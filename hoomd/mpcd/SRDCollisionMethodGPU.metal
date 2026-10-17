// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file SRDCollisionMethodGPU.metal
    \brief Metal port of SRDCollisionMethodGPU.cu

    The cell rotation vectors, thermostat factors, and cell averages are double precision arrays
    on the host. Apple GPUs have no double arithmetic, so the kernels read and write them through
    the hoomd_double storage types and compute in single precision. The CUDA template parameter
    use_thermostat of srd_draw_vectors is Metal function constant 0.
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#ifndef __METAL_VERSION__
#include "SRDCollisionMethodGPU.cuh"
#else
// double precision arrays are read and written through hoomd_double
#define double hoomd_double
#endif

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel_args
    {
struct srd_draw_vectors_args
    {
    HOOMD_DEVPTR(double3) d_rotvec;
    HOOMD_DEVPTR(double) d_factors;
    HOOMD_DEVPTR(const double3) d_cell_energy;
    Index3D ci;
    int3 origin;
    uint3 global_dim;
    Index3D global_ci;
    uint64_t timestep;
    Scalar T_set;
    unsigned int seed;
    unsigned int n_dimensions;
    unsigned int Ncell;
    };

struct srd_rotate_args
    {
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(Scalar4) d_vel_embed;
    HOOMD_DEVPTR(const unsigned int) d_embed_group;
    HOOMD_DEVPTR(const unsigned int) d_embed_cell_ids;
    HOOMD_DEVPTR(const double4) d_cell_vel;
    HOOMD_DEVPTR(const double3) d_rotvec;
    HOOMD_DEVPTR(const double) d_factors;
    Scalar cos_a;
    Scalar one_minus_cos_a;
    Scalar sin_a;
    unsigned int N_mpcd;
    unsigned int N_tot;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant bool use_thermostat [[function_constant(0)]];

//! Draw the cell rotation vectors (and thermostat factors), see srd_draw_vectors in
//! SRDCollisionMethodGPU.cu
kernel void srd_draw_vectors(HOOMD_KERNEL_PARAMS(kernel_args::srd_draw_vectors_args))
    {
    // one thread per cell
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.Ncell)
        return;

    Index3D ci = args.ci;
    Index3D global_ci = args.global_ci;
    const int3 origin = args.origin;
    const uint3 global_dim = args.global_dim;

    // get local cell triple from 1d index
    const uint3 cell = ci.getTriple(idx);
    // shift local cell by local origin, and wrap through global boundaries
    int3 global_cell
        = make_int3(origin.x + (int)cell.x, origin.y + (int)cell.y, origin.z + (int)cell.z);
    if (global_cell.x >= (int)global_dim.x)
        global_cell.x -= global_dim.x;
    else if (global_cell.x < 0)
        global_cell.x += global_dim.x;

    if (global_cell.y >= (int)global_dim.y)
        global_cell.y -= global_dim.y;
    else if (global_cell.y < 0)
        global_cell.y += global_dim.y;

    if (global_cell.z >= (int)global_dim.z)
        global_cell.z -= global_dim.z;
    else if (global_cell.z < 0)
        global_cell.z += global_dim.z;

    // convert global triple to 1d global index
    const unsigned int global_idx = global_ci(global_cell.x, global_cell.y, global_cell.z);

    // Initialize the PRNG using the cell index, timestep, and seed for the hash
    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::SRDCollisionMethod, args.timestep, uint16_t(args.seed)),
        hoomd::Counter(global_idx));

    // draw rotation vector off the surface of the sphere
    Scalar3 rotvec;
    hoomd::SpherePointGenerator<Scalar> sphgen;
    sphgen(rng, rotvec);
    args.d_rotvec[idx] = make_double3(rotvec.x, rotvec.y, rotvec.z);

    if (use_thermostat)
        {
        const double3 cell_energy = args.d_cell_energy[idx];
        const unsigned int np = __double_as_int(cell_energy.z);
        Scalar factor = 1.0;
        if (np > 1)
            {
            // the total number of degrees of freedom in the cell divided by 2
            const Scalar alpha = args.n_dimensions * (np - 1) / Scalar(2.);

            // draw a random kinetic energy for the cell at the set temperature
            hoomd::GammaDistribution<Scalar> gamma_gen(alpha, args.T_set);
            const Scalar rand_ke = gamma_gen(rng);

            // generate the scale factor from the current temperature
            // (don't use the kinetic energy of this cell, since this
            // is total not relative to COM)
            const Scalar cur_ke = alpha * Scalar(cell_energy.y);
            factor = (cur_ke > Scalar(0.)) ? fast::sqrt(rand_ke / cur_ke) : Scalar(1.);
            }
        args.d_factors[idx] = factor;
        }
    }

//! Apply the SRD rotation, see srd_rotate in SRDCollisionMethodGPU.cu
kernel void srd_rotate(HOOMD_KERNEL_PARAMS(kernel_args::srd_rotate_args))
    {
    // one thread per particle
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= args.N_tot)
        return;

    // load particle data
    Scalar3 vel;
    unsigned int cell;
    // these properties are needed for the embedded particles only
    unsigned int idx(0);
    Scalar mass(0);
    if (tid < args.N_mpcd)
        {
        const Scalar4 vel_cell = args.d_vel[tid];
        vel = make_scalar3(vel_cell.x, vel_cell.y, vel_cell.z);
        cell = __scalar_as_int(vel_cell.w);
        }
    else
        {
        idx = args.d_embed_group[tid - args.N_mpcd];

        const Scalar4 vel_mass = args.d_vel_embed[idx];
        vel = make_scalar3(vel_mass.x, vel_mass.y, vel_mass.z);
        mass = vel_mass.w;
        cell = args.d_embed_cell_ids[tid - args.N_mpcd];
        }

    // subtract average velocity
    const double4 avg_vel_d = args.d_cell_vel[cell];
    const Scalar3 avg_vel = make_scalar3(avg_vel_d.x, avg_vel_d.y, avg_vel_d.z);
    vel -= avg_vel;

    // get rotation vector
    const double3 rot_vec_d = args.d_rotvec[cell];
    const Scalar3 rot_vec = make_scalar3(rot_vec_d.x, rot_vec_d.y, rot_vec_d.z);
    const Scalar cos_a = args.cos_a;
    const Scalar one_minus_cos_a = args.one_minus_cos_a;
    const Scalar sin_a = args.sin_a;

    // perform the rotation
    Scalar3 new_vel;
    new_vel.x = (cos_a + rot_vec.x * rot_vec.x * one_minus_cos_a) * vel.x;
    new_vel.x += (rot_vec.x * rot_vec.y * one_minus_cos_a - sin_a * rot_vec.z) * vel.y;
    new_vel.x += (rot_vec.x * rot_vec.z * one_minus_cos_a + sin_a * rot_vec.y) * vel.z;

    new_vel.y = (cos_a + rot_vec.y * rot_vec.y * one_minus_cos_a) * vel.y;
    new_vel.y += (rot_vec.x * rot_vec.y * one_minus_cos_a + sin_a * rot_vec.z) * vel.x;
    new_vel.y += (rot_vec.y * rot_vec.z * one_minus_cos_a - sin_a * rot_vec.x) * vel.z;

    new_vel.z = (cos_a + rot_vec.z * rot_vec.z * one_minus_cos_a) * vel.z;
    new_vel.z += (rot_vec.x * rot_vec.z * one_minus_cos_a - sin_a * rot_vec.y) * vel.x;
    new_vel.z += (rot_vec.y * rot_vec.z * one_minus_cos_a + sin_a * rot_vec.x) * vel.y;

    // rescale the velocity if factor is available
    if (args.d_factors != nullptr)
        {
        const Scalar factor = args.d_factors[cell];
        new_vel *= factor;
        }

    new_vel += avg_vel;

    // set the new velocity
    if (tid < args.N_mpcd)
        {
        args.d_vel[tid] = make_scalar4(new_vel.x, new_vel.y, new_vel.z, __int_as_scalar(cell));
        }
    else
        {
        args.d_vel_embed[idx] = make_scalar4(new_vel.x, new_vel.y, new_vel.z, mass);
        }
    }

#else

cudaError_t srd_draw_vectors(double3* d_rotvec,
                             double* d_factors,
                             const double3* d_cell_energy,
                             const Index3D& ci,
                             const int3 origin,
                             const uint3 global_dim,
                             const Index3D& global_ci,
                             const uint64_t timestep,
                             const uint16_t seed,
                             const Scalar T_set,
                             const unsigned int n_dimensions,
                             const unsigned int block_size)
    {
    const bool thermostat = d_factors != NULL;
    std::initializer_list<mtl::FunctionConstant> constants
        = {mtl::FunctionConstant::make_bool(0, thermostat)};
    const char* name = "hoomd::mpcd::gpu::srd_draw_vectors";
    unsigned int run_block_size
        = std::min(block_size, mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name, constants));

    kernel_args::srd_draw_vectors_args args;
    args.d_rotvec = d_rotvec;
    args.d_factors = d_factors;
    args.d_cell_energy = d_cell_energy;
    args.ci = ci;
    args.origin = origin;
    args.global_dim = global_dim;
    args.global_ci = global_ci;
    args.timestep = timestep;
    args.T_set = T_set;
    args.seed = seed;
    args.n_dimensions = n_dimensions;
    args.Ncell = ci.getNumElements();
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(args.Ncell / run_block_size + 1),
                       dim3(run_block_size),
                       args,
                       0,
                       constants);
    }

cudaError_t srd_rotate(Scalar4* d_vel,
                       Scalar4* d_vel_embed,
                       const unsigned int* d_embed_group,
                       const unsigned int* d_embed_cell_ids,
                       const double4* d_cell_vel,
                       const double3* d_rotvec,
                       const double angle,
                       const double* d_factors,
                       const unsigned int N_mpcd,
                       const unsigned int N_tot,
                       const unsigned int block_size)
    {
    const char* name = "hoomd::mpcd::gpu::srd_rotate";
    unsigned int run_block_size
        = std::min(block_size, mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name));

    // precompute angles for rotation
    const double cos_a = slow::cos(angle);
    const double one_minus_cos_a = 1.0 - cos_a;
    const double sin_a = slow::sin(angle);

    kernel_args::srd_rotate_args args;
    args.d_vel = d_vel;
    args.d_vel_embed = d_vel_embed;
    args.d_embed_group = d_embed_group;
    args.d_embed_cell_ids = d_embed_cell_ids;
    args.d_cell_vel = d_cell_vel;
    args.d_rotvec = d_rotvec;
    args.d_factors = d_factors;
    args.cos_a = Scalar(cos_a);
    args.one_minus_cos_a = Scalar(one_minus_cos_a);
    args.sin_a = Scalar(sin_a);
    args.N_mpcd = N_mpcd;
    args.N_tot = N_tot;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(N_tot / run_block_size + 1),
                       dim3(run_block_size),
                       args);
    }

#endif

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
