// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ATCollisionMethodGPU.metal
    \brief Metal port of ATCollisionMethodGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "ParticleDataUtilities.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifndef __METAL_VERSION__
#include "ATCollisionMethodGPU.cuh"
#endif

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel_args
    {
struct at_draw_velocity_args
    {
    HOOMD_DEVPTR(Scalar4) d_alt_vel;
    HOOMD_DEVPTR(Scalar4) d_alt_vel_embed;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    Scalar mpcd_mass;
    HOOMD_DEVPTR(const unsigned int) d_embed_idx;
    HOOMD_DEVPTR(const Scalar4) d_vel_embed;
    HOOMD_DEVPTR(const unsigned int) d_tag_embed;
    uint64_t timestep;
    uint16_t seed;
    Scalar T;
    unsigned int N_mpcd;
    unsigned int N_tot;
    };

struct at_apply_velocity_args
    {
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(Scalar4) d_vel_embed;
    HOOMD_DEVPTR(const Scalar4) d_vel_alt;
    HOOMD_DEVPTR(const unsigned int) d_embed_idx;
    HOOMD_DEVPTR(const Scalar4) d_vel_alt_embed;
    HOOMD_DEVPTR(const unsigned int) d_embed_cell_ids;
    HOOMD_DEVPTR(const double4) d_cell_vel;
    HOOMD_DEVPTR(const double4) d_rand_vel;
    unsigned int N_mpcd;
    unsigned int N_tot;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void at_draw_velocity(HOOMD_KERNEL_PARAMS(kernel_args::at_draw_velocity_args))
    {
    device Scalar4* d_alt_vel = args.d_alt_vel;
    device Scalar4* d_alt_vel_embed = args.d_alt_vel_embed;
    device const unsigned int* d_tag = args.d_tag;
    const Scalar mpcd_mass = args.mpcd_mass;
    device const unsigned int* d_embed_idx = args.d_embed_idx;
    device const Scalar4* d_vel_embed = args.d_vel_embed;
    device const unsigned int* d_tag_embed = args.d_tag_embed;
    const uint64_t timestep = args.timestep;
    const uint16_t seed = args.seed;
    const Scalar T = args.T;
    const unsigned int N_mpcd = args.N_mpcd;
    const unsigned int N_tot = args.N_tot;

    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_tot)
        return;

    unsigned int pidx;
    unsigned int tag;
    Scalar mass;
    if (idx < N_mpcd)
        {
        pidx = idx;
        mass = mpcd_mass;
        tag = d_tag[idx];
        }
    else
        {
        pidx = d_embed_idx[idx - N_mpcd];
        mass = d_vel_embed[pidx].w;
        tag = d_tag_embed[pidx];
        }

    // draw random velocities from normal distribution
    hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::ATCollisionMethod, timestep, seed),
                               hoomd::Counter(tag));
    hoomd::NormalDistribution<Scalar> gen(fast::sqrt(T / mass), 0.0);
    Scalar3 vel;
    {
    Scalar r1, r2;
    gen(r1, r2, rng);
    vel.x = r1;
    vel.y = r2;
    }
    vel.z = gen(rng);

    // save out velocities
    if (idx < N_mpcd)
        {
        d_alt_vel[pidx] = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
        }
    else
        {
        d_alt_vel_embed[pidx] = make_scalar4(vel.x, vel.y, vel.z, mass);
        }
    }

kernel void at_apply_velocity(HOOMD_KERNEL_PARAMS(kernel_args::at_apply_velocity_args))
    {
    device Scalar4* d_vel = args.d_vel;
    device Scalar4* d_vel_embed = args.d_vel_embed;
    device const Scalar4* d_vel_alt = args.d_vel_alt;
    device const unsigned int* d_embed_idx = args.d_embed_idx;
    device const Scalar4* d_vel_alt_embed = args.d_vel_alt_embed;
    device const unsigned int* d_embed_cell_ids = args.d_embed_cell_ids;
    device const double4* d_cell_vel = args.d_cell_vel;
    device const double4* d_rand_vel = args.d_rand_vel;
    const unsigned int N_mpcd = args.N_mpcd;
    const unsigned int N_tot = args.N_tot;

    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_tot)
        return;

    unsigned int cell, pidx;
    Scalar4 vel_rand;
    if (idx < N_mpcd)
        {
        pidx = idx;
        const Scalar4 vel_cell = d_vel[idx];
        cell = __scalar_as_int(vel_cell.w);
        vel_rand = d_vel_alt[idx];
        }
    else
        {
        pidx = d_embed_idx[idx - N_mpcd];
        cell = d_embed_cell_ids[idx - N_mpcd];
        vel_rand = d_vel_alt_embed[pidx];
        }

    // load cell data
    const double4 v_c = d_cell_vel[cell];
    const double4 vrand_c = d_rand_vel[cell];

    // compute new velocity using the cell + the random draw
    const Scalar3 vnew = make_scalar3(v_c.x - vrand_c.x + vel_rand.x,
                                      v_c.y - vrand_c.y + vel_rand.y,
                                      v_c.z - vrand_c.z + vel_rand.z);

    if (idx < N_mpcd)
        {
        d_vel[pidx] = make_scalar4(vnew.x, vnew.y, vnew.z, __int_as_scalar(cell));
        }
    else
        {
        d_vel_embed[pidx] = make_scalar4(vnew.x, vnew.y, vnew.z, vel_rand.w);
        }
    }

#else

namespace kernel
    {




    } // end namespace kernel

cudaError_t at_draw_velocity(Scalar4* d_alt_vel,
                             Scalar4* d_alt_vel_embed,
                             const unsigned int* d_tag,
                             const Scalar mpcd_mass,
                             const unsigned int* d_embed_idx,
                             const Scalar4* d_vel_embed,
                             const unsigned int* d_tag_embed,
                             const uint64_t timestep,
                             const uint16_t seed,
                             const Scalar T,
                             const unsigned int N_mpcd,
                             const unsigned int N_tot,
                             const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::mpcd::gpu::at_draw_velocity");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(N_tot / run_block_size + 1);
    {
        kernel_args::at_draw_velocity_args _args;
        _args.d_alt_vel = d_alt_vel;
        _args.d_alt_vel_embed = d_alt_vel_embed;
        _args.d_tag = d_tag;
        _args.mpcd_mass = mpcd_mass;
        _args.d_embed_idx = d_embed_idx;
        _args.d_vel_embed = d_vel_embed;
        _args.d_tag_embed = d_tag_embed;
        _args.timestep = timestep;
        _args.seed = seed;
        _args.T = T;
        _args.N_mpcd = N_mpcd;
        _args.N_tot = N_tot;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::mpcd::gpu::at_draw_velocity",
                           grid,
                           run_block_size,
                           _args,
                           0);
        }

    return cudaSuccess;
    }

cudaError_t at_apply_velocity(Scalar4* d_vel,
                              Scalar4* d_vel_embed,
                              const Scalar4* d_vel_alt,
                              const unsigned int* d_embed_idx,
                              const Scalar4* d_vel_alt_embed,
                              const unsigned int* d_embed_cell_ids,
                              const double4* d_cell_vel,
                              const double4* d_rand_vel,
                              const unsigned int N_mpcd,
                              const unsigned int N_tot,
                              const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::mpcd::gpu::at_apply_velocity");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(N_tot / run_block_size + 1);
    {
        kernel_args::at_apply_velocity_args _args;
        _args.d_vel = d_vel;
        _args.d_vel_embed = d_vel_embed;
        _args.d_vel_alt = d_vel_alt;
        _args.d_embed_idx = d_embed_idx;
        _args.d_vel_alt_embed = d_vel_alt_embed;
        _args.d_embed_cell_ids = d_embed_cell_ids;
        _args.d_cell_vel = d_cell_vel;
        _args.d_rand_vel = d_rand_vel;
        _args.N_mpcd = N_mpcd;
        _args.N_tot = N_tot;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::mpcd::gpu::at_apply_velocity",
                           grid,
                           run_block_size,
                           _args,
                           0);
        }

    return cudaSuccess;
    }

#endif

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
