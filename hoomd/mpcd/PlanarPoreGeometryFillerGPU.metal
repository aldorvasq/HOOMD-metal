// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file PlanarPoreGeometryFillerGPU.metal
    \brief Metal port of PlanarPoreGeometryFillerGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "ParticleDataUtilities.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "PlanarPoreGeometry.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifndef __METAL_VERSION__
#include "PlanarPoreGeometryFillerGPU.cuh"
#endif

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel_args
    {
struct slit_pore_draw_particles_args
    {
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(unsigned int) d_tag;
    BoxDim box;
    HOOMD_DEVPTR(const Scalar4) d_boxes;
    HOOMD_DEVPTR(const uint2) d_ranges;
    unsigned int num_boxes;
    unsigned int N_tot;
    unsigned int type;
    unsigned int first_tag;
    unsigned int first_idx;
    Scalar vel_factor;
    uint64_t timestep;
    uint16_t seed;
    unsigned int filler_id;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void slit_pore_draw_particles(HOOMD_KERNEL_PARAMS(kernel_args::slit_pore_draw_particles_args))
    {
    device Scalar4* d_pos = args.d_pos;
    device Scalar4* d_vel = args.d_vel;
    device unsigned int* d_tag = args.d_tag;
    const BoxDim box = args.box;
    device const Scalar4* d_boxes = args.d_boxes;
    device const uint2* d_ranges = args.d_ranges;
    const unsigned int num_boxes = args.num_boxes;
    const unsigned int N_tot = args.N_tot;
    const unsigned int type = args.type;
    const unsigned int first_tag = args.first_tag;
    const unsigned int first_idx = args.first_idx;
    const Scalar vel_factor = args.vel_factor;
    const uint64_t timestep = args.timestep;
    const uint16_t seed = args.seed;
    const unsigned int filler_id = args.filler_id;

    // num_boxes should be 6, so this will all fit in shmem
    threadgroup char* s_data = (threadgroup char*)hoomd_dynamic_shared;
    threadgroup Scalar4* s_boxes = (threadgroup Scalar4*)(&s_data[0]);
    threadgroup uint2* s_ranges = (threadgroup uint2*)(&s_data[sizeof(Scalar4) * num_boxes]);
    for (unsigned int offset = 0; offset < num_boxes; offset += blockDim.x)
        {
        if (offset + threadIdx.x < num_boxes)
            {
            const unsigned int boxid = offset + threadIdx.x;
            s_boxes[boxid] = d_boxes[boxid];
            s_ranges[boxid] = d_ranges[boxid];
            }
        }
    __syncthreads();

    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_tot)
        return;

    // linear search for box matching thread (num_boxes is small)
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();
    for (unsigned int boxid = 0; boxid < num_boxes; ++boxid)
        {
        const uint2 range = s_ranges[boxid];
        if (idx >= range.x && idx < range.y)
            {
            const Scalar4 fillbox = s_boxes[boxid];
            lo.x = fillbox.x;
            hi.x = fillbox.y;
            lo.y = fillbox.z;
            hi.y = fillbox.w;
            break;
            }
        }

    // particle tag and index
    const unsigned int tag = first_tag + idx;
    const unsigned int pidx = first_idx + idx;
    d_tag[pidx] = tag;

    // initialize random number generator for positions and velocity
    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::VirtualParticleFiller, timestep, seed),
        hoomd::Counter(tag, filler_id));
    d_pos[pidx] = make_scalar4(hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng),
                               hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng),
                               hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng),
                               __int_as_scalar(type));

    hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
    Scalar3 vel;
    {
    Scalar r1, r2;
    gen(r1, r2, rng);
    vel.x = r1;
    vel.y = r2;
    }
    vel.z = gen(rng);
    // TODO: should these be given zero net-momentum contribution (relative to the frame of
    // reference?)
    d_vel[pidx] = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
    }

#else

namespace kernel
    {
/*!
 * \param d_pos Particle positions
 * \param d_vel Particle velocities
 * \param d_tag Particle tags
 * \param box Local simulation box
 * \param d_boxes List of 2d bounding boxes for filling
 * \param d_ranges Particle ranges for each box
 * \param num_boxes Number of bounding boxes to fill
 * \param N_tot Total number of particles
 * \param type Type of fill particles
 * \param first_tag First tag of filled particles
 * \param first_idx First (local) particle index of filled particles
 * \param vel_factor Scale factor for uniform normal velocities consistent with particle mass /
 * temperature
 * \param timestep Current timestep
 * \param seed User seed to PRNG for drawing velocities
 * \param filler_id ID for this filler
 *
 * \b Implementation:
 *
 * Using one thread per particle, the thread is assigned to a fill range matching a 2d bounding box,
 * which defines a cuboid of volume to fill. The thread index is translated into a particle tag
 * and local particle index. A random position is drawn within the cuboid. A random velocity
 * is drawn consistent with the speed of the moving wall.
 */

    } // end namespace kernel

/*!
 * \param d_pos Particle positions
 * \param d_vel Particle velocities
 * \param d_tag Particle tags
 * \param box Local simulation box
 * \param d_boxes List of 2d bounding boxes for filling
 * \param d_ranges Particle ranges for each box
 * \param num_boxes Number of bounding boxes to fill
 * \param N_tot Total number of particles
 * \param mass Mass of fill particles
 * \param type Type of fill particles
 * \param first_tag First tag of filled particles
 * \param first_idx First (local) particle index of filled particles
 * \param kT Temperature for fill particles
 * \param timestep Current timestep
 * \param seed User seed to PRNG for drawing velocities
 * \param filler_id ID for this filler
 * \param block_size Number of threads per block
 *
 * \sa kernel::slit_pore_draw_particles
 */
cudaError_t slit_pore_draw_particles(Scalar4* d_pos,
                                     Scalar4* d_vel,
                                     unsigned int* d_tag,
                                     const BoxDim& box,
                                     const Scalar4* d_boxes,
                                     const uint2* d_ranges,
                                     const unsigned int num_boxes,
                                     const unsigned int N_tot,
                                     const Scalar mass,
                                     const unsigned int type,
                                     const unsigned int first_tag,
                                     const unsigned int first_idx,
                                     const Scalar kT,
                                     const uint64_t timestep,
                                     const uint16_t seed,
                                     const unsigned int filler_id,
                                     const unsigned int block_size)
    {
    if (N_tot == 0)
        return cudaSuccess;

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::mpcd::gpu::slit_pore_draw_particles");
    max_block_size = attr.maxThreadsPerBlock;

    // precompute factor for rescaling the velocities since it is the same for all particles
    const Scalar vel_factor = fast::sqrt(kT / mass);

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_tot / run_block_size + 1);
    const size_t shared_bytes = num_boxes * (sizeof(Scalar4) + sizeof(uint2));
    {
    kernel_args::slit_pore_draw_particles_args _args;
    _args.d_pos = d_pos;
    _args.d_vel = d_vel;
    _args.d_tag = d_tag;
    _args.box = box;
    _args.d_boxes = d_boxes;
    _args.d_ranges = d_ranges;
    _args.num_boxes = num_boxes;
    _args.N_tot = N_tot;
    _args.type = type;
    _args.first_tag = first_tag;
    _args.first_idx = first_idx;
    _args.vel_factor = vel_factor;
    _args.timestep = timestep;
    _args.seed = seed;
    _args.filler_id = filler_id;
    hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                       "hoomd::mpcd::gpu::slit_pore_draw_particles",
                       grid,
                       run_block_size,
                       _args,
                       shared_bytes);
    }

    return cudaSuccess;
    }

#endif

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
