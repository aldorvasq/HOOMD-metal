// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BounceBackNVEGPU.metal
    \brief Metal port of BounceBackNVEGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifndef __METAL_VERSION__
#include "BounceBackNVEGPU.cuh"
#endif

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel_args
    {
struct nve_bounce_step_two_args
    {
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(Scalar3) d_accel;
    HOOMD_DEVPTR(const Scalar4) d_net_force;
    HOOMD_DEVPTR(const unsigned int) d_group;
    Scalar dt;
    unsigned int N;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void nve_bounce_step_two(HOOMD_KERNEL_PARAMS(kernel_args::nve_bounce_step_two_args))
    {
    device Scalar4* d_vel = args.d_vel;
    device Scalar3* d_accel = args.d_accel;
    device const Scalar4* d_net_force = args.d_net_force;
    device const unsigned int* d_group = args.d_group;
    const Scalar dt = args.dt;
    const unsigned int N = args.N;

    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    const unsigned int pid = d_group[idx];

    const Scalar4 net_force = d_net_force[pid];
    Scalar3 accel = make_scalar3(net_force.x, net_force.y, net_force.z);
    Scalar4 vel = d_vel[pid];
    accel.x /= vel.w;
    accel.y /= vel.w;
    accel.z /= vel.w;

    // then, update the velocity
    vel.x += Scalar(0.5) * accel.x * dt;
    vel.y += Scalar(0.5) * accel.y * dt;
    vel.z += Scalar(0.5) * accel.z * dt;

    d_vel[pid] = vel;
    d_accel[pid] = accel;
    }

#else

namespace kernel
    {
//! Kernel for applying second step of velocity Verlet algorithm with bounce back
/*!
 * \param d_vel Particle velocities
 * \param d_accel Particle accelerations
 * \param d_net_force Net force on each particle
 * \param d_group Indexes in particle group
 * \param dt Timestep
 * \param N Number of particles in group
 *
 * \b Implementation:
 * Using one thread per particle, the particle velocities are updated according to the second step
 * of the velocity Verlet algorithm. This is the standard update as in MD, and is only reimplemented
 * here in case future modifications are necessary.
 */

    } // end namespace kernel

/*!
 * \param d_vel Particle velocities
 * \param d_accel Particle accelerations
 * \param d_net_force Net force on each particle
 * \param d_group Indexes in particle group
 * \param dt Timestep
 * \param N Number of particles in group
 * \param block_size Number of threads per block
 *
 * \sa kernel::nve_bounce_step_two
 */
cudaError_t nve_bounce_step_two(Scalar4* d_vel,
                                Scalar3* d_accel,
                                const Scalar4* d_net_force,
                                const unsigned int* d_group,
                                const Scalar dt,
                                const unsigned int N,
                                const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::mpcd::gpu::nve_bounce_step_two");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    {
    kernel_args::nve_bounce_step_two_args _args;
    _args.d_vel = d_vel;
    _args.d_accel = d_accel;
    _args.d_net_force = d_net_force;
    _args.d_group = d_group;
    _args.dt = dt;
    _args.N = N;
    hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                       "hoomd::mpcd::gpu::nve_bounce_step_two",
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
