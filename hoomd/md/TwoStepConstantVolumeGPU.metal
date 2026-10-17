// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file TwoStepConstantVolumeGPU.metal
    \brief Metal port of TwoStepConstantVolumeGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifndef __METAL_VERSION__
#include "TwoStepConstantVolumeGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct nvt_rescale_step_one_args
    {
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(const Scalar3) d_accel;
    HOOMD_DEVPTR(int3) d_image;
    HOOMD_DEVPTR(const unsigned int) d_group_members;
    BoxDim box;
    unsigned int work_size;
    Scalar rescale_factor;
    Scalar deltaT;
    unsigned int n_dimensions;
    unsigned int limit;
    Scalar maximum_displacement;
    };

struct nvt_rescale_step_two_args
    {
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(Scalar3) d_accel;
    HOOMD_DEVPTR(const unsigned int) d_group_members;
    HOOMD_DEVPTR(const Scalar4) d_net_force;
    unsigned int work_size;
    Scalar deltaT;
    Scalar rescale_factor;
    unsigned int n_dimensions;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

//! First half step of the NVT integration with velocity rescaling
kernel void gpu_nvt_rescale_step_one_kernel(
    HOOMD_KERNEL_PARAMS(kernel_args::nvt_rescale_step_one_args))
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= args.work_size)
        return;

    unsigned int idx = args.d_group_members[group_idx];
    const Scalar deltaT = args.deltaT;

    Scalar4 postype = args.d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    Scalar4 velmass = args.d_vel[idx];
    Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
    Scalar3 accel = args.d_accel[idx];
    if (args.n_dimensions == 2)
        accel.z = Scalar(0.0);

    // update velocity
    vel = vel + Scalar(1.0 / 2.0) * accel * deltaT;
    vel *= args.rescale_factor;

    if (args.limit)
        {
        Scalar displacement = sqrtf(dot(vel, vel));
        if (displacement * deltaT > args.maximum_displacement)
            vel = vel * args.maximum_displacement / displacement * deltaT;
        }

    // update position
    pos += vel * deltaT;

    // read in the image flags
    int3 image = args.d_image[idx];

    // time to fix the periodic boundary conditions
    BoxDim box = args.box;
    box.wrap(pos, vel, image);

    args.d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
    args.d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, velmass.w);
    args.d_image[idx] = image;
    }

//! Second half step of the NVT integration with velocity rescaling
kernel void gpu_nvt_rescale_step_two_kernel(
    HOOMD_KERNEL_PARAMS(kernel_args::nvt_rescale_step_two_args))
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= args.work_size)
        return;

    unsigned int idx = args.d_group_members[group_idx];

    // read in the net force and calculate the acceleration
    Scalar4 net_force = args.d_net_force[idx];
    if (args.n_dimensions == 2)
        net_force.z = Scalar(0.0);
    Scalar3 accel = make_scalar3(net_force.x, net_force.y, net_force.z);

    Scalar4 vel = args.d_vel[idx];
    Scalar3 v = make_scalar3(vel.x, vel.y, vel.z);
    Scalar mass = vel.w;
    accel = accel / mass;

    v *= args.rescale_factor;
    v += Scalar(1.0 / 2.0) * args.deltaT * accel;

    args.d_vel[idx] = make_scalar4(v.x, v.y, v.z, vel.w);
    args.d_accel[idx] = accel;
    }

#else

namespace kernel
    {
hipError_t gpu_nvt_rescale_step_one(Scalar4* d_pos,
                                    Scalar4* d_vel,
                                    const Scalar3* d_accel,
                                    int3* d_image,
                                    unsigned int* d_group_members,
                                    unsigned int group_size,
                                    const BoxDim& box,
                                    unsigned int block_size,
                                    Scalar rescale_factor,
                                    Scalar deltaT,
                                    unsigned int n_dimensions,
                                    bool use_limit,
                                    Scalar maximum_displacement)
    {
    const char* name = "hoomd::md::gpu_nvt_rescale_step_one_kernel";
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name);
    unsigned int run_block_size = std::min(block_size, max_block_size);

    kernel_args::nvt_rescale_step_one_args args;
    args.d_pos = d_pos;
    args.d_vel = d_vel;
    args.d_accel = d_accel;
    args.d_image = d_image;
    args.d_group_members = d_group_members;
    args.box = box;
    args.work_size = group_size;
    args.rescale_factor = rescale_factor;
    args.deltaT = deltaT;
    args.n_dimensions = n_dimensions;
    args.limit = use_limit;
    args.maximum_displacement = maximum_displacement;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3((group_size / run_block_size) + 1),
                       dim3(run_block_size),
                       args);
    }

hipError_t gpu_nvt_rescale_step_two(Scalar4* d_vel,
                                    Scalar3* d_accel,
                                    unsigned int* d_group_members,
                                    unsigned int group_size,
                                    Scalar4* d_net_force,
                                    unsigned int block_size,
                                    Scalar deltaT,
                                    Scalar rescale_factor,
                                    unsigned int n_dimensions)
    {
    const char* name = "hoomd::md::gpu_nvt_rescale_step_two_kernel";
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name);
    unsigned int run_block_size = std::min(block_size, max_block_size);

    kernel_args::nvt_rescale_step_two_args args;
    args.d_vel = d_vel;
    args.d_accel = d_accel;
    args.d_group_members = d_group_members;
    args.d_net_force = d_net_force;
    args.work_size = group_size;
    args.deltaT = deltaT;
    args.rescale_factor = rescale_factor;
    args.n_dimensions = n_dimensions;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3((group_size / run_block_size) + 1),
                       dim3(run_block_size),
                       args);
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
