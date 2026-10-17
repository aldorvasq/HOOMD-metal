// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxResizeUpdaterGPU.metal
    \brief Metal port of BoxResizeUpdaterGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifndef __METAL_VERSION__
#include "BoxResizeUpdaterGPU.cuh"
#endif

namespace hoomd
    {
namespace kernel_args
    {
struct box_resize_scale_args
    {
    HOOMD_DEVPTR(Scalar4) d_pos;
    BoxDim cur_box;
    BoxDim new_box;
    HOOMD_DEVPTR(const unsigned int) d_group_members;
    unsigned int group_size;
    };

struct box_resize_wrap_args
    {
    unsigned int N;
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(int3) d_image;
    BoxDim new_box;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

kernel void gpu_box_resize_scale_kernel(HOOMD_KERNEL_PARAMS(kernel_args::box_resize_scale_args))
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= args.group_size)
        return;

    BoxDim cur_box = args.cur_box;
    BoxDim new_box = args.new_box;
    unsigned int idx = args.d_group_members[group_idx];
    Scalar4 pos = args.d_pos[idx];
    Scalar3 fractional_pos = cur_box.makeFraction(make_scalar3(pos.x, pos.y, pos.z));
    Scalar3 scaled_pos = new_box.makeCoordinates(fractional_pos);
    args.d_pos[idx].x = scaled_pos.x;
    args.d_pos[idx].y = scaled_pos.y;
    args.d_pos[idx].z = scaled_pos.z;
    }

kernel void gpu_box_resize_wrap_kernel(HOOMD_KERNEL_PARAMS(kernel_args::box_resize_wrap_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.N)
        return;

    BoxDim new_box = args.new_box;
    Scalar4 pos = args.d_pos[idx];
    Scalar4 vel = args.d_vel[idx];
    int3 image = args.d_image[idx];
    new_box.wrap(pos, vel, image);
    args.d_pos[idx] = pos;
    args.d_vel[idx] = vel;
    args.d_image[idx] = image;
    }

#else

namespace kernel
    {
hipError_t gpu_box_resize_scale(Scalar4* d_pos,
                                const BoxDim& cur_box,
                                const BoxDim& new_box,
                                const unsigned int* d_group_members,
                                const unsigned int group_size,
                                unsigned int block_size)
    {
    const char* name = "hoomd::gpu_box_resize_scale_kernel";
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name);
    unsigned int run_block_size = std::min(block_size, max_block_size);

    kernel_args::box_resize_scale_args args;
    args.d_pos = d_pos;
    args.cur_box = cur_box;
    args.new_box = new_box;
    args.d_group_members = d_group_members;
    args.group_size = group_size;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3((group_size / run_block_size) + 1),
                       dim3(run_block_size),
                       args);
    }

hipError_t gpu_box_resize_wrap(const unsigned int N,
                               Scalar4* d_pos,
                               Scalar4* d_vel,
                               int3* d_image,
                               const BoxDim& new_box,
                               unsigned int block_size)
    {
    const char* name = "hoomd::gpu_box_resize_wrap_kernel";
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name);
    unsigned int run_block_size = std::min(block_size, max_block_size);

    kernel_args::box_resize_wrap_args args;
    args.N = N;
    args.d_pos = d_pos;
    args.d_vel = d_vel;
    args.d_image = d_image;
    args.new_box = new_box;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3((N / run_block_size) + 1),
                       dim3(run_block_size),
                       args);
    }
    } // end namespace kernel

#endif

    } // end namespace hoomd
