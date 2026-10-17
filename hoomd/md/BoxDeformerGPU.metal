// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxDeformerGPU.metal
    \brief Metal port of BoxDeformerGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#ifndef __METAL_VERSION__
#include "BoxDeformerGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct gpu_boxdeformer_wrap_kernel_args
    {
    unsigned int N;
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(int3) d_image;
    BoxDim new_box;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void gpu_boxdeformer_wrap_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_boxdeformer_wrap_kernel_args))
    {
    unsigned int N = args.N;
    device Scalar4* d_pos = args.d_pos;
    device Scalar4* d_vel = args.d_vel;
    device int3* d_image = args.d_image;
    const BoxDim new_box = args.new_box;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        // device memory cannot bind to the references wrap takes: wrap local copies
        Scalar4 pos = d_pos[idx];
        Scalar4 vel = d_vel[idx];
        int3 image = d_image[idx];
        new_box.wrap(pos, vel, image);
        d_pos[idx] = pos;
        d_vel[idx] = vel;
        d_image[idx] = image;
        }
    }

#else

namespace kernel
    {
//! Wrap particles after box deformation


hipError_t gpu_boxdeformer_wrap(const unsigned int N,
                                Scalar4* d_pos,
                                Scalar4* d_vel,
                                int3* d_image,
                                const BoxDim& new_box,
                                unsigned int block_size)
    {
    unsigned int max_block_size;

    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_boxdeformer_wrap_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid((N / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    {
        kernel_args::gpu_boxdeformer_wrap_kernel_args _args;
        _args.N = N;
        _args.d_pos = d_pos;
        _args.d_vel = d_vel;
        _args.d_image = d_image;
        _args.new_box = new_box;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_boxdeformer_wrap_kernel",
                           grid,
                           threads,
                           _args,
                           0);
        }

    return hipSuccess;
    }

    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
