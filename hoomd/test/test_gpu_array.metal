// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file test_gpu_array.metal
    \brief Metal port of test_gpu_array.cu
*/

#include "hoomd/mtl/MetalKernel.h"


#ifndef __METAL_VERSION__
#include "test_gpu_array.cuh"

#include <cmath>
#endif

namespace hoomd
    {
namespace test
    {
namespace kernel_args
    {
struct gpu_add_one_kernel_args
    {
    HOOMD_DEVPTR(int) d_data;
    size_t num;
    };

struct gpu_fill_test_pattern_kernel_args
    {
    HOOMD_DEVPTR(int) d_data;
    size_t num;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void gpu_add_one_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_add_one_kernel_args))
    {
    device int* d_data = args.d_data;
    size_t num = args.num;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num)
        d_data[idx] = d_data[idx] + 1;
    }

kernel void gpu_fill_test_pattern_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_fill_test_pattern_kernel_args))
    {
    device int* d_data = args.d_data;
    size_t num = args.num;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num)
        d_data[idx] = idx * idx;
    }

#else

/*! \param d_data Device pointer to the array where the data is held
    \param num Number of elements in the array

    \post All \a num elements in d_data are incremented by 1
*/


/*! \param d_data Device pointer to the array where the data is held
    \param num Number of elements in the array

    gpu_add_one is just a driver for gpu_add_one_kernel()
*/
hipError_t gpu_add_one(int* d_data, size_t num)
    {
    unsigned int block_size = 256;

    // setup the grid to run the kernel
    dim3 grid((int)ceil((double)num / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    {
        kernel_args::gpu_add_one_kernel_args _args;
        _args.d_data = d_data;
        _args.num = num;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::test::gpu_add_one_kernel",
                           dim3(grid),
                           dim3(threads),
                           _args,
                           0);
        }

    hipDeviceSynchronize();
    return hipPeekAtLastError();
    }

/*! \param d_data Device pointer to the array where the data is held
    \param num Number of elements in the array

    \post Element \a i in \a d_data is set to \a i * \a i
*/


/*! \param d_data Device pointer to the array where the data is held
    \param num Number of elements in the array

    gpu_fill_test_pattern is just a driver for gpu_fill_test_pattern_kernel()
*/
hipError_t gpu_fill_test_pattern(int* d_data, size_t num)
    {
    unsigned int block_size = 256;

    // setup the grid to run the kernel
    dim3 grid((int)ceil((double)num / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    {
        kernel_args::gpu_fill_test_pattern_kernel_args _args;
        _args.d_data = d_data;
        _args.num = num;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::test::gpu_fill_test_pattern_kernel",
                           dim3(grid),
                           dim3(threads),
                           _args,
                           0);
        }

    hipDeviceSynchronize();
    return hipPeekAtLastError();
    }

#endif

    } // end namespace test
    } // end namespace hoomd
