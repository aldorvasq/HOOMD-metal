// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file test_warp_tools.metal
    \brief Metal port of test_warp_tools.cu

    The logical warp width tpp is Metal function constant 0. A switch over the supported widths
    (which the compiler folds) instantiates the WarpReduce and WarpScan templates for each width.
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/Index1D.h"

#ifdef __METAL_VERSION__
#include "hoomd/mtl/WarpTools.h"
#else
#include "test_warp_tools.cuh"
#endif

#define BLOCK_SIZE 32
#define MAX_TPP 32

namespace hoomd
    {
namespace test
    {
namespace kernel_args
    {
struct warp_reduce_args
    {
    HOOMD_DEVPTR(const int) d_data;
    HOOMD_DEVPTR(int) d_reduce;
    HOOMD_DEVPTR(int) d_sum;
    unsigned int N;
    unsigned int width;
    Index2D reduce_idx;
    };

struct warp_scan_args
    {
    HOOMD_DEVPTR(const int) d_data;
    HOOMD_DEVPTR(int) d_scan;
    HOOMD_DEVPTR(int) d_sum;
    unsigned int N;
    unsigned int width;
    Index3D scan_idx;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant uint tpp_constant [[function_constant(0)]];

template<int tpp>
inline void warp_reduce_body(constant kernel_args::warp_reduce_args& args,
                             unsigned int tid,
                             unsigned int thread_in_block,
                             ushort lane)
    {
    const unsigned int idx = tid / tpp;
    const unsigned int cta_idx = thread_in_block % tpp;
    if (idx >= args.N)
        return;
    Index2D reduce_idx = args.reduce_idx;

    int sum(0), cntr(0);
    unsigned int offset = cta_idx;
    bool done = false;
    while (!done)
        {
        int thread_data;
        if (offset < args.width)
            {
            thread_data = args.d_data[idx * args.width + offset];
            }
        else
            {
            thread_data = 0;
            done = true;
            }
        offset += tpp;

        done = hoomd::detail::WarpScan<bool, tpp>(lane).Broadcast(done, 0);
        if (!done)
            {
            int sum_iter = hoomd::detail::WarpReduce<int, tpp>(lane).Sum(thread_data);

            if (cta_idx == 0)
                args.d_reduce[reduce_idx(idx, cntr)] = sum_iter;

            sum += sum_iter;

            ++cntr;
            }
        }

    if (cta_idx == 0)
        {
        args.d_sum[idx] = sum;
        }
    }

template<int tpp>
inline void warp_scan_body(constant kernel_args::warp_scan_args& args,
                           unsigned int tid,
                           unsigned int thread_in_block,
                           ushort lane)
    {
    const unsigned int idx = tid / tpp;
    const unsigned int cta_idx = thread_in_block % tpp;
    if (idx >= args.N)
        return;
    Index3D scan_idx = args.scan_idx;

    int sum(0), cntr(0);
    unsigned int offset = cta_idx;
    bool done = false;
    while (!done)
        {
        int thread_data;
        if (offset < args.width)
            {
            thread_data = args.d_data[idx * args.width + offset];
            }
        else
            {
            thread_data = 0;
            done = true;
            }
        offset += tpp;

        done = hoomd::detail::WarpScan<bool, tpp>(lane).Broadcast(done, 0);
        if (!done)
            {
            int sum_iter(0);
            hoomd::detail::WarpScan<int, tpp>(lane).ExclusiveSum(thread_data, thread_data, sum_iter);

            args.d_scan[scan_idx(idx, cta_idx, cntr)] = thread_data;
            if (cta_idx == 0)
                args.d_scan[scan_idx(idx, tpp, cntr)] = sum_iter;

            sum += sum_iter;

            ++cntr;
            }
        }

    if (cta_idx == 0)
        {
        args.d_sum[idx] = sum;
        }
    }

#define HOOMD_TPP_SWITCH(BODY)                                                                 \
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;                            \
    switch (tpp_constant)                                                                      \
        {                                                                                      \
    case 1:                                                                                    \
        BODY<1>(args, tid, threadIdx.x, hoomd_lane);                                           \
        break;                                                                                 \
    case 2:                                                                                    \
        BODY<2>(args, tid, threadIdx.x, hoomd_lane);                                           \
        break;                                                                                 \
    case 4:                                                                                    \
        BODY<4>(args, tid, threadIdx.x, hoomd_lane);                                           \
        break;                                                                                 \
    case 8:                                                                                    \
        BODY<8>(args, tid, threadIdx.x, hoomd_lane);                                           \
        break;                                                                                 \
    case 16:                                                                                   \
        BODY<16>(args, tid, threadIdx.x, hoomd_lane);                                          \
        break;                                                                                 \
    case 32:                                                                                   \
        BODY<32>(args, tid, threadIdx.x, hoomd_lane);                                          \
        break;                                                                                 \
        }

kernel void warp_reduce_kernel(HOOMD_KERNEL_PARAMS(kernel_args::warp_reduce_args))
    {
    HOOMD_TPP_SWITCH(warp_reduce_body)
    }

kernel void warp_scan_kernel(HOOMD_KERNEL_PARAMS(kernel_args::warp_scan_args))
    {
    HOOMD_TPP_SWITCH(warp_scan_body)
    }

#else

void warp_reduce(const reduce_params& params)
    {
    hipMemset(params.reduce, 0, params.reduce_idx.getNumElements() * sizeof(int));
    hipMemset(params.sum, 0, params.N * sizeof(int));

    kernel_args::warp_reduce_args args;
    args.d_data = params.data;
    args.d_reduce = params.reduce;
    args.d_sum = params.sum;
    args.N = params.N;
    args.width = params.width;
    args.reduce_idx = params.reduce_idx;
    mtl::launch(HOOMD_MTL_SOURCE,
                "hoomd::test::warp_reduce_kernel",
                dim3((params.N * params.tpp + BLOCK_SIZE - 1) / BLOCK_SIZE),
                dim3(BLOCK_SIZE),
                args,
                0,
                {mtl::FunctionConstant::make_uint(0, params.tpp)});
    }

void warp_scan(const scan_params& params)
    {
    hipMemset(params.scan, 0, params.scan_idx.getNumElements() * sizeof(int));
    hipMemset(params.sum, 0, params.N * sizeof(int));

    kernel_args::warp_scan_args args;
    args.d_data = params.data;
    args.d_scan = params.scan;
    args.d_sum = params.sum;
    args.N = params.N;
    args.width = params.width;
    args.scan_idx = params.scan_idx;
    mtl::launch(HOOMD_MTL_SOURCE,
                "hoomd::test::warp_scan_kernel",
                dim3((params.N * params.tpp + BLOCK_SIZE - 1) / BLOCK_SIZE),
                dim3(BLOCK_SIZE),
                args,
                0,
                {mtl::FunctionConstant::make_uint(0, params.tpp)});
    }

#endif

    } // end namespace test
    } // end namespace hoomd
