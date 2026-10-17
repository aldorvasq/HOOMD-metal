// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file NeighborListGPU.metal
    \brief Metal port of NeighborListGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifndef __METAL_VERSION__
#include "NeighborListGPU.cuh"
#include "hoomd/mtl/Primitives.h"

#include <cmath>
#endif

//! Number of exclusions processed per filter kernel launch
#define HOOMD_NLIST_FILTER_BATCH_SIZE 4

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct nlist_needs_update_args
    {
    HOOMD_DEVPTR(unsigned int) d_result;
    HOOMD_DEVPTR(const Scalar4) d_last_pos;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar) d_rcut_max;
    BoxDim box;
    Scalar3 lambda;
    Scalar r_buff;
    Scalar lambda_min;
    unsigned int nwork;
    unsigned int checkn;
    };

struct nlist_filter_args
    {
    HOOMD_DEVPTR(unsigned int) d_n_neigh;
    HOOMD_DEVPTR(unsigned int) d_nlist;
    HOOMD_DEVPTR(const size_t) d_head_list;
    HOOMD_DEVPTR(const unsigned int) d_n_ex;
    HOOMD_DEVPTR(const unsigned int) d_ex_list;
    Index2D exli;
    unsigned int N;
    unsigned int ex_start;
    };

struct update_exclusion_list_args
    {
    HOOMD_DEVPTR(const unsigned int) tags;
    HOOMD_DEVPTR(const unsigned int) rtags;
    HOOMD_DEVPTR(const unsigned int) n_ex_tag;
    HOOMD_DEVPTR(const unsigned int) ex_list_tag;
    HOOMD_DEVPTR(unsigned int) n_ex_idx;
    HOOMD_DEVPTR(unsigned int) ex_list_idx;
    Index2D ex_list_tag_indexer;
    Index2D ex_list_indexer;
    unsigned int N;
    };

struct nlist_head_list_args
    {
    HOOMD_DEVPTR(size_t) d_head_list;
    HOOMD_DEVPTR(size_t) d_req_size_nlist;
    HOOMD_DEVPTR(const unsigned int) d_Nmax;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    unsigned int N;
    unsigned int ntypes;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

/*! Set *d_result to checkn when any particle moved more than half the buffer since the last
    neighbor list build (accounting for box scaling lambda).
*/
kernel void
gpu_nlist_needs_update_check_new_kernel(HOOMD_KERNEL_PARAMS(kernel_args::nlist_needs_update_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.nwork)
        return;

    BoxDim box = args.box;
    Scalar4 cur_postype = args.d_pos[idx];
    Scalar3 cur_pos = make_scalar3(cur_postype.x, cur_postype.y, cur_postype.z);
    const unsigned int cur_type = __scalar_as_int(cur_postype.w);

    Scalar4 last_postype = args.d_last_pos[idx];
    Scalar3 last_pos = make_scalar3(last_postype.x, last_postype.y, last_postype.z);

    Scalar3 dx = cur_pos - args.lambda * last_pos;
    dx = box.minImage(dx);

    const Scalar rmin = args.d_rcut_max[cur_type];
    const Scalar rmax = rmin + args.r_buff;
    const Scalar delta_max = (rmax * args.lambda_min - rmin) / Scalar(2.0);
    Scalar maxshiftsq = (delta_max > 0) ? delta_max * delta_max : 0.0f;

    if (dot(dx, dx) >= maxshiftsq)
        atomicMax(args.d_result, args.checkn);
    }

//! Remove up to HOOMD_NLIST_FILTER_BATCH_SIZE exclusions per particle from the neighbor list
kernel void gpu_nlist_filter_kernel(HOOMD_KERNEL_PARAMS(kernel_args::nlist_filter_args))
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= args.N)
        return;

    const unsigned int n_neigh = args.d_n_neigh[idx];
    const unsigned int n_ex = args.d_n_ex[idx];
    unsigned int new_n_neigh = 0;

    // quit early if this thread doesn't need to process any more exclusions
    if (args.ex_start >= n_ex)
        return;

    const unsigned int n_ex_process = n_ex - args.ex_start;
    Index2D exli = args.exli;

    unsigned int l_ex_list[HOOMD_NLIST_FILTER_BATCH_SIZE];
    for (unsigned int cur_ex_idx = 0; cur_ex_idx < HOOMD_NLIST_FILTER_BATCH_SIZE; cur_ex_idx++)
        {
        if (cur_ex_idx < n_ex_process)
            l_ex_list[cur_ex_idx] = args.d_ex_list[exli(idx, cur_ex_idx + args.ex_start)];
        else
            l_ex_list[cur_ex_idx] = 0xffffffffu;
        }

    const size_t my_head = args.d_head_list[idx];
    for (unsigned int cur_neigh_idx = 0; cur_neigh_idx < n_neigh; cur_neigh_idx++)
        {
        unsigned int cur_neigh = args.d_nlist[my_head + cur_neigh_idx];

        bool excluded = false;
        for (unsigned int cur_ex_idx = 0; cur_ex_idx < HOOMD_NLIST_FILTER_BATCH_SIZE; cur_ex_idx++)
            {
            if (cur_neigh == l_ex_list[cur_ex_idx])
                excluded = true;
            }

        if (!excluded)
            {
            if (new_n_neigh != cur_neigh_idx)
                args.d_nlist[my_head + new_n_neigh] = cur_neigh;
            new_n_neigh++;
            }
        }

    args.d_n_neigh[idx] = new_n_neigh;
    }

//! Translate the exclusion list from tags to particle indices
kernel void
gpu_update_exclusion_list_kernel(HOOMD_KERNEL_PARAMS(kernel_args::update_exclusion_list_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.N)
        return;

    Index2D ex_list_tag_indexer = args.ex_list_tag_indexer;
    Index2D ex_list_indexer = args.ex_list_indexer;

    unsigned int tag = args.tags[idx];
    unsigned int n = args.n_ex_tag[tag];
    args.n_ex_idx[idx] = n;
    for (unsigned int offset = 0; offset < n; offset++)
        {
        unsigned int ex_tag = args.ex_list_tag[ex_list_tag_indexer(tag, offset)];
        unsigned int ex_idx = args.rtags[ex_tag];
        args.ex_list_idx[ex_list_indexer(idx, offset)] = ex_idx;
        }
    }

//! Initialize the head list with the number of neighbors each particle's type allows
kernel void gpu_nlist_init_head_list_kernel(HOOMD_KERNEL_PARAMS(kernel_args::nlist_head_list_args))
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= args.N)
        return;

    const Scalar4 postype_i = args.d_pos[idx];
    const unsigned int type_i = __scalar_as_int(postype_i.w);
    const unsigned int Nmax_i = args.d_Nmax[type_i];

    args.d_head_list[idx] = Nmax_i;

    // the last thread stores its count so the total size can be computed after the scan
    if (idx == (args.N - 1))
        *args.d_req_size_nlist = Nmax_i;
    }

//! Complete the total size of the neighbor list after the head list scan
kernel void gpu_nlist_get_nlist_size_kernel(HOOMD_KERNEL_PARAMS(kernel_args::nlist_head_list_args))
    {
    *args.d_req_size_nlist += args.d_head_list[args.N - 1];
    }

#else

namespace kernel
    {
hipError_t gpu_nlist_needs_update_check_new(unsigned int* d_result,
                                            const Scalar4* d_last_pos,
                                            const Scalar4* d_pos,
                                            const unsigned int N,
                                            const BoxDim& box,
                                            const Scalar* d_rcut_max,
                                            const Scalar r_buff,
                                            const unsigned int ntypes,
                                            const Scalar lambda_min,
                                            const Scalar3 lambda,
                                            const unsigned int checkn)
    {
    kernel_args::nlist_needs_update_args args;
    args.d_result = d_result;
    args.d_last_pos = d_last_pos;
    args.d_pos = d_pos;
    args.d_rcut_max = d_rcut_max;
    args.box = box;
    args.lambda = lambda;
    args.r_buff = r_buff;
    args.lambda_min = lambda_min;
    args.nwork = N;
    args.checkn = checkn;

    unsigned int block_size = 128;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       "hoomd::md::gpu_nlist_needs_update_check_new_kernel",
                       dim3(N / block_size + 1),
                       dim3(block_size),
                       args);
    }

hipError_t gpu_nlist_filter(unsigned int* d_n_neigh,
                            unsigned int* d_nlist,
                            const size_t* d_head_list,
                            const unsigned int* d_n_ex,
                            const unsigned int* d_ex_list,
                            const Index2D& exli,
                            const unsigned int N,
                            const unsigned int block_size)
    {
    const char* name = "hoomd::md::gpu_nlist_filter_kernel";
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name);
    unsigned int run_block_size = std::min(block_size, max_block_size);

    kernel_args::nlist_filter_args args;
    args.d_n_neigh = d_n_neigh;
    args.d_nlist = d_nlist;
    args.d_head_list = d_head_list;
    args.d_n_ex = d_n_ex;
    args.d_ex_list = d_ex_list;
    args.exli = exli;
    args.N = N;

    // the exclusion list is processed in batches
    unsigned int n_batches = (unsigned int)std::ceil(double(exli.getH())
                                                     / double(HOOMD_NLIST_FILTER_BATCH_SIZE));
    hipError_t error = hipSuccess;
    for (unsigned int batch = 0; batch < n_batches; batch++)
        {
        args.ex_start = batch * HOOMD_NLIST_FILTER_BATCH_SIZE;
        error = mtl::launch(HOOMD_MTL_SOURCE,
                            name,
                            dim3(N / run_block_size + 1),
                            dim3(run_block_size),
                            args);
        }
    return error;
    }

hipError_t gpu_update_exclusion_list(const unsigned int* d_tag,
                                     const unsigned int* d_rtag,
                                     const unsigned int* d_n_ex_tag,
                                     const unsigned int* d_ex_list_tag,
                                     const Index2D& ex_list_tag_indexer,
                                     unsigned int* d_n_ex_idx,
                                     unsigned int* d_ex_list_idx,
                                     const Index2D& ex_list_indexer,
                                     const unsigned int N)
    {
    kernel_args::update_exclusion_list_args args;
    args.tags = d_tag;
    args.rtags = d_rtag;
    args.n_ex_tag = d_n_ex_tag;
    args.ex_list_tag = d_ex_list_tag;
    args.n_ex_idx = d_n_ex_idx;
    args.ex_list_idx = d_ex_list_idx;
    args.ex_list_tag_indexer = ex_list_tag_indexer;
    args.ex_list_indexer = ex_list_indexer;
    args.N = N;

    unsigned int block_size = 256;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       "hoomd::md::gpu_update_exclusion_list_kernel",
                       dim3(N / block_size + 1),
                       dim3(block_size),
                       args);
    }

hipError_t gpu_nlist_build_head_list(size_t* d_head_list,
                                     size_t* d_req_size_nlist,
                                     const unsigned int* d_Nmax,
                                     const Scalar4* d_pos,
                                     const unsigned int N,
                                     const unsigned int ntypes,
                                     const unsigned int block_size)
    {
    const char* name = "hoomd::md::gpu_nlist_init_head_list_kernel";
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name);
    unsigned int run_block_size = std::min(block_size, max_block_size);

    kernel_args::nlist_head_list_args args;
    args.d_head_list = d_head_list;
    args.d_req_size_nlist = d_req_size_nlist;
    args.d_Nmax = d_Nmax;
    args.d_pos = d_pos;
    args.N = N;
    args.ntypes = ntypes;

    // initialize each particle with its number of neighbors
    mtl::launch(HOOMD_MTL_SOURCE, name, dim3(N / run_block_size + 1), dim3(run_block_size), args);

    // exclusive prefix sum gives each particle's starting index
    mtl::exclusive_scan(d_head_list, d_head_list, N);

    // one thread completes the total neighbor list size
    return mtl::launch(HOOMD_MTL_SOURCE,
                       "hoomd::md::gpu_nlist_get_nlist_size_kernel",
                       dim3(1),
                       dim3(1),
                       args);
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
