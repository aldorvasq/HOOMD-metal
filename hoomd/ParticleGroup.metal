// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ParticleGroup.metal
    \brief Metal port of ParticleGroup.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#ifndef __METAL_VERSION__
#include "ParticleGroup.cuh"
#include "hoomd/mtl/Primitives.h"
#endif

namespace hoomd
    {
namespace kernel_args
    {
struct rebuild_index_list_args
    {
    unsigned int N;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(const unsigned int) d_is_member_tag;
    HOOMD_DEVPTR(unsigned int) d_is_member;
    };

struct scatter_member_indices_args
    {
    unsigned int N;
    HOOMD_DEVPTR(const unsigned int) d_scan;
    HOOMD_DEVPTR(const unsigned int) d_is_member;
    HOOMD_DEVPTR(unsigned int) d_member_idx;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

kernel void gpu_rebuild_index_list_kernel(
    HOOMD_KERNEL_PARAMS(kernel_args::rebuild_index_list_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.N)
        return;
    unsigned int tag = args.d_tag[idx];
    args.d_is_member[idx] = args.d_is_member_tag[tag];
    }

kernel void gpu_scatter_member_indices(HOOMD_KERNEL_PARAMS(kernel_args::scatter_member_indices_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.N)
        return;
    if (args.d_is_member[idx])
        args.d_member_idx[args.d_scan[idx]] = idx;
    }

#else

namespace kernel
    {
hipError_t gpu_rebuild_index_list(unsigned int N,
                                  unsigned int* d_is_member_tag,
                                  unsigned int* d_is_member,
                                  unsigned int* d_tag)
    {
    assert(d_is_member);
    assert(d_is_member_tag);
    assert(d_tag);

    kernel_args::rebuild_index_list_args args;
    args.N = N;
    args.d_tag = d_tag;
    args.d_is_member_tag = d_is_member_tag;
    args.d_is_member = d_is_member;

    unsigned int block_size = 256;
    unsigned int n_blocks = N / block_size + 1;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       "hoomd::gpu_rebuild_index_list_kernel",
                       dim3(n_blocks),
                       dim3(block_size),
                       args);
    }

hipError_t gpu_compact_index_list(unsigned int N,
                                  unsigned int* d_is_member,
                                  unsigned int* d_member_idx,
                                  unsigned int& num_local_members,
                                  unsigned int* d_tmp,
                                  CachedAllocator& alloc)
    {
    assert(d_is_member);
    assert(d_member_idx);

    mtl::exclusive_scan(d_is_member, d_tmp, N);
    num_local_members = mtl::reduce_sum(d_is_member, N);

    kernel_args::scatter_member_indices_args args;
    args.N = N;
    args.d_scan = d_tmp;
    args.d_is_member = d_is_member;
    args.d_member_idx = d_member_idx;

    unsigned int block_size = 256;
    unsigned int n_blocks = N / block_size + 1;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       "hoomd::gpu_scatter_member_indices",
                       dim3(n_blocks),
                       dim3(block_size),
                       args);
    }

    } // end namespace kernel

#endif

    } // end namespace hoomd
