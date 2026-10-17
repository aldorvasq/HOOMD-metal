// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BondedGroupData.metal
    \brief Metal port of BondedGroupData.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "BondedGroupData.cuh"
#include "ParticleData.cuh"

#ifndef __METAL_VERSION__
#include "hoomd/mtl/Primitives.h"

#include <string>
#endif

namespace hoomd
    {
namespace kernel_args
    {
template<unsigned int group_size> struct count_groups_args
    {
    HOOMD_DEVPTR(const group_storage<group_size>) d_group_table;
    HOOMD_DEVPTR(const unsigned int) d_rtag;
    HOOMD_DEVPTR(unsigned int) d_scratch_idx;
    HOOMD_DEVPTR(unsigned int) d_scratch_g;
    HOOMD_DEVPTR(unsigned int) d_n_groups;
    HOOMD_DEVPTR(unsigned int) d_condition;
    unsigned int n_groups;
    unsigned int max_n_groups;
    unsigned int next_flag;
    };

template<unsigned int group_size> struct group_scatter_args
    {
    HOOMD_DEVPTR(const unsigned int) d_scratch_g;
    HOOMD_DEVPTR(const unsigned int) d_scratch_idx;
    HOOMD_DEVPTR(const unsigned int) d_offset;
    HOOMD_DEVPTR(const group_storage<group_size>) d_members;
    HOOMD_DEVPTR(const typeval_union) d_group_typeval;
    HOOMD_DEVPTR(const unsigned int) d_rtag;
    HOOMD_DEVPTR(group_storage<group_size>) d_pidx_group_table;
    HOOMD_DEVPTR(unsigned int) d_pidx_gpos_table;
    unsigned int n_scratch;
    unsigned int pidx_group_table_pitch;
    unsigned int has_type_mapping;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

//! Count the groups each particle belongs to and record (particle, group) pairs
template<unsigned int group_size>
inline void count_groups_impl(constant kernel_args::count_groups_args<group_size>& args,
                              unsigned int group_idx)
    {
    if (group_idx >= args.n_groups)
        return;

    group_storage<group_size> g = args.d_group_table[group_idx];

    for (unsigned int i = 0; i < group_size; ++i)
        {
        unsigned int tag_i = g.tag[i];
        unsigned int pidx_i = args.d_rtag[tag_i];

        // detect incomplete groups
        if (pidx_i == NOT_LOCAL)
            atomicMax(args.d_condition, args.next_flag + 1 + group_idx);

        // write out group_idx to temporary array
        args.d_scratch_g[i * args.n_groups + group_idx] = group_idx;
        args.d_scratch_idx[i * args.n_groups + group_idx] = pidx_i;

        // atomically increment number of groups
        unsigned int n = 0;
        if (pidx_i != NOT_LOCAL)
            n = atomicInc(&args.d_n_groups[pidx_i], 0xffffffffu);

        if (n >= args.max_n_groups)
            // set flag to indicate we need to grow the output array
            atomicMax(args.d_condition, args.next_flag);
        }
    }

//! Write each particle's groups into the per-particle group table
template<unsigned int group_size>
inline void group_scatter_impl(constant kernel_args::group_scatter_args<group_size>& args,
                               unsigned int i)
    {
    if (i >= args.n_scratch)
        return;

    unsigned int pidx = args.d_scratch_idx[i];
    unsigned int offset = args.d_offset[i] * args.pidx_group_table_pitch + pidx;

    // load group
    unsigned int group_idx = args.d_scratch_g[i];
    group_storage<group_size> g = args.d_members[group_idx];

    // construct compact group representation, excluding current particle
    group_storage<group_size> p;

    // last element = group type or index
    if (args.has_type_mapping)
        p.idx[group_size - 1] = args.d_group_typeval[group_idx].type;
    else
        p.idx[group_size - 1] = group_idx;

    unsigned int j = 0;

    // position in group
    unsigned int gpos = 0;

    for (unsigned int k = 0; k < group_size; ++k)
        {
        unsigned int tag_k = g.tag[k];
        unsigned int pidx_k = args.d_rtag[tag_k];
        if (pidx_k == pidx)
            {
            gpos = k;
            continue;
            }

        p.idx[j++] = pidx_k;
        }

    args.d_pidx_group_table[offset] = p;
    args.d_pidx_gpos_table[offset] = gpos;
    }

#define HOOMD_BONDED_GROUP_KERNELS(N)                                                          \
    kernel void gpu_count_groups_kernel_##N(                                                   \
        HOOMD_KERNEL_PARAMS(kernel_args::count_groups_args<N>))                                \
        {                                                                                      \
        count_groups_impl<N>(args, blockIdx.x * blockDim.x + threadIdx.x);                     \
        }                                                                                      \
    kernel void gpu_group_scatter_kernel_##N(                                                  \
        HOOMD_KERNEL_PARAMS(kernel_args::group_scatter_args<N>))                               \
        {                                                                                      \
        group_scatter_impl<N>(args, blockIdx.x * blockDim.x + threadIdx.x);                    \
        }

HOOMD_BONDED_GROUP_KERNELS(2)
HOOMD_BONDED_GROUP_KERNELS(3)
HOOMD_BONDED_GROUP_KERNELS(4)
HOOMD_BONDED_GROUP_KERNELS(6)

#undef HOOMD_BONDED_GROUP_KERNELS

#else

template<unsigned int group_size, typename group_t>
void gpu_update_group_table(const unsigned int n_groups,
                            const unsigned int N,
                            const group_t* d_group_table,
                            const typeval_union* d_group_typeval,
                            const unsigned int* d_rtag,
                            unsigned int* d_n_groups,
                            unsigned int max_n_groups,
                            unsigned int* d_condition,
                            unsigned int next_flag,
                            unsigned int& flag,
                            group_t* d_pidx_group_table,
                            unsigned int* d_pidx_gpos_table,
                            const unsigned int pidx_group_table_pitch,
                            unsigned int* d_scratch_g,
                            unsigned int* d_scratch_idx,
                            unsigned int* d_offsets,
                            bool has_type_mapping,
                            CachedAllocator& alloc)
    {
    const std::string suffix = std::to_string(group_size);

    // construct scratch table by expanding the group table by particle index
    unsigned int block_size = 256;
    unsigned n_blocks = n_groups / block_size + 1;

    // reset number of groups
    hipMemsetAsync(d_n_groups, 0, sizeof(unsigned int) * N);

    kernel_args::count_groups_args<group_size> count;
    count.d_group_table = d_group_table;
    count.d_rtag = d_rtag;
    count.d_scratch_idx = d_scratch_idx;
    count.d_scratch_g = d_scratch_g;
    count.d_n_groups = d_n_groups;
    count.d_condition = d_condition;
    count.n_groups = n_groups;
    count.max_n_groups = max_n_groups;
    count.next_flag = next_flag;
    mtl::launch(HOOMD_MTL_SOURCE,
                ("hoomd::gpu_count_groups_kernel_" + suffix).c_str(),
                dim3(n_blocks),
                dim3(block_size),
                count);

    // read back flag
    hipMemcpy(&flag, d_condition, sizeof(unsigned int), hipMemcpyDeviceToHost);

    if (!(flag >= next_flag) && n_groups)
        {
        // we are good, fill group table
        const unsigned int n_scratch = group_size * n_groups;

        // sort groups by particle idx
        mtl::sort_pairs(d_scratch_idx, d_scratch_g, n_scratch);

        // perform a segmented scan of d_scratch_idx to get each group's rank in its particle
        mtl::exclusive_scan_by_key(d_scratch_idx, nullptr, d_offsets, n_scratch);

        // scatter groups to destinations
        kernel_args::group_scatter_args<group_size> scatter;
        scatter.d_scratch_g = d_scratch_g;
        scatter.d_scratch_idx = d_scratch_idx;
        scatter.d_offset = d_offsets;
        scatter.d_members = d_group_table;
        scatter.d_group_typeval = d_group_typeval;
        scatter.d_rtag = d_rtag;
        scatter.d_pidx_group_table = d_pidx_group_table;
        scatter.d_pidx_gpos_table = d_pidx_gpos_table;
        scatter.n_scratch = n_scratch;
        scatter.pidx_group_table_pitch = pidx_group_table_pitch;
        scatter.has_type_mapping = has_type_mapping;
        mtl::launch(HOOMD_MTL_SOURCE,
                    ("hoomd::gpu_group_scatter_kernel_" + suffix).c_str(),
                    dim3(n_scratch / block_size + 1),
                    dim3(block_size),
                    scatter);
        }
    }

/*
 * Explicit template instantiations
 */

#define HOOMD_INSTANTIATE_GROUP_TABLE(N)                                                       \
    template void gpu_update_group_table<N>(const unsigned int n_groups,                       \
                                            const unsigned int N_ptl,                          \
                                            const union group_storage<N>* d_group_table,       \
                                            const typeval_union* d_group_typeval,              \
                                            const unsigned int* d_rtag,                        \
                                            unsigned int* d_n_groups,                          \
                                            unsigned int max_n_groups,                         \
                                            unsigned int* d_condition,                         \
                                            unsigned int next_flag,                            \
                                            unsigned int& flag,                                \
                                            group_storage<N>* d_pidx_group_table,              \
                                            unsigned int* d_pidx_gpos_table,                   \
                                            const unsigned int pidx_group_table_pitch,         \
                                            unsigned int* d_scratch_g,                         \
                                            unsigned int* d_scratch_idx,                       \
                                            unsigned int* d_offsets,                           \
                                            bool has_type_mapping,                             \
                                            CachedAllocator& alloc);

HOOMD_INSTANTIATE_GROUP_TABLE(2)
HOOMD_INSTANTIATE_GROUP_TABLE(3)
HOOMD_INSTANTIATE_GROUP_TABLE(4)
HOOMD_INSTANTIATE_GROUP_TABLE(6)

#undef HOOMD_INSTANTIATE_GROUP_TABLE

#endif

    } // end namespace hoomd
