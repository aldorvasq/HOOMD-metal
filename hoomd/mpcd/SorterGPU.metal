// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file SorterGPU.metal
    \brief Metal port of SorterGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifndef __METAL_VERSION__
#include "SorterGPU.cuh"
#include "CellListGPU.cuh"

#include "hoomd/mtl/Primitives.h"
#endif

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel_args
    {
struct sort_apply_args
    {
    HOOMD_DEVPTR(Scalar4) d_pos_alt;
    HOOMD_DEVPTR(Scalar4) d_vel_alt;
    HOOMD_DEVPTR(unsigned int) d_tag_alt;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar4) d_vel;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(const unsigned int) d_order;
    unsigned int N;
    };

struct sort_set_sentinel_args
    {
    HOOMD_DEVPTR(unsigned int) d_cell_list;
    HOOMD_DEVPTR(const unsigned int) d_cell_np;
    Index2D cli;
    unsigned int sentinel;
    unsigned int N_cli;
    };

struct sort_gen_reverse_args
    {
    HOOMD_DEVPTR(unsigned int) d_rorder;
    HOOMD_DEVPTR(const unsigned int) d_order;
    unsigned int N;
    };

struct sort_flag_mpcd_args
    {
    HOOMD_DEVPTR(unsigned int) d_flags;
    HOOMD_DEVPTR(const unsigned int) d_cell_list;
    unsigned int num_items;
    unsigned int N_mpcd;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__

kernel void sort_apply(HOOMD_KERNEL_PARAMS(kernel_args::sort_apply_args))
    {
    device Scalar4* d_pos_alt = args.d_pos_alt;
    device Scalar4* d_vel_alt = args.d_vel_alt;
    device unsigned int* d_tag_alt = args.d_tag_alt;
    device const Scalar4* d_pos = args.d_pos;
    device const Scalar4* d_vel = args.d_vel;
    device const unsigned int* d_tag = args.d_tag;
    device const unsigned int* d_order = args.d_order;
    const unsigned int N = args.N;

    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const unsigned int old_idx = d_order[idx];
    d_pos_alt[idx] = d_pos[old_idx];
    d_vel_alt[idx] = d_vel[old_idx];
    d_tag_alt[idx] = d_tag[old_idx];
    }

kernel void sort_set_sentinel(HOOMD_KERNEL_PARAMS(kernel_args::sort_set_sentinel_args))
    {
    device unsigned int* d_cell_list = args.d_cell_list;
    device const unsigned int* d_cell_np = args.d_cell_np;
    const Index2D cli = args.cli;
    const unsigned int sentinel = args.sentinel;
    const unsigned int N_cli = args.N_cli;

    // one thread per cell-list entry
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_cli)
        return;

    // convert the entry 1D index into a 2D index
    const unsigned int cell = idx / cli.getW();
    const unsigned int offset = idx - (cell * cli.getW());

    // if the offset lies outside the number of particles in the cell, fill it with sentinel
    const unsigned int np = d_cell_np[cell];
    if (offset >= np)
        d_cell_list[idx] = sentinel;
    }

kernel void sort_gen_reverse(HOOMD_KERNEL_PARAMS(kernel_args::sort_gen_reverse_args))
    {
    device unsigned int* d_rorder = args.d_rorder;
    device const unsigned int* d_order = args.d_order;
    const unsigned int N = args.N;

    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // inverse map the ordering
    // d_order maps the new index (idx) onto the old index (pid), so we need to flip this around
    const unsigned int pid = d_order[idx];
    d_rorder[pid] = idx;
    }


//! Flag the cell list entries that are MPCD particles (for sort_cell_compact)
kernel void sort_flag_mpcd(HOOMD_KERNEL_PARAMS(kernel_args::sort_flag_mpcd_args))
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.num_items)
        return;
    args.d_flags[idx] = args.d_cell_list[idx] < args.N_mpcd ? 1 : 0;
    }

#else

namespace kernel
    {
//! Kernel to apply sorted particle order
/*!
 * \param d_pos_alt Alternate array of particle positions (output)
 * \param d_vel_alt Alternate array of particle velocities (output)
 * \param d_tag_alt Alternate array of particle tags (output)
 * \param d_pos Particle positions
 * \param d_vel Particle velocities
 * \param d_tag Particle tags
 * \param d_order Mapping of new particle index onto old particle index
 * \param N Number of particles
 *
 * \b Implementation
 * Using one thread per particle, particle data is reordered from the old arrays
 * into the new arrays. This coalesces writes but fragments reads.
 */


//! Kernel to set the empty-cell-entry sentinel
/*!
 * \param d_cell_list Cell list to fill in
 * \param d_cell_np Number of particles per cell
 * \param cli Two-dimensional cell-list indexer
 * \param sentinel Value to fill into empty cell-list entries
 * \param N_cli Number of total entries (filled and empty) in the cell list
 *
 * \b Implementation
 * Using one thread per cell-list entry, the 1D kernel index is mapped onto
 * the 2D entry in the cell list. If the current entry is not filled, the entry
 * is filled with the value of \a sentinel. Typically, \a sentinel should be
 * larger than the number of MPCD particles in the system. (A good value would be
 * 0xffffffff). The cell list can subsequently be compacted by mpcd::gpu::sort_cell_compact.
 */


//! Kernel to generate the reverse particle mapping
/*!
 * \param d_rorder Map of old particle indexes onto new particle indexes (output)
 * \param d_order Map of new particle indexes onto old particle indexes
 * \param N Number of particles
 *
 * \b Implementation
 * Using one thread per particle, the map generated by mpcd::gpu::sort_cell_compact
 * is read and reversed so that new particle indexes can be looked up from old
 * particle indexes.
 */


    } // end namespace kernel

/*!
 * \param d_pos_alt Alternate array of particle positions (output)
 * \param d_vel_alt Alternate array of particle velocities (output)
 * \param d_tag_alt Alternate array of particle tags (output)
 * \param d_pos Particle positions
 * \param d_vel Particle velocities
 * \param d_tag Particle tags
 * \param d_order Mapping of new particle index onto old particle index
 * \param N Number of particles
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion
 *
 * \sa sort_apply
 */
cudaError_t sort_apply(Scalar4* d_pos_alt,
                       Scalar4* d_vel_alt,
                       unsigned int* d_tag_alt,
                       const Scalar4* d_pos,
                       const Scalar4* d_vel,
                       const unsigned int* d_tag,
                       const unsigned int* d_order,
                       const unsigned int N,
                       const unsigned int block_size)
    {
    if (N == 0)
        return cudaSuccess;

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::mpcd::gpu::sort_apply");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    {
        kernel_args::sort_apply_args _args;
        _args.d_pos_alt = d_pos_alt;
        _args.d_vel_alt = d_vel_alt;
        _args.d_tag_alt = d_tag_alt;
        _args.d_pos = d_pos;
        _args.d_vel = d_vel;
        _args.d_tag = d_tag;
        _args.d_order = d_order;
        _args.N = N;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::mpcd::gpu::sort_apply",
                           grid,
                           run_block_size,
                           _args,
                           0);
        }

    return cudaSuccess;
    }

/*!
 * \param d_cell_list Cell list to fill in
 * \param d_cell_np Number of particles per cell
 * \param cli Two-dimensional cell-list indexer
 * \param sentinel Value to fill into empty cell-list entries
 * \param N_cli Number of total entries (filled and empty) in the cell list
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion
 *
 * \sa sort_set_sentinel
 */
cudaError_t sort_set_sentinel(unsigned int* d_cell_list,
                              const unsigned int* d_cell_np,
                              const Index2D& cli,
                              const unsigned int sentinel,
                              const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::mpcd::gpu::sort_set_sentinel");
    max_block_size = attr.maxThreadsPerBlock;

    const unsigned int N_cli = cli.getNumElements();

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_cli / run_block_size + 1);
    {
        kernel_args::sort_set_sentinel_args _args;
        _args.d_cell_list = d_cell_list;
        _args.d_cell_np = d_cell_np;
        _args.cli = cli;
        _args.sentinel = sentinel;
        _args.N_cli = N_cli;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::mpcd::gpu::sort_set_sentinel",
                           grid,
                           run_block_size,
                           _args,
                           0);
        }

    return cudaSuccess;
    }

//! Less-than comparison functor for cell-list compaction
unsigned int sort_cell_compact(unsigned int* d_order,
                               const unsigned int* d_cell_list,
                               const unsigned int num_items,
                               const unsigned int N_mpcd)
    {
    if (num_items == 0)
        return 0;

    mtl::ScratchScope scratch;
    unsigned int* d_flags = scratch.allocate<unsigned int>(num_items);
    unsigned int* d_num = scratch.allocate<unsigned int>(1);

    kernel_args::sort_flag_mpcd_args args;
    args.d_flags = d_flags;
    args.d_cell_list = d_cell_list;
    args.num_items = num_items;
    args.N_mpcd = N_mpcd;
    const unsigned int block_size = 256;
    mtl::launch(HOOMD_MTL_SOURCE,
                "hoomd::mpcd::gpu::sort_flag_mpcd",
                dim3(num_items / block_size + 1),
                dim3(block_size),
                args);

    // keep the MPCD particles in cell list order (thrust::copy_if)
    mtl::select_flagged(d_cell_list, d_flags, d_order, d_num, num_items);
    unsigned int num = 0;
    hipMemcpy(&num, d_num, sizeof(unsigned int), hipMemcpyDeviceToHost);
    return num;
    }

/*!
 * \param d_rorder Map of old particle indexes onto new particle indexes (output)
 * \param d_order Map of new particle indexes onto old particle indexes
 * \param N Number of particles
 *
 * \returns cudaSuccess on completion
 *
 * \sa sort_gen_reverse
 */
cudaError_t sort_gen_reverse(unsigned int* d_rorder,
                             const unsigned int* d_order,
                             const unsigned int N,
                             const unsigned int block_size)
    {
    if (N == 0)
        return cudaSuccess;

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::mpcd::gpu::sort_gen_reverse");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    {
        kernel_args::sort_gen_reverse_args _args;
        _args.d_rorder = d_rorder;
        _args.d_order = d_order;
        _args.N = N;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::mpcd::gpu::sort_gen_reverse",
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
