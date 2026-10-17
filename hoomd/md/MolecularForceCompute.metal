// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MolecularForceCompute.metal
    \brief Metal port of MolecularForceCompute.cu

    The thrust and hipcub algorithms in MolecularForceCompute.cu map onto the primitives in
    hoomd/mtl/Primitives.h. CUB's out-of-place radix sorts become a copy followed by an in-place
    sort, and thrust permutation iterators become explicit gathers and scatters.
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifndef __METAL_VERSION__
#include "MolecularForceCompute.cuh"

#include "hoomd/mtl/Primitives.h"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct gpu_fill_molecule_table_args
    {
    unsigned int nptl;
    Index2D molecule_idx;
    HOOMD_DEVPTR(const unsigned int) d_molecule_idx;
    HOOMD_DEVPTR(unsigned int) d_molecule_list;
    HOOMD_DEVPTR(unsigned int) d_molecule_order;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant unsigned int NO_MOLECULE = 0xffffffff;

kernel void gpu_fill_molecule_table_kernel(
    HOOMD_KERNEL_PARAMS(kernel_args::gpu_fill_molecule_table_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= args.nptl)
        return;

    unsigned int molidx = args.d_molecule_idx[idx];
    if (molidx != NO_MOLECULE)
        {
        Index2D molecule_idx = args.molecule_idx;
        args.d_molecule_list[molecule_idx(args.d_molecule_order[idx], molidx)] = idx;
        }
    }

#else

namespace kernel
    {
namespace
    {
#ifndef __HIPCC__
//! MolecularForceCompute.cuh defines NO_MOLECULE only for HIP compilers
const unsigned int NO_MOLECULE = 0xffffffff;
#endif

//! Copy n unsigned ints on the GPU
void copy_uint(unsigned int* d_dst, const unsigned int* d_src, unsigned int n)
    {
    if (n > 0)
        hipMemcpyAsync(d_dst, d_src, sizeof(unsigned int) * n, hipMemcpyDeviceToDevice);
    }

//! Out-of-place stable key/value radix sort (hipcub::DeviceRadixSort::SortPairs)
void sort_pairs_copy(const unsigned int* d_keys_in,
                     unsigned int* d_keys_out,
                     const unsigned int* d_values_in,
                     unsigned int* d_values_out,
                     unsigned int n)
    {
    copy_uint(d_keys_out, d_keys_in, n);
    copy_uint(d_values_out, d_values_in, n);
    mtl::sort_pairs(d_keys_out, d_values_out, n);
    }

//! Read one unsigned int from device memory
unsigned int read_uint(const unsigned int* d_value)
    {
    unsigned int value = 0;
    hipMemcpy(&value, d_value, sizeof(unsigned int), hipMemcpyDeviceToHost);
    return value;
    }
    } // end anonymous namespace

//! Sort local molecules and assign local molecule indices to particles
hipError_t gpu_sort_by_molecule(unsigned int nptl,
                                const unsigned int* d_tag,
                                const unsigned int* d_molecule_tag,
                                unsigned int* d_local_molecule_tags,
                                unsigned int* d_local_molecules_lowest_idx,
                                unsigned int* d_local_unique_molecule_tags,
                                unsigned int* d_local_molecule_idx,
                                unsigned int* d_sorted_by_tag,
                                unsigned int* d_idx_sorted_by_tag,
                                unsigned int* d_idx_sorted_by_molecule_and_tag,
                                unsigned int* d_lowest_idx,
                                unsigned int* d_lowest_idx_sort,
                                unsigned int* d_lowest_idx_in_molecules,
                                unsigned int* d_lowest_idx_by_molecule_tag,
                                unsigned int* d_molecule_length,
                                unsigned int& n_local_molecules,
                                unsigned int& max_len,
                                unsigned int& n_local_ptls_in_molecules,
                                CachedAllocator& alloc,
                                bool check_cuda)
    {
    mtl::ScratchScope scratch;
    unsigned int* d_molecule_length_tmp = scratch.allocate<unsigned int>(nptl + 1);
    unsigned int* d_local_unique_molecule_tags_tmp = scratch.allocate<unsigned int>(nptl + 1);
    unsigned int* d_tmp = scratch.allocate<unsigned int>(nptl + 1);
    unsigned int* d_tmp2 = scratch.allocate<unsigned int>(nptl + 1);
    unsigned int* d_scalar = scratch.allocate<unsigned int>(2);

    // sort local particles by tag
    mtl::sequence(d_tmp, nptl);
    sort_pairs_copy(d_tag, d_sorted_by_tag, d_tmp, d_idx_sorted_by_tag, nptl);

    // molecule tag of each particle in tag order: molecule_tag[tag[idx_sorted_by_tag[i]]]
    unsigned int* d_molecule_by_idx = d_tmp;
    mtl::gather(d_sorted_by_tag, d_molecule_tag, d_molecule_by_idx, nptl, sizeof(unsigned int));

    // sort local particle indices by global molecule tag, keeping tag order (radix sort is stable)
    sort_pairs_copy(d_molecule_by_idx,
                    d_local_molecule_tags,
                    d_idx_sorted_by_tag,
                    d_idx_sorted_by_molecule_and_tag,
                    nptl);

    // find the end of the molecule list
    mtl::fill(d_scalar, NO_MOLECULE, 1);
    mtl::lower_bound(d_local_molecule_tags, nptl, d_scalar, 1, d_scalar + 1);
    n_local_ptls_in_molecules = read_uint(d_scalar + 1);

    // gather unique molecule tags, and reduce their lengths by key
    mtl::fill(d_tmp, 1, n_local_ptls_in_molecules);
    mtl::reduce_by_key(d_local_molecule_tags,
                       d_tmp,
                       d_local_unique_molecule_tags_tmp,
                       d_molecule_length_tmp,
                       d_scalar,
                       n_local_ptls_in_molecules);
    n_local_molecules = n_local_ptls_in_molecules > 0 ? read_uint(d_scalar) : 0;

    // find the index of the particle with lowest tag in every molecule
    mtl::lower_bound(d_local_molecule_tags,
                     n_local_ptls_in_molecules,
                     d_local_unique_molecule_tags_tmp,
                     n_local_molecules,
                     d_lowest_idx_in_molecules);
    mtl::gather(d_lowest_idx_in_molecules,
                d_idx_sorted_by_molecule_and_tag,
                d_lowest_idx,
                n_local_molecules,
                sizeof(unsigned int));

    // compute maximum molecule length
    max_len = n_local_molecules > 0 ? mtl::reduce_max(d_molecule_length_tmp, n_local_molecules)
                                    : 0;

    // order the molecules by the index of their first particle
    sort_pairs_copy(d_lowest_idx,
                    d_lowest_idx_sort,
                    d_local_unique_molecule_tags_tmp,
                    d_local_unique_molecule_tags,
                    n_local_molecules);
    sort_pairs_copy(d_lowest_idx, d_tmp, d_molecule_length_tmp, d_molecule_length, n_local_molecules);

    // create a global lookup table for lowest idx by molecule tag
    mtl::scatter(d_local_unique_molecule_tags,
                 d_lowest_idx_sort,
                 d_lowest_idx_by_molecule_tag,
                 n_local_molecules,
                 sizeof(unsigned int));

    // sort the list of particles in molecules again according to first particle index, keeping
    // order in molecule
    unsigned int* d_local_molecules_lowest_idx_unsorted = d_tmp;
    mtl::gather(d_local_molecule_tags,
                d_lowest_idx_by_molecule_tag,
                d_local_molecules_lowest_idx_unsorted,
                n_local_ptls_in_molecules,
                sizeof(unsigned int));
    sort_pairs_copy(d_local_molecules_lowest_idx_unsorted,
                    d_local_molecules_lowest_idx,
                    d_idx_sorted_by_molecule_and_tag,
                    d_idx_sorted_by_tag,
                    n_local_ptls_in_molecules);

    // assign local molecule tags to particles
    mtl::fill(d_local_molecule_idx, NO_MOLECULE, nptl);
    mtl::lower_bound(d_lowest_idx_sort,
                     n_local_molecules,
                     d_local_molecules_lowest_idx,
                     n_local_ptls_in_molecules,
                     d_tmp2);
    mtl::scatter(d_idx_sorted_by_tag,
                 d_tmp2,
                 d_local_molecule_idx,
                 n_local_ptls_in_molecules,
                 sizeof(unsigned int));

    if (check_cuda)
        {
        hipError_t error = hipDeviceSynchronize();
        if (error != hipSuccess)
            return error;
        }
    return hipGetLastError();
    }

hipError_t gpu_fill_molecule_table(unsigned int nptl,
                                   unsigned int n_local_ptls_in_molecules,
                                   Index2D molecule_idx,
                                   const unsigned int* d_molecule_idx,
                                   const unsigned int* d_local_molecule_tags,
                                   const unsigned int* d_idx_sorted_by_tag,
                                   unsigned int* d_molecule_list,
                                   unsigned int* d_molecule_order,
                                   unsigned int block_size,
                                   CachedAllocator& alloc)
    {
    // generate ascending index for every molecule
        {
        mtl::ScratchScope scratch;
        unsigned int* d_order = scratch.allocate<unsigned int>(n_local_ptls_in_molecules + 1);
        mtl::exclusive_scan_by_key(d_local_molecule_tags,
                                   nullptr,
                                   d_order,
                                   n_local_ptls_in_molecules);
        mtl::scatter(d_idx_sorted_by_tag,
                     d_order,
                     d_molecule_order,
                     n_local_ptls_in_molecules,
                     sizeof(unsigned int));
        }

    // write out the table
    const char* name = "hoomd::md::gpu_fill_molecule_table_kernel";
    unsigned int run_block_size
        = std::min(block_size, mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name));

    kernel_args::gpu_fill_molecule_table_args args;
    args.nptl = nptl;
    args.molecule_idx = molecule_idx;
    args.d_molecule_idx = d_molecule_idx;
    args.d_molecule_list = d_molecule_list;
    args.d_molecule_order = d_molecule_order;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(nptl / run_block_size + 1),
                       dim3(run_block_size),
                       args);
    }

    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
