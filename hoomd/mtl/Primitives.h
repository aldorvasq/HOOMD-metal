// Copyright (c) 2026 Aldo Vasquez-Briceno.
// Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

/*! \file Primitives.h
    \brief Device-wide parallel primitives for the Metal GPU platform.

    These replace the CUB and Thrust algorithms used by HOOMD-blue's CUDA and HIP kernels. All
    functions operate on device pointers from the HIP API. They are asynchronous unless they return
    a value to the host. Temporary storage comes from an internal pool, so callers do not allocate
    it.
*/

#pragma once

#ifdef __METAL_VERSION__
#error Primitives.h is a host-only header.
#endif

#include "hoomd/mtl/MetalRuntime.h"

namespace hoomd
    {
namespace mtl
    {
//! Exclusive prefix sum: d_out[i] = d_in[0] + ... + d_in[i-1]. d_out may equal d_in.
HIPMETAL_API void exclusive_scan(const unsigned int* d_in, unsigned int* d_out, unsigned int n);
HIPMETAL_API void exclusive_scan(const int* d_in, int* d_out, unsigned int n);
HIPMETAL_API void exclusive_scan(const float* d_in, float* d_out, unsigned int n);
HIPMETAL_API void exclusive_scan(const size_t* d_in, size_t* d_out, unsigned int n);

//! Inclusive prefix sum: d_out[i] = d_in[0] + ... + d_in[i]. d_out may equal d_in.
HIPMETAL_API void inclusive_scan(const unsigned int* d_in, unsigned int* d_out, unsigned int n);
HIPMETAL_API void inclusive_scan(const int* d_in, int* d_out, unsigned int n);
HIPMETAL_API void inclusive_scan(const float* d_in, float* d_out, unsigned int n);
HIPMETAL_API void inclusive_scan(const size_t* d_in, size_t* d_out, unsigned int n);

//! Inclusive prefix maximum
HIPMETAL_API void inclusive_scan_max(const unsigned int* d_in, unsigned int* d_out, unsigned int n);

//! Sum of d_in[0..n) written to *d_out (device memory). Writes 0 when n is 0.
HIPMETAL_API void reduce_sum(const unsigned int* d_in, unsigned int* d_out, unsigned int n);
HIPMETAL_API void reduce_sum(const int* d_in, int* d_out, unsigned int n);
HIPMETAL_API void reduce_sum(const float* d_in, float* d_out, unsigned int n);

//! Maximum of d_in[0..n) written to *d_out. Writes the lowest value of the type when n is 0.
HIPMETAL_API void reduce_max(const unsigned int* d_in, unsigned int* d_out, unsigned int n);
HIPMETAL_API void reduce_max(const int* d_in, int* d_out, unsigned int n);
HIPMETAL_API void reduce_max(const float* d_in, float* d_out, unsigned int n);

//! Sum of d_in[0..n) returned to the host (synchronizes)
HIPMETAL_API unsigned int reduce_sum(const unsigned int* d_in, unsigned int n);

//! Maximum of d_in[0..n) returned to the host (synchronizes)
HIPMETAL_API unsigned int reduce_max(const unsigned int* d_in, unsigned int n);

//! Stable radix sort of key/value pairs in place
/*! Only bits [begin_bit, end_bit) of the keys are compared. d_values may be nullptr to sort keys
    only.
*/
HIPMETAL_API void sort_pairs(unsigned int* d_keys,
                             unsigned int* d_values,
                             unsigned int n,
                             unsigned int begin_bit = 0,
                             unsigned int end_bit = 32);

//! Stable radix sort of keys in place
inline void sort_keys(unsigned int* d_keys,
                      unsigned int n,
                      unsigned int begin_bit = 0,
                      unsigned int end_bit = 32)
    {
    sort_pairs(d_keys, nullptr, n, begin_bit, end_bit);
    }

//! Copy the flagged items of d_in to d_out, preserving order (CUB DeviceSelect::Flagged)
/*! Items with a nonzero flag are selected. The number selected is written to *d_num_selected
    (device memory).
*/
HIPMETAL_API void select_flagged(const unsigned int* d_in,
                                 const unsigned int* d_flags,
                                 unsigned int* d_out,
                                 unsigned int* d_num_selected,
                                 unsigned int n);
HIPMETAL_API void select_flagged(const unsigned int* d_in,
                                 const unsigned char* d_flags,
                                 unsigned int* d_out,
                                 unsigned int* d_num_selected,
                                 unsigned int n);

//! Partition d_in into d_out (CUB DevicePartition::Flagged)
/*! Flagged items are copied to the front of d_out in order and unflagged items to the back in
    reverse order. The number of flagged items is written to *d_num_selected (device memory).
*/
HIPMETAL_API void partition_flagged(const unsigned int* d_in,
                                    const unsigned int* d_flags,
                                    unsigned int* d_out,
                                    unsigned int* d_num_selected,
                                    unsigned int n);
HIPMETAL_API void partition_flagged(const unsigned int* d_in,
                                    const unsigned char* d_flags,
                                    unsigned int* d_out,
                                    unsigned int* d_num_selected,
                                    unsigned int n);

//! For each value, the index of the first element of d_sorted that is not less than it
HIPMETAL_API void lower_bound(const unsigned int* d_sorted,
                              unsigned int n,
                              const unsigned int* d_values,
                              unsigned int m,
                              unsigned int* d_out);

//! For each value, the index of the first element of d_sorted that is greater than it
HIPMETAL_API void upper_bound(const unsigned int* d_sorted,
                              unsigned int n,
                              const unsigned int* d_values,
                              unsigned int m,
                              unsigned int* d_out);

//! Exclusive prefix sum within runs of equal keys (thrust::exclusive_scan_by_key)
/*! d_values may be nullptr, in which case every value is 1 and d_out[i] is the rank of element i
    within its run.
*/
HIPMETAL_API void exclusive_scan_by_key(const unsigned int* d_keys,
                                        const unsigned int* d_values,
                                        unsigned int* d_out,
                                        unsigned int n);

//! Sum values over runs of equal keys (thrust::reduce_by_key)
/*! Writes the unique keys to d_keys_out, the sums to d_values_out, and the number of runs to
    *d_num_runs (device memory).
*/
HIPMETAL_API void reduce_by_key(const unsigned int* d_keys,
                                const unsigned int* d_values,
                                unsigned int* d_keys_out,
                                unsigned int* d_values_out,
                                unsigned int* d_num_runs,
                                unsigned int n);

//! Fill d_out[0..n) with value
HIPMETAL_API void fill(unsigned int* d_out, unsigned int value, unsigned int n);

//! d_out[i] = first + i
HIPMETAL_API void sequence(unsigned int* d_out, unsigned int n, unsigned int first = 0);

//! d_out[i] = d_in[d_map[i]] for 32-bit (4 byte) elements
HIPMETAL_API void gather(const unsigned int* d_map,
                         const void* d_in,
                         void* d_out,
                         unsigned int n,
                         unsigned int element_bytes);

//! d_out[d_map[i]] = d_in[i] for elements of 4, 8, or 16 bytes
HIPMETAL_API void scatter(const unsigned int* d_map,
                          const void* d_in,
                          void* d_out,
                          unsigned int n,
                          unsigned int element_bytes);

//! Temporary device memory from the primitives' pool
/*! Allocations made while a ScratchScope is alive remain valid until it is destroyed. Scopes nest.
    The memory may be reused by later GPU work only after earlier work has completed, which the
    in-order command queue guarantees.
*/
class HIPMETAL_API ScratchScope
    {
    public:
    ScratchScope();
    ~ScratchScope();
    ScratchScope(const ScratchScope&) = delete;
    ScratchScope& operator=(const ScratchScope&) = delete;

    //! Allocate bytes (aligned to 256 bytes)
    void* allocate(size_t bytes);

    template<class T> T* allocate(size_t n)
        {
        return reinterpret_cast<T*>(allocate(n * sizeof(T)));
        }

    private:
    size_t m_mark_chunk;
    size_t m_mark_offset;
    };

    } // end namespace mtl
    } // end namespace hoomd
