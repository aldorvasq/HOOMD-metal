// Copyright (c) 2026 Aldo Vásquez-Briceño.
// Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

/*! \file Primitives.metal
    \brief Metal kernels and host drivers for the device-wide primitives in Primitives.h.

    Scans use three phases: reduce each tile, scan the tile totals (recursively), then scan each
    tile with its offset. Apple GPUs do not guarantee forward progress between threadgroups, so
    single pass decoupled look-back scans are not used.

    The radix sort is a stable least significant digit sort with 4-bit digits. Each pass builds a
    per-tile digit histogram, scans the histograms in digit-major order, and scatters each tile.
    Ranks within a tile come from SIMD-group ballots, which keeps the sort stable.
*/

#include "hoomd/mtl/MetalKernel.h"

#ifndef __METAL_VERSION__
#include "hoomd/mtl/Primitives.h"

#include <algorithm>
#include <stdexcept>
#include <vector>
#endif

//! Threads per threadgroup
#define HOOMD_MTL_NT 256
//! Items per thread
#define HOOMD_MTL_ITEMS 8
//! Items per tile
#define HOOMD_MTL_TILE (HOOMD_MTL_NT * HOOMD_MTL_ITEMS)
//! SIMD-groups per threadgroup
#define HOOMD_MTL_NWARPS (HOOMD_MTL_NT / 32)
//! Radix sort digit size in bits
#define HOOMD_MTL_RADIX_BITS 4
#define HOOMD_MTL_RADIX (1 << HOOMD_MTL_RADIX_BITS)

namespace hoomd
    {
namespace mtl
    {
template<class T> struct scan_args
    {
    HOOMD_DEVPTR(const T) d_in;
    HOOMD_DEVPTR(T) d_out;
    HOOMD_DEVPTR(T) d_tile_values; //!< Tile totals (reduce) or scanned tile offsets (scan)
    unsigned int n;
    unsigned int n_tiles;
    };

template<class T> struct reduce_args
    {
    HOOMD_DEVPTR(const T) d_in;
    HOOMD_DEVPTR(T) d_partial;
    HOOMD_DEVPTR(T) d_out;
    unsigned int n;
    unsigned int n_tiles;
    unsigned int n_partial;
    };

struct radix_args
    {
    HOOMD_DEVPTR(const unsigned int) d_keys_in;
    HOOMD_DEVPTR(const unsigned int) d_values_in;
    HOOMD_DEVPTR(unsigned int) d_keys_out;
    HOOMD_DEVPTR(unsigned int) d_values_out;
    HOOMD_DEVPTR(unsigned int) d_hist;
    unsigned int n;
    unsigned int n_tiles;
    unsigned int shift;
    unsigned int mask;
    unsigned int has_values;
    };

template<class F> struct flags_args
    {
    HOOMD_DEVPTR(const F) d_flags;
    HOOMD_DEVPTR(unsigned int) d_out;
    unsigned int n;
    };

struct select_args
    {
    HOOMD_DEVPTR(const unsigned int) d_in;
    HOOMD_DEVPTR(const unsigned int) d_flags;
    HOOMD_DEVPTR(const unsigned int) d_scan;
    HOOMD_DEVPTR(unsigned int) d_out;
    HOOMD_DEVPTR(unsigned int) d_num_selected;
    unsigned int n;
    };

struct search_args
    {
    HOOMD_DEVPTR(const unsigned int) d_sorted;
    HOOMD_DEVPTR(const unsigned int) d_values;
    HOOMD_DEVPTR(unsigned int) d_out;
    unsigned int n;
    unsigned int m;
    };

struct segment_args
    {
    HOOMD_DEVPTR(const unsigned int) d_keys;
    HOOMD_DEVPTR(const unsigned int) d_values;
    HOOMD_DEVPTR(const unsigned int) d_aux;
    HOOMD_DEVPTR(unsigned int) d_out;
    HOOMD_DEVPTR(unsigned int) d_keys_out;
    HOOMD_DEVPTR(unsigned int) d_num_runs;
    unsigned int n;
    };

struct fill_args
    {
    HOOMD_DEVPTR(unsigned int) d_out;
    unsigned int value;
    unsigned int n;
    };

struct move_args
    {
    HOOMD_DEVPTR(const unsigned int) d_map;
    HOOMD_DEVPTR(const unsigned int) d_in;
    HOOMD_DEVPTR(unsigned int) d_out;
    unsigned int n;
    unsigned int words; //!< 32-bit words per element
    };

    } // end namespace mtl
    } // end namespace hoomd

#ifdef __METAL_VERSION__

namespace hoomd
    {
namespace mtl
    {
template<class T> struct OpSum
    {
    static T identity()
        {
        return T(0);
        }
    static T apply(T a, T b)
        {
        return a + b;
        }
    };

template<class T> struct OpMax
    {
    static T identity()
        {
        return numeric_limits<T>::lowest();
        }
    static T apply(T a, T b)
        {
        return max(a, b);
        }
    };

//! SIMD-group shuffles, including 64-bit types (shuffled as two 32-bit halves)
template<class T> inline T prim_shuffle_up(T v, ushort delta)
    {
    return simd_shuffle_up(v, delta);
    }
inline ulong prim_shuffle_up(ulong v, ushort delta)
    {
    return as_type<ulong>(simd_shuffle_up(as_type<uint2>(v), delta));
    }
template<class T> inline T prim_shuffle_xor(T v, ushort mask)
    {
    return simd_shuffle_xor(v, mask);
    }
inline ulong prim_shuffle_xor(ulong v, ushort mask)
    {
    return as_type<ulong>(simd_shuffle_xor(as_type<uint2>(v), mask));
    }

//! Inclusive scan across a SIMD-group
template<class Op, class T> inline T simd_inclusive(T v, ushort lane)
    {
    for (ushort offset = 1; offset < 32; offset <<= 1)
        {
        T other = prim_shuffle_up(v, offset);
        if (lane >= offset)
            v = Op::apply(other, v);
        }
    return v;
    }

//! Reduction across a SIMD-group (result in every lane)
template<class Op, class T> inline T simd_reduce(T v)
    {
    for (ushort offset = 16; offset > 0; offset >>= 1)
        v = Op::apply(v, prim_shuffle_xor(v, offset));
    return v;
    }

//! Exclusive scan across the threadgroup. All threads must call it.
template<class Op, class T>
inline T block_exclusive_scan(T v, uint tid, ushort lane, threadgroup T* s_warp, thread T& total)
    {
    uint warp = tid / 32;
    T inclusive = simd_inclusive<Op>(v, lane);
    if (lane == 31)
        s_warp[warp] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0)
        {
        T w = (lane < HOOMD_MTL_NWARPS) ? s_warp[lane] : Op::identity();
        T wi = simd_inclusive<Op>(w, lane);
        if (lane < HOOMD_MTL_NWARPS)
            s_warp[lane] = wi;
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    T warp_prefix = (warp == 0) ? Op::identity() : s_warp[warp - 1];
    total = s_warp[HOOMD_MTL_NWARPS - 1];
    T exclusive = prim_shuffle_up(inclusive, 1);
    if (lane == 0)
        exclusive = Op::identity();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return Op::apply(warp_prefix, exclusive);
    }

//! Reduction across the threadgroup. The result is valid in thread 0.
template<class Op, class T> inline T block_reduce(T v, uint tid, ushort lane, threadgroup T* s_warp)
    {
    uint warp = tid / 32;
    v = simd_reduce<Op>(v);
    if (lane == 0)
        s_warp[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    T r = Op::identity();
    if (warp == 0)
        {
        T w = (lane < HOOMD_MTL_NWARPS) ? s_warp[lane] : Op::identity();
        r = simd_reduce<Op>(w);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
    }

template<class Op, class T>
inline void tile_reduce_impl(constant scan_args<T>& args,
                             uint tid,
                             ushort lane,
                             uint tile,
                             threadgroup T* s_warp)
    {
    uint start = tile * HOOMD_MTL_TILE;
    T v = Op::identity();
    for (uint k = 0; k < HOOMD_MTL_ITEMS; k++)
        {
        uint i = start + k * HOOMD_MTL_NT + tid;
        if (i < args.n)
            v = Op::apply(v, args.d_in[i]);
        }
    T r = block_reduce<Op>(v, tid, lane, s_warp);
    if (tid == 0)
        args.d_tile_values[tile] = r;
    }

template<class Op, class T>
inline void scan_tile_impl(constant scan_args<T>& args,
                           bool inclusive,
                           uint tid,
                           ushort lane,
                           uint tile,
                           threadgroup T* s_data,
                           threadgroup T* s_warp)
    {
    uint start = tile * HOOMD_MTL_TILE;
    for (uint k = 0; k < HOOMD_MTL_ITEMS; k++)
        {
        uint i = start + k * HOOMD_MTL_NT + tid;
        s_data[k * HOOMD_MTL_NT + tid] = (i < args.n) ? args.d_in[i] : Op::identity();
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    T vals[HOOMD_MTL_ITEMS];
    T sum = Op::identity();
    for (uint j = 0; j < HOOMD_MTL_ITEMS; j++)
        {
        vals[j] = s_data[tid * HOOMD_MTL_ITEMS + j];
        sum = Op::apply(sum, vals[j]);
        }

    T total;
    T prefix = block_exclusive_scan<Op>(sum, tid, lane, s_warp, total);
    if (args.n_tiles > 1)
        prefix = Op::apply(args.d_tile_values[tile], prefix);

    for (uint j = 0; j < HOOMD_MTL_ITEMS; j++)
        {
        if (inclusive)
            {
            prefix = Op::apply(prefix, vals[j]);
            s_data[tid * HOOMD_MTL_ITEMS + j] = prefix;
            }
        else
            {
            s_data[tid * HOOMD_MTL_ITEMS + j] = prefix;
            prefix = Op::apply(prefix, vals[j]);
            }
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < HOOMD_MTL_ITEMS; k++)
        {
        uint i = start + k * HOOMD_MTL_NT + tid;
        if (i < args.n)
            args.d_out[i] = s_data[k * HOOMD_MTL_NT + tid];
        }
    }

template<class Op, class T>
inline void reduce_partial_impl(constant reduce_args<T>& args,
                                uint tid,
                                ushort lane,
                                uint group,
                                uint n_groups,
                                threadgroup T* s_warp)
    {
    T v = Op::identity();
    for (uint tile = group; tile < args.n_tiles; tile += n_groups)
        {
        uint start = tile * HOOMD_MTL_TILE;
        for (uint k = 0; k < HOOMD_MTL_ITEMS; k++)
            {
            uint i = start + k * HOOMD_MTL_NT + tid;
            if (i < args.n)
                v = Op::apply(v, args.d_in[i]);
            }
        }
    T r = block_reduce<Op>(v, tid, lane, s_warp);
    if (tid == 0)
        args.d_partial[group] = r;
    }

template<class Op, class T>
inline void reduce_final_impl(constant reduce_args<T>& args,
                              uint tid,
                              ushort lane,
                              threadgroup T* s_warp)
    {
    T v = Op::identity();
    for (uint i = tid; i < args.n_partial; i += HOOMD_MTL_NT)
        v = Op::apply(v, args.d_partial[i]);
    T r = block_reduce<Op>(v, tid, lane, s_warp);
    if (tid == 0)
        *args.d_out = r;
    }

// Kernels for each operation and type

#define HOOMD_MTL_SCAN_KERNELS(OP, T)                                                          \
    kernel void tile_reduce_##OP##_##T(HOOMD_KERNEL_PARAMS(scan_args<T>))                     \
        {                                                                                      \
        threadgroup T s_warp[HOOMD_MTL_NWARPS];                                               \
        tile_reduce_impl<Op##OP<T>>(args, threadIdx.x, hoomd_lane, blockIdx.x, s_warp);        \
        }                                                                                      \
    kernel void scan_exclusive_##OP##_##T(HOOMD_KERNEL_PARAMS(scan_args<T>))                  \
        {                                                                                      \
        threadgroup T s_data[HOOMD_MTL_TILE];                                                  \
        threadgroup T s_warp[HOOMD_MTL_NWARPS];                                               \
        scan_tile_impl<Op##OP<T>>(args,                                                        \
                                  false,                                                       \
                                  threadIdx.x,                                                 \
                                  hoomd_lane,                                                  \
                                  blockIdx.x,                                                  \
                                  s_data,                                                      \
                                  s_warp);                                                     \
        }                                                                                      \
    kernel void scan_inclusive_##OP##_##T(HOOMD_KERNEL_PARAMS(scan_args<T>))                  \
        {                                                                                      \
        threadgroup T s_data[HOOMD_MTL_TILE];                                                  \
        threadgroup T s_warp[HOOMD_MTL_NWARPS];                                               \
        scan_tile_impl<Op##OP<T>>(args,                                                        \
                                  true,                                                        \
                                  threadIdx.x,                                                 \
                                  hoomd_lane,                                                  \
                                  blockIdx.x,                                                  \
                                  s_data,                                                      \
                                  s_warp);                                                     \
        }                                                                                      \
    kernel void reduce_partial_##OP##_##T(HOOMD_KERNEL_PARAMS(reduce_args<T>))                \
        {                                                                                      \
        threadgroup T s_warp[HOOMD_MTL_NWARPS];                                               \
        reduce_partial_impl<Op##OP<T>>(args,                                                   \
                                       threadIdx.x,                                            \
                                       hoomd_lane,                                             \
                                       blockIdx.x,                                             \
                                       gridDim.x,                                              \
                                       s_warp);                                                \
        }                                                                                      \
    kernel void reduce_final_##OP##_##T(HOOMD_KERNEL_PARAMS(reduce_args<T>))                  \
        {                                                                                      \
        threadgroup T s_warp[HOOMD_MTL_NWARPS];                                               \
        reduce_final_impl<Op##OP<T>>(args, threadIdx.x, hoomd_lane, s_warp);                   \
        }

HOOMD_MTL_SCAN_KERNELS(Sum, uint)
HOOMD_MTL_SCAN_KERNELS(Sum, int)
HOOMD_MTL_SCAN_KERNELS(Sum, float)
HOOMD_MTL_SCAN_KERNELS(Sum, ulong)
HOOMD_MTL_SCAN_KERNELS(Max, uint)
HOOMD_MTL_SCAN_KERNELS(Max, int)
HOOMD_MTL_SCAN_KERNELS(Max, float)

#undef HOOMD_MTL_SCAN_KERNELS

//! Per-tile digit histogram for one radix sort pass
kernel void radix_histogram(HOOMD_KERNEL_PARAMS(radix_args))
    {
    threadgroup atomic_uint s_count[HOOMD_MTL_RADIX];
    uint tid = threadIdx.x;
    uint tile = blockIdx.x;
    if (tid < HOOMD_MTL_RADIX)
        atomic_store_explicit(&s_count[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint start = tile * HOOMD_MTL_TILE;
    for (uint k = 0; k < HOOMD_MTL_ITEMS; k++)
        {
        uint i = start + k * HOOMD_MTL_NT + tid;
        if (i < args.n)
            {
            uint digit = (args.d_keys_in[i] >> args.shift) & args.mask;
            atomic_fetch_add_explicit(&s_count[digit], 1u, memory_order_relaxed);
            }
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HOOMD_MTL_RADIX)
        args.d_hist[tid * args.n_tiles + tile]
            = atomic_load_explicit(&s_count[tid], memory_order_relaxed);
    }

//! Stable scatter of one tile for a radix sort pass
kernel void radix_scatter(HOOMD_KERNEL_PARAMS(radix_args))
    {
    threadgroup uint s_digit_offset[HOOMD_MTL_RADIX];
    threadgroup uint s_warp_count[HOOMD_MTL_RADIX * HOOMD_MTL_NWARPS];

    uint tid = threadIdx.x;
    uint tile = blockIdx.x;
    uint warp = tid / 32;
    uint lane = hoomd_lane;
    uint lanes_below = (1u << lane) - 1u;

    if (tid < HOOMD_MTL_RADIX)
        s_digit_offset[tid] = args.d_hist[tid * args.n_tiles + tile];

    uint start = tile * HOOMD_MTL_TILE;
    for (uint k = 0; k < HOOMD_MTL_ITEMS; k++)
        {
        if (tid < HOOMD_MTL_RADIX * HOOMD_MTL_NWARPS)
            s_warp_count[tid] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint i = start + k * HOOMD_MTL_NT + tid;
        bool valid = i < args.n;
        uint key = valid ? args.d_keys_in[i] : 0u;
        uint digit = (key >> args.shift) & args.mask;

        // lanes of this SIMD-group with the same digit
        uint same = uint(static_cast<simd_vote::vote_t>(simd_ballot(valid)));
        for (uint b = 0; b < HOOMD_MTL_RADIX_BITS; b++)
            {
            bool bit = (digit >> b) & 1u;
            uint vote = uint(static_cast<simd_vote::vote_t>(simd_ballot(bit)));
            same &= bit ? vote : ~vote;
            }
        uint rank = popcount(same & lanes_below);
        if (valid && rank == 0)
            s_warp_count[digit * HOOMD_MTL_NWARPS + warp] = popcount(same);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // exclusive prefix over SIMD-groups for each digit, continuing from previous rounds
        if (tid < HOOMD_MTL_RADIX)
            {
            uint running = s_digit_offset[tid];
            for (uint w = 0; w < HOOMD_MTL_NWARPS; w++)
                {
                uint c = s_warp_count[tid * HOOMD_MTL_NWARPS + w];
                s_warp_count[tid * HOOMD_MTL_NWARPS + w] = running;
                running += c;
                }
            s_digit_offset[tid] = running;
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid)
            {
            uint position = s_warp_count[digit * HOOMD_MTL_NWARPS + warp] + rank;
            args.d_keys_out[position] = key;
            if (args.has_values)
                args.d_values_out[position] = args.d_values_in[i];
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

//! Convert flags of any type to 0 or 1
kernel void flags_to_uint_uint(HOOMD_KERNEL_PARAMS(flags_args<uint>))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < args.n)
        args.d_out[i] = args.d_flags[i] != 0 ? 1u : 0u;
    }

kernel void flags_to_uint_uchar(HOOMD_KERNEL_PARAMS(flags_args<uchar>))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < args.n)
        args.d_out[i] = args.d_flags[i] != 0 ? 1u : 0u;
    }

kernel void select_scatter(HOOMD_KERNEL_PARAMS(select_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n)
        return;
    uint f = args.d_flags[i];
    uint s = args.d_scan[i];
    if (f)
        args.d_out[s] = args.d_in[i];
    if (i == args.n - 1)
        *args.d_num_selected = s + f;
    }

kernel void partition_scatter(HOOMD_KERNEL_PARAMS(select_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n)
        return;
    uint f = args.d_flags[i];
    uint s = args.d_scan[i];
    if (f)
        args.d_out[s] = args.d_in[i];
    else
        args.d_out[args.n - 1 - (i - s)] = args.d_in[i];
    if (i == args.n - 1)
        *args.d_num_selected = s + f;
    }

kernel void search_lower_bound(HOOMD_KERNEL_PARAMS(search_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.m)
        return;
    uint v = args.d_values[i];
    uint lo = 0;
    uint hi = args.n;
    while (lo < hi)
        {
        uint mid = lo + (hi - lo) / 2;
        if (args.d_sorted[mid] < v)
            lo = mid + 1;
        else
            hi = mid;
        }
    args.d_out[i] = lo;
    }

kernel void search_upper_bound(HOOMD_KERNEL_PARAMS(search_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.m)
        return;
    uint v = args.d_values[i];
    uint lo = 0;
    uint hi = args.n;
    while (lo < hi)
        {
        uint mid = lo + (hi - lo) / 2;
        if (args.d_sorted[mid] <= v)
            lo = mid + 1;
        else
            hi = mid;
        }
    args.d_out[i] = lo;
    }

//! d_out[i] = i if element i starts a run of equal keys, else 0
kernel void segment_heads(HOOMD_KERNEL_PARAMS(segment_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n)
        return;
    bool head = (i == 0) || (args.d_keys[i] != args.d_keys[i - 1]);
    args.d_out[i] = head ? i : 0u;
    }

//! d_out[i] = i - head(i), with head(i) in d_aux
kernel void segment_rank(HOOMD_KERNEL_PARAMS(segment_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n)
        return;
    args.d_out[i] = i - args.d_aux[i];
    }

//! d_out[i] = scan[i] - scan[head(i)], with the exclusive scan in d_values and heads in d_aux
kernel void segment_subtract(HOOMD_KERNEL_PARAMS(segment_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n)
        return;
    args.d_out[i] = args.d_values[i] - args.d_values[args.d_aux[i]];
    }

//! d_out[i] = 1 if element i starts a run of equal keys, else 0
kernel void segment_head_flags(HOOMD_KERNEL_PARAMS(segment_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n)
        return;
    args.d_out[i] = ((i == 0) || (args.d_keys[i] != args.d_keys[i - 1])) ? 1u : 0u;
    }

//! Zero the output of each run, with the inclusive scan of the head flags in d_aux
kernel void segment_zero(HOOMD_KERNEL_PARAMS(segment_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n)
        return;
    if ((i == 0) || (args.d_keys[i] != args.d_keys[i - 1]))
        args.d_out[args.d_aux[i] - 1] = 0;
    }

//! Accumulate values by run, with the inclusive scan of the head flags in d_aux
kernel void segment_accumulate(HOOMD_KERNEL_PARAMS(segment_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n)
        return;
    uint run = args.d_aux[i] - 1;
    atomic_fetch_add_explicit((device atomic_uint*)&args.d_out[run],
                              args.d_values[i],
                              memory_order_relaxed);
    if ((i == 0) || (args.d_keys[i] != args.d_keys[i - 1]))
        args.d_keys_out[run] = args.d_keys[i];
    if (i == args.n - 1)
        *args.d_num_runs = args.d_aux[i];
    }

kernel void fill_uint(HOOMD_KERNEL_PARAMS(fill_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < args.n)
        args.d_out[i] = args.value;
    }

kernel void sequence_uint(HOOMD_KERNEL_PARAMS(fill_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < args.n)
        args.d_out[i] = args.value + i;
    }

kernel void gather_words(HOOMD_KERNEL_PARAMS(move_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n)
        return;
    uint src = args.d_map[i];
    for (uint w = 0; w < args.words; w++)
        args.d_out[i * args.words + w] = args.d_in[src * args.words + w];
    }

kernel void scatter_words(HOOMD_KERNEL_PARAMS(move_args))
    {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n)
        return;
    uint dst = args.d_map[i];
    for (uint w = 0; w < args.words; w++)
        args.d_out[dst * args.words + w] = args.d_in[i * args.words + w];
    }

    } // end namespace mtl
    } // end namespace hoomd

#else // host drivers

namespace hoomd
    {
namespace mtl
    {
namespace
    {
//! Pool of device memory chunks used as a stack by ScratchScope
struct ScratchPool
    {
    std::vector<std::pair<char*, size_t>> chunks;
    size_t chunk = 0;
    size_t offset = 0;
    };

ScratchPool& scratch_pool()
    {
    static ScratchPool* pool = new ScratchPool();
    return *pool;
    }

const unsigned int block_size = 256;

dim3 grid_for(unsigned int n)
    {
    return dim3((n + block_size - 1) / block_size);
    }

unsigned int n_tiles_for(unsigned int n)
    {
    return (n + HOOMD_MTL_TILE - 1) / HOOMD_MTL_TILE;
    }

void check(hipError_t error, const char* what)
    {
    if (error != hipSuccess)
        throw std::runtime_error(std::string("Metal primitive ") + what
                                 + " failed: " + hipGetErrorString(error));
    }

template<class T> const char* type_name();
template<> const char* type_name<unsigned int>()
    {
    return "uint";
    }
template<> const char* type_name<int>()
    {
    return "int";
    }
template<> const char* type_name<float>()
    {
    return "float";
    }
template<> const char* type_name<size_t>()
    {
    return "ulong";
    }

std::string kernel_name(const char* base, const char* op, const char* type)
    {
    return std::string("hoomd::mtl::") + base + "_" + op + "_" + type;
    }

template<class Args> void run(const char* kernel, dim3 grid, dim3 block, const Args& args)
    {
    check(launch(HOOMD_MTL_SOURCE, kernel, grid, block, args), kernel);
    }

template<class T>
void scan(const char* op, const T* d_in, T* d_out, unsigned int n, bool inclusive)
    {
    if (n == 0)
        return;
    unsigned int n_tiles = n_tiles_for(n);
    ScratchScope scratch;
    T* d_tiles = n_tiles > 1 ? scratch.allocate<T>(n_tiles) : nullptr;

    scan_args<T> args;
    args.d_in = d_in;
    args.d_out = d_out;
    args.d_tile_values = d_tiles;
    args.n = n;
    args.n_tiles = n_tiles;

    if (n_tiles > 1)
        {
        run(kernel_name("tile_reduce", op, type_name<T>()).c_str(),
            dim3(n_tiles),
            dim3(HOOMD_MTL_NT),
            args);
        scan<T>(op, d_tiles, d_tiles, n_tiles, false);
        }

    run(kernel_name(inclusive ? "scan_inclusive" : "scan_exclusive", op, type_name<T>()).c_str(),
        dim3(n_tiles),
        dim3(HOOMD_MTL_NT),
        args);
    }

template<class T> void reduce(const char* op, const T* d_in, T* d_out, unsigned int n)
    {
    unsigned int n_tiles = n_tiles_for(n);
    unsigned int n_groups = std::max(1u, std::min(n_tiles, 1024u));
    ScratchScope scratch;

    reduce_args<T> args;
    args.d_in = d_in;
    args.d_partial = scratch.allocate<T>(n_groups);
    args.d_out = d_out;
    args.n = n;
    args.n_tiles = n_tiles;
    args.n_partial = n_groups;

    run(kernel_name("reduce_partial", op, type_name<T>()).c_str(),
        dim3(n_groups),
        dim3(HOOMD_MTL_NT),
        args);
    run(kernel_name("reduce_final", op, type_name<T>()).c_str(), dim3(1), dim3(HOOMD_MTL_NT), args);
    }

template<class F>
void select(const char* kernel,
            const char* flag_kernel,
            const unsigned int* d_in,
            const F* d_flags,
            unsigned int* d_out,
            unsigned int* d_num_selected,
            unsigned int n)
    {
    if (n == 0)
        {
        check(hipMemsetAsync(d_num_selected, 0, sizeof(unsigned int)), "select");
        return;
        }

    ScratchScope scratch;
    unsigned int* d_ones = scratch.allocate<unsigned int>(n);
    unsigned int* d_scan = scratch.allocate<unsigned int>(n);

    flags_args<F> fargs;
    fargs.d_flags = d_flags;
    fargs.d_out = d_ones;
    fargs.n = n;
    run(flag_kernel, grid_for(n), dim3(block_size), fargs);

    exclusive_scan(d_ones, d_scan, n);

    select_args sargs;
    sargs.d_in = d_in;
    sargs.d_flags = d_ones;
    sargs.d_scan = d_scan;
    sargs.d_out = d_out;
    sargs.d_num_selected = d_num_selected;
    sargs.n = n;
    run(kernel, grid_for(n), dim3(block_size), sargs);
    }

void search(const char* kernel,
            const unsigned int* d_sorted,
            unsigned int n,
            const unsigned int* d_values,
            unsigned int m,
            unsigned int* d_out)
    {
    if (m == 0)
        return;
    search_args args;
    args.d_sorted = d_sorted;
    args.d_values = d_values;
    args.d_out = d_out;
    args.n = n;
    args.m = m;
    run(kernel, grid_for(m), dim3(block_size), args);
    }

void move(const char* kernel,
          const unsigned int* d_map,
          const void* d_in,
          void* d_out,
          unsigned int n,
          unsigned int element_bytes)
    {
    if (n == 0)
        return;
    if (element_bytes == 0 || element_bytes % 4 != 0)
        throw std::runtime_error("Metal gather/scatter requires elements that are a multiple of 4 "
                                 "bytes");
    move_args args;
    args.d_map = d_map;
    args.d_in = reinterpret_cast<const unsigned int*>(d_in);
    args.d_out = reinterpret_cast<unsigned int*>(d_out);
    args.n = n;
    args.words = element_bytes / 4;
    run(kernel, grid_for(n), dim3(block_size), args);
    }

    } // end anonymous namespace

ScratchScope::ScratchScope()
    {
    ScratchPool& pool = scratch_pool();
    m_mark_chunk = pool.chunk;
    m_mark_offset = pool.offset;
    }

ScratchScope::~ScratchScope()
    {
    ScratchPool& pool = scratch_pool();
    pool.chunk = m_mark_chunk;
    pool.offset = m_mark_offset;
    }

void* ScratchScope::allocate(size_t bytes)
    {
    ScratchPool& pool = scratch_pool();
    bytes = std::max<size_t>(256, (bytes + 255) & ~size_t(255));
    while (pool.chunk < pool.chunks.size())
        {
        if (pool.offset + bytes <= pool.chunks[pool.chunk].second)
            {
            void* result = pool.chunks[pool.chunk].first + pool.offset;
            pool.offset += bytes;
            return result;
            }
        pool.chunk++;
        pool.offset = 0;
        }

    size_t previous = pool.chunks.empty() ? 0 : pool.chunks.back().second;
    size_t size = std::max(bytes, std::max<size_t>(size_t(1) << 20, 2 * previous));
    void* ptr = nullptr;
    check(hipMalloc(&ptr, size), "scratch allocation");
    pool.chunks.emplace_back(reinterpret_cast<char*>(ptr), size);
    pool.chunk = pool.chunks.size() - 1;
    pool.offset = bytes;
    return ptr;
    }

void exclusive_scan(const unsigned int* d_in, unsigned int* d_out, unsigned int n)
    {
    scan("Sum", d_in, d_out, n, false);
    }

void exclusive_scan(const int* d_in, int* d_out, unsigned int n)
    {
    scan("Sum", d_in, d_out, n, false);
    }

void exclusive_scan(const float* d_in, float* d_out, unsigned int n)
    {
    scan("Sum", d_in, d_out, n, false);
    }

void exclusive_scan(const size_t* d_in, size_t* d_out, unsigned int n)
    {
    scan("Sum", d_in, d_out, n, false);
    }

void inclusive_scan(const size_t* d_in, size_t* d_out, unsigned int n)
    {
    scan("Sum", d_in, d_out, n, true);
    }

void inclusive_scan(const unsigned int* d_in, unsigned int* d_out, unsigned int n)
    {
    scan("Sum", d_in, d_out, n, true);
    }

void inclusive_scan(const int* d_in, int* d_out, unsigned int n)
    {
    scan("Sum", d_in, d_out, n, true);
    }

void inclusive_scan(const float* d_in, float* d_out, unsigned int n)
    {
    scan("Sum", d_in, d_out, n, true);
    }

void inclusive_scan_max(const unsigned int* d_in, unsigned int* d_out, unsigned int n)
    {
    scan("Max", d_in, d_out, n, true);
    }

void reduce_sum(const unsigned int* d_in, unsigned int* d_out, unsigned int n)
    {
    reduce("Sum", d_in, d_out, n);
    }

void reduce_sum(const int* d_in, int* d_out, unsigned int n)
    {
    reduce("Sum", d_in, d_out, n);
    }

void reduce_sum(const float* d_in, float* d_out, unsigned int n)
    {
    reduce("Sum", d_in, d_out, n);
    }

void reduce_max(const unsigned int* d_in, unsigned int* d_out, unsigned int n)
    {
    reduce("Max", d_in, d_out, n);
    }

void reduce_max(const int* d_in, int* d_out, unsigned int n)
    {
    reduce("Max", d_in, d_out, n);
    }

void reduce_max(const float* d_in, float* d_out, unsigned int n)
    {
    reduce("Max", d_in, d_out, n);
    }

unsigned int reduce_sum(const unsigned int* d_in, unsigned int n)
    {
    ScratchScope scratch;
    unsigned int* d_result = scratch.allocate<unsigned int>(1);
    reduce_sum(d_in, d_result, n);
    check(hipDeviceSynchronize(), "reduce_sum");
    return *d_result;
    }

unsigned int reduce_max(const unsigned int* d_in, unsigned int n)
    {
    ScratchScope scratch;
    unsigned int* d_result = scratch.allocate<unsigned int>(1);
    reduce_max(d_in, d_result, n);
    check(hipDeviceSynchronize(), "reduce_max");
    return *d_result;
    }

void sort_pairs(unsigned int* d_keys,
                unsigned int* d_values,
                unsigned int n,
                unsigned int begin_bit,
                unsigned int end_bit)
    {
    if (n <= 1 || begin_bit >= end_bit)
        return;
    end_bit = std::min(end_bit, 32u);

    unsigned int n_tiles = n_tiles_for(n);
    ScratchScope scratch;
    unsigned int* keys_alt = scratch.allocate<unsigned int>(n);
    unsigned int* values_alt = d_values ? scratch.allocate<unsigned int>(n) : nullptr;
    unsigned int* hist = scratch.allocate<unsigned int>(HOOMD_MTL_RADIX * n_tiles);

    unsigned int* keys_in = d_keys;
    unsigned int* values_in = d_values;
    unsigned int* keys_out = keys_alt;
    unsigned int* values_out = values_alt;

    for (unsigned int shift = begin_bit; shift < end_bit; shift += HOOMD_MTL_RADIX_BITS)
        {
        unsigned int bits = std::min<unsigned int>(HOOMD_MTL_RADIX_BITS, end_bit - shift);
        radix_args args;
        args.d_keys_in = keys_in;
        args.d_values_in = values_in;
        args.d_keys_out = keys_out;
        args.d_values_out = values_out;
        args.d_hist = hist;
        args.n = n;
        args.n_tiles = n_tiles;
        args.shift = shift;
        args.mask = (1u << bits) - 1u;
        args.has_values = d_values != nullptr;

        run("hoomd::mtl::radix_histogram", dim3(n_tiles), dim3(HOOMD_MTL_NT), args);
        exclusive_scan(hist, hist, HOOMD_MTL_RADIX * n_tiles);
        run("hoomd::mtl::radix_scatter", dim3(n_tiles), dim3(HOOMD_MTL_NT), args);

        std::swap(keys_in, keys_out);
        std::swap(values_in, values_out);
        }

    if (keys_in != d_keys)
        {
        check(hipMemcpyAsync(d_keys,
                             keys_in,
                             sizeof(unsigned int) * n,
                             hipMemcpyDeviceToDevice),
              "sort_pairs");
        if (d_values)
            check(hipMemcpyAsync(d_values,
                                 values_in,
                                 sizeof(unsigned int) * n,
                                 hipMemcpyDeviceToDevice),
                  "sort_pairs");
        }
    }

void select_flagged(const unsigned int* d_in,
                    const unsigned int* d_flags,
                    unsigned int* d_out,
                    unsigned int* d_num_selected,
                    unsigned int n)
    {
    select("hoomd::mtl::select_scatter",
           "hoomd::mtl::flags_to_uint_uint",
           d_in,
           d_flags,
           d_out,
           d_num_selected,
           n);
    }

void select_flagged(const unsigned int* d_in,
                    const unsigned char* d_flags,
                    unsigned int* d_out,
                    unsigned int* d_num_selected,
                    unsigned int n)
    {
    select("hoomd::mtl::select_scatter",
           "hoomd::mtl::flags_to_uint_uchar",
           d_in,
           d_flags,
           d_out,
           d_num_selected,
           n);
    }

void partition_flagged(const unsigned int* d_in,
                       const unsigned int* d_flags,
                       unsigned int* d_out,
                       unsigned int* d_num_selected,
                       unsigned int n)
    {
    select("hoomd::mtl::partition_scatter",
           "hoomd::mtl::flags_to_uint_uint",
           d_in,
           d_flags,
           d_out,
           d_num_selected,
           n);
    }

void partition_flagged(const unsigned int* d_in,
                       const unsigned char* d_flags,
                       unsigned int* d_out,
                       unsigned int* d_num_selected,
                       unsigned int n)
    {
    select("hoomd::mtl::partition_scatter",
           "hoomd::mtl::flags_to_uint_uchar",
           d_in,
           d_flags,
           d_out,
           d_num_selected,
           n);
    }

void lower_bound(const unsigned int* d_sorted,
                 unsigned int n,
                 const unsigned int* d_values,
                 unsigned int m,
                 unsigned int* d_out)
    {
    search("hoomd::mtl::search_lower_bound", d_sorted, n, d_values, m, d_out);
    }

void upper_bound(const unsigned int* d_sorted,
                 unsigned int n,
                 const unsigned int* d_values,
                 unsigned int m,
                 unsigned int* d_out)
    {
    search("hoomd::mtl::search_upper_bound", d_sorted, n, d_values, m, d_out);
    }

void exclusive_scan_by_key(const unsigned int* d_keys,
                           const unsigned int* d_values,
                           unsigned int* d_out,
                           unsigned int n)
    {
    if (n == 0)
        return;
    ScratchScope scratch;
    unsigned int* d_heads = scratch.allocate<unsigned int>(n);

    segment_args args;
    args.d_keys = d_keys;
    args.d_out = d_heads;
    args.n = n;
    run("hoomd::mtl::segment_heads", grid_for(n), dim3(block_size), args);
    inclusive_scan_max(d_heads, d_heads, n);

    args.d_aux = d_heads;
    args.d_out = d_out;
    if (!d_values)
        {
        run("hoomd::mtl::segment_rank", grid_for(n), dim3(block_size), args);
        }
    else
        {
        unsigned int* d_scan = scratch.allocate<unsigned int>(n);
        exclusive_scan(d_values, d_scan, n);
        args.d_values = d_scan;
        run("hoomd::mtl::segment_subtract", grid_for(n), dim3(block_size), args);
        }
    }

void reduce_by_key(const unsigned int* d_keys,
                   const unsigned int* d_values,
                   unsigned int* d_keys_out,
                   unsigned int* d_values_out,
                   unsigned int* d_num_runs,
                   unsigned int n)
    {
    if (n == 0)
        {
        check(hipMemsetAsync(d_num_runs, 0, sizeof(unsigned int)), "reduce_by_key");
        return;
        }
    ScratchScope scratch;
    unsigned int* d_runs = scratch.allocate<unsigned int>(n);

    segment_args args;
    args.d_keys = d_keys;
    args.d_values = d_values;
    args.d_out = d_runs;
    args.n = n;
    run("hoomd::mtl::segment_head_flags", grid_for(n), dim3(block_size), args);
    inclusive_scan(d_runs, d_runs, n);

    args.d_aux = d_runs;
    args.d_out = d_values_out;
    args.d_keys_out = d_keys_out;
    args.d_num_runs = d_num_runs;
    run("hoomd::mtl::segment_zero", grid_for(n), dim3(block_size), args);
    run("hoomd::mtl::segment_accumulate", grid_for(n), dim3(block_size), args);
    }

void fill(unsigned int* d_out, unsigned int value, unsigned int n)
    {
    if (n == 0)
        return;
    fill_args args;
    args.d_out = d_out;
    args.value = value;
    args.n = n;
    run("hoomd::mtl::fill_uint", grid_for(n), dim3(block_size), args);
    }

void sequence(unsigned int* d_out, unsigned int n, unsigned int first)
    {
    if (n == 0)
        return;
    fill_args args;
    args.d_out = d_out;
    args.value = first;
    args.n = n;
    run("hoomd::mtl::sequence_uint", grid_for(n), dim3(block_size), args);
    }

void gather(const unsigned int* d_map,
            const void* d_in,
            void* d_out,
            unsigned int n,
            unsigned int element_bytes)
    {
    move("hoomd::mtl::gather_words", d_map, d_in, d_out, n, element_bytes);
    }

void scatter(const unsigned int* d_map,
             const void* d_in,
             void* d_out,
             unsigned int n,
             unsigned int element_bytes)
    {
    move("hoomd::mtl::scatter_words", d_map, d_in, d_out, n, element_bytes);
    }

    } // end namespace mtl
    } // end namespace hoomd

#endif
