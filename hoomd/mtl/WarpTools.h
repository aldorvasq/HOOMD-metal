// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file WarpTools.h
    \brief Metal Shading Language versions of the warp primitives in hoomd/WarpTools.cuh.

    The classes have the API of hoomd::detail::WarpReduce and hoomd::detail::WarpScan, which wrap
    CUB. Metal code has no implicit lane index, so the constructors take the thread's index in its
    SIMD-group (the kernel parameter hoomd_lane). A logical warp is a group of LOGICAL_WARP_THREADS
    consecutive lanes. As with CUB, every lane of the SIMD-group must call the collective.
*/

#pragma once

#ifndef __METAL_VERSION__
#error hoomd/mtl/WarpTools.h is Metal Shading Language only.
#endif

#include "hoomd/mtl/MetalDevice.h"

namespace hoomd
    {
namespace detail
    {
//! Shuffle any 4-byte based scalar, vector, or packed vector type
template<class T> inline T warp_shuffle(T v, ushort lane)
    {
    return simd_shuffle(v, lane);
    }
inline packed_float3 warp_shuffle(packed_float3 v, ushort lane)
    {
    return packed_float3(simd_shuffle(float3(v), lane));
    }
inline packed_int3 warp_shuffle(packed_int3 v, ushort lane)
    {
    return packed_int3(simd_shuffle(int3(v), lane));
    }
inline packed_uint3 warp_shuffle(packed_uint3 v, ushort lane)
    {
    return packed_uint3(simd_shuffle(uint3(v), lane));
    }
inline ulong warp_shuffle(ulong v, ushort lane)
    {
    return as_type<ulong>(simd_shuffle(as_type<uint2>(v), lane));
    }
inline bool warp_shuffle(bool v, ushort lane)
    {
    return simd_shuffle(uint(v), lane) != 0;
    }

template<class T> inline T warp_shuffle_down(T v, ushort delta)
    {
    return simd_shuffle_down(v, delta);
    }
inline packed_float3 warp_shuffle_down(packed_float3 v, ushort delta)
    {
    return packed_float3(simd_shuffle_down(float3(v), delta));
    }
inline bool warp_shuffle_down(bool v, ushort delta)
    {
    return simd_shuffle_down(uint(v), delta) != 0;
    }

template<class T> inline T warp_shuffle_up(T v, ushort delta)
    {
    return simd_shuffle_up(v, delta);
    }
inline packed_float3 warp_shuffle_up(packed_float3 v, ushort delta)
    {
    return packed_float3(simd_shuffle_up(float3(v), delta));
    }
inline bool warp_shuffle_up(bool v, ushort delta)
    {
    return simd_shuffle_up(uint(v), delta) != 0;
    }

struct WarpSum
    {
    template<class T> T operator()(T a, T b) const
        {
        return a + b;
        }
    };

struct WarpMax
    {
    template<class T> T operator()(T a, T b) const
        {
        return max(a, b);
        }
    };

struct WarpMin
    {
    template<class T> T operator()(T a, T b) const
        {
        return min(a, b);
        }
    };

//! Reduction over a logical warp. The result is valid in the logical warp's first lane.
template<typename T, int LOGICAL_WARP_THREADS = 32> class WarpReduce
    {
    static_assert(LOGICAL_WARP_THREADS <= 32, "Logical warp size cannot exceed 32");
    static_assert(LOGICAL_WARP_THREADS && !(LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)),
                  "Logical warp size must be a power of 2");

    public:
    explicit WarpReduce(ushort lane) : m_lane(lane) { }

    T Sum(T input)
        {
        return Reduce(input, WarpSum());
        }

    T Sum(T input, int valid_items)
        {
        return Reduce(input, WarpSum(), valid_items);
        }

    template<typename ReduceOpT> T Reduce(T input, ReduceOpT reduce_op)
        {
        T v = input;
        for (ushort offset = LOGICAL_WARP_THREADS / 2; offset > 0; offset >>= 1)
            {
            T other = warp_shuffle_down(v, offset);
            if ((m_lane % LOGICAL_WARP_THREADS) + offset < LOGICAL_WARP_THREADS)
                v = reduce_op(v, other);
            }
        return v;
        }

    template<typename ReduceOpT> T Reduce(T input, ReduceOpT reduce_op, int valid_items)
        {
        T v = input;
        int logical_lane = m_lane % LOGICAL_WARP_THREADS;
        for (ushort offset = LOGICAL_WARP_THREADS / 2; offset > 0; offset >>= 1)
            {
            T other = warp_shuffle_down(v, offset);
            if (logical_lane + offset < valid_items)
                v = reduce_op(v, other);
            }
        return v;
        }

    private:
    ushort m_lane;
    };

//! Scans over a logical warp
template<typename T, int LOGICAL_WARP_THREADS = 32> class WarpScan
    {
    static_assert(LOGICAL_WARP_THREADS <= 32, "Logical warp size cannot exceed 32");
    static_assert(LOGICAL_WARP_THREADS && !(LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)),
                  "Logical warp size must be a power of 2");

    public:
    explicit WarpScan(ushort lane) : m_lane(lane) { }

    void InclusiveSum(T input, thread T& output)
        {
        InclusiveScan(input, output, WarpSum());
        }

    void InclusiveSum(T input, thread T& output, thread T& aggregate)
        {
        InclusiveScan(input, output, WarpSum(), aggregate);
        }

    template<class ScanOpT> void InclusiveScan(T input, thread T& output, ScanOpT scan_op)
        {
        T v = input;
        ushort logical_lane = m_lane % LOGICAL_WARP_THREADS;
        for (ushort offset = 1; offset < LOGICAL_WARP_THREADS; offset <<= 1)
            {
            T other = warp_shuffle_up(v, offset);
            if (logical_lane >= offset)
                v = scan_op(other, v);
            }
        output = v;
        }

    template<class ScanOpT>
    void InclusiveScan(T input, thread T& output, ScanOpT scan_op, thread T& aggregate)
        {
        InclusiveScan(input, output, scan_op);
        aggregate = warp_shuffle(output, last_lane());
        }

    void ExclusiveSum(T input, thread T& output)
        {
        T inclusive;
        InclusiveScan(input, inclusive, WarpSum());
        output = inclusive - input;
        }

    void ExclusiveSum(T input, thread T& output, thread T& aggregate)
        {
        T inclusive;
        InclusiveScan(input, inclusive, WarpSum());
        output = inclusive - input;
        aggregate = warp_shuffle(inclusive, last_lane());
        }

    template<class ScanOpT> void ExclusiveScan(T input, thread T& output, T initial, ScanOpT scan_op)
        {
        T inclusive;
        InclusiveScan(input, inclusive, scan_op);
        T shifted = warp_shuffle_up(inclusive, 1);
        output = (m_lane % LOGICAL_WARP_THREADS == 0) ? initial : scan_op(initial, shifted);
        }

    template<class ScanOpT>
    void ExclusiveScan(T input, thread T& output, T initial, ScanOpT scan_op, thread T& aggregate)
        {
        T inclusive;
        InclusiveScan(input, inclusive, scan_op);
        T shifted = warp_shuffle_up(inclusive, 1);
        output = (m_lane % LOGICAL_WARP_THREADS == 0) ? initial : scan_op(initial, shifted);
        aggregate = warp_shuffle(inclusive, last_lane());
        }

    T Broadcast(T input, unsigned int src_lane)
        {
        return warp_shuffle(input, ushort(first_lane() + src_lane));
        }

    private:
    ushort m_lane;

    ushort first_lane() const
        {
        return m_lane - (m_lane % LOGICAL_WARP_THREADS);
        }

    ushort last_lane() const
        {
        return first_lane() + LOGICAL_WARP_THREADS - 1;
        }
    };

// Logical warp operations with a run-time width. Kernels specialize the width with a Metal
// function constant, which the compiler folds like a template parameter.

//! Value of v in lane src of the caller's logical warp of width lanes
template<class T> inline T warp_broadcast(T v, ushort lane, uint width, uint src)
    {
    return warp_shuffle(v, ushort(lane - lane % width + src));
    }

//! Exclusive prefix sum over the caller's logical warp. aggregate is set to the warp's total.
template<class T> inline T warp_exclusive_sum(T v, ushort lane, uint width, thread T& aggregate)
    {
    ushort logical_lane = lane % width;
    T inclusive = v;
    for (ushort offset = 1; offset < width; offset <<= 1)
        {
        T other = warp_shuffle_up(inclusive, offset);
        if (logical_lane >= offset)
            inclusive += other;
        }
    aggregate = warp_shuffle(inclusive, ushort(lane - logical_lane + width - 1));
    return inclusive - v;
    }

//! Sum over the caller's logical warp. The result is valid in the first lane of the logical warp.
template<class T> inline T warp_reduce_sum(T v, ushort lane, uint width)
    {
    ushort logical_lane = lane % width;
    for (ushort offset = width / 2; offset > 0; offset >>= 1)
        {
        T other = warp_shuffle_down(v, offset);
        if (logical_lane + offset < width)
            v += other;
        }
    return v;
    }

    } // end namespace detail
    } // end namespace hoomd
