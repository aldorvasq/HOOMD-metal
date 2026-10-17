// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MuellerPlatheFlowGPU.metal
    \brief Metal port of MuellerPlatheFlowGPU.cu

    MuellerPlatheFlowGPU.cu searches for the extreme momenta with thrust::transform_reduce and a
    binary operator that keeps only particles in the requested slab. Here, one threadgroup filters
    the particles to the slab and reduces them in threadgroup memory with the same comparison.
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#include "MuellerPlatheFlowEnum.h"

#ifndef __METAL_VERSION__
#include "MuellerPlatheFlowGPU.cuh"

#include "hoomd/mtl/Primitives.h"

#include <algorithm>
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct gpu_search_min_max_velocity_args
    {
    HOOMD_DEVPTR(const Scalar4) d_vel;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(const unsigned int) d_rtag;
    HOOMD_DEVPTR(const unsigned int) d_group_members;
    HOOMD_DEVPTR(Scalar3) d_result;
    BoxDim gl_box;
    Scalar3 init;
    unsigned int group_size;
    unsigned int Nslabs;
    unsigned int slab_index;
    int flow_direction;
    int slab_direction;
    int find_max;
    };

struct gpu_update_min_max_velocity_args
    {
    HOOMD_DEVPTR(const unsigned int) d_rtag;
    HOOMD_DEVPTR(Scalar4) d_vel;
    unsigned int Ntotal;
    Scalar3 last_max_vel;
    Scalar3 last_min_vel;
    int flow_direction;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

//! Component of v along a flow_enum::Direction
inline Scalar mp_component(Scalar4 v, int direction)
    {
    if (direction == flow_enum::X)
        return v.x;
    if (direction == flow_enum::Y)
        return v.y;
    return v.z;
    }

//! Slab index of the particle with the given tag (see vel_search_binary_opt)
inline unsigned int mp_slab(constant kernel_args::gpu_search_min_max_velocity_args& args,
                            unsigned int tag)
    {
    unsigned int idx = args.d_rtag[tag];
    Scalar4 pos = args.d_pos[idx];
    BoxDim box = args.gl_box;
    Scalar3 L = box.getL();
    Scalar x = mp_component(pos, args.slab_direction);
    Scalar Ld = args.slab_direction == flow_enum::X   ? L.x
                : args.slab_direction == flow_enum::Y ? L.y
                                                      : L.z;
    unsigned int index = (unsigned int)((x / Ld + Scalar(0.5)) * Scalar(args.Nslabs));
    return index % args.Nslabs;
    }

//! Combine two candidates that are both in the slab (or invalid)
/*! The invalid tag is UINT_MAX, a NaN when viewed as a float, so tags compare as bits.
 */
inline Scalar3 mp_combine(Scalar3 a, Scalar3 b, Scalar invalid_tag, bool find_max)
    {
    if (__scalar_as_int(a.z) == __scalar_as_int(invalid_tag))
        return b;
    if (__scalar_as_int(b.z) == __scalar_as_int(invalid_tag))
        return a;
    bool a_wins = find_max ? (a.x > b.x) : (a.x < b.x);
    return a_wins ? a : b;
    }

kernel void gpu_search_min_max_velocity_kernel(
    HOOMD_KERNEL_PARAMS(kernel_args::gpu_search_min_max_velocity_args))
    {
    threadgroup Scalar3* s_best = (threadgroup Scalar3*)hoomd_dynamic_shared;
    const Scalar invalid_tag = args.init.z;
    const bool find_max = args.find_max != 0;

    // the initial value is the invalid sentinel (see vel_search_binary_opt)
    Scalar3 best = args.init;

    for (unsigned int i = threadIdx.x; i < args.group_size; i += blockDim.x)
        {
        unsigned int idx = args.d_group_members[i];
        unsigned int tag = args.d_tag[idx];
        if (mp_slab(args, tag) != args.slab_index)
            continue;

        Scalar4 vel = args.d_vel[idx];
        Scalar mass = vel.w;
        Scalar3 candidate;
        candidate.x = mp_component(vel, args.flow_direction) * mass;
        candidate.y = mass;
        candidate.z = __int_as_scalar(tag);
        best = mp_combine(best, candidate, invalid_tag, find_max);
        }

    s_best[threadIdx.x] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1)
        {
        if (threadIdx.x < offset)
            s_best[threadIdx.x]
                = mp_combine(s_best[threadIdx.x], s_best[threadIdx.x + offset], invalid_tag, find_max);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    if (threadIdx.x == 0)
        *args.d_result = s_best[0];
    }

kernel void gpu_update_min_max_velocity_kernel(
    HOOMD_KERNEL_PARAMS(kernel_args::gpu_update_min_max_velocity_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1)
        return;
    const Scalar3 last_max_vel = args.last_max_vel;
    const Scalar3 last_min_vel = args.last_min_vel;
    const unsigned int min_tag = __scalar_as_int(last_min_vel.z);
    const unsigned int min_idx = args.d_rtag[min_tag];
    const unsigned int max_tag = __scalar_as_int(last_max_vel.z);
    const unsigned int max_idx = args.d_rtag[max_tag];

    // Is the particle local on the processor?
    // Swap the particles the new velocities.
    if (min_idx < args.Ntotal)
        {
        const Scalar new_min_vel = last_max_vel.x / last_min_vel.y;
        if (args.flow_direction == flow_enum::X)
            args.d_vel[min_idx].x = new_min_vel;
        else if (args.flow_direction == flow_enum::Y)
            args.d_vel[min_idx].y = new_min_vel;
        else
            args.d_vel[min_idx].z = new_min_vel;
        }

    if (max_idx < args.Ntotal)
        {
        const Scalar new_max_vel = last_min_vel.x / last_max_vel.y;
        if (args.flow_direction == flow_enum::X)
            args.d_vel[max_idx].x = new_max_vel;
        else if (args.flow_direction == flow_enum::Y)
            args.d_vel[max_idx].y = new_max_vel;
        else
            args.d_vel[max_idx].z = new_max_vel;
        }
    }

#else

namespace kernel
    {
hipError_t gpu_search_min_max_velocity(const unsigned int group_size,
                                       const Scalar4* const d_vel,
                                       const Scalar4* const d_pos,
                                       const unsigned int* const d_tag,
                                       const unsigned int* const d_rtag,
                                       const unsigned int* const d_group_members,
                                       const BoxDim gl_box,
                                       const unsigned int Nslabs,
                                       const unsigned int max_slab,
                                       const unsigned int min_slab,
                                       Scalar3* const last_max_vel,
                                       Scalar3* const last_min_vel,
                                       const bool has_max_slab,
                                       const bool has_min_slab,
                                       const unsigned int blocksize,
                                       const flow_enum::Direction flow_direction,
                                       const flow_enum::Direction slab_direction)
    {
    const char* name = "hoomd::md::gpu_search_min_max_velocity_kernel";
    unsigned int block_size = 1;
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name);
    // the threadgroup reduction requires a power of two
    while (block_size * 2 <= std::min(max_block_size, 1024u))
        block_size *= 2;

    mtl::ScratchScope scratch;
    Scalar3* d_result = scratch.allocate<Scalar3>(2);

    kernel_args::gpu_search_min_max_velocity_args args;
    args.d_vel = d_vel;
    args.d_pos = d_pos;
    args.d_tag = d_tag;
    args.d_rtag = d_rtag;
    args.d_group_members = d_group_members;
    args.gl_box = gl_box;
    args.group_size = group_size;
    args.Nslabs = Nslabs;
    args.flow_direction = flow_direction;
    args.slab_direction = slab_direction;

    if (has_max_slab)
        {
        args.d_result = d_result;
        args.init = *last_max_vel;
        args.slab_index = max_slab;
        args.find_max = 1;
        mtl::launch(HOOMD_MTL_SOURCE,
                    name,
                    dim3(1),
                    dim3(block_size),
                    args,
                    block_size * sizeof(Scalar3));
        }

    if (has_min_slab)
        {
        args.d_result = d_result + 1;
        args.init = *last_min_vel;
        args.slab_index = min_slab;
        args.find_max = 0;
        mtl::launch(HOOMD_MTL_SOURCE,
                    name,
                    dim3(1),
                    dim3(block_size),
                    args,
                    block_size * sizeof(Scalar3));
        }

    Scalar3 result[2];
    hipError_t error = hipMemcpy(result, d_result, sizeof(result), hipMemcpyDeviceToHost);
    if (has_max_slab)
        *last_max_vel = result[0];
    if (has_min_slab)
        *last_min_vel = result[1];
    return error;
    }

hipError_t gpu_update_min_max_velocity(const unsigned int* const d_rtag,
                                       Scalar4* const d_vel,
                                       const unsigned int Ntotal,
                                       const Scalar3 last_max_vel,
                                       const Scalar3 last_min_vel,
                                       const flow_enum::Direction flow_direction)
    {
    kernel_args::gpu_update_min_max_velocity_args args;
    args.d_rtag = d_rtag;
    args.d_vel = d_vel;
    args.Ntotal = Ntotal;
    args.last_max_vel = last_max_vel;
    args.last_min_vel = last_min_vel;
    args.flow_direction = flow_direction;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       "hoomd::md::gpu_update_min_max_velocity_kernel",
                       dim3(1),
                       dim3(1),
                       args);
    }

    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
