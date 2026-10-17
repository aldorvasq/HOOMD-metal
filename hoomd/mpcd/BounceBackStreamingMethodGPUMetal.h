// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BounceBackStreamingMethodGPUMetal.h
    \brief Metal port of the confined streaming kernel in BounceBackStreamingMethodGPU.cuh

    Each geometry and force has a generated .metal file (from BulkStreamingMethodGPU.metal.inc or
    BounceBackStreamingMethodGPU.metal.inc) that defines GEOMETRY_CLASS and FORCE_CLASS and
    includes this header. The host driver is an explicit specialization of confined_stream.
*/

#pragma once

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#include "ParticleDataUtilities.h"

#ifndef __METAL_VERSION__
#include "BounceBackStreamingMethodGPU.cuh"

#include <cstring>
#include <stdexcept>
#endif

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel_args
    {
struct confined_stream_args
    {
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_vel;
    BoxDim box;
    Scalar mass;
    Scalar dt;
    unsigned int N;
    // the geometry and force as bytes (Metal does not allow empty classes in argument buffers)
    alignas(8) unsigned char geom[sizeof(GEOMETRY_CLASS)];
    alignas(8) unsigned char force[sizeof(FORCE_CLASS)];
    };

//! Sizes of the types shared with the host, written by the layout check kernel
struct confined_stream_layout_args
    {
    HOOMD_DEVPTR(unsigned int) d_out;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

//! Stream particles with bounce back from the geometry, see confined_stream in
//! BounceBackStreamingMethodGPU.cuh
kernel void confined_stream(HOOMD_KERNEL_PARAMS(kernel_args::confined_stream_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.N)
        return;

    const GEOMETRY_CLASS geom = *(constant GEOMETRY_CLASS*)args.geom;
    const FORCE_CLASS force = *(constant FORCE_CLASS*)args.force;
    const BoxDim box = args.box;
    const Scalar dt = args.dt;
    const Scalar mass = args.mass;

    const Scalar4 postype = args.d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const unsigned int type = __scalar_as_int(postype.w);

    const Scalar4 vel_cell = args.d_vel[idx];
    Scalar3 vel = make_scalar3(vel_cell.x, vel_cell.y, vel_cell.z);
    vel += Scalar(0.5) * dt * force.evaluate(pos) / mass;

    Scalar dt_remain = dt;
    bool collide = true;
    do
        {
        pos += dt_remain * vel;
        collide = geom.detectCollision(pos, vel, dt_remain);
        } while (dt_remain > 0 && collide);
    vel += Scalar(0.5) * dt * force.evaluate(pos) / mass;

    int3 image = make_int3(0, 0, 0);
    box.wrap(pos, vel, image);

    args.d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));
    args.d_vel[idx]
        = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
    }

kernel void confined_stream_layout(HOOMD_KERNEL_PARAMS(kernel_args::confined_stream_layout_args))
    {
    args.d_out[0] = sizeof(kernel_args::confined_stream_args);
    }

#else

template<>
cudaError_t __attribute__((visibility("default")))
confined_stream<GEOMETRY_CLASS, FORCE_CLASS>(const stream_args_t& args,
                                             const GEOMETRY_CLASS& geom,
                                             const FORCE_CLASS& force)
    {
    // verify once that the host and Metal layouts of the arguments agree
    static bool checked = false;
    if (!checked)
        {
        unsigned int* d_out = nullptr;
        hipMalloc(&d_out, sizeof(unsigned int));
        kernel_args::confined_stream_layout_args largs;
        largs.d_out = d_out;
        mtl::launch(HOOMD_MTL_SOURCE,
                    "hoomd::mpcd::gpu::confined_stream_layout",
                    dim3(1),
                    dim3(1),
                    largs);
        unsigned int size = 0;
        hipMemcpy(&size, d_out, sizeof(unsigned int), hipMemcpyDeviceToHost);
        hipFree(d_out);
        if (size != sizeof(kernel_args::confined_stream_args))
            throw std::runtime_error("Metal and host layouts of the MPCD geometry or force differ");
        checked = true;
        }

    const char* name = "hoomd::mpcd::gpu::confined_stream";
    unsigned int run_block_size
        = std::min(args.block_size, mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name));

    kernel_args::confined_stream_args kargs;
    kargs.d_pos = args.d_pos;
    kargs.d_vel = args.d_vel;
    kargs.box = args.box;
    kargs.mass = args.mass;
    kargs.dt = args.dt;
    kargs.N = args.N;
    std::memcpy(kargs.geom, &geom, sizeof(GEOMETRY_CLASS));
    std::memcpy(kargs.force, &force, sizeof(FORCE_CLASS));
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(args.N / run_block_size + 1),
                       dim3(run_block_size),
                       kargs);
    }

#endif

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
