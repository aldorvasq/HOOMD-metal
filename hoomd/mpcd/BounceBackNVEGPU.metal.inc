// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Metal port of nve_bounce_step_one in BounceBackNVEGPU.cuh for @_geometry@ (generated from
// BounceBackNVEGPU.metal.inc)

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/mpcd/@_geometry@.h"

#ifndef __METAL_VERSION__
#include "hoomd/mpcd/BounceBackNVEGPU.cuh"

#include <cstring>
#endif

#define GEOMETRY_CLASS @_geometry@

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel_args
    {
struct nve_bounce_step_one_args
    {
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(int3) d_image;
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(const Scalar3) d_accel;
    HOOMD_DEVPTR(const unsigned int) d_group;
    BoxDim box;
    Scalar dt;
    unsigned int N;
    // the geometry as bytes (Metal does not allow empty classes in argument buffers)
    alignas(8) unsigned char geom[sizeof(GEOMETRY_CLASS)];
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

//! First step of velocity Verlet with bounce back from the geometry
kernel void nve_bounce_step_one(HOOMD_KERNEL_PARAMS(kernel_args::nve_bounce_step_one_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.N)
        return;
    const unsigned int pid = args.d_group[idx];
    const GEOMETRY_CLASS geom = *(constant GEOMETRY_CLASS*)args.geom;
    const BoxDim box = args.box;
    const Scalar dt = args.dt;

    const Scalar4 velmass = args.d_vel[pid];
    Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
    const Scalar mass = velmass.w;

    const Scalar3 accel = args.d_accel[pid];
    vel += Scalar(0.5) * dt * accel;

    const Scalar4 postype = args.d_pos[pid];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const Scalar type = postype.w;

    Scalar dt_remain = dt;
    bool collide = false;
    do
        {
        pos += dt_remain * vel;
        collide = geom.detectCollision(pos, vel, dt_remain);
        } while (dt_remain > 0 && collide);

    int3 img = args.d_image[pid];
    box.wrap(pos, vel, img);

    args.d_pos[pid] = make_scalar4(pos.x, pos.y, pos.z, type);
    args.d_vel[pid] = make_scalar4(vel.x, vel.y, vel.z, mass);
    args.d_image[pid] = img;
    }

#else

template<>
cudaError_t __attribute__((visibility("default")))
nve_bounce_step_one<GEOMETRY_CLASS>(const bounce_args_t& args, const GEOMETRY_CLASS& geom)
    {
    const char* name = "hoomd::mpcd::gpu::nve_bounce_step_one";
    unsigned int run_block_size
        = std::min(args.block_size, mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name));

    kernel_args::nve_bounce_step_one_args kargs;
    kargs.d_pos = args.d_pos;
    kargs.d_image = args.d_image;
    kargs.d_vel = args.d_vel;
    kargs.d_accel = args.d_accel;
    kargs.d_group = args.d_group;
    kargs.box = args.box;
    kargs.dt = args.dt;
    kargs.N = args.N;
    std::memcpy(kargs.geom, &geom, sizeof(GEOMETRY_CLASS));
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
