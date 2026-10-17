// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Metal port of draw_virtual_particles in RejectionVirtualParticleFillerGPU.cuh for @_geometry@
// (generated from RejectionVirtualParticleFillerGPUDrawKernel.metal.inc)

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/mpcd/@_geometry@.h"
#include "hoomd/mpcd/ParticleDataUtilities.h"

#ifndef __METAL_VERSION__
#include "hoomd/mpcd/RejectionVirtualParticleFillerGPU.cuh"

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
struct draw_virtual_particles_args
    {
    HOOMD_DEVPTR(Scalar4) d_tmp_pos;
    HOOMD_DEVPTR(Scalar4) d_tmp_vel;
    HOOMD_DEVPTR(bool) d_keep_particles;
    Scalar3 lo;
    Scalar3 hi;
    uint64_t timestep;
    Scalar vel_factor;
    unsigned int first_tag;
    unsigned int type;
    unsigned int N_virt_max;
    unsigned int seed;
    unsigned int filler_id;
    // the geometry as bytes (Metal does not allow empty classes in argument buffers)
    alignas(8) unsigned char geom[sizeof(GEOMETRY_CLASS)];
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

//! Draw candidate virtual particles, see draw_virtual_particles in
//! RejectionVirtualParticleFillerGPU.cuh
kernel void draw_virtual_particles(HOOMD_KERNEL_PARAMS(kernel_args::draw_virtual_particles_args))
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.N_virt_max)
        return;
    const GEOMETRY_CLASS geom = *(constant GEOMETRY_CLASS*)args.geom;
    const Scalar3 lo = args.lo;
    const Scalar3 hi = args.hi;

    const unsigned int tag = args.first_tag + idx;
    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::VirtualParticleFiller, args.timestep, args.seed),
        hoomd::Counter(tag, args.filler_id));
    // evaluation order matches the CUDA kernel (left to right)
    const Scalar x = hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng);
    const Scalar y = hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng);
    const Scalar z = hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng);
    Scalar3 pos = make_scalar3(x, y, z);
    args.d_tmp_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(args.type));

    args.d_keep_particles[idx] = geom.isOutside(pos);

    hoomd::NormalDistribution<Scalar> gen(args.vel_factor, 0.0);
    Scalar3 vel;
        {
        Scalar r1, r2;
        gen(r1, r2, rng);
        vel.x = r1;
        vel.y = r2;
        }
    vel.z = gen(rng);
    geom.addToVirtualParticleVelocity(vel, pos);
    args.d_tmp_vel[idx]
        = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
    }

#else

template<>
cudaError_t __attribute__((visibility("default")))
draw_virtual_particles<GEOMETRY_CLASS>(const draw_virtual_particles_args_t& args,
                                       const GEOMETRY_CLASS& geom)
    {
    const char* name = "hoomd::mpcd::gpu::draw_virtual_particles";
    unsigned int run_block_size
        = std::min(args.block_size, mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name));

    kernel_args::draw_virtual_particles_args kargs;
    kargs.d_tmp_pos = args.d_tmp_pos;
    kargs.d_tmp_vel = args.d_tmp_vel;
    kargs.d_keep_particles = args.d_keep_particles;
    kargs.lo = args.lo;
    kargs.hi = args.hi;
    kargs.timestep = args.timestep;
    kargs.vel_factor = args.vel_factor;
    kargs.first_tag = args.first_tag;
    kargs.type = args.type;
    kargs.N_virt_max = args.N_virt_max;
    kargs.seed = args.seed;
    kargs.filler_id = args.filler_id;
    std::memcpy(kargs.geom, &geom, sizeof(GEOMETRY_CLASS));
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(args.N_virt_max / run_block_size + 1),
                       dim3(run_block_size),
                       kargs);
    }

#endif

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
