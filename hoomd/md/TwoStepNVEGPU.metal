// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file TwoStepNVEGPU.metal
    \brief Metal port of TwoStepNVEGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#ifndef __METAL_VERSION__
#include "TwoStepNVEGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct nve_step_one_args
    {
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(const Scalar3) d_accel;
    HOOMD_DEVPTR(int3) d_image;
    HOOMD_DEVPTR(const unsigned int) d_group_members;
    BoxDim box;
    Scalar deltaT;
    Scalar limit_val;
    unsigned int nwork;
    unsigned int limit;
    unsigned int zero_force;
    unsigned int n_dimensions;
    };

struct nve_angular_args
    {
    HOOMD_DEVPTR(Scalar4) d_orientation;
    HOOMD_DEVPTR(Scalar4) d_angmom;
    HOOMD_DEVPTR(const Scalar3) d_inertia;
    HOOMD_DEVPTR(const Scalar4) d_net_torque;
    HOOMD_DEVPTR(const unsigned int) d_group_members;
    unsigned int nwork;
    Scalar deltaT;
    unsigned int n_dimensions;
    Scalar scale;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

//! First half step of velocity Verlet: update positions and half step velocities
kernel void gpu_nve_step_one_kernel(HOOMD_KERNEL_PARAMS(kernel_args::nve_step_one_args))
    {
    unsigned int work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= args.nwork)
        return;

    unsigned int idx = args.d_group_members[work_idx];
    const Scalar deltaT = args.deltaT;

    // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    Scalar4 postype = args.d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    Scalar4 velmass = args.d_vel[idx];
    Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);

    Scalar3 accel = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    if (!args.zero_force)
        accel = args.d_accel[idx];
    if (args.n_dimensions == 2)
        accel.z = Scalar(0.0);

    Scalar3 dx = vel * deltaT + (Scalar(1.0) / Scalar(2.0)) * accel * deltaT * deltaT;

    // limit the movement of the particles
    if (args.limit)
        {
        Scalar len = sqrtf(dot(dx, dx));
        if (len > args.limit_val)
            dx = dx / len * args.limit_val;
        }

    pos += dx;
    vel += (Scalar(1.0) / Scalar(2.0)) * accel * deltaT;

    // fix the periodic boundary conditions
    int3 image = args.d_image[idx];
    BoxDim box = args.box;
    box.wrap(pos, vel, image);

    args.d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
    args.d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, velmass.w);
    args.d_image[idx] = image;
    }

//! First half step of the angular momentum integration (Miller et al. symplectic scheme)
kernel void gpu_nve_angular_step_one_kernel(HOOMD_KERNEL_PARAMS(kernel_args::nve_angular_args))
    {
    unsigned int work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= args.nwork)
        return;

    unsigned int idx = args.d_group_members[work_idx];
    const Scalar deltaT = args.deltaT;

    // read the particle's orientation, conjugate quaternion, moment of inertia and net torque
    quat<Scalar> q(args.d_orientation[idx]);
    quat<Scalar> p(args.d_angmom[idx]);
    vec3<Scalar> t(args.d_net_torque[idx]);
    vec3<Scalar> I(args.d_inertia[idx]);

    // rotate torque into principal frame
    t = rotate(conj(q), t);

    // check for zero moment of inertia
    bool x_zero, y_zero, z_zero;
    x_zero = (I.x == 0 || args.n_dimensions == 2);
    y_zero = (I.y == 0 || args.n_dimensions == 2);
    z_zero = (I.z == 0);

    // ignore torque component along an axis for which the moment of inertia zero
    if (x_zero)
        t.x = Scalar(0.0);
    if (y_zero)
        t.y = Scalar(0.0);
    if (z_zero)
        t.z = Scalar(0.0);

    // advance p(t)->p(t+deltaT/2), q(t)->q(t+deltaT)
    // using Trotter factorization of rotation Liouvillian
    p += deltaT * q * t;

    // rescale
    p = p * args.scale;

    quat<Scalar> p1, p2, p3; // permutated quaternions
    quat<Scalar> q1, q2, q3;
    Scalar phi1, cphi1, sphi1;
    Scalar phi2, cphi2, sphi2;
    Scalar phi3, cphi3, sphi3;

    if (!z_zero)
        {
        p3 = quat<Scalar>(-p.v.z, vec3<Scalar>(p.v.y, -p.v.x, p.s));
        q3 = quat<Scalar>(-q.v.z, vec3<Scalar>(q.v.y, -q.v.x, q.s));
        phi3 = Scalar(1. / 4.) / I.z * dot(p, q3);
        cphi3 = slow::cos(Scalar(1. / 2.) * deltaT * phi3);
        sphi3 = slow::sin(Scalar(1. / 2.) * deltaT * phi3);

        p = cphi3 * p + sphi3 * p3;
        q = cphi3 * q + sphi3 * q3;
        }

    if (!y_zero)
        {
        p2 = quat<Scalar>(-p.v.y, vec3<Scalar>(-p.v.z, p.s, p.v.x));
        q2 = quat<Scalar>(-q.v.y, vec3<Scalar>(-q.v.z, q.s, q.v.x));
        phi2 = Scalar(1. / 4.) / I.y * dot(p, q2);
        cphi2 = slow::cos(Scalar(1. / 2.) * deltaT * phi2);
        sphi2 = slow::sin(Scalar(1. / 2.) * deltaT * phi2);

        p = cphi2 * p + sphi2 * p2;
        q = cphi2 * q + sphi2 * q2;
        }

    if (!x_zero)
        {
        p1 = quat<Scalar>(-p.v.x, vec3<Scalar>(p.s, p.v.z, -p.v.y));
        q1 = quat<Scalar>(-q.v.x, vec3<Scalar>(q.s, q.v.z, -q.v.y));
        phi1 = Scalar(1. / 4.) / I.x * dot(p, q1);
        cphi1 = slow::cos(deltaT * phi1);
        sphi1 = slow::sin(deltaT * phi1);

        p = cphi1 * p + sphi1 * p1;
        q = cphi1 * q + sphi1 * q1;
        }

    if (!y_zero)
        {
        p2 = quat<Scalar>(-p.v.y, vec3<Scalar>(-p.v.z, p.s, p.v.x));
        q2 = quat<Scalar>(-q.v.y, vec3<Scalar>(-q.v.z, q.s, q.v.x));
        phi2 = Scalar(1. / 4.) / I.y * dot(p, q2);
        cphi2 = slow::cos(Scalar(1. / 2.) * deltaT * phi2);
        sphi2 = slow::sin(Scalar(1. / 2.) * deltaT * phi2);

        p = cphi2 * p + sphi2 * p2;
        q = cphi2 * q + sphi2 * q2;
        }

    if (!z_zero)
        {
        p3 = quat<Scalar>(-p.v.z, vec3<Scalar>(p.v.y, -p.v.x, p.s));
        q3 = quat<Scalar>(-q.v.z, vec3<Scalar>(q.v.y, -q.v.x, q.s));
        phi3 = Scalar(1. / 4.) / I.z * dot(p, q3);
        cphi3 = slow::cos(Scalar(1. / 2.) * deltaT * phi3);
        sphi3 = slow::sin(Scalar(1. / 2.) * deltaT * phi3);

        p = cphi3 * p + sphi3 * p3;
        q = cphi3 * q + sphi3 * q3;
        }

    // renormalize (improves stability)
    q = q * (Scalar(1.0) / slow::sqrt(norm2(q)));

    args.d_orientation[idx] = quat_to_scalar4(q);
    args.d_angmom[idx] = quat_to_scalar4(p);
    }

//! Second half step of the angular momentum integration
kernel void gpu_nve_angular_step_two_kernel(HOOMD_KERNEL_PARAMS(kernel_args::nve_angular_args))
    {
    unsigned int work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= args.nwork)
        return;

    unsigned int idx = args.d_group_members[work_idx];

    quat<Scalar> q(args.d_orientation[idx]);
    quat<Scalar> p(args.d_angmom[idx]);
    vec3<Scalar> t(args.d_net_torque[idx]);
    vec3<Scalar> I(args.d_inertia[idx]);

    // rotate torque into principal frame
    t = rotate(conj(q), t);

    // check for zero moment of inertia
    bool x_zero, y_zero, z_zero;
    x_zero = (I.x == 0 || args.n_dimensions == 2);
    y_zero = (I.y == 0 || args.n_dimensions == 2);
    z_zero = (I.z == 0);

    // ignore torque component along an axis for which the moment of inertia zero
    if (x_zero)
        t.x = Scalar(0.0);
    if (y_zero)
        t.y = Scalar(0.0);
    if (z_zero)
        t.z = Scalar(0.0);

    // rescale
    p = p * args.scale;

    // advance p(t+deltaT/2)->p(t+deltaT)
    p += args.deltaT * q * t;

    args.d_angmom[idx] = quat_to_scalar4(p);
    }

#else

namespace kernel
    {
namespace
    {
dim3 grid_for(unsigned int nwork, unsigned int block_size)
    {
    return dim3((nwork / block_size) + 1);
    }

hipError_t launch_angular(const char* name,
                          const Scalar4* d_orientation,
                          Scalar4* d_angmom,
                          const Scalar3* d_inertia,
                          const Scalar4* d_net_torque,
                          const unsigned int* d_group_members,
                          unsigned int group_size,
                          Scalar deltaT,
                          unsigned int n_dimensions,
                          Scalar scale,
                          unsigned int block_size)
    {
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name);
    unsigned int run_block_size = std::min(block_size, max_block_size);

    kernel_args::nve_angular_args args;
    args.d_orientation = const_cast<Scalar4*>(d_orientation);
    args.d_angmom = d_angmom;
    args.d_inertia = d_inertia;
    args.d_net_torque = d_net_torque;
    args.d_group_members = d_group_members;
    args.nwork = group_size;
    args.deltaT = deltaT;
    args.n_dimensions = n_dimensions;
    args.scale = scale;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       grid_for(group_size, run_block_size),
                       dim3(run_block_size),
                       args);
    }
    } // end anonymous namespace

hipError_t gpu_nve_step_one(Scalar4* d_pos,
                            Scalar4* d_vel,
                            const Scalar3* d_accel,
                            int3* d_image,
                            unsigned int* d_group_members,
                            const unsigned int group_size,
                            const BoxDim& box,
                            Scalar deltaT,
                            bool limit,
                            Scalar limit_val,
                            bool zero_force,
                            unsigned int block_size,
                            unsigned int n_dimensions)
    {
    const char* name = "hoomd::md::gpu_nve_step_one_kernel";
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name);
    unsigned int run_block_size = std::min(block_size, max_block_size);

    kernel_args::nve_step_one_args args;
    args.d_pos = d_pos;
    args.d_vel = d_vel;
    args.d_accel = d_accel;
    args.d_image = d_image;
    args.d_group_members = d_group_members;
    args.box = box;
    args.deltaT = deltaT;
    args.limit_val = limit_val;
    args.nwork = group_size;
    args.limit = limit;
    args.zero_force = zero_force;
    args.n_dimensions = n_dimensions;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       grid_for(group_size, run_block_size),
                       dim3(run_block_size),
                       args);
    }

hipError_t gpu_nve_angular_step_one(Scalar4* d_orientation,
                                    Scalar4* d_angmom,
                                    const Scalar3* d_inertia,
                                    const Scalar4* d_net_torque,
                                    unsigned int* d_group_members,
                                    const unsigned int group_size,
                                    Scalar deltaT,
                                    unsigned int n_dimensions,
                                    Scalar scale,
                                    const unsigned int block_size)
    {
    return launch_angular("hoomd::md::gpu_nve_angular_step_one_kernel",
                          d_orientation,
                          d_angmom,
                          d_inertia,
                          d_net_torque,
                          d_group_members,
                          group_size,
                          deltaT,
                          n_dimensions,
                          scale,
                          block_size);
    }

hipError_t gpu_nve_angular_step_two(const Scalar4* d_orientation,
                                    Scalar4* d_angmom,
                                    const Scalar3* d_inertia,
                                    const Scalar4* d_net_torque,
                                    unsigned int* d_group_members,
                                    const unsigned int group_size,
                                    Scalar deltaT,
                                    unsigned int n_dimensions,
                                    Scalar scale,
                                    const unsigned int block_size)
    {
    return launch_angular("hoomd::md::gpu_nve_angular_step_two_kernel",
                          d_orientation,
                          d_angmom,
                          d_inertia,
                          d_net_torque,
                          d_group_members,
                          group_size,
                          deltaT,
                          n_dimensions,
                          scale,
                          block_size);
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
