// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file Integrator.metal
    \brief Metal port of Integrator.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/HOOMDMath.h"

#ifndef __METAL_VERSION__
#include "Integrator.cuh"
#endif

namespace hoomd
    {
namespace kernel_args
    {
struct sum_net_force_args
    {
    HOOMD_DEVPTR(Scalar4) d_net_force;
    HOOMD_DEVPTR(Scalar) d_net_virial;
    HOOMD_DEVPTR(Scalar4) d_net_torque;
    HOOMD_DEVPTR(Scalar4) d_f[6];
    HOOMD_DEVPTR(Scalar4) d_t[6];
    HOOMD_DEVPTR(Scalar) d_v[6];
    unsigned int vpitch[6];
    unsigned int net_virial_pitch;
    unsigned int nwork;
    unsigned int clear;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant bool compute_virial [[function_constant(0)]];

//! Sum the forces, virials, and torques of up to 6 force computes into the net arrays
kernel void gpu_integrator_sum_net_force_kernel(HOOMD_KERNEL_PARAMS(kernel_args::sum_net_force_args))
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= args.nwork)
        return;

    Scalar4 net_force;
    Scalar net_virial[6];
    Scalar4 net_torque;
    if (args.clear)
        {
        net_force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
        for (int i = 0; i < 6; i++)
            net_virial[i] = Scalar(0.0);
        net_torque = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
        }
    else
        {
        net_force = args.d_net_force[idx];
        if (compute_virial)
            {
            for (int i = 0; i < 6; i++)
                net_virial[i] = args.d_net_virial[i * args.net_virial_pitch + idx];
            }
        net_torque = args.d_net_torque[idx];
        }

    for (int k = 0; k < 6; k++)
        {
        device Scalar4* d_f = args.d_f[k];
        device Scalar* d_v = args.d_v[k];
        device Scalar4* d_t = args.d_t[k];
        if (d_f != nullptr && d_v != nullptr && d_t != nullptr)
            {
            net_force += d_f[idx];
            if (compute_virial)
                {
                for (int i = 0; i < 6; i++)
                    net_virial[i] += d_v[i * args.vpitch[k] + idx];
                }
            net_torque += d_t[idx];
            }
        }

    args.d_net_force[idx] = net_force;
    if (compute_virial)
        {
        for (int i = 0; i < 6; i++)
            args.d_net_virial[i * args.net_virial_pitch + idx] = net_virial[i];
        }
    args.d_net_torque[idx] = net_torque;
    }

#else

namespace kernel
    {
hipError_t gpu_integrator_sum_net_force(Scalar4* d_net_force,
                                        Scalar* d_net_virial,
                                        size_t net_virial_pitch,
                                        Scalar4* d_net_torque,
                                        const gpu_force_list& force_list,
                                        unsigned int nparticles,
                                        bool clear,
                                        bool compute_virial)
    {
    assert(d_net_force);
    assert(d_net_virial);
    assert(d_net_torque);

    kernel_args::sum_net_force_args args;
    args.d_net_force = d_net_force;
    args.d_net_virial = d_net_virial;
    args.d_net_torque = d_net_torque;
    Scalar4* f[6]
        = {force_list.f0, force_list.f1, force_list.f2, force_list.f3, force_list.f4, force_list.f5};
    Scalar4* t[6]
        = {force_list.t0, force_list.t1, force_list.t2, force_list.t3, force_list.t4, force_list.t5};
    Scalar* v[6]
        = {force_list.v0, force_list.v1, force_list.v2, force_list.v3, force_list.v4, force_list.v5};
    size_t vpitch[6] = {force_list.vpitch0,
                        force_list.vpitch1,
                        force_list.vpitch2,
                        force_list.vpitch3,
                        force_list.vpitch4,
                        force_list.vpitch5};
    for (int k = 0; k < 6; k++)
        {
        args.d_f[k] = f[k];
        args.d_t[k] = t[k];
        args.d_v[k] = v[k];
        args.vpitch[k] = static_cast<unsigned int>(vpitch[k]);
        }
    args.net_virial_pitch = static_cast<unsigned int>(net_virial_pitch);
    args.nwork = nparticles;
    args.clear = clear;

    const unsigned int block_size = 256;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       "hoomd::gpu_integrator_sum_net_force_kernel",
                       dim3(nparticles / block_size + 1),
                       dim3(block_size),
                       args,
                       0,
                       {mtl::FunctionConstant::make_bool(0, compute_virial)});
    }
    } // end namespace kernel

#endif

    } // end namespace hoomd
