// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file PotentialExternalGPUMetal.h
    \brief Metal port of the external force kernel in PotentialExternalGPU.cuh

    Each external potential has a generated .metal file (from PotentialExternalGPUKernel.metal.inc
    or PotentialExternalWallGPUKernel.metal.inc) that includes this header with EVALUATOR_CLASS
    defined.
*/

#pragma once

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#ifndef __METAL_VERSION__
#include "PotentialExternalGPU.cuh"

#include <stdexcept>
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
template<class evaluator> struct external_forces_args
    {
    HOOMD_DEVPTR(Scalar4) d_force;
    HOOMD_DEVPTR(Scalar4) d_torque;
    HOOMD_DEVPTR(Scalar) d_virial;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar4) d_orientation;
    HOOMD_DEVPTR(const Scalar) d_charge;
    HOOMD_DEVPTR(const typename evaluator::param_type) params;
    HOOMD_DEVPTR(const int) d_field; //!< field_type data as 32-bit words
    BoxDim box;
    unsigned int virial_pitch;
    unsigned int N;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

/*! Compute the external forces on all N particles. See gpu_compute_external_forces_kernel in
    PotentialExternalGPU.cuh.
*/
template<class evaluator>
inline void
gpu_compute_external_forces_impl(constant kernel_args::external_forces_args<evaluator>& args,
                                 metal::vec<uint, 3> threadIdx,
                                 metal::vec<uint, 3> blockIdx,
                                 metal::vec<uint, 3> blockDim,
                                 threadgroup char* s_data)
    {
    typedef typename evaluator::field_type field_type;

    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // copy the field into threadgroup memory
    threadgroup field_type* s_field = (threadgroup field_type*)s_data;
        {
        unsigned int tidx = threadIdx.x;
        unsigned int block_size = blockDim.x;
        unsigned int field_size = sizeof(field_type) / sizeof(int);
        device const int* d_field_int = args.d_field;
        threadgroup int* s_field_int = (threadgroup int*)s_field;
        for (unsigned int cur_offset = 0; cur_offset < field_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < field_size)
                s_field_int[cur_offset + tidx] = d_field_int[cur_offset + tidx];
            }
        }
    threadgroup const field_type& field = *s_field;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (idx >= args.N)
        return;

    BoxDim box = args.box;

    // read in the position of our particle.
    Scalar4 posi = args.d_pos[idx];
    Scalar qi;
    if (evaluator::needsCharge())
        qi = args.d_charge[idx];
    else
        qi = Scalar(0.0);

    // initialize the force to 0
    Scalar3 force = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar3 torque = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virial[6];
    for (unsigned int k = 0; k < 6; k++)
        virial[k] = Scalar(0.0);
    Scalar energy = Scalar(0.0);

    unsigned int typei = __scalar_as_int(posi.w);
    Scalar3 Xi = make_scalar3(posi.x, posi.y, posi.z);
    quat<Scalar> q(args.d_orientation[idx]);
    evaluator eval(Xi, q, box, args.params[typei], field);

    if (evaluator::needsCharge())
        eval.setCharge(qi);

    eval.evalForceTorqueEnergyAndVirial(force, torque, energy, virial);

    // now that the force calculation is complete, write out the result
    args.d_force[idx] = make_scalar4(force.x, force.y, force.z, energy);
    for (unsigned int k = 0; k < 6; k++)
        args.d_virial[k * args.virial_pitch + idx] = virial[k];

    Scalar4 t = args.d_torque[idx];
    args.d_torque[idx] = make_scalar4(torque.x, torque.y, torque.z, t.w);
    }

//! Define the kernel entry point for an evaluator
#define HOOMD_EXTERNAL_FORCE_KERNEL(...)                                                       \
    kernel void gpu_compute_external_forces_kernel(                                            \
        HOOMD_KERNEL_PARAMS(kernel_args::external_forces_args<__VA_ARGS__>))                   \
        {                                                                                      \
        gpu_compute_external_forces_impl<__VA_ARGS__>(args,                                    \
                                                      threadIdx,                               \
                                                      blockIdx,                                \
                                                      blockDim,                                \
                                                      hoomd_dynamic_shared);                   \
        }

#else

namespace kernel
    {
template<class evaluator>
hipError_t gpu_compute_potential_external_forces(
    const kernel::external_potential_args_t& external_potential_args,
    const typename evaluator::param_type* d_params,
    const typename evaluator::field_type* d_field)
    {
    const char* name = "hoomd::md::gpu_compute_external_forces_kernel";
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name);
    unsigned int run_block_size = std::min(external_potential_args.block_size, max_block_size);

    unsigned int bytes = (sizeof(typename evaluator::field_type) / sizeof(int) + 1) * sizeof(int);
    if (bytes > external_potential_args.devprop.sharedMemPerBlock)
        {
        throw std::runtime_error("External potential parameters exceed the available shared memory "
                                 "per block.");
        }

    kernel_args::external_forces_args<evaluator> args;
    args.d_force = external_potential_args.d_force;
    args.d_torque = external_potential_args.d_torque;
    args.d_virial = external_potential_args.d_virial;
    args.d_pos = external_potential_args.d_pos;
    args.d_orientation = external_potential_args.d_orientation;
    args.d_charge = evaluator::needsCharge() ? external_potential_args.d_charge : nullptr;
    args.params = d_params;
    args.d_field = reinterpret_cast<const int*>(d_field);
    args.box = external_potential_args.box;
    args.virial_pitch = static_cast<unsigned int>(external_potential_args.virial_pitch);
    args.N = external_potential_args.N;

    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(external_potential_args.N / run_block_size + 1),
                       dim3(run_block_size),
                       args,
                       bytes);
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
