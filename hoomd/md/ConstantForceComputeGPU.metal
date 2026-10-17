// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ConstantForceComputeGPU.metal
    \brief Metal port of ConstantForceComputeGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#ifndef __METAL_VERSION__
#include "ConstantForceComputeGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct gpu_compute_constant_force_set_forces_kernel_args
    {
    unsigned int group_size;
    HOOMD_DEVPTR(unsigned int) d_index_array;
    HOOMD_DEVPTR(Scalar4) d_force;
    HOOMD_DEVPTR(Scalar4) d_torque;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar3) d_f_const;
    HOOMD_DEVPTR(const Scalar3) d_t_const;
    unsigned int N;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void gpu_compute_constant_force_set_forces_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_compute_constant_force_set_forces_kernel_args))
    {
    const unsigned int group_size = args.group_size;
    device unsigned int* d_index_array = args.d_index_array;
    device Scalar4* d_force = args.d_force;
    device Scalar4* d_torque = args.d_torque;
    device const Scalar4* d_pos = args.d_pos;
    device const Scalar3* d_f_const = args.d_f_const;
    device const Scalar3* d_t_const = args.d_t_const;
    const unsigned int N = args.N;

    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int idx = d_index_array[group_idx];
    Scalar4 posidx = __ldg(d_pos + idx);
    unsigned int type = __scalar_as_int(posidx.w);

    Scalar3 fconst = d_f_const[type];

    vec3<Scalar> fi(fconst.x, fconst.y, fconst.z);
    d_force[idx] = vec_to_scalar4(fi, 0);

    Scalar3 tconst = d_t_const[type];
    vec3<Scalar> ti(tconst.x, tconst.y, tconst.z);
    d_torque[idx] = vec_to_scalar4(ti, 0);
    }

#else

namespace kernel
    {
//! Kernel for setting constant force vectors on the GPU
/*! \param group_size number of particles
    \param d_index_array stores list to convert group index to global tag
    \param d_force particle force on device
    \param d_torque particle torque on device
    \param d_f_const particle constant force unit vector
    \param d_t_const particle constant torque unit vector
    \param orientationLink check if particle orientation is linked to constant force vector
*/


hipError_t gpu_compute_constant_force_set_forces(const unsigned int group_size,
                                                 unsigned int* d_index_array,
                                                 Scalar4* d_force,
                                                 Scalar4* d_torque,
                                                 const Scalar4* d_pos,
                                                 const Scalar3* d_f_const,
                                                 const Scalar3* d_t_const,
                                                 const unsigned int N,
                                                 unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    {
        kernel_args::gpu_compute_constant_force_set_forces_kernel_args _args;
        _args.group_size = group_size;
        _args.d_index_array = d_index_array;
        _args.d_force = d_force;
        _args.d_torque = d_torque;
        _args.d_pos = d_pos;
        _args.d_f_const = d_f_const;
        _args.d_t_const = d_t_const;
        _args.N = N;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_compute_constant_force_set_forces_kernel",
                           dim3(grid),
                           dim3(threads),
                           _args,
                           0);
        }
    return hipSuccess;
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
