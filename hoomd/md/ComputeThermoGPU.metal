// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ComputeThermoGPU.metal
    \brief Metal port of ComputeThermoGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "ComputeThermoTypes.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/VectorMath.h"

#ifndef __METAL_VERSION__
#include "ComputeThermoGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct thermo_partial_args
    {
    HOOMD_DEVPTR(Scalar4) d_scratch;
    HOOMD_DEVPTR(Scalar) d_scratch_pressure_tensor;
    HOOMD_DEVPTR(Scalar) d_scratch_rot;
    HOOMD_DEVPTR(const Scalar4) d_net_force;
    HOOMD_DEVPTR(const Scalar) d_net_virial;
    HOOMD_DEVPTR(const Scalar4) d_velocity;
    HOOMD_DEVPTR(const Scalar4) d_orientation;
    HOOMD_DEVPTR(const Scalar4) d_angmom;
    HOOMD_DEVPTR(const Scalar3) d_inertia;
    HOOMD_DEVPTR(const unsigned int) d_body;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(const unsigned int) d_group_members;
    unsigned int virial_pitch;
    unsigned int work_size;
    unsigned int num_blocks;
    };

struct thermo_final_args
    {
    HOOMD_DEVPTR(Scalar) d_properties;
    HOOMD_DEVPTR(const Scalar4) d_scratch;
    HOOMD_DEVPTR(const Scalar) d_scratch_rot;
    HOOMD_DEVPTR(const Scalar) d_scratch_pressure_tensor;
    BoxDim box;
    Scalar ndof;
    unsigned int D;
    unsigned int group_size;
    unsigned int num_partial_sums;
    Scalar external_virial;
    Scalar external_energy;
    Scalar external_virial_xx;
    Scalar external_virial_xy;
    Scalar external_virial_xz;
    Scalar external_virial_yy;
    Scalar external_virial_yz;
    Scalar external_virial_zz;
    unsigned int twod;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

//! Is the particle counted in the sum? Rigid body constituents are not, their centers are.
inline bool thermo_counts(unsigned int body, unsigned int tag)
    {
    return body >= MIN_FLOPPY || body == tag;
    }

/*! Partial sums per block: 2*kinetic energy in .x, potential energy in .y, and the isotropic
    virial in .z. Requires sizeof(Scalar3)*block_size bytes of dynamic threadgroup memory.
*/
kernel void gpu_compute_thermo_partial_sums(HOOMD_KERNEL_PARAMS(kernel_args::thermo_partial_args))
    {
    threadgroup Scalar3* sdata = (threadgroup Scalar3*)hoomd_dynamic_shared;
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pitch = args.virial_pitch;

    Scalar3 my_element = make_scalar3(0, 0, 0);
    if (group_idx < args.work_size)
        {
        unsigned int idx = args.d_group_members[group_idx];
        if (thermo_counts(args.d_body[idx], args.d_tag[idx]))
            {
            Scalar4 net_force = args.d_net_force[idx];
            Scalar net_isotropic_virial = Scalar(1.0 / 3.0)
                                          * (args.d_net_virial[0 * pitch + idx]
                                             + args.d_net_virial[3 * pitch + idx]
                                             + args.d_net_virial[5 * pitch + idx]);
            Scalar4 vel = args.d_velocity[idx];
            Scalar mass = vel.w;

            my_element.x = mass * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
            my_element.y = net_force.w;
            my_element.z = net_isotropic_virial;
            }
        }

    sdata[threadIdx.x] = my_element;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    unsigned int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            {
            sdata[threadIdx.x].x += sdata[threadIdx.x + offs].x;
            sdata[threadIdx.x].y += sdata[threadIdx.x + offs].y;
            sdata[threadIdx.x].z += sdata[threadIdx.x + offs].z;
            }
        offs >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    if (threadIdx.x == 0)
        {
        Scalar3 res = sdata[0];
        args.d_scratch[blockIdx.x] = make_scalar4(res.x, res.y, res.z, 0);
        }
    }

//! Partial sums of the six pressure tensor components (6*sizeof(Scalar)*block_size bytes)
kernel void
gpu_compute_pressure_tensor_partial_sums(HOOMD_KERNEL_PARAMS(kernel_args::thermo_partial_args))
    {
    threadgroup Scalar* sdata = (threadgroup Scalar*)hoomd_dynamic_shared;
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pitch = args.virial_pitch;

    Scalar my_element[6] = {0, 0, 0, 0, 0, 0};
    if (group_idx < args.work_size)
        {
        unsigned int idx = args.d_group_members[group_idx];
        if (thermo_counts(args.d_body[idx], args.d_tag[idx]))
            {
            Scalar4 vel = args.d_velocity[idx];
            Scalar mass = vel.w;
            my_element[0] = mass * vel.x * vel.x + args.d_net_virial[0 * pitch + idx]; // xx
            my_element[1] = mass * vel.x * vel.y + args.d_net_virial[1 * pitch + idx]; // xy
            my_element[2] = mass * vel.x * vel.z + args.d_net_virial[2 * pitch + idx]; // xz
            my_element[3] = mass * vel.y * vel.y + args.d_net_virial[3 * pitch + idx]; // yy
            my_element[4] = mass * vel.y * vel.z + args.d_net_virial[4 * pitch + idx]; // yz
            my_element[5] = mass * vel.z * vel.z + args.d_net_virial[5 * pitch + idx]; // zz
            }
        }

    for (unsigned int i = 0; i < 6; i++)
        sdata[i * blockDim.x + threadIdx.x] = my_element[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    unsigned int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            {
            for (unsigned int i = 0; i < 6; i++)
                sdata[i * blockDim.x + threadIdx.x] += sdata[i * blockDim.x + threadIdx.x + offs];
            }
        offs >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    if (threadIdx.x == 0)
        {
        for (unsigned int i = 0; i < 6; i++)
            args.d_scratch_pressure_tensor[args.num_blocks * i + blockIdx.x]
                = sdata[i * blockDim.x];
        }
    }

//! Partial sums of the rotational kinetic energy (sizeof(Scalar)*block_size bytes)
kernel void
gpu_compute_rotational_ke_partial_sums(HOOMD_KERNEL_PARAMS(kernel_args::thermo_partial_args))
    {
    threadgroup Scalar* sdata = (threadgroup Scalar*)hoomd_dynamic_shared;
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar my_element = Scalar(0.0);
    if (group_idx < args.work_size)
        {
        unsigned int idx = args.d_group_members[group_idx];
        if (thermo_counts(args.d_body[idx], args.d_tag[idx]))
            {
            quat<Scalar> q(args.d_orientation[idx]);
            quat<Scalar> p(args.d_angmom[idx]);
            vec3<Scalar> I(args.d_inertia[idx]);
            quat<Scalar> s(Scalar(0.5) * conj(q) * p);

            Scalar ke_rot(0.0);
            if (I.x > 0)
                ke_rot += s.v.x * s.v.x / I.x;
            if (I.y > 0)
                ke_rot += s.v.y * s.v.y / I.y;
            if (I.z > 0)
                ke_rot += s.v.z * s.v.z / I.z;
            my_element = ke_rot * Scalar(1.0 / 2.0);
            }
        }

    sdata[threadIdx.x] = my_element;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    unsigned int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            sdata[threadIdx.x] += sdata[threadIdx.x + offs];
        offs >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    if (threadIdx.x == 0)
        args.d_scratch_rot[blockIdx.x] = sdata[0];
    }

//! Reduce the partial sums and compute the thermodynamic properties (one threadgroup)
kernel void gpu_compute_thermo_final_sums(HOOMD_KERNEL_PARAMS(kernel_args::thermo_final_args))
    {
    threadgroup Scalar4* sdata = (threadgroup Scalar4*)hoomd_dynamic_shared;

    Scalar4 final_sum = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    for (unsigned int start = 0; start < args.num_partial_sums; start += blockDim.x)
        {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (start + threadIdx.x < args.num_partial_sums)
            {
            Scalar4 scratch = args.d_scratch[start + threadIdx.x];
            Scalar scratch_rot = args.d_scratch_rot[start + threadIdx.x];
            sdata[threadIdx.x] = make_scalar4(scratch.x, scratch.y, scratch.z, scratch_rot);
            }
        else
            sdata[threadIdx.x] = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        unsigned int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                sdata[threadIdx.x] += sdata[threadIdx.x + offs];
            offs >>= 1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            }

        if (threadIdx.x == 0)
            final_sum += sdata[0];
        }

    if (threadIdx.x == 0)
        {
        Scalar ke_trans_total = final_sum.x * Scalar(0.5);
        Scalar pe_total = final_sum.y + args.external_energy;
        Scalar W = final_sum.z + args.external_virial;
        Scalar ke_rot_total = final_sum.w;

        // compute the pressure
        BoxDim box = args.box;
        Scalar volume;
        Scalar3 L = box.getL();
        if (args.D == 2)
            {
            // "volume" is area in 2D
            volume = L.x * L.y;
            // W needs to be corrected since the 1/3 factor is built in
            W *= Scalar(3.0) / Scalar(2.0);
            }
        else
            {
            volume = L.x * L.y * L.z;
            }

        // pressure: P = (N * K_B * T + W)/V
        Scalar pressure = (Scalar(2.0) * ke_trans_total / Scalar(args.D) + W) / volume;

        args.d_properties[thermo_index::translational_kinetic_energy] = Scalar(ke_trans_total);
        args.d_properties[thermo_index::rotational_kinetic_energy] = Scalar(ke_rot_total);
        args.d_properties[thermo_index::potential_energy] = Scalar(pe_total);
        args.d_properties[thermo_index::pressure] = pressure;
        }
    }

//! Reduce the pressure tensor partial sums (one threadgroup)
kernel void
gpu_compute_pressure_tensor_final_sums(HOOMD_KERNEL_PARAMS(kernel_args::thermo_final_args))
    {
    threadgroup Scalar* sdata = (threadgroup Scalar*)hoomd_dynamic_shared;

    Scalar final_sum[6];
    final_sum[0] = args.external_virial_xx;
    final_sum[1] = args.external_virial_xy;
    final_sum[2] = args.external_virial_xz;
    final_sum[3] = args.external_virial_yy;
    final_sum[4] = args.external_virial_yz;
    final_sum[5] = args.external_virial_zz;

    for (unsigned int start = 0; start < args.num_partial_sums; start += blockDim.x)
        {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (start + threadIdx.x < args.num_partial_sums)
            {
            for (unsigned int i = 0; i < 6; i++)
                sdata[i * blockDim.x + threadIdx.x]
                    = args.d_scratch_pressure_tensor[i * args.num_partial_sums + start
                                                     + threadIdx.x];
            }
        else
            {
            for (unsigned int i = 0; i < 6; i++)
                sdata[i * blockDim.x + threadIdx.x] = Scalar(0.0);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        unsigned int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                {
                for (unsigned int i = 0; i < 6; i++)
                    sdata[i * blockDim.x + threadIdx.x]
                        += sdata[i * blockDim.x + threadIdx.x + offs];
                }
            offs >>= 1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            }

        if (threadIdx.x == 0)
            {
            for (unsigned int i = 0; i < 6; i++)
                final_sum[i] += sdata[i * blockDim.x];
            }
        }

    if (threadIdx.x == 0)
        {
        BoxDim box = args.box;
        Scalar V = box.getVolume(args.twod);
        args.d_properties[thermo_index::pressure_xx] = final_sum[0] / V;
        args.d_properties[thermo_index::pressure_xy] = final_sum[1] / V;
        args.d_properties[thermo_index::pressure_xz] = final_sum[2] / V;
        args.d_properties[thermo_index::pressure_yy] = final_sum[3] / V;
        args.d_properties[thermo_index::pressure_yz] = final_sum[4] / V;
        args.d_properties[thermo_index::pressure_zz] = final_sum[5] / V;
        }
    }

#else

namespace kernel
    {
hipError_t gpu_compute_thermo_partial(Scalar* d_properties,
                                      Scalar4* d_vel,
                                      unsigned int* d_body,
                                      unsigned int* d_tag,
                                      unsigned int* d_group_members,
                                      unsigned int group_size,
                                      const BoxDim& box,
                                      const compute_thermo_args& args,
                                      bool compute_pressure_tensor,
                                      bool compute_rotational_energy)
    {
    assert(d_properties);
    assert(d_vel);
    assert(d_group_members);
    assert(args.d_net_force);
    assert(args.d_net_virial);
    assert(args.d_scratch);

    kernel_args::thermo_partial_args kargs;
    kargs.d_scratch = args.d_scratch;
    kargs.d_scratch_pressure_tensor = args.d_scratch_pressure_tensor;
    kargs.d_scratch_rot = args.d_scratch_rot;
    kargs.d_net_force = args.d_net_force;
    kargs.d_net_virial = args.d_net_virial;
    kargs.d_velocity = d_vel;
    kargs.d_orientation = args.d_orientation;
    kargs.d_angmom = args.d_angmom;
    kargs.d_inertia = args.d_inertia;
    kargs.d_body = d_body;
    kargs.d_tag = d_tag;
    kargs.d_group_members = d_group_members;
    kargs.virial_pitch = static_cast<unsigned int>(args.virial_pitch);
    kargs.work_size = group_size;
    kargs.num_blocks = args.n_blocks;

    dim3 grid(group_size / args.block_size + 1);
    dim3 threads(args.block_size);

    hipError_t error = mtl::launch(HOOMD_MTL_SOURCE,
                                   "hoomd::md::gpu_compute_thermo_partial_sums",
                                   grid,
                                   threads,
                                   kargs,
                                   sizeof(Scalar3) * args.block_size);

    if (compute_pressure_tensor)
        {
        assert(args.d_scratch_pressure_tensor);
        error = mtl::launch(HOOMD_MTL_SOURCE,
                            "hoomd::md::gpu_compute_pressure_tensor_partial_sums",
                            grid,
                            threads,
                            kargs,
                            6 * sizeof(Scalar) * args.block_size);
        }

    if (compute_rotational_energy)
        {
        error = mtl::launch(HOOMD_MTL_SOURCE,
                            "hoomd::md::gpu_compute_rotational_ke_partial_sums",
                            grid,
                            threads,
                            kargs,
                            sizeof(Scalar) * args.block_size);
        }
    return error;
    }

hipError_t gpu_compute_thermo_final(Scalar* d_properties,
                                    Scalar4* d_vel,
                                    unsigned int* d_body,
                                    unsigned int* d_tag,
                                    unsigned int* d_group_members,
                                    unsigned int group_size,
                                    const BoxDim& box,
                                    const compute_thermo_args& args,
                                    bool compute_pressure_tensor,
                                    bool compute_rotational_energy)
    {
    assert(d_properties);
    assert(args.d_scratch);

    const unsigned int final_block_size = 256;

    kernel_args::thermo_final_args kargs;
    kargs.d_properties = d_properties;
    kargs.d_scratch = args.d_scratch;
    kargs.d_scratch_rot = args.d_scratch_rot;
    kargs.d_scratch_pressure_tensor = args.d_scratch_pressure_tensor;
    kargs.box = box;
    kargs.ndof = args.ndof;
    kargs.D = args.D;
    kargs.group_size = group_size;
    kargs.num_partial_sums = args.n_blocks;
    kargs.external_virial
        = Scalar(1.0 / 3.0)
          * (args.external_virial_xx + args.external_virial_yy + args.external_virial_zz);
    kargs.external_energy = args.external_energy;
    kargs.external_virial_xx = args.external_virial_xx;
    kargs.external_virial_xy = args.external_virial_xy;
    kargs.external_virial_xz = args.external_virial_xz;
    kargs.external_virial_yy = args.external_virial_yy;
    kargs.external_virial_yz = args.external_virial_yz;
    kargs.external_virial_zz = args.external_virial_zz;
    kargs.twod = args.D == 2;

    hipError_t error = mtl::launch(HOOMD_MTL_SOURCE,
                                   "hoomd::md::gpu_compute_thermo_final_sums",
                                   dim3(1),
                                   dim3(final_block_size),
                                   kargs,
                                   sizeof(Scalar4) * final_block_size);

    if (compute_pressure_tensor)
        {
        error = mtl::launch(HOOMD_MTL_SOURCE,
                            "hoomd::md::gpu_compute_pressure_tensor_final_sums",
                            dim3(1),
                            dim3(final_block_size),
                            kargs,
                            6 * sizeof(Scalar) * final_block_size);
        }
    return error;
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
