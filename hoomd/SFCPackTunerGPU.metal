// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file SFCPackTunerGPU.metal
    \brief Metal port of SFCPackTunerGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifndef __METAL_VERSION__
#include "SFCPackTunerGPU.cuh"
#include "hoomd/mtl/Primitives.h"
#endif

namespace hoomd
    {
namespace kernel_args
    {
struct sfc_bin_particles_args
    {
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(unsigned int) d_particle_bins;
    HOOMD_DEVPTR(const unsigned int) d_traversal_order;
    HOOMD_DEVPTR(unsigned int) d_sorted_order;
    BoxDim box;
    unsigned int N;
    unsigned int n_grid;
    unsigned int twod;
    };

struct apply_sorted_order_args
    {
    HOOMD_DEVPTR(const unsigned int) d_sorted_order;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_pos_alt;
    HOOMD_DEVPTR(const Scalar4) d_vel;
    HOOMD_DEVPTR(Scalar4) d_vel_alt;
    HOOMD_DEVPTR(const Scalar3) d_accel;
    HOOMD_DEVPTR(Scalar3) d_accel_alt;
    HOOMD_DEVPTR(const Scalar) d_charge;
    HOOMD_DEVPTR(Scalar) d_charge_alt;
    HOOMD_DEVPTR(const Scalar) d_diameter;
    HOOMD_DEVPTR(Scalar) d_diameter_alt;
    HOOMD_DEVPTR(const int3) d_image;
    HOOMD_DEVPTR(int3) d_image_alt;
    HOOMD_DEVPTR(const unsigned int) d_body;
    HOOMD_DEVPTR(unsigned int) d_body_alt;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(unsigned int) d_tag_alt;
    HOOMD_DEVPTR(const Scalar4) d_orientation;
    HOOMD_DEVPTR(Scalar4) d_orientation_alt;
    HOOMD_DEVPTR(const Scalar4) d_angmom;
    HOOMD_DEVPTR(Scalar4) d_angmom_alt;
    HOOMD_DEVPTR(const Scalar3) d_inertia;
    HOOMD_DEVPTR(Scalar3) d_inertia_alt;
    HOOMD_DEVPTR(const Scalar) d_net_virial;
    HOOMD_DEVPTR(Scalar) d_net_virial_alt;
    HOOMD_DEVPTR(const Scalar4) d_net_force;
    HOOMD_DEVPTR(Scalar4) d_net_force_alt;
    HOOMD_DEVPTR(const Scalar4) d_net_torque;
    HOOMD_DEVPTR(Scalar4) d_net_torque_alt;
    HOOMD_DEVPTR(unsigned int) d_rtag;
    unsigned int virial_pitch;
    unsigned int N;
    unsigned int n_ghost;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

//! Assign each particle to its bin on the space filling curve
kernel void gpu_sfc_bin_particles_kernel(HOOMD_KERNEL_PARAMS(kernel_args::sfc_bin_particles_args))
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= args.N)
        return;

    BoxDim box = args.box;
    unsigned int n_grid = args.n_grid;
    Scalar4 postype = args.d_pos[idx];
    Scalar3 p = make_scalar3(postype.x, postype.y, postype.z);
    Scalar3 f = box.makeFraction(p);
    int ib = (unsigned int)(f.x * n_grid) % n_grid;
    int jb = (unsigned int)(f.y * n_grid) % n_grid;
    int kb = (unsigned int)(f.z * n_grid) % n_grid;

    // if the particle is slightly outside, move back into grid
    if (ib < 0)
        ib = 0;
    if (ib >= (int)n_grid)
        ib = n_grid - 1;
    if (jb < 0)
        jb = 0;
    if (jb >= (int)n_grid)
        jb = n_grid - 1;
    if (kb < 0)
        kb = 0;
    if (kb >= (int)n_grid)
        kb = n_grid - 1;

    if (args.twod)
        {
        args.d_particle_bins[idx] = ib * n_grid + jb;
        }
    else
        {
        unsigned int bin = ib * (n_grid * n_grid) + jb * n_grid + kb;
        args.d_particle_bins[idx] = args.d_traversal_order[bin];
        }
    args.d_sorted_order[idx] = idx;
    }

kernel void gpu_apply_sorted_order_kernel(HOOMD_KERNEL_PARAMS(kernel_args::apply_sorted_order_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.N + args.n_ghost)
        return;

    // apply sorted order only for local particles
    unsigned int old_idx = (idx < args.N ? args.d_sorted_order[idx] : idx);
    unsigned int pitch = args.virial_pitch;

    args.d_pos_alt[idx] = args.d_pos[old_idx];
    args.d_vel_alt[idx] = args.d_vel[old_idx];
    args.d_accel_alt[idx] = args.d_accel[old_idx];
    args.d_charge_alt[idx] = args.d_charge[old_idx];
    args.d_diameter_alt[idx] = args.d_diameter[old_idx];
    args.d_image_alt[idx] = args.d_image[old_idx];
    args.d_body_alt[idx] = args.d_body[old_idx];
    unsigned int tag = args.d_tag[old_idx];
    args.d_tag_alt[idx] = tag;
    args.d_orientation_alt[idx] = args.d_orientation[old_idx];
    args.d_angmom_alt[idx] = args.d_angmom[old_idx];
    args.d_inertia_alt[idx] = args.d_inertia[old_idx];
    for (unsigned int j = 0; j < 6; j++)
        args.d_net_virial_alt[j * pitch + idx] = args.d_net_virial[j * pitch + old_idx];
    args.d_net_force_alt[idx] = args.d_net_force[old_idx];
    args.d_net_torque_alt[idx] = args.d_net_torque[old_idx];

    if (idx < args.N)
        {
        // update rtag to point to particle position in new arrays
        args.d_rtag[tag] = idx;
        }
    }

#else

namespace kernel
    {
void gpu_generate_sorted_order(unsigned int N,
                               const Scalar4* d_pos,
                               unsigned int* d_particle_bins,
                               unsigned int* d_traversal_order,
                               unsigned int n_grid,
                               unsigned int* d_sorted_order,
                               const BoxDim& box,
                               bool twod,
                               CachedAllocator& alloc)
    {
    kernel_args::sfc_bin_particles_args args;
    args.d_pos = d_pos;
    args.d_particle_bins = d_particle_bins;
    args.d_traversal_order = d_traversal_order;
    args.d_sorted_order = d_sorted_order;
    args.box = box;
    args.N = N;
    args.n_grid = n_grid;
    args.twod = twod;

    unsigned int block_size = 256;
    unsigned int n_blocks = N / block_size + 1;
    mtl::launch(HOOMD_MTL_SOURCE,
                "hoomd::gpu_sfc_bin_particles_kernel",
                dim3(n_blocks),
                dim3(block_size),
                args);

    // sort particles by bin
    if (N)
        mtl::sort_pairs(d_particle_bins, d_sorted_order, N);
    }

void gpu_apply_sorted_order(unsigned int N,
                            unsigned int n_ghost,
                            const unsigned int* d_sorted_order,
                            const Scalar4* d_pos,
                            Scalar4* d_pos_alt,
                            const Scalar4* d_vel,
                            Scalar4* d_vel_alt,
                            const Scalar3* d_accel,
                            Scalar3* d_accel_alt,
                            const Scalar* d_charge,
                            Scalar* d_charge_alt,
                            const Scalar* d_diameter,
                            Scalar* d_diameter_alt,
                            const int3* d_image,
                            int3* d_image_alt,
                            const unsigned int* d_body,
                            unsigned int* d_body_alt,
                            const unsigned int* d_tag,
                            unsigned int* d_tag_alt,
                            const Scalar4* d_orientation,
                            Scalar4* d_orientation_alt,
                            const Scalar4* d_angmom,
                            Scalar4* d_angmom_alt,
                            const Scalar3* d_inertia,
                            Scalar3* d_inertia_alt,
                            const Scalar* d_net_virial,
                            Scalar* d_net_virial_alt,
                            size_t virial_pitch,
                            const Scalar4* d_net_force,
                            Scalar4* d_net_force_alt,
                            const Scalar4* d_net_torque,
                            Scalar4* d_net_torque_alt,
                            unsigned int* d_rtag)
    {
    kernel_args::apply_sorted_order_args args;
    args.d_sorted_order = d_sorted_order;
    args.d_pos = d_pos;
    args.d_pos_alt = d_pos_alt;
    args.d_vel = d_vel;
    args.d_vel_alt = d_vel_alt;
    args.d_accel = d_accel;
    args.d_accel_alt = d_accel_alt;
    args.d_charge = d_charge;
    args.d_charge_alt = d_charge_alt;
    args.d_diameter = d_diameter;
    args.d_diameter_alt = d_diameter_alt;
    args.d_image = d_image;
    args.d_image_alt = d_image_alt;
    args.d_body = d_body;
    args.d_body_alt = d_body_alt;
    args.d_tag = d_tag;
    args.d_tag_alt = d_tag_alt;
    args.d_orientation = d_orientation;
    args.d_orientation_alt = d_orientation_alt;
    args.d_angmom = d_angmom;
    args.d_angmom_alt = d_angmom_alt;
    args.d_inertia = d_inertia;
    args.d_inertia_alt = d_inertia_alt;
    args.d_net_virial = d_net_virial;
    args.d_net_virial_alt = d_net_virial_alt;
    args.d_net_force = d_net_force;
    args.d_net_force_alt = d_net_force_alt;
    args.d_net_torque = d_net_torque;
    args.d_net_torque_alt = d_net_torque_alt;
    args.d_rtag = d_rtag;
    args.virial_pitch = static_cast<unsigned int>(virial_pitch);
    args.N = N;
    args.n_ghost = n_ghost;

    unsigned int block_size = 256;
    unsigned int n_blocks = (N + n_ghost) / block_size + 1;
    mtl::launch(HOOMD_MTL_SOURCE,
                "hoomd::gpu_apply_sorted_order_kernel",
                dim3(n_blocks),
                dim3(block_size),
                args);
    }
    } // end namespace kernel

#endif

    } // end namespace hoomd
