// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file HarmonicAngleForceGPU.metal
    \brief Metal port of HarmonicAngleForceGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifndef __METAL_VERSION__
#include "HarmonicAngleForceGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct harmonic_angle_args
    {
    HOOMD_DEVPTR(Scalar4) d_force;
    HOOMD_DEVPTR(Scalar) d_virial;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar2) d_params;
    HOOMD_DEVPTR(const group_storage<3>) alist;
    HOOMD_DEVPTR(const unsigned int) apos_list;
    HOOMD_DEVPTR(const unsigned int) n_angles_list;
    BoxDim box;
    unsigned int virial_pitch;
    unsigned int N;
    unsigned int pitch;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

#define SMALL Scalar(0.001)

//! Compute the harmonic angle forces on all N particles (one thread per particle)
kernel void
gpu_compute_harmonic_angle_forces_kernel(HOOMD_KERNEL_PARAMS(kernel_args::harmonic_angle_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.N)
        return;

    BoxDim box = args.box;
    const unsigned int pitch = args.pitch;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    unsigned int n_angles = args.n_angles_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = args.d_pos[idx]; // we can be either a, b, or c in the a-b-c triplet
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 a_pos, b_pos, c_pos; // allocate space for the a,b, and c atom in the a-b-c triplet

    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    Scalar fab[3], fcb[3];

    // initialize the virial to 0
    Scalar virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);

    // loop over all angles
    for (unsigned int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {
        group_storage<3> cur_angle = args.alist[pitch * angle_idx + idx];

        int cur_angle_x_idx = cur_angle.idx[0];
        int cur_angle_y_idx = cur_angle.idx[1];
        int cur_angle_type = cur_angle.idx[2];

        int cur_angle_abc = args.apos_list[pitch * angle_idx + idx];

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 x_postype = args.d_pos[cur_angle_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 y_postype = args.d_pos[cur_angle_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);

        if (cur_angle_abc == 0)
            {
            a_pos = idx_pos;
            b_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 1)
            {
            b_pos = idx_pos;
            a_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 2)
            {
            c_pos = idx_pos;
            a_pos = x_pos;
            b_pos = y_pos;
            }

        // calculate dr for a-b,c-b,and a-c
        Scalar3 dab = a_pos - b_pos;
        Scalar3 dcb = c_pos - b_pos;
        Scalar3 dac = a_pos - c_pos;

        // apply periodic boundary conditions
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        dac = box.minImage(dac);

        // get the angle parameters (MEM TRANSFER: 8 bytes)
        Scalar2 params = args.d_params[cur_angle_type];
        Scalar K = params.x;
        Scalar t_0 = params.y;

        Scalar rsqab = dot(dab, dab);
        Scalar rab = sqrtf(rsqab);
        Scalar rsqcb = dot(dcb, dcb);
        Scalar rcb = sqrtf(rsqcb);

        Scalar c_abbc = dot(dab, dcb);
        c_abbc /= rab * rcb;

        if (c_abbc > Scalar(1.0))
            c_abbc = Scalar(1.0);
        if (c_abbc < -Scalar(1.0))
            c_abbc = -Scalar(1.0);

        Scalar s_abbc = sqrtf(Scalar(1.0) - c_abbc * c_abbc);
        if (s_abbc < SMALL)
            s_abbc = SMALL;
        s_abbc = Scalar(1.0) / s_abbc;

        // actually calculate the force
        Scalar dth = fast::acos(c_abbc) - t_0;
        Scalar tk = K * dth;

        Scalar a = -Scalar(1.0) * tk * s_abbc;
        Scalar a11 = a * c_abbc / rsqab;
        Scalar a12 = -a / (rab * rcb);
        Scalar a22 = a * c_abbc / rsqcb;

        fab[0] = a11 * dab.x + a12 * dcb.x;
        fab[1] = a11 * dab.y + a12 * dcb.y;
        fab[2] = a11 * dab.z + a12 * dcb.z;

        fcb[0] = a22 * dcb.x + a12 * dab.x;
        fcb[1] = a22 * dcb.y + a12 * dab.y;
        fcb[2] = a22 * dcb.z + a12 * dab.z;

        // compute 1/3 of the energy, 1/3 for each atom in the angle
        Scalar angle_eng = tk * dth * Scalar(Scalar(1.0) / Scalar(6.0));

        // upper triangular version of virial tensor
        Scalar angle_virial[6];
        angle_virial[0] = Scalar(1. / 3.) * (dab.x * fab[0] + dcb.x * fcb[0]);
        angle_virial[1] = Scalar(1. / 3.) * (dab.y * fab[0] + dcb.y * fcb[0]);
        angle_virial[2] = Scalar(1. / 3.) * (dab.z * fab[0] + dcb.z * fcb[0]);
        angle_virial[3] = Scalar(1. / 3.) * (dab.y * fab[1] + dcb.y * fcb[1]);
        angle_virial[4] = Scalar(1. / 3.) * (dab.z * fab[1] + dcb.z * fcb[1]);
        angle_virial[5] = Scalar(1. / 3.) * (dab.z * fab[2] + dcb.z * fcb[2]);

        if (cur_angle_abc == 0)
            {
            force_idx.x += fab[0];
            force_idx.y += fab[1];
            force_idx.z += fab[2];
            }
        if (cur_angle_abc == 1)
            {
            force_idx.x -= fab[0] + fcb[0];
            force_idx.y -= fab[1] + fcb[1];
            force_idx.z -= fab[2] + fcb[2];
            }
        if (cur_angle_abc == 2)
            {
            force_idx.x += fcb[0];
            force_idx.y += fcb[1];
            force_idx.z += fcb[2];
            }

        force_idx.w += angle_eng;

        for (int i = 0; i < 6; i++)
            virial[i] += angle_virial[i];
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    args.d_force[idx] = force_idx;
    for (int i = 0; i < 6; i++)
        args.d_virial[i * args.virial_pitch + idx] = virial[i];
    }

#else

namespace kernel
    {
hipError_t gpu_compute_harmonic_angle_forces(Scalar4* d_force,
                                             Scalar* d_virial,
                                             const size_t virial_pitch,
                                             const unsigned int N,
                                             const Scalar4* d_pos,
                                             const BoxDim& box,
                                             const group_storage<3>* atable,
                                             const unsigned int* apos_list,
                                             const unsigned int pitch,
                                             const unsigned int* n_angles_list,
                                             Scalar2* d_params,
                                             unsigned int n_angle_types,
                                             int block_size)
    {
    assert(d_params);

    const char* name = "hoomd::md::gpu_compute_harmonic_angle_forces_kernel";
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name);
    unsigned int run_block_size = std::min((unsigned int)block_size, max_block_size);

    kernel_args::harmonic_angle_args args;
    args.d_force = d_force;
    args.d_virial = d_virial;
    args.d_pos = d_pos;
    args.d_params = d_params;
    args.alist = atable;
    args.apos_list = apos_list;
    args.n_angles_list = n_angles_list;
    args.box = box;
    args.virial_pitch = static_cast<unsigned int>(virial_pitch);
    args.N = N;
    args.pitch = pitch;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(N / run_block_size + 1),
                       dim3(run_block_size),
                       args);
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
