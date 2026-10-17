// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file TwoStepConstantPressureGPU.metal
    \brief Metal port of TwoStepConstantPressureGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/VectorMath.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#ifndef __METAL_VERSION__
#include "TwoStepConstantPressureGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct gpu_npt_mtk_step_one_kernel_args
    {
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(const Scalar3) d_accel;
    HOOMD_DEVPTR(unsigned int) d_group_members;
    unsigned int nwork;
    Scalar thermo_rescale;
    Scalar mat_exp_v_xx;
    Scalar mat_exp_v_xy;
    Scalar mat_exp_v_xz;
    Scalar mat_exp_v_yy;
    Scalar mat_exp_v_yz;
    Scalar mat_exp_v_zz;
    Scalar mat_exp_r_xx;
    Scalar mat_exp_r_xy;
    Scalar mat_exp_r_xz;
    Scalar mat_exp_r_yy;
    Scalar mat_exp_r_yz;
    Scalar mat_exp_r_zz;
    Scalar mat_exp_r_int_xx;
    Scalar mat_exp_r_int_xy;
    Scalar mat_exp_r_int_xz;
    Scalar mat_exp_r_int_yy;
    Scalar mat_exp_r_int_yz;
    Scalar mat_exp_r_int_zz;
    Scalar deltaT;
    bool rescale_all;
    unsigned int n_dimensions;
    };

struct gpu_npt_mtk_wrap_kernel_args
    {
    unsigned int nwork;
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(int3) d_image;
    BoxDim box;
    };

struct gpu_npt_mtk_step_two_kernel_args
    {
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(Scalar3) d_accel;
    HOOMD_DEVPTR(const Scalar4) d_net_force;
    HOOMD_DEVPTR(unsigned int) d_group_members;
    unsigned int nwork;
    Scalar mat_exp_v_xx;
    Scalar mat_exp_v_xy;
    Scalar mat_exp_v_xz;
    Scalar mat_exp_v_yy;
    Scalar mat_exp_v_yz;
    Scalar mat_exp_v_zz;
    Scalar deltaT;
    Scalar thermo_rescale;
    unsigned int n_dimensions;
    };

struct gpu_npt_mtk_rescale_kernel_args
    {
    unsigned int nwork;
    HOOMD_DEVPTR(Scalar4) d_postype;
    Scalar mat_exp_r_xx;
    Scalar mat_exp_r_xy;
    Scalar mat_exp_r_xz;
    Scalar mat_exp_r_yy;
    Scalar mat_exp_r_yz;
    Scalar mat_exp_r_zz;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void gpu_npt_mtk_step_one_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_npt_mtk_step_one_kernel_args))
    {
    device Scalar4* d_pos = args.d_pos;
    device Scalar4* d_vel = args.d_vel;
    device const Scalar3* d_accel = args.d_accel;
    device unsigned int* d_group_members = args.d_group_members;
    const unsigned int nwork = args.nwork;
    Scalar thermo_rescale = args.thermo_rescale;
    Scalar mat_exp_v_xx = args.mat_exp_v_xx;
    Scalar mat_exp_v_xy = args.mat_exp_v_xy;
    Scalar mat_exp_v_xz = args.mat_exp_v_xz;
    Scalar mat_exp_v_yy = args.mat_exp_v_yy;
    Scalar mat_exp_v_yz = args.mat_exp_v_yz;
    Scalar mat_exp_v_zz = args.mat_exp_v_zz;
    Scalar mat_exp_r_xx = args.mat_exp_r_xx;
    Scalar mat_exp_r_xy = args.mat_exp_r_xy;
    Scalar mat_exp_r_xz = args.mat_exp_r_xz;
    Scalar mat_exp_r_yy = args.mat_exp_r_yy;
    Scalar mat_exp_r_yz = args.mat_exp_r_yz;
    Scalar mat_exp_r_zz = args.mat_exp_r_zz;
    Scalar mat_exp_r_int_xx = args.mat_exp_r_int_xx;
    Scalar mat_exp_r_int_xy = args.mat_exp_r_int_xy;
    Scalar mat_exp_r_int_xz = args.mat_exp_r_int_xz;
    Scalar mat_exp_r_int_yy = args.mat_exp_r_int_yy;
    Scalar mat_exp_r_int_yz = args.mat_exp_r_int_yz;
    Scalar mat_exp_r_int_zz = args.mat_exp_r_int_zz;
    Scalar deltaT = args.deltaT;
    bool rescale_all = args.rescale_all;
    unsigned int n_dimensions = args.n_dimensions;

    // determine which particle this thread works on
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize eigenvectors
    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx;
        unsigned int idx = d_group_members[group_idx];

        // fetch particle position
        Scalar4 pos = d_pos[idx];

        // fetch particle velocity and acceleration
        Scalar4 vel = d_vel[idx];
        Scalar3 v = make_scalar3(vel.x, vel.y, vel.z);
        Scalar3 accel = d_accel[idx];
        ;
        Scalar3 r = make_scalar3(pos.x, pos.y, pos.z);

        if (n_dimensions == 2)
            {
            accel.z = Scalar(0.0);
            }

        // advance velocity
        v += deltaT / Scalar(2.0) * accel;

        // propagate velocity by half a time step and position by the full time step
        // by multiplying with upper triangular matrix
        v.x = mat_exp_v_xx * v.x + mat_exp_v_xy * v.y + mat_exp_v_xz * v.z;
        v.y = mat_exp_v_yy * v.y + mat_exp_v_yz * v.z;
        v.z = mat_exp_v_zz * v.z;

        // apply thermostat update of velocity
        v *= thermo_rescale;

        if (!rescale_all)
            {
            // rescale this group of particles
            r.x = mat_exp_r_xx * r.x + mat_exp_r_xy * r.y + mat_exp_r_xz * r.z;
            r.y = mat_exp_r_yy * r.y + mat_exp_r_yz * r.z;
            r.z = mat_exp_r_zz * r.z;
            }

        r.x += mat_exp_r_int_xx * v.x + mat_exp_r_int_xy * v.y + mat_exp_r_int_xz * v.z;
        r.y += mat_exp_r_int_yy * v.y + mat_exp_r_int_yz * v.z;
        r.z += mat_exp_r_int_zz * v.z;

        // write out the results
        d_pos[idx] = make_scalar4(r.x, r.y, r.z, pos.w);
        d_vel[idx] = make_scalar4(v.x, v.y, v.z, vel.w);
        }
    }

kernel void gpu_npt_mtk_wrap_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_npt_mtk_wrap_kernel_args))
    {
    const unsigned int nwork = args.nwork;
    device Scalar4* d_pos = args.d_pos;
    device Scalar4* d_vel = args.d_vel;
    device int3* d_image = args.d_image;
    BoxDim box = args.box;

    // determine which particle this thread works on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // wrap ALL particles in the box
    if (idx < nwork)
        {
        // fetch particle position
        Scalar4 postype = d_pos[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        Scalar4 velmass = d_vel[idx];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);

        // read in the image flags
        int3 image = d_image[idx];

        // fix periodic boundary conditions
        box.wrap(pos, vel, image);

        // write out the results
        d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
        d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, velmass.w);
        d_image[idx] = image;
        }
    }

kernel void gpu_npt_mtk_step_two_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_npt_mtk_step_two_kernel_args))
    {
    device Scalar4* d_vel = args.d_vel;
    device Scalar3* d_accel = args.d_accel;
    device const Scalar4* d_net_force = args.d_net_force;
    device unsigned int* d_group_members = args.d_group_members;
    const unsigned int nwork = args.nwork;
    Scalar mat_exp_v_xx = args.mat_exp_v_xx;
    Scalar mat_exp_v_xy = args.mat_exp_v_xy;
    Scalar mat_exp_v_xz = args.mat_exp_v_xz;
    Scalar mat_exp_v_yy = args.mat_exp_v_yy;
    Scalar mat_exp_v_yz = args.mat_exp_v_yz;
    Scalar mat_exp_v_zz = args.mat_exp_v_zz;
    Scalar deltaT = args.deltaT;
    Scalar thermo_rescale = args.thermo_rescale;
    unsigned int n_dimensions = args.n_dimensions;

    // determine which particle this thread works on
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx;
        unsigned int idx = d_group_members[group_idx];

        // fetch particle velocity and acceleration
        Scalar4 vel = d_vel[idx];

        // compute acceleration
        Scalar minv = Scalar(1.0) / vel.w;
        Scalar4 net_force = d_net_force[idx];

        if (n_dimensions == 2)
            {
            net_force.z = Scalar(0.0);
            }

        Scalar3 accel = make_scalar3(net_force.x, net_force.y, net_force.z);
        accel *= minv;

        Scalar3 v = make_scalar3(vel.x, vel.y, vel.z);

        // apply thermostat rescaling
        v = v * thermo_rescale;

        // propagate velocity by half a time step by multiplying with an upper triangular matrix
        v.x = mat_exp_v_xx * v.x + mat_exp_v_xy * v.y + mat_exp_v_xz * v.z;
        v.y = mat_exp_v_yy * v.y + mat_exp_v_yz * v.z;
        v.z = mat_exp_v_zz * v.z;

        // advance velocity
        v += deltaT / Scalar(2.0) * accel;

        // write out velocity
        d_vel[idx] = make_scalar4(v.x, v.y, v.z, vel.w);

        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = accel;
        }
    }

kernel void gpu_npt_mtk_rescale_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_npt_mtk_rescale_kernel_args))
    {
    const unsigned int nwork = args.nwork;
    device Scalar4* d_postype = args.d_postype;
    Scalar mat_exp_r_xx = args.mat_exp_r_xx;
    Scalar mat_exp_r_xy = args.mat_exp_r_xy;
    Scalar mat_exp_r_xz = args.mat_exp_r_xz;
    Scalar mat_exp_r_yy = args.mat_exp_r_yy;
    Scalar mat_exp_r_yz = args.mat_exp_r_yz;
    Scalar mat_exp_r_zz = args.mat_exp_r_zz;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nwork)
        return;

    // rescale position
    Scalar4 postype = d_postype[idx];
    Scalar3 r = make_scalar3(postype.x, postype.y, postype.z);

    r.x = mat_exp_r_xx * r.x + mat_exp_r_xy * r.y + mat_exp_r_xz * r.z;
    r.y = mat_exp_r_yy * r.y + mat_exp_r_yz * r.z;
    r.z = mat_exp_r_zz * r.z;

    d_postype[idx] = make_scalar4(r.x, r.y, r.z, postype.w);
    }

#else

namespace kernel
    {
//! Kernel to propagate the positions and velocities, first half of NPT update


/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param thermo_rescale Update factor for thermostat
    \param mat_exp_v Matrix exponential for velocity update
    \param mat_exp_r Matrix exponential for position update
    \param mat_exp_r_int Integrated matrix exp for position update
    \param deltaT Time to advance (for one full step)
    \param deltaT Time to move forward in one whole step
    \param rescale_all True if all particles in the system should be rescaled at once

    This is just a kernel driver for gpu_npt_mtk_step_one_kernel(). See it for more details.
*/
hipError_t gpu_npt_rescale_step_one(Scalar4* d_pos,
                                    Scalar4* d_vel,
                                    const Scalar3* d_accel,
                                    unsigned int* d_group_members,
                                    const unsigned int group_size,
                                    Scalar thermo_rescale,
                                    Scalar* mat_exp_v,
                                    Scalar* mat_exp_r,
                                    Scalar* mat_exp_r_int,
                                    Scalar deltaT,
                                    bool rescale_all,
                                    const unsigned int block_size,
                                    unsigned int n_dimensions)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_npt_mtk_step_one_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    unsigned int nwork = group_size;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    {
        kernel_args::gpu_npt_mtk_step_one_kernel_args _args;
        _args.d_pos = d_pos;
        _args.d_vel = d_vel;
        _args.d_accel = d_accel;
        _args.d_group_members = d_group_members;
        _args.nwork = nwork;
        _args.thermo_rescale = thermo_rescale;
        _args.mat_exp_v_xx = mat_exp_v[0];
        _args.mat_exp_v_xy = mat_exp_v[1];
        _args.mat_exp_v_xz = mat_exp_v[2];
        _args.mat_exp_v_yy = mat_exp_v[3];
        _args.mat_exp_v_yz = mat_exp_v[4];
        _args.mat_exp_v_zz = mat_exp_v[5];
        _args.mat_exp_r_xx = mat_exp_r[0];
        _args.mat_exp_r_xy = mat_exp_r[1];
        _args.mat_exp_r_xz = mat_exp_r[2];
        _args.mat_exp_r_yy = mat_exp_r[3];
        _args.mat_exp_r_yz = mat_exp_r[4];
        _args.mat_exp_r_zz = mat_exp_r[5];
        _args.mat_exp_r_int_xx = mat_exp_r_int[0];
        _args.mat_exp_r_int_xy = mat_exp_r_int[1];
        _args.mat_exp_r_int_xz = mat_exp_r_int[2];
        _args.mat_exp_r_int_yy = mat_exp_r_int[3];
        _args.mat_exp_r_int_yz = mat_exp_r_int[4];
        _args.mat_exp_r_int_zz = mat_exp_r_int[5];
        _args.deltaT = deltaT;
        _args.rescale_all = rescale_all;
        _args.n_dimensions = n_dimensions;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_npt_mtk_step_one_kernel",
                           dim3(grid),
                           dim3(threads),
                           _args,
                           0);
        }

    return hipSuccess;
    }

/*! \param N number of particles in the system
    \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_image array of particle images
    \param box The new box the particles where the particles now reside

    Wrap particle positions for all particles in the box
*/


/*! \param N number of particles in the system
    \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_image array of particle images
    \param box The new box the particles where the particles now reside

    This is just a kernel driver for gpu_npt_mtk_wrap_kernel(). See it for more details.
*/
hipError_t gpu_npt_rescale_wrap(const unsigned int N,
                                Scalar4* d_pos,
                                Scalar4* d_vel,
                                int3* d_image,
                                const BoxDim& box,
                                const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_npt_mtk_wrap_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    unsigned int nwork = N;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    {
        kernel_args::gpu_npt_mtk_wrap_kernel_args _args;
        _args.nwork = nwork;
        _args.d_pos = d_pos;
        _args.d_vel = d_vel;
        _args.d_image = d_image;
        _args.box = box;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_npt_mtk_wrap_kernel",
                           dim3(grid),
                           dim3(threads),
                           _args,
                           0);
        }

    return hipSuccess;
    }

//! Kernel to propagate the positions and velocities, second half of NPT update


/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param mat_exp_v Matrix exponential for velocity update
    \param d_net_force Net force on each particle

    \param deltaT Time to move forward in one whole step

    This is just a kernel driver for gpu_npt_mtk_step_kernel(). See it for more details.
*/
hipError_t gpu_npt_rescale_step_two(Scalar4* d_vel,
                                    Scalar3* d_accel,
                                    unsigned int* d_group_members,
                                    const unsigned int group_size,
                                    Scalar4* d_net_force,
                                    Scalar* mat_exp_v,
                                    Scalar deltaT,
                                    Scalar thermo_rescale,
                                    const unsigned int block_size,
                                    unsigned int n_dimensions)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_npt_mtk_step_two_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    unsigned int nwork = group_size;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    {
        kernel_args::gpu_npt_mtk_step_two_kernel_args _args;
        _args.d_vel = d_vel;
        _args.d_accel = d_accel;
        _args.d_net_force = d_net_force;
        _args.d_group_members = d_group_members;
        _args.nwork = nwork;
        _args.mat_exp_v_xx = mat_exp_v[0];
        _args.mat_exp_v_xy = mat_exp_v[1];
        _args.mat_exp_v_xz = mat_exp_v[2];
        _args.mat_exp_v_yy = mat_exp_v[3];
        _args.mat_exp_v_yz = mat_exp_v[4];
        _args.mat_exp_v_zz = mat_exp_v[5];
        _args.deltaT = deltaT;
        _args.thermo_rescale = thermo_rescale;
        _args.n_dimensions = n_dimensions;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_npt_mtk_step_two_kernel",
                           dim3(grid),
                           dim3(threads),
                           _args,
                           0);
        }

    return hipSuccess;
    }



void gpu_npt_rescale_rescale(const unsigned int N,
                             Scalar4* d_postype,
                             Scalar mat_exp_r_xx,
                             Scalar mat_exp_r_xy,
                             Scalar mat_exp_r_xz,
                             Scalar mat_exp_r_yy,
                             Scalar mat_exp_r_yz,
                             Scalar mat_exp_r_zz,
                             const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_npt_mtk_rescale_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    unsigned int nwork = N;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    {
        kernel_args::gpu_npt_mtk_rescale_kernel_args _args;
        _args.nwork = nwork;
        _args.d_postype = d_postype;
        _args.mat_exp_r_xx = mat_exp_r_xx;
        _args.mat_exp_r_xy = mat_exp_r_xy;
        _args.mat_exp_r_xz = mat_exp_r_xz;
        _args.mat_exp_r_yy = mat_exp_r_yy;
        _args.mat_exp_r_yz = mat_exp_r_yz;
        _args.mat_exp_r_zz = mat_exp_r_zz;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_npt_mtk_rescale_kernel",
                           dim3(grid),
                           dim3(threads),
                           _args,
                           0);
        }
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
