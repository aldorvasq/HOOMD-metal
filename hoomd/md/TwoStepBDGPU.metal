// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file TwoStepBDGPU.metal
    \brief Metal port of TwoStepBDGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/ParticleData.cuh"

#ifndef __METAL_VERSION__
#include "TwoStepBDGPU.cuh"
#include "TwoStepLangevinGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct gpu_brownian_step_one_kernel_args
    {
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(int3) d_image;
    BoxDim box;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(const unsigned int) d_group_members;
    unsigned int nwork;
    HOOMD_DEVPTR(const Scalar4) d_net_force;
    HOOMD_DEVPTR(const Scalar3) d_gamma_r;
    HOOMD_DEVPTR(Scalar4) d_orientation;
    HOOMD_DEVPTR(Scalar4) d_torque;
    HOOMD_DEVPTR(const Scalar3) d_inertia;
    HOOMD_DEVPTR(Scalar4) d_angmom;
    HOOMD_DEVPTR(const Scalar) d_gamma;
    unsigned int n_types;
    uint64_t timestep;
    uint16_t seed;
    Scalar T;
    bool aniso;
    Scalar deltaT;
    unsigned int D;
    bool d_noiseless_t;
    bool d_noiseless_r;
    bool enable_shared_cache;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void gpu_brownian_step_one_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_brownian_step_one_kernel_args))
    {
    device Scalar4* d_pos = args.d_pos;
    device Scalar4* d_vel = args.d_vel;
    device int3* d_image = args.d_image;
    const BoxDim box = args.box;
    device const unsigned int* d_tag = args.d_tag;
    device const unsigned int* d_group_members = args.d_group_members;
    const unsigned int nwork = args.nwork;
    device const Scalar4* d_net_force = args.d_net_force;
    device const Scalar3* d_gamma_r = args.d_gamma_r;
    device Scalar4* d_orientation = args.d_orientation;
    device Scalar4* d_torque = args.d_torque;
    device const Scalar3* d_inertia = args.d_inertia;
    device Scalar4* d_angmom = args.d_angmom;
    device const Scalar* d_gamma = args.d_gamma;
    const unsigned int n_types = args.n_types;
    const uint64_t timestep = args.timestep;
    const uint16_t seed = args.seed;
    const Scalar T = args.T;
    const bool aniso = args.aniso;
    const Scalar deltaT = args.deltaT;
    unsigned int D = args.D;
    const bool d_noiseless_t = args.d_noiseless_t;
    const bool d_noiseless_r = args.d_noiseless_r;
    bool enable_shared_cache = args.enable_shared_cache;

    HIP_DYNAMIC_SHARED(char, s_data)

    threadgroup Scalar3* s_gammas_r = (threadgroup Scalar3*)s_data;
    threadgroup Scalar* s_gammas = (threadgroup Scalar*)(s_gammas_r + n_types);

    if (enable_shared_cache)
        {
        // read in the gamma (1 dimensional array), stored in s_gammas[0: n_type] (Pythonic
        // convention)
        for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < n_types)
                s_gammas[cur_offset + threadIdx.x] = d_gamma[cur_offset + threadIdx.x];
            }
        __syncthreads();

        // read in the gamma_r, stored in s_gammas_r[0: n_type], which is s_gamma_r[0:n_type]

        for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < n_types)
                s_gammas_r[cur_offset + threadIdx.x] = d_gamma_r[cur_offset + threadIdx.x];
            }
        __syncthreads();
        }

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_idx < nwork)
        {
        const unsigned int group_idx = local_idx;

        // determine the particle to work on
        unsigned int idx = d_group_members[group_idx];
        Scalar4 postype = d_pos[idx];
        Scalar4 vel = d_vel[idx];
        Scalar4 net_force = d_net_force[idx];
        int3 image = d_image[idx];

        // read in the tag of our particle.
        unsigned int ptag = d_tag[idx];

        // compute the random force
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),
                            hoomd::Counter(ptag));
        UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
        Scalar rx = uniform(rng);
        Scalar ry = uniform(rng);
        Scalar rz = uniform(rng);

        // calculate the magnitude of the random force
        Scalar gamma;
        // determine gamma from type
        unsigned int typ = __scalar_as_int(postype.w);
        if (enable_shared_cache)
            {
            gamma = s_gammas[typ];
            }
        else
            {
            gamma = d_gamma[typ];
            }

        // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform -1,1
        // distribution it is not the dimensionality of the system
        Scalar coeff = fast::sqrt(Scalar(3.0) * Scalar(2.0) * gamma * T / deltaT);
        if (d_noiseless_t)
            coeff = Scalar(0.0);
        Scalar Fr_x = rx * coeff;
        Scalar Fr_y = ry * coeff;
        Scalar Fr_z = rz * coeff;

        if (D < 3)
            Fr_z = Scalar(0.0);

        // update position
        postype.x += (net_force.x + Fr_x) * deltaT / gamma;
        postype.y += (net_force.y + Fr_y) * deltaT / gamma;
        if (D > 2)
            {
            postype.z += (net_force.z + Fr_z) * deltaT / gamma;
            }

        // particles may have been moved slightly outside the box by the above steps, wrap them back
        // into place
        box.wrap(postype, vel, image);

        if (d_noiseless_t)
            {
            vel.x = net_force.x / gamma;
            vel.y = net_force.y / gamma;
            if (D > 2)
                vel.z = net_force.z / gamma;
            else
                vel.z = 0;
            }
        else
            {
            // draw a new random velocity for particle j
            Scalar mass = vel.w;
            Scalar sigma = fast::sqrt(T / mass);
            NormalDistribution<Scalar> normal(sigma);
            vel.x = normal(rng);
            vel.y = normal(rng);
            if (D > 2)
                vel.z = normal(rng);
            else
                vel.z = 0;
            }

        // write out data
        d_pos[idx] = postype;
        d_vel[idx] = vel;
        d_image[idx] = image;

        // rotational random force and orientation quaternion updates
        if (aniso)
            {
            unsigned int type_r = __scalar_as_int(d_pos[idx].w);

            // gamma_r is stored in the second half of s_gammas a.k.a s_gammas_r
            Scalar3 gamma_r;
            if (enable_shared_cache)
                {
                gamma_r = s_gammas_r[type_r];
                }
            else
                {
                gamma_r = d_gamma_r[type_r];
                }

            if (gamma_r.x > 0 || gamma_r.y > 0 || gamma_r.z > 0)
                {
                vec3<Scalar> p_vec;
                quat<Scalar> q(d_orientation[idx]);
                vec3<Scalar> t(d_torque[idx]);
                vec3<Scalar> I(d_inertia[idx]);

                // check if the shape is degenerate
                bool x_zero, y_zero, z_zero;
                x_zero = (I.x == 0);
                y_zero = (I.y == 0);
                z_zero = (I.z == 0);

                Scalar3 sigma_r = make_scalar3(fast::sqrt(Scalar(2.0) * gamma_r.x * T / deltaT),
                                               fast::sqrt(Scalar(2.0) * gamma_r.y * T / deltaT),
                                               fast::sqrt(Scalar(2.0) * gamma_r.z * T / deltaT));
                if (d_noiseless_r)
                    sigma_r = make_scalar3(0, 0, 0);

                // original Gaussian random torque
                // Gaussian random distribution is preferred in terms of preserving the exact math
                vec3<Scalar> bf_torque;
                bf_torque.x = NormalDistribution<Scalar>(sigma_r.x)(rng);
                bf_torque.y = NormalDistribution<Scalar>(sigma_r.y)(rng);
                bf_torque.z = NormalDistribution<Scalar>(sigma_r.z)(rng);

                if (x_zero || D == 2)
                    {
                    bf_torque.x = 0;
                    t.x = 0;
                    }
                if (y_zero || D == 2)
                    {
                    bf_torque.y = 0;
                    t.y = 0;
                    }
                if (z_zero)
                    {
                    bf_torque.z = 0;
                    t.z = 0;
                    }

                // use the damping by gamma_r and rotate back to lab frame
                // For Future Updates: take special care when have anisotropic gamma_r
                bf_torque = rotate(q, bf_torque);

                // do the integration for quaternion
                q += Scalar(0.5) * deltaT * ((t + bf_torque) / vec3<Scalar>(gamma_r)) * q;
                q = q * (Scalar(1.0) / slow::sqrt(norm2(q)));
                d_orientation[idx] = quat_to_scalar4(q);

                if (d_noiseless_r)
                    {
                    p_vec.x = t.x / gamma_r.x;
                    p_vec.y = t.y / gamma_r.y;
                    p_vec.z = t.z / gamma_r.z;
                    }
                else
                    {
                    // draw a new random ang_mom for particle j in body frame
                    p_vec.x = NormalDistribution<Scalar>(fast::sqrt(T * I.x))(rng);
                    p_vec.y = NormalDistribution<Scalar>(fast::sqrt(T * I.y))(rng);
                    p_vec.z = NormalDistribution<Scalar>(fast::sqrt(T * I.z))(rng);
                    }

                if (x_zero || D == 2)
                    p_vec.x = 0;
                if (y_zero || D == 2)
                    p_vec.y = 0;
                if (z_zero)
                    p_vec.z = 0;

                // store ang_mom quaternion
                quat<Scalar> p = Scalar(2.0) * q * p_vec;
                d_angmom[idx] = quat_to_scalar4(p);
                }
            }
        }
    }

#else

namespace kernel
    {
//! Takes the second half-step forward in the Langevin integration on a group of particles with
/*! \param d_pos array of particle positions and types
    \param d_vel array of particle positions and masses
    \param d_image array of particle images
    \param box simulation box
    \param d_tag array of particle tags
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param nwork Number of group members to process on this GPU
    \param d_net_force Net force on each particle
    \param d_gamma_r List of per-type gamma_rs (rotational drag coeff.)
    \param d_orientation Device array of orientation quaternion
    \param d_torque Device array of net torque on each particle
    \param d_inertia Device array of moment of inertial of each particle
    \param d_angmom Device array of transformed angular momentum quaternion of each particle (see
        online documentation)
    \param d_gamma List of per-type gammas
    \param n_types Number of particle types in the simulation
    \param d_noiseless_t If set true, there will be no translational noise (random force)
    \param d_noiseless_r If set true, there will be no rotational noise (random torque)

    This kernel is implemented in a very similar manner to gpu_nve_step_one_kernel(), see it for
   design details.

    This kernel must be launched with enough dynamic shared memory per block to read in d_gamma
*/


/*! \param d_pos array of particle positions and types
    \param d_vel array of particle positions and masses
    \param d_image array of particle images
    \param box simulation box
    \param d_tag array of particle tags
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param d_gamma_r List of per-type gamma_rs (rotational drag coeff.)
    \param d_orientation Device array of orientation quaternion
    \param d_torque Device array of net torque on each particle
    \param d_inertia Device array of moment of inertial of each particle
    \param d_angmom Device array of transformed angular momentum quaternion of each particle (see
   online documentation) \param langevin_args Collected arguments for gpu_brownian_step_one_kernel()
    \param aniso If set true, the system would go through rigid body updates for its orientation
    \param deltaT Amount of real time to step forward in one time step
    \param D Dimensionality of the system
    \param d_noiseless_t If set true, there will be no translational noise (random force)
    \param d_noiseless_r If set true, there will be no rotational noise (random torque)

    This is just a driver for gpu_brownian_step_one_kernel(), see it for details.
*/
hipError_t gpu_brownian_step_one(Scalar4* d_pos,
                                 Scalar4* d_vel,
                                 int3* d_image,
                                 const BoxDim& box,
                                 const unsigned int* d_tag,
                                 const unsigned int* d_group_members,
                                 const unsigned int group_size,
                                 const Scalar4* d_net_force,
                                 const Scalar3* d_gamma_r,
                                 Scalar4* d_orientation,
                                 Scalar4* d_torque,
                                 const Scalar3* d_inertia,
                                 Scalar4* d_angmom,
                                 const langevin_step_two_args& langevin_args,
                                 const bool aniso,
                                 const Scalar deltaT,
                                 const unsigned int D,
                                 const bool d_noiseless_t,
                                 const bool d_noiseless_r)
    {
    unsigned int run_block_size = 256;

    unsigned int nwork = group_size;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    auto shared_bytes
        = (sizeof(Scalar) * langevin_args.n_types + sizeof(Scalar3) * langevin_args.n_types);

    bool enable_shared_cache = true;

    if (shared_bytes > langevin_args.devprop.sharedMemPerBlock)
        {
        enable_shared_cache = false;
        shared_bytes = 0;
        }

    // run the kernel
    {
        kernel_args::gpu_brownian_step_one_kernel_args _args;
        _args.d_pos = d_pos;
        _args.d_vel = d_vel;
        _args.d_image = d_image;
        _args.box = box;
        _args.d_tag = d_tag;
        _args.d_group_members = d_group_members;
        _args.nwork = nwork;
        _args.d_net_force = d_net_force;
        _args.d_gamma_r = d_gamma_r;
        _args.d_orientation = d_orientation;
        _args.d_torque = d_torque;
        _args.d_inertia = d_inertia;
        _args.d_angmom = d_angmom;
        _args.d_gamma = langevin_args.d_gamma;
        _args.n_types = langevin_args.n_types;
        _args.timestep = langevin_args.timestep;
        _args.seed = langevin_args.seed;
        _args.T = langevin_args.T;
        _args.aniso = aniso;
        _args.deltaT = deltaT;
        _args.D = D;
        _args.d_noiseless_t = d_noiseless_t;
        _args.d_noiseless_r = d_noiseless_r;
        _args.enable_shared_cache = enable_shared_cache;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_brownian_step_one_kernel",
                           dim3(grid),
                           dim3(threads),
                           _args,
                           shared_bytes);
        }

    return hipSuccess;
    }

    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
