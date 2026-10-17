// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file TwoStepRATTLEManifoldGPU.metal
    \brief Metal port of TwoStepRATTLEManifoldGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/BoxDim.h"

#ifndef __METAL_VERSION__
#include "TwoStepRATTLEBDGPU.cuh"
#include "TwoStepRATTLELangevinGPU.cuh"
#include "TwoStepRATTLENVEGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct gpu_rattle_bdtally_reduce_partial_sum_kernel_args
    {
    HOOMD_DEVPTR(Scalar) d_sum;
    HOOMD_DEVPTR(Scalar) d_partial_sum;
    unsigned int num_blocks;
    };

struct gpu_rattle_brownian_step_one_kernel_args
    {
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(int3) d_image;
    HOOMD_DEVPTR(Scalar4) d_vel;
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
    size_t n_types;
    uint64_t timestep;
    uint16_t seed;
    Scalar T;
    MANIFOLD_CLASS manifold;
    bool aniso;
    Scalar deltaT;
    unsigned int D;
    bool d_noiseless_t;
    bool d_noiseless_r;
    };

struct gpu_include_rattle_force_bd_kernel_args
    {
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_net_force;
    HOOMD_DEVPTR(Scalar) d_net_virial;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(const unsigned int) d_group_members;
    unsigned int nwork;
    HOOMD_DEVPTR(const Scalar) d_gamma;
    size_t n_types;
    uint64_t timestep;
    uint16_t seed;
    Scalar T;
    Scalar tolerance;
    MANIFOLD_CLASS manifold;
    size_t net_virial_pitch;
    Scalar deltaT;
    bool d_noiseless_t;
    };

struct gpu_rattle_langevin_step_two_kernel_args
    {
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(Scalar3) d_accel;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(unsigned int) d_group_members;
    unsigned int group_size;
    HOOMD_DEVPTR(Scalar4) d_net_force;
    HOOMD_DEVPTR(Scalar) d_gamma;
    size_t n_types;
    uint64_t timestep;
    uint16_t seed;
    Scalar T;
    Scalar tolerance;
    bool noiseless_t;
    MANIFOLD_CLASS manifold;
    Scalar deltaT;
    unsigned int D;
    bool tally;
    HOOMD_DEVPTR(Scalar) d_partial_sum_bdenergy;
    };

struct gpu_rattle_nve_step_two_kernel_args
    {
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(Scalar3) d_accel;
    HOOMD_DEVPTR(unsigned int) d_group_members;
    unsigned int nwork;
    HOOMD_DEVPTR(Scalar4) d_net_force;
    MANIFOLD_CLASS manifold;
    Scalar tolerance;
    Scalar deltaT;
    bool limit;
    Scalar limit_val;
    bool zero_force;
    };

struct gpu_include_rattle_force_nve_kernel_args
    {
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar4) d_vel;
    HOOMD_DEVPTR(Scalar3) d_accel;
    HOOMD_DEVPTR(Scalar4) d_net_force;
    HOOMD_DEVPTR(Scalar) d_net_virial;
    HOOMD_DEVPTR(unsigned int) d_group_members;
    unsigned int nwork;
    size_t net_virial_pitch;
    MANIFOLD_CLASS manifold;
    Scalar tolerance;
    Scalar deltaT;
    bool zero_force;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void gpu_rattle_bdtally_reduce_partial_sum_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_rattle_bdtally_reduce_partial_sum_kernel_args))
    {
    device Scalar* d_sum = args.d_sum;
    device Scalar* d_partial_sum = args.d_partial_sum;
    unsigned int num_blocks = args.num_blocks;

    Scalar sum = Scalar(0.0);
    threadgroup char* s_data = (threadgroup char*)hoomd_dynamic_shared;
    threadgroup Scalar* bdtally_sdata = (threadgroup Scalar*)&s_data[0];

    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_blocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_blocks)
            bdtally_sdata[threadIdx.x] = d_partial_sum[start + threadIdx.x];
        else
            bdtally_sdata[threadIdx.x] = Scalar(0.0);
        __syncthreads();

        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                bdtally_sdata[threadIdx.x] += bdtally_sdata[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }

        // everybody sums up sum2K
        sum += bdtally_sdata[0];
        }

    if (threadIdx.x == 0)
        *d_sum = sum;
    }

kernel void gpu_rattle_brownian_step_one_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_rattle_brownian_step_one_kernel_args))
    {
    device Scalar4* d_pos = args.d_pos;
    device int3* d_image = args.d_image;
    device Scalar4* d_vel = args.d_vel;
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
    const size_t n_types = args.n_types;
    const uint64_t timestep = args.timestep;
    const uint16_t seed = args.seed;
    const Scalar T = args.T;
    MANIFOLD_CLASS manifold = args.manifold;
    const bool aniso = args.aniso;
    const Scalar deltaT = args.deltaT;
    unsigned int D = args.D;
    const bool d_noiseless_t = args.d_noiseless_t;
    const bool d_noiseless_r = args.d_noiseless_r;

    HIP_DYNAMIC_SHARED(char, s_data)

    threadgroup Scalar3* s_gammas_r = (threadgroup Scalar3*)s_data;
    threadgroup Scalar* s_gammas = (threadgroup Scalar*)(s_gammas_r + n_types);

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

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_idx < nwork)
        {
        const unsigned int group_idx = local_idx;

        // determine the particle to work on
        unsigned int idx = d_group_members[group_idx];
        unsigned int tag = d_tag[idx];

        Scalar4 postype = d_pos[idx];
        Scalar4 vel = d_vel[idx];
        Scalar4 net_force = d_net_force[idx];
        int3 image = d_image[idx];

        // calculate the magnitude of the random force
        Scalar gamma;
        // determine gamma from type
        unsigned int typ = __scalar_as_int(postype.w);
        gamma = s_gammas[typ];

        Scalar deltaT_gamma = deltaT / gamma;

        // compute the random force
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),
                            hoomd::Counter(tag, 1));

        // compute the random force
        RandomGenerator rng_b(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),
                              hoomd::Counter(tag, 2));

        Scalar3 next_pos;
        next_pos.x = postype.x;
        next_pos.y = postype.y;
        next_pos.z = postype.z;
        Scalar3 normal = manifold.derivative(next_pos);

        if (d_noiseless_t)
            {
            vel.x = net_force.x / gamma;
            vel.y = net_force.y / gamma;
            vel.z = net_force.y / gamma;
            }
        else
            {
            // draw a new random velocity for particle j
            Scalar mass = vel.w;
            Scalar sigma = fast::sqrt(T / mass);
            NormalDistribution<Scalar> norm(sigma);
            vel.x = norm(rng);
            vel.y = norm(rng);
            vel.z = norm(rng);
            }

        Scalar norm_normal = fast::rsqrt(dot(normal, normal));
        normal.x *= norm_normal;
        normal.y *= norm_normal;
        normal.z *= norm_normal;

        Scalar rand_norm = vel.x * normal.x + vel.y * normal.y + vel.z * normal.z;
        vel.x -= rand_norm * normal.x;
        vel.y -= rand_norm * normal.y;
        vel.z -= rand_norm * normal.z;

        Scalar rx, ry, rz, coeff;

        if (T > 0)
            {
            UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
            rx = uniform(rng_b);
            ry = uniform(rng_b);
            rz = uniform(rng_b);

            Scalar normal_r = rx * normal.x + ry * normal.y + rz * normal.z;

            rx = rx - normal_r * normal.x;
            ry = ry - normal_r * normal.y;
            rz = rz - normal_r * normal.z;

            // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform
            // -1,1 distribution it is not the dimensionality of the system
            coeff = fast::sqrt(Scalar(6.0) * T / deltaT_gamma);
            if (d_noiseless_t)
                coeff = Scalar(0.0);
            }
        else
            {
            rx = 0;
            ry = 0;
            rz = 0;
            coeff = 0;
            }

        Scalar dx = (net_force.x + rx * coeff) * deltaT_gamma;
        Scalar dy = (net_force.y + ry * coeff) * deltaT_gamma;
        Scalar dz = (net_force.z + rz * coeff) * deltaT_gamma;

        postype.x += dx;
        postype.y += dy;
        postype.z += dz;
        // particles may have been moved slightly outside the box by the above steps, wrap them back
        // into place
        box.wrap(postype, vel, image);

        // write out data
        d_pos[idx] = postype;
        d_image[idx] = image;
        d_vel[idx] = vel;

        // rotational random force and orientation quaternion updates
        if (aniso)
            {
            unsigned int type_r = __scalar_as_int(d_pos[idx].w);

            // gamma_r is stored in the second half of s_gammas a.k.a s_gammas_r
            Scalar3 gamma_r = s_gammas_r[type_r];
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

                if (x_zero)
                    {
                    bf_torque.x = 0;
                    t.x = 0;
                    }
                if (y_zero)
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
                if (D < 3)
                    {
                    bf_torque.x = 0;
                    bf_torque.y = 0;
                    t.x = 0;
                    t.y = 0;
                    }

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

                if (x_zero)
                    p_vec.x = 0;
                if (y_zero)
                    p_vec.y = 0;
                if (z_zero)
                    p_vec.z = 0;

                // !! Note this ang_mom isn't well-behaving in 2D,
                // !! because may have effective non-zero ang_mom in x,y

                // store ang_mom quaternion
                quat<Scalar> p = Scalar(2.0) * q * p_vec;
                d_angmom[idx] = quat_to_scalar4(p);
                }
            }
        }
    }

kernel void gpu_include_rattle_force_bd_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_include_rattle_force_bd_kernel_args))
    {
    device const Scalar4* d_pos = args.d_pos;
    device Scalar4* d_net_force = args.d_net_force;
    device Scalar* d_net_virial = args.d_net_virial;
    device const unsigned int* d_tag = args.d_tag;
    device const unsigned int* d_group_members = args.d_group_members;
    const unsigned int nwork = args.nwork;
    device const Scalar* d_gamma = args.d_gamma;
    const size_t n_types = args.n_types;
    const uint64_t timestep = args.timestep;
    const uint16_t seed = args.seed;
    const Scalar T = args.T;
    const Scalar tolerance = args.tolerance;
    MANIFOLD_CLASS manifold = args.manifold;
    size_t net_virial_pitch = args.net_virial_pitch;
    const Scalar deltaT = args.deltaT;
    const bool d_noiseless_t = args.d_noiseless_t;

    HIP_DYNAMIC_SHARED(char, s_data2)

    threadgroup Scalar3* s_gammas_r = (threadgroup Scalar3*)s_data2;
    threadgroup Scalar* s_gammas = (threadgroup Scalar*)(s_gammas_r + n_types);

    // read in the gamma (1 dimensional array), stored in s_gammas[0: n_type] (Pythonic
    // convention)
    for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < n_types)
            s_gammas[cur_offset + threadIdx.x] = d_gamma[cur_offset + threadIdx.x];
        }
    __syncthreads();

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_idx < nwork)
        {
        const unsigned int group_idx = local_idx;

        // determine the particle to work on
        unsigned int idx = d_group_members[group_idx];
        unsigned int tag = d_tag[idx];

        Scalar4 postype = d_pos[idx];
        Scalar4 net_force = d_net_force[idx];
        Scalar3 brownian_force = make_scalar3(0, 0, 0);

        Scalar virial0 = d_net_virial[0 * net_virial_pitch + idx];
        Scalar virial1 = d_net_virial[1 * net_virial_pitch + idx];
        Scalar virial2 = d_net_virial[2 * net_virial_pitch + idx];
        Scalar virial3 = d_net_virial[3 * net_virial_pitch + idx];
        Scalar virial4 = d_net_virial[4 * net_virial_pitch + idx];
        Scalar virial5 = d_net_virial[5 * net_virial_pitch + idx];

        // calculate the magnitude of the random force
        Scalar gamma;
        // determine gamma from type
        unsigned int typ = __scalar_as_int(postype.w);
        gamma = s_gammas[typ];

        Scalar deltaT_gamma = deltaT / gamma;

        // compute the random force
        RandomGenerator rng_b(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),
                              hoomd::Counter(tag, 2));

        Scalar3 next_pos;
        next_pos.x = postype.x;
        next_pos.y = postype.y;
        next_pos.z = postype.z;

        Scalar3 normal = manifold.derivative(next_pos);
        Scalar norm_normal = fast::rsqrt(dot(normal, normal));
        normal.x *= norm_normal;
        normal.y *= norm_normal;
        normal.z *= norm_normal;

        Scalar rx, ry, rz, coeff;

        if (T > 0)
            {
            UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
            rx = uniform(rng_b);
            ry = uniform(rng_b);
            rz = uniform(rng_b);

            Scalar normal_r = rx * normal.x + ry * normal.y + rz * normal.z;

            rx = rx - normal_r * normal.x;
            ry = ry - normal_r * normal.y;
            rz = rz - normal_r * normal.z;

            // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform
            // -1,1 distribution it is not the dimensionality of the system
            coeff = fast::sqrt(Scalar(6.0) * T / deltaT_gamma);
            if (d_noiseless_t)
                coeff = Scalar(0.0);
            }
        else
            {
            rx = 0;
            ry = 0;
            rz = 0;
            coeff = 0;
            }

        brownian_force.x = rx * coeff;
        brownian_force.y = ry * coeff;
        brownian_force.z = rz * coeff;

        // update position

        Scalar mu = 0;

        Scalar inv_alpha = -deltaT_gamma;
        inv_alpha = Scalar(1.0) / inv_alpha;

        Scalar3 residual;
        Scalar resid;
        unsigned int iteration = 0;
        const unsigned int maxiteration = 10;

        do
            {
            iteration++;
            residual.x = postype.x - next_pos.x
                         + (net_force.x + brownian_force.x - mu * normal.x) * deltaT_gamma;
            residual.y = postype.y - next_pos.y
                         + (net_force.y + brownian_force.y - mu * normal.y) * deltaT_gamma;
            residual.z = postype.z - next_pos.z
                         + (net_force.z + brownian_force.z - mu * normal.z) * deltaT_gamma;
            resid = manifold.implicitFunction(next_pos);

            Scalar3 next_normal = manifold.derivative(next_pos);

            Scalar nndotr = dot(next_normal, residual);
            Scalar nndotn = dot(next_normal, normal);
            Scalar beta = (resid + nndotr) / nndotn;

            next_pos.x = next_pos.x - beta * normal.x + residual.x;
            next_pos.y = next_pos.y - beta * normal.y + residual.y;
            next_pos.z = next_pos.z - beta * normal.z + residual.z;
            mu = mu - beta * inv_alpha;

            resid = fabs(resid);
            Scalar vec_norm = sqrt(dot(residual, residual));
            if (vec_norm > resid)
                resid = vec_norm;

            } while (resid > tolerance && iteration < maxiteration);

        net_force.x -= mu * normal.x;
        net_force.y -= mu * normal.y;
        net_force.z -= mu * normal.z;

        virial0 -= mu * normal.x * postype.x;
        virial1 -= 0.5 * mu * (normal.x * postype.y + normal.y * postype.x);
        virial2 -= 0.5 * mu * (normal.x * postype.z + normal.z * postype.x);
        virial3 -= mu * normal.y * postype.y;
        virial4 -= 0.5 * mu * (normal.y * postype.z + normal.z * postype.y);
        virial5 -= mu * normal.z * postype.z;

        d_net_force[idx] = net_force;
        d_net_virial[0 * net_virial_pitch + idx] = virial0;
        d_net_virial[1 * net_virial_pitch + idx] = virial1;
        d_net_virial[2 * net_virial_pitch + idx] = virial2;
        d_net_virial[3 * net_virial_pitch + idx] = virial3;
        d_net_virial[4 * net_virial_pitch + idx] = virial4;
        d_net_virial[5 * net_virial_pitch + idx] = virial5;
        }
    }

kernel void gpu_rattle_langevin_step_two_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_rattle_langevin_step_two_kernel_args))
    {
    device const Scalar4* d_pos = args.d_pos;
    device Scalar4* d_vel = args.d_vel;
    device Scalar3* d_accel = args.d_accel;
    device const unsigned int* d_tag = args.d_tag;
    device unsigned int* d_group_members = args.d_group_members;
    unsigned int group_size = args.group_size;
    device Scalar4* d_net_force = args.d_net_force;
    device Scalar* d_gamma = args.d_gamma;
    size_t n_types = args.n_types;
    uint64_t timestep = args.timestep;
    uint16_t seed = args.seed;
    Scalar T = args.T;
    Scalar tolerance = args.tolerance;
    bool noiseless_t = args.noiseless_t;
    MANIFOLD_CLASS manifold = args.manifold;
    Scalar deltaT = args.deltaT;
    unsigned int D = args.D;
    bool tally = args.tally;
    device Scalar* d_partial_sum_bdenergy = args.d_partial_sum_bdenergy;

    HIP_DYNAMIC_SHARED(char, s_data)
    threadgroup Scalar* s_gammas = (threadgroup Scalar*)s_data;

    // read in the gammas (1 dimensional array)
    for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < n_types)
            s_gammas[cur_offset + threadIdx.x] = d_gamma[cur_offset + threadIdx.x];
        }
    __syncthreads();

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar bd_energy_transfer = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // ******** first, calculate the additional BD force
        // read the current particle velocity (MEM TRANSFER: 16 bytes)
        Scalar4 vel = d_vel[idx];
        // read in the tag of our particle.
        // (MEM TRANSFER: 4 bytes)
        unsigned int ptag = d_tag[idx];

        // calculate the magnitude of the random force
        Scalar gamma;
        // read in the type of our particle. A texture read of only the fourth part of the
        // position Scalar4 (where type is stored) is used.
        unsigned int typ = __scalar_as_int(d_pos[idx].w);
        gamma = s_gammas[typ];

        // read in the net force and calculate the acceleration MEM TRANSFER: 16 bytes
        Scalar4 net_force = d_net_force[idx];
        Scalar3 accel = d_accel[idx];

        Scalar3 pos = make_scalar3(d_pos[idx].x, d_pos[idx].y, d_pos[idx].z);

        // Initialize the Random Number Generator and generate the 3 random numbers
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepLangevin, timestep, seed),
                            hoomd::Counter(ptag));

        Scalar3 normal = manifold.derivative(pos);
        Scalar ndotn = dot(normal, normal);

        Scalar randomx, randomy, randomz, coeff;

        if (T > 0)
            {
            UniformDistribution<Scalar> uniform(-1, 1);

            randomx = uniform(rng);
            randomy = uniform(rng);
            randomz = uniform(rng);

            coeff = sqrtf(Scalar(6.0) * gamma * T / deltaT);
            Scalar3 bd_force = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
            if (noiseless_t)
                coeff = Scalar(0.0);

            Scalar proj_x = normal.x / fast::sqrt(ndotn);
            Scalar proj_y = normal.y / fast::sqrt(ndotn);
            Scalar proj_z = normal.z / fast::sqrt(ndotn);

            Scalar proj_r = randomx * proj_x + randomy * proj_y + randomz * proj_z;
            randomx = randomx - proj_r * proj_x;
            randomy = randomy - proj_r * proj_y;
            randomz = randomz - proj_r * proj_z;
            }
        else
            {
            randomx = 0;
            randomy = 0;
            randomz = 0;
            coeff = 0;
            }

        Scalar3 bd_force;

        bd_force.x = randomx * coeff - gamma * vel.x;
        bd_force.y = randomy * coeff - gamma * vel.y;
        bd_force.z = randomz * coeff - gamma * vel.z;

        // MEM TRANSFER: 4 bytes   FLOPS: 3
        Scalar mass = vel.w;
        Scalar minv = Scalar(1.0) / mass;
        accel.x = (net_force.x + bd_force.x) * minv;
        accel.y = (net_force.y + bd_force.y) * minv;
        accel.z = (net_force.z + bd_force.z) * minv;

        // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
        // update the velocity (FLOPS: 6)

        Scalar3 next_vel;
        next_vel.x = vel.x + Scalar(1.0 / 2.0) * deltaT * accel.x;
        next_vel.y = vel.y + Scalar(1.0 / 2.0) * deltaT * accel.y;
        next_vel.z = vel.z + Scalar(1.0 / 2.0) * deltaT * accel.z;

        Scalar mu = 0;
        Scalar inv_alpha = -Scalar(1.0 / 2.0) * deltaT;
        inv_alpha = Scalar(1.0) / inv_alpha;

        Scalar3 residual;
        Scalar resid;
        Scalar3 vel_dot;

        const unsigned int maxiteration = 10;
        unsigned int iteration = 0;
        do
            {
            iteration++;
            vel_dot.x = accel.x - mu * minv * normal.x;
            vel_dot.y = accel.y - mu * minv * normal.y;
            vel_dot.z = accel.z - mu * minv * normal.z;

            residual.x = vel.x - next_vel.x + Scalar(1.0 / 2.0) * deltaT * vel_dot.x;
            residual.y = vel.y - next_vel.y + Scalar(1.0 / 2.0) * deltaT * vel_dot.y;
            residual.z = vel.z - next_vel.z + Scalar(1.0 / 2.0) * deltaT * vel_dot.z;
            resid = dot(normal, next_vel) * minv;

            Scalar ndotr = dot(normal, residual);
            Scalar beta = (mass * resid + ndotr) / ndotn;
            next_vel.x = next_vel.x - normal.x * beta + residual.x;
            next_vel.y = next_vel.y - normal.y * beta + residual.y;
            next_vel.z = next_vel.z - normal.z * beta + residual.z;
            mu = mu - mass * beta * inv_alpha;

            resid = fabs(resid);
            Scalar vec_norm = sqrt(dot(residual, residual));
            if (vec_norm > resid)
                resid = vec_norm;

            } while (resid * mass > tolerance && iteration < maxiteration);

        vel.x += (Scalar(1.0) / Scalar(2.0)) * (accel.x - mu * minv * normal.x) * deltaT;
        vel.y += (Scalar(1.0) / Scalar(2.0)) * (accel.y - mu * minv * normal.y) * deltaT;
        vel.z += (Scalar(1.0) / Scalar(2.0)) * (accel.z - mu * minv * normal.z) * deltaT;

        // tally the energy transfer from the bd thermal reservoir to the particles (FLOPS: 6)
        bd_energy_transfer = bd_force.x * vel.x + bd_force.y * vel.y + bd_force.z * vel.z;

        // write out data (MEM TRANSFER: 32 bytes)
        d_vel[idx] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = accel;
        }

    threadgroup Scalar* bdtally_sdata = (threadgroup Scalar*)&s_data[0];
    if (tally)
        {
        // don't overwrite values in the s_gammas array with bd_energy transfer
        __syncthreads();
        bdtally_sdata[threadIdx.x] = bd_energy_transfer;
        __syncthreads();

        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                bdtally_sdata[threadIdx.x] += bdtally_sdata[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }

        // write out our partial sum
        if (threadIdx.x == 0)
            {
            d_partial_sum_bdenergy[blockIdx.x] = bdtally_sdata[0];
            }
        }
    }

kernel void gpu_rattle_nve_step_two_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_rattle_nve_step_two_kernel_args))
    {
    device Scalar4* d_pos = args.d_pos;
    device Scalar4* d_vel = args.d_vel;
    device Scalar3* d_accel = args.d_accel;
    device unsigned int* d_group_members = args.d_group_members;
    const unsigned int nwork = args.nwork;
    device Scalar4* d_net_force = args.d_net_force;
    MANIFOLD_CLASS manifold = args.manifold;
    Scalar tolerance = args.tolerance;
    Scalar deltaT = args.deltaT;
    bool limit = args.limit;
    Scalar limit_val = args.limit_val;
    bool zero_force = args.zero_force;

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx;
        unsigned int idx = d_group_members[group_idx];

        Scalar3 pos = make_scalar3(d_pos[idx].x, d_pos[idx].y, d_pos[idx].z);

        // read in the net forc and calculate the acceleration MEM TRANSFER: 16 bytes
        Scalar3 accel = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));

        // read the current particle velocity (MEM TRANSFER: 16 bytes)
        Scalar4 vel = d_vel[idx];

        if (!zero_force)
            {
            Scalar4 net_force = d_net_force[idx];
            accel = make_scalar3(net_force.x, net_force.y, net_force.z);
            // MEM TRANSFER: 4 bytes   FLOPS: 3
            Scalar mass = vel.w;
            accel.x /= mass;
            accel.y /= mass;
            accel.z /= mass;
            }

        // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT

        // update the velocity (FLOPS: 6)

        Scalar mu = 0;
        Scalar inv_alpha = -Scalar(1.0 / 2.0) * deltaT;
        inv_alpha = Scalar(1.0) / inv_alpha;
        Scalar mass = vel.w;
        Scalar inv_mass = Scalar(1.0) / mass;

        Scalar3 normal = manifold.derivative(pos);

        Scalar3 next_vel;
        next_vel.x = vel.x + Scalar(1.0 / 2.0) * deltaT * accel.x;
        next_vel.y = vel.y + Scalar(1.0 / 2.0) * deltaT * accel.y;
        next_vel.z = vel.z + Scalar(1.0 / 2.0) * deltaT * accel.z;

        Scalar3 residual;
        Scalar resid;
        Scalar3 vel_dot;

        const unsigned int maxiteration = 10;
        unsigned int iteration = 0;
        do
            {
            iteration++;
            vel_dot.x = accel.x - mu * inv_mass * normal.x;
            vel_dot.y = accel.y - mu * inv_mass * normal.y;
            vel_dot.z = accel.z - mu * inv_mass * normal.z;

            residual.x = vel.x - next_vel.x + Scalar(1.0 / 2.0) * deltaT * vel_dot.x;
            residual.y = vel.y - next_vel.y + Scalar(1.0 / 2.0) * deltaT * vel_dot.y;
            residual.z = vel.z - next_vel.z + Scalar(1.0 / 2.0) * deltaT * vel_dot.z;
            resid = dot(normal, next_vel) * inv_mass;

            Scalar ndotr = dot(normal, residual);
            Scalar ndotn = dot(normal, normal);
            Scalar beta = (mass * resid + ndotr) / ndotn;
            next_vel.x = next_vel.x - normal.x * beta + residual.x;
            next_vel.y = next_vel.y - normal.y * beta + residual.y;
            next_vel.z = next_vel.z - normal.z * beta + residual.z;
            mu = mu - mass * beta * inv_alpha;

            resid = fabs(resid);
            Scalar vec_norm = sqrt(dot(residual, residual));
            if (vec_norm > resid)
                resid = vec_norm;

            } while (resid * mass > tolerance && iteration < maxiteration);

        vel.x += (Scalar(1.0) / Scalar(2.0)) * (accel.x - mu * inv_mass * normal.x) * deltaT;
        vel.y += (Scalar(1.0) / Scalar(2.0)) * (accel.y - mu * inv_mass * normal.y) * deltaT;
        vel.z += (Scalar(1.0) / Scalar(2.0)) * (accel.z - mu * inv_mass * normal.z) * deltaT;

        if (limit)
            {
            Scalar vel_len = sqrtf(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
            if ((vel_len * deltaT) > limit_val)
                {
                vel.x = vel.x / vel_len * limit_val / deltaT;
                vel.y = vel.y / vel_len * limit_val / deltaT;
                vel.z = vel.z / vel_len * limit_val / deltaT;
                }
            }

        // write out data (MEM TRANSFER: 32 bytes)
        d_vel[idx] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = accel;
        }
    }

kernel void gpu_include_rattle_force_nve_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_include_rattle_force_nve_kernel_args))
    {
    device const Scalar4* d_pos = args.d_pos;
    device const Scalar4* d_vel = args.d_vel;
    device Scalar3* d_accel = args.d_accel;
    device Scalar4* d_net_force = args.d_net_force;
    device Scalar* d_net_virial = args.d_net_virial;
    device unsigned int* d_group_members = args.d_group_members;
    const unsigned int nwork = args.nwork;
    size_t net_virial_pitch = args.net_virial_pitch;
    MANIFOLD_CLASS manifold = args.manifold;
    Scalar tolerance = args.tolerance;
    Scalar deltaT = args.deltaT;
    bool zero_force = args.zero_force;

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx;
        unsigned int idx = d_group_members[group_idx];

        // do velocity verlet update
        // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
        // v(t+deltaT/2) = v(t) + (1/2)a*deltaT

        // read the particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 postype = d_pos[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

        Scalar3 normal
            = manifold.derivative(pos); // the normal vector to which the particles are confined.

        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        Scalar4 velmass = d_vel[idx];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);

        Scalar3 accel = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
        if (!zero_force)
            accel = d_accel[idx];

        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        Scalar4 forcetype = d_net_force[idx];
        Scalar3 force = make_scalar3(forcetype.x, forcetype.y, forcetype.z);

        Scalar virial0 = d_net_virial[0 * net_virial_pitch + idx];
        Scalar virial1 = d_net_virial[1 * net_virial_pitch + idx];
        Scalar virial2 = d_net_virial[2 * net_virial_pitch + idx];
        Scalar virial3 = d_net_virial[3 * net_virial_pitch + idx];
        Scalar virial4 = d_net_virial[4 * net_virial_pitch + idx];
        Scalar virial5 = d_net_virial[5 * net_virial_pitch + idx];

        Scalar lambda = 0.0;
        Scalar inv_mass = Scalar(1.0) / velmass.w;
        Scalar deltaT_half = Scalar(1.0 / 2.0) * deltaT;
        Scalar inv_alpha = -deltaT_half * deltaT * inv_mass;
        inv_alpha = Scalar(1.0) / inv_alpha;

        Scalar3 next_pos = pos;
        Scalar3 residual;
        Scalar resid;
        Scalar3 half_vel;

        const unsigned int maxiteration = 10;
        unsigned int iteration = 0;
        do
            {
            iteration++;
            half_vel = vel + deltaT_half * accel - deltaT_half * inv_mass * lambda * normal;

            residual = pos - next_pos + deltaT * half_vel;
            resid = manifold.implicitFunction(next_pos);

            Scalar3 next_normal = manifold.derivative(next_pos);
            Scalar nndotr = dot(next_normal, residual);
            Scalar nndotn = dot(next_normal, normal);
            Scalar beta = (resid + nndotr) / nndotn;

            next_pos = next_pos - beta * normal + residual;
            lambda = lambda - beta * inv_alpha;

            resid = fabs(resid);
            Scalar vec_norm = sqrt(dot(residual, residual));
            if (vec_norm > resid)
                resid = vec_norm;

            } while (resid > tolerance && iteration < maxiteration);

        force = force - lambda * normal;

        accel = accel - inv_mass * lambda * normal;

        virial0 -= lambda * normal.x * pos.x;
        virial1 -= 0.5 * lambda * (normal.x * pos.y + normal.y * pos.x);
        virial2 -= 0.5 * lambda * (normal.x * pos.z + normal.z * pos.x);
        virial3 -= lambda * normal.y * pos.y;
        virial4 -= 0.5 * lambda * (normal.y * pos.z + normal.z * pos.y);
        virial5 -= lambda * normal.z * pos.z;

        d_net_force[idx] = make_scalar4(force.x, force.y, force.z, forcetype.w);
        d_accel[idx] = accel;
        d_net_virial[0 * net_virial_pitch + idx] = virial0;
        d_net_virial[1 * net_virial_pitch + idx] = virial1;
        d_net_virial[2 * net_virial_pitch + idx] = virial2;
        d_net_virial[3 * net_virial_pitch + idx] = virial3;
        d_net_virial[4 * net_virial_pitch + idx] = virial4;
        d_net_virial[5 * net_virial_pitch + idx] = virial5;
        }
    }

#else

namespace kernel
    {






template<>
__attribute__((visibility("default"))) hipError_t gpu_rattle_brownian_step_one<MANIFOLD_CLASS>(Scalar4* d_pos,
                                        int3* d_image,
                                        Scalar4* d_vel,
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
                                        const rattle_bd_step_one_args& rattle_bd_args,
                                        MANIFOLD_CLASS manifold,
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

    const auto shared_bytes
        = (sizeof(Scalar) * rattle_bd_args.n_types + sizeof(Scalar3) * rattle_bd_args.n_types);

    if (shared_bytes > rattle_bd_args.devprop.sharedMemPerBlock)
        {
        throw std::runtime_error("Brownian gamma parameters exceed the available shared "
                                 "memory per block.");
        }

    // run the kernel
    {
        kernel_args::gpu_rattle_brownian_step_one_kernel_args _args;
        _args.d_pos = d_pos;
        _args.d_image = d_image;
        _args.d_vel = d_vel;
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
        _args.d_gamma = rattle_bd_args.d_gamma;
        _args.n_types = rattle_bd_args.n_types;
        _args.timestep = rattle_bd_args.timestep;
        _args.seed = rattle_bd_args.seed;
        _args.T = rattle_bd_args.T;
        _args.manifold = manifold;
        _args.aniso = aniso;
        _args.deltaT = deltaT;
        _args.D = D;
        _args.d_noiseless_t = d_noiseless_t;
        _args.d_noiseless_r = d_noiseless_r;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_rattle_brownian_step_one_kernel",
                           dim3(grid),
                           dim3(threads),
                           _args,
                           shared_bytes);
        }

    return hipSuccess;
    }



template<>
__attribute__((visibility("default"))) hipError_t gpu_include_rattle_force_bd<MANIFOLD_CLASS>(const Scalar4* d_pos,
                                       Scalar4* d_net_force,
                                       Scalar* d_net_virial,
                                       const unsigned int* d_tag,
                                       const unsigned int* d_group_members,
                                       const unsigned int group_size,
                                       const rattle_bd_step_one_args& rattle_bd_args,
                                       MANIFOLD_CLASS manifold,
                                       size_t net_virial_pitch,
                                       const Scalar deltaT,
                                       const bool d_noiseless_t)
    {
    unsigned int run_block_size = 256;

    unsigned int nwork = group_size;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    const auto shared_bytes
        = (sizeof(Scalar) * rattle_bd_args.n_types + sizeof(Scalar3) * rattle_bd_args.n_types);

    if (shared_bytes > rattle_bd_args.devprop.sharedMemPerBlock)
        {
        throw std::runtime_error("Brownian gamma parameters exceed the available shared "
                                 "memory per block.");
        }

    // run the kernel
    {
        kernel_args::gpu_include_rattle_force_bd_kernel_args _args;
        _args.d_pos = d_pos;
        _args.d_net_force = d_net_force;
        _args.d_net_virial = d_net_virial;
        _args.d_tag = d_tag;
        _args.d_group_members = d_group_members;
        _args.nwork = nwork;
        _args.d_gamma = rattle_bd_args.d_gamma;
        _args.n_types = rattle_bd_args.n_types;
        _args.timestep = rattle_bd_args.timestep;
        _args.seed = rattle_bd_args.seed;
        _args.T = rattle_bd_args.T;
        _args.tolerance = rattle_bd_args.tolerance;
        _args.manifold = manifold;
        _args.net_virial_pitch = net_virial_pitch;
        _args.deltaT = deltaT;
        _args.d_noiseless_t = d_noiseless_t;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_include_rattle_force_bd_kernel",
                           dim3(grid),
                           dim3(threads),
                           _args,
                           shared_bytes);
        }

    return hipSuccess;
    }




/*! \file TwoStepRATTLELangevinGPU.cu
    \brief Defines GPU kernel code for RATTLELangevin integration on the GPU. Used by
   TwoStepRATTLELangevinGPU.
*/

//! Takes the second half-step forward in the RATTLELangevin integration on a group of particles
//! with
/*! \param d_pos array of particle positions and types
    \param d_vel array of particle positions and masses
    \param d_accel array of particle accelerations
    \param d_tag array of particle tags
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param d_gamma List of per-type gammas
    \param n_types Number of particle types in the simulation
    \param timestep Current timestep of the simulation
    \param seed User chosen random number seed
    \param T Temperature set point
    \param deltaT Amount of real time to step forward in one time step
    \param D Dimensionality of the system
    \param tally Boolean indicating whether energy tally is performed or not
    \param d_partial_sum_bdenergy Placeholder for the partial sum

    This kernel is implemented in a very similar manner to gpu_nve_step_two_kernel(), see it for
   design details.

    This kernel will tally the energy transfer from the bd thermal reservoir and the particle system

    This kernel must be launched with enough dynamic shared memory per block to read in d_gamma
*/



/*! \param d_pos array of particle positions and types
    \param d_vel array of particle positions and masses
    \param d_accel array of particle accelerations
    \param d_tag array of particle tags
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param rattle_langevin_args Collected arguments for gpu_rattle_langevin_step_two_kernel() and
   gpu_rattle_langevin_angular_step_two() \param deltaT Amount of real time to step forward in one
   time step \param D Dimensionality of the system

    This is just a driver for gpu_rattle_langevin_step_two_kernel(), see it for details.
*/
template<>
__attribute__((visibility("default"))) hipError_t gpu_rattle_langevin_step_two<MANIFOLD_CLASS>(const Scalar4* d_pos,
                                        Scalar4* d_vel,
                                        Scalar3* d_accel,
                                        const unsigned int* d_tag,
                                        unsigned int* d_group_members,
                                        unsigned int group_size,
                                        Scalar4* d_net_force,
                                        const rattle_langevin_step_two_args& rattle_langevin_args,
                                        MANIFOLD_CLASS manifold,
                                        Scalar deltaT,
                                        unsigned int D)
    {
    // setup the grid to run the kernel
    dim3 grid(rattle_langevin_args.num_blocks, 1, 1);
    dim3 grid1(1, 1, 1);
    dim3 threads(rattle_langevin_args.block_size, 1, 1);
    dim3 threads1(256, 1, 1);

    size_t shared_bytes = max((unsigned int)(sizeof(Scalar) * rattle_langevin_args.n_types),
                              (unsigned int)(rattle_langevin_args.block_size * sizeof(Scalar)));

    if (shared_bytes > rattle_langevin_args.devprop.sharedMemPerBlock)
        {
        throw std::runtime_error("Langevin gamma parameters exceed the available shared "
                                 "memory per block.");
        }

    // run the kernel
    {
        kernel_args::gpu_rattle_langevin_step_two_kernel_args _args;
        _args.d_pos = d_pos;
        _args.d_vel = d_vel;
        _args.d_accel = d_accel;
        _args.d_tag = d_tag;
        _args.d_group_members = d_group_members;
        _args.group_size = group_size;
        _args.d_net_force = d_net_force;
        _args.d_gamma = rattle_langevin_args.d_gamma;
        _args.n_types = rattle_langevin_args.n_types;
        _args.timestep = rattle_langevin_args.timestep;
        _args.seed = rattle_langevin_args.seed;
        _args.T = rattle_langevin_args.T;
        _args.tolerance = rattle_langevin_args.tolerance;
        _args.noiseless_t = rattle_langevin_args.noiseless_t;
        _args.manifold = manifold;
        _args.deltaT = deltaT;
        _args.D = D;
        _args.tally = rattle_langevin_args.tally;
        _args.d_partial_sum_bdenergy = rattle_langevin_args.d_partial_sum_bdenergy;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_rattle_langevin_step_two_kernel",
                           grid,
                           threads,
                           _args,
                           shared_bytes);
        }

    // run the summation kernel
    if (rattle_langevin_args.tally)
        {
            kernel_args::gpu_rattle_bdtally_reduce_partial_sum_kernel_args _args;
            _args.d_sum = &rattle_langevin_args.d_sum_bdenergy[0];
            _args.d_partial_sum = rattle_langevin_args.d_partial_sum_bdenergy;
            _args.num_blocks = rattle_langevin_args.num_blocks;
            hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                               "hoomd::md::gpu_rattle_bdtally_reduce_partial_sum_kernel",
                               dim3(grid1),
                               dim3(threads1),
                               _args,
                               rattle_langevin_args.block_size * sizeof(Scalar));
            }

    return hipSuccess;
    }




/*! \file TwoStepNVEGPU.cu
    \brief Defines GPU kernel code for NVE integration on the GPU. Used by TwoStepNVEGPU.
*/

//! Takes the second half-step forward in the velocity-verlet NVE integration on a group of
//! particles
/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param deltaT Amount of real time to step forward in one time step
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not
   move a distance further than \a limit_val in one step. \param limit_val Length to limit particle
   distance movement to \param zero_force Set to true to always assign an acceleration of 0 to all
   particles in the group This kernel is implemented in a very similar manner to
   gpu_rattle_nve_step_one_kernel(), see it for design details.
*/


/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param deltaT Amount of real time to step forward in one time step
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not
   move a distance further than \a limit_val in one step. \param limit_val Length to limit particle
   distance movement to \param zero_force Set to true to always assign an acceleration of 0 to all
   particles in the group This is just a driver for gpu_rattle_nve_step_two_kernel(), see it for
   details.
*/
template<>
__attribute__((visibility("default"))) hipError_t gpu_rattle_nve_step_two<MANIFOLD_CLASS>(Scalar4* d_pos,
                                   Scalar4* d_vel,
                                   Scalar3* d_accel,
                                   unsigned int* d_group_members,
                                   const unsigned int group_size,
                                   Scalar4* d_net_force,
                                   MANIFOLD_CLASS manifold,
                                   Scalar tolerance,
                                   Scalar deltaT,
                                   bool limit,
                                   Scalar limit_val,
                                   bool zero_force,
                                   unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_rattle_nve_step_two_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    unsigned int nwork = group_size;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    {
        kernel_args::gpu_rattle_nve_step_two_kernel_args _args;
        _args.d_pos = d_pos;
        _args.d_vel = d_vel;
        _args.d_accel = d_accel;
        _args.d_group_members = d_group_members;
        _args.nwork = nwork;
        _args.d_net_force = d_net_force;
        _args.manifold = manifold;
        _args.tolerance = tolerance;
        _args.deltaT = deltaT;
        _args.limit = limit;
        _args.limit_val = limit_val;
        _args.zero_force = zero_force;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_rattle_nve_step_two_kernel",
                           dim3(grid),
                           dim3(threads),
                           _args,
                           0);
        }

    return hipSuccess;
    }



template<>
__attribute__((visibility("default"))) hipError_t gpu_include_rattle_force_nve<MANIFOLD_CLASS>(const Scalar4* d_pos,
                                        const Scalar4* d_vel,
                                        Scalar3* d_accel,
                                        Scalar4* d_net_force,
                                        Scalar* d_net_virial,
                                        unsigned int* d_group_members,
                                        const unsigned int group_size,
                                        size_t net_virial_pitch,
                                        MANIFOLD_CLASS manifold,
                                        Scalar tolerance,
                                        Scalar deltaT,
                                        bool zero_force,
                                        unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_include_rattle_force_nve_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    unsigned int nwork = group_size;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    {
        kernel_args::gpu_include_rattle_force_nve_kernel_args _args;
        _args.d_pos = d_pos;
        _args.d_vel = d_vel;
        _args.d_accel = d_accel;
        _args.d_net_force = d_net_force;
        _args.d_net_virial = d_net_virial;
        _args.d_group_members = d_group_members;
        _args.nwork = nwork;
        _args.net_virial_pitch = net_virial_pitch;
        _args.manifold = manifold;
        _args.tolerance = tolerance;
        _args.deltaT = deltaT;
        _args.zero_force = zero_force;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_include_rattle_force_nve_kernel",
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
