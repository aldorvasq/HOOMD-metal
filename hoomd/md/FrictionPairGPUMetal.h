// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file FrictionPairGPUMetal.h
    \brief Metal port of the friction pair force kernel in FrictionPairGPU.cuh

    A .metal file includes this header and defines one kernel per evaluator with
    HOOMD_FRICTION_PAIR_FORCE_KERNEL(name, evaluator). The host driver finds the kernel through
    friction_kernel_name<evaluator>, which the .metal file specializes. The CUDA template
    parameters of the kernel are Metal function constants: 0: compute_virial, 1: threads per
    particle. The per type pair parameters are read from device memory into thread-local copies.
*/

#pragma once

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/VectorMath.h"

#ifdef __METAL_VERSION__
#include "hoomd/mtl/WarpTools.h"
#else
#include "FrictionPairGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
template<class evaluator> struct friction_pair_forces_args
    {
    HOOMD_DEVPTR(Scalar4) d_force;
    HOOMD_DEVPTR(Scalar4) d_torque;
    HOOMD_DEVPTR(Scalar) d_virial;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar4) d_vel;
    HOOMD_DEVPTR(const Scalar) d_charge;
    HOOMD_DEVPTR(const Scalar4) d_orientation;
    HOOMD_DEVPTR(const Scalar4) d_angmom;
    HOOMD_DEVPTR(const Scalar) d_diameter;
    HOOMD_DEVPTR(const Scalar3) d_moment_inertia;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(const unsigned int) d_n_neigh;
    HOOMD_DEVPTR(const unsigned int) d_nlist;
    HOOMD_DEVPTR(const size_t) d_head_list;
    HOOMD_DEVPTR(const typename evaluator::param_type) d_params;
    HOOMD_DEVPTR(const Scalar) d_rcutsq;
    BoxDim box;
    uint64_t timestep;
    Scalar deltaT;
    unsigned int virial_pitch;
    unsigned int N;
    unsigned int ntypes;
    unsigned int dim;
    unsigned int third_law;
    unsigned int seed;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant bool friction_compute_virial [[function_constant(0)]];
constant uint friction_tpp [[function_constant(1)]];

/*! Compute the friction pair forces and torques on all N particles. Each group of friction_tpp
    threads computes the force on one particle. See gpu_compute_pair_friction_forces_kernel in
    FrictionPairGPU.cuh.
*/
template<class evaluator>
inline void gpu_compute_pair_friction_forces_impl(
    constant kernel_args::friction_pair_forces_args<evaluator>& args,
    metal::vec<uint, 3> threadIdx,
    metal::vec<uint, 3> blockIdx,
    metal::vec<uint, 3> blockDim,
    ushort lane)
    {
    typedef typename evaluator::param_type param_type;

    const unsigned int tpp = friction_tpp;
    Index2D typpair_idx(args.ntypes);
    BoxDim box = args.box;

    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;

    // need to mask this thread, but still participate in warp-level reduction
    bool active = idx < args.N;

    // initialize the force to 0
    Scalar4 force = make_scalar4(Scalar(0), Scalar(0), Scalar(0), Scalar(0));
    Scalar4 torque = make_scalar4(Scalar(0), Scalar(0), Scalar(0), Scalar(0));
    Scalar virialxx = Scalar(0);
    Scalar virialxy = Scalar(0);
    Scalar virialxz = Scalar(0);
    Scalar virialyy = Scalar(0);
    Scalar virialyz = Scalar(0);
    Scalar virialzz = Scalar(0);

    if (active)
        {
        // load in the length of the neighbor list
        unsigned int n_neigh = args.d_n_neigh[idx];

        // read in the particle data
        Scalar4 postypei = args.d_pos[idx];
        Scalar4 veltypei = args.d_vel[idx];
        Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);
        Scalar3 veli = make_scalar3(veltypei.x, veltypei.y, veltypei.z);
        quat<Scalar> quati(args.d_orientation[idx]);
        quat<Scalar> angmomi(args.d_angmom[idx]);
        Scalar diai = args.d_diameter[idx];
        Scalar3 moment_inertia_i = args.d_moment_inertia[idx];

        Scalar qi = Scalar(0);
        Scalar massi = Scalar(0);
        if (evaluator::needsCharge())
            qi = args.d_charge[idx];
        if (evaluator::needsNu())
            massi = veltypei.w;

        // calculate angular momentum of i-th particle in the body frame
        quat<Scalar> qxP_i = conj(quati) * angmomi;
        vec3<Scalar> bf_vel_i = Scalar(0.5)
                                * vec3<Scalar>(qxP_i.v.x / moment_inertia_i.z,
                                               qxP_i.v.y / moment_inertia_i.z,
                                               qxP_i.v.z / moment_inertia_i.z);

        // Rotate angular velocity into global frame
        vec3<Scalar> gf_angvel_i = rotate(quati, bf_vel_i);
        Scalar3 angvel_i = make_scalar3(gf_angvel_i.x, gf_angvel_i.y, gf_angvel_i.z);

        size_t my_head = args.d_head_list[idx];
        unsigned int tagi = args.d_tag[idx];

        // loop over neighbors
        for (unsigned int neigh_idx = threadIdx.x % tpp; neigh_idx < n_neigh; neigh_idx += tpp)
            {
            unsigned int cur_j = args.d_nlist[my_head + neigh_idx];

            // get the neighbor's particle data
            Scalar4 postypej = args.d_pos[cur_j];
            Scalar4 veltypej = args.d_vel[cur_j];
            Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);
            Scalar3 velj = make_scalar3(veltypej.x, veltypej.y, veltypej.z);

            quat<Scalar> quatj(args.d_orientation[cur_j]);
            quat<Scalar> angmomj(args.d_angmom[cur_j]);
            Scalar diaj = args.d_diameter[cur_j];
            Scalar3 moment_inertia_j = args.d_moment_inertia[cur_j];

            Scalar qj = Scalar(0);
            Scalar massj = Scalar(0);
            if (evaluator::needsCharge())
                qj = args.d_charge[cur_j];
            if (evaluator::needsNu())
                massj = veltypej.w;

            // calculate dv_ij
            Scalar3 dv = velj - veli;

            // calculate angular momentum of j-th particle in the body frame
            quat<Scalar> qxP_j = conj(quatj) * angmomj;
            vec3<Scalar> bf_vel_j = Scalar(0.5)
                                    * vec3<Scalar>(qxP_j.v.x / moment_inertia_j.z,
                                                   qxP_j.v.y / moment_inertia_j.z,
                                                   qxP_j.v.z / moment_inertia_j.z);

            // Rotate angular velocity into global frame
            vec3<Scalar> gf_angvel_j = rotate(quatj, bf_vel_j);
            Scalar3 angvel_j = make_scalar3(gf_angvel_j.x, gf_angvel_j.y, gf_angvel_j.z);

            // calculate dr (with periodic boundary conditions)
            Scalar3 dx = posi - posj;
            dx = box.minImage(dx);

            // access the per type pair parameters
            unsigned int typpair
                = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypej.w));
            Scalar rcutsq = args.d_rcutsq[typpair];
            const param_type param = args.d_params[typpair];

            // evaluate the potential
            Scalar3 jforce = make_scalar3(Scalar(0), Scalar(0), Scalar(0));
            Scalar3 torquei = make_scalar3(Scalar(0), Scalar(0), Scalar(0));
            Scalar3 torquej = make_scalar3(Scalar(0), Scalar(0), Scalar(0));
            Scalar pair_eng = Scalar(0);

            evaluator eval(dx, angvel_i, angvel_j, dv, diai, diaj, rcutsq, param);

            // set seed using global tags
            unsigned int tagj = args.d_tag[cur_j];
            eval.set_seed_ij_timestep(uint16_t(args.seed), tagi, tagj, args.timestep);
            eval.setDeltaT(args.deltaT);
            eval.setThirdLaw(args.third_law != 0);

            if (evaluator::needsCharge())
                eval.setCharge(qi, qj);
            if (evaluator::needsTags())
                eval.setTags(tagi, tagj);

            if (evaluator::needsNu())
                {
                // Calculate nu for the Ito formalism
                Scalar nu_ito = ((Scalar(1.0) / massi) + (Scalar(1.0) / massj))
                                + (((diaj * diaj / Scalar(4.0)) / moment_inertia_j.z)
                                   + ((diai * diai / Scalar(4.0)) / moment_inertia_i.z));
                eval.setNu(nu_ito);
                }

            eval.evaluate(jforce, pair_eng, torquei, torquej);

            if (args.dim == 2)
                {
                jforce.z = 0.0;
                torquei.x = 0.0;
                torquei.y = 0.0;
                torquej.x = 0.0;
                torquej.y = 0.0;
                }

            // calculate the virial
            if (friction_compute_virial)
                {
                Scalar3 jforce2 = Scalar(0.5) * jforce;
                virialxx += dx.x * jforce2.x;
                virialxy += dx.y * jforce2.x;
                virialxz += dx.z * jforce2.x;
                virialyy += dx.y * jforce2.y;
                virialyz += dx.z * jforce2.y;
                virialzz += dx.z * jforce2.z;
                }

            // add up the force vector components
            force.x += jforce.x;
            force.y += jforce.y;
            force.z += jforce.z;
            torque.x += torquei.x;
            torque.y += torquei.y;
            torque.z += torquei.z;

            force.w += pair_eng;
            }

        // potential energy per particle must be halved
        force.w *= Scalar(0.5);
        }

    // reduce force over threads in the logical warp
    force.x = detail::warp_reduce_sum(force.x, lane, tpp);
    force.y = detail::warp_reduce_sum(force.y, lane, tpp);
    force.z = detail::warp_reduce_sum(force.z, lane, tpp);
    force.w = detail::warp_reduce_sum(force.w, lane, tpp);

    torque.x = detail::warp_reduce_sum(torque.x, lane, tpp);
    torque.y = detail::warp_reduce_sum(torque.y, lane, tpp);
    torque.z = detail::warp_reduce_sum(torque.z, lane, tpp);

    // now that the force calculation is complete, write out the result
    if (active && threadIdx.x % tpp == 0)
        {
        args.d_force[idx] = force;
        args.d_torque[idx] = torque;
        }

    if (friction_compute_virial)
        {
        virialxx = detail::warp_reduce_sum(virialxx, lane, tpp);
        virialxy = detail::warp_reduce_sum(virialxy, lane, tpp);
        virialxz = detail::warp_reduce_sum(virialxz, lane, tpp);
        virialyy = detail::warp_reduce_sum(virialyy, lane, tpp);
        virialyz = detail::warp_reduce_sum(virialyz, lane, tpp);
        virialzz = detail::warp_reduce_sum(virialzz, lane, tpp);

        if (active && threadIdx.x % tpp == 0)
            {
            const unsigned int virial_pitch = args.virial_pitch;
            args.d_virial[0 * virial_pitch + idx] = virialxx;
            args.d_virial[1 * virial_pitch + idx] = virialxy;
            args.d_virial[2 * virial_pitch + idx] = virialxz;
            args.d_virial[3 * virial_pitch + idx] = virialyy;
            args.d_virial[4 * virial_pitch + idx] = virialyz;
            args.d_virial[5 * virial_pitch + idx] = virialzz;
            }
        }
    }

//! Define the kernel entry point for an evaluator
#define HOOMD_FRICTION_PAIR_FORCE_KERNEL(NAME, ...)                                            \
    kernel void NAME(HOOMD_KERNEL_PARAMS(kernel_args::friction_pair_forces_args<__VA_ARGS__>)) \
        {                                                                                      \
        gpu_compute_pair_friction_forces_impl<__VA_ARGS__>(args,                               \
                                                           threadIdx,                          \
                                                           blockIdx,                           \
                                                           blockDim,                           \
                                                           hoomd_lane);                        \
        }

#else

namespace kernel
    {
//! Name of the Metal kernel for an evaluator (specialized next to the kernel definitions)
template<class evaluator> struct friction_kernel_name;

template<class evaluator>
hipError_t gpu_compute_pair_friction_forces(const a_pair_args_t& pair_args,
                                            const typename evaluator::param_type* d_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.ntypes > 0);

    const unsigned int tpp = pair_args.threads_per_particle;
    std::initializer_list<mtl::FunctionConstant> constants
        = {mtl::FunctionConstant::make_bool(0, pair_args.compute_virial != 0),
           mtl::FunctionConstant::make_uint(1, tpp)};

    const char* name = friction_kernel_name<evaluator>::value;
    unsigned int max_block_size
        = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name, constants);
    max_block_size -= max_block_size % gpu_friction_pair_force_max_tpp;
    unsigned int block_size = std::min(pair_args.block_size, max_block_size);

    kernel_args::friction_pair_forces_args<evaluator> args;
    args.d_force = pair_args.d_force;
    args.d_torque = pair_args.d_torque;
    args.d_virial = pair_args.d_virial;
    args.d_pos = pair_args.d_pos;
    args.d_vel = pair_args.d_vel;
    args.d_charge = pair_args.d_charge;
    args.d_orientation = pair_args.d_orientation;
    args.d_angmom = pair_args.d_angmom;
    args.d_diameter = pair_args.d_diameter;
    args.d_moment_inertia = pair_args.d_moment_inertia;
    args.d_tag = pair_args.d_tag;
    args.d_n_neigh = pair_args.d_n_neigh;
    args.d_nlist = pair_args.d_nlist;
    args.d_head_list = pair_args.d_head_list;
    args.d_params = d_params;
    args.d_rcutsq = pair_args.d_rcutsq;
    args.box = pair_args.box;
    args.timestep = pair_args.timestep;
    args.deltaT = pair_args.deltaT;
    args.virial_pitch = static_cast<unsigned int>(pair_args.virial_pitch);
    args.N = pair_args.N;
    args.ntypes = pair_args.ntypes;
    args.dim = pair_args.dim;
    args.third_law = pair_args.third_law ? 1 : 0;
    args.seed = pair_args.seed;

    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(pair_args.N / (block_size / tpp) + 1),
                       dim3(block_size),
                       args,
                       0,
                       constants);
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
