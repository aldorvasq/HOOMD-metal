// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file AnisoPotentialPairGPUMetal.h
    \brief Metal port of the anisotropic pair force kernel in AnisoPotentialPairGPU.cuh

    Each anisotropic pair potential has a .metal file that includes this header with
    EVALUATOR_CLASS defined. The CUDA template parameters of the kernel are Metal function
    constants: 0: shift_mode, 1: compute_virial, 2: threads per particle.

    The CUDA kernel stages the per type pair parameters, the per type shapes, and the arrays they
    reference (load_shared) in shared memory. Here, the parameters and shapes are read from device
    memory into thread-local copies, and ManagedArray data stays in device memory, so evaluators
    see thread-address-space parameters and shapes.
*/

#pragma once

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifdef __METAL_VERSION__
#include "hoomd/mtl/WarpTools.h"
#else
#include "AnisoPotentialPairGPU.cuh"

#include <sstream>
#include <stdexcept>
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
template<class evaluator> struct aniso_pair_forces_args
    {
    HOOMD_DEVPTR(Scalar4) d_force;
    HOOMD_DEVPTR(Scalar4) d_torque;
    HOOMD_DEVPTR(Scalar) d_virial;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar) d_charge;
    HOOMD_DEVPTR(const Scalar4) d_orientation;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(const unsigned int) d_n_neigh;
    HOOMD_DEVPTR(const unsigned int) d_nlist;
    HOOMD_DEVPTR(const size_t) d_head_list;
    HOOMD_DEVPTR(const typename evaluator::param_type) d_params;
    HOOMD_DEVPTR(const typename evaluator::shape_type) d_shape_params;
    HOOMD_DEVPTR(const Scalar) d_rcutsq;
    BoxDim box;
    unsigned int virial_pitch;
    unsigned int N;
    unsigned int ntypes;
    };

//! Sizes of the types shared with the host, written by the layout check kernel
struct aniso_pair_layout_args
    {
    HOOMD_DEVPTR(unsigned int) d_out;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant uint aniso_shift_mode [[function_constant(0)]];
constant bool aniso_compute_virial [[function_constant(1)]];
constant uint aniso_tpp [[function_constant(2)]];

/*! Compute the anisotropic pair forces and torques on all N particles. Each group of aniso_tpp
    threads computes the force on one particle. See gpu_compute_pair_aniso_forces_kernel in
    AnisoPotentialPairGPU.cuh.
*/
template<class evaluator>
inline void
gpu_compute_pair_aniso_forces_impl(constant kernel_args::aniso_pair_forces_args<evaluator>& args,
                                   metal::vec<uint, 3> threadIdx,
                                   metal::vec<uint, 3> blockIdx,
                                   metal::vec<uint, 3> blockDim,
                                   ushort lane)
    {
    typedef typename evaluator::param_type param_type;
    typedef typename evaluator::shape_type shape_type;

    const unsigned int tpp = aniso_tpp;
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

        // read in the position of our particle
        Scalar4 postypei = args.d_pos[idx];
        Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);
        Scalar4 quati = args.d_orientation[idx];
        unsigned int typei = __scalar_as_int(postypei.w);

        Scalar qi = Scalar(0);
        if (evaluator::needsCharge())
            qi = args.d_charge[idx];

        shape_type shape_i;
        if (evaluator::needsShape())
            shape_i = args.d_shape_params[typei];

        size_t my_head = args.d_head_list[idx];

        // loop over neighbors
        for (unsigned int neigh_idx = threadIdx.x % tpp; neigh_idx < n_neigh; neigh_idx += tpp)
            {
            unsigned int cur_j = args.d_nlist[my_head + neigh_idx];

            // get the neighbor's position
            Scalar4 postypej = args.d_pos[cur_j];
            Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);
            Scalar4 quatj = args.d_orientation[cur_j];
            unsigned int typej = __scalar_as_int(postypej.w);

            Scalar qj = Scalar(0);
            if (evaluator::needsCharge())
                qj = args.d_charge[cur_j];

            // calculate dr (with periodic boundary conditions)
            Scalar3 dx = posi - posj;
            dx = box.minImage(dx);

            // access the per type pair parameters
            unsigned int typpair = typpair_idx(typei, typej);
            Scalar rcutsq = args.d_rcutsq[typpair];
            const param_type param = args.d_params[typpair];

            // design specifies that energies are shifted if shift mode is set to shift
            bool energy_shift = (aniso_shift_mode == 1);

            // evaluate the potential
            Scalar3 jforce = make_scalar3(Scalar(0), Scalar(0), Scalar(0));
            Scalar3 torquei = make_scalar3(Scalar(0), Scalar(0), Scalar(0));
            Scalar3 torquej = make_scalar3(Scalar(0), Scalar(0), Scalar(0));
            Scalar pair_eng = Scalar(0);

            evaluator eval(dx, quati, quatj, rcutsq, param);
            if (evaluator::needsCharge())
                eval.setCharge(qi, qj);
            shape_type shape_j;
            if (evaluator::needsShape())
                {
                shape_j = args.d_shape_params[typej];
                eval.setShape(&shape_i, &shape_j);
                }
            if (evaluator::needsTags())
                eval.setTags(args.d_tag[idx], args.d_tag[cur_j]);

            eval.evaluate(jforce, pair_eng, energy_shift, torquei, torquej);

            // calculate the virial
            if (aniso_compute_virial)
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

    if (aniso_compute_virial)
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

//! Define the kernel entry points for an evaluator
#define HOOMD_ANISO_PAIR_FORCE_KERNELS(...)                                                    \
    kernel void gpu_compute_pair_aniso_forces_kernel(                                          \
        HOOMD_KERNEL_PARAMS(kernel_args::aniso_pair_forces_args<__VA_ARGS__>))                 \
        {                                                                                      \
        gpu_compute_pair_aniso_forces_impl<__VA_ARGS__>(args,                                  \
                                                        threadIdx,                             \
                                                        blockIdx,                              \
                                                        blockDim,                              \
                                                        hoomd_lane);                           \
        }                                                                                      \
    kernel void gpu_aniso_pair_layout_kernel(                                                  \
        HOOMD_KERNEL_PARAMS(kernel_args::aniso_pair_layout_args))                              \
        {                                                                                      \
        args.d_out[0] = sizeof(__VA_ARGS__::param_type);                                       \
        args.d_out[1] = sizeof(__VA_ARGS__::shape_type);                                       \
        args.d_out[2] = sizeof(BoxDim);                                                        \
        }

#else

namespace kernel
    {
//! Verify that the host and Metal layouts of the shared types agree
template<class evaluator> void check_aniso_pair_layout(const mtl::KernelSource& source)
    {
    static bool checked = false;
    if (checked)
        return;

    unsigned int* d_out = nullptr;
    hipMalloc(&d_out, 3 * sizeof(unsigned int));
    kernel_args::aniso_pair_layout_args args;
    args.d_out = d_out;
    mtl::launch(source, "hoomd::md::gpu_aniso_pair_layout_kernel", dim3(1), dim3(1), args);
    unsigned int sizes[3];
    hipMemcpy(sizes, d_out, sizeof(sizes), hipMemcpyDeviceToHost);
    hipFree(d_out);

    if (sizes[0] != sizeof(typename evaluator::param_type)
        || sizes[1] != sizeof(typename evaluator::shape_type) || sizes[2] != sizeof(BoxDim))
        {
        std::ostringstream s;
        s << "Metal and host layouts differ in " << source.name << ": param_type " << sizes[0]
          << " != " << sizeof(typename evaluator::param_type) << " bytes, shape_type "
          << sizes[1] << " != " << sizeof(typename evaluator::shape_type) << " bytes, BoxDim "
          << sizes[2] << " != " << sizeof(BoxDim) << " bytes";
        throw std::runtime_error(s.str());
        }
    checked = true;
    }

template<class evaluator>
hipError_t gpu_compute_pair_aniso_forces(const a_pair_args_t& pair_args,
                                         const typename evaluator::param_type* d_params,
                                         const typename evaluator::shape_type* d_shape_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.ntypes > 0);

    if (pair_args.shift_mode > 1)
        return hipErrorUnknown;

    check_aniso_pair_layout<evaluator>(HOOMD_MTL_SOURCE);

    const unsigned int tpp = pair_args.threads_per_particle;
    const unsigned int shift_mode
        = (pair_args.shift_mode == 1 && evaluator::implementsEnergyShift()) ? 1 : 0;

    std::initializer_list<mtl::FunctionConstant> constants
        = {mtl::FunctionConstant::make_uint(0, shift_mode),
           mtl::FunctionConstant::make_bool(1, pair_args.compute_virial != 0),
           mtl::FunctionConstant::make_uint(2, tpp)};

    const char* name = "hoomd::md::gpu_compute_pair_aniso_forces_kernel";
    unsigned int max_block_size
        = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name, constants);
    max_block_size -= max_block_size % gpu_aniso_pair_force_max_tpp;
    unsigned int block_size = std::min(pair_args.block_size, max_block_size);

    kernel_args::aniso_pair_forces_args<evaluator> args;
    args.d_force = pair_args.d_force;
    args.d_torque = pair_args.d_torque;
    args.d_virial = pair_args.d_virial;
    args.d_pos = pair_args.d_pos;
    args.d_charge = pair_args.d_charge;
    args.d_orientation = pair_args.d_orientation;
    args.d_tag = pair_args.d_tag;
    args.d_n_neigh = pair_args.d_n_neigh;
    args.d_nlist = pair_args.d_nlist;
    args.d_head_list = pair_args.d_head_list;
    args.d_params = d_params;
    args.d_shape_params = d_shape_params;
    args.d_rcutsq = pair_args.d_rcutsq;
    args.box = pair_args.box;
    args.virial_pitch = static_cast<unsigned int>(pair_args.virial_pitch);
    args.N = pair_args.N;
    args.ntypes = pair_args.ntypes;

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
