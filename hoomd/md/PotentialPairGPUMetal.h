// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file PotentialPairGPUMetal.h
    \brief Metal port of the pair force kernel in PotentialPairGPU.cuh

    Each pair potential has a generated .metal file (from PotentialPairGPUKernel.metal.inc) that
    includes this header with EVALUATOR_CLASS defined. The CUDA template parameters of the kernel
    are Metal function constants: 0: shift_mode, 1: compute_virial, 2: threads per particle,
    3: enable_shared_cache.
*/

#pragma once

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifdef __METAL_VERSION__
#include "hoomd/mtl/WarpTools.h"
#else
#include "PotentialPairGPU.cuh"

#include <sstream>
#include <stdexcept>
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
template<class evaluator> struct pair_forces_args
    {
    HOOMD_DEVPTR(Scalar4) d_force;
    HOOMD_DEVPTR(Scalar) d_virial;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar) d_charge;
    HOOMD_DEVPTR(const unsigned int) d_n_neigh;
    HOOMD_DEVPTR(const unsigned int) d_nlist;
    HOOMD_DEVPTR(const size_t) d_head_list;
    HOOMD_DEVPTR(const typename evaluator::param_type) d_params;
    HOOMD_DEVPTR(const Scalar) d_rcutsq;
    HOOMD_DEVPTR(const Scalar) d_ronsq;
    BoxDim box;
    unsigned int virial_pitch;
    unsigned int N;
    unsigned int ntypes;
    };

//! Sizes of the types shared with the host, written by the layout check kernel
struct pair_layout_args
    {
    HOOMD_DEVPTR(unsigned int) d_out;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant uint pair_shift_mode [[function_constant(0)]];
constant bool pair_compute_virial [[function_constant(1)]];
constant uint pair_tpp [[function_constant(2)]];
constant bool pair_enable_shared_cache [[function_constant(3)]];

/*! Compute the pair forces on all N particles. Each group of pair_tpp threads computes the force
    on one particle. See gpu_compute_pair_forces_shared_kernel in PotentialPairGPU.cuh.
*/
template<class evaluator>
inline void gpu_compute_pair_forces_impl(constant kernel_args::pair_forces_args<evaluator>& args,
                                         metal::vec<uint, 3> threadIdx,
                                         metal::vec<uint, 3> blockIdx,
                                         metal::vec<uint, 3> blockDim,
                                         ushort lane,
                                         threadgroup char* s_data)
    {
    typedef typename evaluator::param_type param_type;

    Index2D typpair_idx(args.ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared arrays for per type pair parameters
    threadgroup param_type* s_params = (threadgroup param_type*)s_data;
    threadgroup Scalar* s_rcutsq
        = (threadgroup Scalar*)(s_data + num_typ_parameters * sizeof(param_type));
    threadgroup Scalar* s_ronsq = s_rcutsq + num_typ_parameters;

    // load in the per type pair parameters
    if (pair_enable_shared_cache)
        {
        for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < num_typ_parameters)
                {
                s_rcutsq[cur_offset + threadIdx.x] = args.d_rcutsq[cur_offset + threadIdx.x];
                if (pair_shift_mode == 2)
                    s_ronsq[cur_offset + threadIdx.x] = args.d_ronsq[cur_offset + threadIdx.x];
                }
            }

        unsigned int param_size = num_typ_parameters * sizeof(param_type) / sizeof(int);
        device const int* d_params_int = (device const int*)args.d_params;
        threadgroup int* s_params_int = (threadgroup int*)s_params;
        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < param_size)
                s_params_int[cur_offset + threadIdx.x] = d_params_int[cur_offset + threadIdx.x];
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    const unsigned int tpp = pair_tpp;

    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    bool active = idx < args.N;

    // initialize the force to 0
    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virialxx = Scalar(0.0);
    Scalar virialxy = Scalar(0.0);
    Scalar virialxz = Scalar(0.0);
    Scalar virialyy = Scalar(0.0);
    Scalar virialyz = Scalar(0.0);
    Scalar virialzz = Scalar(0.0);

    if (active)
        {
        BoxDim box = args.box;

        // load in the length of the neighbor list
        unsigned int n_neigh = args.d_n_neigh[idx];

        // read in the position of our particle
        Scalar4 postypei = args.d_pos[idx];
        Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

        Scalar qi = Scalar(0);
        if (evaluator::needsCharge())
            qi = args.d_charge[idx];

        size_t my_head = args.d_head_list[idx];
        unsigned int cur_j = 0;

        unsigned int next_j(0);
        next_j = threadIdx.x % tpp < n_neigh ? args.d_nlist[my_head + threadIdx.x % tpp] : 0;

        // loop over neighbors
        for (unsigned int neigh_idx = threadIdx.x % tpp; neigh_idx < n_neigh; neigh_idx += tpp)
            {
            // read the current neighbor index (MEM TRANSFER: 4 bytes)
            cur_j = next_j;
            if (neigh_idx + tpp < n_neigh)
                next_j = args.d_nlist[my_head + neigh_idx + tpp];

            // get the neighbor's position
            Scalar4 postypej = args.d_pos[cur_j];
            Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

            Scalar qj = Scalar(0.0);
            if (evaluator::needsCharge())
                qj = args.d_charge[cur_j];

            // calculate dr (with periodic boundary conditions)
            Scalar3 dx = posi - posj;

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r squared
            Scalar rsq = dot(dx, dx);

            // access the per type pair parameters
            unsigned int typpair
                = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypej.w));
            Scalar rcutsq;
            param_type param;
            Scalar ronsq = Scalar(0.0);

            if (pair_enable_shared_cache)
                {
                rcutsq = s_rcutsq[typpair];
                param = s_params[typpair];
                if (pair_shift_mode == 2)
                    ronsq = s_ronsq[typpair];
                }
            else
                {
                rcutsq = args.d_rcutsq[typpair];
                param = args.d_params[typpair];
                if (pair_shift_mode == 2)
                    ronsq = args.d_ronsq[typpair];
                }

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            // or 2) shift mode is explor and ron > rcut
            bool energy_shift = false;
            if (pair_shift_mode == 1)
                energy_shift = true;
            else if (pair_shift_mode == 2)
                {
                if (ronsq > rcutsq)
                    energy_shift = true;
                }

            // evaluate the potential
            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);

            evaluator eval(rsq, rcutsq, param);
            if (evaluator::needsCharge())
                eval.setCharge(qi, qj);

            eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

            if (pair_shift_mode == 2)
                {
                if (rsq >= ronsq && rsq < rcutsq)
                    {
                    // Implement XPLOR smoothing
                    Scalar old_pair_eng = pair_eng;
                    Scalar old_force_divr = force_divr;

                    // calculate 1.0 / (xplor denominator)
                    Scalar xplor_denom_inv
                        = Scalar(1.0) / ((rcutsq - ronsq) * (rcutsq - ronsq) * (rcutsq - ronsq));

                    Scalar rsq_minus_r_cut_sq = rsq - rcutsq;
                    Scalar s = rsq_minus_r_cut_sq * rsq_minus_r_cut_sq
                               * (rcutsq + Scalar(2.0) * rsq - Scalar(3.0) * ronsq)
                               * xplor_denom_inv;
                    Scalar ds_dr_divr
                        = Scalar(12.0) * (rsq - ronsq) * rsq_minus_r_cut_sq * xplor_denom_inv;

                    // make modifications to the old pair energy and force
                    pair_eng = old_pair_eng * s;
                    force_divr = s * old_force_divr - ds_dr_divr * old_pair_eng;
                    }
                }

            // calculate the virial
            if (pair_compute_virial)
                {
                Scalar force_div2r = Scalar(0.5) * force_divr;
                virialxx += dx.x * dx.x * force_div2r;
                virialxy += dx.x * dx.y * force_div2r;
                virialxz += dx.x * dx.z * force_div2r;
                virialyy += dx.y * dx.y * force_div2r;
                virialyz += dx.y * dx.z * force_div2r;
                virialzz += dx.z * dx.z * force_div2r;
                }

            // add up the force vector components
            force.x += dx.x * force_divr;
            force.y += dx.y * force_divr;
            force.z += dx.z * force_divr;

            force.w += pair_eng;
            }

        // potential energy per particle must be halved
        force.w *= Scalar(0.5);
        }

    // reduce the force over the threads of each particle (all lanes participate)
    force.x = detail::warp_reduce_sum(force.x, lane, tpp);
    force.y = detail::warp_reduce_sum(force.y, lane, tpp);
    force.z = detail::warp_reduce_sum(force.z, lane, tpp);
    force.w = detail::warp_reduce_sum(force.w, lane, tpp);

    // now that the force calculation is complete, write out the result
    if (active && threadIdx.x % tpp == 0)
        args.d_force[idx] = force;

    if (pair_compute_virial)
        {
        virialxx = detail::warp_reduce_sum(virialxx, lane, tpp);
        virialxy = detail::warp_reduce_sum(virialxy, lane, tpp);
        virialxz = detail::warp_reduce_sum(virialxz, lane, tpp);
        virialyy = detail::warp_reduce_sum(virialyy, lane, tpp);
        virialyz = detail::warp_reduce_sum(virialyz, lane, tpp);
        virialzz = detail::warp_reduce_sum(virialzz, lane, tpp);

        // if we are the first thread in the cta, write out virial to global mem
        if (active && threadIdx.x % tpp == 0)
            {
            const unsigned int pitch = args.virial_pitch;
            args.d_virial[0 * pitch + idx] = virialxx;
            args.d_virial[1 * pitch + idx] = virialxy;
            args.d_virial[2 * pitch + idx] = virialxz;
            args.d_virial[3 * pitch + idx] = virialyy;
            args.d_virial[4 * pitch + idx] = virialyz;
            args.d_virial[5 * pitch + idx] = virialzz;
            }
        }
    }

//! Define the kernel entry points for an evaluator
#define HOOMD_PAIR_FORCE_KERNELS(EVALUATOR)                                                    \
    kernel void gpu_compute_pair_forces_shared_kernel(                                         \
        HOOMD_KERNEL_PARAMS(kernel_args::pair_forces_args<EVALUATOR>))                         \
        {                                                                                      \
        gpu_compute_pair_forces_impl<EVALUATOR>(args,                                          \
                                                threadIdx,                                     \
                                                blockIdx,                                      \
                                                blockDim,                                      \
                                                hoomd_lane,                                    \
                                                hoomd_dynamic_shared);                         \
        }                                                                                      \
    kernel void gpu_pair_layout_kernel(HOOMD_KERNEL_PARAMS(kernel_args::pair_layout_args))     \
        {                                                                                      \
        args.d_out[0] = sizeof(EVALUATOR::param_type);                                         \
        args.d_out[1] = sizeof(BoxDim);                                                        \
        }

#else

namespace kernel
    {
//! Verify that the host and Metal layouts of the shared types agree
template<class evaluator> void check_pair_layout(const mtl::KernelSource& source)
    {
    static bool checked = false;
    if (checked)
        return;

    unsigned int* d_out = nullptr;
    hipMalloc(&d_out, 2 * sizeof(unsigned int));
    kernel_args::pair_layout_args args;
    args.d_out = d_out;
    mtl::launch(source, "hoomd::md::gpu_pair_layout_kernel", dim3(1), dim3(1), args);
    unsigned int sizes[2];
    hipMemcpy(sizes, d_out, sizeof(sizes), hipMemcpyDeviceToHost);
    hipFree(d_out);

    if (sizes[0] != sizeof(typename evaluator::param_type) || sizes[1] != sizeof(BoxDim))
        {
        std::ostringstream s;
        s << "Metal and host layouts differ in " << source.name << ": param_type "
          << sizes[0] << " != " << sizeof(typename evaluator::param_type) << " bytes, BoxDim "
          << sizes[1] << " != " << sizeof(BoxDim) << " bytes";
        throw std::runtime_error(s.str());
        }
    checked = true;
    }

template<class evaluator>
hipError_t gpu_compute_pair_forces(const pair_args_t& pair_args,
                                   const typename evaluator::param_type* d_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.d_ronsq);
    assert(pair_args.ntypes > 0);

    check_pair_layout<evaluator>(HOOMD_MTL_SOURCE);

    const unsigned int tpp = pair_args.threads_per_particle;

    // the per type pair parameters are cached in threadgroup memory when they fit
    Index2D typpair_idx(pair_args.ntypes);
    size_t param_shared_bytes = (2 * sizeof(Scalar) + sizeof(typename evaluator::param_type))
                                * typpair_idx.getNumElements();
    bool enable_shared_cache = param_shared_bytes <= pair_args.devprop.sharedMemPerBlock;
    if (!enable_shared_cache)
        param_shared_bytes = 0;

    std::initializer_list<mtl::FunctionConstant> constants
        = {mtl::FunctionConstant::make_uint(0, pair_args.shift_mode),
           mtl::FunctionConstant::make_bool(1, pair_args.compute_virial != 0),
           mtl::FunctionConstant::make_uint(2, tpp),
           mtl::FunctionConstant::make_bool(3, enable_shared_cache)};

    const char* name = "hoomd::md::gpu_compute_pair_forces_shared_kernel";
    unsigned int max_block_size
        = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name, constants);
    max_block_size -= max_block_size % gpu_pair_force_max_tpp;
    unsigned int block_size = std::min(pair_args.block_size, max_block_size);

    kernel_args::pair_forces_args<evaluator> args;
    args.d_force = pair_args.d_force;
    args.d_virial = pair_args.d_virial;
    args.d_pos = pair_args.d_pos;
    args.d_charge = evaluator::needsCharge() ? pair_args.d_charge : nullptr;
    args.d_n_neigh = pair_args.d_n_neigh;
    args.d_nlist = pair_args.d_nlist;
    args.d_head_list = pair_args.d_head_list;
    args.d_params = d_params;
    args.d_rcutsq = pair_args.d_rcutsq;
    args.d_ronsq = pair_args.d_ronsq;
    args.box = pair_args.box;
    args.virial_pitch = static_cast<unsigned int>(pair_args.virial_pitch);
    args.N = pair_args.N;
    args.ntypes = pair_args.ntypes;

    dim3 grid(pair_args.N / (block_size / tpp) + 1);
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       grid,
                       dim3(block_size),
                       args,
                       param_shared_bytes,
                       constants);
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
