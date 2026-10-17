// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file PotentialPairDPDThermoGPUMetal.h
    \brief Metal port of the DPD force kernel in PotentialPairDPDThermoGPU.cuh

    Each DPD potential has a generated .metal file (from PotentialPairDPDThermoGPUKernel.metal.inc)
    that includes this header with EVALUATOR_CLASS defined. The CUDA template parameters are Metal
    function constants: 0: shift_mode, 1: compute_virial, 2: threads per particle.
*/

#pragma once

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifdef __METAL_VERSION__
#include "hoomd/mtl/WarpTools.h"
#else
#include "PotentialPairDPDThermoGPU.cuh"

#include <stdexcept>
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
template<class evaluator> struct dpd_forces_args
    {
    HOOMD_DEVPTR(Scalar4) d_force;
    HOOMD_DEVPTR(Scalar) d_virial;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar4) d_vel;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(const unsigned int) d_n_neigh;
    HOOMD_DEVPTR(const unsigned int) d_nlist;
    HOOMD_DEVPTR(const size_t) d_head_list;
    HOOMD_DEVPTR(const typename evaluator::param_type) d_params;
    HOOMD_DEVPTR(const Scalar) d_rcutsq;
    BoxDim box;
    uint64_t timestep;
    unsigned int virial_pitch;
    unsigned int N;
    Scalar deltaT;
    Scalar T;
    unsigned int ntypes;
    uint16_t seed;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant uint dpd_shift_mode [[function_constant(0)]];
constant bool dpd_compute_virial [[function_constant(1)]];
constant uint dpd_tpp [[function_constant(2)]];

/*! Compute the DPD forces (conservative, dissipative, and random) on all N particles. See
    gpu_compute_dpd_forces_kernel in PotentialPairDPDThermoGPU.cuh.
*/
template<class evaluator>
inline void gpu_compute_dpd_forces_impl(constant kernel_args::dpd_forces_args<evaluator>& args,
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

    // load in the per type pair parameters
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            s_rcutsq[cur_offset + threadIdx.x] = args.d_rcutsq[cur_offset + threadIdx.x];
            s_params[cur_offset + threadIdx.x] = args.d_params[cur_offset + threadIdx.x];
            }
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const unsigned int tpp = dpd_tpp;

    // start by identifying which particle we are to handle
    const unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    bool active = idx < args.N;

    // initialize the force to 0
    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virial[6];
    for (unsigned int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);

    if (active)
        {
        BoxDim box = args.box;

        // load in the length of the neighbor list (MEM_TRANSFER: 4 bytes)
        unsigned int n_neigh = args.d_n_neigh[idx];

        // read in the position of our particle.
        Scalar4 postypei = args.d_pos[idx];
        Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

        // read in the velocity of our particle.
        Scalar4 velmassi = args.d_vel[idx];
        Scalar3 veli = make_scalar3(velmassi.x, velmassi.y, velmassi.z);

        const size_t head_idx = args.d_head_list[idx];

        // prefetch neighbor index
        unsigned int cur_j = 0;
        unsigned int next_j(0);
        next_j = (threadIdx.x % tpp < n_neigh) ? args.d_nlist[head_idx + threadIdx.x % tpp] : 0;

        // this particle's tag
        unsigned int tagi = args.d_tag[idx];

        // loop over neighbors
        for (unsigned int neigh_idx = threadIdx.x % tpp; neigh_idx < n_neigh; neigh_idx += tpp)
            {
            // read the current neighbor index (MEM TRANSFER: 4 bytes)
            cur_j = next_j;
            if (neigh_idx + tpp < n_neigh)
                next_j = args.d_nlist[head_idx + neigh_idx + tpp];

            // get the neighbor's position (MEM TRANSFER: 16 bytes)
            Scalar4 postypej = args.d_pos[cur_j];
            Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

            // get the neighbor's velocity (MEM TRANSFER: 16 bytes)
            Scalar4 velmassj = args.d_vel[cur_j];
            Scalar3 velj = make_scalar3(velmassj.x, velmassj.y, velmassj.z);

            // calculate dr (with periodic boundary conditions) (FLOPS: 3)
            Scalar3 dx = posi - posj;

            // calculate dv (FLOPS: 3)
            Scalar3 dv = veli - velj;

            // apply periodic boundary conditions: (FLOPS 12)
            box.minImage(dx, dv);

            // calculate r squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            // calculate the drag term r \dot v
            Scalar rdotv = dot(dx, dv);

            // access the per type pair parameters
            unsigned int typpair
                = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypej.w));
            Scalar rcutsq = s_rcutsq[typpair];
            param_type param = s_params[typpair];

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            bool energy_shift = false;
            if (dpd_shift_mode == 1)
                energy_shift = true;

            evaluator eval(rsq, rcutsq, param);

            // evaluate the potential
            Scalar force_divr = Scalar(0.0);
            Scalar force_divr_cons = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);

            // Special Potential Pair DPD Requirements
            // use particle i's and j's tags
            unsigned int tagj = args.d_tag[cur_j];
            eval.set_seed_ij_timestep(args.seed, tagi, tagj, args.timestep);
            eval.setDeltaT(args.deltaT);
            eval.setRDotV(rdotv);
            eval.setT(args.T);

            eval.evalForceEnergyThermo(force_divr, force_divr_cons, pair_eng, energy_shift);

            // calculate the virial (FLOPS: 3)
            if (dpd_compute_virial)
                {
                Scalar force_div2r_cons = Scalar(0.5) * force_divr_cons;
                virial[0] += dx.x * dx.x * force_div2r_cons;
                virial[1] += dx.x * dx.y * force_div2r_cons;
                virial[2] += dx.x * dx.z * force_div2r_cons;
                virial[3] += dx.y * dx.y * force_div2r_cons;
                virial[4] += dx.y * dx.z * force_div2r_cons;
                virial[5] += dx.z * dx.z * force_div2r_cons;
                }

            // add up the force vector components (FLOPS: 7)
            force.x += dx.x * force_divr;
            force.y += dx.y * force_divr;
            force.z += dx.z * force_divr;

            force.w += pair_eng;
            }

        // potential energy per particle must be halved
        force.w *= Scalar(0.5);
        }

    // reduce force over threads in a logical warp (all lanes participate)
    force.x = detail::warp_reduce_sum(force.x, lane, tpp);
    force.y = detail::warp_reduce_sum(force.y, lane, tpp);
    force.z = detail::warp_reduce_sum(force.z, lane, tpp);
    force.w = detail::warp_reduce_sum(force.w, lane, tpp);

    // now that the force calculation is complete, write out the result
    if (active && threadIdx.x % tpp == 0)
        args.d_force[idx] = force;

    if (dpd_compute_virial)
        {
        for (unsigned int i = 0; i < 6; ++i)
            virial[i] = detail::warp_reduce_sum(virial[i], lane, tpp);

        // if we are the first thread in the cta, write out virial to global mem
        if (active && threadIdx.x % tpp == 0)
            for (unsigned int i = 0; i < 6; i++)
                args.d_virial[i * args.virial_pitch + idx] = virial[i];
        }
    }

//! Define the kernel entry point for an evaluator
#define HOOMD_DPD_FORCE_KERNEL(EVALUATOR)                                                      \
    kernel void gpu_compute_dpd_forces_kernel(                                                 \
        HOOMD_KERNEL_PARAMS(kernel_args::dpd_forces_args<EVALUATOR>))                          \
        {                                                                                      \
        gpu_compute_dpd_forces_impl<EVALUATOR>(args,                                           \
                                               threadIdx,                                      \
                                               blockIdx,                                       \
                                               blockDim,                                       \
                                               hoomd_lane,                                     \
                                               hoomd_dynamic_shared);                          \
        }

#else

namespace kernel
    {
template<class evaluator>
hipError_t gpu_compute_dpd_forces(const dpd_pair_args_t& args,
                                  const typename evaluator::param_type* d_params)
    {
    assert(d_params);
    assert(args.d_rcutsq);
    assert(args.ntypes > 0);

    if (args.shift_mode > 1)
        return hipErrorUnknown;

    Index2D typpair_idx(args.ntypes);
    const size_t shared_bytes
        = (sizeof(Scalar) + sizeof(typename evaluator::param_type)) * typpair_idx.getNumElements();
    if (shared_bytes > args.devprop.sharedMemPerBlock)
        {
        throw std::runtime_error("Pair potential parameters exceed the available shared "
                                 "memory per block.");
        }

    const unsigned int tpp = args.threads_per_particle;
    std::initializer_list<mtl::FunctionConstant> constants
        = {mtl::FunctionConstant::make_uint(0, args.shift_mode),
           mtl::FunctionConstant::make_bool(1, args.compute_virial != 0),
           mtl::FunctionConstant::make_uint(2, tpp)};

    const char* name = "hoomd::md::gpu_compute_dpd_forces_kernel";
    unsigned int max_block_size
        = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name, constants);
    max_block_size -= max_block_size % gpu_dpd_pair_force_max_tpp;
    unsigned int block_size = std::min(args.block_size, max_block_size);

    kernel_args::dpd_forces_args<evaluator> kargs;
    kargs.d_force = args.d_force;
    kargs.d_virial = args.d_virial;
    kargs.d_pos = args.d_pos;
    kargs.d_vel = args.d_vel;
    kargs.d_tag = args.d_tag;
    kargs.d_n_neigh = args.d_n_neigh;
    kargs.d_nlist = args.d_nlist;
    kargs.d_head_list = args.d_head_list;
    kargs.d_params = d_params;
    kargs.d_rcutsq = args.d_rcutsq;
    kargs.box = args.box;
    kargs.timestep = args.timestep;
    kargs.virial_pitch = static_cast<unsigned int>(args.virial_pitch);
    kargs.N = args.N;
    kargs.deltaT = args.deltaT;
    kargs.T = args.T;
    kargs.ntypes = args.ntypes;
    kargs.seed = args.seed;

    dim3 grid(args.N / (block_size / tpp) + 1);
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       grid,
                       dim3(block_size),
                       kargs,
                       shared_bytes,
                       constants);
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
