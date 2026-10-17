// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file PotentialBondGPUMetal.h
    \brief Metal port of the bond force kernel in PotentialBondGPU.cuh

    Each bond potential has generated .metal files (from PotentialBondGPUKernel.metal.inc and
    PotentialMeshBondGPUKernel.metal.inc) that include this header with EVALUATOR_CLASS defined.
    The kernel's enable_shared_cache template parameter is Metal function constant 0.
*/

#pragma once

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifndef __METAL_VERSION__
#include "PotentialBondGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
template<class evaluator, int group_size> struct bond_forces_args
    {
    HOOMD_DEVPTR(Scalar4) d_force;
    HOOMD_DEVPTR(Scalar) d_virial;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar) d_charge;
    HOOMD_DEVPTR(const group_storage<group_size>) blist;
    HOOMD_DEVPTR(const unsigned int) bpos_list;
    HOOMD_DEVPTR(const unsigned int) n_bonds_list;
    HOOMD_DEVPTR(const typename evaluator::param_type) d_params;
    HOOMD_DEVPTR(unsigned int) d_flags;
    BoxDim box;
    Index2D blist_idx;
    unsigned int virial_pitch;
    unsigned int N;
    unsigned int n_bond_type;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant bool bond_enable_shared_cache [[function_constant(0)]];

/*! Compute the bond forces on all N particles. One thread per particle loops over the particle's
    bonds. See gpu_compute_bond_forces_kernel in PotentialBondGPU.cuh.
*/
template<class evaluator, int group_size>
inline void
gpu_compute_bond_forces_impl(constant kernel_args::bond_forces_args<evaluator, group_size>& args,
                             metal::vec<uint, 3> threadIdx,
                             metal::vec<uint, 3> blockIdx,
                             metal::vec<uint, 3> blockDim,
                             threadgroup char* s_data)
    {
    typedef typename evaluator::param_type param_type;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load in parameters to shared memory
    threadgroup param_type* s_params = (threadgroup param_type*)s_data;
    if (bond_enable_shared_cache)
        {
        for (unsigned int cur_offset = 0; cur_offset < args.n_bond_type; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < args.n_bond_type)
                s_params[cur_offset + threadIdx.x] = args.d_params[cur_offset + threadIdx.x];
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    if (idx >= args.N)
        return;

    BoxDim box = args.box;
    Index2D blist_idx = args.blist_idx;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    unsigned int n_bonds = args.n_bonds_list[idx];

    // read in the position of our particle
    Scalar4 postype = args.d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    Scalar q(0);
    if (evaluator::needsCharge())
        q = args.d_charge[idx];

    // initialize the force to 0
    Scalar4 force = make_scalar4(0, 0, 0, 0);

    // initialize the virial to 0
    Scalar virial[6];
    for (unsigned int i = 0; i < 6; i++)
        virial[i] = 0;

    // loop over neighbors
    for (unsigned int bond_idx = 0; bond_idx < n_bonds; bond_idx++)
        {
        // the second half of the group list stores the mesh triangle neighbors, skip them
        unsigned int cur_bond_pos = args.bpos_list[blist_idx(idx, bond_idx)];
        if (cur_bond_pos > 1)
            continue;

        group_storage<group_size> cur_bond = args.blist[blist_idx(idx, bond_idx)];

        int cur_bond_idx = cur_bond.idx[0];
        int cur_bond_type = cur_bond.idx[group_size - 1];

        // get the bonded particle's position (MEM_TRANSFER: 16 bytes)
        Scalar4 neigh_postypej = args.d_pos[cur_bond_idx];
        Scalar3 neigh_pos = make_scalar3(neigh_postypej.x, neigh_postypej.y, neigh_postypej.z);

        // calculate dr (FLOPS: 3)
        Scalar3 dx = pos - neigh_pos;

        // apply periodic boundary conditions (FLOPS: 12)
        dx = box.minImage(dx);

        // get parameters for this bond type
        param_type param;
        if (bond_enable_shared_cache)
            param = s_params[cur_bond_type];
        else
            param = args.d_params[cur_bond_type];

        // calculate r squared (FLOPS: 5)
        Scalar rsq = dot(dx, dx);

        // compute the force and potential energy
        Scalar force_divr = Scalar(0.0);
        Scalar bond_eng = Scalar(0.0);
        evaluator eval(rsq, param);

        // get the bonded particle's charge (if needed)
        if (evaluator::needsCharge())
            {
            Scalar neigh_q = args.d_charge[cur_bond_idx];
            eval.setCharge(q, neigh_q);
            }

        bool evaluated = eval.evalForceAndEnergy(force_divr, bond_eng);

        if (evaluated)
            {
            // add up the virial (double counting, multiply by 0.5)
            Scalar force_div2r = force_divr / Scalar(2.0);
            virial[0] += dx.x * dx.x * force_div2r; // xx
            virial[1] += dx.x * dx.y * force_div2r; // xy
            virial[2] += dx.x * dx.z * force_div2r; // xz
            virial[3] += dx.y * dx.y * force_div2r; // yy
            virial[4] += dx.y * dx.z * force_div2r; // yz
            virial[5] += dx.z * dx.z * force_div2r; // zz

            // add up the forces
            force.x += dx.x * force_divr;
            force.y += dx.y * force_divr;
            force.z += dx.z * force_divr;
            // energy is double counted: multiply by 0.5
            force.w += bond_eng * Scalar(0.5);
            }
        else
            {
            *args.d_flags = 1;
            return;
            }
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes);
    args.d_force[idx] = force;

    for (unsigned int i = 0; i < 6; i++)
        args.d_virial[i * args.virial_pitch + idx] = virial[i];
    }

//! Define the kernel entry point for an evaluator and group size
#define HOOMD_BOND_FORCE_KERNEL(EVALUATOR, GROUP_SIZE)                                         \
    kernel void gpu_compute_bond_forces_kernel(                                                \
        HOOMD_KERNEL_PARAMS(kernel_args::bond_forces_args<EVALUATOR, GROUP_SIZE>))             \
        {                                                                                      \
        gpu_compute_bond_forces_impl<EVALUATOR, GROUP_SIZE>(args,                              \
                                                            threadIdx,                         \
                                                            blockIdx,                          \
                                                            blockDim,                          \
                                                            hoomd_dynamic_shared);             \
        }

#else

namespace kernel
    {
template<class evaluator, int group_size>
hipError_t gpu_compute_bond_forces(const kernel::bond_args_t<group_size>& bond_args,
                                   const typename evaluator::param_type* d_params,
                                   unsigned int* d_flags)
    {
    assert(d_params);
    assert(bond_args.n_bond_types > 0);
    assert(bond_args.block_size != 0);

    // cache the per type parameters in threadgroup memory when they fit
    size_t shared_bytes = sizeof(typename evaluator::param_type) * bond_args.n_bond_types;
    bool enable_shared_cache = shared_bytes <= bond_args.devprop.sharedMemPerBlock;
    if (!enable_shared_cache)
        shared_bytes = 0;

    std::initializer_list<mtl::FunctionConstant> constants
        = {mtl::FunctionConstant::make_bool(0, enable_shared_cache)};

    const char* name = "hoomd::md::gpu_compute_bond_forces_kernel";
    unsigned int max_block_size
        = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name, constants);
    unsigned int run_block_size = std::min(bond_args.block_size, max_block_size);

    kernel_args::bond_forces_args<evaluator, group_size> args;
    args.d_force = bond_args.d_force;
    args.d_virial = bond_args.d_virial;
    args.d_pos = bond_args.d_pos;
    args.d_charge = evaluator::needsCharge() ? bond_args.d_charge : nullptr;
    args.blist = bond_args.d_gpu_bondlist;
    args.bpos_list = bond_args.d_gpu_bond_pos;
    args.n_bonds_list = bond_args.d_gpu_n_bonds;
    args.d_params = d_params;
    args.d_flags = d_flags;
    args.box = bond_args.box;
    args.blist_idx = bond_args.gpu_table_indexer;
    args.virial_pitch = static_cast<unsigned int>(bond_args.virial_pitch);
    args.N = bond_args.N;
    args.n_bond_type = bond_args.n_bond_types;

    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(bond_args.N / run_block_size + 1),
                       dim3(run_block_size),
                       args,
                       shared_bytes,
                       constants);
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
