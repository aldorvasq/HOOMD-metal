// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file TersoffGPU.metal
    \brief Metal port of TersoffGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/BoxDim.h"

#ifdef __METAL_VERSION__
#include "hoomd/mtl/WarpTools.h"
#else
#include "PotentialTersoffGPU.cuh"

#include <stdexcept>
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct gpu_compute_triplet_forces_kernel_args
    {
    HOOMD_DEVPTR(Scalar4) d_force;
    unsigned int N;
    HOOMD_DEVPTR(Scalar) d_virial;
    size_t virial_pitch;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    BoxDim box;
    HOOMD_DEVPTR(const unsigned int) d_n_neigh;
    HOOMD_DEVPTR(const unsigned int) d_nlist;
    HOOMD_DEVPTR(const size_t) d_head_list;
    HOOMD_DEVPTR(const typename EVALUATOR_CLASS::param_type) d_params;
    HOOMD_DEVPTR(const Scalar) d_rcutsq;
    unsigned int ntypes;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant bool compute_virial [[function_constant(0)]];
constant uint tpp [[function_constant(1)]];

//! Kernel for calculating the triplet forces (see gpu_compute_triplet_forces_kernel in
//! PotentialTersoffGPU.cuh)
kernel void gpu_compute_triplet_forces_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_compute_triplet_forces_kernel_args))
    {
    device Scalar4* d_force = args.d_force;
    const unsigned int N = args.N;
    device Scalar* d_virial = args.d_virial;
    size_t virial_pitch = args.virial_pitch;
    device const Scalar4* d_pos = args.d_pos;
    const BoxDim box = args.box;
    device const unsigned int* d_n_neigh = args.d_n_neigh;
    device const unsigned int* d_nlist = args.d_nlist;
    device const size_t* d_head_list = args.d_head_list;
    device const typename EVALUATOR_CLASS::param_type* d_params = args.d_params;
    device const Scalar* d_rcutsq = args.d_rcutsq;
    const unsigned int ntypes = args.ntypes;

    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared arrays for per type pair parameters
    HIP_DYNAMIC_SHARED(char, s_data)
    threadgroup typename EVALUATOR_CLASS::param_type* s_params = (threadgroup typename EVALUATOR_CLASS::param_type*)(&s_data[0]);
    threadgroup Scalar* s_rcutsq = (threadgroup Scalar*)(&s_data[num_typ_parameters * sizeof(typename EVALUATOR_CLASS::param_type)]);

    threadgroup Scalar* s_phi_ab = s_rcutsq + num_typ_parameters;

    // load in the per type pair parameters
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            s_rcutsq[cur_offset + threadIdx.x] = d_rcutsq[cur_offset + threadIdx.x];
            s_params[cur_offset + threadIdx.x] = d_params[cur_offset + threadIdx.x];
            }
        }

    __syncthreads();

    for (unsigned int i = 0; i < ntypes; ++i)
        {
        // reset phi term
        s_phi_ab[threadIdx.x * ntypes + i] = Scalar(0.0);
        }

    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;

    if (idx >= N)
        return;

    // load in the length of the neighbor list (MEM_TRANSFER: 4 bytes)
    unsigned int n_neigh = d_n_neigh[idx];

    // read in the position of the particle
    Scalar4 postypei = __ldg(d_pos + idx);
    Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

    // initialize the force to 0
    Scalar4 forcei = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    Scalar viriali_xx(0.0);
    Scalar viriali_xy(0.0);
    Scalar viriali_xz(0.0);
    Scalar viriali_yy(0.0);
    Scalar viriali_yz(0.0);
    Scalar viriali_zz(0.0);

    // check if this is the Tersoff/SquareDensity potential or the RevCross
    if (EVALUATOR_CLASS::flag_for_RevCross)
        {
        // this is the RevCross potential
        // prefetch neighbor index
        const size_t head_idx = d_head_list[idx];
        unsigned int cur_j = 0;
        unsigned int next_j(0);
        size_t my_head = d_head_list[idx];

        next_j = threadIdx.x % tpp < n_neigh ? __ldg(d_nlist + my_head + threadIdx.x % tpp) : 0;

        // loop over neighbors in strided way
        for (int neigh_idx = threadIdx.x % tpp; neigh_idx < n_neigh; neigh_idx += tpp)
            {
            // read the current neighbor index (MEM TRANSFER: 4 bytes)
            // prefetch the next value and set the current one
            cur_j = next_j;
            if (neigh_idx + tpp < n_neigh)
                {
                next_j = __ldg(d_nlist + head_idx + neigh_idx + tpp);
                }

            // read the position of j (MEM TRANSFER: 16 bytes)
            Scalar4 postypej = __ldg(d_pos + cur_j);
            Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

            // initialize the force on j
            Scalar4 forcej = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

            // compute r_ij (FLOPS: 3)
            Scalar3 dxij = posi - posj;

            // apply periodic boundary conditions (FLOPS: 12)
            dxij = box.minImage(dxij);

            // compute rij_sq (FLOPS: 5)
            Scalar rij_sq = dot(dxij, dxij);

            // access the per type-pair parameters
            unsigned int typpair
                = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypej.w));
            Scalar rcutsq = s_rcutsq[typpair];
            const typename EVALUATOR_CLASS::param_type param = s_params[typpair];

            // compute the base repulsive and attractive terms of the potential
            Scalar invratio = 0.0;
            Scalar invratio2 = 0.0;
            EVALUATOR_CLASS eval(rij_sq, rcutsq, param);
            bool evaluatedij = eval.evalRepulsiveAndAttractive(invratio, invratio2);

            if (evaluatedij)
                {
                // evaluate the force and energy from the ij interaction
                Scalar force_divr = Scalar(0.0);
                Scalar potential_eng = Scalar(0.0);
                Scalar bij = Scalar(0.0); // not used
                eval.evalForceij(invratio,
                                 invratio2,
                                 Scalar(0.0),
                                 Scalar(0.0),
                                 bij,
                                 force_divr,
                                 potential_eng);

                // add the forces and energies to their respective particles
                forcei.x += dxij.x * force_divr;
                forcei.y += dxij.y * force_divr;
                forcei.z += dxij.z * force_divr;

                forcej.x -= dxij.x * force_divr;
                forcej.y -= dxij.y * force_divr;
                forcej.z -= dxij.z * force_divr;

                forcej.w += potential_eng;
                forcei.w += potential_eng;

                // calculate the virial
                if (compute_virial)
                    {
                    viriali_xx += dxij.x * dxij.x * force_divr;
                    viriali_xy += dxij.x * dxij.y * force_divr;
                    viriali_xz += dxij.x * dxij.z * force_divr;
                    viriali_yy += dxij.y * dxij.y * force_divr;
                    viriali_yz += dxij.y * dxij.z * force_divr;
                    viriali_zz += dxij.z * dxij.z * force_divr;
                    }

                // now evaluate the force from the ik interactions. As in the CPU implementation
                // (PotentialTersoff::computeForces), every particle i is the vertex of the
                // triplets formed with each pair of its neighbors {j, k}, with k after j in the
                // neighbor list. PotentialTersoffGPU.cuh instead requires idx < j < k by particle
                // index and starts with particle 0 in place of the first neighbor, which does not
                // match the CPU forces.
                for (int neigh_idy = neigh_idx + 1; neigh_idy < n_neigh; neigh_idy++)
                    {
                    unsigned int cur_k = __ldg(d_nlist + head_idx + neigh_idy);

                        {
                        // get the position of neighbor k
                        Scalar4 postypek = __ldg(d_pos + cur_k);
                        Scalar3 posk = make_scalar3(postypek.x, postypek.y, postypek.z);

                        // get the type pair parameters for i and k
                        typpair
                            = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypek.w));
                        Scalar temp_rcutsq = s_rcutsq[typpair];
                        typename EVALUATOR_CLASS::param_type temp_param = s_params[typpair];

                        // compute rik
                        Scalar3 dxik = posi - posk;
                        // apply the periodic boundary conditions
                        dxik = box.minImage(dxik);
                        // compute rik_sq
                        Scalar rik_sq = dot(dxik, dxik);

                        EVALUATOR_CLASS temp_eval(rij_sq, temp_rcutsq, temp_param);
                        temp_eval.setRik(rik_sq);
                        bool temp_evaluated = temp_eval.areInteractive();

                        if (temp_evaluated)
                            {
                            Scalar4 forcek
                                = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

                            // set up the EVALUATOR_CLASS
                            eval.setRik(rik_sq);

                            // compute the force
                            Scalar force_divr_ij = 0.0;
                            Scalar force_divr_ik = 0.0;
                            Scalar3 force_divr_ij_vec = make_scalar3(0.0, 0.0, 0.0);
                            Scalar3 force_divr_ik_vec = make_scalar3(0.0, 0.0, 0.0);
                            bool evaluatedjk = eval.evalForceik(invratio,
                                                                invratio2,
                                                                Scalar(0.0),
                                                                Scalar(0.0),
                                                                force_divr_ij_vec,
                                                                force_divr_ik_vec);

                            if (evaluatedjk)
                                {
                                // I stored the modulus of the force in the first component
                                force_divr_ij = force_divr_ij_vec.x;
                                force_divr_ik = force_divr_ik_vec.x;

                                // add the forces to their respective particles
                                forcei.x += force_divr_ij * dxij.x + force_divr_ik * dxik.x;
                                forcei.y += force_divr_ij * dxij.y + force_divr_ik * dxik.y;
                                forcei.z += force_divr_ij * dxij.z + force_divr_ik * dxik.z;

                                forcej.x -= force_divr_ij * dxij.x;
                                forcej.y -= force_divr_ij * dxij.y;
                                forcej.z -= force_divr_ij * dxij.z;

                                forcek.x -= force_divr_ik * dxik.x;
                                forcek.y -= force_divr_ik * dxik.y;
                                forcek.z -= force_divr_ik * dxik.z;

                                atomicAdd((device Scalar*)(d_force + cur_k) + 0, forcek.x);
                                atomicAdd((device Scalar*)(d_force + cur_k) + 1, forcek.y);
                                atomicAdd((device Scalar*)(d_force + cur_k) + 2, forcek.z);

                                // evaluate the virial contribute of this 3 body interaction
                                if (compute_virial)
                                    {
                                    // a single term is needed to account for all of the 3 body
                                    // virial that is stored in the 'i' particle data look at S.
                                    // Ciarella and W.G. Ellenbroek 2019
                                    // https://arxiv.org/abs/1912.08569 for the definition of the
                                    // virial tensor
                                    viriali_xx += (force_divr_ij * dxij.x * dxij.x
                                                   + force_divr_ik * dxik.x * dxik.x);
                                    viriali_yy += (force_divr_ij * dxij.y * dxij.y
                                                   + force_divr_ik * dxik.y * dxik.y);
                                    viriali_zz += (force_divr_ij * dxij.z * dxij.z
                                                   + force_divr_ik * dxik.z * dxik.z);
                                    viriali_xy += (force_divr_ij * dxij.x * dxij.y
                                                   + force_divr_ik * dxik.x * dxik.y);
                                    viriali_xz += (force_divr_ij * dxij.x * dxij.z
                                                   + force_divr_ik * dxik.x * dxik.z);
                                    viriali_yz += (force_divr_ij * dxij.y * dxij.z
                                                   + force_divr_ik * dxik.y * dxik.z);
                                    }
                                }
                            }
                        }
                    }

                // potential energy of j must be halved
                // write out the result for particle j
                atomicAdd((device Scalar*)(d_force + cur_j) + 0, forcej.x);
                atomicAdd((device Scalar*)(d_force + cur_j) + 1, forcej.y);
                atomicAdd((device Scalar*)(d_force + cur_j) + 2, forcej.z);
                atomicAdd((device Scalar*)(d_force + cur_j) + 3, forcej.w);
                }
            }
        // potential energy per particle must be halved
        // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
        atomicAdd((device Scalar*)(d_force + idx) + 0, forcei.x);
        atomicAdd((device Scalar*)(d_force + idx) + 1, forcei.y);
        atomicAdd((device Scalar*)(d_force + idx) + 2, forcei.z);
        atomicAdd((device Scalar*)(d_force + idx) + 3, forcei.w);

        if (compute_virial)
            {
            atomicAdd(&d_virial[0 * virial_pitch + idx], viriali_xx);
            atomicAdd(&d_virial[1 * virial_pitch + idx], viriali_xy);
            atomicAdd(&d_virial[2 * virial_pitch + idx], viriali_xz);
            atomicAdd(&d_virial[3 * virial_pitch + idx], viriali_yy);
            atomicAdd(&d_virial[4 * virial_pitch + idx], viriali_yz);
            atomicAdd(&d_virial[5 * virial_pitch + idx], viriali_zz);
            }
        }
    else
        {
        // this is the Tersoff potential
        // loop over neighbors to calculate per-particle energy
        if (EVALUATOR_CLASS::hasPerParticleEnergy())
            {
            // prefetch neighbor index
            const size_t head_idx = d_head_list[idx];
            unsigned int cur_j = 0;
            unsigned int next_j(0);
            size_t my_head = d_head_list[idx];

            next_j = threadIdx.x % tpp < n_neigh ? __ldg(d_nlist + my_head + threadIdx.x % tpp) : 0;

            // loop over neighbors in strided way
            for (int neigh_idx = threadIdx.x % tpp; neigh_idx < n_neigh; neigh_idx += tpp)
                {
                // read the current neighbor index (MEM TRANSFER: 4 bytes)
                // prefetch the next value and set the current one
                cur_j = next_j;
                if (neigh_idx + tpp < n_neigh)
                    {
                    next_j = __ldg(d_nlist + head_idx + neigh_idx + tpp);
                    }

                // read the position of j (MEM TRANSFER: 16 bytes)
                Scalar4 postypej = __ldg(d_pos + cur_j);
                Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

                // initialize the force on j
                Scalar4 forcej = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

                // compute r_ij (FLOPS: 3)
                Scalar3 dxij = posi - posj;

                // apply periodic boundary conditions (FLOPS: 12)
                dxij = box.minImage(dxij);

                // compute rij_sq (FLOPS: 5)
                Scalar rij_sq = dot(dxij, dxij);

                // access the per type-pair parameters
                unsigned int typpair
                    = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypej.w));
                Scalar rcutsq = s_rcutsq[typpair];
                const typename EVALUATOR_CLASS::param_type param = s_params[typpair];

                EVALUATOR_CLASS eval(rij_sq, rcutsq, param);
                Scalar phi_j = s_phi_ab[threadIdx.x * ntypes + __scalar_as_int(postypej.w)];
                eval.evalPhi(phi_j);
                s_phi_ab[threadIdx.x * ntypes + __scalar_as_int(postypej.w)] = phi_j;
                }

            // self-energy
            for (unsigned int typ_b = 0; typ_b < ntypes; ++typ_b)
                {
                Scalar phi = s_phi_ab[threadIdx.x * ntypes + typ_b];

                // reduce in warp
                phi = detail::warp_reduce_sum(phi, hoomd_lane, tpp);

                // broadcast into shared mem
                s_phi_ab[threadIdx.x * ntypes + typ_b]
                    = detail::warp_broadcast(phi, hoomd_lane, tpp, 0);

                if (threadIdx.x % tpp == 0)
                    {
                    unsigned int typpair = typpair_idx(__scalar_as_int(postypei.w), typ_b);
                    Scalar rcutsq = s_rcutsq[typpair];
                    const typename EVALUATOR_CLASS::param_type param = s_params[typpair];

                    EVALUATOR_CLASS eval(Scalar(0.0), rcutsq, param);
                    Scalar energy(0.0);

                    eval.evalSelfEnergy(energy, phi);
                    forcei.w += energy;
                    }
                }
            }

        // prefetch neighbor index
        const size_t head_idx = d_head_list[idx];
        unsigned int cur_j = 0;
        unsigned int next_j(0);
        size_t my_head = d_head_list[idx];

        next_j = threadIdx.x % tpp < n_neigh ? __ldg(d_nlist + my_head + threadIdx.x % tpp) : 0;

        // loop over neighbors in strided way
        for (int neigh_idx = threadIdx.x % tpp; neigh_idx < n_neigh; neigh_idx += tpp)
            {
            // read the current neighbor index (MEM TRANSFER: 4 bytes)
            // prefetch the next value and set the current one
            cur_j = next_j;
            if (neigh_idx + tpp < n_neigh)
                {
                next_j = __ldg(d_nlist + head_idx + neigh_idx + tpp);
                }

            // read the position of j (MEM TRANSFER: 16 bytes)
            Scalar4 postypej = __ldg(d_pos + cur_j);
            Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

            // initialize the force on j
            Scalar4 forcej = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

            Scalar virialj_xx(0.0);
            Scalar virialj_xy(0.0);
            Scalar virialj_xz(0.0);
            Scalar virialj_yy(0.0);
            Scalar virialj_yz(0.0);
            Scalar virialj_zz(0.0);

            // compute r_ij (FLOPS: 3)
            Scalar3 dxij = posi - posj;

            // apply periodic boundary conditions (FLOPS: 12)
            dxij = box.minImage(dxij);

            // compute rij_sq (FLOPS: 5)
            Scalar rij_sq = dot(dxij, dxij);

            // access the per type-pair parameters
            unsigned int typpair
                = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypej.w));
            Scalar rcutsq = s_rcutsq[typpair];
            const typename EVALUATOR_CLASS::param_type param = s_params[typpair];

            // compute the base repulsive and attractive terms of the potential
            Scalar fR = Scalar(0.0);
            Scalar fA = Scalar(0.0);
            EVALUATOR_CLASS eval(rij_sq, rcutsq, param);
            bool evaluatedij = eval.evalRepulsiveAndAttractive(fR, fA);

            if (evaluatedij)
                {
                Scalar chi = Scalar(0.0);

                if (EVALUATOR_CLASS::needsChi())
                    {
                    // compute chi
                    unsigned int cur_k = 0;
                    unsigned int next_k(0);
                    next_k = __ldg(d_nlist + head_idx);

                    // loop over neighbors one by one
                    for (int neigh_idy = 0; neigh_idy < n_neigh; neigh_idy++)
                        {
                        // read the current index of k and prefetch the next one
                        cur_k = next_k;
                        if (neigh_idy + 1 < n_neigh)
                        next_k = __ldg(d_nlist + head_idx + neigh_idy + 1);

                        // get the position of neighbor k
                        Scalar4 postypek = __ldg(d_pos + cur_k);
                        Scalar3 posk = make_scalar3(postypek.x, postypek.y, postypek.z);

                        // get the type pair parameters for i and k
                        typpair
                            = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypek.w));
                        Scalar temp_rcutsq = s_rcutsq[typpair];
                        typename EVALUATOR_CLASS::param_type temp_param = s_params[typpair];

                        EVALUATOR_CLASS temp_eval(rij_sq, temp_rcutsq, temp_param);
                        bool temp_evaluated = temp_eval.areInteractive();

                        if (cur_k != cur_j && temp_evaluated)
                            {
                            // compute rik
                            Scalar3 dxik = posi - posk;

                            // apply the periodic boundary conditions
                            dxik = box.minImage(dxik);

                            // compute rik_sq
                            Scalar rik_sq = dot(dxik, dxik);

                            // compute the bond angle (if needed)
                            Scalar cos_th = Scalar(0.0);
                            if (EVALUATOR_CLASS::needsAngle())
                                cos_th = dot(dxij, dxik) * fast::rsqrt(rij_sq * rik_sq);
                            else
                                cos_th += Scalar(1.0); // shuts up the compiler warning

                            // set up the EVALUATOR_CLASS
                            eval.setRik(rik_sq);
                            if (EVALUATOR_CLASS::needsAngle())
                                eval.setAngle(cos_th);

                            // compute the partial chi term
                            eval.evalChi(chi);
                            }
                        }
                    }
                // evaluate the force and energy from the ij interaction
                Scalar force_divr = Scalar(0.0);
                Scalar potential_eng = Scalar(0.0);
                Scalar bij = Scalar(0.0);
                const Scalar phi = s_phi_ab[threadIdx.x * ntypes + __scalar_as_int(postypej.w)];
                eval.evalForceij(fR, fA, chi, phi, bij, force_divr, potential_eng);

                // add the forces and energies to their respective particles
                Scalar2 v_coeffs = make_scalar2(Scalar(1.0 / 6.0) * rij_sq, Scalar(0.0));
                forcei.x += dxij.x * force_divr;
                forcei.y += dxij.y * force_divr;
                forcei.z += dxij.z * force_divr;

                if (compute_virial)
                    {
                    Scalar force_div2r = Scalar(0.5) * force_divr;
                    viriali_xx += force_div2r * dxij.x * dxij.x;
                    viriali_xy += force_div2r * dxij.x * dxij.y;
                    viriali_xz += force_div2r * dxij.x * dxij.z;
                    viriali_yy += force_div2r * dxij.y * dxij.y;
                    viriali_yz += force_div2r * dxij.y * dxij.z;
                    viriali_zz += force_div2r * dxij.z * dxij.z;
                    }

                forcej.x -= dxij.x * force_divr;
                forcej.y -= dxij.y * force_divr;
                forcej.z -= dxij.z * force_divr;

                if (compute_virial)
                    {
                    Scalar force_div2r = Scalar(0.5) * force_divr;
                    virialj_xx += force_div2r * dxij.x * dxij.x;
                    virialj_xy += force_div2r * dxij.x * dxij.y;
                    virialj_xz += force_div2r * dxij.x * dxij.z;
                    virialj_yy += force_div2r * dxij.y * dxij.y;
                    virialj_yz += force_div2r * dxij.y * dxij.z;
                    virialj_zz += force_div2r * dxij.z * dxij.z;
                    }

                // potential energy of j must be halved
                forcej.w += Scalar(0.5) * potential_eng;

                // potential energy per particle must be halved
                forcei.w += Scalar(0.5) * potential_eng;

                if (EVALUATOR_CLASS::hasIkForce())
                    {
                    // now evaluate the force from the ik interactions
                    unsigned int cur_k = 0;
                    unsigned int next_k(0);
                    next_k = __ldg(d_nlist + head_idx);

                    // loop over neighbors one by one
                    for (int neigh_idy = 0; neigh_idy < n_neigh; neigh_idy++)
                        {
                        // read the current neighbor index and prefetch the next one
                        cur_k = next_k;
                        if (neigh_idy + 1 < n_neigh)
                        next_k = __ldg(d_nlist + head_idx + neigh_idy + 1);

                        // get the position of neighbor k
                        Scalar4 postypek = __ldg(d_pos + cur_k);
                        Scalar3 posk = make_scalar3(postypek.x, postypek.y, postypek.z);

                        // get the type pair parameters for i and k
                        typpair
                            = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypek.w));
                        Scalar temp_rcutsq = s_rcutsq[typpair];
                        typename EVALUATOR_CLASS::param_type temp_param = s_params[typpair];

                        EVALUATOR_CLASS temp_eval(rij_sq, temp_rcutsq, temp_param);
                        bool temp_evaluated = temp_eval.areInteractive();

                        if (cur_k != cur_j && temp_evaluated)
                            {
                            Scalar4 forcek
                                = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

                            // compute rik
                            Scalar3 dxik = posi - posk;

                            // apply the periodic boundary conditions
                            dxik = box.minImage(dxik);

                            // compute rik_sq
                            Scalar rik_sq = dot(dxik, dxik);

                            // compute the bond angle (if needed)
                            Scalar cos_th = Scalar(0.0);
                            if (EVALUATOR_CLASS::needsAngle())
                                cos_th = dot(dxij, dxik) * fast::rsqrt(rij_sq * rik_sq);
                            else
                                cos_th += Scalar(1.0); // shuts up the compiler warning

                            // set up the EVALUATOR_CLASS
                            eval.setRik(rik_sq);
                            if (EVALUATOR_CLASS::needsAngle())
                                eval.setAngle(cos_th);

                            // compute the force
                            Scalar3 force_divr_ij
                                = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
                            Scalar3 force_divr_ik
                                = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
                            bool evaluatedjk
                                = eval.evalForceik(fR, fA, chi, bij, force_divr_ij, force_divr_ik);

                            if (evaluatedjk)
                                {
                                // add the forces to their respective particles
                                v_coeffs.y = Scalar(1.0 / 6.0) * rik_sq;
                                forcei.x += force_divr_ij.x * dxij.x + force_divr_ik.x * dxik.x;
                                forcei.y += force_divr_ij.x * dxij.y + force_divr_ik.x * dxik.y;
                                forcei.z += force_divr_ij.x * dxij.z + force_divr_ik.x * dxik.z;

                                // NOTE: virial for ik forces not tested
                                if (compute_virial)
                                    {
                                    Scalar force_div2r_ij = Scalar(0.5) * force_divr_ij.x;
                                    Scalar force_div2r_ik = Scalar(0.5) * force_divr_ik.x;
                                    viriali_xx += force_div2r_ij * dxij.x * dxij.x
                                                  + force_div2r_ik * dxik.x * dxik.x;
                                    viriali_xy += force_div2r_ij * dxij.x * dxij.y
                                                  + force_div2r_ik * dxik.x * dxik.y;
                                    viriali_xz += force_div2r_ij * dxij.x * dxij.z
                                                  + force_div2r_ik * dxik.x * dxik.z;
                                    viriali_yy += force_div2r_ij * dxij.y * dxij.y
                                                  + force_div2r_ik * dxik.y * dxik.y;
                                    viriali_yz += force_div2r_ij * dxij.y * dxij.z
                                                  + force_div2r_ik * dxik.y * dxik.z;
                                    viriali_zz += force_div2r_ij * dxij.z * dxij.z
                                                  + force_div2r_ik * dxik.z * dxik.z;
                                    }

                                forcej.x += force_divr_ij.y * dxij.x + force_divr_ik.y * dxik.x;
                                forcej.y += force_divr_ij.y * dxij.y + force_divr_ik.y * dxik.y;
                                forcej.z += force_divr_ij.y * dxij.z + force_divr_ik.y * dxik.z;

                                if (compute_virial)
                                    {
                                    Scalar force_div2r_ij = Scalar(0.5) * force_divr_ij.y;
                                    Scalar force_div2r_ik = Scalar(0.5) * force_divr_ik.y;
                                    virialj_xx += force_div2r_ij * dxij.x * dxij.x
                                                  + force_div2r_ik * dxik.x * dxik.x;
                                    virialj_xy += force_div2r_ij * dxij.x * dxij.y
                                                  + force_div2r_ik * dxik.x * dxik.y;
                                    virialj_xz += force_div2r_ij * dxij.x * dxij.z
                                                  + force_div2r_ik * dxik.x * dxik.z;
                                    virialj_yy += force_div2r_ij * dxij.y * dxij.y
                                                  + force_div2r_ik * dxik.y * dxik.y;
                                    virialj_yz += force_div2r_ij * dxij.y * dxij.z
                                                  + force_div2r_ik * dxik.y * dxik.z;
                                    virialj_zz += force_div2r_ij * dxij.z * dxij.z
                                                  + force_div2r_ik * dxik.z * dxik.z;
                                    }

                                forcek.x += force_divr_ij.z * dxij.x + force_divr_ik.z * dxik.x;
                                forcek.y += force_divr_ij.z * dxij.y + force_divr_ik.z * dxik.y;
                                forcek.z += force_divr_ij.z * dxij.z + force_divr_ik.z * dxik.z;

                                atomicAdd((device Scalar*)(d_force + cur_k) + 0, forcek.x);
                                atomicAdd((device Scalar*)(d_force + cur_k) + 1, forcek.y);
                                atomicAdd((device Scalar*)(d_force + cur_k) + 2, forcek.z);

                                if (compute_virial)
                                    {
                                    Scalar force_div2r_ij = Scalar(0.5) * force_divr_ij.z;
                                    Scalar force_div2r_ik = Scalar(0.5) * force_divr_ik.z;
                                    atomicAdd(&d_virial[0 * virial_pitch + cur_k],
                                                force_div2r_ij * dxij.x * dxij.x
                                                    + force_div2r_ik * dxik.x * dxik.x);
                                    atomicAdd(&d_virial[1 * virial_pitch + cur_k],
                                                force_div2r_ij * dxij.x * dxij.y
                                                    + force_div2r_ik * dxik.x * dxik.y);
                                    atomicAdd(&d_virial[2 * virial_pitch + cur_k],
                                                force_div2r_ij * dxij.x * dxij.z
                                                    + force_div2r_ik * dxik.x * dxik.z);
                                    atomicAdd(&d_virial[3 * virial_pitch + cur_k],
                                                force_div2r_ij * dxij.y * dxij.y
                                                    + force_div2r_ik * dxik.y * dxik.y);
                                    atomicAdd(&d_virial[4 * virial_pitch + cur_k],
                                                force_div2r_ij * dxij.y * dxij.z
                                                    + force_div2r_ik * dxik.y * dxik.z);
                                    atomicAdd(&d_virial[5 * virial_pitch + cur_k],
                                                force_div2r_ij * dxij.z * dxij.z
                                                    + force_div2r_ik * dxik.z * dxik.z);
                                    }
                                }
                            }
                        }
                    }

                // write out the result for particle j
                atomicAdd((device Scalar*)(d_force + cur_j) + 0, forcej.x);
                atomicAdd((device Scalar*)(d_force + cur_j) + 1, forcej.y);
                atomicAdd((device Scalar*)(d_force + cur_j) + 2, forcej.z);
                atomicAdd((device Scalar*)(d_force + cur_j) + 3, forcej.w);

                if (compute_virial)
                    {
                    atomicAdd(&d_virial[0 * virial_pitch + cur_j], virialj_xx);
                    atomicAdd(&d_virial[1 * virial_pitch + cur_j], virialj_xy);
                    atomicAdd(&d_virial[2 * virial_pitch + cur_j], virialj_xz);
                    atomicAdd(&d_virial[3 * virial_pitch + cur_j], virialj_yy);
                    atomicAdd(&d_virial[4 * virial_pitch + cur_j], virialj_yz);
                    atomicAdd(&d_virial[5 * virial_pitch + cur_j], virialj_zz);
                    }
                }
            }
        // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
        atomicAdd((device Scalar*)(d_force + idx) + 0, forcei.x);
        atomicAdd((device Scalar*)(d_force + idx) + 1, forcei.y);
        atomicAdd((device Scalar*)(d_force + idx) + 2, forcei.z);
        atomicAdd((device Scalar*)(d_force + idx) + 3, forcei.w);

        if (compute_virial)
            {
            atomicAdd(&d_virial[0 * virial_pitch + idx], viriali_xx);
            atomicAdd(&d_virial[1 * virial_pitch + idx], viriali_xy);
            atomicAdd(&d_virial[2 * virial_pitch + idx], viriali_xz);
            atomicAdd(&d_virial[3 * virial_pitch + idx], viriali_yy);
            atomicAdd(&d_virial[4 * virial_pitch + idx], viriali_yz);
            atomicAdd(&d_virial[5 * virial_pitch + idx], viriali_zz);
            }
        }
    }

#else

namespace kernel
    {
//! Kernel driver that computes the three-body forces
template<>
__attribute__((visibility("default"))) hipError_t
gpu_compute_triplet_forces<EVALUATOR_CLASS>(const tersoff_args_t& pair_args,
                                            const typename EVALUATOR_CLASS::param_type* d_params)
    {
    typedef typename EVALUATOR_CLASS::param_type param_type;
    const char* name = "hoomd::md::gpu_compute_triplet_forces_kernel";
    std::initializer_list<mtl::FunctionConstant> constants
        = {mtl::FunctionConstant::make_bool(0, pair_args.compute_virial),
           mtl::FunctionConstant::make_uint(1, pair_args.tpp)};

    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name, constants);
    max_block_size &= ~(pair_args.devprop.warpSize - 1);
    unsigned int run_block_size = std::min(pair_args.block_size, max_block_size);

    // size shared bytes
    Index2D typpair_idx(pair_args.ntypes);
    auto shared_size = [&](unsigned int block_size)
    {
        return (sizeof(Scalar) + sizeof(param_type)) * typpair_idx.getNumElements()
               + pair_args.ntypes * block_size * sizeof(Scalar);
    };
    size_t shared_bytes = shared_size(run_block_size);
    while (shared_bytes >= pair_args.devprop.sharedMemPerBlock
           && run_block_size > (unsigned int)pair_args.devprop.warpSize)
        {
        run_block_size -= pair_args.devprop.warpSize;
        shared_bytes = shared_size(run_block_size);
        }

    if (shared_bytes > pair_args.devprop.sharedMemPerBlock)
        {
        throw std::runtime_error("Triplet potential parameters exceed the available shared "
                                 "memory per block.");
        }

    // zero the forces
    hipMemsetAsync(pair_args.d_force, 0, sizeof(Scalar4) * (pair_args.N + pair_args.Nghosts));
    hipMemsetAsync(pair_args.d_virial, 0, sizeof(Scalar) * pair_args.virial_pitch * 6);

    kernel_args::gpu_compute_triplet_forces_kernel_args args;
    args.d_force = pair_args.d_force;
    args.N = pair_args.N;
    args.d_virial = pair_args.d_virial;
    args.virial_pitch = pair_args.virial_pitch;
    args.d_pos = pair_args.d_pos;
    args.box = pair_args.box;
    args.d_n_neigh = pair_args.d_n_neigh;
    args.d_nlist = pair_args.d_nlist;
    args.d_head_list = pair_args.d_head_list;
    args.d_params = d_params;
    args.d_rcutsq = pair_args.d_rcutsq;
    args.ntypes = pair_args.ntypes;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(pair_args.N / (run_block_size / pair_args.tpp) + 1),
                       dim3(run_block_size),
                       args,
                       shared_bytes,
                       constants);
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
