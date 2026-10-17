// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ForceDistanceConstraintGPU.metal
    \brief Metal port of ForceDistanceConstraintGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifndef __METAL_VERSION__
#include "ForceDistanceConstraintGPU.cuh"
#else
// The constraint matrix and vectors are double precision arrays. Apple GPUs compute in single
// precision and read and write the arrays through hoomd_double.
#define double hoomd_double
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct gpu_fill_matrix_vector_kernel_args
    {
    unsigned int n_constraint;
    unsigned int nptl_local;
    HOOMD_DEVPTR(double) d_matrix;
    HOOMD_DEVPTR(double) d_vec;
    HOOMD_DEVPTR(double) d_csr_val;
    HOOMD_DEVPTR(const int) d_csr_idxlookup;
    HOOMD_DEVPTR(unsigned int) d_sparsity_pattern_changed;
    Scalar rel_tol;
    HOOMD_DEVPTR(unsigned int) d_constraint_violated;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar4) d_vel;
    HOOMD_DEVPTR(const Scalar4) d_netforce;
    HOOMD_DEVPTR(const group_storage<2>) d_gpu_clist;
    Index2D gpu_clist_indexer;
    HOOMD_DEVPTR(const unsigned int) d_gpu_n_constraints;
    HOOMD_DEVPTR(const unsigned int) d_gpu_cpos;
    HOOMD_DEVPTR(const typeval_union) d_group_typeval;
    Scalar deltaT;
    BoxDim box;
    };

struct gpu_fill_constraint_forces_kernel_args
    {
    unsigned int nptl_local;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const group_storage<2>) d_gpu_clist;
    Index2D gpu_clist_indexer;
    HOOMD_DEVPTR(const unsigned int) d_gpu_n_constraints;
    HOOMD_DEVPTR(const unsigned int) d_gpu_cpos;
    HOOMD_DEVPTR(double) d_lagrange;
    HOOMD_DEVPTR(Scalar4) d_force;
    HOOMD_DEVPTR(Scalar) d_virial;
    size_t virial_pitch;
    BoxDim box;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void gpu_fill_matrix_vector_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_fill_matrix_vector_kernel_args))
    {
    unsigned int n_constraint = args.n_constraint;
    unsigned int nptl_local = args.nptl_local;
    device double* d_matrix = args.d_matrix;
    device double* d_vec = args.d_vec;
    device double* d_csr_val = args.d_csr_val;
    device const int* d_csr_idxlookup = args.d_csr_idxlookup;
    device unsigned int* d_sparsity_pattern_changed = args.d_sparsity_pattern_changed;
    Scalar rel_tol = args.rel_tol;
    device unsigned int* d_constraint_violated = args.d_constraint_violated;
    device const Scalar4* d_pos = args.d_pos;
    device const Scalar4* d_vel = args.d_vel;
    device const Scalar4* d_netforce = args.d_netforce;
    device const group_storage<2>* d_gpu_clist = args.d_gpu_clist;
    const Index2D gpu_clist_indexer = args.gpu_clist_indexer;
    device const unsigned int* d_gpu_n_constraints = args.d_gpu_n_constraints;
    device const unsigned int* d_gpu_cpos = args.d_gpu_cpos;
    device const typeval_union* d_group_typeval = args.d_group_typeval;
    Scalar deltaT = args.deltaT;
    const BoxDim box = args.box;

    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // if beyond the number of local ptls, exit early
    if (idx >= nptl_local)
        return;

    // load number of constraints per this ptl
    unsigned int n_constraint_ptl = d_gpu_n_constraints[idx];

    // small O(N^2) loop over ptl connectivity
    for (unsigned int cidx_i = 0; cidx_i < n_constraint_ptl; cidx_i++)
        {
        group_storage<2> cur_constraint_i = d_gpu_clist[gpu_clist_indexer(idx, cidx_i)];

        // the other ptl in the constraint
        unsigned int cur_constraint_idx_i = cur_constraint_i.idx[0];

        // constraint index
        unsigned int n = cur_constraint_i.idx[1];

        // the constraint distance
        Scalar d = d_group_typeval[n].val;

        // indices of constrained ptls in correct order
        unsigned int idx_na, idx_nb;
        unsigned int cpos = d_gpu_cpos[gpu_clist_indexer(idx, cidx_i)];
        if (cpos == 0)
            {
            idx_na = idx;
            idx_nb = cur_constraint_idx_i;
            }
        else
            {
            idx_na = cur_constraint_idx_i;
            idx_nb = idx;
            }

        // constraint separation
        vec3<Scalar> rn(vec3<Scalar>(d_pos[idx_na]) - vec3<Scalar>(d_pos[idx_nb]));

        // get masses
        Scalar ma = d_vel[idx_na].w;
        Scalar mb = d_vel[idx_nb].w;

        // constraint time derivative
        vec3<Scalar> rndot(vec3<Scalar>(d_vel[idx_na]) - vec3<Scalar>(d_vel[idx_nb]));

        // apply minimum image
        box.minImage(rn, rndot);

        // constraint separation at t+2*deltaT
        vec3<Scalar> qn(rn + deltaT * rndot);

        // load number of constraints per this ptl
        unsigned int n_constraint_i = d_gpu_n_constraints[cur_constraint_idx_i];

        for (unsigned int cidx_j = 0; cidx_j < n_constraint_i; cidx_j++)
            {
            group_storage<2> cur_constraint_j
                = d_gpu_clist[gpu_clist_indexer(cur_constraint_idx_i, cidx_j)];

            // the other ptl in the constraint
            unsigned int cur_constraint_idx_j = cur_constraint_j.idx[0];

            // indices of constrained ptls in correct order
            unsigned int idx_ma, idx_mb;
            if (d_gpu_cpos[gpu_clist_indexer(cur_constraint_idx_i, cidx_j)] == 0)
                {
                idx_ma = cur_constraint_idx_i;
                idx_mb = cur_constraint_idx_j;
                }
            else
                {
                idx_ma = cur_constraint_idx_j;
                idx_mb = cur_constraint_idx_i;
                }

            unsigned int m = cur_constraint_j.idx[1];

            // constraint separation
            vec3<Scalar> rm(vec3<Scalar>(d_pos[idx_ma]) - vec3<Scalar>(d_pos[idx_mb]));

            rm = box.minImage(rm);

            double mat_element(0.0);
            if (idx_na == idx_ma)
                {
                mat_element += double(4.0) * dot(qn, rm) / ma;
                }
            if (idx_na == idx_mb)
                {
                mat_element -= double(4.0) * dot(qn, rm) / ma;
                }
            if (idx_nb == idx_ma)
                {
                mat_element -= double(4.0) * dot(qn, rm) / mb;
                }
            if (idx_nb == idx_mb)
                {
                mat_element += double(4.0) * dot(qn, rm) / mb;
                }

            // write out matrix element in column-major
            d_matrix[m * n_constraint + n] = mat_element;

            // update sparse matrix
            int k = d_csr_idxlookup[m * n_constraint + n];

            if ((k == -1 && mat_element != double(0.0)) || (k != -1 && mat_element == double(0.0)))
                {
                *d_sparsity_pattern_changed = 1;
                }

            if (k != -1)
                {
                d_csr_val[k] = mat_element;
                }
            }

        if (fast::sqrt(dot(rn, rn)) - d >= rel_tol * d || isnan(dot(rn, rn)))
            {
            *d_constraint_violated = n + 1;
            }

        if (cpos == 0 || idx_na >= nptl_local || idx_nb >= nptl_local)
            {
            // fill vector component
            d_vec[n] = (dot(qn, qn) - d * d) / deltaT / deltaT
                       + double(2.0)
                             * dot(qn,
                                   vec3<Scalar>(d_netforce[idx_na]) / ma
                                       - vec3<Scalar>(d_netforce[idx_nb]) / mb);
            }
        }
    }

kernel void gpu_fill_constraint_forces_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_fill_constraint_forces_kernel_args))
    {
    unsigned int nptl_local = args.nptl_local;
    device const Scalar4* d_pos = args.d_pos;
    device const group_storage<2>* d_gpu_clist = args.d_gpu_clist;
    const Index2D gpu_clist_indexer = args.gpu_clist_indexer;
    device const unsigned int* d_gpu_n_constraints = args.d_gpu_n_constraints;
    device const unsigned int* d_gpu_cpos = args.d_gpu_cpos;
    device double* d_lagrange = args.d_lagrange;
    device Scalar4* d_force = args.d_force;
    device Scalar* d_virial = args.d_virial;
    size_t virial_pitch = args.virial_pitch;
    const BoxDim box = args.box;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nptl_local)
        return;

    // load number of constraints per this ptl
    unsigned int n_constraint_ptl = d_gpu_n_constraints[idx];

    // the accumulated force on this ptl
    vec3<Scalar> f(0.0, 0.0, 0.0);

    // accumulated virial tensor components
    Scalar virialxx(0.0);
    Scalar virialxy(0.0);
    Scalar virialxz(0.0);
    Scalar virialyy(0.0);
    Scalar virialyz(0.0);
    Scalar virialzz(0.0);

    // iterate over constraints involving ptl with index idx
    for (unsigned int cidx = 0; cidx < n_constraint_ptl; cidx++)
        {
        group_storage<2> cur_constraint = d_gpu_clist[gpu_clist_indexer(idx, cidx)];

        // the other ptl in the constraint
        unsigned int cur_constraint_idx = cur_constraint.idx[0];

        // group idx
        unsigned int n = cur_constraint.idx[1];

        // position of ptl in constraint
        unsigned int cpos = d_gpu_cpos[gpu_clist_indexer(idx, cidx)];

        // indices of constrained ptls in correct order
        unsigned int idx_na, idx_nb;
        if (cpos == 0)
            {
            idx_na = idx;
            idx_nb = cur_constraint_idx;
            }
        else
            {
            idx_na = cur_constraint_idx;
            idx_nb = idx;
            }

        // constraint separation
        vec3<Scalar> rn(vec3<Scalar>(d_pos[idx_na]) - vec3<Scalar>(d_pos[idx_nb]));

        // apply minimum image
        rn = box.minImage(rn);

        if (idx < nptl_local)
            {
            // write out force
            if (cpos == 0)
                {
                f += -Scalar(2.0) * (Scalar)d_lagrange[n] * rn;
                }
            else
                {
                f += Scalar(2.0) * (Scalar)d_lagrange[n] * rn;
                }

            // virial
            virialxx -= (Scalar)d_lagrange[n] * rn.x * rn.x;
            virialxy -= (Scalar)d_lagrange[n] * rn.x * rn.y;
            virialxz -= (Scalar)d_lagrange[n] * rn.x * rn.z;
            virialyy -= (Scalar)d_lagrange[n] * rn.y * rn.y;
            virialyz -= (Scalar)d_lagrange[n] * rn.y * rn.z;
            virialzz -= (Scalar)d_lagrange[n] * rn.z * rn.z;
            }
        }

    d_force[idx] = make_scalar4(f.x, f.y, f.z, Scalar(0.0));

    d_virial[0 * virial_pitch + idx] = virialxx;
    d_virial[1 * virial_pitch + idx] = virialxy;
    d_virial[2 * virial_pitch + idx] = virialxz;
    d_virial[3 * virial_pitch + idx] = virialyy;
    d_virial[4 * virial_pitch + idx] = virialyz;
    d_virial[5 * virial_pitch + idx] = virialzz;
    }

#else

namespace kernel
    {
//! Kernel to fill the matrix for the linear constraint equation


hipError_t gpu_fill_matrix_vector(unsigned int n_constraint,
                                  unsigned int nptl_local,
                                  double* d_matrix,
                                  double* d_vec,
                                  double* d_csr_val,
                                  const int* d_csr_idxlookup,
                                  unsigned int* d_sparsity_pattern_changed,
                                  Scalar rel_tol,
                                  unsigned int* d_constraint_violated,
                                  const Scalar4* d_pos,
                                  const Scalar4* d_vel,
                                  const Scalar4* d_netforce,
                                  const group_storage<2>* d_gpu_clist,
                                  const Index2D& gpu_clist_indexer,
                                  const unsigned int* d_gpu_n_constraints,
                                  const unsigned int* d_gpu_cpos,
                                  const typeval_union* d_group_typeval,
                                  Scalar deltaT,
                                  const BoxDim box,
                                  unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_fill_matrix_vector_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    // run configuration
    unsigned int run_block_size = min(block_size, max_block_size);
    unsigned int n_blocks = nptl_local / run_block_size + 1;

    // run GPU kernel
    {
        kernel_args::gpu_fill_matrix_vector_kernel_args _args;
        _args.n_constraint = n_constraint;
        _args.nptl_local = nptl_local;
        _args.d_matrix = d_matrix;
        _args.d_vec = d_vec;
        _args.d_csr_val = d_csr_val;
        _args.d_csr_idxlookup = d_csr_idxlookup;
        _args.d_sparsity_pattern_changed = d_sparsity_pattern_changed;
        _args.rel_tol = rel_tol;
        _args.d_constraint_violated = d_constraint_violated;
        _args.d_pos = d_pos;
        _args.d_vel = d_vel;
        _args.d_netforce = d_netforce;
        _args.d_gpu_clist = d_gpu_clist;
        _args.gpu_clist_indexer = gpu_clist_indexer;
        _args.d_gpu_n_constraints = d_gpu_n_constraints;
        _args.d_gpu_cpos = d_gpu_cpos;
        _args.d_group_typeval = d_group_typeval;
        _args.deltaT = deltaT;
        _args.box = box;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_fill_matrix_vector_kernel",
                           dim3(n_blocks),
                           dim3(run_block_size),
                           _args,
                           0);
        }

    return hipSuccess;
    }



#ifdef CUSOLVER_AVAILABLE
hipError_t gpu_count_nnz(unsigned int n_constraint,
                         double* d_matrix,
                         int* d_nnz,
                         int& nnz,
                         cusparseHandle_t cusparse_handle,
                         cusparseMatDescr_t cusparse_mat_descr)
    {
    // count zeros
    cusparseDnnz(cusparse_handle,
                 CUSPARSE_DIRECTION_ROW,
                 n_constraint,
                 n_constraint,
                 cusparse_mat_descr,
                 d_matrix,
                 n_constraint,
                 d_nnz,
                 &nnz);
    return hipSuccess;
    }

#ifndef CUSPARSE_NEW_API
hipError_t gpu_dense2sparse(unsigned int n_constraint,
                            double* d_matrix,
                            int* d_nnz,
                            cusparseHandle_t cusparse_handle,
                            cusparseMatDescr_t cusparse_mat_descr,
                            int* d_csr_rowptr,
                            int* d_csr_colind,
                            double* d_csr_val)
    {
    // convert dense matrix to compressed sparse row

    // update values in CSR format
    cusparseDdense2csr(cusparse_handle,
                       n_constraint,
                       n_constraint,
                       cusparse_mat_descr,
                       d_matrix,
                       n_constraint,
                       d_nnz,
                       d_csr_val,
                       d_csr_rowptr,
                       d_csr_colind);

    return hipSuccess;
    }
#endif
#endif

hipError_t gpu_compute_constraint_forces(const Scalar4* d_pos,
                                         const group_storage<2>* d_gpu_clist,
                                         const Index2D& gpu_clist_indexer,
                                         const unsigned int* d_gpu_n_constraints,
                                         const unsigned int* d_gpu_cpos,
                                         Scalar4* d_force,
                                         Scalar* d_virial,
                                         size_t virial_pitch,
                                         const BoxDim box,
                                         unsigned int nptl_local,
                                         unsigned int block_size,
                                         double* d_lagrange)
    {
    // d_lagrange contains the Lagrange multipliers

    // fill out force array
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_fill_constraint_forces_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    // run configuration
    unsigned int run_block_size = min(block_size, max_block_size);
    unsigned int n_blocks = nptl_local / run_block_size + 1;

    // invoke kernel
    {
        kernel_args::gpu_fill_constraint_forces_kernel_args _args;
        _args.nptl_local = nptl_local;
        _args.d_pos = d_pos;
        _args.d_gpu_clist = d_gpu_clist;
        _args.gpu_clist_indexer = gpu_clist_indexer;
        _args.d_gpu_n_constraints = d_gpu_n_constraints;
        _args.d_gpu_cpos = d_gpu_cpos;
        _args.d_lagrange = d_lagrange;
        _args.d_force = d_force;
        _args.d_virial = d_virial;
        _args.virial_pitch = virial_pitch;
        _args.box = box;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_fill_constraint_forces_kernel",
                           dim3(n_blocks),
                           dim3(run_block_size),
                           _args,
                           0);
        }

    return hipSuccess;
    }

    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
