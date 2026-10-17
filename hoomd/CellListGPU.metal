// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file CellListGPU.metal
    \brief Metal port of CellListGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifndef __METAL_VERSION__
#include "CellListGPU.cuh"
#include "hoomd/mtl/Primitives.h"
#endif

namespace hoomd
    {
namespace kernel_args
    {
struct compute_cell_list_args
    {
    HOOMD_DEVPTR(unsigned int) d_cell_size;
    HOOMD_DEVPTR(Scalar4) d_xyzf;
    HOOMD_DEVPTR(uint2) d_type_body;
    HOOMD_DEVPTR(Scalar4) d_cell_orientation;
    HOOMD_DEVPTR(unsigned int) d_cell_idx;
    HOOMD_DEVPTR(unsigned int) d_conditions; //!< uint3 as 3 words
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar4) d_orientation;
    HOOMD_DEVPTR(const Scalar) d_charge;
    HOOMD_DEVPTR(const unsigned int) d_body;
    BoxDim box;
    Index3D ci;
    Index2D cli;
    Scalar3 ghost_width;
    unsigned int N;
    unsigned int Nmax;
    unsigned int nwork;
    unsigned int flag_charge;
    unsigned int flag_type;
    };

struct fill_indices_args
    {
    HOOMD_DEVPTR(unsigned int) d_cell_key;
    HOOMD_DEVPTR(unsigned int) d_pidx_key;
    HOOMD_DEVPTR(unsigned int) d_sort_permutation;
    HOOMD_DEVPTR(const unsigned int) d_cell_idx;
    HOOMD_DEVPTR(const unsigned int) d_cell_size;
    Index3D ci;
    Index2D cli;
    unsigned int cl_size;
    };

struct apply_sorted_order_args
    {
    HOOMD_DEVPTR(const unsigned int) d_cell_idx;
    HOOMD_DEVPTR(unsigned int) d_cell_idx_new;
    HOOMD_DEVPTR(const Scalar4) d_xyzf;
    HOOMD_DEVPTR(Scalar4) d_xyzf_new;
    HOOMD_DEVPTR(const uint2) d_type_body;
    HOOMD_DEVPTR(uint2) d_type_body_new;
    HOOMD_DEVPTR(const Scalar4) d_cell_orientation;
    HOOMD_DEVPTR(Scalar4) d_cell_orientation_new;
    HOOMD_DEVPTR(const unsigned int) d_sort_permutation;
    unsigned int cl_size;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

/*! Compute the cell list. Each thread places one particle (or ghost) in its cell.
    d_conditions flags: x: cell overflow (max cell size + 1), y: NaN position (index + 1), z: out
    of the box (index + 1).
*/
kernel void gpu_compute_cell_list_kernel(HOOMD_KERNEL_PARAMS(kernel_args::compute_cell_list_args))
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= args.nwork)
        return;

    BoxDim box = args.box;
    Index3D ci = args.ci;
    Index2D cli = args.cli;

    Scalar4 postype = args.d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    Scalar flag = 0;
    unsigned int body = 0;
    Scalar type = postype.w;
    Scalar4 orientation = make_scalar4(0, 0, 0, 0);
    if (args.d_type_body != nullptr)
        body = args.d_body[idx];
    if (args.d_cell_orientation != nullptr)
        orientation = args.d_orientation[idx];

    if (args.flag_charge)
        flag = args.d_charge[idx];
    else if (args.flag_type)
        flag = type;
    else
        flag = __int_as_scalar(idx);

    if (isnan(pos.x) || isnan(pos.y) || isnan(pos.z))
        {
        args.d_conditions[1] = idx + 1;
        return;
        }

    uchar3 periodic = box.getPeriodic();
    Scalar3 ghost_width = args.ghost_width;
    Scalar3 f = box.makeFraction(pos, ghost_width);

    // check if the particle is inside the unit cell + ghost layer in all dimensions
    if ((f.x < Scalar(-0.00001) || f.x >= Scalar(1.00001))
        || (f.y < Scalar(-0.00001) || f.y >= Scalar(1.00001))
        || (f.z < Scalar(-0.00001) || f.z >= Scalar(1.00001)))
        {
        // if a ghost particle is out of bounds, silently ignore it
        if (idx < args.N)
            args.d_conditions[2] = idx + 1;
        return;
        }

    int ib = (int)(f.x * ci.getW());
    int jb = (int)(f.y * ci.getH());
    int kb = (int)(f.z * ci.getD());

    // need to handle the case where the particle is exactly at the box hi
    if (ib == (int)ci.getW() && periodic.x)
        ib = 0;
    if (jb == (int)ci.getH() && periodic.y)
        jb = 0;
    if (kb == (int)ci.getD() && periodic.z)
        kb = 0;

    // identify the bin
    if (ib < 0 || ib >= (int)ci.getW() || jb < 0 || jb >= (int)ci.getH() || kb < 0
        || kb >= (int)ci.getD())
        {
        if (idx < args.N)
            atomicMax(&args.d_conditions[2], idx + 1);
        return;
        }
    unsigned int bin = ci(ib, jb, kb);

    unsigned int size = atomicInc(&args.d_cell_size[bin], 0xffffffffu);
    if (size < args.Nmax)
        {
        unsigned int write_pos = cli(size, bin);
        if (args.d_xyzf != nullptr)
            args.d_xyzf[write_pos] = make_scalar4(pos.x, pos.y, pos.z, flag);
        if (args.d_type_body != nullptr)
            args.d_type_body[write_pos] = make_uint2(__scalar_as_int(type), body);
        if (args.d_cell_orientation != nullptr)
            args.d_cell_orientation[write_pos] = orientation;
        if (args.d_cell_idx != nullptr)
            args.d_cell_idx[write_pos] = idx;
        }
    else
        {
        // handle overflow
        atomicMax(&args.d_conditions[0], size + 1);
        }
    }

//! Build the (cell, particle index) sort keys and the identity permutation
kernel void gpu_fill_indices_kernel(HOOMD_KERNEL_PARAMS(kernel_args::fill_indices_args))
    {
    unsigned int cell_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell_idx >= args.cl_size)
        return;

    Index3D ci = args.ci;
    Index2D cli = args.cli;
    unsigned int icell = cell_idx / cli.getW();
    unsigned int pidx = 0xffffffffu;

    if (icell < ci.getNumElements())
        {
        unsigned int my_cell_size = args.d_cell_size[icell];
        unsigned int ilocal = cell_idx % cli.getW();
        if (ilocal < my_cell_size)
            pidx = args.d_cell_idx[cell_idx];
        }

    args.d_cell_key[cell_idx] = icell;
    args.d_pidx_key[cell_idx] = pidx;
    args.d_sort_permutation[cell_idx] = cell_idx;
    }

kernel void gpu_apply_sorted_cell_list_order(
    HOOMD_KERNEL_PARAMS(kernel_args::apply_sorted_order_args))
    {
    unsigned int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= args.cl_size)
        return;

    unsigned int perm_idx = args.d_sort_permutation[cell_idx];
    if (args.d_xyzf != nullptr)
        args.d_xyzf_new[cell_idx] = args.d_xyzf[perm_idx];
    if (args.d_cell_idx != nullptr)
        args.d_cell_idx_new[cell_idx] = args.d_cell_idx[perm_idx];
    if (args.d_type_body != nullptr)
        args.d_type_body_new[cell_idx] = args.d_type_body[perm_idx];
    if (args.d_cell_orientation != nullptr)
        args.d_cell_orientation_new[cell_idx] = args.d_cell_orientation[perm_idx];
    }

#else

void gpu_compute_cell_list(unsigned int* d_cell_size,
                           Scalar4* d_xyzf,
                           uint2* d_type_body,
                           Scalar4* d_cell_orientation,
                           unsigned int* d_cell_idx,
                           uint3* d_conditions,
                           const Scalar4* d_pos,
                           const Scalar4* d_orientation,
                           const Scalar* d_charge,
                           const Scalar* d_diameter,
                           const unsigned int* d_body,
                           const unsigned int N,
                           const unsigned int n_ghost,
                           const unsigned int Nmax,
                           const bool flag_charge,
                           const bool flag_type,
                           const BoxDim& box,
                           const Index3D& ci,
                           const Index2D& cli,
                           const Scalar3& ghost_width,
                           const unsigned int block_size)
    {
    const char* name = "hoomd::gpu_compute_cell_list_kernel";
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name);

    kernel_args::compute_cell_list_args args;
    args.d_cell_size = d_cell_size;
    args.d_xyzf = d_xyzf;
    args.d_type_body = d_type_body;
    args.d_cell_orientation = d_cell_orientation;
    args.d_cell_idx = d_cell_idx;
    args.d_conditions = reinterpret_cast<unsigned int*>(d_conditions);
    args.d_pos = d_pos;
    args.d_orientation = d_orientation;
    args.d_charge = d_charge;
    args.d_body = d_body;
    args.box = box;
    args.ci = ci;
    args.cli = cli;
    args.ghost_width = ghost_width;
    args.N = N;
    args.Nmax = Nmax;
    args.nwork = N + n_ghost;
    args.flag_charge = flag_charge;
    args.flag_type = flag_type;

    unsigned int run_block_size = std::min(block_size, max_block_size);
    unsigned int n_blocks = args.nwork / run_block_size + 1;
    mtl::launch(HOOMD_MTL_SOURCE, name, dim3(n_blocks), dim3(run_block_size), args);
    }

/*! Sort the cell list contents into lexicographical order of (cell index, particle index). Two
    stable radix sort passes, first by particle index and then by cell index, produce the
    permutation.
*/
hipError_t gpu_sort_cell_list(unsigned int* d_cell_size,
                              Scalar4* d_xyzf,
                              Scalar4* d_xyzf_new,
                              uint2* d_type_body,
                              uint2* d_type_body_new,
                              Scalar4* d_cell_orientation,
                              Scalar4* d_cell_orientation_new,
                              unsigned int* d_cell_idx,
                              unsigned int* d_cell_idx_new,
                              uint2* d_sort_idx,
                              unsigned int* d_sort_permutation,
                              const Index3D ci,
                              const Index2D cli)
    {
    const unsigned int cl_size = cli.getNumElements();
    const unsigned int block_size = 256;
    dim3 grid(cl_size / block_size + 1);

    mtl::ScratchScope scratch;
    unsigned int* d_cell_key = scratch.allocate<unsigned int>(cl_size);
    unsigned int* d_pidx_key = scratch.allocate<unsigned int>(cl_size);
    unsigned int* d_sorted_cell_key = scratch.allocate<unsigned int>(cl_size);

    kernel_args::fill_indices_args fill;
    fill.d_cell_key = d_cell_key;
    fill.d_pidx_key = d_pidx_key;
    fill.d_sort_permutation = d_sort_permutation;
    fill.d_cell_idx = d_cell_idx;
    fill.d_cell_size = d_cell_size;
    fill.ci = ci;
    fill.cli = cli;
    fill.cl_size = cl_size;
    mtl::launch(HOOMD_MTL_SOURCE, "hoomd::gpu_fill_indices_kernel", grid, dim3(block_size), fill);

    // secondary key first, then a stable sort on the primary key
    mtl::sort_pairs(d_pidx_key, d_sort_permutation, cl_size);
    mtl::gather(d_sort_permutation, d_cell_key, d_sorted_cell_key, cl_size, sizeof(unsigned int));
    mtl::sort_pairs(d_sorted_cell_key, d_sort_permutation, cl_size);

    kernel_args::apply_sorted_order_args apply;
    apply.d_cell_idx = d_cell_idx;
    apply.d_cell_idx_new = d_cell_idx_new;
    apply.d_xyzf = d_xyzf;
    apply.d_xyzf_new = d_xyzf_new;
    apply.d_type_body = d_type_body;
    apply.d_type_body_new = d_type_body_new;
    apply.d_cell_orientation = d_cell_orientation;
    apply.d_cell_orientation_new = d_cell_orientation_new;
    apply.d_sort_permutation = d_sort_permutation;
    apply.cl_size = cl_size;
    mtl::launch(HOOMD_MTL_SOURCE,
                "hoomd::gpu_apply_sorted_cell_list_order",
                grid,
                dim3(block_size),
                apply);

    // copy back permuted arrays to cell list
    if (d_xyzf)
        hipMemcpyAsync(d_xyzf,
                       d_xyzf_new,
                       sizeof(Scalar4) * cl_size,
                       hipMemcpyDeviceToDevice);
    hipMemcpyAsync(d_cell_idx,
                   d_cell_idx_new,
                   sizeof(unsigned int) * cl_size,
                   hipMemcpyDeviceToDevice);
    if (d_type_body)
        hipMemcpyAsync(d_type_body,
                       d_type_body_new,
                       sizeof(uint2) * cl_size,
                       hipMemcpyDeviceToDevice);
    if (d_cell_orientation)
        hipMemcpyAsync(d_cell_orientation,
                       d_cell_orientation_new,
                       sizeof(Scalar4) * cl_size,
                       hipMemcpyDeviceToDevice);
    return hipSuccess;
    }

#endif

    } // end namespace hoomd
