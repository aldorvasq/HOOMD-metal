// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file NeighborListGPUBinned.metal
    \brief Metal port of NeighborListGPUBinned.cu

    The CUDA template parameters of the neighbor list kernel are Metal function constants:
    0: filter_body, 1: enable_shared_cache, 2: use_index, 3: threads_per_particle.
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifdef __METAL_VERSION__
#include "hoomd/mtl/WarpTools.h"
#else
#include "NeighborListGPUBinned.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct nlist_binned_args
    {
    HOOMD_DEVPTR(unsigned int) d_nlist;
    HOOMD_DEVPTR(unsigned int) d_n_neigh;
    HOOMD_DEVPTR(Scalar4) d_last_updated_pos;
    HOOMD_DEVPTR(unsigned int) d_conditions;
    HOOMD_DEVPTR(const unsigned int) d_Nmax;
    HOOMD_DEVPTR(const size_t) d_head_list;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const unsigned int) d_body;
    HOOMD_DEVPTR(const unsigned int) d_cell_size;
    HOOMD_DEVPTR(const Scalar4) d_cell_xyzf;
    HOOMD_DEVPTR(const unsigned int) d_cell_idx;
    HOOMD_DEVPTR(const uint2) d_cell_type_body;
    HOOMD_DEVPTR(const unsigned int) d_cell_adj;
    HOOMD_DEVPTR(const Scalar) d_r_cut;
    Index3D ci;
    Index2D cli;
    Index2D cadji;
    BoxDim box;
    Scalar3 ghost_width;
    Scalar r_buff;
    unsigned int ntypes;
    unsigned int N;
    unsigned int nwork;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant bool filter_body [[function_constant(0)]];
constant bool enable_shared_cache [[function_constant(1)]];
constant bool use_index [[function_constant(2)]];
constant uint threads_per_particle [[function_constant(3)]];

/*! Build the neighbor list from the cell list. threads_per_particle threads cooperate on each
    particle: they iterate over the adjacent cells' members and compact the neighbors they find
    with a logical warp prefix sum.
*/
kernel void gpu_compute_nlist_binned_kernel(HOOMD_KERNEL_PARAMS(kernel_args::nlist_binned_args))
    {
    const unsigned int tpp = threads_per_particle;
    Index2D typpair_idx(args.ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared data for the cutoff radii per type pair
    threadgroup Scalar* s_r_list = (threadgroup Scalar*)hoomd_dynamic_shared;
    if (enable_shared_cache)
        {
        for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < num_typ_parameters)
                {
                Scalar r_cut = args.d_r_cut[cur_offset + threadIdx.x];
                s_r_list[cur_offset + threadIdx.x]
                    = (r_cut > Scalar(0.0)) ? r_cut + args.r_buff : Scalar(-1.0);
                }
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    // each set of threads_per_particle threads processes one particle
    unsigned int my_pidx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;

    // one logical warp exits together
    if (my_pidx >= args.nwork)
        return;

    BoxDim box = args.box;
    Index3D ci = args.ci;
    Index2D cli = args.cli;
    Index2D cadji = args.cadji;

    Scalar4 my_postype = args.d_pos[my_pidx];
    Scalar3 my_pos = make_scalar3(my_postype.x, my_postype.y, my_postype.z);

    unsigned int my_type = __scalar_as_int(my_postype.w);
    unsigned int my_body = args.d_body[my_pidx];
    size_t my_head = args.d_head_list[my_pidx];

    Scalar3 ghost_width = args.ghost_width;
    Scalar3 f = box.makeFraction(my_pos, ghost_width);

    // find the bin each particle belongs in
    int ib = (int)(f.x * ci.getW());
    int jb = (int)(f.y * ci.getH());
    int kb = (int)(f.z * ci.getD());

    uchar3 periodic = box.getPeriodic();

    // need to handle the case where the particle is exactly at the box hi
    if (ib == (int)ci.getW() && periodic.x)
        ib = 0;
    if (jb == (int)ci.getH() && periodic.y)
        jb = 0;
    if (kb == (int)ci.getD() && periodic.z)
        kb = 0;

    int my_cell = ci(ib, jb, kb);

    // loop over all adjacent bins
    unsigned int cur_adj = 0;
    unsigned int neigh_cell = args.d_cell_adj[cadji(cur_adj, my_cell)];
    unsigned int neigh_size = args.d_cell_size[neigh_cell];

    unsigned int cur_offset = threadIdx.x % tpp;

    bool done = false;

    // total number of neighbors
    unsigned int nneigh = 0;

    unsigned int my_n_max = args.d_Nmax[my_type];

    while (!done)
        {
        // initialize with default
        unsigned int neighbor = 0;
        unsigned int has_neighbor = 0;

        // advance neighbor cell
        while (cur_offset >= neigh_size && !done)
            {
            cur_offset -= neigh_size;
            cur_adj++;
            if (cur_adj >= cadji.getW())
                {
                done = true;
                neigh_size = 0;
                }
            if (!done)
                {
                neigh_cell = args.d_cell_adj[cadji(cur_adj, my_cell)];
                neigh_size = args.d_cell_size[neigh_cell];
                }
            }

        // check for a neighbor if thread is still working
        if (!done)
            {
            Scalar4 cur_xyzf;
            unsigned int j = 0;
            Scalar4 postype_j;
            if (!use_index)
                {
                cur_xyzf = args.d_cell_xyzf[cli(cur_offset, neigh_cell)];
                }
            else
                {
                j = args.d_cell_idx[cli(cur_offset, neigh_cell)];
                postype_j = args.d_pos[j];
                cur_xyzf = make_scalar4(postype_j.x, postype_j.y, postype_j.z, __int_as_scalar(j));
                }

            uint2 cur_type_body;
            if (!use_index)
                cur_type_body = args.d_cell_type_body[cli(cur_offset, neigh_cell)];
            else
                cur_type_body = make_uint2(__scalar_as_int(postype_j.w), args.d_body[j]);

            // advance cur_offset
            cur_offset += tpp;

            unsigned int neigh_type = cur_type_body.x;

            // Only do the hard work if the particle should be included by r_cut(i,j)
            Scalar r_list;
            if (enable_shared_cache)
                {
                r_list = s_r_list[typpair_idx(my_type, neigh_type)];
                }
            else
                {
                Scalar r_cut = args.d_r_cut[typpair_idx(my_type, neigh_type)];
                r_list = (r_cut > Scalar(0.0)) ? r_cut + args.r_buff : Scalar(-1.0);
                }

            if (r_list > Scalar(0.0))
                {
                unsigned int neigh_body = cur_type_body.y;

                Scalar3 neigh_pos = make_scalar3(cur_xyzf.x, cur_xyzf.y, cur_xyzf.z);
                int cur_neigh = __scalar_as_int(cur_xyzf.w);

                // compute the distance between the two particles
                Scalar3 dx = my_pos - neigh_pos;

                // wrap the periodic boundary conditions
                dx = box.minImage(dx);

                // compute dr squared
                Scalar drsq = dot(dx, dx);

                bool excluded = ((int)my_pidx == cur_neigh);

                if (filter_body && my_body != 0xffffffffu)
                    excluded = excluded | (my_body == neigh_body);

                // store result in shared memory
                if (drsq <= r_list * r_list && !excluded)
                    {
                    neighbor = cur_neigh;
                    has_neighbor = 1;
                    }
                }
            }

        // no syncthreads here, we assume tpp <= warp size

        // broadcast whether we are done iterating
        done = detail::warp_broadcast(uint(done), hoomd_lane, tpp, 0) != 0;

        if (!done)
            {
            // scan over flags
            unsigned int n = 0;
            unsigned int k = detail::warp_exclusive_sum(has_neighbor, hoomd_lane, tpp, n);

            if (has_neighbor && (nneigh + k) < my_n_max)
                args.d_nlist[my_head + nneigh + k] = neighbor;

            // increment total neighbor count
            nneigh += n;
            }
        } // end while

    if (threadIdx.x % tpp == 0)
        {
        // flag if we need to grow the neighbor list
        if (nneigh >= my_n_max)
            atomicMax(&args.d_conditions[my_type], nneigh);

        args.d_n_neigh[my_pidx] = nneigh;
        args.d_last_updated_pos[my_pidx] = my_postype;
        }
    }

#else

namespace kernel
    {
hipError_t gpu_compute_nlist_binned(unsigned int* d_nlist,
                                    unsigned int* d_n_neigh,
                                    Scalar4* d_last_updated_pos,
                                    unsigned int* d_conditions,
                                    const unsigned int* d_Nmax,
                                    const size_t* d_head_list,
                                    const Scalar4* d_pos,
                                    const unsigned int* d_body,
                                    const unsigned int N,
                                    const unsigned int* d_cell_size,
                                    const Scalar4* d_cell_xyzf,
                                    const unsigned int* d_cell_idx,
                                    const uint2* d_cell_type_body,
                                    const unsigned int* d_cell_adj,
                                    const Index3D& ci,
                                    const Index2D& cli,
                                    const Index2D& cadji,
                                    const BoxDim& box,
                                    const Scalar* d_r_cut,
                                    const Scalar r_buff,
                                    const unsigned int ntypes,
                                    const unsigned int threads_per_particle,
                                    const unsigned int block_size,
                                    bool filter_body,
                                    const Scalar3& ghost_width,
                                    bool use_index,
                                    const hipDeviceProp_t& devprop)
    {
    // shared memory caches the cutoff of every type pair when it fits
    Index2D typpair_idx(ntypes);
    unsigned int shared_size = (unsigned int)(sizeof(Scalar) * typpair_idx.getNumElements());
    bool enable_shared = true;
    if (shared_size > devprop.sharedMemPerBlock)
        {
        enable_shared = false;
        shared_size = 0;
        }

    std::initializer_list<mtl::FunctionConstant> constants
        = {mtl::FunctionConstant::make_bool(0, filter_body),
           mtl::FunctionConstant::make_bool(1, enable_shared),
           mtl::FunctionConstant::make_bool(2, use_index),
           mtl::FunctionConstant::make_uint(3, threads_per_particle)};

    const char* name = "hoomd::md::gpu_compute_nlist_binned_kernel";
    unsigned int max_block_size
        = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name, constants);
    max_block_size -= max_block_size % max_threads_per_particle;
    unsigned int run_block_size = std::min(block_size, max_block_size);

    kernel_args::nlist_binned_args args;
    args.d_nlist = d_nlist;
    args.d_n_neigh = d_n_neigh;
    args.d_last_updated_pos = d_last_updated_pos;
    args.d_conditions = d_conditions;
    args.d_Nmax = d_Nmax;
    args.d_head_list = d_head_list;
    args.d_pos = d_pos;
    args.d_body = d_body;
    args.d_cell_size = d_cell_size;
    args.d_cell_xyzf = use_index ? nullptr : d_cell_xyzf;
    args.d_cell_idx = use_index ? d_cell_idx : nullptr;
    args.d_cell_type_body = use_index ? nullptr : d_cell_type_body;
    args.d_cell_adj = d_cell_adj;
    args.d_r_cut = d_r_cut;
    args.ci = ci;
    args.cli = cli;
    args.cadji = cadji;
    args.box = box;
    args.ghost_width = ghost_width;
    args.r_buff = r_buff;
    args.ntypes = ntypes;
    args.N = N;
    args.nwork = N;

    dim3 grid(N / (run_block_size / threads_per_particle) + 1);
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       grid,
                       dim3(run_block_size),
                       args,
                       shared_size,
                       constants);
    }
    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
