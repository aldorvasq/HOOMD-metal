// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file NeighborListGPUStencil.metal
    \brief Metal port of NeighborListGPUStencil.cu

    The CUDA template parameters of the neighbor list kernel are Metal function constants:
    0: filter_body, 1: threads_per_particle.
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifdef __METAL_VERSION__
#include "hoomd/mtl/WarpTools.h"
#else
#include "NeighborListGPUStencil.cuh"

#include "hoomd/mtl/Primitives.h"

#include <stdexcept>
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct nlist_stencil_args
    {
    HOOMD_DEVPTR(unsigned int) d_nlist;
    HOOMD_DEVPTR(unsigned int) d_n_neigh;
    HOOMD_DEVPTR(Scalar4) d_last_updated_pos;
    HOOMD_DEVPTR(unsigned int) d_conditions;
    HOOMD_DEVPTR(const unsigned int) d_Nmax;
    HOOMD_DEVPTR(const size_t) d_head_list;
    HOOMD_DEVPTR(const unsigned int) d_pid_map;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const unsigned int) d_body;
    HOOMD_DEVPTR(const unsigned int) d_cell_size;
    HOOMD_DEVPTR(const Scalar4) d_cell_xyzf;
    HOOMD_DEVPTR(const uint2) d_cell_type_body;
    HOOMD_DEVPTR(const Scalar4) d_stencil;
    HOOMD_DEVPTR(const unsigned int) d_n_stencil;
    HOOMD_DEVPTR(const Scalar) d_r_cut;
    Index3D ci;
    Index2D cli;
    Index2D stencil_idx;
    BoxDim box;
    Scalar3 ghost_width;
    Scalar r_buff;
    unsigned int ntypes;
    unsigned int N;
    };

struct nlist_stencil_fill_types_args
    {
    HOOMD_DEVPTR(unsigned int) d_pids;
    HOOMD_DEVPTR(unsigned int) d_types;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    unsigned int N;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant bool filter_body [[function_constant(0)]];
constant uint threads_per_particle [[function_constant(1)]];

/*! Build the neighbor list with per-type cell stencils. See gpu_compute_nlist_stencil_kernel in
    NeighborListGPUStencil.cu.
*/
kernel void gpu_compute_nlist_stencil_kernel(HOOMD_KERNEL_PARAMS(kernel_args::nlist_stencil_args))
    {
    const unsigned int tpp = threads_per_particle;

    // cache the r_listsq parameters into shared memory
    Index2D typpair_idx(args.ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // pointer for the r_listsq data
    threadgroup Scalar* s_r_list = (threadgroup Scalar*)hoomd_dynamic_shared;
    threadgroup unsigned int* s_Nmax
        = (threadgroup unsigned int*)(hoomd_dynamic_shared + sizeof(Scalar) * num_typ_parameters);

    // load in the per type pair r_list
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            Scalar r_cut = args.d_r_cut[cur_offset + threadIdx.x];
            // force the r_list(i,j) to a skippable value if r_cut(i,j) is skippable
            s_r_list[cur_offset + threadIdx.x]
                = (r_cut > Scalar(0.0)) ? r_cut + args.r_buff : Scalar(-1.0);
            }
        if (cur_offset + threadIdx.x < args.ntypes)
            {
            s_Nmax[cur_offset + threadIdx.x] = args.d_Nmax[cur_offset + threadIdx.x];
            }
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // each set of threads_per_particle threads is going to compute the neighbor list for a single
    // particle
    const unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;

    // one logical warp exits together
    if (idx >= args.N)
        return;

    BoxDim box = args.box;
    Index3D ci = args.ci;
    Index2D cli = args.cli;
    Index2D stencil_idx = args.stencil_idx;

    // get the write particle id
    int my_pidx = args.d_pid_map[idx];

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

    // number of available stencils
    unsigned int n_stencil = args.d_n_stencil[my_type];

    // index of current stencil (-1 to initialize)
    int cur_adj = -1;
    Scalar cell_dist2 = 0.0;

    // current cell (0 to initialize)
    unsigned int neigh_cell = 0;

    // size of current cell (0 to initialize)
    unsigned int neigh_size = 0;

    // current index in cell
    unsigned int cur_offset = threadIdx.x % tpp;

    bool done = false;

    // total number of neighbors
    unsigned int nneigh = 0;

    const unsigned int my_n_max = s_Nmax[my_type];

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

            if (cur_adj < (int)n_stencil)
                {
                // compute the stenciled cell cartesian coordinates
                Scalar4 stencil = args.d_stencil[stencil_idx(cur_adj, my_type)];
                int sib = ib + __scalar_as_int(stencil.x);
                int sjb = jb + __scalar_as_int(stencil.y);
                int skb = kb + __scalar_as_int(stencil.z);
                cell_dist2 = stencil.w;

                // wrap through the boundary
                if (sib >= (int)ci.getW() && periodic.x)
                    sib -= ci.getW();
                if (sib < 0 && periodic.x)
                    sib += ci.getW();
                if (sjb >= (int)ci.getH() && periodic.y)
                    sjb -= ci.getH();
                if (sjb < 0 && periodic.y)
                    sjb += ci.getH();
                if (skb >= (int)ci.getD() && periodic.z)
                    skb -= ci.getD();
                if (skb < 0 && periodic.z)
                    skb += ci.getD();

                neigh_cell = ci(sib, sjb, skb);
                neigh_size = args.d_cell_size[neigh_cell];
                }
            else
                {
                // we are past the end of the cell neighbors
                done = true;
                }
            }

        // check for a neighbor if thread is still working
        if (!done)
            {
            // use a do {} while(0) loop to process this particle so we can break for exclusions
            do
                {
                // read in the particle type and body
                const uint2 neigh_type_body = args.d_cell_type_body[cli(cur_offset, neigh_cell)];
                const unsigned int type_j = neigh_type_body.x;
                const unsigned int body_j = neigh_type_body.y;

                // skip any particles belonging to the same rigid body if requested
                if (filter_body && my_body != 0xffffffff && my_body == body_j)
                    break;

                // compute the rlist based on the particle type we're interacting with
                Scalar r_list = s_r_list[typpair_idx(my_type, type_j)];
                if (r_list <= Scalar(0.0))
                    break;
                Scalar r_listsq = r_list * r_list;

                // compare the check distance to the minimum cell distance, and pass without
                // distance check if unnecessary
                if (cell_dist2 > r_listsq)
                    break;

                // only load in the particle position and id if distance check is required
                const Scalar4 neigh_xyzf = args.d_cell_xyzf[cli(cur_offset, neigh_cell)];
                const Scalar3 neigh_pos = make_scalar3(neigh_xyzf.x, neigh_xyzf.y, neigh_xyzf.z);
                unsigned int cur_neigh = __scalar_as_int(neigh_xyzf.w);

                // a particle cannot neighbor itself
                if (my_pidx == (int)cur_neigh)
                    break;

                Scalar3 dx = my_pos - neigh_pos;
                dx = box.minImage(dx);

                Scalar dr_sq = dot(dx, dx);

                if (dr_sq <= r_listsq)
                    {
                    neighbor = cur_neigh;
                    has_neighbor = 1;
                    }
                } while (0); // particle is processed exactly once

            // advance cur_offset
            cur_offset += tpp;
            }

        // now that possible neighbor checks are finished, done (for the cta) depends only on first
        // thread neighbor list only needs to get written into if thread 0 is not done
        done = detail::warp_broadcast(uint(done), hoomd_lane, tpp, 0) != 0;
        if (!done)
            {
            // scan over flags
            unsigned int n = 0;
            unsigned int k = detail::warp_exclusive_sum(has_neighbor, hoomd_lane, tpp, n);

            // write neighbor if it fits in list
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

kernel void gpu_compute_nlist_stencil_fill_types_kernel(
    HOOMD_KERNEL_PARAMS(kernel_args::nlist_stencil_fill_types_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.N)
        return;

    Scalar4 pos_i = args.d_pos[idx];
    unsigned int type = __scalar_as_int(pos_i.w);
    args.d_types[idx] = type;
    args.d_pids[idx] = idx;
    }

#else

namespace kernel
    {
hipError_t gpu_compute_nlist_stencil(unsigned int* d_nlist,
                                     unsigned int* d_n_neigh,
                                     Scalar4* d_last_updated_pos,
                                     unsigned int* d_conditions,
                                     const unsigned int* d_Nmax,
                                     const size_t* d_head_list,
                                     const unsigned int* d_pid_map,
                                     const Scalar4* d_pos,
                                     const unsigned int* d_body,
                                     const unsigned int N,
                                     const unsigned int* d_cell_size,
                                     const Scalar4* d_cell_xyzf,
                                     const uint2* d_cell_type_body,
                                     const Index3D& ci,
                                     const Index2D& cli,
                                     const Scalar4* d_stencil,
                                     const unsigned int* d_n_stencil,
                                     const Index2D& stencil_idx,
                                     const BoxDim& box,
                                     const Scalar* d_r_cut,
                                     const Scalar r_buff,
                                     const unsigned int ntypes,
                                     const Scalar3& ghost_width,
                                     bool filter_body,
                                     const unsigned int threads_per_particle,
                                     const unsigned int block_size,
                                     const hipDeviceProp_t& devprop)
    {
    // shared memory = r_listsq + Nmax
    Index2D typpair_idx(ntypes);
    unsigned int shared_size = (unsigned int)(sizeof(Scalar) * typpair_idx.getNumElements()
                                              + sizeof(unsigned int) * ntypes);

    if (shared_size > devprop.sharedMemPerBlock)
        {
        throw std::runtime_error("Neighborlist r_cut matrix exceeds the available shared memory "
                                 "per block.");
        }

    std::initializer_list<mtl::FunctionConstant> constants
        = {mtl::FunctionConstant::make_bool(0, filter_body),
           mtl::FunctionConstant::make_uint(1, threads_per_particle)};

    const char* name = "hoomd::md::gpu_compute_nlist_stencil_kernel";
    unsigned int max_block_size
        = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name, constants);
    max_block_size -= max_block_size % max_threads_per_particle;
    unsigned int run_block_size = std::min(block_size, max_block_size);

    kernel_args::nlist_stencil_args args;
    args.d_nlist = d_nlist;
    args.d_n_neigh = d_n_neigh;
    args.d_last_updated_pos = d_last_updated_pos;
    args.d_conditions = d_conditions;
    args.d_Nmax = d_Nmax;
    args.d_head_list = d_head_list;
    args.d_pid_map = d_pid_map;
    args.d_pos = d_pos;
    args.d_body = d_body;
    args.d_cell_size = d_cell_size;
    args.d_cell_xyzf = d_cell_xyzf;
    args.d_cell_type_body = d_cell_type_body;
    args.d_stencil = d_stencil;
    args.d_n_stencil = d_n_stencil;
    args.d_r_cut = d_r_cut;
    args.ci = ci;
    args.cli = cli;
    args.stencil_idx = stencil_idx;
    args.box = box;
    args.ghost_width = ghost_width;
    args.r_buff = r_buff;
    args.ntypes = ntypes;
    args.N = N;

    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(N / (run_block_size / threads_per_particle) + 1),
                       dim3(run_block_size),
                       args,
                       shared_size,
                       constants);
    }

hipError_t gpu_compute_nlist_stencil_fill_types(unsigned int* d_pids,
                                                unsigned int* d_types,
                                                const Scalar4* d_pos,
                                                const unsigned int N)
    {
    const unsigned int block_size = 128;

    kernel_args::nlist_stencil_fill_types_args args;
    args.d_pids = d_pids;
    args.d_types = d_types;
    args.d_pos = d_pos;
    args.N = N;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       "hoomd::md::gpu_compute_nlist_stencil_fill_types_kernel",
                       dim3(N / block_size + 1),
                       dim3(block_size),
                       args);
    }

/*! The CUB two pass interface is kept: the first call (d_tmp_storage == NULL) requests a token
    amount of temporary storage, and the second sorts d_types and d_pids in place, so the sorted
    values never need to be swapped from the alternate buffers.
*/
void gpu_compute_nlist_stencil_sort_types(unsigned int* d_pids,
                                          unsigned int* d_pids_alt,
                                          unsigned int* d_types,
                                          unsigned int* d_types_alt,
                                          void* d_tmp_storage,
                                          size_t& tmp_storage_bytes,
                                          bool& swap,
                                          const unsigned int N)
    {
    if (d_tmp_storage == NULL)
        {
        tmp_storage_bytes = 4;
        return;
        }
    mtl::sort_pairs(d_types, d_pids, N);
    swap = false;
    }

    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
