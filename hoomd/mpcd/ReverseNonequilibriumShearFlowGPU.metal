// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ReverseNonequilibriumShearFlowGPU.metal
    \brief Metal port of ReverseNonequilibriumShearFlowGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/HOOMDMath.h"

#ifndef __METAL_VERSION__
#include "ReverseNonequilibriumShearFlowGPU.cuh"
#include "ReverseNonequilibriumShearFlowUtilities.h"

#include <algorithm>
#endif

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel_args
    {
struct rnes_filter_particles_args
    {
    HOOMD_DEVPTR(Scalar2) d_particles_lo;
    HOOMD_DEVPTR(Scalar2) d_particles_hi;
    HOOMD_DEVPTR(unsigned int) d_num_lo_hi;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    HOOMD_DEVPTR(const Scalar4) d_vel;
    Scalar mass;
    Scalar2 pos_lo;
    Scalar2 pos_hi;
    unsigned int num_lo_alloc;
    unsigned int num_hi_alloc;
    unsigned int N;
    };

struct rnes_swap_particles_args
    {
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(Scalar) d_momentum_sum;
    HOOMD_DEVPTR(const Scalar2) d_particles_staged;
    Scalar mass;
    unsigned int num_staged;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void rnes_filter_particles(HOOMD_KERNEL_PARAMS(kernel_args::rnes_filter_particles_args))
    {
    device Scalar2* d_particles_lo = args.d_particles_lo;
    device Scalar2* d_particles_hi = args.d_particles_hi;
    device unsigned int* d_num_lo_hi = args.d_num_lo_hi;
    device const Scalar4* d_pos = args.d_pos;
    device const Scalar4* d_vel = args.d_vel;
    const Scalar mass = args.mass;
    const Scalar2 pos_lo = args.pos_lo;
    const Scalar2 pos_hi = args.pos_hi;
    const unsigned int num_lo_alloc = args.num_lo_alloc;
    const unsigned int num_hi_alloc = args.num_hi_alloc;
    const unsigned int N = args.N;

    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const Scalar4 vel = d_vel[idx];
    const Scalar momentum = vel.x * mass;
    const Scalar y = d_pos[idx].y;

    if (pos_lo.x <= y && y < pos_lo.y
        && momentum > Scalar(0.0)) // lower slab, search for positive momentum
        {
        const unsigned int num_lo = atomicInc(d_num_lo_hi, 0xffffffff);
        if (num_lo < num_lo_alloc)
            {
            d_particles_lo[num_lo] = make_scalar2(__int_as_scalar(idx), momentum);
            }
        }
    else if (pos_hi.x <= y && y < pos_hi.y
             && momentum < Scalar(0.0)) // higher slab, search for negative momentum
        {
        const unsigned int num_hi = atomicInc(d_num_lo_hi + 1, 0xffffffff);
        if (num_hi < num_hi_alloc)
            {
            d_particles_hi[num_hi] = make_scalar2(__int_as_scalar(idx), momentum);
            }
        }
    }

kernel void rnes_swap_particles(HOOMD_KERNEL_PARAMS(kernel_args::rnes_swap_particles_args))
    {
    device Scalar4* d_vel = args.d_vel;
    device Scalar* d_momentum_sum = args.d_momentum_sum;
    device const Scalar2* d_particles_staged = args.d_particles_staged;
    const Scalar mass = args.mass;
    const unsigned int num_staged = args.num_staged;

    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_staged)
        return;

    const Scalar2 pidx_mom = d_particles_staged[idx];
    const unsigned int pidx = __scalar_as_int(pidx_mom.x);
    const Scalar new_momentum = pidx_mom.y;

    const Scalar current_momentum = d_vel[pidx].x * mass;

    d_vel[pidx].x = new_momentum / mass;
    atomicAdd(d_momentum_sum, fabs(new_momentum - current_momentum));
    }

#else

namespace kernel
    {





    } // end namespace kernel

cudaError_t rnes_filter_particles(Scalar2* d_particles_lo,
                                  Scalar2* d_particles_hi,
                                  unsigned int* d_num_lo_hi,
                                  const Scalar4* d_pos,
                                  const Scalar4* d_vel,
                                  const Scalar mass,
                                  const Scalar2& pos_lo,
                                  const Scalar2& pos_hi,
                                  const unsigned int num_lo_alloc,
                                  const unsigned int num_hi_alloc,
                                  const unsigned int N,
                                  const unsigned int block_size)
    {
    // zero counts in each bin
    cudaError_t error = cudaMemset(d_num_lo_hi, 0, 2 * sizeof(unsigned int));
    if (error != cudaSuccess)
        return error;

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::mpcd::gpu::rnes_filter_particles");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(N / run_block_size + 1);
    {
        kernel_args::rnes_filter_particles_args _args;
        _args.d_particles_lo = d_particles_lo;
        _args.d_particles_hi = d_particles_hi;
        _args.d_num_lo_hi = d_num_lo_hi;
        _args.d_pos = d_pos;
        _args.d_vel = d_vel;
        _args.mass = mass;
        _args.pos_lo = pos_lo;
        _args.pos_hi = pos_hi;
        _args.num_lo_alloc = num_lo_alloc;
        _args.num_hi_alloc = num_hi_alloc;
        _args.N = N;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::mpcd::gpu::rnes_filter_particles",
                           grid,
                           run_block_size,
                           _args,
                           0);
        }

    return cudaSuccess;
    }

/*
 * thrust doesn't like to be included in host code, so explicitly list the ones
 * we need as wrappers to a template.
 */
//! Sort the staged particles on the host (thrust::sort in ReverseNonequilibriumShearFlowGPU.cu)
/*! The lists hold the particles in the slabs, so they are short. Memory is shared with the GPU,
    so the host sorts them in place after the GPU work that filled them completes.
*/
template<class T> void rnes_thrust_sort(Scalar2* d_particles, const unsigned int N, const T& comp)
    {
    hipDeviceSynchronize();
    std::sort(d_particles, d_particles + N, comp);
    }
void rnes_sort(Scalar2* d_particles,
               const unsigned int N,
               const mpcd::detail::MinimumMomentum& comp)
    {
    rnes_thrust_sort(d_particles, N, comp);
    }
void rnes_sort(Scalar2* d_particles,
               const unsigned int N,
               const mpcd::detail::MaximumMomentum& comp)
    {
    rnes_thrust_sort(d_particles, N, comp);
    }
void rnes_sort(Scalar2* d_particles,
               const unsigned int N,
               const mpcd::detail::CompareMomentumToTarget& comp)
    {
    rnes_thrust_sort(d_particles, N, comp);
    }

cudaError_t rnes_copy_top_particles(Scalar2* h_top_particles,
                                    const Scalar2* d_particles,
                                    const unsigned int num_top)
    {
    return cudaMemcpy(h_top_particles,
                      d_particles,
                      num_top * sizeof(Scalar2),
                      cudaMemcpyDeviceToHost);
    }

cudaError_t rnes_swap_particles(Scalar4* d_vel,
                                Scalar* d_momentum_sum,
                                const Scalar2* d_particles_staged,
                                const Scalar mass,
                                const unsigned int num_staged,
                                const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::mpcd::gpu::rnes_swap_particles");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_staged / run_block_size + 1);
    {
        kernel_args::rnes_swap_particles_args _args;
        _args.d_vel = d_vel;
        _args.d_momentum_sum = d_momentum_sum;
        _args.d_particles_staged = d_particles_staged;
        _args.mass = mass;
        _args.num_staged = num_staged;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::mpcd::gpu::rnes_swap_particles",
                           grid,
                           run_block_size,
                           _args,
                           0);
        }

    return cudaSuccess;
    }

#endif

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
