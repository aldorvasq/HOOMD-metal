// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file RejectionVirtualParticleFillerGPU.metal
    \brief Metal port of RejectionVirtualParticleFillerGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "ParticleDataUtilities.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#ifndef __METAL_VERSION__
#include "RejectionVirtualParticleFillerGPU.cuh"

#include "hoomd/mtl/Primitives.h"
#endif

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel_args
    {
struct copy_virtual_particles_args
    {
    HOOMD_DEVPTR(unsigned int) d_keep_indices;
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(Scalar4) d_vel;
    HOOMD_DEVPTR(unsigned int) d_tags;
    HOOMD_DEVPTR(const Scalar4) d_tmp_pos;
    HOOMD_DEVPTR(const Scalar4) d_tmp_vel;
    unsigned int first_idx;
    unsigned int first_tag;
    unsigned int n_virtual;
    unsigned int block_size;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void copy_virtual_particles(HOOMD_KERNEL_PARAMS(kernel_args::copy_virtual_particles_args))
    {
    device unsigned int* d_keep_indices = args.d_keep_indices;
    device Scalar4* d_pos = args.d_pos;
    device Scalar4* d_vel = args.d_vel;
    device unsigned int* d_tags = args.d_tags;
    device const Scalar4* d_tmp_pos = args.d_tmp_pos;
    device const Scalar4* d_tmp_vel = args.d_tmp_vel;
    const unsigned int first_idx = args.first_idx;
    const unsigned int first_tag = args.first_tag;
    const unsigned int n_virtual = args.n_virtual;
    const unsigned int block_size = args.block_size;

    // one thread per virtual particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_virtual)
        return;

    // d_keep_indices holds accepted particle indices from the temporary arrays
    const unsigned int tmp_pidx = d_keep_indices[idx];
    const unsigned int pidx = first_idx + idx;
    d_pos[pidx] = d_tmp_pos[tmp_pidx];
    d_vel[pidx] = d_tmp_vel[tmp_pidx];
    d_tags[pidx] = first_tag + idx;
    }

#else

namespace kernel
    {

/*!
 * \b implementation
 * Using one thread per particle, we assign the particle position, velocity and tags using the
 * compacted indices array as an input.
 */


    } // end namespace kernel

cudaError_t __attribute__((visibility("default")))
compact_virtual_particle_indices(void* d_tmp,
                                 size_t& tmp_bytes,
                                 const bool* d_keep_particles,
                                 const unsigned int num_particles,
                                 unsigned int* d_keep_indices,
                                 unsigned int* d_num_keep)
    {
    // select the indices of the particles to keep (cub::DeviceSelect::Flagged on a counting
    // iterator): the first call sizes temporary storage for the index sequence
    if (d_tmp == NULL)
        {
        tmp_bytes = (num_particles > 0 ? num_particles : 1) * sizeof(unsigned int);
        return cudaSuccess;
        }
    unsigned int* d_sequence = static_cast<unsigned int*>(d_tmp);
    mtl::sequence(d_sequence, num_particles);
    mtl::select_flagged(d_sequence,
                        reinterpret_cast<const unsigned char*>(d_keep_particles),
                        d_keep_indices,
                        d_num_keep,
                        num_particles);
    return cudaSuccess;
    }

cudaError_t __attribute__((visibility("default")))
copy_virtual_particles(unsigned int* d_keep_indices,
                       Scalar4* d_pos,
                       Scalar4* d_vel,
                       unsigned int* d_tags,
                       const Scalar4* d_tmp_pos,
                       const Scalar4* d_tmp_vel,
                       const unsigned int first_idx,
                       const unsigned int first_tag,
                       const unsigned int n_virtual,
                       const unsigned int block_size)
    {
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::mpcd::gpu::copy_virtual_particles");
    const unsigned int max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(n_virtual / run_block_size + 1);
    {
        kernel_args::copy_virtual_particles_args _args;
        _args.d_keep_indices = d_keep_indices;
        _args.d_pos = d_pos;
        _args.d_vel = d_vel;
        _args.d_tags = d_tags;
        _args.d_tmp_pos = d_tmp_pos;
        _args.d_tmp_vel = d_tmp_vel;
        _args.first_idx = first_idx;
        _args.first_tag = first_tag;
        _args.n_virtual = n_virtual;
        _args.block_size = block_size;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::mpcd::gpu::copy_virtual_particles",
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
