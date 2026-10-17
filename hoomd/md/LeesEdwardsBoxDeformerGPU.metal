// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file LeesEdwardsBoxDeformerGPU.metal
    \brief Metal port of LeesEdwardsBoxDeformerGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#ifndef __METAL_VERSION__
#include "LeesEdwardsBoxDeformerGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct gpu_lees_edwards_remap_kernel_args
    {
    unsigned int N;
    HOOMD_DEVPTR(Scalar4) d_pos;
    HOOMD_DEVPTR(int3) d_image;
    BoxDim flipped_box;
    Scalar xy;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


kernel void gpu_lees_edwards_remap_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_lees_edwards_remap_kernel_args))
    {
    const unsigned int N = args.N;
    device Scalar4* d_pos = args.d_pos;
    device int3* d_image = args.d_image;
    const BoxDim flipped_box = args.flipped_box;
    const Scalar xy = args.xy;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar3 L = flipped_box.getL();
    Scalar xy_flip = flipped_box.getTiltFactorXY();

    if (idx < N)
        {
        // Transformation of x-images
        d_image[idx].x -= d_image[idx].y * static_cast<int>(std::round((xy_flip - xy) * L.y / L.x));

        // Convert position to fractional coordinates
        Scalar3 pos = make_scalar3(d_pos[idx].x, d_pos[idx].y, d_pos[idx].z);
        Scalar3 fpos = flipped_box.makeFraction(pos);

        int ix = static_cast<int>(std::floor(fpos.x));
        int iy = static_cast<int>(std::floor(fpos.y));
        int iz = static_cast<int>(std::floor(fpos.z));

        // Wrap into flipped box and update image flags
        fpos.x -= ix;
        fpos.y -= iy;
        fpos.z -= iz;

        d_image[idx].x += ix;
        d_image[idx].y += iy;
        d_image[idx].z += iz;

        // Convert back to Cartesian coordinates
        pos = flipped_box.makeCoordinates(fpos);

        d_pos[idx].x = pos.x;
        d_pos[idx].y = pos.y;
        d_pos[idx].z = pos.z;
        }
    }

#else

namespace kernel
    {
/*! Particle remap procedure
    First update the x image index using the transformation below:
        i1' = i1 - i2*(xy' - xy)*Ly/Lx
    This preserves the particle's physical periodic image under the new lattice basis.

    Convert the particle position to fractional coordinates with respect to the flipped box.
    Wrap the fractional coordinates into the interval [0,1), updating the image flags for any
    periodic crossings.

    Convert the wrapped fractional coordinates back to Cartesian coordinates in the flipped box.
*/


hipError_t gpu_lees_edwards_remap(const unsigned int N,
                                  Scalar4* d_pos,
                                  int3* d_image,
                                  const BoxDim& flipped_box,
                                  const Scalar xy,
                                  unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_lees_edwards_remap_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid((N / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    {
        kernel_args::gpu_lees_edwards_remap_kernel_args _args;
        _args.N = N;
        _args.d_pos = d_pos;
        _args.d_image = d_image;
        _args.flipped_box = flipped_box;
        _args.xy = xy;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_lees_edwards_remap_kernel",
                           grid,
                           threads,
                           _args,
                           0);
        }

    return hipSuccess;
    }

    } // namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
