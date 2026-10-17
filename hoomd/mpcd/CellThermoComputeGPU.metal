// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file CellThermoComputeGPU.metal
    \brief Metal port of CellThermoComputeGPU.cu

    The cell velocities and energies are double precision arrays on the host. Apple GPUs have no
    double arithmetic, so the kernels sum each cell in single precision and read and write the
    arrays through hoomd_double. The net reduction over all cells sums blocks of cells on the GPU
    and adds the block sums in double precision on the host.

    The CUDA template parameters are Metal function constants: 0: need_energy, 1: threads per cell.
    begin_cell_thermo and end_cell_thermo (outer cells with MPI) are not ported because MPI is not
    available on Metal.
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifdef __METAL_VERSION__
#include "hoomd/mtl/WarpTools.h"
// double precision arrays are read and written through hoomd_double
#define double hoomd_double
#else
#include "CellThermoComputeGPU.cuh"
#endif

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel_args
    {
struct inner_cell_thermo_args
    {
    HOOMD_DEVPTR(double4) d_cell_vel;
    HOOMD_DEVPTR(double3) d_cell_energy;
    HOOMD_DEVPTR(const unsigned int) d_cell_np;
    HOOMD_DEVPTR(const unsigned int) d_cell_list;
    HOOMD_DEVPTR(const Scalar4) d_vel;
    HOOMD_DEVPTR(const Scalar4) d_embed_vel;
    HOOMD_DEVPTR(const unsigned int) d_embed_idx;
    Index3D ci;
    Index3D inner_ci;
    uint3 offset;
    Index2D cli;
    Scalar mpcd_mass;
    unsigned int N_mpcd;
    unsigned int n_dimensions;
    };

//! Device layout of mpcd::detail::cell_thermo_element
struct cell_thermo_element_storage
    {
    double3 momentum;
    double energy;
    double temperature;
    unsigned int flag;
    };

struct stage_net_cell_thermo_args
    {
    HOOMD_DEVPTR(cell_thermo_element_storage) d_tmp_thermo;
    HOOMD_DEVPTR(const double4) d_cell_vel;
    HOOMD_DEVPTR(const double3) d_cell_energy;
    Index3D tmp_ci;
    Index3D ci;
    };

struct reduce_net_cell_thermo_args
    {
    HOOMD_DEVPTR(cell_thermo_element_storage) d_partial;
    HOOMD_DEVPTR(const cell_thermo_element_storage) d_tmp_thermo;
    unsigned int N;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant bool need_energy [[function_constant(0)]];
constant uint tpp [[function_constant(1)]];

//! Computes the cell thermo for inner cells, see inner_cell_thermo in CellThermoComputeGPU.cu
kernel void inner_cell_thermo(HOOMD_KERNEL_PARAMS(kernel_args::inner_cell_thermo_args))
    {
    Index3D ci = args.ci;
    Index3D inner_ci = args.inner_ci;
    Index2D cli = args.cli;

    // tpp threads per cell
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tpp * inner_ci.getNumElements())
        return;

    // reinterpret the thread id as a cell by first mapping the thread into the inner indexer,
    // shifting by the offset of the inner indexer from the full indexer, and then compressing
    // back into a 1D cell id
    const uint3 inner_cell = inner_ci.getTriple(idx / tpp);
    const uint3 cell = make_uint3(inner_cell.x + args.offset.x,
                                  inner_cell.y + args.offset.y,
                                  inner_cell.z + args.offset.z);
    const unsigned int cell_id = ci(cell.x, cell.y, cell.z);

    const unsigned int np = args.d_cell_np[cell_id];
    Scalar4 momentum = make_scalar4(0.0, 0.0, 0.0, 0.0);
    Scalar ke(0.0);

    for (unsigned int offset = (idx % tpp); offset < np; offset += tpp)
        {
        // Load particle data
        const unsigned int cur_p = args.d_cell_list[cli(offset, cell_id)];
        Scalar3 vel_i;
        Scalar mass_i;
        if (cur_p < args.N_mpcd)
            {
            Scalar4 vel_cell = args.d_vel[cur_p];
            vel_i = make_scalar3(vel_cell.x, vel_cell.y, vel_cell.z);
            mass_i = args.mpcd_mass;
            }
        else
            {
            Scalar4 vel_m = args.d_embed_vel[args.d_embed_idx[cur_p - args.N_mpcd]];
            vel_i = make_scalar3(vel_m.x, vel_m.y, vel_m.z);
            mass_i = vel_m.w;
            }

        // add momentum
        momentum.x += mass_i * vel_i.x;
        momentum.y += mass_i * vel_i.y;
        momentum.z += mass_i * vel_i.z;
        momentum.w += mass_i;

        // also compute ke of the particle
        if (need_energy)
            ke += Scalar(0.5) * mass_i * dot(vel_i, vel_i);
        }

    // reduce quantities down into the 0-th lane per logical warp
    if (tpp > 1)
        {
        momentum.x = detail::warp_reduce_sum(momentum.x, hoomd_lane, tpp);
        momentum.y = detail::warp_reduce_sum(momentum.y, hoomd_lane, tpp);
        momentum.z = detail::warp_reduce_sum(momentum.z, hoomd_lane, tpp);
        momentum.w = detail::warp_reduce_sum(momentum.w, hoomd_lane, tpp);
        if (need_energy)
            ke = detail::warp_reduce_sum(ke, hoomd_lane, tpp);
        }

    // 0-th lane in each warp writes the result
    if (idx % tpp == 0)
        {
        const Scalar mass = momentum.w;
        Scalar3 vel_cm = make_scalar3(0.0, 0.0, 0.0);
        if (mass > Scalar(0.))
            {
            vel_cm.x = momentum.x / mass;
            vel_cm.y = momentum.y / mass;
            vel_cm.z = momentum.z / mass;
            }
        args.d_cell_vel[cell_id] = make_double4(vel_cm.x, vel_cm.y, vel_cm.z, mass);

        if (need_energy)
            {
            Scalar temp(0.0);
            if (np > 1)
                {
                const Scalar ke_cm = Scalar(0.5) * mass * dot(vel_cm, vel_cm);
                temp = Scalar(2.) * (ke - ke_cm) / (args.n_dimensions * (np - 1));
                }
            hoomd_double3 energy;
            energy.x = ke;
            energy.y = temp;
            energy.z = __int_as_double(np);
            args.d_cell_energy[cell_id] = energy;
            }
        }
    }

//! Stage the cell properties for the net reduction, see stage_net_cell_thermo in
//! CellThermoComputeGPU.cu
kernel void stage_net_cell_thermo(HOOMD_KERNEL_PARAMS(kernel_args::stage_net_cell_thermo_args))
    {
    Index3D tmp_ci = args.tmp_ci;
    Index3D ci = args.ci;

    // one thread per cell
    unsigned int tmp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tmp_idx >= tmp_ci.getNumElements())
        return;

    // use the temporary cell indexer to map to a cell, then use the real cell indexer to
    // get the read index
    uint3 cell = tmp_ci.getTriple(tmp_idx);
    const unsigned int idx = ci(cell.x, cell.y, cell.z);

    const double4 vel_mass = args.d_cell_vel[idx];
    const Scalar mass = vel_mass.w;

    kernel_args::cell_thermo_element_storage thermo;
    thermo.momentum = make_double3(mass * Scalar(vel_mass.x),
                                   mass * Scalar(vel_mass.y),
                                   mass * Scalar(vel_mass.z));

    if (need_energy)
        {
        const double3 cell_energy = args.d_cell_energy[idx];
        thermo.energy = cell_energy.x;
        if (__double_as_int(cell_energy.z) > 1)
            {
            thermo.temperature = cell_energy.y;
            thermo.flag = 1;
            }
        else
            {
            thermo.temperature = Scalar(0.0);
            thermo.flag = 0;
            }
        }
    else
        {
        thermo.energy = Scalar(0.);
        thermo.temperature = Scalar(0.);
        thermo.flag = 0;
        }

    args.d_tmp_thermo[tmp_idx] = thermo;
    }

//! Sum blocks of staged cell properties (one threadgroup per block, power of 2 threads)
kernel void reduce_net_cell_thermo(HOOMD_KERNEL_PARAMS(kernel_args::reduce_net_cell_thermo_args))
    {
    threadgroup Scalar* s_data = (threadgroup Scalar*)hoomd_dynamic_shared;
    const unsigned int n = blockDim.x;

    Scalar momentum_x(0), momentum_y(0), momentum_z(0), energy(0), temperature(0);
    unsigned int flag = 0;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < args.N)
        {
        momentum_x = args.d_tmp_thermo[idx].momentum.x;
        momentum_y = args.d_tmp_thermo[idx].momentum.y;
        momentum_z = args.d_tmp_thermo[idx].momentum.z;
        energy = args.d_tmp_thermo[idx].energy;
        temperature = args.d_tmp_thermo[idx].temperature;
        flag = args.d_tmp_thermo[idx].flag;
        }

    s_data[0 * n + threadIdx.x] = momentum_x;
    s_data[1 * n + threadIdx.x] = momentum_y;
    s_data[2 * n + threadIdx.x] = momentum_z;
    s_data[3 * n + threadIdx.x] = energy;
    s_data[4 * n + threadIdx.x] = temperature;
    s_data[5 * n + threadIdx.x] = as_type<Scalar>(flag);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (unsigned int offset = n / 2; offset > 0; offset >>= 1)
        {
        if (threadIdx.x < offset)
            {
            for (unsigned int k = 0; k < 5; ++k)
                s_data[k * n + threadIdx.x] += s_data[k * n + threadIdx.x + offset];
            s_data[5 * n + threadIdx.x] = as_type<Scalar>(
                as_type<unsigned int>(s_data[5 * n + threadIdx.x])
                + as_type<unsigned int>(s_data[5 * n + threadIdx.x + offset]));
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    if (threadIdx.x == 0)
        {
        kernel_args::cell_thermo_element_storage sum;
        sum.momentum = make_double3(s_data[0], s_data[n], s_data[2 * n]);
        sum.energy = s_data[3 * n];
        sum.temperature = s_data[4 * n];
        sum.flag = as_type<unsigned int>(s_data[5 * n]);
        args.d_partial[blockIdx.x] = sum;
        }
    }

#else

static_assert(sizeof(kernel_args::cell_thermo_element_storage)
                  == sizeof(mpcd::detail::cell_thermo_element),
              "cell_thermo_element layout");

cudaError_t inner_cell_thermo(const mpcd::detail::thermo_args_t& args,
                              const Index3D& ci,
                              const Index3D& inner_ci,
                              const uint3& offset,
                              const unsigned int n_dimensions,
                              const unsigned int block_size,
                              const unsigned int tpp)
    {
    if (inner_ci.getNumElements() == 0)
        return cudaSuccess;

    std::initializer_list<mtl::FunctionConstant> constants
        = {mtl::FunctionConstant::make_bool(0, args.need_energy),
           mtl::FunctionConstant::make_uint(1, tpp)};
    const char* name = "hoomd::mpcd::gpu::inner_cell_thermo";
    unsigned int max_block_size = mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name, constants);
    max_block_size -= max_block_size % 32;
    unsigned int run_block_size = std::min(block_size, max_block_size);

    kernel_args::inner_cell_thermo_args kargs;
    kargs.d_cell_vel = args.cell_vel;
    kargs.d_cell_energy = args.cell_energy;
    kargs.d_cell_np = args.cell_np;
    kargs.d_cell_list = args.cell_list;
    kargs.d_vel = args.vel;
    kargs.d_embed_vel = args.embed_vel;
    kargs.d_embed_idx = args.embed_idx;
    kargs.ci = ci;
    kargs.inner_ci = inner_ci;
    kargs.offset = offset;
    kargs.cli = args.cli;
    kargs.mpcd_mass = args.mass;
    kargs.N_mpcd = args.N_mpcd;
    kargs.n_dimensions = n_dimensions;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(tpp * inner_ci.getNumElements() / run_block_size + 1),
                       dim3(run_block_size),
                       kargs,
                       0,
                       constants);
    }

cudaError_t stage_net_cell_thermo(mpcd::detail::cell_thermo_element* d_tmp_thermo,
                                  const double4* d_cell_vel,
                                  const double3* d_cell_energy,
                                  const Index3D& tmp_ci,
                                  const Index3D& ci,
                                  const bool need_energy,
                                  const unsigned int block_size)
    {
    std::initializer_list<mtl::FunctionConstant> constants
        = {mtl::FunctionConstant::make_bool(0, need_energy)};
    const char* name = "hoomd::mpcd::gpu::stage_net_cell_thermo";
    unsigned int run_block_size
        = std::min(block_size, mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name, constants));

    kernel_args::stage_net_cell_thermo_args args;
    args.d_tmp_thermo
        = reinterpret_cast<kernel_args::cell_thermo_element_storage*>(d_tmp_thermo);
    args.d_cell_vel = d_cell_vel;
    args.d_cell_energy = d_cell_energy;
    args.tmp_ci = tmp_ci;
    args.ci = ci;
    return mtl::launch(HOOMD_MTL_SOURCE,
                       name,
                       dim3(tmp_ci.getNumElements() / run_block_size + 1),
                       dim3(run_block_size),
                       args,
                       0,
                       constants);
    }

/*! The CUB two pass interface is kept. The first call (d_tmp == NULL) sizes the temporary storage
    for the block sums. The second sums blocks of cells on the GPU and then adds the block sums in
    double precision on the host (which waits for the GPU).
*/
cudaError_t reduce_net_cell_thermo(mpcd::detail::cell_thermo_element* d_reduced,
                                   void* d_tmp,
                                   size_t& tmp_bytes,
                                   const mpcd::detail::cell_thermo_element* d_tmp_thermo,
                                   const size_t Ncell)
    {
    const unsigned int block_size = 256;
    const unsigned int num_blocks = static_cast<unsigned int>((Ncell + block_size - 1) / block_size);
    if (d_tmp == NULL)
        {
        tmp_bytes = std::max(num_blocks, 1u) * sizeof(mpcd::detail::cell_thermo_element);
        return cudaSuccess;
        }

    mpcd::detail::cell_thermo_element* d_partial
        = static_cast<mpcd::detail::cell_thermo_element*>(d_tmp);
    if (num_blocks > 0)
        {
        kernel_args::reduce_net_cell_thermo_args args;
        args.d_partial = reinterpret_cast<kernel_args::cell_thermo_element_storage*>(d_partial);
        args.d_tmp_thermo
            = reinterpret_cast<const kernel_args::cell_thermo_element_storage*>(d_tmp_thermo);
        args.N = static_cast<unsigned int>(Ncell);
        mtl::launch(HOOMD_MTL_SOURCE,
                    "hoomd::mpcd::gpu::reduce_net_cell_thermo",
                    dim3(num_blocks),
                    dim3(block_size),
                    args,
                    6 * block_size * sizeof(Scalar));
        }

    // add the block sums in double precision
    cudaError_t error = hipDeviceSynchronize();
    mpcd::detail::cell_thermo_element sum = {};
    for (unsigned int i = 0; i < num_blocks; ++i)
        sum = sum + d_partial[i];
    *d_reduced = sum;
    return error;
    }

#endif

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
