// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ForceCompositeGPU.metal
    \brief Metal port of ForceCompositeGPU.cu
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/VectorMath.h"
#include "hoomd/HOOMDMath.h"

#ifndef __METAL_VERSION__
#include "ForceCompositeGPU.cuh"

#include "hoomd/mtl/Primitives.h"
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct gpu_rigid_force_sliding_kernel_args
    {
    HOOMD_DEVPTR(Scalar4) d_force;
    HOOMD_DEVPTR(Scalar4) d_torque;
    HOOMD_DEVPTR(const unsigned int) d_molecule_len;
    HOOMD_DEVPTR(const unsigned int) d_molecule_list;
    HOOMD_DEVPTR(const unsigned int) d_molecule_idx;
    HOOMD_DEVPTR(const unsigned int) d_rigid_center;
    Index2D molecule_indexer;
    HOOMD_DEVPTR(const Scalar4) d_postype;
    HOOMD_DEVPTR(const Scalar4) d_orientation;
    Index2D body_indexer;
    HOOMD_DEVPTR(Scalar3) d_body_pos;
    HOOMD_DEVPTR(Scalar4) d_body_orientation;
    HOOMD_DEVPTR(const unsigned int) d_body_len;
    HOOMD_DEVPTR(const unsigned int) d_body;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(uint2) d_flag;
    HOOMD_DEVPTR(Scalar4) d_net_force;
    HOOMD_DEVPTR(Scalar4) d_net_torque;
    unsigned int n_mol;
    unsigned int N;
    unsigned int window_size;
    unsigned int thread_mask;
    unsigned int n_bodies_per_block;
    bool zero_force;
    unsigned int nwork;
    };

struct gpu_rigid_virial_sliding_kernel_args
    {
    HOOMD_DEVPTR(Scalar) d_virial;
    HOOMD_DEVPTR(const unsigned int) d_molecule_len;
    HOOMD_DEVPTR(const unsigned int) d_molecule_list;
    HOOMD_DEVPTR(const unsigned int) d_molecule_idx;
    HOOMD_DEVPTR(const unsigned int) d_rigid_center;
    Index2D molecule_indexer;
    HOOMD_DEVPTR(const Scalar4) d_postype;
    HOOMD_DEVPTR(const Scalar4) d_orientation;
    Index2D body_indexer;
    HOOMD_DEVPTR(Scalar3) d_body_pos;
    HOOMD_DEVPTR(Scalar4) d_body_orientation;
    HOOMD_DEVPTR(Scalar4) d_net_force;
    HOOMD_DEVPTR(Scalar) d_net_virial;
    HOOMD_DEVPTR(const unsigned int) d_body;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    unsigned int n_mol;
    unsigned int N;
    size_t net_virial_pitch;
    size_t virial_pitch;
    unsigned int window_size;
    unsigned int thread_mask;
    unsigned int n_bodies_per_block;
    unsigned int nwork;
    };

struct gpu_update_composite_kernel_args
    {
    unsigned int N;
    unsigned int nwork;
    unsigned int n_ghost;
    HOOMD_DEVPTR(const unsigned int) d_lookup_center;
    HOOMD_DEVPTR(Scalar4) d_postype;
    HOOMD_DEVPTR(Scalar4) d_velocity;
    HOOMD_DEVPTR(Scalar4) d_orientation;
    HOOMD_DEVPTR(const Scalar4) d_angmom;
    HOOMD_DEVPTR(const Scalar3) d_inertia;
    Index2D body_indexer;
    HOOMD_DEVPTR(const Scalar3) d_body_pos;
    HOOMD_DEVPTR(const Scalar4) d_body_orientation;
    HOOMD_DEVPTR(const unsigned int) d_body_types;
    HOOMD_DEVPTR(const unsigned int) d_body_len;
    HOOMD_DEVPTR(const unsigned int) d_molecule_order;
    HOOMD_DEVPTR(const unsigned int) d_molecule_len;
    HOOMD_DEVPTR(const unsigned int) d_molecule_idx;
    HOOMD_DEVPTR(int3) d_image;
    BoxDim box;
    BoxDim global_box;
    HOOMD_DEVPTR(uint2) d_flag;
    };

struct gpu_find_rigid_centers_kernel_args
    {
    HOOMD_DEVPTR(const unsigned int) d_body;
    HOOMD_DEVPTR(const unsigned int) d_tag;
    HOOMD_DEVPTR(const unsigned int) d_rtag;
    HOOMD_DEVPTR(unsigned int) d_is_center;
    HOOMD_DEVPTR(unsigned int) d_lookup_center;
    unsigned int n;
    };

    } // end namespace kernel_args

#ifdef __METAL_VERSION__


//! Flag rigid body centers and build the lookup table particle idx -> center idx
kernel void gpu_find_rigid_centers_kernel(
    HOOMD_KERNEL_PARAMS(kernel_args::gpu_find_rigid_centers_kernel_args))
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.n)
        return;

    unsigned int body = args.d_body[idx];
    args.d_is_center[idx] = body == args.d_tag[idx] ? 1 : 0;
    args.d_lookup_center[idx] = (body < MIN_FLOPPY) ? args.d_rtag[body] : NO_BODY;
    }

kernel void gpu_rigid_force_sliding_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_rigid_force_sliding_kernel_args))
    {
    device Scalar4* d_force = args.d_force;
    device Scalar4* d_torque = args.d_torque;
    device const unsigned int* d_molecule_len = args.d_molecule_len;
    device const unsigned int* d_molecule_list = args.d_molecule_list;
    device const unsigned int* d_molecule_idx = args.d_molecule_idx;
    device const unsigned int* d_rigid_center = args.d_rigid_center;
    Index2D molecule_indexer = args.molecule_indexer;
    device const Scalar4* d_postype = args.d_postype;
    device const Scalar4* d_orientation = args.d_orientation;
    Index2D body_indexer = args.body_indexer;
    device Scalar3* d_body_pos = args.d_body_pos;
    device Scalar4* d_body_orientation = args.d_body_orientation;
    device const unsigned int* d_body_len = args.d_body_len;
    device const unsigned int* d_body = args.d_body;
    device const unsigned int* d_tag = args.d_tag;
    device uint2* d_flag = args.d_flag;
    device Scalar4* d_net_force = args.d_net_force;
    device Scalar4* d_net_torque = args.d_net_torque;
    unsigned int n_mol = args.n_mol;
    unsigned int N = args.N;
    unsigned int window_size = args.window_size;
    unsigned int thread_mask = args.thread_mask;
    unsigned int n_bodies_per_block = args.n_bodies_per_block;
    bool zero_force = args.zero_force;
    unsigned int nwork = args.nwork;

    threadgroup char* sum = (threadgroup char*)hoomd_dynamic_shared;

    // determine which body (0 ... n_bodies_per_block-1) this thread is working on
    // assign threads 0, 1, 2, ... to body 0, n, n+1, n+2, ... to body 1, and so on.
    unsigned int m = threadIdx.x / (blockDim.x / n_bodies_per_block);

    // arrays in shared memory
    threadgroup Scalar4* body_force = (threadgroup Scalar4*)sum; // blockDim.x elements
    threadgroup Scalar4* body_orientation = body_force + blockDim.x; // n_bodies_per_block elements
    threadgroup Scalar3* body_torque
        = (threadgroup Scalar3*)(body_orientation + n_bodies_per_block); // blockDim.x elements
    threadgroup unsigned int* body_type
        = (threadgroup unsigned int*)(body_torque + blockDim.x); // n_bodies_per_block elements
    threadgroup unsigned int* mol_idx = body_type + n_bodies_per_block; // n_bodies_per_block elements
    threadgroup unsigned int* central_idx = mol_idx + n_bodies_per_block; // n_bodies_per_block

    // each thread makes partial sums of force and torque of all the particles that this thread
    // loops over
    Scalar4 sum_force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar3 sum_torque = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // thread_mask is a bitmask that masks out the high bits in threadIdx.x.
    // threadIdx.x & thread_mask is an index from 0 to block_size/n_bodies_per_block-1 and
    // determines what offset this thread is to use when accessing the particles in the body
    if ((threadIdx.x & thread_mask) == 0)
        {
        // thread 0 for this body reads in the body id and orientation and stores them in shared
        // memory
        int group_idx = blockIdx.x * n_bodies_per_block + m;
        if (group_idx < nwork)
            {
            central_idx[m] = d_rigid_center[group_idx];
            mol_idx[m] = d_molecule_idx[central_idx[m]];

            if (d_tag[central_idx[m]] != d_body[central_idx[m]])
                {
                // this is not the central ptl, molecule is incomplete - mark as such
                body_type[m] = 0xffffffff;
                body_orientation[m] = make_scalar4(1, 0, 0, 0);
                }
            else
                {
                body_type[m] = __scalar_as_int(d_postype[central_idx[m]].w);
                body_orientation[m] = d_orientation[central_idx[m]];
                }
            }
        else
            {
            mol_idx[m] = NO_BODY;
            }
        }

    __syncthreads();

    if (mol_idx[m] < MIN_FLOPPY)
        {
        // compute the number of windows that we need to loop over
        unsigned int mol_len = d_molecule_len[mol_idx[m]];
        unsigned int n_windows = mol_len / window_size + 1;

        // slide the window throughout the block
        for (unsigned int start = 0; start < n_windows; start++)
            {
            // determine the index with this body that this particle should handle
            unsigned int k = start * window_size + (threadIdx.x & thread_mask);

            // if that index is in the body we are actually handling a real body
            if (k < mol_len)
                {
                // determine the particle idx of the particle
                unsigned int pidx = d_molecule_list[molecule_indexer(k, mol_idx[m])];

                // if this particle is not the central particle
                if (body_type[m] != 0xffffffff && pidx != central_idx[m])
                    {
                    Scalar4 fi = d_net_force[pidx];

                    // will likely need to rotate these components too
                    vec3<Scalar> ti(d_net_torque[pidx]);

                    // zero net torque on constituent particles
                    d_net_torque[pidx] = make_scalar4(0.0, 0.0, 0.0, 0.0);

                    // zero force only if we don't need it later
                    if (zero_force)
                        {
                        // zero net energy on constituent ptls to avoid double counting
                        // also zero net force for consistency
                        d_net_force[pidx] = make_scalar4(0.0, 0.0, 0.0, 0.0);
                        }

                    if (central_idx[m] < N)
                        {
                        // at this point, the molecule needs to be complete
                        if (mol_len != d_body_len[body_type[m]] + 1)
                            {
                            // incomplete molecule
                            atomicMax((device unsigned int*)d_flag, d_body[central_idx[m]] + 1);
                            }

                        // calculate body force and torques
                        vec3<Scalar> particle_pos(d_body_pos[body_indexer(body_type[m], k - 1)]);

                        // tally the force in the per thread counter
                        sum_force.x += fi.x;
                        sum_force.y += fi.y;
                        sum_force.z += fi.z;

                        // sum up energy
                        sum_force.w += fi.w;

                        vec3<Scalar> ri = rotate(quat<Scalar>(body_orientation[m]), particle_pos);

                        // torque = r x f
                        vec3<Scalar> del_torque(cross(ri, vec3<Scalar>(fi)));

                        // tally the torque in the per thread counter
                        sum_torque.x += ti.x + del_torque.x;
                        sum_torque.y += ti.y + del_torque.y;
                        sum_torque.z += ti.z + del_torque.z;
                        }
                    }
                }
            }
        }

    __syncthreads();

    // put the partial sums into shared memory
    body_force[threadIdx.x] = sum_force;
    body_torque[threadIdx.x] = sum_torque;

    __syncthreads();

    // perform a set of partial reductions. Each block_size/n_bodies_per_block threads performs a
    // sum reduction just within its own group
    unsigned int offset = window_size >> 1;
    while (offset > 0)
        {
        if ((threadIdx.x & thread_mask) < offset)
            {
            body_force[threadIdx.x].x += body_force[threadIdx.x + offset].x;
            body_force[threadIdx.x].y += body_force[threadIdx.x + offset].y;
            body_force[threadIdx.x].z += body_force[threadIdx.x + offset].z;
            body_force[threadIdx.x].w += body_force[threadIdx.x + offset].w;

            body_torque[threadIdx.x].x += body_torque[threadIdx.x + offset].x;
            body_torque[threadIdx.x].y += body_torque[threadIdx.x + offset].y;
            body_torque[threadIdx.x].z += body_torque[threadIdx.x + offset].z;
            }

        offset >>= 1;

        __syncthreads();
        }

    // thread 0 within this body writes out the total force and torque for the body
    if ((threadIdx.x & thread_mask) == 0 && mol_idx[m] < MIN_FLOPPY && central_idx[m] < N)
        {
        d_force[central_idx[m]] = body_force[threadIdx.x];
        d_torque[central_idx[m]] = make_scalar4(body_torque[threadIdx.x].x,
                                                body_torque[threadIdx.x].y,
                                                body_torque[threadIdx.x].z,
                                                0.0f);
        }
    }

kernel void gpu_rigid_virial_sliding_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_rigid_virial_sliding_kernel_args))
    {
    device Scalar* d_virial = args.d_virial;
    device const unsigned int* d_molecule_len = args.d_molecule_len;
    device const unsigned int* d_molecule_list = args.d_molecule_list;
    device const unsigned int* d_molecule_idx = args.d_molecule_idx;
    device const unsigned int* d_rigid_center = args.d_rigid_center;
    Index2D molecule_indexer = args.molecule_indexer;
    device const Scalar4* d_postype = args.d_postype;
    device const Scalar4* d_orientation = args.d_orientation;
    Index2D body_indexer = args.body_indexer;
    device Scalar3* d_body_pos = args.d_body_pos;
    device Scalar4* d_body_orientation = args.d_body_orientation;
    device Scalar4* d_net_force = args.d_net_force;
    device Scalar* d_net_virial = args.d_net_virial;
    device const unsigned int* d_body = args.d_body;
    device const unsigned int* d_tag = args.d_tag;
    unsigned int n_mol = args.n_mol;
    unsigned int N = args.N;
    size_t net_virial_pitch = args.net_virial_pitch;
    size_t virial_pitch = args.virial_pitch;
    unsigned int window_size = args.window_size;
    unsigned int thread_mask = args.thread_mask;
    unsigned int n_bodies_per_block = args.n_bodies_per_block;
    unsigned int nwork = args.nwork;

    threadgroup char* sum = (threadgroup char*)hoomd_dynamic_shared;

    // determine which body (0 ... n_bodies_per_block-1) this thread is working on
    // assign threads 0, 1, 2, ... to body 0, n, n+1, n+2, ... to body 1, and so on.
    unsigned int m = threadIdx.x / (blockDim.x / n_bodies_per_block);

    // arrays in shared memory
    threadgroup Scalar4* body_orientation = (threadgroup Scalar4*)sum; // n_bodies_per_block elements
    threadgroup Scalar* sum_virial
        = (threadgroup Scalar*)(body_orientation + n_bodies_per_block); // 6*blockDim.x elements
    threadgroup unsigned int* body_type
        = (threadgroup unsigned int*)(sum_virial + 6 * blockDim.x); // n_bodies_per_block elements
    threadgroup unsigned int* mol_idx = body_type + n_bodies_per_block; // n_bodies_per_block elements
    threadgroup unsigned int* central_idx = mol_idx + n_bodies_per_block; // n_bodies_per_block

    // body_force and body_torque are each shared memory arrays with 1 element per threads
    threadgroup Scalar* body_virial_xx = sum_virial;
    threadgroup Scalar* body_virial_xy = &sum_virial[1 * blockDim.x];
    threadgroup Scalar* body_virial_xz = &sum_virial[2 * blockDim.x];
    threadgroup Scalar* body_virial_yy = &sum_virial[3 * blockDim.x];
    threadgroup Scalar* body_virial_yz = &sum_virial[4 * blockDim.x];
    threadgroup Scalar* body_virial_zz = &sum_virial[5 * blockDim.x];

    // each thread makes partial sums of the virial of all the particles that this thread loops over
    Scalar sum_virial_xx(0.0);
    Scalar sum_virial_xy(0.0);
    Scalar sum_virial_xz(0.0);
    Scalar sum_virial_yy(0.0);
    Scalar sum_virial_yz(0.0);
    Scalar sum_virial_zz(0.0);

    // thread_mask is a bitmask that masks out the high bits in threadIdx.x.
    // threadIdx.x & thread_mask is an index from 0 to block_size/n_bodies_per_block-1 and
    // determines what offset this thread is to use when accessing the particles in the body
    if ((threadIdx.x & thread_mask) == 0)
        {
        // thread 0 for this body reads in the body id and orientation and stores them in shared
        // memory
        int group_idx = blockIdx.x * n_bodies_per_block + m;
        if (group_idx < nwork)
            {
            central_idx[m] = d_rigid_center[group_idx];
            mol_idx[m] = d_molecule_idx[central_idx[m]];

            if (d_tag[central_idx[m]] != d_body[central_idx[m]])
                {
                // this is not the central ptl, molecule is incomplete - mark as such
                body_type[m] = NO_BODY;
                body_orientation[m] = make_scalar4(1, 0, 0, 0);
                }
            else
                {
                body_type[m] = __scalar_as_int(d_postype[central_idx[m]].w);
                body_orientation[m] = d_orientation[central_idx[m]];
                }
            }
        else
            {
            mol_idx[m] = NO_BODY;
            }
        }

    __syncthreads();

    if (mol_idx[m] < MIN_FLOPPY)
        {
        // compute the number of windows that we need to loop over
        unsigned int mol_len = d_molecule_len[mol_idx[m]];
        unsigned int n_windows = mol_len / window_size + 1;

        // slide the window throughout the block
        for (unsigned int start = 0; start < n_windows; start++)
            {
            // determine the index with this body that this particle should handle
            unsigned int k = start * window_size + (threadIdx.x & thread_mask);

            // if that index is in the body we are actually handling a real body
            if (k < mol_len)
                {
                // determine the particle idx of the particle
                unsigned int pidx = d_molecule_list[molecule_indexer(k, mol_idx[m])];

                if (body_type[m] < MIN_FLOPPY && pidx != central_idx[m])
                    {
                    // calculate body force and torques
                    Scalar4 fi = d_net_force[pidx];

                    // sum up virial
                    Scalar virialxx = d_net_virial[0 * net_virial_pitch + pidx];
                    Scalar virialxy = d_net_virial[1 * net_virial_pitch + pidx];
                    Scalar virialxz = d_net_virial[2 * net_virial_pitch + pidx];
                    Scalar virialyy = d_net_virial[3 * net_virial_pitch + pidx];
                    Scalar virialyz = d_net_virial[4 * net_virial_pitch + pidx];
                    Scalar virialzz = d_net_virial[5 * net_virial_pitch + pidx];

                    // zero force and virial on constituent particles
                    d_net_force[pidx] = make_scalar4(0.0, 0.0, 0.0, 0.0);

                    d_net_virial[0 * net_virial_pitch + pidx] = Scalar(0.0);
                    d_net_virial[1 * net_virial_pitch + pidx] = Scalar(0.0);
                    d_net_virial[2 * net_virial_pitch + pidx] = Scalar(0.0);
                    d_net_virial[3 * net_virial_pitch + pidx] = Scalar(0.0);
                    d_net_virial[4 * net_virial_pitch + pidx] = Scalar(0.0);
                    d_net_virial[5 * net_virial_pitch + pidx] = Scalar(0.0);

                    // if this particle is not the central particle (incomplete molecules can't have
                    // local members)
                    if (central_idx[m] < N)
                        {
                        vec3<Scalar> particle_pos(d_body_pos[body_indexer(body_type[m], k - 1)]);
                        vec3<Scalar> ri = rotate(quat<Scalar>(body_orientation[m]), particle_pos);

                        // subtract intra-body virial prt
                        sum_virial_xx += virialxx - fi.x * ri.x;
                        sum_virial_xy += virialxy - fi.x * ri.y;
                        sum_virial_xz += virialxz - fi.x * ri.z;
                        sum_virial_yy += virialyy - fi.y * ri.y;
                        sum_virial_yz += virialyz - fi.y * ri.z;
                        sum_virial_zz += virialzz - fi.z * ri.z;
                        }
                    }
                }
            }
        }

    __syncthreads();

    // put the partial sums into shared memory
    body_virial_xx[threadIdx.x] = sum_virial_xx;
    body_virial_xy[threadIdx.x] = sum_virial_xy;
    body_virial_xz[threadIdx.x] = sum_virial_xz;
    body_virial_yy[threadIdx.x] = sum_virial_yy;
    body_virial_yz[threadIdx.x] = sum_virial_yz;
    body_virial_zz[threadIdx.x] = sum_virial_zz;

    __syncthreads();

    // perform a set of partial reductions. Each block_size/n_bodies_per_block threads performs a
    // sum reduction just within its own group
    unsigned int offset = window_size >> 1;
    while (offset > 0)
        {
        if ((threadIdx.x & thread_mask) < offset)
            {
            body_virial_xx[threadIdx.x] += body_virial_xx[threadIdx.x + offset];
            body_virial_xy[threadIdx.x] += body_virial_xy[threadIdx.x + offset];
            body_virial_xz[threadIdx.x] += body_virial_xz[threadIdx.x + offset];
            body_virial_yy[threadIdx.x] += body_virial_yy[threadIdx.x + offset];
            body_virial_yz[threadIdx.x] += body_virial_yz[threadIdx.x + offset];
            body_virial_zz[threadIdx.x] += body_virial_zz[threadIdx.x + offset];
            }

        offset >>= 1;

        __syncthreads();
        }

    // thread 0 within this body writes out the total virial for the body
    if ((threadIdx.x & thread_mask) == 0 && mol_idx[m] < MIN_FLOPPY && central_idx[m] < N)
        {
        d_virial[0 * virial_pitch + central_idx[m]] = body_virial_xx[threadIdx.x];
        d_virial[1 * virial_pitch + central_idx[m]] = body_virial_xy[threadIdx.x];
        d_virial[2 * virial_pitch + central_idx[m]] = body_virial_xz[threadIdx.x];
        d_virial[3 * virial_pitch + central_idx[m]] = body_virial_yy[threadIdx.x];
        d_virial[4 * virial_pitch + central_idx[m]] = body_virial_yz[threadIdx.x];
        d_virial[5 * virial_pitch + central_idx[m]] = body_virial_zz[threadIdx.x];
        }
    }

kernel void gpu_update_composite_kernel(HOOMD_KERNEL_PARAMS(kernel_args::gpu_update_composite_kernel_args))
    {
    unsigned int N = args.N;
    unsigned int nwork = args.nwork;
    unsigned int n_ghost = args.n_ghost;
    device const unsigned int* d_lookup_center = args.d_lookup_center;
    device Scalar4* d_postype = args.d_postype;
    device Scalar4* d_velocity = args.d_velocity;
    device Scalar4* d_orientation = args.d_orientation;
    device const Scalar4* d_angmom = args.d_angmom;
    device const Scalar3* d_inertia = args.d_inertia;
    Index2D body_indexer = args.body_indexer;
    device const Scalar3* d_body_pos = args.d_body_pos;
    device const Scalar4* d_body_orientation = args.d_body_orientation;
    device const unsigned int* d_body_types = args.d_body_types;
    device const unsigned int* d_body_len = args.d_body_len;
    device const unsigned int* d_molecule_order = args.d_molecule_order;
    device const unsigned int* d_molecule_len = args.d_molecule_len;
    device const unsigned int* d_molecule_idx = args.d_molecule_idx;
    device int3* d_image = args.d_image;
    const BoxDim box = args.box;
    const BoxDim global_box = args.global_box;
    device uint2* d_flag = args.d_flag;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nwork)
        return;

    unsigned int central_idx = d_lookup_center[idx];
    if (central_idx >= MIN_FLOPPY)
        return;

    if (central_idx >= N + n_ghost)
        {
        // If the central particle is not local, then we cannot update the position and orientation
        // of this particle. Ideally, this would perform an error check. However, that is not
        // feasible as ForceComposite does not have knowledge of which ghost particles are within
        // the interaction ghost width (and need therefore need to be updated) vs those that are
        // communicated to make bodies whole.
        return;
        }

    // do not overwrite central ptl
    if (idx == central_idx)
        return;

    Scalar4 postype = d_postype[central_idx];
    vec3<Scalar> pos(postype);
    quat<Scalar> orientation(d_orientation[central_idx]);

    unsigned int body_type = __scalar_as_int(postype.w);

    unsigned int body_len = d_body_len[body_type];
    unsigned int mol_idx = d_molecule_idx[idx];

    // Checks if the number of local particle in the molecule is equal to the number of particles in
    // the rigid body definition `body_len`. As above, this error check *should* be performed for
    // all local and ghost particles within the interaction ghost width. However, that check is not
    // feasible here. At least catch this error for particles local to this rank.
    if (body_len != d_molecule_len[mol_idx] - 1)
        {
        if (idx < N)
            {
            atomicMax((device unsigned int*)d_flag + 1, idx + 1);
            }

        // otherwise, ignore
        return;
        }

    int3 img = d_image[central_idx];

    unsigned int idx_in_body = d_molecule_order[idx] - 1;

    vec3<Scalar> local_pos(d_body_pos[body_indexer(body_type, idx_in_body)]);
    vec3<Scalar> dr_space = rotate(orientation, local_pos);

    vec3<Scalar> updated_pos(pos);
    updated_pos += dr_space;

    quat<Scalar> local_orientation(d_body_orientation[body_indexer(body_type, idx_in_body)]);
    quat<Scalar> updated_orientation = orientation * local_orientation;

    /* Update velocity of constituents
     *
     * Calculate the angular momentum in the body frame as a vector,
     * then divide by the moment of inertia to get the angular velocity.
     * The tangential velocity in the body frame is the angular velocity
     * cross the position relative to the center of mass, which is then
     * rotated into the space frame and added to the velocity of the center
     * of mass.
     */
    const quat<Scalar> angmom(d_angmom[central_idx]);
    const vec3<Scalar> inertia(d_inertia[central_idx]);
    const quat<Scalar> angvel_body_quat = Scalar(0.5) * conj(orientation) * angmom;
    vec3<Scalar> angvel_body = angvel_body_quat.v;
    if (inertia.x != Scalar(0))
        {
        angvel_body.x /= inertia.x;
        }
    else
        {
        angvel_body.x = Scalar(0);
        }

    if (inertia.y != Scalar(0))
        {
        angvel_body.y /= inertia.y;
        }
    else
        {
        angvel_body.y = Scalar(0);
        }

    if (inertia.z != Scalar(0))
        {
        angvel_body.z /= inertia.z;
        }
    else
        {
        angvel_body.z = Scalar(0);
        }

    vec3<Scalar> updated_vel = vec3<Scalar>(d_velocity[central_idx])
                               + rotate(orientation, cross(angvel_body, local_pos));
    // this runs before the ForceComputes,
    // wrap into box, allowing rigid bodies to span multiple images
    int3 imgi = box.getImage(vec_to_scalar3(updated_pos));
    int3 negimgi = make_int3(-imgi.x, -imgi.y, -imgi.z);
    global_box.shift(updated_pos, updated_vel, negimgi);

    unsigned int type = __scalar_as_int(d_postype[idx].w);
    const Scalar mass = d_velocity[idx].w;

    d_postype[idx]
        = make_scalar4(updated_pos.x,
                       updated_pos.y,
                       updated_pos.z,
                       __int_as_scalar(d_body_types[body_indexer(body_type, idx_in_body)]));
    d_velocity[idx] = make_scalar4(updated_vel.x, updated_vel.y, updated_vel.z, mass);
    d_orientation[idx] = quat_to_scalar4(updated_orientation);
    d_image[idx] = img + imgi;
    }

#else

namespace kernel
    {
//! Calculates the body forces and torques by summing the constituent particle forces using a fixed
//! sliding window size
/*  Compute the force and torque sum on all bodies in the system from their constituent particles.
   n_bodies_per_block bodies are handled within each block of execution on the GPU. The reason for
   this is to decrease over-parallelism and use the GPU cores more effectively when bodies are
   smaller than the block size. Otherwise, small bodies leave many threads in the block idle with
   nothing to do.

    On start, the properties common to each body are read in, computed, and stored in shared memory
   for all the threads working on that body to access. Then, the threads loop over all particles
   that are part of the body with a sliding window. Each loop of the window computes the force and
   torque for block_size/n_bodies_per_block particles in as many threads in parallel. These
   quantities are summed over enough windows to cover the whole body.

    The block_size/n_bodies_per_block partial sums are stored in shared memory. Then
   n_bodies_per_block partial reductions are performed in parallel using all threads to sum the
   total force and torque on each body. This looks just like a normal reduction, except that it
   terminates at a certain level in the tree. To make the math for the partial reduction work out,
   block_size must be a power of 2 as must n_bodies_per_block.

    Performance testing on GF100 with many different bodies of different sizes ranging from 4-256
   particles per body has found that the optimum block size for most bodies is 64 threads.
   Performance increases for all body sizes as n_bodies_per_block is increased, but only up to 8.
   n_bodies_per_block=16 slows performance significantly. Based on these performance results, this
   kernel is hardcoded to handle only 1,2,4,8 n_bodies_per_block with a power of 2 block size
   (hardcoded to 64 in the kernel launch).
*/




/*!
 */
hipError_t gpu_rigid_force(Scalar4* d_force,
                           Scalar4* d_torque,
                           const unsigned int* d_molecule_len,
                           const unsigned int* d_molecule_list,
                           const unsigned int* d_molecule_idx,
                           const unsigned int* d_rigid_center,
                           Index2D molecule_indexer,
                           const Scalar4* d_postype,
                           const Scalar4* d_orientation,
                           Index2D body_indexer,
                           Scalar3* d_body_pos,
                           Scalar4* d_body_orientation,
                           const unsigned int* d_body_len,
                           const unsigned int* d_body,
                           const unsigned int* d_tag,
                           uint2* d_flag,
                           Scalar4* d_net_force,
                           Scalar4* d_net_torque,
                           unsigned int n_mol,
                           unsigned int N,
                           unsigned int n_bodies_per_block,
                           unsigned int block_size,
                           const hipDeviceProp_t& dev_prop,
                           bool zero_force,
                           unsigned int n_local_bodies)
    {
    unsigned int nwork = n_local_bodies;

    dim3 force_grid(nwork / n_bodies_per_block + 1, 1, 1);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_rigid_force_sliding_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = max_block_size < block_size ? max_block_size : block_size;

    // round down to nearest power of two
    unsigned int b = 1;
    while (b * 2 <= run_block_size)
        {
        b *= 2;
        }
    run_block_size = b;

    unsigned int window_size = run_block_size / n_bodies_per_block;
    unsigned int thread_mask = window_size - 1;

    size_t shared_bytes = run_block_size * (sizeof(Scalar4) + sizeof(Scalar3))
                          + n_bodies_per_block * (sizeof(Scalar4) + 3 * sizeof(unsigned int));

    while (shared_bytes + attr.sharedSizeBytes >= dev_prop.sharedMemPerBlock)
        {
        // block size is power of two
        run_block_size /= 2;

        shared_bytes = run_block_size * (sizeof(Scalar4) + sizeof(Scalar3))
                       + n_bodies_per_block * (sizeof(Scalar4) + 3 * sizeof(unsigned int));

        window_size = run_block_size / n_bodies_per_block;
        thread_mask = window_size - 1;
        }

    {
        kernel_args::gpu_rigid_force_sliding_kernel_args _args;
        _args.d_force = d_force;
        _args.d_torque = d_torque;
        _args.d_molecule_len = d_molecule_len;
        _args.d_molecule_list = d_molecule_list;
        _args.d_molecule_idx = d_molecule_idx;
        _args.d_rigid_center = d_rigid_center;
        _args.molecule_indexer = molecule_indexer;
        _args.d_postype = d_postype;
        _args.d_orientation = d_orientation;
        _args.body_indexer = body_indexer;
        _args.d_body_pos = d_body_pos;
        _args.d_body_orientation = d_body_orientation;
        _args.d_body_len = d_body_len;
        _args.d_body = d_body;
        _args.d_tag = d_tag;
        _args.d_flag = d_flag;
        _args.d_net_force = d_net_force;
        _args.d_net_torque = d_net_torque;
        _args.n_mol = n_mol;
        _args.N = N;
        _args.window_size = window_size;
        _args.thread_mask = thread_mask;
        _args.n_bodies_per_block = n_bodies_per_block;
        _args.zero_force = zero_force;
        _args.nwork = nwork;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_rigid_force_sliding_kernel",
                           dim3(force_grid),
                           dim3(run_block_size),
                           _args,
                           shared_bytes);
        }
    return hipSuccess;
    }

hipError_t gpu_rigid_virial(Scalar* d_virial,
                            const unsigned int* d_molecule_len,
                            const unsigned int* d_molecule_list,
                            const unsigned int* d_molecule_idx,
                            const unsigned int* d_rigid_center,
                            Index2D molecule_indexer,
                            const Scalar4* d_postype,
                            const Scalar4* d_orientation,
                            Index2D body_indexer,
                            Scalar3* d_body_pos,
                            Scalar4* d_body_orientation,
                            Scalar4* d_net_force,
                            Scalar* d_net_virial,
                            const unsigned int* d_body,
                            const unsigned int* d_tag,
                            unsigned int n_mol,
                            unsigned int N,
                            unsigned int n_bodies_per_block,
                            size_t net_virial_pitch,
                            size_t virial_pitch,
                            unsigned int block_size,
                            const hipDeviceProp_t& dev_prop,
                            unsigned int n_local_bodies)
    {
    unsigned int nwork = n_local_bodies;

    dim3 force_grid(nwork / n_bodies_per_block + 1, 1, 1);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_rigid_virial_sliding_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = max_block_size < block_size ? max_block_size : block_size;

    // round down to nearest power of two
    unsigned int b = 1;
    while (b * 2 <= run_block_size)
        {
        b *= 2;
        }
    run_block_size = b;

    unsigned int window_size = run_block_size / n_bodies_per_block;
    unsigned int thread_mask = window_size - 1;

    size_t shared_bytes = 6 * run_block_size * sizeof(Scalar)
                          + n_bodies_per_block * (sizeof(Scalar4) + 3 * sizeof(unsigned int));

    while (shared_bytes + attr.sharedSizeBytes >= dev_prop.sharedMemPerBlock)
        {
        // block size is power of two
        run_block_size /= 2;

        shared_bytes = 6 * run_block_size * sizeof(Scalar)
                       + n_bodies_per_block * (sizeof(Scalar4) + 3 * sizeof(unsigned int));

        window_size = run_block_size / n_bodies_per_block;
        thread_mask = window_size - 1;
        }

    {
        kernel_args::gpu_rigid_virial_sliding_kernel_args _args;
        _args.d_virial = d_virial;
        _args.d_molecule_len = d_molecule_len;
        _args.d_molecule_list = d_molecule_list;
        _args.d_molecule_idx = d_molecule_idx;
        _args.d_rigid_center = d_rigid_center;
        _args.molecule_indexer = molecule_indexer;
        _args.d_postype = d_postype;
        _args.d_orientation = d_orientation;
        _args.body_indexer = body_indexer;
        _args.d_body_pos = d_body_pos;
        _args.d_body_orientation = d_body_orientation;
        _args.d_net_force = d_net_force;
        _args.d_net_virial = d_net_virial;
        _args.d_body = d_body;
        _args.d_tag = d_tag;
        _args.n_mol = n_mol;
        _args.N = N;
        _args.net_virial_pitch = net_virial_pitch;
        _args.virial_pitch = virial_pitch;
        _args.window_size = window_size;
        _args.thread_mask = thread_mask;
        _args.n_bodies_per_block = n_bodies_per_block;
        _args.nwork = nwork;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_rigid_virial_sliding_kernel",
                           dim3(force_grid),
                           dim3(run_block_size),
                           _args,
                           shared_bytes);
        }

    return hipSuccess;
    }



void gpu_update_composite(unsigned int N,
                          unsigned int n_ghost,
                          Scalar4* d_postype,
                          Scalar4* d_velocity,
                          Scalar4* d_orientation,
                          const Scalar4* d_angmom,
                          const Scalar3* d_inertia,
                          Index2D body_indexer,
                          const unsigned int* d_lookup_center,
                          const Scalar3* d_body_pos,
                          const Scalar4* d_body_orientation,
                          const unsigned int* d_body_types,
                          const unsigned int* d_body_len,
                          const unsigned int* d_molecule_order,
                          const unsigned int* d_molecule_len,
                          const unsigned int* d_molecule_idx,
                          int3* d_image,
                          const BoxDim box,
                          const BoxDim global_box,
                          unsigned int block_size,
                          uint2* d_flag)
    {
    unsigned int run_block_size = block_size;

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hoomd::mtl::get_function_attributes(&attr, HOOMD_MTL_SOURCE, "hoomd::md::gpu_update_composite_kernel");
    max_block_size = attr.maxThreadsPerBlock;

    if (max_block_size <= run_block_size)
        {
        run_block_size = max_block_size;
        }

    unsigned int nwork = N + n_ghost;

    unsigned int n_blocks = nwork / run_block_size + 1;
    {
        kernel_args::gpu_update_composite_kernel_args _args;
        _args.N = N;
        _args.nwork = nwork;
        _args.n_ghost = n_ghost;
        _args.d_lookup_center = d_lookup_center;
        _args.d_postype = d_postype;
        _args.d_velocity = d_velocity;
        _args.d_orientation = d_orientation;
        _args.d_angmom = d_angmom;
        _args.d_inertia = d_inertia;
        _args.body_indexer = body_indexer;
        _args.d_body_pos = d_body_pos;
        _args.d_body_orientation = d_body_orientation;
        _args.d_body_types = d_body_types;
        _args.d_body_len = d_body_len;
        _args.d_molecule_order = d_molecule_order;
        _args.d_molecule_len = d_molecule_len;
        _args.d_molecule_idx = d_molecule_idx;
        _args.d_image = d_image;
        _args.box = box;
        _args.global_box = global_box;
        _args.d_flag = d_flag;
        hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                           "hoomd::md::gpu_update_composite_kernel",
                           dim3(n_blocks),
                           dim3(run_block_size),
                           _args,
                           0);
        }
    }

hipError_t gpu_find_rigid_centers(const unsigned int* d_body,
                                  const unsigned int* d_tag,
                                  const unsigned int* d_rtag,
                                  const unsigned int N,
                                  const unsigned int nghost,
                                  unsigned int* d_rigid_center,
                                  unsigned int* d_lookup_center,
                                  unsigned int& n_rigid)
    {
    unsigned int n = N + nghost;
    n_rigid = 0;
    if (n == 0)
        return hipSuccess;

    mtl::ScratchScope scratch;
    unsigned int* d_is_center = scratch.allocate<unsigned int>(n);
    unsigned int* d_count = scratch.allocate<unsigned int>(n);
    unsigned int* d_num_selected = scratch.allocate<unsigned int>(1);

    kernel_args::gpu_find_rigid_centers_kernel_args args;
    args.d_body = d_body;
    args.d_tag = d_tag;
    args.d_rtag = d_rtag;
    args.d_is_center = d_is_center;
    args.d_lookup_center = d_lookup_center;
    args.n = n;
    const unsigned int block_size = 256;
    hipError_t error = mtl::launch(HOOMD_MTL_SOURCE,
                                   "hoomd::md::gpu_find_rigid_centers_kernel",
                                   dim3(n / block_size + 1),
                                   dim3(block_size),
                                   args);
    if (error != hipSuccess)
        return error;

    // create a contiguous list of rigid center indices
    mtl::sequence(d_count, n);
    mtl::select_flagged(d_count, d_is_center, d_rigid_center, d_num_selected, n);
    return hipMemcpy(&n_rigid, d_num_selected, sizeof(unsigned int), hipMemcpyDeviceToHost);
    }

    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
