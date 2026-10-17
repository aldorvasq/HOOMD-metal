// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file NeighborListGPUTree.metal
    \brief Metal port of NeighborListGPUTree.cu and the LBVH of the neighbor library

    NeighborListGPUTree.cu builds and traverses linear bounding volume hierarchies with the
    neighbor library (hoomd/extern/neighbor), which is written for CUDA and HIP. This file
    implements the same algorithms for Metal and provides the neighbor::LBVH and
    neighbor::LBVHTraverser classes that the wrappers in NeighborListGPUTree.cuh hold:

    - LBVH build (Karras 2012): 30-bit Morton codes in a 1024^3 grid, a stable radix sort of the
      codes, the binary radix tree, and a bottom-up merge of the bounding boxes.
    - Traversal (neighbor::LBVHTraverser): the tree is compressed to one int4 per node holding its
      quantized bounding box, left child (or primitive), and rope. Each query sphere is tested
      against the tree for every periodic image that overlaps the root.

    The neighbor library uses directed rounding intrinsics (__fadd_rd and friends) so that the
    quantized boxes and overlap tests are conservative. Metal has none, so the approx_* functions
    below compute the round-to-nearest result and step it by one ulp when the exact error shows
    that the result is on the wrong side.
*/

#include "hoomd/mtl/MetalKernel.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifndef __METAL_VERSION__
#include "NeighborListGPUTree.cuh"

#include "hoomd/mtl/Primitives.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel_args
    {
struct nlist_mark_types_args
    {
    HOOMD_DEVPTR(unsigned int) d_types;
    HOOMD_DEVPTR(unsigned int) d_indexes;
    HOOMD_DEVPTR(unsigned int) d_lbvh_errors;
    HOOMD_DEVPTR(Scalar4) d_last_pos;
    HOOMD_DEVPTR(const Scalar4) d_pos;
    unsigned int N;
    unsigned int nghosts;
    BoxDim box;
    Scalar3 ghost_width;
    };

struct nlist_count_types_args
    {
    HOOMD_DEVPTR(unsigned int) d_first;
    HOOMD_DEVPTR(unsigned int) d_last;
    HOOMD_DEVPTR(const unsigned int) d_types;
    unsigned int ntypes;
    unsigned int N;
    };

struct nlist_copy_primitives_args
    {
    HOOMD_DEVPTR(unsigned int) d_traverse_order;
    HOOMD_DEVPTR(const unsigned int) d_indexes;
    HOOMD_DEVPTR(const unsigned int) d_primitives;
    unsigned int N;
    };

//! Device pointers to an LBVH (neighbor::LBVHData)
struct lbvh_data
    {
    HOOMD_DEVPTR(int) parent;
    HOOMD_DEVPTR(int) left;
    HOOMD_DEVPTR(int) right;
    HOOMD_DEVPTR(unsigned int) primitive;
    HOOMD_DEVPTR(float3) lo;
    HOOMD_DEVPTR(float3) hi;
    };

struct lbvh_gen_codes_args
    {
    HOOMD_DEVPTR(unsigned int) d_codes;
    HOOMD_DEVPTR(unsigned int) d_indexes;
    HOOMD_DEVPTR(const Scalar4) points;
    HOOMD_DEVPTR(const unsigned int) map;
    float3 lo;
    float3 hi;
    unsigned int N;
    };

struct lbvh_gen_tree_args
    {
    lbvh_data tree;
    HOOMD_DEVPTR(const unsigned int) d_codes;
    unsigned int N;
    };

struct lbvh_bubble_aabbs_args
    {
    lbvh_data tree;
    HOOMD_DEVPTR(const Scalar4) points;
    HOOMD_DEVPTR(const unsigned int) map;
    HOOMD_DEVPTR(unsigned int) d_locks;
    unsigned int N;
    };

struct lbvh_compress_ropes_args
    {
    lbvh_data tree;
    HOOMD_DEVPTR(int4) data;
    HOOMD_DEVPTR(float3) ctree_lo;
    HOOMD_DEVPTR(float3) ctree_hi;
    HOOMD_DEVPTR(float3) ctree_bins;
    HOOMD_DEVPTR(const unsigned int) transform_map;
    int root;
    unsigned int N_internal;
    unsigned int N_nodes;
    };

struct lbvh_traverse_ropes_args
    {
    // compressed LBVH
    HOOMD_DEVPTR(const int4) data;
    HOOMD_DEVPTR(const float3) tree_lo;
    HOOMD_DEVPTR(const float3) tree_hi;
    HOOMD_DEVPTR(const float3) tree_bins;
    int root;

    // ParticleQueryOp
    HOOMD_DEVPTR(const Scalar4) positions;
    HOOMD_DEVPTR(const unsigned int) bodies;
    HOOMD_DEVPTR(const unsigned int) order;
    unsigned int N;
    unsigned int Nown;
    Scalar rlist;

    // NeighborListOp
    HOOMD_DEVPTR(uint4) neigh_list;
    HOOMD_DEVPTR(unsigned int) nneigh;
    HOOMD_DEVPTR(unsigned int) new_max_neigh;
    HOOMD_DEVPTR(const size_t) first_neigh;
    unsigned int max_neigh;

    // ImageListOp
    HOOMD_DEVPTR(const Scalar3) images;
    unsigned int n_images;
    };
    } // end namespace kernel_args

#ifdef __METAL_VERSION__

constant unsigned int NeighborListTypeSentinel = 0xffffffff;
constant int LBVHSentinel = 0x7fffffff;

constant bool use_body [[function_constant(0)]];

//! Step x by one ulp toward -inf
inline float approx_down(float x)
    {
    return nextafter(x, -INFINITY);
    }

//! Step x by one ulp toward +inf
inline float approx_up(float x)
    {
    return nextafter(x, INFINITY);
    }

//! Exact error of s = fl(a + b): a + b = s + err (Knuth's TwoSum)
inline float two_sum_error(float a, float b, float s)
    {
    float bb = s - a;
    return (a - (s - bb)) + (b - bb);
    }

inline float approx_fadd_rd(float a, float b)
    {
    float s = a + b;
    return two_sum_error(a, b, s) < 0.0f ? approx_down(s) : s;
    }

inline float approx_fadd_ru(float a, float b)
    {
    float s = a + b;
    return two_sum_error(a, b, s) > 0.0f ? approx_up(s) : s;
    }

inline float approx_fsub_rd(float a, float b)
    {
    return approx_fadd_rd(a, -b);
    }

inline float approx_fsub_ru(float a, float b)
    {
    return approx_fadd_ru(a, -b);
    }

inline float approx_fmul_rd(float a, float b)
    {
    float p = a * b;
    return fma(a, b, -p) < 0.0f ? approx_down(p) : p;
    }

inline float approx_fmul_ru(float a, float b)
    {
    float p = a * b;
    return fma(a, b, -p) > 0.0f ? approx_up(p) : p;
    }

inline float approx_fdiv_rd(float a, float b)
    {
    float q = a / b;
    // a - q*b has the sign of b when q is too small and the opposite sign when it is too large
    for (int i = 0; i < 4; ++i)
        {
        float r = fma(-q, b, a);
        if (r != 0.0f && ((r < 0.0f) != (b < 0.0f)))
            q = approx_down(q);
        else
            break;
        }
    return q;
    }

inline float approx_frcp_rd(float x)
    {
    return approx_fdiv_rd(1.0f, x);
    }

//! a*b + c rounded toward -inf
inline float approx_fmaf_rd(float a, float b, float c)
    {
    // a*b = ph + pl exactly, then ph + c = s + e exactly
    float ph = a * b;
    float pl = fma(a, b, -ph);
    float s = ph + c;
    float e = two_sum_error(ph, c, s);
    float r = s + (e + pl);
    float err = two_sum_error(s, e + pl, r);
    return err < 0.0f ? approx_down(r) : r;
    }

//! Axis aligned bounding box (neighbor::BoundingBox)
struct lbvh_box
    {
    float3 lo;
    float3 hi;
    };

//! Sphere that may be skipped (SkippableBoundingSphere)
struct lbvh_sphere
    {
    float3 origin;
    float Rsq;
    bool skip;
    };

inline lbvh_sphere make_lbvh_sphere(float3 o, float r)
    {
    lbvh_sphere s;
    s.origin = o;
    s.Rsq = approx_fmul_ru(r, r);
    s.skip = !(r > 0.0f);
    return s;
    }

//! Sphere-box overlap test (neighbor::BoundingSphere::overlap)
inline bool lbvh_overlap(lbvh_sphere s, lbvh_box box)
    {
    if (s.skip)
        return false;
    const float3 dr = float3(approx_fsub_rd(fmin(fmax(s.origin.x, box.lo.x), box.hi.x), s.origin.x),
                             approx_fsub_rd(fmin(fmax(s.origin.y, box.lo.y), box.hi.y), s.origin.y),
                             approx_fsub_rd(fmin(fmax(s.origin.z, box.lo.z), box.hi.z), s.origin.z));
    const float dr2
        = approx_fmaf_rd(dr.x, dr.x, approx_fmaf_rd(dr.y, dr.y, approx_fmul_rd(dr.z, dr.z)));
    return (dr2 <= s.Rsq);
    }

//! Bounding box of a point (PointMapInsertOp::get)
inline float3 lbvh_point(device const Scalar4* points, device const unsigned int* map, uint idx)
    {
    const Scalar4 point = points[map[idx]];
    return float3(point.x, point.y, point.z);
    }

//! Expand a 10-bit integer into 30 bits by inserting 2 zeros after each bit
inline unsigned int lbvh_expand_bits(unsigned int v)
    {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
    }

//! Bin a fractional coordinate in [0,1] into 1024 bins
inline unsigned int lbvh_fraction_to_bin(float f)
    {
    return static_cast<unsigned int>(fmin(fmax(f * 1023.f, 0.f), 1023.f));
    }

//! Length of the common prefix of the codes of i and j (ties broken by the index)
inline int lbvh_delta(device const unsigned int* d_codes,
                      const unsigned int code_i,
                      const int i,
                      const int j,
                      const unsigned int N)
    {
    if (j < 0 || j >= (int)N)
        {
        return -1;
        }

    const unsigned int code_j = d_codes[j];
    if (code_i == code_j)
        {
        return (32 + clz(uint(i ^ j)));
        }
    else
        {
        return clz(code_i ^ code_j);
        }
    }

kernel void gpu_nlist_mark_types_kernel(HOOMD_KERNEL_PARAMS(kernel_args::nlist_mark_types_args))
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= args.N + args.nghosts)
        return;

    // get particle information
    Scalar4 postype = args.d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __scalar_as_int(postype.w);

    // get position in simulation box
    BoxDim box = args.box;
    Scalar3 f = box.makeFraction(pos, args.ghost_width);

    /* check if the particle is inside the unit cell + ghost layer.
     * we silently ignore ghosts outside of this width, since they could be in bonds, etc.
     */
    if ((f.x < Scalar(-0.00001) || f.x >= Scalar(1.00001))
        || (f.y < Scalar(-0.00001) || f.y >= Scalar(1.00001))
        || (f.z < Scalar(-0.00001) || f.z >= Scalar(1.00001)))
        {
        // error for owned particle
        if (idx < args.N)
            {
            atomicMax(args.d_lbvh_errors, idx + 1);
            return;
            }
        else // silent for ghosts
            {
            type = NeighborListTypeSentinel;
            }
        }

    args.d_types[idx] = type;
    args.d_indexes[idx] = idx;
    // record as "last" position of owned particles
    if (idx < args.N)
        {
        args.d_last_pos[idx] = postype;
        }
    }

kernel void gpu_nlist_count_types_kernel(HOOMD_KERNEL_PARAMS(kernel_args::nlist_count_types_args))
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int N = args.N;
    if (idx >= N)
        return;

    // my type
    const unsigned int type = args.d_types[idx];
    // look to left if not first
    const unsigned int left = (idx > 0) ? args.d_types[idx - 1] : NeighborListTypeSentinel;
    // look to right if not last
    const unsigned int right = (idx < N - 1) ? args.d_types[idx + 1] : NeighborListTypeSentinel;

    // if left is not same as self (or idx == 0 by use of sentinel), this is the first index
    if (left != type && type < args.ntypes)
        {
        args.d_first[type] = idx;
        }
    // if right is not the same as self (or idx == N-1 by use of sentinel), this is the last index
    if (right != type && type < args.ntypes)
        {
        args.d_last[type] = idx + 1;
        }
    }

kernel void
gpu_nlist_copy_primitives_kernel(HOOMD_KERNEL_PARAMS(kernel_args::nlist_copy_primitives_args))
    {
    // one thread per particle
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= args.N)
        return;

    const unsigned int primitive = args.d_primitives[idx];
    args.d_traverse_order[idx] = args.d_indexes[primitive];
    }

kernel void lbvh_gen_codes(HOOMD_KERNEL_PARAMS(kernel_args::lbvh_gen_codes_args))
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.N)
        return;

    const float3 r = lbvh_point(args.points, args.map, idx);
    const float3 lo = args.lo;
    const float3 hi = args.hi;
    const float3 f = float3((r.x - lo.x) / (hi.x - lo.x),
                            (r.y - lo.y) / (hi.y - lo.y),
                            (r.z - lo.z) / (hi.z - lo.z));
    const unsigned int code = 4 * lbvh_expand_bits(lbvh_fraction_to_bin(f.x))
                              + 2 * lbvh_expand_bits(lbvh_fraction_to_bin(f.y))
                              + lbvh_expand_bits(lbvh_fraction_to_bin(f.z));
    args.d_codes[idx] = code;
    args.d_indexes[idx] = idx;
    }

kernel void lbvh_gen_tree(HOOMD_KERNEL_PARAMS(kernel_args::lbvh_gen_tree_args))
    {
    // one thread per internal node (= N-1 threads)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int N = args.N;
    if (i >= (int)N - 1)
        return;

    device const unsigned int* d_codes = args.d_codes;
    const unsigned int code_i = d_codes[i];
    const int forward_prefix = lbvh_delta(d_codes, code_i, i, i + 1, N);
    const int backward_prefix = lbvh_delta(d_codes, code_i, i, i - 1, N);

    // get direction of the range based on sign
    const int d = (forward_prefix >= backward_prefix) - (forward_prefix < backward_prefix);

    // get minimum prefix
    const int min_prefix = lbvh_delta(d_codes, code_i, i, i - d, N);

    // get maximum prefix by binary search
    int lmax = 2;
    while (lbvh_delta(d_codes, code_i, i, i + d * lmax, N) > min_prefix)
        {
        lmax = lmax << 1;
        }
    int l = 0;
    int t = lmax;
    do
        {
        t = t >> 1;
        if (lbvh_delta(d_codes, code_i, i, i + (l + t) * d, N) > min_prefix)
            l = l + t;
        } while (t > 1);
    const int j = i + l * d;

    // get the length of the common prefix
    const int common_prefix = lbvh_delta(d_codes, code_i, i, j, N);

    // binary search to find split position
    int s = 0;
    t = l;
    do
        {
        t = (t + 1) >> 1;
        // if proposed split lies within range
        if (s + t < l)
            {
            const int split_prefix = lbvh_delta(d_codes, code_i, i, i + (s + t) * d, N);

            // if new split shares a longer number of bits, accept it
            if (split_prefix > common_prefix)
                {
                s = s + t;
                }
            }
        } while (t > 1);
    const int split = i + s * d + min(d, 0);

    const int left = (min(i, j) == split) ? split + ((int)N - 1) : split;
    const int right = (max(i, j) == (split + 1)) ? split + (int)N : split + 1;

    // children
    args.tree.left[i] = left;
    args.tree.right[i] = right;

    // parents
    args.tree.parent[left] = i;
    args.tree.parent[right] = i;

    // root node (index 0) has no parent
    if (i == 0)
        {
        args.tree.parent[0] = LBVHSentinel;
        }
    }

//! Store a bounding box component with device-coherent atomics
inline void lbvh_store_bounds(device float3* p, float3 v)
    {
    device atomic_float* a = (device atomic_float*)p;
    atomic_store_explicit(a + 0, v.x, memory_order_relaxed);
    atomic_store_explicit(a + 1, v.y, memory_order_relaxed);
    atomic_store_explicit(a + 2, v.z, memory_order_relaxed);
    }

//! Load a bounding box component with device-coherent atomics
inline float3 lbvh_load_bounds(device float3* p)
    {
    device atomic_float* a = (device atomic_float*)p;
    return float3(atomic_load_explicit(a + 0, memory_order_relaxed),
                  atomic_load_explicit(a + 1, memory_order_relaxed),
                  atomic_load_explicit(a + 2, memory_order_relaxed));
    }

/*! Merge the bounding boxes from the leaves to the root. The second thread to arrive at each
    internal node continues upward, so every node is processed once its children are complete.
    Bounds pass between threadgroups through device memory, so they are read and written with
    atomics and ordered with device scope fences.
*/
kernel void lbvh_bubble_aabbs(HOOMD_KERNEL_PARAMS(kernel_args::lbvh_bubble_aabbs_args))
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int N = args.N;
    if (idx >= N)
        return;

    // okay to just point to primitives since leaves are points
    const float3 p = lbvh_point(args.points, args.map, args.tree.primitive[idx]);
    float3 lo = p;
    float3 hi = p;

    // set aabb for the leaf node
    int last = N - 1 + idx;
    lbvh_store_bounds(args.tree.lo + last, lo);
    lbvh_store_bounds(args.tree.hi + last, hi);
    __threadfence();

    int current = args.tree.parent[last];
    while (current != LBVHSentinel)
        {
        // only the second thread to reach a node processes it
        unsigned int lock = atomic_fetch_add_explicit((device atomic_uint*)(args.d_locks + current),
                                                      1u,
                                                      memory_order_relaxed);
        if (!lock)
            return;
        __threadfence();

        // look for the sibling of the current thread with speculation
        int sibling = args.tree.left[current];
        if (sibling == last)
            {
            sibling = args.tree.right[current];
            }

        // compute min / max bounds of the current thread with its sibling
        const float3 sib_lo = lbvh_load_bounds(args.tree.lo + sibling);
        if (sib_lo.x < lo.x)
            lo.x = sib_lo.x;
        if (sib_lo.y < lo.y)
            lo.y = sib_lo.y;
        if (sib_lo.z < lo.z)
            lo.z = sib_lo.z;

        const float3 sib_hi = lbvh_load_bounds(args.tree.hi + sibling);
        if (sib_hi.x > hi.x)
            hi.x = sib_hi.x;
        if (sib_hi.y > hi.y)
            hi.y = sib_hi.y;
        if (sib_hi.z > hi.z)
            hi.z = sib_hi.z;

        // write out to memory
        lbvh_store_bounds(args.tree.lo + current, lo);
        lbvh_store_bounds(args.tree.hi + current, hi);

        // move up tree
        __threadfence();
        last = current;
        current = args.tree.parent[current];
        }
    }

kernel void lbvh_one_primitive(HOOMD_KERNEL_PARAMS(kernel_args::lbvh_bubble_aabbs_args))
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1)
        return;

    const float3 p = lbvh_point(args.points, args.map, 0);
    args.tree.parent[0] = LBVHSentinel;
    args.tree.primitive[0] = 0;
    args.tree.lo[0] = p;
    args.tree.hi[0] = p;
    }

/*! Compress the LBVH into one int4 per node: quantized lower and upper bounds, the left child
    (or the bitwise complement of the transformed primitive for a leaf), and the rope (the node to
    visit next when this subtree is skipped).
*/
kernel void lbvh_compress_ropes(HOOMD_KERNEL_PARAMS(kernel_args::lbvh_compress_ropes_args))
    {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load the tree extent for the compression
    const float3 tree_lo = args.tree.lo[args.root];
    const float3 tree_hi = args.tree.hi[args.root];
    float3 L = float3(approx_fsub_ru(tree_hi.x, tree_lo.x),
                      approx_fsub_ru(tree_hi.y, tree_lo.y),
                      approx_fsub_ru(tree_hi.z, tree_lo.z));
    if (L.x <= 0.f)
        L.x = 1.0f;
    if (L.y <= 0.f)
        L.y = 1.0f;
    if (L.z <= 0.f)
        L.z = 1.0f;
    const float3 tree_bininv = float3(approx_fdiv_rd(1023.f, L.x),
                                      approx_fdiv_rd(1023.f, L.y),
                                      approx_fdiv_rd(1023.f, L.z));

    if (idx >= (int)args.N_nodes)
        return;

    // compute the rope by finding the first ancestor (or self) that is a left child
    int rope = LBVHSentinel;
    int current = idx;
    while (current != args.root && rope == LBVHSentinel)
        {
        int parent = args.tree.parent[current];
        int left = args.tree.left[parent];
        if (left == current)
            {
            // if this is a left node, then rope is determined to the right
            rope = args.tree.right[parent];
            }
        else
            {
            // otherwise, keep ascending the tree
            current = parent;
            }
        }

    // compress node data into one byte per box dim
    // low bounds are encoded relative to the low of the box, always rounding down
    const float3 lo = args.tree.lo[idx];
    const uint3 lo_bin = uint3(
        (unsigned int)floor(approx_fmul_rd(approx_fsub_rd(lo.x, tree_lo.x), tree_bininv.x)),
        (unsigned int)floor(approx_fmul_rd(approx_fsub_rd(lo.y, tree_lo.y), tree_bininv.y)),
        (unsigned int)floor(approx_fmul_rd(approx_fsub_rd(lo.z, tree_lo.z), tree_bininv.z)));
    const unsigned int lo_bin3 = (lo_bin.x << 20) + (lo_bin.y << 10) + lo_bin.z;

    // high bounds are encoded relative to the high of the box, always rounding down
    const float3 hi = args.tree.hi[idx];
    const uint3 hi_bin = uint3(
        (unsigned int)floor(approx_fmul_rd(approx_fsub_rd(tree_hi.x, hi.x), tree_bininv.x)),
        (unsigned int)floor(approx_fmul_rd(approx_fsub_rd(tree_hi.y, hi.y), tree_bininv.y)),
        (unsigned int)floor(approx_fmul_rd(approx_fsub_rd(tree_hi.z, hi.z), tree_bininv.z)));
    const unsigned int hi_bin3 = (hi_bin.x << 20) + (hi_bin.y << 10) + hi_bin.z;

    // node holds left child for internal nodes (>= 0) or primitive for leaf (< 0)
    int left_flag = (idx < (int)args.N_internal)
                        ? args.tree.left[idx]
                        : ~(int)args.transform_map[args.tree.primitive[idx - args.N_internal]];

    // stash all the data into one int4
    args.data[idx] = int4(lo_bin3, hi_bin3, left_flag, rope);

    // first thread writes out the compression values, rounding down bin size to ensure box bounds
    // always expand even with floating point error
    if (idx == 0)
        {
        *args.ctree_lo = tree_lo;
        *args.ctree_hi = tree_hi;
        *args.ctree_bins = float3(approx_frcp_rd(tree_bininv.x),
                                  approx_frcp_rd(tree_bininv.y),
                                  approx_frcp_rd(tree_bininv.z));
        }
    }

/*! Traverse the compressed LBVH with one thread per query particle, testing each periodic image
    of the particle's search sphere that overlaps the tree, and append the neighbors with
    NeighborListOp's four-wide stack.
*/
kernel void lbvh_traverse_ropes(HOOMD_KERNEL_PARAMS(kernel_args::lbvh_traverse_ropes_args))
    {
    // one thread per test
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.N)
        return;

    // load tree compression
    lbvh_box tree_box;
    tree_box.lo = *args.tree_lo;
    tree_box.hi = *args.tree_hi;
    const float3 tree_bins = *args.tree_bins;

    // ParticleQueryOp::setup
    const unsigned int pidx = args.order[idx];
    const Scalar4 position = args.positions[pidx];
    const Scalar3 qpos = make_scalar3(position.x, position.y, position.z);
    unsigned int qbody = 0xffffffff;
    if (use_body)
        qbody = args.bodies[pidx];
    const Scalar qr = (pidx < args.Nown) ? args.rlist : Scalar(-1.0);

    // NeighborListOp::setup
    const size_t first = args.first_neigh[pidx];
    unsigned int num_neigh = args.nneigh[pidx];
    unsigned int stack[4] = {0, 0, 0, 0};
    if (num_neigh % 4 != 0)
        {
        uint4 tmp = args.neigh_list[(first + num_neigh - 1) / 4];
        stack[0] = tmp.x;
        stack[1] = tmp.y;
        stack[2] = tmp.z;
        stack[3] = tmp.w;
        }

    // find image flags against root before divergence
    unsigned int flags = 0;
    const int nbits = (args.n_images <= 32u) ? args.n_images : 32;
    for (int i = 0; i < nbits; ++i)
        {
        const Scalar3 image = args.images[i];
        const lbvh_sphere q = make_lbvh_sphere(qpos + image, qr);
        if (lbvh_overlap(q, tree_box))
            flags |= 1u << i;
        }

    // stackless search
    while (flags != 0)
        {
        const int image_bit = ctz(flags);
        flags &= ~(1u << image_bit);

        // move the sphere to the image
        const Scalar3 image = args.images[image_bit];
        const lbvh_sphere q = make_lbvh_sphere(qpos + image, qr);

        int node = args.root;
        while (node != LBVHSentinel)
            {
            const int4 aabb = args.data[node];
            const unsigned int lo = aabb.x;
            lbvh_box box;
            box.lo = float3(
                approx_fadd_rd(tree_box.lo.x, approx_fmul_rd(float((lo >> 20) & 0x3ffu), tree_bins.x)),
                approx_fadd_rd(tree_box.lo.y, approx_fmul_rd(float((lo >> 10) & 0x3ffu), tree_bins.y)),
                approx_fadd_rd(tree_box.lo.z, approx_fmul_rd(float((lo)&0x3ffu), tree_bins.z)));
            const unsigned int hi = aabb.y;
            box.hi = float3(
                approx_fsub_ru(tree_box.hi.x, approx_fmul_rd(float((hi >> 20) & 0x3ffu), tree_bins.x)),
                approx_fsub_ru(tree_box.hi.y, approx_fmul_rd(float((hi >> 10) & 0x3ffu), tree_bins.y)),
                approx_fsub_ru(tree_box.hi.z, approx_fmul_rd(float((hi)&0x3ffu), tree_bins.z)));
            const int left = aabb.z;

            // advance to rope as a preliminary
            node = aabb.w;

            // if overlap, do work with primitive. otherwise, rope ahead
            if (lbvh_overlap(q, box))
                {
                if (left < 0)
                    {
                    const int primitive = ~left;

                    // ParticleQueryOp::refine
                    bool exclude = ((int)pidx == primitive);
                    if (use_body && !exclude && qbody != 0xffffffff)
                        {
                        const unsigned int body = args.bodies[primitive];
                        exclude |= (qbody == body);
                        }

                    // NeighborListOp::process
                    if (!exclude)
                        {
                        if (num_neigh < args.max_neigh)
                            {
                            const unsigned int offset = num_neigh % 4;
                            stack[offset] = primitive;
                            if (offset == 3)
                                {
                                args.neigh_list[(first + num_neigh) / 4]
                                    = uint4(stack[0], stack[1], stack[2], stack[3]);
                                }
                            }
                        ++num_neigh;
                        }

                    // leaf nodes always move to their rope
                    }
                else
                    {
                    // internal node takes left child
                    node = left;
                    }
                }
            } // end stackless search
        }

    // NeighborListOp::finalize
    args.nneigh[pidx] = num_neigh;
    if (num_neigh > args.max_neigh)
        {
        atomicMax(args.new_max_neigh, num_neigh);
        }
    else if (num_neigh % 4 != 0)
        {
        args.neigh_list[(first + num_neigh - 1) / 4]
            = uint4(stack[0], stack[1], stack[2], stack[3]);
        }
    }

#else

    } // end namespace md
    } // end namespace hoomd

namespace neighbor
    {
namespace detail
    {
//! GPU array allocated through the HIP API
template<class T> class DeviceArray
    {
    public:
    DeviceArray() = default;
    ~DeviceArray()
        {
        if (m_data)
            hipFree(m_data);
        }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    //! Grow to hold at least n elements (contents are not preserved)
    void reserve(size_t n)
        {
        if (n <= m_size)
            return;
        if (m_data)
            hipFree(m_data);
        m_data = nullptr;
        if (hipMalloc((void**)&m_data, n * sizeof(T)) != hipSuccess)
            throw std::runtime_error("LBVH: failed to allocate GPU memory");
        m_size = n;
        }

    T* get() const
        {
        return m_data;
        }

    private:
    T* m_data = nullptr;
    size_t m_size = 0;
    };

//! Valid block sizes (neighbor::Tunable<unsigned int>(32, 1024, 32))
inline std::vector<unsigned int> tunable_block_sizes()
    {
    std::vector<unsigned int> params;
    for (unsigned int b = 32; b <= 1024; b += 32)
        params.push_back(b);
    return params;
    }

//! Launch a 1D kernel over n threads, limiting the block size to what the pipeline supports
template<class Args>
hipError_t launch_1d(const char* name,
                     unsigned int n,
                     unsigned int block_size,
                     const Args& args,
                     std::initializer_list<hoomd::mtl::FunctionConstant> constants = {})
    {
    if (n == 0)
        return hipSuccess;
    unsigned int max_block_size
        = hoomd::mtl::max_threads_per_threadgroup(HOOMD_MTL_SOURCE, name, constants);
    unsigned int run_block_size = std::min(block_size, max_block_size);
    unsigned int num_blocks = (n + run_block_size - 1) / run_block_size;
    return hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                              name,
                              dim3(num_blocks),
                              dim3(run_block_size),
                              args,
                              0,
                              constants);
    }
    } // end namespace detail

//! Linear bounding volume hierarchy of points (Metal implementation of neighbor::LBVH)
class LBVH
    {
    public:
    //! Allocate memory for N primitives
    void setup(unsigned int N)
        {
        if (N == m_N)
            return;
        m_N = N;
        m_N_internal = (m_N > 0) ? m_N - 1 : 0;
        m_N_nodes = m_N + m_N_internal;
        m_parent.reserve(m_N_nodes);
        m_left.reserve(m_N_internal);
        m_right.reserve(m_N_internal);
        m_locks.reserve(m_N_internal);
        m_lo.reserve(m_N_nodes);
        m_hi.reserve(m_N_nodes);
        m_codes.reserve(m_N);
        m_indexes.reserve(m_N);
        }

    //! Build the hierarchy of the points points[map[i]], i = 0..N-1
    void build(const hoomd::Scalar4* points,
               const unsigned int* map,
               unsigned int N,
               const float3& lo,
               const float3& hi,
               unsigned int block_size)
        {
        setup(N);
        if (m_N == 0)
            return;

        hoomd::md::kernel_args::lbvh_bubble_aabbs_args bubble;
        bubble.tree = data();
        bubble.points = points;
        bubble.map = map;
        bubble.d_locks = m_locks.get();
        bubble.N = m_N;

        // single-particle LBVH has the root as its only leaf
        if (m_N == 1)
            {
            detail::launch_1d("hoomd::md::lbvh_one_primitive", 1, 1, bubble);
            return;
            }

        // calculate morton codes
        hoomd::md::kernel_args::lbvh_gen_codes_args codes;
        codes.d_codes = m_codes.get();
        codes.d_indexes = m_indexes.get();
        codes.points = points;
        codes.map = map;
        codes.lo = lo;
        codes.hi = hi;
        codes.N = m_N;
        detail::launch_1d("hoomd::md::lbvh_gen_codes", m_N, block_size, codes);

        // sort the codes (30 bits) with the primitive indexes
        hoomd::mtl::sort_pairs(m_codes.get(), m_indexes.get(), m_N, 0, 30);

        // process hierarchy and bubble aabbs
        hoomd::md::kernel_args::lbvh_gen_tree_args tree;
        tree.tree = data();
        tree.d_codes = m_codes.get();
        tree.N = m_N;
        detail::launch_1d("hoomd::md::lbvh_gen_tree", m_N - 1, block_size, tree);

        hipMemsetAsync(m_locks.get(), 0, (m_N - 1) * sizeof(unsigned int));
        detail::launch_1d("hoomd::md::lbvh_bubble_aabbs", m_N, block_size, bubble);
        }

    int getRoot() const
        {
        return 0;
        }
    unsigned int getN() const
        {
        return m_N;
        }
    unsigned int getNInternal() const
        {
        return m_N_internal;
        }
    unsigned int getNNodes() const
        {
        return m_N_nodes;
        }
    const unsigned int* getPrimitives() const
        {
        return m_indexes.get();
        }

    hoomd::md::kernel_args::lbvh_data data() const
        {
        hoomd::md::kernel_args::lbvh_data tree;
        tree.parent = m_parent.get();
        tree.left = m_left.get();
        tree.right = m_right.get();
        tree.primitive = m_indexes.get();
        tree.lo = m_lo.get();
        tree.hi = m_hi.get();
        return tree;
        }

    private:
    unsigned int m_N = 0;          //!< Number of primitives in the tree
    unsigned int m_N_internal = 0; //!< Number of internal nodes in tree
    unsigned int m_N_nodes = 0;    //!< Number of nodes in the tree
    detail::DeviceArray<int> m_parent;
    detail::DeviceArray<int> m_left;
    detail::DeviceArray<int> m_right;
    detail::DeviceArray<unsigned int> m_locks;
    detail::DeviceArray<float3> m_lo;
    detail::DeviceArray<float3> m_hi;
    detail::DeviceArray<unsigned int> m_codes;
    detail::DeviceArray<unsigned int> m_indexes;
    };

//! Rope traverser of an LBVH (Metal implementation of neighbor::LBVHTraverser)
class LBVHTraverser
    {
    public:
    LBVHTraverser()
        {
        m_lo.reserve(1);
        m_hi.reserve(1);
        m_bins.reserve(1);
        }

    //! Compress the LBVH for traversal, mapping its primitives through map
    void setup(const unsigned int* map, const LBVH& lbvh, unsigned int block_size)
        {
        m_replay = false;
        if (lbvh.getN() != 0)
            {
            compress(map, lbvh, block_size);
            m_replay = true;
            m_map = map;
            }
        }

    void traverse(hoomd::md::kernel_args::lbvh_traverse_ropes_args& args,
                  const LBVH& lbvh,
                  const unsigned int* map,
                  bool use_body,
                  unsigned int block_size)
        {
        if (lbvh.getN() == 0)
            return;
        if (args.N == 0 || args.n_images == 0)
            return;
        if (args.n_images > 32)
            {
            throw std::runtime_error(
                "A maximum of 32 image vectores are supported by LBVH traversers.");
            }

        // compress the tree if it was not set up with this map
        if (!m_replay || m_map != map)
            compress(map, lbvh, block_size);

        args.data = m_data.get();
        args.tree_lo = m_lo.get();
        args.tree_hi = m_hi.get();
        args.tree_bins = m_bins.get();
        args.root = lbvh.getRoot();
        detail::launch_1d("hoomd::md::lbvh_traverse_ropes",
                          args.N,
                          block_size,
                          args,
                          {hoomd::mtl::FunctionConstant::make_bool(0, use_body)});
        }

    private:
    void compress(const unsigned int* map, const LBVH& lbvh, unsigned int block_size)
        {
        m_data.reserve(lbvh.getNNodes());
        hoomd::md::kernel_args::lbvh_compress_ropes_args args;
        args.tree = lbvh.data();
        args.data = m_data.get();
        args.ctree_lo = m_lo.get();
        args.ctree_hi = m_hi.get();
        args.ctree_bins = m_bins.get();
        args.transform_map = map;
        args.root = lbvh.getRoot();
        args.N_internal = lbvh.getNInternal();
        args.N_nodes = lbvh.getNNodes();
        detail::launch_1d("hoomd::md::lbvh_compress_ropes", lbvh.getNNodes(), block_size, args);
        }

    detail::DeviceArray<int4> m_data;   //!< Compressed LBVH
    detail::DeviceArray<float3> m_lo;   //!< Lower bound of tree
    detail::DeviceArray<float3> m_hi;   //!< Upper bound of tree
    detail::DeviceArray<float3> m_bins; //!< Bin size for compression
    bool m_replay = false;              //!< True when the compressed tree is current
    const unsigned int* m_map = nullptr; //!< Map used to compress the tree
    };
    } // end namespace neighbor

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_nlist_mark_types(unsigned int* d_types,
                                unsigned int* d_indexes,
                                unsigned int* d_lbvh_errors,
                                Scalar4* d_last_pos,
                                const Scalar4* d_pos,
                                const unsigned int N,
                                const unsigned int nghosts,
                                const BoxDim& box,
                                const Scalar3 ghost_width,
                                const unsigned int block_size)
    {
    kernel_args::nlist_mark_types_args args;
    args.d_types = d_types;
    args.d_indexes = d_indexes;
    args.d_lbvh_errors = d_lbvh_errors;
    args.d_last_pos = d_last_pos;
    args.d_pos = d_pos;
    args.N = N;
    args.nghosts = nghosts;
    args.box = box;
    args.ghost_width = ghost_width;
    return neighbor::detail::launch_1d("hoomd::md::gpu_nlist_mark_types_kernel",
                                       N + nghosts,
                                       block_size,
                                       args);
    }

/*! The CUB two pass interface is kept: the first call (d_tmp == NULL) requests a token amount of
    temporary storage, and the second sorts d_types and d_indexes in place. The sorted data then
    lives in the input buffers, which the caller swaps in when swap is set.
*/
uchar2 gpu_nlist_sort_types(void* d_tmp,
                            size_t& tmp_bytes,
                            unsigned int* d_types,
                            unsigned int* d_sorted_types,
                            unsigned int* d_indexes,
                            unsigned int* d_sorted_indexes,
                            const unsigned int N,
                            const unsigned int num_bits)
    {
    if (d_tmp == NULL)
        {
        tmp_bytes = 4;
        return make_uchar2(0, 0);
        }
    mtl::sort_pairs(d_types, d_indexes, N, 0, num_bits);
    return make_uchar2(1, 1);
    }

hipError_t gpu_nlist_count_types(unsigned int* d_first,
                                 unsigned int* d_last,
                                 const unsigned int* d_types,
                                 const unsigned int ntypes,
                                 const unsigned int N,
                                 const unsigned int block_size)
    {
    // fill first and last with sentinels
    mtl::fill(d_first, NeighborListTypeSentinel, ntypes);
    mtl::fill(d_last, 0, ntypes);

    kernel_args::nlist_count_types_args args;
    args.d_first = d_first;
    args.d_last = d_last;
    args.d_types = d_types;
    args.ntypes = ntypes;
    args.N = N;
    return neighbor::detail::launch_1d("hoomd::md::gpu_nlist_count_types_kernel",
                                       N,
                                       block_size,
                                       args);
    }

hipError_t gpu_nlist_copy_primitives(unsigned int* d_traverse_order,
                                     const unsigned int* d_indexes,
                                     const unsigned int* d_primitives,
                                     const unsigned int N,
                                     const unsigned int block_size)
    {
    kernel_args::nlist_copy_primitives_args args;
    args.d_traverse_order = d_traverse_order;
    args.d_indexes = d_indexes;
    args.d_primitives = d_primitives;
    args.N = N;
    return neighbor::detail::launch_1d("hoomd::md::gpu_nlist_copy_primitives_kernel",
                                       N,
                                       block_size,
                                       args);
    }

LBVHWrapper::LBVHWrapper()
    {
    lbvh_ = new neighbor::LBVH();
    }

LBVHWrapper::~LBVHWrapper()
    {
    delete lbvh_;
    }

void LBVHWrapper::setup(const Scalar4* points,
                        const unsigned int* map,
                        unsigned int N,
                        hipStream_t stream)
    {
    lbvh_->setup(N);
    }

void LBVHWrapper::build(const Scalar4* points,
                        const unsigned int* map,
                        unsigned int N,
                        const Scalar3& lo,
                        const Scalar3& hi,
                        hipStream_t stream,
                        unsigned int block_size)
    {
    lbvh_->build(points, map, N, lo, hi, block_size);
    }

unsigned int LBVHWrapper::getN() const
    {
    return lbvh_->getN();
    }

const unsigned int* LBVHWrapper::getPrimitives() const
    {
    return lbvh_->getPrimitives();
    }

std::vector<unsigned int> LBVHWrapper::getTunableParameters() const
    {
    return neighbor::detail::tunable_block_sizes();
    }

LBVHTraverserWrapper::LBVHTraverserWrapper()
    {
    trav_ = new neighbor::LBVHTraverser();
    };

LBVHTraverserWrapper::~LBVHTraverserWrapper()
    {
    delete trav_;
    }

void LBVHTraverserWrapper::setup(const unsigned int* map, neighbor::LBVH& lbvh, hipStream_t stream)
    {
    trav_->setup(map, lbvh, 32);
    }

void LBVHTraverserWrapper::traverse(TraverserArgs& args,
                                    neighbor::LBVH& lbvh,
                                    const Scalar3* images,
                                    const unsigned int Nimages,
                                    hipStream_t stream,
                                    unsigned int block_size)
    {
    kernel_args::lbvh_traverse_ropes_args targs;
    targs.positions = args.positions;
    targs.bodies = args.bodies;
    targs.order = args.order;
    targs.N = args.N;
    targs.Nown = args.Nown;
    targs.rlist = args.rlist;
    targs.neigh_list = reinterpret_cast<uint4*>(args.neigh_list);
    targs.nneigh = args.nneigh;
    targs.new_max_neigh = args.new_max_neigh;
    targs.first_neigh = args.first_neigh;
    targs.max_neigh = args.max_neigh;
    targs.images = images;
    targs.n_images = Nimages;
    trav_->traverse(targs, lbvh, args.map, args.bodies != NULL, block_size);
    }

std::vector<unsigned int> LBVHTraverserWrapper::getTunableParameters() const
    {
    return neighbor::detail::tunable_block_sizes();
    }

    } // end namespace kernel

#endif

    } // end namespace md
    } // end namespace hoomd
