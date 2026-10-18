// Copyright (c) 2026 Aldo Vasquez-Briceno.
// Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

/*! \file MetalDevice.h
    \brief CUDA/HIP device code compatibility layer for the Metal Shading Language.

    HOOMD-blue's device code (evaluators, box and vector math, kernels) is written for CUDA and
    HIP. This header provides the CUDA built-ins that code relies on so that the shared headers
    compile as Metal Shading Language with minimal changes:

    - Function qualifiers (__device__, __host__, __forceinline__, ...).
    - Address space qualifier macros for code shared between C++ and MSL: HOOMD_THREAD (thread
      local references and pointers), HOOMD_SHARED (CUDA shared memory), HOOMD_GLOBAL (device
      memory). They expand to nothing in C++.
    - 3-component vector types with CUDA's 12 byte layout. MSL's float3 is 16 bytes, so the CUDA
      names map to MSL's packed vectors.
    - make_* vector constructors, the CUDA math library and intrinsics, atomics, warp shuffles,
      votes, and barriers.

    Only single precision is available: Apple GPUs have no double precision arithmetic. Metal
    builds of HOOMD-blue use HOOMD_LONGREAL_SIZE=32 and HOOMD_SHORTREAL_SIZE=32.
*/

#pragma once

#ifndef __METAL_VERSION__
#error MetalDevice.h must only be included in Metal Shading Language code.
#endif

// At build time, HOOMD's MSL sources are preprocessed with the host compiler (which defines
// HOOMD_MSL_PREPROCESS) and metal_stdlib is included by the generated source.
#ifndef HOOMD_MSL_PREPROCESS
#include <metal_stdlib>
#endif

using namespace metal;

// ---------------------------------------------------------------------------------------------
// Qualifiers

#define __host__
#define __device__
#define __forceinline__ inline __attribute__((always_inline))
#define __noinline__ __attribute__((noinline))
#define __restrict__
#define __launch_bounds__(...)
#define __shared__ threadgroup
#define __constant__ constant

#define HOOMD_THREAD thread
#define HOOMD_SHARED threadgroup
#define HOOMD_GLOBAL device
#define HOOMD_CONSTANT constant
//! Read-only parameters are passed by value so that arguments may come from any address space
#define HOOMD_CREF
//! Reference to data in threadgroup (CUDA shared) memory
#define HOOMD_SHARED_REF threadgroup&

// ---------------------------------------------------------------------------------------------
// Vector types: CUDA's 3-component vectors are packed (12 bytes for float3)

#define float3 packed_float3
#define int3 packed_int3
#define uint3 packed_uint3
#define char3 packed_char3
#define uchar3 packed_uchar3
#define short3 packed_short3
#define ushort3 packed_ushort3

#define HOOMD_MSL_MAKE_VECTOR(T, NAME)                                                          \
    inline vec<T, 2> make_##NAME##2(T x, T y)                                                   \
        {                                                                                       \
        return vec<T, 2>(x, y);                                                                 \
        }                                                                                       \
    inline packed_##NAME##3 make_##NAME##3(T x, T y, T z)                                       \
        {                                                                                       \
        return packed_##NAME##3(x, y, z);                                                       \
        }                                                                                       \
    inline vec<T, 4> make_##NAME##4(T x, T y, T z, T w)                                         \
        {                                                                                       \
        return vec<T, 4>(x, y, z, w);                                                           \
        }

HOOMD_MSL_MAKE_VECTOR(float, float)
HOOMD_MSL_MAKE_VECTOR(int, int)
HOOMD_MSL_MAKE_VECTOR(uint, uint)
HOOMD_MSL_MAKE_VECTOR(char, char)
HOOMD_MSL_MAKE_VECTOR(uchar, uchar)
HOOMD_MSL_MAKE_VECTOR(short, short)
HOOMD_MSL_MAKE_VECTOR(ushort, ushort)

#undef HOOMD_MSL_MAKE_VECTOR

// ---------------------------------------------------------------------------------------------
// Constants

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880f
#endif
#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390f
#endif
#ifndef M_LN2
#define M_LN2 0.69314718055994530942f
#endif
#ifndef M_E
#define M_E 2.71828182845904523536f
#endif
#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif

// ---------------------------------------------------------------------------------------------
// Math library: the CUDA single precision functions (IEEE versions use metal::precise)

inline float sqrtf(float x)
    {
    return precise::sqrt(x);
    }
inline float rsqrtf(float x)
    {
    return precise::rsqrt(x);
    }
inline float expf(float x)
    {
    return precise::exp(x);
    }
inline float exp2f(float x)
    {
    return precise::exp2(x);
    }
inline float exp10f(float x)
    {
    return precise::exp10(x);
    }
inline float logf(float x)
    {
    return precise::log(x);
    }
inline float log2f(float x)
    {
    return precise::log2(x);
    }
inline float log10f(float x)
    {
    return precise::log10(x);
    }
inline float powf(float x, float y)
    {
    return precise::pow(x, y);
    }
inline float sinf(float x)
    {
    return precise::sin(x);
    }
inline float cosf(float x)
    {
    return precise::cos(x);
    }
inline float tanf(float x)
    {
    return precise::tan(x);
    }
inline float asinf(float x)
    {
    return precise::asin(x);
    }
inline float acosf(float x)
    {
    return precise::acos(x);
    }
inline float atanf(float x)
    {
    return precise::atan(x);
    }
inline float atan2f(float y, float x)
    {
    return precise::atan2(y, x);
    }
inline float sinhf(float x)
    {
    return precise::sinh(x);
    }
inline float coshf(float x)
    {
    return precise::cosh(x);
    }
inline float tanhf(float x)
    {
    return precise::tanh(x);
    }
inline float asinhf(float x)
    {
    return precise::asinh(x);
    }
inline float acoshf(float x)
    {
    return precise::acosh(x);
    }
inline float atanhf(float x)
    {
    return precise::atanh(x);
    }
inline float fabsf(float x)
    {
    return fabs(x);
    }
inline float floorf(float x)
    {
    return floor(x);
    }
inline float ceilf(float x)
    {
    return ceil(x);
    }
inline float truncf(float x)
    {
    return trunc(x);
    }
inline float roundf(float x)
    {
    return round(x);
    }
inline float rintf(float x)
    {
    return rint(x);
    }
inline float nearbyintf(float x)
    {
    return rint(x);
    }
inline float fminf(float x, float y)
    {
    return fmin(x, y);
    }
inline float fmaxf(float x, float y)
    {
    return fmax(x, y);
    }
inline float fmodf(float x, float y)
    {
    return precise::fmod(x, y);
    }
inline float copysignf(float x, float y)
    {
    return copysign(x, y);
    }
inline float fmaf(float x, float y, float z)
    {
    return fma(x, y, z);
    }
inline float ldexpf(float x, int e)
    {
    return ldexp(x, e);
    }
inline float frexpf(float x, thread int* e)
    {
    return frexp(x, *e);
    }
inline float modff(float x, thread float* i)
    {
    return modf(x, *i);
    }
inline float nextafterf(float x, float y)
    {
    return nextafter(x, y);
    }
inline long lrintf(float x)
    {
    return long(rint(x));
    }
inline long lroundf(float x)
    {
    return long(round(x));
    }
inline void sincosf(float x, thread float* s, thread float* c)
    {
    *s = precise::sincos(x, *c);
    }
inline void sincospif(float x, thread float* s, thread float* c)
    {
    *s = precise::sinpi(x);
    *c = precise::cospi(x);
    }

//! Cube root
inline float cbrtf(float x)
    {
    if (x == 0.0f || isinf(x) || isnan(x))
        return x;
    float a = fabs(x);
    float r = precise::pow(a, 1.0f / 3.0f);
    // one Newton step to correct the rounding of the exponent 1/3
    r = r - (r * r * r - a) / (3.0f * r * r);
    return copysign(r, x);
    }

//! sqrt(x*x + y*y) without undue overflow or underflow
inline float hypotf(float x, float y)
    {
    float a = fabs(x);
    float b = fabs(y);
    float m = fmax(a, b);
    if (m == 0.0f || isinf(m))
        return m;
    float n = fmin(a, b) / m;
    return m * precise::sqrt(1.0f + n * n);
    }

//! exp(x) - 1 without cancellation near 0
inline float expm1f(float x)
    {
    float u = precise::exp(x);
    if (u == 1.0f)
        return x;
    float um1 = u - 1.0f;
    if (um1 == -1.0f)
        return -1.0f;
    return um1 * x / precise::log(u);
    }

//! log(1 + x) without cancellation near 0
inline float log1pf(float x)
    {
    float u = 1.0f + x;
    if (u == 1.0f)
        return x;
    return precise::log(u) * x / (u - 1.0f);
    }

//! Error function and complementary error function
/*! For |x| < 0.5, erf is evaluated from its Taylor series. Otherwise,
    erfc(z) = t exp(-z^2 + Q(t)) with t = 1 / (1 + z/2) and Q a degree 11 polynomial fit to erfc
    for HOOMD-blue. The z^2 term is split to avoid the rounding error of z*z. The maximum error is
    about 6 ulp for z where erfc(z) is a normal float.
*/
inline float hoomd_msl_erf_small(float x)
    {
    float x2 = x * x;
    float p = -1.492565036e-05f;
    p = fma(p, x2, 1.205533298e-04f);
    p = fma(p, x2, -8.548327023e-04f);
    p = fma(p, x2, 5.223977625e-03f);
    p = fma(p, x2, -2.686617065e-02f);
    p = fma(p, x2, 1.128379167e-01f);
    p = fma(p, x2, -3.761263890e-01f);
    p = fma(p, x2, 1.128379167e+00f);
    return x * p;
    }

inline float hoomd_msl_erfc_large(float z)
    {
    float t = 1.0f / (1.0f + 0.5f * z);
    float q = -3.103494535e-01f;
    q = fma(q, t, 1.668039396e+00f);
    q = fma(q, t, -3.729758352e+00f);
    q = fma(q, t, 4.377066704e+00f);
    q = fma(q, t, -2.860443841e+00f);
    q = fma(q, t, 1.234977569e+00f);
    q = fma(q, t, -5.682909668e-01f);
    q = fma(q, t, 9.158570394e-03f);
    q = fma(q, t, 6.870629650e-02f);
    q = fma(q, t, 3.764705852e-01f);
    q = fma(q, t, 9.999131197e-01f);
    q = fma(q, t, -1.265509842e+00f);
    float zr = as_type<float>(as_type<uint>(z) & 0xfffff000u);
    return t * precise::exp(-zr * zr) * precise::exp((zr - z) * (zr + z) + q);
    }

inline float erfcf(float x)
    {
    if (isnan(x))
        return x;
    float z = fabs(x);
    float r;
    if (z < 0.5f)
        return 1.0f - hoomd_msl_erf_small(x);
    else if (z < 10.0f)
        r = hoomd_msl_erfc_large(z);
    else
        r = 0.0f;
    return x < 0.0f ? 2.0f - r : r;
    }

inline float erff(float x)
    {
    if (isnan(x))
        return x;
    float z = fabs(x);
    if (z < 0.5f)
        return hoomd_msl_erf_small(x);
    float r = z < 10.0f ? 1.0f - hoomd_msl_erfc_large(z) : 1.0f;
    return copysign(r, x);
    }

inline float erfc(float x)
    {
    return erfcf(x);
    }

inline float erf(float x)
    {
    return erff(x);
    }

//! Natural log of the gamma function for x >= 0.5 (Lanczos approximation, g = 7, n = 9)
inline float hoomd_msl_lgamma_lanczos(float x)
    {
    x -= 1.0f;
    float a = 0.99999999999980993f;
    const float p[8] = {676.5203681218851f,
                        -1259.1392167224028f,
                        771.32342877765313f,
                        -176.61502916214059f,
                        12.507343278686905f,
                        -0.13857109526572012f,
                        9.9843695780195716e-6f,
                        1.5056327351493116e-7f};
    float t = x + 7.5f;
    for (int i = 0; i < 8; i++)
        a += p[i] / (x + float(i + 1));
    return 0.91893853320467274f + (x + 0.5f) * precise::log(t) - t + precise::log(a);
    }

//! Natural log of the absolute value of the gamma function
inline float lgammaf(float x)
    {
    if (x < 0.5f)
        {
        // reflection formula
        return precise::log(M_PI / fabs(precise::sinpi(x))) - hoomd_msl_lgamma_lanczos(1.0f - x);
        }
    return hoomd_msl_lgamma_lanczos(x);
    }

inline float lgamma(float x)
    {
    return lgammaf(x);
    }

inline float tgammaf(float x)
    {
    float r = precise::exp(lgammaf(x));
    if (x < 0.0f && fmod(floor(x), 2.0f) != 0.0f)
        r = -r;
    return r;
    }

// ---------------------------------------------------------------------------------------------
// Double precision storage. Apple GPUs cannot compute in double precision, but kernels can read
// and write arrays of IEEE doubles that the host uses: hoomd_double stores the bits of a double and
// converts to and from float in software (round to nearest even).

//! Bits of the double precision value of f (exact)
inline ulong hoomd_float_to_double_bits(float f)
    {
    uint u = as_type<uint>(f);
    ulong sign = ulong(u >> 31) << 63;
    uint e = (u >> 23) & 0xffu;
    uint m = u & 0x7fffffu;
    if (e == 0xffu)
        return sign | (ulong(0x7ff) << 52) | (ulong(m) << 29); // inf and nan
    if (e == 0u)
        {
        if (m == 0u)
            return sign;
        // normalize the subnormal float
        uint shift = clz(m) - 8u;
        m = (m << shift) & 0x7fffffu;
        return sign | (ulong(1023 - 126 - int(shift)) << 52) | (ulong(m) << 29);
        }
    return sign | (ulong(int(e) - 127 + 1023) << 52) | (ulong(m) << 29);
    }

//! Single precision value of the double with the given bits (round to nearest even)
inline float hoomd_double_bits_to_float(ulong d)
    {
    uint sign = uint(d >> 63) << 31;
    int e = int((d >> 52) & 0x7ff);
    ulong m = d & 0xfffffffffffffull;
    if (e == 0x7ff)
        {
        uint mf = uint(m >> 29);
        if (m != 0 && mf == 0)
            mf = 1; // preserve nan
        return as_type<float>(sign | 0x7f800000u | mf);
        }
    if (e == 0)
        return as_type<float>(sign); // double subnormals underflow to zero
    int ef = e - 1023 + 127;
    if (ef >= 0xff)
        return as_type<float>(sign | 0x7f800000u);
    if (ef <= 0)
        {
        // float subnormal
        if (ef < -23)
            return as_type<float>(sign);
        m |= (1ull << 52);
        uint shift = uint(30 - ef);
        ulong mf = m >> shift;
        ulong rem = m & ((1ull << shift) - 1);
        ulong halfway = 1ull << (shift - 1);
        if (rem > halfway || (rem == halfway && (mf & 1)))
            mf++;
        return as_type<float>(sign | uint(mf));
        }
    uint bits = sign | (uint(ef) << 23) | uint(m >> 29);
    ulong rem = m & 0x1fffffffull;
    if (rem > 0x10000000ull || (rem == 0x10000000ull && (bits & 1u)))
        bits++; // carries into the exponent as needed
    return as_type<float>(bits);
    }

//! Storage for an IEEE double that converts to and from float
struct hoomd_double
    {
    ulong bits;

    hoomd_double() = default;
    hoomd_double(float f) : bits(hoomd_float_to_double_bits(f)) { }

    operator float() const thread
        {
        return hoomd_double_bits_to_float(bits);
        }
    operator float() const device
        {
        return hoomd_double_bits_to_float(bits);
        }
    operator float() const threadgroup
        {
        return hoomd_double_bits_to_float(bits);
        }
    operator float() const constant
        {
        return hoomd_double_bits_to_float(bits);
        }

#define HOOMD_DOUBLE_COMPOUND(SPACE, OP)                                                       \
    SPACE hoomd_double& operator OP##=(float v) SPACE                                          \
        {                                                                                      \
        bits = hoomd_float_to_double_bits(hoomd_double_bits_to_float(bits) OP v);              \
        return *this;                                                                          \
        }
    HOOMD_DOUBLE_COMPOUND(thread, +)
    HOOMD_DOUBLE_COMPOUND(thread, -)
    HOOMD_DOUBLE_COMPOUND(thread, *)
    HOOMD_DOUBLE_COMPOUND(thread, /)
    HOOMD_DOUBLE_COMPOUND(device, +)
    HOOMD_DOUBLE_COMPOUND(device, -)
    HOOMD_DOUBLE_COMPOUND(device, *)
    HOOMD_DOUBLE_COMPOUND(device, /)
#undef HOOMD_DOUBLE_COMPOUND
    };

//! Storage for an IEEE double3 (CUDA double3 layout: 3 doubles, 24 bytes)
struct hoomd_double3
    {
    hoomd_double x, y, z;
    };

//! Storage for an IEEE double4 (CUDA double4 layout: 4 doubles, 32 bytes)
struct hoomd_double4
    {
    hoomd_double x, y, z, w;
    };

// double3 and double4 are reserved names in MSL. Kernels ported from CUDA that read and write
// double3/double4 arrays use the storage types and compute in float.
#define double3 hoomd_double3
#define double4 hoomd_double4

inline hoomd_double3 make_double3(float x, float y, float z)
    {
    hoomd_double3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
    }

inline hoomd_double4 make_double4(float x, float y, float z, float w)
    {
    hoomd_double4 v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
    return v;
    }

//! Stuff an integer inside a double (low 32 bits, as __int_as_double in HOOMDMath.h)
inline hoomd_double __int_as_double(int a)
    {
    hoomd_double d;
    d.bits = ulong(uint(a));
    return d;
    }

//! Extract an integer from a double stuffed by __int_as_double()
inline int __double_as_int(hoomd_double d)
    {
    return int(uint(d.bits));
    }

// ---------------------------------------------------------------------------------------------
// CUDA intrinsics

inline float __sinf(float x)
    {
    return fast::sin(x);
    }
inline float __cosf(float x)
    {
    return fast::cos(x);
    }
inline float __tanf(float x)
    {
    return fast::tan(x);
    }
inline float __expf(float x)
    {
    return fast::exp(x);
    }
inline float __exp10f(float x)
    {
    return fast::exp10(x);
    }
inline float __logf(float x)
    {
    return fast::log(x);
    }
inline float __log2f(float x)
    {
    return fast::log2(x);
    }
inline float __log10f(float x)
    {
    return fast::log10(x);
    }
inline float __powf(float x, float y)
    {
    return fast::pow(x, y);
    }
inline float __fdividef(float x, float y)
    {
    return fast::divide(x, y);
    }
inline void __sincosf(float x, thread float* s, thread float* c)
    {
    *s = fast::sincos(x, *c);
    }
inline float __saturatef(float x)
    {
    return saturate(x);
    }
inline float __fsqrt_rn(float x)
    {
    return precise::sqrt(x);
    }
inline float __frsqrt_rn(float x)
    {
    return precise::rsqrt(x);
    }
inline float __frcp_rn(float x)
    {
    return 1.0f / x;
    }
inline float __fdiv_rn(float x, float y)
    {
    return x / y;
    }
inline float __fadd_rn(float x, float y)
    {
    return x + y;
    }
inline float __fsub_rn(float x, float y)
    {
    return x - y;
    }
inline float __fmul_rn(float x, float y)
    {
    return x * y;
    }
inline float __fmaf_rn(float x, float y, float z)
    {
    return fma(x, y, z);
    }

inline int __float2int_rn(float x)
    {
    return int(rint(x));
    }
inline int __float2int_rz(float x)
    {
    return int(x);
    }
inline int __float2int_rd(float x)
    {
    return int(floor(x));
    }
inline int __float2int_ru(float x)
    {
    return int(ceil(x));
    }
inline uint __float2uint_rn(float x)
    {
    return uint(rint(x));
    }
inline uint __float2uint_rz(float x)
    {
    return uint(x);
    }
inline uint __float2uint_rd(float x)
    {
    return uint(floor(x));
    }
inline uint __float2uint_ru(float x)
    {
    return uint(ceil(x));
    }
inline float __int2float_rn(int x)
    {
    return float(x);
    }
inline float __uint2float_rn(uint x)
    {
    return float(x);
    }

inline int __float_as_int(float x)
    {
    return as_type<int>(x);
    }
inline float __int_as_float(int x)
    {
    return as_type<float>(x);
    }
inline uint __float_as_uint(float x)
    {
    return as_type<uint>(x);
    }
inline float __uint_as_float(uint x)
    {
    return as_type<float>(x);
    }

inline int __popc(uint x)
    {
    return int(popcount(x));
    }
inline int __popcll(ulong x)
    {
    return int(popcount(x));
    }
inline int __clz(int x)
    {
    return int(clz(x));
    }
inline int __ffs(int x)
    {
    return x == 0 ? 0 : int(ctz(x)) + 1;
    }
inline uint __brev(uint x)
    {
    return reverse_bits(x);
    }
inline uint __umulhi(uint x, uint y)
    {
    return mulhi(x, y);
    }
inline int __mulhi(int x, int y)
    {
    return mulhi(x, y);
    }
inline int __mul24(int x, int y)
    {
    return x * y;
    }
inline uint __umul24(uint x, uint y)
    {
    return x * y;
    }

//! Read-only data cache load (Metal caches device const loads)
template<class T> inline T __ldg(device const T* ptr)
    {
    return *ptr;
    }
template<class T> inline T __ldg(constant T* ptr)
    {
    return *ptr;
    }

// ---------------------------------------------------------------------------------------------
// Atomics. Device memory supports integer and float atomics. Threadgroup memory supports integer
// atomics, so threadgroup float atomics use compare and swap.

#define HOOMD_MSL_ATOMIC_INT(SPACE, T, AT)                                                     \
    inline T atomicAdd(SPACE T* p, T v)                                                         \
        {                                                                                       \
        return atomic_fetch_add_explicit((SPACE AT*)p, v, memory_order_relaxed);                \
        }                                                                                       \
    inline T atomicSub(SPACE T* p, T v)                                                         \
        {                                                                                       \
        return atomic_fetch_sub_explicit((SPACE AT*)p, v, memory_order_relaxed);                \
        }                                                                                       \
    inline T atomicExch(SPACE T* p, T v)                                                        \
        {                                                                                       \
        return atomic_exchange_explicit((SPACE AT*)p, v, memory_order_relaxed);                 \
        }                                                                                       \
    inline T atomicMin(SPACE T* p, T v)                                                         \
        {                                                                                       \
        return atomic_fetch_min_explicit((SPACE AT*)p, v, memory_order_relaxed);                \
        }                                                                                       \
    inline T atomicMax(SPACE T* p, T v)                                                         \
        {                                                                                       \
        return atomic_fetch_max_explicit((SPACE AT*)p, v, memory_order_relaxed);                \
        }                                                                                       \
    inline T atomicAnd(SPACE T* p, T v)                                                         \
        {                                                                                       \
        return atomic_fetch_and_explicit((SPACE AT*)p, v, memory_order_relaxed);                \
        }                                                                                       \
    inline T atomicOr(SPACE T* p, T v)                                                          \
        {                                                                                       \
        return atomic_fetch_or_explicit((SPACE AT*)p, v, memory_order_relaxed);                 \
        }                                                                                       \
    inline T atomicXor(SPACE T* p, T v)                                                         \
        {                                                                                       \
        return atomic_fetch_xor_explicit((SPACE AT*)p, v, memory_order_relaxed);                \
        }                                                                                       \
    /* CUDA's atomicCAS is a strong compare and swap */                                        \
    inline T atomicCAS(SPACE T* p, T compare, T v)                                              \
        {                                                                                       \
        T expected = compare;                                                                   \
        while (!atomic_compare_exchange_weak_explicit((SPACE AT*)p,                             \
                                                      &expected,                                \
                                                      v,                                        \
                                                      memory_order_relaxed,                     \
                                                      memory_order_relaxed))                    \
            {                                                                                   \
            if (expected != compare)                                                            \
                break;                                                                          \
            }                                                                                   \
        return expected;                                                                        \
        }

HOOMD_MSL_ATOMIC_INT(device, int, atomic_int)
HOOMD_MSL_ATOMIC_INT(device, uint, atomic_uint)
HOOMD_MSL_ATOMIC_INT(threadgroup, int, atomic_int)
HOOMD_MSL_ATOMIC_INT(threadgroup, uint, atomic_uint)

#undef HOOMD_MSL_ATOMIC_INT

//! CUDA atomicInc: ((old >= limit) ? 0 : (old + 1))
inline uint atomicInc(device uint* p, uint limit)
    {
    uint old = atomic_load_explicit((device atomic_uint*)p, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit((device atomic_uint*)p,
                                                  &old,
                                                  old >= limit ? 0u : old + 1u,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed))
        {
        }
    return old;
    }

//! CUDA atomicDec: (((old == 0) || (old > limit)) ? limit : (old - 1))
inline uint atomicDec(device uint* p, uint limit)
    {
    uint old = atomic_load_explicit((device atomic_uint*)p, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit((device atomic_uint*)p,
                                                  &old,
                                                  (old == 0u || old > limit) ? limit : old - 1u,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed))
        {
        }
    return old;
    }

inline float atomicAdd(device float* p, float v)
    {
    return atomic_fetch_add_explicit((device atomic_float*)p, v, memory_order_relaxed);
    }

inline float atomicSub(device float* p, float v)
    {
    return atomic_fetch_sub_explicit((device atomic_float*)p, v, memory_order_relaxed);
    }

inline float atomicExch(device float* p, float v)
    {
    return atomic_exchange_explicit((device atomic_float*)p, v, memory_order_relaxed);
    }

inline float atomicAdd(threadgroup float* p, float v)
    {
    threadgroup atomic_uint* a = (threadgroup atomic_uint*)p;
    uint old = atomic_load_explicit(a, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(a,
                                                  &old,
                                                  as_type<uint>(as_type<float>(old) + v),
                                                  memory_order_relaxed,
                                                  memory_order_relaxed))
        {
        }
    return as_type<float>(old);
    }

inline float atomicSub(threadgroup float* p, float v)
    {
    return atomicAdd(p, -v);
    }

//! Float max/min through integer compare and swap (device and threadgroup memory)
#define HOOMD_MSL_ATOMIC_FLOAT_MINMAX(SPACE, NAME, OP)                                         \
    inline float NAME(SPACE float* p, float v)                                                  \
        {                                                                                       \
        SPACE atomic_uint* a = (SPACE atomic_uint*)p;                                           \
        uint old = atomic_load_explicit(a, memory_order_relaxed);                               \
        while (OP(as_type<float>(old), v) != as_type<float>(old)                                \
               && !atomic_compare_exchange_weak_explicit(a,                                     \
                                                         &old,                                  \
                                                         as_type<uint>(v),                      \
                                                         memory_order_relaxed,                  \
                                                         memory_order_relaxed))                 \
            {                                                                                   \
            }                                                                                   \
        return as_type<float>(old);                                                             \
        }

HOOMD_MSL_ATOMIC_FLOAT_MINMAX(device, atomicMax, fmax)
HOOMD_MSL_ATOMIC_FLOAT_MINMAX(device, atomicMin, fmin)
HOOMD_MSL_ATOMIC_FLOAT_MINMAX(threadgroup, atomicMax, fmax)
HOOMD_MSL_ATOMIC_FLOAT_MINMAX(threadgroup, atomicMin, fmin)

#undef HOOMD_MSL_ATOMIC_FLOAT_MINMAX

// ---------------------------------------------------------------------------------------------
// Warp (SIMD-group) operations. Apple GPUs execute 32-wide SIMD-groups. Threads of a 1D
// threadgroup map to SIMD-groups in order, so lane = threadIdx.x % 32 as on NVIDIA GPUs.
// Only full-width shuffles are provided: sub-warp shuffles need the caller's lane index.

#define warpSize 32

template<class T> inline T __shfl_sync(uint mask, T v, int src_lane)
    {
    return simd_shuffle(v, ushort(src_lane & 31));
    }

template<class T> inline T __shfl_down_sync(uint mask, T v, uint delta)
    {
    return simd_shuffle_down(v, ushort(delta));
    }

template<class T> inline T __shfl_up_sync(uint mask, T v, uint delta)
    {
    return simd_shuffle_up(v, ushort(delta));
    }

template<class T> inline T __shfl_xor_sync(uint mask, T v, int lane_mask)
    {
    return simd_shuffle_xor(v, ushort(lane_mask));
    }

inline uint __ballot_sync(uint mask, bool predicate)
    {
    return uint(static_cast<simd_vote::vote_t>(simd_ballot(predicate)));
    }

inline bool __any_sync(uint mask, bool predicate)
    {
    return simd_any(predicate);
    }

inline bool __all_sync(uint mask, bool predicate)
    {
    return simd_all(predicate);
    }

inline uint __activemask()
    {
    return uint(static_cast<simd_vote::vote_t>(simd_active_threads_mask()));
    }

// ---------------------------------------------------------------------------------------------
// Barriers and fences

//! __syncthreads also orders global memory accesses within the block
#define __syncthreads() threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#define __syncwarp(...) simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#define __threadfence()                                                                        \
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)
#define __threadfence_block()                                                                  \
    atomic_thread_fence(mem_flags::mem_threadgroup | mem_flags::mem_device,                    \
                        memory_order_seq_cst,                                                  \
                        thread_scope_threadgroup)

// ---------------------------------------------------------------------------------------------
// Kernel declarations

/*! Parameters of a HOOMD-blue Metal kernel.

    The macro argument is the argument struct shared with the host driver.

    Declares the CUDA built-ins threadIdx, blockIdx, blockDim and gridDim, the argument struct
    `args`, and `hoomd_dynamic_shared`, the dynamic threadgroup memory (CUDA's extern __shared__).
    Helper functions that need thread indices take them as arguments.
*/
#define HOOMD_KERNEL_PARAMS(...)                                                               \
    constant __VA_ARGS__& args [[buffer(0)]],                                                  \
        metal::vec<uint, 3> threadIdx [[thread_position_in_threadgroup]],                      \
        metal::vec<uint, 3> blockIdx [[threadgroup_position_in_grid]],                         \
        metal::vec<uint, 3> blockDim [[threads_per_threadgroup]],                              \
        metal::vec<uint, 3> gridDim [[threadgroups_per_grid]],                                 \
        ushort hoomd_lane [[thread_index_in_simdgroup]],                                       \
        threadgroup char* hoomd_dynamic_shared [[threadgroup(0)]]

//! HIP's dynamic shared memory declaration: the kernel's dynamic threadgroup memory
#define HIP_DYNAMIC_SHARED(type, var) threadgroup type* var = (threadgroup type*)hoomd_dynamic_shared;

//! Device pointer member of a kernel argument struct (devptr<T> on the host)
#define HOOMD_DEVPTR(...) device __VA_ARGS__*
