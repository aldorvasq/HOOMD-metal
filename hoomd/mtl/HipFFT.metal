// Copyright (c) 2026 Aldo Vasquez-Briceno.
// Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

/*! \file HipFFT.metal
    \brief hipFFT complex-to-complex transforms on Apple Metal (see hipfft.h).

    A multidimensional transform is a sequence of batched 1D transforms, one per axis. Each
    threadgroup transforms one line in threadgroup memory with a mixed radix Stockham FFT
    (Govindaraju et al., SC 2008). A radix R step computes, for butterfly j with Ns the product of
    the previous radices and k = j mod Ns:

        dst[(j / Ns) Ns R + k + q Ns] = sum_r src[j + r N / R] exp(sign 2 pi i r (k + q Ns) / (Ns R))

    which combines the twiddle factors and the length R DFT. Lines are limited to 2048 elements
    by the threadgroup memory.
*/

#include "hoomd/mtl/MetalKernel.h"

#ifndef __METAL_VERSION__
#include <hipfft.h>

#include <map>
#include <mutex>
#include <vector>
#endif

#define HOOMD_MTL_FFT_MAX_FACTORS 32
#define HOOMD_MTL_FFT_MAX_LENGTH 2048

namespace hoomd
    {
namespace mtl
    {
struct fft_lines_args
    {
    HOOMD_DEVPTR(const float2) d_in;
    HOOMD_DEVPTR(float2) d_out;
    unsigned int length;      //!< Number of elements in a line
    unsigned int stride;      //!< Distance between consecutive elements of a line
    unsigned int n_lines;     //!< Number of lines
    int sign;                 //!< -1 forward, +1 backward
    unsigned int n_factors;   //!< Number of radices
    unsigned int factors[HOOMD_MTL_FFT_MAX_FACTORS];
    };
    } // end namespace mtl
    } // end namespace hoomd

#ifdef __METAL_VERSION__

namespace hoomd
    {
namespace mtl
    {
kernel void fft_lines(HOOMD_KERNEL_PARAMS(fft_lines_args))
    {
    const unsigned int N = args.length;
    const unsigned int S = args.stride;
    const unsigned int line = blockIdx.x;
    const unsigned int T = blockDim.x;
    const unsigned int tid = threadIdx.x;

    // the line's first element: lines interleave with stride S inside blocks of N * S elements
    const unsigned int base = (line / S) * (N * S) + (line % S);

    threadgroup float2* src = (threadgroup float2*)hoomd_dynamic_shared;
    threadgroup float2* dst = src + N;

    for (unsigned int n = tid; n < N; n += T)
        src[n] = args.d_in[base + n * S];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float two_pi = 6.28318530717958647692f * float(args.sign);
    unsigned int Ns = 1;
    for (unsigned int f = 0; f < args.n_factors; f++)
        {
        const unsigned int R = args.factors[f];
        const unsigned int NR = N / R;
        const unsigned int m = Ns * R;
        for (unsigned int j = tid; j < NR; j += T)
            {
            const unsigned int k = j % Ns;
            const unsigned int out = (j / Ns) * m + k;
            for (unsigned int q = 0; q < R; q++)
                {
                float2 acc = float2(0.0f, 0.0f);
                const unsigned int kq = k + q * Ns;
                for (unsigned int r = 0; r < R; r++)
                    {
                    float2 x = src[j + r * NR];
                    unsigned int e = (r * kq) % m;
                    float c;
                    float s = precise::sincos(two_pi * float(e) / float(m), c);
                    acc.x += x.x * c - x.y * s;
                    acc.y += x.x * s + x.y * c;
                    }
                dst[out + q * Ns] = acc;
                }
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2* tmp = src;
        src = dst;
        dst = tmp;
        Ns = m;
        }

    for (unsigned int n = tid; n < N; n += T)
        args.d_out[base + n * S] = src[n];
    }
    } // end namespace mtl
    } // end namespace hoomd

#else

namespace
    {
struct FFTPlan
    {
    std::vector<int> dims; //!< Dimensions, slowest varying first
    int batch;
    };

std::mutex g_fft_mutex;
std::map<hipfftHandle, FFTPlan> g_fft_plans;
hipfftHandle g_fft_next_handle = 1;

//! Factor n into radices 4, 2, 3, 5, 7, then remaining primes
bool factorize(unsigned int n, std::vector<unsigned int>& factors)
    {
    factors.clear();
    for (unsigned int r : {4u, 2u, 3u, 5u, 7u})
        {
        while (n % r == 0)
            {
            factors.push_back(r);
            n /= r;
            }
        }
    for (unsigned int p = 11; n > 1; p += 2)
        {
        while (n % p == 0)
            {
            factors.push_back(p);
            n /= p;
            }
        }
    return factors.size() <= HOOMD_MTL_FFT_MAX_FACTORS;
    }

hipfftResult make_plan(hipfftHandle* plan, std::vector<int> dims, int batch, hipfftType type)
    {
    if (!plan)
        return HIPFFT_INVALID_VALUE;
    if (type != HIPFFT_C2C)
        return HIPFFT_NOT_SUPPORTED;
    for (int d : dims)
        {
        if (d <= 0)
            return HIPFFT_INVALID_SIZE;
        if (d > HOOMD_MTL_FFT_MAX_LENGTH)
            return HIPFFT_NOT_SUPPORTED;
        }
    std::lock_guard<std::mutex> lock(g_fft_mutex);
    *plan = g_fft_next_handle++;
    g_fft_plans[*plan] = FFTPlan {dims, batch};
    return HIPFFT_SUCCESS;
    }
    } // end anonymous namespace

hipfftResult hipfftPlan1d(hipfftHandle* plan, int nx, hipfftType type, int batch)
    {
    return make_plan(plan, {nx}, batch, type);
    }

hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny, hipfftType type)
    {
    return make_plan(plan, {nx, ny}, 1, type);
    }

hipfftResult hipfftPlan3d(hipfftHandle* plan, int nx, int ny, int nz, hipfftType type)
    {
    return make_plan(plan, {nx, ny, nz}, 1, type);
    }

hipfftResult hipfftDestroy(hipfftHandle plan)
    {
    std::lock_guard<std::mutex> lock(g_fft_mutex);
    return g_fft_plans.erase(plan) ? HIPFFT_SUCCESS : HIPFFT_INVALID_PLAN;
    }

hipfftResult
hipfftExecC2C(hipfftHandle plan, hipfftComplex* idata, hipfftComplex* odata, int direction)
    {
    FFTPlan p;
        {
        std::lock_guard<std::mutex> lock(g_fft_mutex);
        auto it = g_fft_plans.find(plan);
        if (it == g_fft_plans.end())
            return HIPFFT_INVALID_PLAN;
        p = it->second;
        }
    if (direction != HIPFFT_FORWARD && direction != HIPFFT_BACKWARD)
        return HIPFFT_INVALID_VALUE;

    unsigned int total = static_cast<unsigned int>(p.batch);
    for (int d : p.dims)
        total *= static_cast<unsigned int>(d);

    // transform along each axis, fastest varying first
    unsigned int stride = 1;
    const hipfftComplex* in = idata;
    for (int axis = static_cast<int>(p.dims.size()) - 1; axis >= 0; axis--)
        {
        unsigned int length = static_cast<unsigned int>(p.dims[axis]);
        if (length > 1)
            {
            hoomd::mtl::fft_lines_args args;
            args.d_in = in;
            args.d_out = odata;
            args.length = length;
            args.stride = stride;
            args.n_lines = total / length;
            args.sign = direction == HIPFFT_FORWARD ? -1 : 1;
            std::vector<unsigned int> factors;
            if (!factorize(length, factors))
                return HIPFFT_INVALID_SIZE;
            args.n_factors = static_cast<unsigned int>(factors.size());
            for (size_t i = 0; i < factors.size(); i++)
                args.factors[i] = factors[i];

            unsigned int threads = std::min(256u, std::max(32u, ((length / 2 + 31) / 32) * 32));
            hipError_t error = hoomd::mtl::launch(HOOMD_MTL_SOURCE,
                                                  "hoomd::mtl::fft_lines",
                                                  dim3(args.n_lines),
                                                  dim3(threads),
                                                  args,
                                                  2 * length * sizeof(float2));
            if (error != hipSuccess)
                return HIPFFT_EXEC_FAILED;
            in = odata;
            }
        else if (in != odata)
            {
            if (hipMemcpyAsync(odata, in, total * sizeof(hipfftComplex), hipMemcpyDeviceToDevice)
                != hipSuccess)
                return HIPFFT_EXEC_FAILED;
            in = odata;
            }
        stride *= length;
        }
    return HIPFFT_SUCCESS;
    }

#endif
