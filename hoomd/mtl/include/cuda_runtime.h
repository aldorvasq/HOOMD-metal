// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file cuda_runtime.h
    \brief The subset of the CUDA runtime API that HOOMD-blue uses directly (in mpcd), mapped to
    the HIP API of the Metal runtime
*/

#pragma once

#include <hip/hip_runtime.h>

typedef hipError_t cudaError_t;
typedef hipFuncAttributes cudaFuncAttributes;
typedef hipDeviceProp_t cudaDeviceProp;
typedef hipStream_t cudaStream_t;
typedef hipMemcpyKind cudaMemcpyKind;

#define cudaSuccess hipSuccess
#define cudaErrorUnknown hipErrorUnknown
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault

#define cudaFuncGetAttributes hipFuncGetAttributes
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetErrorString hipGetErrorString
