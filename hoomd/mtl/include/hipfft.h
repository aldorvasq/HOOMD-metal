// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file hipfft.h
    \brief The subset of the hipFFT API that HOOMD-blue uses, implemented on Apple Metal.

    Single precision complex-to-complex transforms in 1, 2, and 3 dimensions. Like cuFFT and
    hipFFT, the transforms are unnormalized and the data are in row major order (the last
    dimension is contiguous).
*/

#pragma once

#include <hip/hip_runtime.h>

typedef float2 hipfftComplex;
typedef int hipfftHandle;

typedef enum hipfftResult_t
    {
    HIPFFT_SUCCESS = 0,
    HIPFFT_INVALID_PLAN = 1,
    HIPFFT_ALLOC_FAILED = 2,
    HIPFFT_INVALID_TYPE = 3,
    HIPFFT_INVALID_VALUE = 4,
    HIPFFT_INTERNAL_ERROR = 5,
    HIPFFT_EXEC_FAILED = 6,
    HIPFFT_SETUP_FAILED = 7,
    HIPFFT_INVALID_SIZE = 8,
    HIPFFT_UNALIGNED_DATA = 9,
    HIPFFT_INCOMPLETE_PARAMETER_LIST = 10,
    HIPFFT_INVALID_DEVICE = 11,
    HIPFFT_PARSE_ERROR = 12,
    HIPFFT_NO_WORKSPACE = 13,
    HIPFFT_NOT_IMPLEMENTED = 14,
    HIPFFT_NOT_SUPPORTED = 16
    } hipfftResult;

typedef enum hipfftType_t
    {
    HIPFFT_R2C = 0x2a,
    HIPFFT_C2R = 0x2c,
    HIPFFT_C2C = 0x29
    } hipfftType;

#define HIPFFT_FORWARD -1
#define HIPFFT_BACKWARD 1

HIPMETAL_API hipfftResult hipfftPlan1d(hipfftHandle* plan, int nx, hipfftType type, int batch);
HIPMETAL_API hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny, hipfftType type);
HIPMETAL_API hipfftResult hipfftPlan3d(hipfftHandle* plan, int nx, int ny, int nz, hipfftType type);
HIPMETAL_API hipfftResult hipfftExecC2C(hipfftHandle plan,
                                        hipfftComplex* idata,
                                        hipfftComplex* odata,
                                        int direction);
HIPMETAL_API hipfftResult hipfftDestroy(hipfftHandle plan);
