// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MetalKernel.h
    \brief Include first in every HOOMD-blue .metal kernel file.

    A .metal file holds the Metal port of a CUDA/HIP .cu file. It is compiled twice:

    - As Metal Shading Language (__METAL_VERSION__ is defined): the kernels.
    - As C++ for the host: the driver functions declared in the matching .cuh header. They pack
      the kernel arguments into a struct and call hoomd::mtl::launch.

    The kernel argument structs are defined once, outside of any `#ifdef __METAL_VERSION__`, so that
    the host and device layouts match. Declare pointer members with HOOMD_DEVPTR(T).

    The build defines HOOMD_MTL_SOURCE as the name of the embedded KernelSource for the file. Pass
    it to hoomd::mtl::launch.
*/

#pragma once

#ifdef __METAL_VERSION__

#include "hoomd/mtl/MetalDevice.h"

#else

#include "hoomd/mtl/MetalRuntime.h"

//! Device pointer member of a kernel argument struct
#define HOOMD_DEVPTR(...) ::hoomd::mtl::devptr<__VA_ARGS__>

#ifdef HOOMD_MTL_SOURCE
//! The preprocessed Metal source of this file, embedded at build time
extern const ::hoomd::mtl::KernelSource HOOMD_MTL_SOURCE;
#endif

// CUDA provides host overloads of min and max, including mixed signed and unsigned integers
// (which compare as unsigned). Driver code ported from .cu files uses them.
inline int min(int a, int b)
    {
    return a < b ? a : b;
    }
inline unsigned int min(unsigned int a, unsigned int b)
    {
    return a < b ? a : b;
    }
inline unsigned int min(int a, unsigned int b)
    {
    return min(static_cast<unsigned int>(a), b);
    }
inline unsigned int min(unsigned int a, int b)
    {
    return min(a, static_cast<unsigned int>(b));
    }
inline float min(float a, float b)
    {
    return a < b ? a : b;
    }
inline int max(int a, int b)
    {
    return a > b ? a : b;
    }
inline unsigned int max(unsigned int a, unsigned int b)
    {
    return a > b ? a : b;
    }
inline unsigned int max(int a, unsigned int b)
    {
    return max(static_cast<unsigned int>(a), b);
    }
inline unsigned int max(unsigned int a, int b)
    {
    return max(a, static_cast<unsigned int>(b));
    }
inline float max(float a, float b)
    {
    return a > b ? a : b;
    }
inline unsigned long min(unsigned long a, unsigned long b)
    {
    return a < b ? a : b;
    }
inline unsigned long max(unsigned long a, unsigned long b)
    {
    return a > b ? a : b;
    }

#endif
