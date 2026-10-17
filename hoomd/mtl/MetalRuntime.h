// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MetalRuntime.h
    \brief Host-side API to launch Metal compute kernels from HOOMD-blue's GPU driver functions.

    Kernel sources are Metal Shading Language translation units (the .metal files next to the
    classes they serve). At build time each one is preprocessed into a single string and embedded
    in the library as a KernelSource. The runtime compiles a source the first time one of its
    kernels launches and caches the resulting pipelines.

    Kernel arguments are passed as a single struct by value, the same struct definition compiled
    into the host driver and the Metal kernel. Pointer members use devptr<T> on the host, which
    stores the GPU virtual address of the HIP allocation the pointer belongs to, and `device T*` in
    Metal Shading Language.
*/

#pragma once

#ifdef __METAL_VERSION__
#error MetalRuntime.h is a host-only header.
#endif

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace hoomd
    {
namespace mtl
    {
//! A Metal Shading Language translation unit embedded in the library
struct KernelSource
    {
    const char* name;   //!< Name of the translation unit (for diagnostics)
    const char* source; //!< Preprocessed source code
    };

//! Value of a Metal function constant used to specialize a kernel
struct FunctionConstant
    {
    enum Type
        {
        Bool,
        Int,
        UInt,
        Float
        };

    unsigned int index;
    Type type;
    union
        {
        bool b;
        int i;
        unsigned int u;
        float f;
        };

    static FunctionConstant make_bool(unsigned int index, bool value)
        {
        FunctionConstant c;
        c.index = index;
        c.type = Bool;
        c.u = 0;
        c.b = value;
        return c;
        }

    static FunctionConstant make_int(unsigned int index, int value)
        {
        FunctionConstant c;
        c.index = index;
        c.type = Int;
        c.i = value;
        return c;
        }

    static FunctionConstant make_uint(unsigned int index, unsigned int value)
        {
        FunctionConstant c;
        c.index = index;
        c.type = UInt;
        c.u = value;
        return c;
        }

    static FunctionConstant make_float(unsigned int index, float value)
        {
        FunctionConstant c;
        c.index = index;
        c.type = Float;
        c.f = value;
        return c;
        }
    };

//! Translate a pointer from a HIP allocation into a GPU virtual address
/*! Returns 0 for nullptr. Throws std::runtime_error when the pointer is not in any allocation made
    through the HIP API.
*/
HIPMETAL_API uint64_t device_address(const void* ptr);

//! Launch a kernel
/*! \param source Translation unit that defines the kernel
    \param kernel Name of the kernel function
    \param grid Number of threadgroups (CUDA blocks) in each dimension
    \param block Number of threads per threadgroup in each dimension
    \param args Pointer to the argument struct
    \param args_bytes Size of the argument struct
    \param shared_bytes Dynamic threadgroup memory (CUDA's dynamic shared memory) in bytes
    \param constants Function constant values that specialize the kernel
    \param n_constants Number of function constants

    Launches are asynchronous. Errors are reported through the return value and
    hipGetLastError().
*/
HIPMETAL_API hipError_t launch(const KernelSource& source,
                               const char* kernel,
                               dim3 grid,
                               dim3 block,
                               const void* args,
                               size_t args_bytes,
                               size_t shared_bytes = 0,
                               const FunctionConstant* constants = nullptr,
                               unsigned int n_constants = 0);

//! Launch a kernel with an argument struct
template<class Args>
inline hipError_t launch(const KernelSource& source,
                         const char* kernel,
                         dim3 grid,
                         dim3 block,
                         const Args& args,
                         size_t shared_bytes = 0,
                         std::initializer_list<FunctionConstant> constants = {})
    {
    return launch(source,
                  kernel,
                  grid,
                  block,
                  &args,
                  sizeof(Args),
                  shared_bytes,
                  constants.begin(),
                  static_cast<unsigned int>(constants.size()));
    }

//! Maximum number of threads per threadgroup the kernel's pipeline supports
/*! This plays the role of hipFuncAttributes::maxThreadsPerBlock.
 */
HIPMETAL_API unsigned int max_threads_per_threadgroup(const KernelSource& source,
                                                      const char* kernel,
                                                      const FunctionConstant* constants = nullptr,
                                                      unsigned int n_constants = 0);

inline unsigned int max_threads_per_threadgroup(const KernelSource& source,
                                                const char* kernel,
                                                std::initializer_list<FunctionConstant> constants)
    {
    return max_threads_per_threadgroup(source,
                                       kernel,
                                       constants.begin(),
                                       static_cast<unsigned int>(constants.size()));
    }

//! Static threadgroup memory used by the kernel's pipeline in bytes
HIPMETAL_API size_t static_threadgroup_memory(const KernelSource& source,
                                              const char* kernel,
                                              const FunctionConstant* constants = nullptr,
                                              unsigned int n_constants = 0);

//! Fill hipFuncAttributes for a kernel (replaces hipFuncGetAttributes)
inline hipError_t
get_function_attributes(hipFuncAttributes* attr, const KernelSource& source, const char* kernel)
    {
    attr->maxThreadsPerBlock = static_cast<int>(max_threads_per_threadgroup(source, kernel));
    attr->sharedSizeBytes = static_threadgroup_memory(source, kernel);
    attr->constSizeBytes = 0;
    attr->localSizeBytes = 0;
    attr->numRegs = 0;
    attr->maxDynamicSharedSizeBytes = 0;
    return hipSuccess;
    }

//! Submit all encoded work to the GPU without waiting for it to complete
HIPMETAL_API void flush();

//! Device pointer member of a kernel argument struct
/*! The Metal side of the struct declares the member as `device T*`. Both are 8 bytes.
 */
template<class T> struct devptr
    {
    uint64_t address = 0;

    devptr() = default;
    devptr(T* ptr) : address(device_address(ptr)) { }
    devptr(std::nullptr_t) : address(0) { }
    };

static_assert(sizeof(devptr<float>) == 8, "devptr must match the size of a Metal device pointer");

    } // end namespace mtl
    } // end namespace hoomd
