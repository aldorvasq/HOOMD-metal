// Copyright (c) 2026 Aldo Vasquez-Briceno.
// Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

/*! \file hip/hip_runtime.h
    \brief The subset of the HIP runtime API that HOOMD-blue uses, implemented on Apple Metal.

    HOOMD-blue's GPU code paths are written against the HIP runtime API. When HOOMD_GPU_PLATFORM is
    Metal, this header takes the place of the HIP headers:

    - Host code sees HIP types and functions. They are implemented in hoomd/mtl/HipMetalRuntime.mm.
    - Metal Shading Language code sees the device prelude in hoomd/mtl/MetalDevice.h.

    Apple GPUs share memory with the CPU. Every allocation made through this API is a Metal buffer
    in shared storage and the pointers handed to HOOMD are valid on both the host and the device.
    Device pointers are translated to GPU virtual addresses when kernels are launched (see
    hoomd/mtl/MetalRuntime.h).
*/

#pragma once

#ifndef __HIP_PLATFORM_METAL__
#define __HIP_PLATFORM_METAL__ 1
#endif

#ifdef __METAL_VERSION__

#include "hoomd/mtl/MetalDevice.h"

#else

#include <cstddef>
#include <cstdint>

#include "hoomd/extern/cudacpu_vector_functions.h"
#include "hoomd/extern/cudacpu_vector_types.h"

#define HIPMETAL_API __attribute__((visibility("default")))

#ifndef HIP_VERSION_MAJOR
#define HIP_VERSION_MAJOR 6
#endif
#ifndef HIP_VERSION_MINOR
#define HIP_VERSION_MINOR 0
#endif
#ifndef HIP_VERSION_PATCH
#define HIP_VERSION_PATCH 0
#endif

typedef enum hipError_t
    {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorMemoryAllocation = 2,
    hipErrorNotInitialized = 3,
    hipErrorInitializationError = 3,
    hipErrorDeinitialized = 4,
    hipErrorInvalidConfiguration = 9,
    hipErrorInvalidSymbol = 13,
    hipErrorInvalidDevicePointer = 17,
    hipErrorInvalidMemcpyDirection = 21,
    hipErrorInvalidDeviceFunction = 98,
    hipErrorNoDevice = 100,
    hipErrorInvalidDevice = 101,
    hipErrorInvalidImage = 200,
    hipErrorInvalidKernelFile = 218,
    hipErrorInvalidHandle = 400,
    hipErrorInvalidResourceHandle = 400,
    hipErrorNotReady = 600,
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701,
    hipErrorLaunchTimeOut = 702,
    hipErrorHostMemoryAlreadyRegistered = 712,
    hipErrorHostMemoryNotRegistered = 713,
    hipErrorLaunchFailure = 719,
    hipErrorNotSupported = 801,
    hipErrorUnknown = 999
    } hipError_t;

typedef enum hipMemcpyKind
    {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
    } hipMemcpyKind;

typedef enum hipComputeMode
    {
    hipComputeModeDefault = 0,
    hipComputeModeExclusive = 1,
    hipComputeModeProhibited = 2,
    hipComputeModeExclusiveProcess = 3
    } hipComputeMode;

typedef enum hipMemoryAdvise
    {
    hipMemAdviseSetReadMostly = 1,
    hipMemAdviseUnsetReadMostly = 2,
    hipMemAdviseSetPreferredLocation = 3,
    hipMemAdviseUnsetPreferredLocation = 4,
    hipMemAdviseSetAccessedBy = 5,
    hipMemAdviseUnsetAccessedBy = 6
    } hipMemoryAdvise;

#define hipHostRegisterDefault 0x0
#define hipHostRegisterPortable 0x1
#define hipHostRegisterMapped 0x2
#define hipHostMallocDefault 0x0
#define hipHostMallocPortable 0x1
#define hipHostMallocMapped 0x2
#define hipDeviceScheduleAuto 0x0
#define hipDeviceMapHost 0x8
#define hipEventDefault 0x0
#define hipEventBlockingSync 0x1
#define hipEventDisableTiming 0x2
#define hipMemAttachGlobal 0x1
#define hipMemAttachHost 0x2
#define hipStreamDefault 0x0
#define hipStreamNonBlocking 0x1
#define hipCpuDeviceId (-1)

//! Kernel attributes (see hoomd::mtl::get_function_attributes)
typedef struct hipFuncAttributes
    {
    int maxThreadsPerBlock;
    size_t sharedSizeBytes;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int numRegs;
    int maxDynamicSharedSizeBytes;
    } hipFuncAttributes;

typedef struct ihipStream_t* hipStream_t;
typedef struct ihipEvent_t* hipEvent_t;

//! Device properties, filled from the Metal device
typedef struct hipDeviceProp_t
    {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    int memoryClockRate;
    int memoryBusWidth;
    size_t totalConstMem;
    int major;
    int minor;
    int multiProcessorCount;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int computeMode;
    int canMapHostMemory;
    int integrated;
    int concurrentManagedAccess;
    int managedMemory;
    int unifiedAddressing;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int ECCEnabled;
    size_t sharedMemPerMultiprocessor;
    size_t maxSharedMemoryPerMultiProcessor;
    } hipDeviceProp_t;

// Device management
HIPMETAL_API hipError_t hipInit(unsigned int flags);
HIPMETAL_API hipError_t hipGetDeviceCount(int* count);
HIPMETAL_API hipError_t hipGetDevice(int* device);
HIPMETAL_API hipError_t hipSetDevice(int device);
HIPMETAL_API hipError_t hipSetDeviceFlags(unsigned int flags);
HIPMETAL_API hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device);
HIPMETAL_API hipError_t hipDeviceSynchronize();
HIPMETAL_API hipError_t hipDeviceReset();
HIPMETAL_API hipError_t hipMemGetInfo(size_t* free, size_t* total);

// Error handling
HIPMETAL_API hipError_t hipGetLastError();
HIPMETAL_API hipError_t hipPeekAtLastError();
HIPMETAL_API const char* hipGetErrorString(hipError_t error);
HIPMETAL_API const char* hipGetErrorName(hipError_t error);

// Memory management
HIPMETAL_API hipError_t hipMalloc(void** ptr, size_t size);
HIPMETAL_API hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags = hipMemAttachGlobal);
HIPMETAL_API hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags = hipHostMallocDefault);
HIPMETAL_API hipError_t hipFree(void* ptr);
HIPMETAL_API hipError_t hipHostFree(void* ptr);
HIPMETAL_API hipError_t hipHostRegister(void* ptr, size_t size, unsigned int flags);
HIPMETAL_API hipError_t hipHostUnregister(void* ptr);
HIPMETAL_API hipError_t hipHostGetDevicePointer(void** device_ptr, void* host_ptr, unsigned int flags);
HIPMETAL_API hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind);
HIPMETAL_API hipError_t hipMemcpyAsync(void* dst,
                                      const void* src,
                                      size_t size,
                                      hipMemcpyKind kind,
                                      hipStream_t stream = 0);
HIPMETAL_API hipError_t hipMemset(void* ptr, int value, size_t size);
HIPMETAL_API hipError_t hipMemsetAsync(void* ptr, int value, size_t size, hipStream_t stream = 0);
HIPMETAL_API hipError_t hipMemAdvise(const void* ptr, size_t size, int advice, int device);
HIPMETAL_API hipError_t
hipMemPrefetchAsync(const void* ptr, size_t size, int device, hipStream_t stream = 0);

template<class T> inline hipError_t hipMalloc(T** ptr, size_t size)
    {
    return hipMalloc(reinterpret_cast<void**>(ptr), size);
    }

template<class T>
inline hipError_t hipMallocManaged(T** ptr, size_t size, unsigned int flags = hipMemAttachGlobal)
    {
    return hipMallocManaged(reinterpret_cast<void**>(ptr), size, flags);
    }

template<class T>
inline hipError_t hipHostMalloc(T** ptr, size_t size, unsigned int flags = hipHostMallocDefault)
    {
    return hipHostMalloc(reinterpret_cast<void**>(ptr), size, flags);
    }

template<class T>
inline hipError_t hipHostGetDevicePointer(T** device_ptr, void* host_ptr, unsigned int flags)
    {
    return hipHostGetDevicePointer(reinterpret_cast<void**>(device_ptr), host_ptr, flags);
    }

// Streams. All streams map to the single in-order Metal command queue.
HIPMETAL_API hipError_t hipStreamCreate(hipStream_t* stream);
HIPMETAL_API hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
HIPMETAL_API hipError_t hipStreamDestroy(hipStream_t stream);
HIPMETAL_API hipError_t hipStreamSynchronize(hipStream_t stream);
HIPMETAL_API hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);

// Events. Elapsed time is the GPU execution time of the work submitted between two events.
HIPMETAL_API hipError_t hipEventCreate(hipEvent_t* event);
HIPMETAL_API hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags);
HIPMETAL_API hipError_t hipEventDestroy(hipEvent_t event);
HIPMETAL_API hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = 0);
HIPMETAL_API hipError_t hipEventSynchronize(hipEvent_t event);
HIPMETAL_API hipError_t hipEventQuery(hipEvent_t event);
HIPMETAL_API hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);

// Profiling: start and stop a Metal GPU trace capture when one is configured.
HIPMETAL_API hipError_t hipProfilerStart();
HIPMETAL_API hipError_t hipProfilerStop();
#define hipProfileStart hipProfilerStart
#define hipProfileStop hipProfilerStop

#endif // __METAL_VERSION__
