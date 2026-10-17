// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file HipMetalRuntime.mm
    \brief Implements the HIP runtime API subset and the kernel launch API on Apple Metal.

    Design:

    - Every allocation is an MTLBuffer in shared storage. The pointer handed to HOOMD is the
      buffer's CPU address, so "device" pointers are also valid host pointers. All allocations are
      members of one residency set attached to the command queue.
    - Kernel arguments are passed by value with setBytes. Pointers inside the argument struct are
      GPU virtual addresses obtained from device_address().
    - All HIP streams map to one in-order MTLCommandQueue. Launches are encoded into a serial
      compute encoder. A fence orders each encoder after the previous one because the buffers are
      not hazard tracked. A command buffer is committed whenever fewer than
      kCommandBuffersInFlight are queued or executing (so the GPU starts work while the host
      encodes more), after kDispatchesPerCommandBuffer launches, and when the host synchronizes.
    - Each command buffer signals a shared event with its sequence number when it completes, and
      a completion handler records the same. The host waits by spinning on these before blocking
      in waitUntilCompleted, which has a wake-up latency comparable to a short simulation step.
    - Synchronous memory operations (hipMemcpy, hipMemset) wait for the GPU and then operate on
      the shared memory with the CPU. hipMemcpyAsync between allocations and hipMemsetAsync are
      encoded as GPU kernels.
    - Event elapsed time is the GPU execution time of the command buffers submitted between the
      two events. Recording an event commits the pending command buffer.

    Environment variables:

    - HOOMD_METAL_SYNC=1: wait for each kernel to complete and check for errors (debugging).
    - HOOMD_METAL_DEBUG=1: report Metal errors with more detail.
    - HOOMD_METAL_PROFILE=1: run each kernel in its own command buffer and print the GPU time of
      each kernel at exit.
    - HOOMD_METAL_STATS=1: print command buffer, GPU utilization, and synchronization statistics
      (with the code that caused each synchronization) at exit.
    - HOOMD_METAL_BUFFERS_IN_FLIGHT=n: command buffers to keep queued before batching launches.
    - HOOMD_METAL_SPIN_US=t: microseconds to spin while waiting before blocking (0 to block).
    - HOOMD_METAL_CAPTURE=path: record a GPU trace for Xcode.
*/

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>

#include "hoomd/mtl/MetalRuntime.h"

#include <algorithm>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

struct ihipStream_t
    {
    unsigned int flags;
    };

struct ihipEvent_t
    {
    uint64_t sequence = 0;          //!< Number of command buffers committed before the event
    id<MTLCommandBuffer> last = nil; //!< Last command buffer committed before the event
    bool recorded = false;
    };

namespace
    {
//! Number of kernel launches encoded before a command buffer is committed
constexpr unsigned int kDispatchesPerCommandBuffer = 64;

//! Command buffers kept queued or executing on the GPU before kernels accumulate on the host
constexpr uint64_t kCommandBuffersInFlight = 2;

//! Number of committed command buffers remembered for event timing
constexpr size_t kTimedCommandBuffers = 4096;

//! Built-in kernels used to implement asynchronous memset and memcpy
const char* const kBuiltinSource = R"(
#include <metal_stdlib>
using namespace metal;

struct hipmetal_fill_args
    {
    device uchar* ptr;
    ulong n;
    uint value;
    uint aligned;
    };

kernel void hipmetal_fill(constant hipmetal_fill_args& args [[buffer(0)]],
                          uint i [[thread_position_in_grid]])
    {
    ulong start = ulong(i) * 16;
    if (start >= args.n)
        return;
    uchar v = uchar(args.value);
    if (args.aligned && start + 16 <= args.n)
        {
        uint w = uint(v) * 0x01010101u;
        *((device uint4*)(args.ptr + start)) = uint4(w);
        }
    else
        {
        ulong end = min(start + 16, args.n);
        for (ulong k = start; k < end; k++)
            args.ptr[k] = v;
        }
    }

struct hipmetal_copy_args
    {
    device uchar* dst;
    device const uchar* src;
    ulong n;
    uint aligned;
    };

kernel void hipmetal_copy(constant hipmetal_copy_args& args [[buffer(0)]],
                          uint i [[thread_position_in_grid]])
    {
    ulong start = ulong(i) * 16;
    if (start >= args.n)
        return;
    if (args.aligned && start + 16 <= args.n)
        {
        *((device uint4*)(args.dst + start)) = *((device const uint4*)(args.src + start));
        }
    else
        {
        ulong end = min(start + 16, args.n);
        for (ulong k = start; k < end; k++)
            args.dst[k] = args.src[k];
        }
    }
)";

const hoomd::mtl::KernelSource kBuiltinKernels = {"hipmetal_builtin", kBuiltinSource};

struct FillArgs
    {
    uint64_t ptr;
    uint64_t n;
    uint32_t value;
    uint32_t aligned;
    };

struct CopyArgs
    {
    uint64_t dst;
    uint64_t src;
    uint64_t n;
    uint32_t aligned;
    };

//! A region of memory that kernels may access
struct Allocation
    {
    size_t size;          //!< Size of the region in bytes
    id<MTLBuffer> buffer; //!< Buffer that contains the region
    uint64_t offset;      //!< Offset of the region's first byte in the buffer
    bool registered;      //!< True when the memory belongs to the caller (hipHostRegister)
    };

//! Read the GPU core count from the IO registry
int query_gpu_core_count()
    {
    int cores = 0;
    io_iterator_t iterator;
    if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                     IOServiceMatching("AGXAccelerator"),
                                     &iterator)
        == KERN_SUCCESS)
        {
        io_object_t service;
        while ((service = IOIteratorNext(iterator)) != 0)
            {
            CFTypeRef value = IORegistryEntrySearchCFProperty(service,
                                                              kIOServicePlane,
                                                              CFSTR("gpu-core-count"),
                                                              kCFAllocatorDefault,
                                                              kIORegistryIterateRecursively);
            if (value)
                {
                if (CFGetTypeID(value) == CFNumberGetTypeID())
                    CFNumberGetValue((CFNumberRef)value, kCFNumberIntType, &cores);
                CFRelease(value);
                }
            IOObjectRelease(service);
            if (cores > 0)
                break;
            }
        IOObjectRelease(iterator);
        }
    return cores;
    }

//! Report the Apple GPU family as the "compute capability" major version
int query_gpu_family(id<MTLDevice> device)
    {
    const MTLGPUFamily families[] = {MTLGPUFamilyApple10,
                                     MTLGPUFamilyApple9,
                                     MTLGPUFamilyApple8,
                                     MTLGPUFamilyApple7};
    const int numbers[] = {10, 9, 8, 7};
    for (size_t i = 0; i < sizeof(numbers) / sizeof(int); i++)
        {
        if ([device supportsFamily:families[i]])
            return numbers[i];
        }
    return 0;
    }

class Runtime
    {
    public:
    //! The process-wide runtime. Intentionally never destroyed so that static destructors that
    //! free HIP memory at exit remain valid.
    static Runtime& instance()
        {
        static Runtime* runtime = new Runtime();
        return *runtime;
        }

    bool available() const
        {
        return m_device != nil;
        }

    id<MTLDevice> device() const
        {
        return m_device;
        }

    //! Record an error to report through hipGetLastError
    hipError_t set_error(hipError_t error, const std::string& message)
        {
        std::lock_guard<std::mutex> lock(m_error_mutex);
        m_last_error = error;
        m_message_error = error;
        m_last_error_message = message;
        if (m_debug)
            {
            fprintf(stderr, "*** HIP-on-Metal error: %s\n", message.c_str());
            }
        return error;
        }

    hipError_t peek_error()
        {
        std::lock_guard<std::mutex> lock(m_error_mutex);
        return m_last_error;
        }

    hipError_t get_error()
        {
        std::lock_guard<std::mutex> lock(m_error_mutex);
        hipError_t error = m_last_error;
        m_last_error = hipSuccess;
        return error;
        }

    //! Detailed message of the most recent error with the given code (empty if none)
    std::string error_message(hipError_t error)
        {
        std::lock_guard<std::mutex> lock(m_error_mutex);
        return error == m_message_error ? m_last_error_message : std::string();
        }

    //! Register a region of memory
    void add_allocation(void* ptr, const Allocation& allocation)
        {
        std::lock_guard<std::mutex> lock(m_allocation_mutex);
        m_allocations[reinterpret_cast<uintptr_t>(ptr)] = allocation;
        [m_residency addAllocation:allocation.buffer];
        m_residency_dirty = true;
        }

    //! Find the allocation that starts at ptr
    bool find_exact(const void* ptr, Allocation& allocation)
        {
        std::lock_guard<std::mutex> lock(m_allocation_mutex);
        auto it = m_allocations.find(reinterpret_cast<uintptr_t>(ptr));
        if (it == m_allocations.end())
            return false;
        allocation = it->second;
        return true;
        }

    //! Find the allocation that contains ptr (or ends at ptr)
    bool find_containing(const void* ptr, Allocation& allocation, uintptr_t& start)
        {
        std::lock_guard<std::mutex> lock(m_allocation_mutex);
        uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        auto it = m_allocations.upper_bound(p);
        if (it == m_allocations.begin())
            return false;
        --it;
        if (p - it->first > it->second.size)
            return false;
        allocation = it->second;
        start = it->first;
        return true;
        }

    //! Remove the allocation that starts at ptr. Waits for the GPU first.
    void remove_allocation(const void* ptr)
        {
        synchronize();
        std::lock_guard<std::mutex> lock(m_allocation_mutex);
        auto it = m_allocations.find(reinterpret_cast<uintptr_t>(ptr));
        if (it == m_allocations.end())
            return;
        // the buffer may back other registrations (hipHostRegister on a shared page)
        id<MTLBuffer> buffer = it->second.buffer;
        m_allocations.erase(it);
        bool in_use = false;
        for (const auto& entry : m_allocations)
            {
            if (entry.second.buffer == buffer)
                {
                in_use = true;
                break;
                }
            }
        if (!in_use)
            {
            [m_residency removeAllocation:buffer];
            m_residency_dirty = true;
            }
        }

    uint64_t device_address(const void* ptr)
        {
        if (!ptr)
            return 0;
        Allocation allocation;
        uintptr_t start;
        if (!find_containing(ptr, allocation, start))
            {
            std::ostringstream s;
            s << "Pointer " << ptr
              << " passed to a Metal kernel is not part of any allocation made through the HIP "
                 "API.";
            throw std::runtime_error(s.str());
            }
        return allocation.buffer.gpuAddress + allocation.offset
               + (reinterpret_cast<uintptr_t>(ptr) - start);
        }

    //! Get a compute encoder for the current command buffer
    id<MTLComputeCommandEncoder> encoder()
        {
        if (!m_encoder)
            {
            if (!m_command_buffer)
                {
                // The runtime keeps every buffer and pipeline alive until the GPU is idle, so
                // command buffers do not need to retain them.
                m_command_buffer = [m_queue commandBufferWithUnretainedReferences];
                m_dispatches = 0;
                }
            m_encoder = [m_command_buffer
                computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            [m_encoder waitForFence:m_fence];
            }
        return m_encoder;
        }

    //! Called after each dispatch
    void dispatched()
        {
        m_dispatches++;
        if (m_profile)
            {
            commit();
            id<MTLCommandBuffer> command_buffer = m_last_command_buffer;
            [command_buffer waitUntilCompleted];
            check_status(command_buffer);
            m_last_command_buffer = nil;
            auto& entry = m_profile_data[m_current_kernel ? m_current_kernel : "(unknown)"];
            entry.first++;
            entry.second += command_buffer.GPUEndTime - command_buffer.GPUStartTime;
            }
        else if (m_sync_launches)
            {
            synchronize();
            }
        else if (m_dispatches >= kDispatchesPerCommandBuffer
                 || in_flight() < m_buffers_in_flight)
            {
            // Keep the GPU fed: submit work whenever fewer than kCommandBuffersInFlight command
            // buffers are queued or executing, so that the GPU starts the next buffer as soon as
            // it finishes the current one while the host encodes the following kernels. Once
            // enough work is queued, kernels accumulate in the open command buffer.
            commit();
            }
        }

    //! Number of committed command buffers that have not completed
    uint64_t in_flight() const
        {
        return m_submitted - m_completed.load(std::memory_order_acquire);
        }

    //! Wait for the command buffer with the given sequence number to complete
    /*! Spin briefly before blocking: waking from waitUntilCompleted adds latency that is
        significant compared to the time a typical simulation step takes on the GPU.
    */
    void wait_for(uint64_t sequence, id<MTLCommandBuffer> command_buffer)
        {
        auto start = std::chrono::steady_clock::now();
        unsigned int spins = 0;
        while (m_completed.load(std::memory_order_acquire) < sequence
               && m_completion_event.signaledValue < sequence)
            {
            if (++spins % 256 == 0 && std::chrono::steady_clock::now() - start > m_spin_limit)
                {
                [command_buffer waitUntilCompleted];
                break;
                }
            }
        }

    //! Print command buffer statistics (HOOMD_METAL_STATS)
    void print_stats()
        {
        std::lock_guard<std::mutex> lock(m_stats_mutex);
        double span = m_stats_last - m_stats_first;
        fprintf(stderr,
                "\nHOOMD Metal: %lu command buffers, %lu dispatches (%.1f per buffer), GPU busy "
                "%.3f s of %.3f s (%.1f%%)\n",
                m_stats_buffers,
                m_stats_dispatches,
                m_stats_buffers ? double(m_stats_dispatches) / m_stats_buffers : 0.0,
                m_stats_busy,
                span,
                span > 0 ? 100 * m_stats_busy / span : 0.0);
        fprintf(stderr,
                "HOOMD Metal: %lu synchronizations, %lu residency set commits\n",
                m_stats_syncs,
                m_stats_residency_commits);
        std::vector<std::pair<unsigned long, std::string>> sites;
        for (const auto& entry : m_stats_sync_sites)
            sites.emplace_back(entry.second, entry.first);
        std::sort(sites.rbegin(), sites.rend());
        for (size_t i = 0; i < sites.size() && i < 15; ++i)
            fprintf(stderr, "  %8lu  %s\n", sites[i].first, sites[i].second.c_str());
        }

    //! Print the kernel profile (HOOMD_METAL_PROFILE) sorted by total GPU time
    void print_profile()
        {
        std::vector<std::pair<std::string, std::pair<unsigned long, double>>> rows(
            m_profile_data.begin(),
            m_profile_data.end());
        std::sort(rows.begin(),
                  rows.end(),
                  [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
        double total = 0;
        for (const auto& row : rows)
            total += row.second.second;
        fprintf(stderr, "\nHOOMD Metal kernel profile (GPU time, %.3f s total)\n", total);
        fprintf(stderr, "%8s %7s %10s %10s  %s\n", "time(s)", "%", "launches", "us/launch", "kernel");
        for (const auto& row : rows)
            {
            fprintf(stderr,
                    "%8.3f %6.1f%% %10lu %10.2f  %s\n",
                    row.second.second,
                    total > 0 ? 100.0 * row.second.second / total : 0.0,
                    row.second.first,
                    1e6 * row.second.second / row.second.first,
                    row.first.c_str());
            }
        }

    //! Commit the current command buffer
    void commit()
        {
        if (!m_command_buffer)
            return;

        if (m_encoder)
            {
            [m_encoder updateFence:m_fence];
            [m_encoder endEncoding];
            m_encoder = nil;
            }

        if (m_residency_dirty)
            {
            m_stats_residency_commits++;
            [m_residency commit];
            m_residency_dirty = false;
            }

        id<MTLCommandBuffer> command_buffer = m_command_buffer;
        m_command_buffer = nil;

        if (!m_transient.empty())
            {
            // keep oversized argument buffers alive until the GPU is done with them
            __block std::vector<id<MTLBuffer>> transient = std::move(m_transient);
            m_transient.clear();
            [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
              transient.clear();
            }];
            }

        // Track completion with a handler: the handler runs promptly when the GPU finishes, while
        // the command buffer status property can report completion late.
            {
            const uint64_t sequence = m_submitted + 1;
            [command_buffer encodeSignalEvent:m_completion_event value:sequence];
            [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
              std::atomic<uint64_t>& completed = Runtime::instance().m_completed;
              uint64_t current = completed.load(std::memory_order_relaxed);
              while (current < sequence
                     && !completed.compare_exchange_weak(current,
                                                         sequence,
                                                         std::memory_order_release,
                                                         std::memory_order_relaxed))
                  {
                  }
            }];
            }
        if (m_stats)
            {
            unsigned long dispatches = m_dispatches;
            [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
              Runtime& rt = Runtime::instance();
              std::lock_guard<std::mutex> lock(rt.m_stats_mutex);
              rt.m_stats_buffers++;
              rt.m_stats_dispatches += dispatches;
              rt.m_stats_busy += cb.GPUEndTime - cb.GPUStartTime;
              if (rt.m_stats_first < 0)
                  rt.m_stats_first = cb.GPUStartTime;
              rt.m_stats_last = std::max(rt.m_stats_last, cb.GPUEndTime);
            }];
            }
        [command_buffer commit];
        m_submitted++;
        m_last_command_buffer = command_buffer;
        m_recent.emplace_back(m_submitted, command_buffer);
        while (m_recent.size() > kTimedCommandBuffers)
            m_recent.pop_front();
        }

    //! Commit pending work and wait for the GPU to complete it
    hipError_t synchronize()
        {
        m_stats_syncs++;
        if (m_stats)
            {
            // attribute the synchronization to the first caller outside of the HIP layer
            void* frames[12];
            int n = backtrace(frames, 12);
            std::string site, via;
            for (int i = 1; i < n; ++i)
                {
                Dl_info info;
                if (!dladdr(frames[i], &info) || !info.dli_sname)
                    continue;
                int status = 0;
                char* name = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
                std::string symbol = (status == 0 && name) ? name : info.dli_sname;
                free(name);
                if (via.empty()
                    && (symbol.find("::acquire") != std::string::npos
                        || symbol.find("hipMemcpy") != std::string::npos
                        || symbol.find("hipMemset") != std::string::npos
                        || symbol.find("GPUFlags") != std::string::npos))
                    via = symbol.substr(0, 60);
                if (symbol.find("Runtime") != std::string::npos
                    || symbol.find("hipDeviceSynchronize") != std::string::npos
                    || symbol.find("hipMemcpy") != std::string::npos
                    || symbol.find("hipMemset") != std::string::npos
                    || symbol.find("guarded") != std::string::npos
                    || symbol.find("::acquire") != std::string::npos
                    || symbol.find("ArrayHandle") != std::string::npos
                    || symbol.find("GPUFlags") != std::string::npos)
                    continue;
                site = symbol.substr(0, 100);
                break;
                }
            m_stats_sync_sites[site + "  [" + via + "]"]++;
            }
        commit();
        if (m_last_command_buffer)
            {
            id<MTLCommandBuffer> command_buffer = m_last_command_buffer;
            wait_for(m_submitted, command_buffer);
            check_status(command_buffer);
            m_last_command_buffer = nil;
            }
        return peek_error();
        }

    //! Report errors from a completed command buffer
    void check_status(id<MTLCommandBuffer> command_buffer)
        {
        if (command_buffer.status == MTLCommandBufferStatusError)
            {
            NSError* error = command_buffer.error;
            std::string message = "Metal command buffer failed";
            if (error)
                {
                message += ": ";
                message += error.localizedDescription.UTF8String;
                }
            hipError_t code = hipErrorLaunchFailure;
            if (error && error.code == MTLCommandBufferErrorPageFault)
                code = hipErrorIllegalAddress;
            else if (error && error.code == MTLCommandBufferErrorTimeout)
                code = hipErrorLaunchTimeOut;
            else if (error && error.code == MTLCommandBufferErrorOutOfMemory)
                code = hipErrorOutOfMemory;
            set_error(code, message);
            if (m_sync_launches)
                throw std::runtime_error(message);
            }
        }

    //! Record an event: commit the pending work and remember where it ends
    void record(ihipEvent_t* event)
        {
        commit();
        event->sequence = m_submitted;
        event->last = m_last_command_buffer;
        event->recorded = true;
        }

    //! GPU execution time of the command buffers submitted between two events
    float elapsed_ms(const ihipEvent_t* start, const ihipEvent_t* stop)
        {
        if (stop->last)
            [stop->last waitUntilCompleted];

        double seconds = 0;
        for (const auto& entry : m_recent)
            {
            if (entry.first > start->sequence && entry.first <= stop->sequence)
                {
                id<MTLCommandBuffer> command_buffer = entry.second;
                [command_buffer waitUntilCompleted];
                if (command_buffer.GPUEndTime > command_buffer.GPUStartTime)
                    seconds += command_buffer.GPUEndTime - command_buffer.GPUStartTime;
                }
            }
        return static_cast<float>(seconds * 1000.0);
        }

    //! Compile (once) and return the library for a kernel source
    id<MTLLibrary> library(const hoomd::mtl::KernelSource& source)
        {
        auto it = m_libraries.find(&source);
        if (it != m_libraries.end())
            return it->second;

        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_2;
        // match the IEEE semantics of the CUDA and HIP builds: no fast math unless a kernel
        // explicitly calls a fast:: function
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;

        NSError* error = nil;
        NSString* text = [NSString stringWithUTF8String:source.source];
        id<MTLLibrary> library = [m_device newLibraryWithSource:text options:options error:&error];
        if (!library)
            {
            std::string message = std::string("Failed to compile Metal kernels in ") + source.name;
            if (error)
                message += std::string(":\n") + error.localizedDescription.UTF8String;
            throw std::runtime_error(message);
            }
        m_libraries[&source] = library;
        return library;
        }

    //! Get (and cache) a pipeline for a kernel
    id<MTLComputePipelineState> pipeline(const hoomd::mtl::KernelSource& source,
                                         const char* kernel,
                                         const hoomd::mtl::FunctionConstant* constants,
                                         unsigned int n_constants)
        {
        std::string key = std::string(source.name) + "/" + kernel;
        for (unsigned int i = 0; i < n_constants; i++)
            {
            key += "/" + std::to_string(constants[i].index) + ":"
                   + std::to_string(int(constants[i].type)) + ":"
                   + std::to_string(constants[i].u);
            }

        auto it = m_pipelines.find(key);
        if (it != m_pipelines.end())
            return it->second;

        id<MTLLibrary> lib = library(source);
        NSError* error = nil;
        id<MTLFunction> function = nil;
        NSString* name = [NSString stringWithUTF8String:kernel];
        if (n_constants == 0)
            {
            function = [lib newFunctionWithName:name];
            }
        else
            {
            MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
            for (unsigned int i = 0; i < n_constants; i++)
                {
                const hoomd::mtl::FunctionConstant& c = constants[i];
                switch (c.type)
                    {
                case hoomd::mtl::FunctionConstant::Bool:
                    [values setConstantValue:&c.b type:MTLDataTypeBool atIndex:c.index];
                    break;
                case hoomd::mtl::FunctionConstant::Int:
                    [values setConstantValue:&c.i type:MTLDataTypeInt atIndex:c.index];
                    break;
                case hoomd::mtl::FunctionConstant::UInt:
                    [values setConstantValue:&c.u type:MTLDataTypeUInt atIndex:c.index];
                    break;
                case hoomd::mtl::FunctionConstant::Float:
                    [values setConstantValue:&c.f type:MTLDataTypeFloat atIndex:c.index];
                    break;
                    }
                }
            function = [lib newFunctionWithName:name constantValues:values error:&error];
            }

        if (!function)
            {
            std::ostringstream s;
            s << "Metal kernel " << kernel << " not found in " << source.name;
            if (error)
                s << ": " << error.localizedDescription.UTF8String;
            s << ". Available kernels:";
            for (NSString* n in lib.functionNames)
                s << " " << n.UTF8String;
            throw std::runtime_error(s.str());
            }

        MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = function;
        descriptor.label = name;
        id<MTLComputePipelineState> state =
            [m_device newComputePipelineStateWithDescriptor:descriptor
                                                    options:MTLPipelineOptionNone
                                                 reflection:nil
                                                      error:&error];
        if (!state)
            {
            std::string message = std::string("Failed to create Metal pipeline for ") + kernel;
            if (error)
                message += std::string(": ") + error.localizedDescription.UTF8String;
            throw std::runtime_error(message);
            }
        m_pipelines[key] = state;
        return state;
        }

    hipError_t launch(const hoomd::mtl::KernelSource& source,
                      const char* kernel,
                      dim3 grid,
                      dim3 block,
                      const void* args,
                      size_t args_bytes,
                      size_t shared_bytes,
                      const hoomd::mtl::FunctionConstant* constants,
                      unsigned int n_constants)
        {
        if (grid.x == 0 || grid.y == 0 || grid.z == 0)
            return hipSuccess;

        id<MTLComputePipelineState> state = pipeline(source, kernel, constants, n_constants);

        size_t threads = size_t(block.x) * block.y * block.z;
        if (threads == 0 || threads > state.maxTotalThreadsPerThreadgroup)
            {
            std::ostringstream s;
            s << "Invalid threadgroup size " << block.x << "x" << block.y << "x" << block.z
              << " for Metal kernel " << kernel << " (maximum "
              << state.maxTotalThreadsPerThreadgroup << ")";
            return set_error(hipErrorInvalidConfiguration, s.str());
            }

        size_t threadgroup_bytes = std::max<size_t>(16, (shared_bytes + 15) & ~size_t(15));
        if (threadgroup_bytes + state.staticThreadgroupMemoryLength
            > m_device.maxThreadgroupMemoryLength)
            {
            std::ostringstream s;
            s << "Metal kernel " << kernel << " requests " << shared_bytes
              << " bytes of dynamic threadgroup memory, more than the device supports";
            return set_error(hipErrorLaunchOutOfResources, s.str());
            }

        id<MTLComputeCommandEncoder> enc = encoder();
        [enc setComputePipelineState:state];
        if (args_bytes <= 4096)
            {
            [enc setBytes:args length:args_bytes atIndex:0];
            }
        else
            {
            id<MTLBuffer> buffer = [m_device newBufferWithBytes:args
                                                          length:args_bytes
                                                         options:MTLResourceStorageModeShared];
            m_transient.push_back(buffer);
            [enc setBuffer:buffer offset:0 atIndex:0];
            }
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(grid.x, grid.y, grid.z)
            threadsPerThreadgroup:MTLSizeMake(block.x, block.y, block.z)];
        m_current_kernel = kernel;
        dispatched();
        return hipSuccess;
        }

    //! Fill memory with a byte value on the GPU
    hipError_t fill_async(void* ptr, int value, size_t size)
        {
        FillArgs args;
        args.ptr = device_address(ptr);
        args.n = size;
        args.value = static_cast<uint32_t>(value) & 0xff;
        args.aligned = (args.ptr % 16) == 0;
        size_t threads = (size + 15) / 16;
        unsigned int block = 256;
        return launch(kBuiltinKernels,
                      "hipmetal_fill",
                      dim3(static_cast<unsigned int>((threads + block - 1) / block)),
                      dim3(block),
                      &args,
                      sizeof(args),
                      0,
                      nullptr,
                      0);
        }

    //! Copy memory on the GPU
    hipError_t copy_async(void* dst, const void* src, size_t size)
        {
        CopyArgs args;
        args.dst = device_address(dst);
        args.src = device_address(src);
        args.n = size;
        args.aligned = (args.dst % 16) == 0 && (args.src % 16) == 0;
        size_t threads = (size + 15) / 16;
        unsigned int block = 256;
        return launch(kBuiltinKernels,
                      "hipmetal_copy",
                      dim3(static_cast<unsigned int>((threads + block - 1) / block)),
                      dim3(block),
                      &args,
                      sizeof(args),
                      0,
                      nullptr,
                      0);
        }

    void fill_properties(hipDeviceProp_t* prop)
        {
        memset(prop, 0, sizeof(hipDeviceProp_t));
        strncpy(prop->name, m_device.name.UTF8String, sizeof(prop->name) - 1);
        prop->totalGlobalMem = m_device.recommendedMaxWorkingSetSize;
        prop->sharedMemPerBlock = m_device.maxThreadgroupMemoryLength;
        prop->regsPerBlock = 65536;
        prop->warpSize = 32;
        prop->maxThreadsPerBlock = static_cast<int>(m_device.maxThreadsPerThreadgroup.width);
        prop->maxThreadsDim[0] = static_cast<int>(m_device.maxThreadsPerThreadgroup.width);
        prop->maxThreadsDim[1] = static_cast<int>(m_device.maxThreadsPerThreadgroup.height);
        prop->maxThreadsDim[2] = static_cast<int>(m_device.maxThreadsPerThreadgroup.depth);
        prop->maxGridSize[0] = 2147483647;
        prop->maxGridSize[1] = 65535;
        prop->maxGridSize[2] = 65535;
        prop->clockRate = 0;
        prop->totalConstMem = 65536;
        prop->major = m_gpu_family;
        prop->minor = 0;
        prop->multiProcessorCount = m_gpu_cores;
        prop->maxThreadsPerMultiProcessor = 2048;
        prop->computeMode = hipComputeModeDefault;
        prop->canMapHostMemory = 1;
        prop->integrated = 1;
        prop->concurrentManagedAccess = 1;
        prop->managedMemory = 1;
        prop->unifiedAddressing = 1;
        prop->sharedMemPerMultiprocessor = m_device.maxThreadgroupMemoryLength;
        prop->maxSharedMemoryPerMultiProcessor = m_device.maxThreadgroupMemoryLength;
        }

    bool debug() const
        {
        return m_debug;
        }

    private:
    Runtime()
        {
        @autoreleasepool
            {
            m_device = MTLCreateSystemDefaultDevice();
            if (!m_device)
                return;

            m_queue = [m_device newCommandQueue];
            m_fence = [m_device newFence];
            m_completion_event = [m_device newSharedEvent];

            MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
            descriptor.label = @"HOOMD-blue allocations";
            descriptor.initialCapacity = 256;
            NSError* error = nil;
            m_residency = [m_device newResidencySetWithDescriptor:descriptor error:&error];
            if (!m_residency)
                {
                fprintf(stderr,
                        "HIP-on-Metal: failed to create a residency set: %s\n",
                        error ? error.localizedDescription.UTF8String : "unknown error");
                m_device = nil;
                return;
                }
            [m_queue addResidencySet:m_residency];

            m_gpu_cores = query_gpu_core_count();
            if (m_gpu_cores <= 0)
                m_gpu_cores = 8;
            m_gpu_family = query_gpu_family(m_device);

            const char* sync = getenv("HOOMD_METAL_SYNC");
            m_sync_launches = sync && strcmp(sync, "0") != 0;
            const char* debug = getenv("HOOMD_METAL_DEBUG");
            m_debug = m_sync_launches || (debug && strcmp(debug, "0") != 0);

            // HOOMD_METAL_PROFILE=1 runs each kernel in its own command buffer and reports the GPU
            // time spent in each kernel when the process exits
            const char* spin = getenv("HOOMD_METAL_SPIN_US");
            if (spin)
                m_spin_limit = std::chrono::microseconds(strtoll(spin, nullptr, 10));

            const char* in_flight = getenv("HOOMD_METAL_BUFFERS_IN_FLIGHT");
            if (in_flight)
                m_buffers_in_flight = strtoull(in_flight, nullptr, 10);

            const char* stats = getenv("HOOMD_METAL_STATS");
            m_stats = stats && strcmp(stats, "0") != 0;
            if (m_stats)
                std::atexit([] { Runtime::instance().print_stats(); });

            const char* profile = getenv("HOOMD_METAL_PROFILE");
            m_profile = profile && strcmp(profile, "0") != 0;
            if (m_profile)
                std::atexit([] { Runtime::instance().print_profile(); });
            }
        }

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    id<MTLFence> m_fence = nil;
    id<MTLResidencySet> m_residency = nil;
    bool m_residency_dirty = false;

    id<MTLCommandBuffer> m_command_buffer = nil;
    id<MTLComputeCommandEncoder> m_encoder = nil;
    unsigned int m_dispatches = 0;
    std::vector<id<MTLBuffer>> m_transient;

    id<MTLCommandBuffer> m_last_command_buffer = nil;
    uint64_t m_submitted = 0;
    //! Sequence number of the last command buffer known to have completed
    std::atomic<uint64_t> m_completed{0};
    //! Signaled by the GPU with the sequence number at the end of each command buffer
    id<MTLSharedEvent> m_completion_event = nil;
    std::deque<std::pair<uint64_t, id<MTLCommandBuffer>>> m_recent;

    std::mutex m_allocation_mutex;
    std::map<uintptr_t, Allocation> m_allocations;

    std::unordered_map<const hoomd::mtl::KernelSource*, id<MTLLibrary>> m_libraries;
    std::unordered_map<std::string, id<MTLComputePipelineState>> m_pipelines;

    std::mutex m_error_mutex;
    hipError_t m_last_error = hipSuccess;
    hipError_t m_message_error = hipSuccess;
    std::string m_last_error_message;

    int m_gpu_cores = 0;
    int m_gpu_family = 0;
    bool m_sync_launches = false;
    bool m_debug = false;
    uint64_t m_buffers_in_flight = kCommandBuffersInFlight;
    //! Maximum time to spin before blocking in wait_for (HOOMD_METAL_SPIN_US)
    std::chrono::microseconds m_spin_limit{5000};

    //! Command buffer statistics (HOOMD_METAL_STATS)
    bool m_stats = false;
    std::mutex m_stats_mutex;
    unsigned long m_stats_buffers = 0;
    unsigned long m_stats_dispatches = 0;
    unsigned long m_stats_syncs = 0;
    std::map<std::string, unsigned long> m_stats_sync_sites;
    unsigned long m_stats_residency_commits = 0;
    double m_stats_busy = 0;
    double m_stats_first = -1;
    double m_stats_last = 0;

    //! Kernel profiling (HOOMD_METAL_PROFILE): GPU time and launch count per kernel
    bool m_profile = false;
    const char* m_current_kernel = nullptr;
    std::map<std::string, std::pair<unsigned long, double>> m_profile_data;
    };

//! Translate exceptions from the runtime into HIP error codes
template<class F> hipError_t guarded(hipError_t code, F&& f)
    {
    try
        {
        return f();
        }
    catch (const std::exception& e)
        {
        Runtime::instance().set_error(code, e.what());
        if (Runtime::instance().debug())
            throw;
        return code;
        }
    }

hipError_t require_device()
    {
    if (!Runtime::instance().available())
        return Runtime::instance().set_error(hipErrorNoDevice, "No Metal device is available");
    return hipSuccess;
    }

//! Allocate a shared-storage buffer and register it
hipError_t allocate(void** ptr, size_t size)
    {
    if (!ptr)
        return hipErrorInvalidValue;
    *ptr = nullptr;
    if (hipError_t error = require_device())
        return error;
    if (size == 0)
        return hipSuccess;

    Runtime& rt = Runtime::instance();
    @autoreleasepool
        {
        id<MTLBuffer> buffer = [rt.device() newBufferWithLength:size
                                                         options:MTLResourceStorageModeShared
                                                                 | MTLResourceHazardTrackingModeUntracked];
        if (!buffer)
            {
            std::ostringstream s;
            s << "Failed to allocate " << size << " bytes of Metal memory";
            return rt.set_error(hipErrorOutOfMemory, s.str());
            }
        *ptr = buffer.contents;
        rt.add_allocation(*ptr, Allocation{size, buffer, 0, false});
        }
    return hipSuccess;
    }

hipError_t release(void* ptr, bool registered)
    {
    if (!ptr)
        return hipSuccess;
    Runtime& rt = Runtime::instance();
    Allocation allocation;
    if (!rt.find_exact(ptr, allocation) || allocation.registered != registered)
        {
        return rt.set_error(registered ? hipErrorHostMemoryNotRegistered : hipErrorInvalidValue,
                            "Attempt to free memory that was not allocated through the HIP API");
        }
    rt.remove_allocation(ptr);
    return hipSuccess;
    }

bool is_allocated(const void* ptr)
    {
    Allocation allocation;
    uintptr_t start;
    return Runtime::instance().find_containing(ptr, allocation, start);
    }

    } // end anonymous namespace

// ---------------------------------------------------------------------------------------------
// Device management

hipError_t hipInit(unsigned int)
    {
    return require_device();
    }

hipError_t hipGetDeviceCount(int* count)
    {
    if (!count)
        return hipErrorInvalidValue;
    *count = Runtime::instance().available() ? 1 : 0;
    return hipSuccess;
    }

hipError_t hipGetDevice(int* device)
    {
    if (!device)
        return hipErrorInvalidValue;
    *device = 0;
    return require_device();
    }

hipError_t hipSetDevice(int device)
    {
    if (hipError_t error = require_device())
        return error;
    if (device != 0)
        return Runtime::instance().set_error(hipErrorInvalidDevice, "Invalid Metal device id");
    return hipSuccess;
    }

hipError_t hipSetDeviceFlags(unsigned int)
    {
    return require_device();
    }

hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device)
    {
    if (!prop)
        return hipErrorInvalidValue;
    if (hipError_t error = require_device())
        return error;
    if (device != 0)
        return Runtime::instance().set_error(hipErrorInvalidDevice, "Invalid Metal device id");
    Runtime::instance().fill_properties(prop);
    return hipSuccess;
    }

hipError_t hipDeviceSynchronize()
    {
    if (!Runtime::instance().available())
        return hipSuccess;
    return guarded(hipErrorLaunchFailure, [] { return Runtime::instance().synchronize(); });
    }

hipError_t hipDeviceReset()
    {
    return hipDeviceSynchronize();
    }

hipError_t hipMemGetInfo(size_t* free, size_t* total)
    {
    if (hipError_t error = require_device())
        return error;
    id<MTLDevice> device = Runtime::instance().device();
    size_t t = device.recommendedMaxWorkingSetSize;
    size_t used = device.currentAllocatedSize;
    if (total)
        *total = t;
    if (free)
        *free = used < t ? t - used : 0;
    return hipSuccess;
    }

// ---------------------------------------------------------------------------------------------
// Error handling

hipError_t hipGetLastError()
    {
    return Runtime::instance().get_error();
    }

hipError_t hipPeekAtLastError()
    {
    return Runtime::instance().peek_error();
    }

const char* hipGetErrorName(hipError_t error)
    {
    switch (error)
        {
    case hipSuccess:
        return "hipSuccess";
    case hipErrorInvalidValue:
        return "hipErrorInvalidValue";
    case hipErrorOutOfMemory:
        return "hipErrorOutOfMemory";
    case hipErrorNotInitialized:
        return "hipErrorNotInitialized";
    case hipErrorInvalidConfiguration:
        return "hipErrorInvalidConfiguration";
    case hipErrorNoDevice:
        return "hipErrorNoDevice";
    case hipErrorInvalidDevice:
        return "hipErrorInvalidDevice";
    case hipErrorInvalidKernelFile:
        return "hipErrorInvalidKernelFile";
    case hipErrorNotReady:
        return "hipErrorNotReady";
    case hipErrorIllegalAddress:
        return "hipErrorIllegalAddress";
    case hipErrorLaunchOutOfResources:
        return "hipErrorLaunchOutOfResources";
    case hipErrorLaunchTimeOut:
        return "hipErrorLaunchTimeOut";
    case hipErrorHostMemoryAlreadyRegistered:
        return "hipErrorHostMemoryAlreadyRegistered";
    case hipErrorHostMemoryNotRegistered:
        return "hipErrorHostMemoryNotRegistered";
    case hipErrorLaunchFailure:
        return "hipErrorLaunchFailure";
    case hipErrorNotSupported:
        return "hipErrorNotSupported";
    default:
        return "hipErrorUnknown";
        }
    }

const char* hipGetErrorString(hipError_t error)
    {
    if (error == hipSuccess)
        return "no error";

    // Include the detailed message of the most recent error with this code. The string is stored
    // per thread so that it remains valid for the caller.
    thread_local std::string message;
    std::string detail = Runtime::instance().error_message(error);
    if (!detail.empty())
        {
        message = std::string(hipGetErrorName(error)) + ": " + detail;
        return message.c_str();
        }
    return hipGetErrorName(error);
    }

// ---------------------------------------------------------------------------------------------
// Memory management

hipError_t hipMalloc(void** ptr, size_t size)
    {
    return allocate(ptr, size);
    }

hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int)
    {
    return allocate(ptr, size);
    }

hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int)
    {
    return allocate(ptr, size);
    }

hipError_t hipFree(void* ptr)
    {
    return release(ptr, false);
    }

hipError_t hipHostFree(void* ptr)
    {
    return release(ptr, false);
    }

hipError_t hipHostRegister(void* ptr, size_t size, unsigned int)
    {
    if (!ptr)
        return hipErrorInvalidValue;
    if (hipError_t error = require_device())
        return error;
    if (size == 0)
        return hipSuccess;

    Runtime& rt = Runtime::instance();
    Allocation existing;
    if (rt.find_exact(ptr, existing))
        {
        return rt.set_error(hipErrorHostMemoryAlreadyRegistered,
                            "Host memory is already registered");
        }

    // Metal can only wrap whole pages
    const uintptr_t page = static_cast<uintptr_t>(getpagesize());
    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t base = p & ~(page - 1);
    uintptr_t end = (p + size + page - 1) & ~(page - 1);

    @autoreleasepool
        {
        id<MTLBuffer> buffer =
            [rt.device() newBufferWithBytesNoCopy:reinterpret_cast<void*>(base)
                                           length:end - base
                                          options:MTLResourceStorageModeShared
                                                  | MTLResourceHazardTrackingModeUntracked
                                      deallocator:nil];
        if (!buffer)
            {
            std::ostringstream s;
            s << "Failed to register " << size << " bytes of host memory at " << ptr
              << " with Metal";
            return rt.set_error(hipErrorOutOfMemory, s.str());
            }
        rt.add_allocation(ptr, Allocation{size, buffer, p - base, true});
        }
    return hipSuccess;
    }

hipError_t hipHostUnregister(void* ptr)
    {
    return release(ptr, true);
    }

hipError_t hipHostGetDevicePointer(void** device_ptr, void* host_ptr, unsigned int)
    {
    if (!device_ptr)
        return hipErrorInvalidValue;
    if (!is_allocated(host_ptr))
        {
        *device_ptr = nullptr;
        return Runtime::instance().set_error(hipErrorInvalidValue,
                                             "Host pointer is not registered with the HIP API");
        }
    // allocations are shared: the device pointer is the host pointer
    *device_ptr = host_ptr;
    return hipSuccess;
    }

hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind)
    {
    if (size == 0)
        return hipSuccess;
    if (!dst || !src)
        return hipErrorInvalidValue;
    hipError_t error = hipDeviceSynchronize();
    memmove(dst, src, size);
    return error;
    }

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size, hipMemcpyKind kind, hipStream_t)
    {
    if (size == 0)
        return hipSuccess;
    if (!dst || !src)
        return hipErrorInvalidValue;
    if (Runtime::instance().available() && is_allocated(dst) && is_allocated(src)
        && (kind == hipMemcpyDeviceToDevice || kind == hipMemcpyDefault))
        {
        return guarded(hipErrorLaunchFailure,
                       [&] { return Runtime::instance().copy_async(dst, src, size); });
        }
    return hipMemcpy(dst, src, size, kind);
    }

hipError_t hipMemset(void* ptr, int value, size_t size)
    {
    if (size == 0)
        return hipSuccess;
    if (!ptr)
        return hipErrorInvalidValue;
    hipError_t error = hipDeviceSynchronize();
    memset(ptr, value, size);
    return error;
    }

hipError_t hipMemsetAsync(void* ptr, int value, size_t size, hipStream_t)
    {
    if (size == 0)
        return hipSuccess;
    if (!ptr)
        return hipErrorInvalidValue;
    if (Runtime::instance().available() && is_allocated(ptr))
        {
        return guarded(hipErrorLaunchFailure,
                       [&] { return Runtime::instance().fill_async(ptr, value, size); });
        }
    return hipMemset(ptr, value, size);
    }

hipError_t hipMemAdvise(const void*, size_t, int, int)
    {
    return hipSuccess;
    }

hipError_t hipMemPrefetchAsync(const void*, size_t, int, hipStream_t)
    {
    return hipSuccess;
    }

// ---------------------------------------------------------------------------------------------
// Streams

hipError_t hipStreamCreate(hipStream_t* stream)
    {
    return hipStreamCreateWithFlags(stream, hipStreamDefault);
    }

hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags)
    {
    if (!stream)
        return hipErrorInvalidValue;
    *stream = new ihipStream_t {flags};
    return hipSuccess;
    }

hipError_t hipStreamDestroy(hipStream_t stream)
    {
    delete stream;
    return hipSuccess;
    }

hipError_t hipStreamSynchronize(hipStream_t)
    {
    return hipDeviceSynchronize();
    }

hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned int)
    {
    // there is only one in-order queue
    return hipSuccess;
    }

// ---------------------------------------------------------------------------------------------
// Events

hipError_t hipEventCreate(hipEvent_t* event)
    {
    return hipEventCreateWithFlags(event, hipEventDefault);
    }

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int)
    {
    if (!event)
        return hipErrorInvalidValue;
    *event = new ihipEvent_t();
    return hipSuccess;
    }

hipError_t hipEventDestroy(hipEvent_t event)
    {
    delete event;
    return hipSuccess;
    }

hipError_t hipEventRecord(hipEvent_t event, hipStream_t)
    {
    if (!event)
        return hipErrorInvalidHandle;
    if (!Runtime::instance().available())
        return hipSuccess;
    return guarded(hipErrorLaunchFailure,
                   [&]
                   {
                       Runtime::instance().record(event);
                       return hipSuccess;
                   });
    }

hipError_t hipEventSynchronize(hipEvent_t event)
    {
    if (!event)
        return hipErrorInvalidHandle;
    if (event->last)
        {
        Runtime::instance().wait_for(event->sequence, event->last);
        Runtime::instance().check_status(event->last);
        }
    return hipSuccess;
    }

hipError_t hipEventQuery(hipEvent_t event)
    {
    if (!event)
        return hipErrorInvalidHandle;
    if (event->last && event->last.status < MTLCommandBufferStatusCompleted)
        return hipErrorNotReady;
    return hipSuccess;
    }

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop)
    {
    if (!ms || !start || !stop)
        return hipErrorInvalidValue;
    if (!start->recorded || !stop->recorded)
        return hipErrorInvalidHandle;
    *ms = Runtime::instance().elapsed_ms(start, stop);
    return hipSuccess;
    }

// ---------------------------------------------------------------------------------------------
// Profiling

hipError_t hipProfilerStart()
    {
    const char* path = getenv("HOOMD_METAL_CAPTURE");
    if (!path || !Runtime::instance().available())
        return hipSuccess;
    @autoreleasepool
        {
        MTLCaptureDescriptor* descriptor = [MTLCaptureDescriptor new];
        descriptor.captureObject = Runtime::instance().device();
        descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
        descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        NSError* error = nil;
        if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor
                                                                             error:&error])
            {
            fprintf(stderr,
                    "HIP-on-Metal: failed to start GPU capture (set METAL_CAPTURE_ENABLED=1): %s\n",
                    error ? error.localizedDescription.UTF8String : "unknown error");
            }
        }
    return hipSuccess;
    }

hipError_t hipProfilerStop()
    {
    if (getenv("HOOMD_METAL_CAPTURE") && Runtime::instance().available())
        {
        hipDeviceSynchronize();
        [[MTLCaptureManager sharedCaptureManager] stopCapture];
        }
    return hipSuccess;
    }

// ---------------------------------------------------------------------------------------------
// Kernel launch API

namespace hoomd
    {
namespace mtl
    {
uint64_t device_address(const void* ptr)
    {
    return Runtime::instance().device_address(ptr);
    }

hipError_t launch(const KernelSource& source,
                  const char* kernel,
                  dim3 grid,
                  dim3 block,
                  const void* args,
                  size_t args_bytes,
                  size_t shared_bytes,
                  const FunctionConstant* constants,
                  unsigned int n_constants)
    {
    if (hipError_t error = require_device())
        return error;
    return guarded(hipErrorInvalidDeviceFunction,
                   [&]
                   {
                       return Runtime::instance().launch(source,
                                                         kernel,
                                                         grid,
                                                         block,
                                                         args,
                                                         args_bytes,
                                                         shared_bytes,
                                                         constants,
                                                         n_constants);
                   });
    }

unsigned int max_threads_per_threadgroup(const KernelSource& source,
                                         const char* kernel,
                                         const FunctionConstant* constants,
                                         unsigned int n_constants)
    {
    if (!Runtime::instance().available())
        return 0;
    return static_cast<unsigned int>(
        Runtime::instance().pipeline(source, kernel, constants, n_constants)
            .maxTotalThreadsPerThreadgroup);
    }

size_t static_threadgroup_memory(const KernelSource& source,
                                 const char* kernel,
                                 const FunctionConstant* constants,
                                 unsigned int n_constants)
    {
    if (!Runtime::instance().available())
        return 0;
    return Runtime::instance().pipeline(source, kernel, constants, n_constants)
        .staticThreadgroupMemoryLength;
    }

void flush()
    {
    if (Runtime::instance().available())
        Runtime::instance().commit();
    }

    } // end namespace mtl
    } // end namespace hoomd

//! Called by the stubs that make_unported_stubs.py generates for GPU driver functions that have no
//! Metal implementation yet.
extern "C" HIPMETAL_API [[noreturn]] void hoomd_mtl_unported_kernel(const char* name)
    {
    throw std::runtime_error(std::string(name)
                             + " is not yet available on the Metal GPU platform. Run this "
                               "simulation with hoomd.device.CPU().");
    }
