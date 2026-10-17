// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Unit tests for the HIP-on-Metal runtime and the Metal primitives.

#include "hoomd/mtl/Primitives.h"

#include <hipfft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                                            \
    do                                                                                         \
        {                                                                                      \
        if (!(cond))                                                                           \
            {                                                                                  \
            fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                  \
            failures++;                                                                        \
            }                                                                                  \
        } while (0)

#define HIP_CHECK(expr) CHECK((expr) == hipSuccess)

template<class T> T* device_array(const std::vector<T>& h)
    {
    T* d = nullptr;
    HIP_CHECK(hipMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty())
        HIP_CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
    }

template<class T> std::vector<T> host_array(const T* d, size_t n)
    {
    std::vector<T> h(n);
    if (n)
        HIP_CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
    }

void test_device()
    {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    CHECK(count == 1);
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("device: %s, %d cores, family %d, %zu MiB, %zu bytes threadgroup memory\n",
           prop.name,
           prop.multiProcessorCount,
           prop.major,
           prop.totalGlobalMem >> 20,
           prop.sharedMemPerBlock);
    CHECK(prop.warpSize == 32);
    CHECK(prop.canMapHostMemory == 1);
    }

void test_memory()
    {
    const size_t n = 100003;
    std::vector<unsigned int> h(n);
    std::iota(h.begin(), h.end(), 7u);
    unsigned int* d = device_array(h);
    unsigned int* d2 = nullptr;
    HIP_CHECK(hipMalloc(&d2, n * sizeof(unsigned int)));

    // async device to device copy with an unaligned offset
    HIP_CHECK(hipMemcpyAsync(d2 + 1, d + 3, (n - 3) * sizeof(unsigned int), hipMemcpyDeviceToDevice));
    auto r = host_array(d2, n);
    bool ok = true;
    for (size_t i = 0; i < n - 3; i++)
        ok = ok && r[i + 1] == h[i + 3];
    CHECK(ok);

    // async memset of a byte range that is not a multiple of 16
    HIP_CHECK(hipMemsetAsync(reinterpret_cast<char*>(d2) + 5, 0xab, 1001));
    HIP_CHECK(hipDeviceSynchronize());
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(d2);
    ok = true;
    for (size_t i = 5; i < 1006; i++)
        ok = ok && bytes[i] == 0xab;
    CHECK(ok);
    CHECK(bytes[1006] != 0xab || bytes[1006] == reinterpret_cast<const unsigned char*>(r.data())[1006]);

    // registered host memory
    void* host = nullptr;
    CHECK(posix_memalign(&host, 32, 1000 * sizeof(unsigned int)) == 0);
    HIP_CHECK(hipHostRegister(host, 1000 * sizeof(unsigned int), hipHostRegisterMapped));
    unsigned int* dptr = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(&dptr, host, 0));
    CHECK(dptr == host);
    hoomd::mtl::fill(dptr, 42u, 1000);
    HIP_CHECK(hipDeviceSynchronize());
    CHECK(static_cast<unsigned int*>(host)[999] == 42u);
    HIP_CHECK(hipHostUnregister(host));
    free(host);

    // freeing memory that was not allocated fails
    int x;
    CHECK(hipFree(&x) != hipSuccess);
    (void)hipGetLastError();

    HIP_CHECK(hipFree(d));
    HIP_CHECK(hipFree(d2));
    }

template<class T> void test_scan(unsigned int n, std::mt19937& rng)
    {
    std::uniform_int_distribution<int> dist(0, 9);
    std::vector<T> h(n);
    for (auto& v : h)
        v = T(dist(rng));
    T* d_in = device_array(h);
    T* d_out = device_array(std::vector<T>(n));

    hoomd::mtl::exclusive_scan(d_in, d_out, n);
    auto exc = host_array(d_out, n);
    hoomd::mtl::inclusive_scan(d_in, d_out, n);
    auto inc = host_array(d_out, n);

    T running = 0;
    bool ok_exc = true, ok_inc = true;
    for (unsigned int i = 0; i < n; i++)
        {
        ok_exc = ok_exc && exc[i] == running;
        running += h[i];
        ok_inc = ok_inc && inc[i] == running;
        }
    CHECK(ok_exc);
    CHECK(ok_inc);

    // in place
    hoomd::mtl::exclusive_scan(d_in, d_in, n);
    auto inplace = host_array(d_in, n);
    CHECK(inplace == exc);

    // reductions
    T* d_result = device_array(std::vector<T>(1));
    HIP_CHECK(hipMemcpy(d_in, h.data(), n * sizeof(T), hipMemcpyHostToDevice));
    hoomd::mtl::reduce_sum(d_in, d_result, n);
    CHECK(host_array(d_result, 1)[0] == running);
    hoomd::mtl::reduce_max(d_in, d_result, n);
    T expected_max = n ? *std::max_element(h.begin(), h.end()) : std::numeric_limits<T>::lowest();
    CHECK(host_array(d_result, 1)[0] == expected_max);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_result));
    }

void test_scan64(unsigned int n)
    {
    std::vector<size_t> h(n);
    for (unsigned int i = 0; i < n; i++)
        h[i] = (size_t(i % 7) << 31) + i; // values that overflow 32 bits when summed
    size_t* d_in = device_array(h);
    size_t* d_out = device_array(std::vector<size_t>(n));
    hoomd::mtl::exclusive_scan(d_in, d_out, n);
    auto out = host_array(d_out, n);
    size_t running = 0;
    bool ok = true;
    for (unsigned int i = 0; i < n; i++)
        {
        ok = ok && out[i] == running;
        running += h[i];
        }
    CHECK(ok);
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    }

void test_sort(unsigned int n, unsigned int end_bit, std::mt19937& rng)
    {
    std::uniform_int_distribution<unsigned int> dist(0, 0xffffffffu);
    std::vector<unsigned int> keys(n), values(n);
    for (unsigned int i = 0; i < n; i++)
        {
        keys[i] = dist(rng) % 1000; // many duplicates to test stability
        if (end_bit < 32)
            keys[i] |= dist(rng) << end_bit; // bits that must be ignored
        values[i] = i;
        }
    unsigned int* d_keys = device_array(keys);
    unsigned int* d_values = device_array(values);
    hoomd::mtl::sort_pairs(d_keys, d_values, n, 0, end_bit);
    auto sorted_keys = host_array(d_keys, n);
    auto sorted_values = host_array(d_values, n);

    unsigned int mask = end_bit < 32 ? (1u << end_bit) - 1u : 0xffffffffu;
    std::vector<unsigned int> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(),
                     order.end(),
                     [&](unsigned int a, unsigned int b)
                     { return (keys[a] & mask) < (keys[b] & mask); });
    bool ok = true;
    for (unsigned int i = 0; i < n; i++)
        ok = ok && sorted_values[i] == order[i] && sorted_keys[i] == keys[order[i]];
    CHECK(ok);
    HIP_CHECK(hipFree(d_keys));
    HIP_CHECK(hipFree(d_values));
    }

void test_select(unsigned int n, std::mt19937& rng)
    {
    std::uniform_int_distribution<unsigned int> dist(0, 3);
    std::vector<unsigned int> in(n), flags(n);
    std::vector<unsigned char> flags8(n);
    for (unsigned int i = 0; i < n; i++)
        {
        in[i] = i * 3;
        flags[i] = dist(rng) == 0 ? 2 : 0;
        flags8[i] = flags[i] ? 1 : 0;
        }
    unsigned int* d_in = device_array(in);
    unsigned int* d_flags = device_array(flags);
    unsigned char* d_flags8 = device_array(flags8);
    unsigned int* d_out = device_array(std::vector<unsigned int>(n));
    unsigned int* d_num = device_array(std::vector<unsigned int>(1));

    std::vector<unsigned int> selected, rejected;
    for (unsigned int i = 0; i < n; i++)
        (flags[i] ? selected : rejected).push_back(in[i]);

    hoomd::mtl::select_flagged(d_in, d_flags, d_out, d_num, n);
    unsigned int num = host_array(d_num, 1)[0];
    CHECK(num == selected.size());
    auto out = host_array(d_out, num);
    CHECK(out == selected);

    hoomd::mtl::partition_flagged(d_in, d_flags8, d_out, d_num, n);
    CHECK(host_array(d_num, 1)[0] == selected.size());
    std::vector<unsigned int> expected = selected;
    expected.insert(expected.end(), rejected.rbegin(), rejected.rend());
    CHECK(host_array(d_out, n) == expected);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_flags));
    HIP_CHECK(hipFree(d_flags8));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_num));
    }

void test_segments(unsigned int n, std::mt19937& rng)
    {
    std::uniform_int_distribution<unsigned int> dist(0, 4);
    std::vector<unsigned int> keys(n), values(n);
    unsigned int k = 0;
    for (unsigned int i = 0; i < n; i++)
        {
        if (dist(rng) == 0)
            k += 1 + dist(rng);
        keys[i] = k;
        values[i] = dist(rng);
        }
    unsigned int* d_keys = device_array(keys);
    unsigned int* d_values = device_array(values);
    unsigned int* d_out = device_array(std::vector<unsigned int>(n));

    hoomd::mtl::exclusive_scan_by_key(d_keys, nullptr, d_out, n);
    auto rank = host_array(d_out, n);
    hoomd::mtl::exclusive_scan_by_key(d_keys, d_values, d_out, n);
    auto scan = host_array(d_out, n);

    bool ok_rank = true, ok_scan = true;
    unsigned int r = 0, s = 0;
    for (unsigned int i = 0; i < n; i++)
        {
        if (i > 0 && keys[i] != keys[i - 1])
            {
            r = 0;
            s = 0;
            }
        ok_rank = ok_rank && rank[i] == r;
        ok_scan = ok_scan && scan[i] == s;
        r++;
        s += values[i];
        }
    CHECK(ok_rank);
    CHECK(ok_scan);

    // reduce by key
    std::vector<unsigned int> unique_keys, sums;
    for (unsigned int i = 0; i < n; i++)
        {
        if (i == 0 || keys[i] != keys[i - 1])
            {
            unique_keys.push_back(keys[i]);
            sums.push_back(0);
            }
        sums.back() += values[i];
        }
    unsigned int* d_keys_out = device_array(std::vector<unsigned int>(n));
    unsigned int* d_num = device_array(std::vector<unsigned int>(1));
    hoomd::mtl::reduce_by_key(d_keys, d_values, d_keys_out, d_out, d_num, n);
    unsigned int runs = host_array(d_num, 1)[0];
    CHECK(runs == unique_keys.size());
    CHECK(host_array(d_keys_out, runs) == unique_keys);
    CHECK(host_array(d_out, runs) == sums);

    // binary search
    std::vector<unsigned int> queries = {0, keys.empty() ? 0 : keys[n / 2], k, k + 1};
    unsigned int* d_queries = device_array(queries);
    unsigned int* d_result = device_array(std::vector<unsigned int>(queries.size()));
    hoomd::mtl::lower_bound(d_keys, n, d_queries, (unsigned int)queries.size(), d_result);
    auto lower = host_array(d_result, queries.size());
    hoomd::mtl::upper_bound(d_keys, n, d_queries, (unsigned int)queries.size(), d_result);
    auto upper = host_array(d_result, queries.size());
    for (size_t q = 0; q < queries.size(); q++)
        {
        CHECK(lower[q]
              == (unsigned int)(std::lower_bound(keys.begin(), keys.end(), queries[q])
                                - keys.begin()));
        CHECK(upper[q]
              == (unsigned int)(std::upper_bound(keys.begin(), keys.end(), queries[q])
                                - keys.begin()));
        }

    for (void* p : {(void*)d_keys,
                    (void*)d_values,
                    (void*)d_out,
                    (void*)d_keys_out,
                    (void*)d_num,
                    (void*)d_queries,
                    (void*)d_result})
        HIP_CHECK(hipFree(p));
    }

void test_events()
    {
    const unsigned int n = 1 << 22;
    unsigned int* d = nullptr;
    HIP_CHECK(hipMalloc(&d, n * sizeof(unsigned int)));
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    hoomd::mtl::sequence(d, n);
    hoomd::mtl::sort_keys(d, n);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    printf("sort of %u keys: %.3f ms GPU time\n", n, ms);
    CHECK(ms > 0);
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d));
    }

//! Direct DFT of a row major array along all axes (double precision reference)
std::vector<std::complex<double>> reference_dft(const std::vector<std::complex<double>>& in,
                                                const std::vector<int>& dims,
                                                int sign)
    {
    std::vector<std::complex<double>> data = in;
    size_t total = data.size();
    size_t stride = 1;
    for (int axis = (int)dims.size() - 1; axis >= 0; axis--)
        {
        size_t n = dims[axis];
        std::vector<std::complex<double>> out(total);
        for (size_t line = 0; line < total / n; line++)
            {
            size_t base = (line / stride) * (n * stride) + (line % stride);
            for (size_t k = 0; k < n; k++)
                {
                std::complex<double> acc = 0;
                for (size_t j = 0; j < n; j++)
                    acc += data[base + j * stride]
                           * std::polar(1.0, sign * 2.0 * M_PI * double(j * k % n) / double(n));
                out[base + k * stride] = acc;
                }
            }
        data = out;
        stride *= n;
        }
    return data;
    }

void test_fft(std::vector<int> dims, bool in_place, std::mt19937& rng)
    {
    size_t total = 1;
    for (int d : dims)
        total *= d;
    std::normal_distribution<double> dist;
    std::vector<std::complex<double>> input(total);
    std::vector<hipfftComplex> h(total);
    for (size_t i = 0; i < total; i++)
        {
        input[i] = {dist(rng), dist(rng)};
        h[i].x = float(input[i].real());
        h[i].y = float(input[i].imag());
        }
    hipfftComplex* d_in = device_array(h);
    hipfftComplex* d_out = in_place ? d_in : device_array(std::vector<hipfftComplex>(total));

    hipfftHandle plan;
    hipfftResult result;
    if (dims.size() == 1)
        result = hipfftPlan1d(&plan, dims[0], HIPFFT_C2C, 1);
    else if (dims.size() == 2)
        result = hipfftPlan2d(&plan, dims[0], dims[1], HIPFFT_C2C);
    else
        result = hipfftPlan3d(&plan, dims[0], dims[1], dims[2], HIPFFT_C2C);
    CHECK(result == HIPFFT_SUCCESS);

    for (int sign : {HIPFFT_FORWARD, HIPFFT_BACKWARD})
        {
        if (!in_place)
            HIP_CHECK(hipMemcpy(d_in, h.data(), total * sizeof(hipfftComplex), hipMemcpyHostToDevice));
        else
            HIP_CHECK(hipMemcpy(d_in, h.data(), total * sizeof(hipfftComplex), hipMemcpyHostToDevice));
        CHECK(hipfftExecC2C(plan, d_in, d_out, sign) == HIPFFT_SUCCESS);
        auto out = host_array(d_out, total);
        auto expected = reference_dft(input, dims, sign);
        double max_err = 0, max_val = 0;
        for (size_t i = 0; i < total; i++)
            {
            max_err = std::max(max_err,
                               std::abs(std::complex<double>(out[i].x, out[i].y) - expected[i]));
            max_val = std::max(max_val, std::abs(expected[i]));
            }
        double rel = max_err / max_val;
        if (rel > 2e-5)
            printf("fft dims %zu: relative error %g\n", dims.size(), rel);
        CHECK(rel < 2e-5);
        }
    CHECK(hipfftDestroy(plan) == HIPFFT_SUCCESS);
    HIP_CHECK(hipFree(d_in));
    if (!in_place)
        HIP_CHECK(hipFree(d_out));
    }

int main()
    {
    std::mt19937 rng(42);
    test_device();
    test_memory();
    for (unsigned int n : {0u, 1u, 31u, 2048u, 2049u, 100000u, 5000000u})
        {
        test_scan<unsigned int>(n, rng);
        test_scan<int>(n, rng);
        if (n <= 100000)
            test_scan<float>(n, rng);
        }
    for (unsigned int n : {2u, 255u, 2048u, 4097u, 300000u})
        {
        test_sort(n, 32, rng);
        test_sort(n, 10, rng);
        test_select(n, rng);
        test_segments(n, rng);
        }
    test_events();
    for (unsigned int n : {1u, 1000u, 3000000u})
        test_scan64(n);
    test_fft({8}, true, rng);
    test_fft({2048}, false, rng);
    test_fft({13}, true, rng);
    test_fft({60}, false, rng);
    test_fft({12, 10, 7}, true, rng);
    test_fft({16, 16, 16}, false, rng);
    test_fft({32, 24, 22}, true, rng);
    test_fft({5, 9}, false, rng);

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    if (failures)
        {
        printf("%d checks failed\n", failures);
        return 1;
        }
    printf("all checks passed\n");
    return 0;
    }
