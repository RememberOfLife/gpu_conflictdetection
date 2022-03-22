#pragma once
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <bitset>
#include "cuda_try.cuh"

#define UNUSED(VAR) (void)(true ? (void)0 : ((void)(VAR)))

void error(const char* error)
{
    fputs(error, stderr);
    fputs("\n", stderr);
    assert(false);
    exit(EXIT_FAILURE);
}
void alloc_failure()
{
    error("memory allocation failed");
}

template <typename T>
void cpu_buffer_print(T* h_buffer, uint32_t offset, uint32_t length)
{
    for (uint32_t i = offset; i < offset + length; i++) {
        std::bitset<sizeof(T) * 8> bits(h_buffer[i]);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
}

template <typename T>
void gpu_buffer_print(T* d_buffer, uint32_t offset, uint32_t length)
{
    T* h_buffer = static_cast<T*>(malloc(length * sizeof(T)));
    CUDA_TRY(cudaMemcpy(
        h_buffer, d_buffer + offset, length * sizeof(T),
        cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < length; i++) {
        std::bitset<sizeof(T) * 8> bits(h_buffer[i]);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
    free(h_buffer);
}

template <class T> struct dont_deduce_t {
    using type = T;
};

size_t ceil2pow2(size_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    if (sizeof(size_t) == sizeof(uint64_t)) {
        v |= v >> 32;
    }
    v++;
    return v;
}

template <typename T>
__device__ __host__ T ceil2mult(T val, typename dont_deduce_t<T>::type mult)
{
    T rem = val % mult;
    if (rem) return val + mult - rem;
    return val;
}

template <typename T>
__device__ __host__ T ceildiv(T div, typename dont_deduce_t<T>::type divisor)
{
    T rem = div / divisor;
    if (rem * divisor == div) return rem;
    return rem + 1;
}

template <typename T>
__device__ __host__ T overlap(T value, typename dont_deduce_t<T>::type align)
{
    T rem = value % align;
    if (rem) return align - rem;
    return 0;
}
