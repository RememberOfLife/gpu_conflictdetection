#pragma once

#include <cstdint>

#include "fast_prng.cuh"

// TODO uses uint32_t for now, make templates?

void cpu_generate_data(
    uint32_t* h_buffer,
    uint64_t element_count,
    uint64_t distinct_elements,
    uint64_t seed = 42)
{
    fast_prng rng(seed);
    for (int i = 0; i < element_count; i++) {
        h_buffer[i] = rng.rand() % distinct_elements;
    }
}

__global__ void gpu_generate_data(
    uint32_t* d_buffer,
    uint64_t element_count,
    uint64_t distinct_elements,
    uint64_t seed = 42)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t gridstride = blockDim.x * gridDim.x;
    fast_prng rng(seed + tid);
    for (uint64_t i = tid; i < element_count; i += gridstride) {
        d_buffer[i] = rng.rand() % distinct_elements;
    }
}
