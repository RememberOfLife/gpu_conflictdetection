#pragma once

#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <cub/cub.cuh>
#include "cuda_try.cuh"

__global__ void kernel_gen_list(size_t* output, size_t element_count)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridstride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < element_count; i += gridstride) {
        output[i] = i;
    }
}

template <typename T>
void conflictdetect_sort(
    T* d_input,
    size_t element_count,
    T* d_output_elements,
    size_t* d_output_indices)
{
    void* cub_temp;
    size_t cub_temp_size = 0;
    size_t* input_indices;
    cub::DeviceRadixSort::SortPairs(
        NULL, cub_temp_size, d_input, d_output_elements, input_indices,
        d_output_indices, element_count);
    CUDA_TRY(cudaMalloc(&cub_temp, cub_temp_size));
    // we rely on the observed, but apparently not documented behavior that
    // cub::DeviceRadixSort::SortPairs can work in place
    // (with d_values_in = d_values_out)
    // to avoid this assumption, the uncommented malloc and free below have to
    // be used
    input_indices = d_output_indices;
    // CUDA_TRY(cudaMalloc(&input_indices, element_count * sizeof(size_t)));
    kernel_gen_list<<<1024, 32>>>(input_indices, element_count);
    cub::DeviceRadixSort::SortPairs(
        cub_temp, cub_temp_size, d_input, d_output_elements, input_indices,
        d_output_indices, element_count);
    CUDA_TRY(cudaFree(cub_temp));
    // CUDA_TRY(cudaFree(input_indices));
}

template <typename T>
void conflictdetect_sort_get_matrix_element(
    T element,
    size_t input_index,
    size_t element_count,
    T* element_list,
    size_t* indices_list,
    size_t** conflicts_start,
    size_t** conflicts_end)
{
    T* l =
        std::lower_bound(element_list, element_list + element_count, element);
    T* r = std::upper_bound(l, element_list + element_count, element);
    *conflicts_start = indices_list + (l - element_list);
    *conflicts_end = std::lower_bound(
        indices_list + (l - element_list), indices_list + (r - element_list),
        input_index);
}