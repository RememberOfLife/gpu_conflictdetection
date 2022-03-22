#pragma once

#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <cub/cub.cuh>
#include "cuda_try.cuh"

template <typename T>
struct sort_buffers {
    T* out_elements;
    size_t* out_indices;
    size_t element_count;
};

template <typename T>
void sort_buffers_init_d(sort_buffers<T>* sb, size_t element_count)
{
    sb->element_count = element_count;
    CUDA_TRY(cudaMalloc(&sb->out_elements, sizeof(T) * element_count));
    CUDA_TRY(cudaMalloc(&sb->out_indices, sizeof(size_t) * element_count));
}

template <typename T>
void sort_buffers_d2h(sort_buffers<T>* h_sb, sort_buffers<T>* d_sb)
{
    h_sb->element_count = d_sb->element_count;
    h_sb->out_indices = (size_t*)malloc(h_sb->element_count * sizeof(size_t));
    h_sb->out_elements = (T*)malloc(h_sb->element_count * sizeof(T));

    CUDA_TRY(cudaMemcpy(
        h_sb->out_elements, d_sb->out_elements, sizeof(T) * h_sb->element_count,
        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(
        h_sb->out_indices, d_sb->out_indices,
        sizeof(size_t) * h_sb->element_count, cudaMemcpyDeviceToHost));
}

template <typename T>
void sort_buffers_fin_d(sort_buffers<T>* sb)
{
    CUDA_TRY(cudaFree(sb->out_indices));
    CUDA_TRY(cudaFree(sb->out_elements));
}

template <typename T>
void sort_buffers_fin_h(sort_buffers<T>* sb)
{
    free(sb->out_indices);
    free(sb->out_elements);
}

__global__ void kernel_gen_list(size_t* output, size_t element_count)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridstride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < element_count; i += gridstride) {
        output[i] = i;
    }
}

template <typename T>
void conflictdetect_sort(T* d_input, sort_buffers<T>* d_sb)
{
    void* cub_temp;
    size_t cub_temp_size = 0;
    size_t* input_indices;
    cub::DeviceRadixSort::SortPairs(
        NULL, cub_temp_size, d_input, d_sb->out_elements, input_indices,
        d_sb->out_indices, d_sb->element_count);
    CUDA_TRY(cudaMalloc(&cub_temp, cub_temp_size));
    // we rely on the observed, but apparently not documented behavior that
    // cub::DeviceRadixSort::SortPairs can work in place
    // (with d_values_in = d_values_out)
    // to avoid this assumption, the uncommented malloc and free below have to
    // be used
    input_indices = d_sb->out_indices;
    // CUDA_TRY(cudaMalloc(&input_indices, element_count * sizeof(size_t)));
    kernel_gen_list<<<1024, 32>>>(input_indices, d_sb->element_count);
    cub::DeviceRadixSort::SortPairs(
        cub_temp, cub_temp_size, d_input, d_sb->out_elements, input_indices,
        d_sb->out_indices, d_sb->element_count);
    CUDA_TRY(cudaFree(cub_temp));
    // CUDA_TRY(cudaFree(input_indices));
}

template <typename T>
void conflictdetect_sort_get_matrix_element(
    T element,
    size_t input_index,
    sort_buffers<T>* h_sb,
    size_t** conflicts_start,
    size_t** conflicts_end)
{
    T* l = std::lower_bound(
        h_sb->out_elements, h_sb->out_elements + h_sb->element_count, element);
    T* r =
        std::upper_bound(l, h_sb->out_elements + h_sb->element_count, element);
    *conflicts_start = h_sb->out_indices + (l - h_sb->out_elements);
    *conflicts_end = std::lower_bound(
        h_sb->out_indices + (l - h_sb->out_elements),
        h_sb->out_indices + (r - h_sb->out_elements), input_index);
}
