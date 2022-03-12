#include <cstdint>
#include <memory>
#include <stdio.h>

#include "cuda_try.cuh"
#include "data_generator.cuh"
#include <cub/cub.cuh>
#include <algorithm>

struct index_iterator {
    size_t idx;
    size_t* storage;

    __host__ __device__ index_iterator(size_t idx = 0, size_t* storage = NULL)
        : idx(idx), storage(storage)
    {
    }

    __host__ __device__ size_t operator[](size_t i)
    {
        return idx + i;
    }

    __host__ __device__ index_iterator operator+(size_t i)
    {
        return index_iterator{idx + i};
    }

    __host__ __device__ index_iterator operator-(size_t i)
    {
        return index_iterator{idx - i};
    }

    __host__ __device__ size_t& operator*()
    {
        return storage[idx];
    }

    __host__ __device__ size_t operator++()
    {
        idx++;
        return idx;
    }

    __host__ __device__ size_t operator--()
    {
        idx--;
        return idx;
    }
};

template <> struct std::iterator_traits<index_iterator> {
    typedef size_t value_type;
};

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
    T* input, size_t count, T** p_output_elements, size_t** p_output_indices)
{
    void* cub_temp;
    size_t cub_temp_size = 0;
    T* output_elements;
    size_t* input_indices;
    size_t* output_indices;
    cub::DeviceRadixSort::SortPairs(
        NULL, cub_temp_size, input, output_elements, input_indices,
        output_indices, count);
    CUDA_TRY(cudaMalloc(&output_elements, count * sizeof(T)));
    CUDA_TRY(cudaMalloc(&input_indices, count * sizeof(size_t)));
    CUDA_TRY(cudaMalloc(&output_indices, count * sizeof(size_t)));
    CUDA_TRY(cudaMalloc(&cub_temp, cub_temp_size));
    kernel_gen_list<<<1024, 32>>>(input_indices, count);
    cub::DeviceRadixSort::SortPairs(
        cub_temp, cub_temp_size, input, output_elements, input_indices,
        output_indices, count);
    CUDA_TRY(cudaFree(cub_temp));
    CUDA_TRY(cudaFree(input_indices));
    *p_output_elements = output_elements;
    *p_output_indices = output_indices;
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

int main(int argc, char** argv)
{
    size_t element_count = 1 << 5;
    size_t elements_size = sizeof(uint32_t) * element_count;
    size_t indices_size = sizeof(size_t) * element_count;
    size_t distinct_elements = 1 << 3;
    uint32_t* h_data = (uint32_t*)malloc(elements_size);
    uint32_t* d_data = NULL;
    CUDA_TRY(cudaMalloc(&d_data, elements_size));
    gpu_generate_data<<<10, 32>>>(d_data, element_count, distinct_elements);
    size_t *d_out_indices, *h_out_indices;
    uint32_t *d_out_elements, *h_out_elements;
    conflictdetect_sort(d_data, element_count, &d_out_elements, &d_out_indices);
    CUDA_TRY(cudaMemcpy(h_data, d_data, elements_size, cudaMemcpyDeviceToHost));

    h_out_elements = (uint32_t*)malloc(elements_size);
    h_out_indices = (size_t*)malloc(indices_size);
    CUDA_TRY(cudaMemcpy(
        h_out_indices, d_out_indices, indices_size, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(
        h_out_elements, d_out_elements, elements_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < element_count; i++) {
        printf("%d : %d [", i, h_data[i]);
        size_t *conflicts_start, *conflicts_end;
        conflictdetect_sort_get_matrix_element(
            h_data[i], i, element_count, h_out_elements, h_out_indices,
            &conflicts_start, &conflicts_end);
        for (size_t* i = conflicts_start; i != conflicts_end; i++) {
            printf("%zu%s", *i, i + 1 == conflicts_end ? "" : ", ");
        }
        printf("]\n");
    }
    return 0;
}
