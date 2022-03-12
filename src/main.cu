#include <cstdint>
#include <memory>
#include <stdio.h>

#include "cuda_try.cuh"
#include "data_generator.cuh"
#include <cub/cub.cuh>

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

int main(int argc, char** argv)
{
    uint32_t element_count = 1 << 10;
    uint32_t distinct_elements = 1 << 5;
    uint32_t* h_data = (uint32_t*)malloc(sizeof(uint32_t) * element_count);
    uint32_t* d_data = NULL;
    CUDA_TRY(cudaMalloc(&d_data, sizeof(uint32_t) * element_count));
    gpu_generate_data<<<10, 32>>>(d_data, element_count, distinct_elements);
    size_t* out_indices;
    uint32_t* out_elements;
    conflictdetect_sort(d_data, element_count, &out_elements, &out_indices);
    CUDA_TRY(cudaMemcpy(
        h_data, d_data, sizeof(uint32_t) * element_count,
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < element_count; i++) {
        printf("%d : %d\n", i, h_data[i]);
    }
    return 0;
}
