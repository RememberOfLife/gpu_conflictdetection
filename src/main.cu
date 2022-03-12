#include <cstdint>
#include <memory>
#include <stdio.h>

#include "cuda_try.cuh"
#include "data_generator.cuh"

int main(int argc, char** argv)
{
    uint32_t element_count = 1 << 10;
    uint32_t distinct_elements = 1 << 5;
    uint32_t* h_data = (uint32_t*)malloc(sizeof(uint32_t) * element_count);
    uint32_t* d_data = NULL;
    CUDA_TRY(cudaMalloc(&d_data, sizeof(uint32_t) * element_count));
    gpu_generate_data<<<10, 32>>>(d_data, element_count, distinct_elements);
    CUDA_TRY(cudaMemcpy(
        h_data, d_data, sizeof(uint32_t) * element_count,
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < element_count; i++) {
        printf("%d : %d\n", i, h_data[i]);
    }
    return 0;
}
