#include <cstdint>
#include <memory>
#include <stdio.h>

#include "cuda_try.cuh"
#include "data_generator.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include <string>
#include <sstream>

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
    T* input, size_t count, T* output_elements, size_t* output_indices)
{
    void* cub_temp;
    size_t cub_temp_size = 0;
    size_t* input_indices;
    cub::DeviceRadixSort::SortPairs(
        NULL, cub_temp_size, input, output_elements, input_indices,
        output_indices, count);
    CUDA_TRY(cudaMalloc(&cub_temp, cub_temp_size));
    // we rely on the observed, but apparently not documented behavior that
    // cub::DeviceRadixSort::SortPairs can work in place
    // (with d_values_in = d_values_out)
    // to avoid this assumption, the uncommented malloc and free below have to
    // be used
    input_indices = output_indices;
    // CUDA_TRY(cudaMalloc(&input_indices, count * sizeof(size_t)));
    kernel_gen_list<<<1024, 32>>>(input_indices, count);
    cub::DeviceRadixSort::SortPairs(
        cub_temp, cub_temp_size, input, output_elements, input_indices,
        output_indices, count);
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
void test_sort()
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
    CUDA_TRY(cudaMalloc(&d_out_elements, elements_size));
    CUDA_TRY(cudaMalloc(&d_out_indices, indices_size));
    conflictdetect_sort(d_data, element_count, d_out_elements, d_out_indices);
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
}
#ifdef CATCH_CONFIG_DISABLE
int main(int argc, char** argv)
{
    test_sort();
    return 0;
}
#else
#define CATCH_CONFIG_MAIN
#endif

#include <catch2/catch.hpp>

template <typename T, size_t ELEMENT_COUNT>
class ConflictdetectSortTestFixture {
  public:
    T* h_data;
    T* d_data;
    T* h_out_elements;
    T* d_out_elements;
    size_t* h_out_indices;
    size_t* d_out_indices;
    size_t element_count;

  public:
    ConflictdetectSortTestFixture()
    {
        element_count = ELEMENT_COUNT;
        CUDA_TRY(cudaMalloc(&d_data, ELEMENT_COUNT * sizeof(T)));
        CUDA_TRY(cudaMalloc(&d_out_elements, ELEMENT_COUNT * sizeof(T)));
        CUDA_TRY(cudaMalloc(&d_out_indices, ELEMENT_COUNT * sizeof(size_t)));
        h_data = (T*)malloc(ELEMENT_COUNT * sizeof(T));
        h_out_elements = (T*)malloc(ELEMENT_COUNT * sizeof(T));
        h_out_indices = (size_t*)malloc(ELEMENT_COUNT * sizeof(size_t));
    }

    ~ConflictdetectSortTestFixture()
    {
        CUDA_TRY(cudaFree(d_data));
        CUDA_TRY(cudaFree(d_out_elements));
        CUDA_TRY(cudaFree(d_out_indices));
        free(h_data);
        free(h_out_elements);
        free(h_out_indices);
    }

    void input2gpu(T* input)
    {
        memcpy(h_data, input, ELEMENT_COUNT * sizeof(T));
        CUDA_TRY(cudaMemcpy(
            d_data, h_data, ELEMENT_COUNT * sizeof(T), cudaMemcpyHostToDevice));
    }

    void output2cpu()
    {
        CUDA_TRY(cudaMemcpy(
            h_out_indices, d_out_indices, element_count * sizeof(size_t),
            cudaMemcpyDeviceToHost));
        CUDA_TRY(cudaMemcpy(
            h_out_elements, d_out_elements, element_count * sizeof(T),
            cudaMemcpyDeviceToHost));
    }

    std::string output2string()
    {
        std::ostringstream ss{};
        for (int i = 0; i < element_count; i++) {
            ss << i;
            ss << " : ";
            ss << h_data[i];
            ss << " [";
            size_t *conflicts_start, *conflicts_end;
            conflictdetect_sort_get_matrix_element(
                h_data[i], i, element_count, h_out_elements, h_out_indices,
                &conflicts_start, &conflicts_end);

            for (size_t* i = conflicts_start; i != conflicts_end; i++) {
                ss << *i;
                if (i + 1 != conflicts_end) ss << ", ";
            }
            ss << "]\n";
        }
        return ss.str();
    }
};

TEST_CASE("conflictdetect_sort basic functionality", "[conflictdetect_sort]")
{
    ConflictdetectSortTestFixture<uint32_t, 10> f{};
    uint32_t input[10] = {2, 1, 1, 1, 2, 2, 3, 7, 4, 0};
    f.input2gpu(input);
    conflictdetect_sort(
        f.d_data, f.element_count, f.d_out_elements, f.d_out_indices);
    f.output2cpu();
    // clang-format off
    REQUIRE(
        f.output2string() == (
R"(0 : 2 []
1 : 1 []
2 : 1 [1]
3 : 1 [1, 2]
4 : 2 [0]
5 : 2 [0, 4]
6 : 3 []
7 : 7 []
8 : 4 []
9 : 0 []
)"
        )
    );
    // clang-format on
}

TEST_CASE("conflictdetect_sort works on empty data", "[conflictdetect_sort]")
{
    ConflictdetectSortTestFixture<uint32_t, 0> f{};
    uint32_t input[0] = {};
    f.input2gpu(input);
    conflictdetect_sort(
        f.d_data, f.element_count, f.d_out_elements, f.d_out_indices);
    f.output2cpu();
    REQUIRE(f.output2string() == "");
}
