#include <cstdint>
#include <memory>
#include <stdio.h>
#include <string>
#include <sstream>
#include <cstdlib>

#include "cuda_try.cuh"
#include "data_generator.cuh"
#include "conflictdetect_sort.cuh"
#include "conflictdetect_hashtable.cuh"

template <typename T> struct input_data {
    size_t element_count;
    T* h_input;
    T* d_input;
};

template <typename T>
void input_data_print_from_sort(
    input_data<T>* in_data,
    std::ostream& ss,
    T* h_out_elements,
    size_t* h_out_indices)
{
    for (int i = 0; i < in_data->element_count; i++) {
        ss << i;
        ss << " : ";
        ss << in_data->h_input[i];
        ss << " [";
        size_t *conflicts_start, *conflicts_end;
        conflictdetect_sort_get_matrix_element(
            in_data->h_input[i], i, in_data->element_count, h_out_elements,
            h_out_indices, &conflicts_start, &conflicts_end);

        for (size_t* i = conflicts_start; i != conflicts_end; i++) {
            ss << *i;
            if (i + 1 != conflicts_end) ss << ", ";
        }
        ss << "]\n";
    }
}

template <typename T>
void input_data_print_from_hashtable(
    input_data<T>* in_data, std::ostream& ss, hashtable<T>* d_ht)
{
    hashtable<T> h_ht;
    hashtable_d2h(d_ht, &h_ht);

    std::vector<size_t> indices{};

    for (int i = 0; i < in_data->element_count; i++) {
        ss << i;
        ss << " : ";
        ss << in_data->h_input[i];
        ss << " [";

        ht_entry<T>* hte = hashtable_get_entry(&h_ht, in_data->h_input[i]);
        cudaSize_t llbi = hte->occurence_list_llbi;
        indices.clear();
        while (true) {
            indices.push_back(h_ht.ll_buffer[llbi].idx);
            llbi = h_ht.ll_buffer[llbi].next_node_llbi;
            if (llbi == HT_LL_BUFFER_SENTINEL) {
                break;
            }
        }
        std::sort(indices.begin(), indices.end());
        auto last = std::lower_bound(indices.begin(), indices.end(), i);
        for (auto i = indices.begin(); i != last; i++) {
            ss << *i;
            if (i + 1 != last) ss << ", ";
        }

        ss << "]\n";
    }

    hashtable_fin_h(&h_ht);
}

template <typename T>
void input_data_init_empty(input_data<T>* in_data, size_t element_count)
{
    in_data->element_count = element_count;
    size_t elements_size = sizeof(T) * element_count;
    in_data->h_input = (T*)malloc(elements_size);
    CUDA_TRY(cudaMalloc(&in_data->d_input, elements_size));
}

template <typename T>
void input_data_init_filled(
    input_data<T>* in_data, size_t element_count, size_t distinct_elements)
{
    input_data_init_empty(in_data, element_count);
    gpu_generate_data<<<10, 32>>>(
        in_data->d_input, element_count, distinct_elements);
    CUDA_TRY(cudaMemcpy(
        in_data->h_input, in_data->d_input, sizeof(T) * element_count,
        cudaMemcpyDeviceToHost));
}

template <typename T> void input_data_fin(input_data<T>* in_data)
{
    free(in_data->h_input);
    CUDA_TRY(cudaFree(in_data->d_input));
}

template <typename T> void test_sort(input_data<T>* in_data)
{
    size_t elements_size = sizeof(T) * in_data->element_count;
    size_t indices_size = sizeof(size_t) * in_data->element_count;
    size_t *d_out_indices, *h_out_indices;
    T *d_out_elements, *h_out_elements;
    CUDA_TRY(cudaMalloc(&d_out_elements, in_data->element_count * sizeof(T)));
    CUDA_TRY(
        cudaMalloc(&d_out_indices, in_data->element_count * sizeof(size_t)));
    conflictdetect_sort(
        in_data->d_input, in_data->element_count, d_out_elements,
        d_out_indices);
    h_out_elements = (T*)malloc(elements_size);
    h_out_indices = (size_t*)malloc(indices_size);
    CUDA_TRY(cudaMemcpy(
        h_out_indices, d_out_indices, indices_size, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(
        h_out_elements, d_out_elements, elements_size, cudaMemcpyDeviceToHost));
    input_data_print_from_sort(
        in_data, std::cout, h_out_elements, h_out_indices);
    CUDA_TRY(cudaFree(d_out_elements));
    CUDA_TRY(cudaFree(d_out_indices));
    free(h_out_elements);
    free(h_out_indices);
}

template <typename T> void test_hashtable(input_data<T>* in_data)
{
    hashtable<uint64_t> ht;
    hashtable_init_d(&ht, in_data->element_count);
    conflictdetect_hashtable(in_data->d_input, in_data->element_count, &ht);
    CUDA_TRY(cudaDeviceSynchronize());
    printf("DONE\n");
    input_data_print_from_hashtable(in_data, std::cout, &ht);
    hashtable_fin_d(&ht);
}

#ifdef CATCH_CONFIG_DISABLE
int main(int argc, char** argv)
{
    input_data<uint64_t> in_data;
    input_data_init_filled(&in_data, 17, 4);
    test_hashtable(&in_data);
    return 0;
}
#else
#define CATCH_CONFIG_MAIN
#endif

#include <catch2/catch.hpp>

template <typename T> class ConflictdetectSortTestFixture {
  public:
    input_data<T> in_data;
    T* h_out_elements;
    T* d_out_elements;
    size_t* h_out_indices;
    size_t* d_out_indices;

  public:
    ConflictdetectSortTestFixture(size_t element_count)
    {
        input_data_init_empty(&in_data, element_count);
        CUDA_TRY(
            cudaMalloc(&d_out_elements, in_data.element_count * sizeof(T)));
        CUDA_TRY(
            cudaMalloc(&d_out_indices, in_data.element_count * sizeof(size_t)));
        h_out_elements = (T*)malloc(in_data.element_count * sizeof(T));
        h_out_indices = (size_t*)malloc(in_data.element_count * sizeof(size_t));
    }

    ~ConflictdetectSortTestFixture()
    {
        CUDA_TRY(cudaFree(d_out_elements));
        CUDA_TRY(cudaFree(d_out_indices));
        free(h_out_elements);
        free(h_out_indices);
        input_data_fin(&in_data);
    }

    void input2gpu(T* input)
    {
        memcpy(in_data.h_input, input, in_data.element_count * sizeof(T));
        CUDA_TRY(cudaMemcpy(
            in_data.d_input, in_data.h_input, in_data.element_count * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    void output2cpu()
    {
        CUDA_TRY(cudaMemcpy(
            h_out_indices, d_out_indices,
            in_data.element_count * sizeof(size_t), cudaMemcpyDeviceToHost));
        CUDA_TRY(cudaMemcpy(
            h_out_elements, d_out_elements, in_data.element_count * sizeof(T),
            cudaMemcpyDeviceToHost));
    }

    std::string output2string()
    {
        std::ostringstream ss{};
        input_data_print_from_sort(&in_data, ss, h_out_elements, h_out_indices);
        return ss.str();
    }
};

TEST_CASE("conflictdetect_sort basic functionality", "[conflictdetect_sort]")
{
    ConflictdetectSortTestFixture<uint32_t> f{10};
    uint32_t input[10] = {2, 1, 1, 1, 2, 2, 3, 7, 4, 0};
    f.input2gpu(input);
    conflictdetect_sort(
        f.in_data.d_input, f.in_data.element_count, f.d_out_elements,
        f.d_out_indices);
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
    ConflictdetectSortTestFixture<uint32_t> f{0};
    uint32_t input[0] = {};
    f.input2gpu(input);
    conflictdetect_sort(
        f.in_data.d_input, f.in_data.element_count, f.d_out_elements,
        f.d_out_indices);
    f.output2cpu();
    REQUIRE(f.output2string() == "");
}
