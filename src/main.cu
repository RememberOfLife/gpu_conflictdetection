#include <cstdint>
#include <memory>
#include <stdio.h>
#include <string>
#include <sstream>
#include <cstdlib>

#include "cuda_try.cuh"
#include "cuda_time.cuh"
#include "data_generator.cuh"
#include "conflictdetect_sort.cuh"
#include "conflictdetect_hashtable.cuh"

template <typename T>
struct input_data {
    size_t element_count;
    T* h_input;
    T* d_input;
};

template <typename T>
void input_data_print_from_sort(
    input_data<T>* in_data, std::ostream& ss, sort_buffers<T>* d_sb)
{
    sort_buffers<T> h_sb;
    sort_buffers_d2h(&h_sb, d_sb);
    for (int i = 0; i < in_data->element_count; i++) {
        ss << i;
        ss << " : ";
        ss << in_data->h_input[i];
        ss << " [";
        size_t *conflicts_start, *conflicts_end;
        conflictdetect_sort_get_matrix_element(
            in_data->h_input[i], i, &h_sb, &conflicts_start, &conflicts_end);

        for (size_t* i = conflicts_start; i != conflicts_end; i++) {
            ss << *i;
            if (i + 1 != conflicts_end) ss << ", ";
        }
        ss << "]\n";
    }
    sort_buffers_fin_h(&h_sb);
}

template <typename T>
void input_data_print_from_hashtable(
    input_data<T>* in_data, std::ostream& ss, hashtable<T>* d_ht)
{
    hashtable<T> h_ht;
    hashtable_d2h(&h_ht, d_ht);

    std::vector<size_t> indices{};

    for (int i = 0; i < in_data->element_count; i++) {
        ss << i;
        ss << " : ";
        ss << in_data->h_input[i];
        ss << " [";

        ht_entry<T>* hte = hashtable_get_entry(&h_ht, in_data->h_input[i]);
        assert(hte);
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
        in_data->h_input,
        in_data->d_input,
        sizeof(T) * element_count,
        cudaMemcpyDeviceToHost));
}

template <typename T>
void input_data_fill(input_data<T>* in_data, T* input)
{
    memcpy(in_data->h_input, input, in_data->element_count * sizeof(T));
    cudaMemcpy(
        in_data->d_input,
        in_data->h_input,
        in_data->element_count * sizeof(T),
        cudaMemcpyHostToDevice);
}

template <typename T>
void input_data_fin(input_data<T>* in_data)
{
    free(in_data->h_input);
    CUDA_TRY(cudaFree(in_data->d_input));
}

template <typename T>
void test_sort(input_data<T>* in_data)
{
    size_t elements_size = sizeof(T) * in_data->element_count;
    size_t indices_size = sizeof(size_t) * in_data->element_count;
    size_t *d_out_indices, *h_out_indices;
    T *d_out_elements, *h_out_elements;
    CUDA_TRY(cudaMalloc(&d_out_elements, in_data->element_count * sizeof(T)));
    CUDA_TRY(
        cudaMalloc(&d_out_indices, in_data->element_count * sizeof(size_t)));
    conflictdetect_sort(
        in_data->d_input,
        in_data->element_count,
        d_out_elements,
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

template <typename T>
void test_hashtable(input_data<T>* in_data)
{
    hashtable<uint64_t> d_ht;
    hashtable_init_d(&d_ht, in_data->element_count);
    conflictdetect_hashtable(in_data->d_input, in_data->element_count, &d_ht);
    input_data_print_from_hashtable(in_data, std::cout, &d_ht);
    hashtable_fin_d(&d_ht);
}

template <typename T>
void benchmark(const char* context)
{
    input_data<T> in_data;
    input_data_init_filled(&in_data, 1 << 27, 1 << 26);

    sort_buffers<T> d_sb;
    sort_buffers_init_d(&d_sb, in_data.element_count);
    fputs(context, stdout);
    CUDA_TIME_PRINT(
        " sort: %f ms\n", conflictdetect_sort(in_data.d_input, &d_sb));
    sort_buffers_fin_d(&d_sb);

    hashtable<T> d_ht;
    hashtable_init_d(&d_ht, in_data.element_count);

    fputs(context, stdout);
    CUDA_TIME_PRINT(
        " hashtable: %f ms\n",
        conflictdetect_hashtable(in_data.d_input, &d_ht));

    hashtable_fin_d(&d_ht);
}
#ifdef CATCH_CONFIG_DISABLE
int main(int argc, char** argv)
{
    benchmark<uint32_t>("uint32");
    benchmark<uint64_t>("uint64");
    return 0;
}
#else
#define CATCH_CONFIG_MAIN
#endif

#include <catch2/catch.hpp>

TEST_CASE("conflictdetect_sort basic functionality", "[conflictdetect_sort]")
{
    uint32_t input[10] = {2, 1, 1, 1, 2, 2, 3, 7, 4, 0};

    input_data<uint32_t> in_data;
    input_data_init_empty(&in_data, 10);
    input_data_fill(&in_data, input);

    sort_buffers<uint32_t> d_sb;
    sort_buffers_init_d(&d_sb, in_data.element_count);

    conflictdetect_sort(in_data.d_input, &d_sb);

    std::ostringstream ss;
    input_data_print_from_sort(&in_data, ss, &d_sb);
    // clang-format off
    REQUIRE(
        ss.str() == (
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

    input_data_fin(&in_data);
    sort_buffers_fin_d(&d_sb);
}

TEST_CASE(
    "conflictdetect_sort works on empty data", "[conflictdetect_sort][empty]")
{
    uint32_t input[0] = {};

    input_data<uint32_t> in_data;
    input_data_init_empty(&in_data, 0);
    input_data_fill(&in_data, input);

    sort_buffers<uint32_t> d_sb;
    sort_buffers_init_d(&d_sb, in_data.element_count);

    conflictdetect_sort(in_data.d_input, &d_sb);

    std::ostringstream ss;
    input_data_print_from_sort(&in_data, ss, &d_sb);
    REQUIRE(ss.str() == "");

    input_data_fin(&in_data);
    sort_buffers_fin_d(&d_sb);
}

TEST_CASE(
    "conflictdetect_hashtable basic functionality",
    "[conflictdetect_hashtable]")
{
    input_data<uint32_t> in_data;
    input_data_init_empty(&in_data, 10);
    uint32_t input[10] = {2, 1, 1, 1, 2, 2, 3, 7, 4, 0};
    input_data_fill(&in_data, input);

    hashtable<uint32_t> d_ht;
    hashtable_init_d(&d_ht, in_data.element_count);

    conflictdetect_hashtable(in_data.d_input, &d_ht);

    std::ostringstream ss;
    input_data_print_from_hashtable(&in_data, ss, &d_ht);
    // clang-format off
    REQUIRE(ss.str() == (
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
    ));
    // clang-format on
    hashtable_fin_d(&d_ht);
    input_data_fin(&in_data);
}

TEST_CASE(
    "conflictdetect_hashtable works on empty data",
    "[conflictdetect_hashtable][empty]")
{
    input_data<uint32_t> in_data;
    input_data_init_empty(&in_data, 0);
    uint32_t input[0] = {};
    input_data_fill(&in_data, input);

    hashtable<uint32_t> d_ht;
    hashtable_init_d(&d_ht, in_data.element_count);

    conflictdetect_hashtable(in_data.d_input, &d_ht);

    std::ostringstream ss;
    input_data_print_from_hashtable(&in_data, ss, &d_ht);

    REQUIRE(ss.str() == "");

    hashtable_fin_d(&d_ht);
    input_data_fin(&in_data);
}
