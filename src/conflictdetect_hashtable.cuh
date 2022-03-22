#pragma once

#include <cstdint>
#include <type_traits>
#include "cuda_try.cuh"
#include "utils.cuh"

#define HT_OVERSIZE_MULT 1.5
#define HT_LL_BUFFER_SENTINEL UINTPTR_MAX

#if ((UINT32_MAX) == (UINTPTR_MAX))
typedef unsigned int cudaSize_t;
#else
typedef unsigned long long cudaSize_t;
#endif

typedef unsigned long long cudaUInt64_t;

template <typename T>
struct ht_type_map {
};

template <>
struct ht_type_map<uint64_t> {
    using type = unsigned long long;
    static constexpr type EMPTY_ELEMENT = UINT64_MAX;
};
template <>
struct ht_type_map<uint32_t> {
    using type = unsigned int;
    static constexpr type EMPTY_ELEMENT = UINT32_MAX;
};

struct ll_node {
    size_t idx;
    cudaSize_t next_node_llbi; // llbi = ll_buffer_index
};

template <typename T>
struct ht_entry {
    T element;
    cudaSize_t occurence_list_llbi;
};

template <typename T>
struct hashtable {
    ll_node* ll_buffer;
    size_t capacity; // max entry count
    ht_entry<T>* table;
    size_t element_count;
};

template <typename T>
static inline void hashtable_init_d(hashtable<T>* ht, size_t element_count)
{
    ht->capacity = ceil2pow2(element_count * HT_OVERSIZE_MULT);
    CUDA_TRY(cudaMalloc(&ht->table, sizeof(ht_entry<T>) * ht->capacity));
    CUDA_TRY(cudaMemset(ht->table, 0xFF, sizeof(ht_entry<T>) * ht->capacity));
    CUDA_TRY(cudaMalloc(&ht->ll_buffer, element_count * sizeof(ll_node)));
    ht->element_count = element_count;
    // ht->table[0] for the ht_type_map<T>::EMPTY_ELEMENT automatically gets its
    // right elem assigned
}

template <typename T>
static inline void hashtable_fin_d(hashtable<T>* ht)
{
    CUDA_TRY(cudaFree(ht->ll_buffer));
    CUDA_TRY(cudaFree(ht->table));
}

template <typename T>
static inline void hashtable_fin_h(hashtable<T>* ht)
{
    free(ht->ll_buffer);
    free(ht->table);
}

template <typename T>
void hashtable_d2h(hashtable<T>* h_ht, hashtable<T>* d_ht)
{
    *h_ht = *d_ht;
    h_ht->ll_buffer = (ll_node*)malloc(h_ht->element_count * sizeof(ll_node));
    h_ht->table = (ht_entry<T>*)malloc(h_ht->capacity * sizeof(ht_entry<T>));
    CUDA_TRY(cudaMemcpy(
        h_ht->ll_buffer,
        d_ht->ll_buffer,
        h_ht->element_count * sizeof(ll_node),
        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(
        h_ht->table,
        d_ht->table,
        h_ht->capacity * sizeof(ht_entry<T>),
        cudaMemcpyDeviceToHost));
}

template <typename T>
ht_entry<T>* hashtable_get_entry(hashtable<T>* ht, T element)
{
    if (element == ht_type_map<T>::EMPTY_ELEMENT) {
        // insert into special list
        return &ht->table[0];
    }
    size_t hash = element & (ht->capacity - 1);
    if (hash == 0) hash++; // skip ht_type_map<T>::EMPTY_ELEMENT
    ht_entry<T>* hte = &ht->table[hash];
    while (true) {
        if (hte->element == element) {
            break;
        }
        if (hte->element == ht_type_map<T>::EMPTY_ELEMENT) {
            return NULL;
        }
        if (hte != &ht->table[ht->capacity - 1]) {
            hte++;
        }
        else {
            // restart from the beginning of the hashtable
            hte = ht->table + 1;
        }
    }
    return hte;
}

template <typename T>
__device__ void hashtable_insert(hashtable<T>* ht, T element, size_t index)
{
    ll_node* new_ll_node = &ht->ll_buffer[index];
    typedef typename ht_type_map<T>::type T_CUDA;
    cudaSize_t* eol;

    if (element == ht_type_map<T>::EMPTY_ELEMENT) {
        // insert into special list
        eol = &ht->table[0].occurence_list_llbi;
    }
    else {
        size_t hash = (size_t)element & (ht->capacity - 1);
        if (hash == 0) hash++; // skip ht_type_map<T>::EMPTY_ELEMENT
        ht_entry<T>* hte = &ht->table[hash];
        while (true) {
            if (hte->element == element) {
                break;
            }
            if (hte->element == ht_type_map<T>::EMPTY_ELEMENT) {
                T found = (T)atomicCAS(
                    (T_CUDA*)&hte->element,
                    ht_type_map<T>::EMPTY_ELEMENT,
                    (T_CUDA)element);
                if (found == ht_type_map<T>::EMPTY_ELEMENT ||
                    found == element) {
                    // got the slot, or someone else did it for us
                    break;
                }
            }
            if (hte != &ht->table[ht->capacity - 1]) {
                hte++;
            }
            else {
                // restart from the beginning of the hashtable
                hte = ht->table + 1;
            }
        }
        eol = &hte->occurence_list_llbi;
    }
    // finally append to linked list
    new_ll_node->idx = index;
    new_ll_node->next_node_llbi = *eol;
    cudaSize_t new_ll_node_llbi = (cudaSize_t)(new_ll_node - ht->ll_buffer);
    while (true) {
        cudaSize_t found = (cudaSize_t)atomicCAS(
            (cudaSize_t*)eol,
            (cudaSize_t)new_ll_node->next_node_llbi,
            new_ll_node_llbi);
        if (found == new_ll_node->next_node_llbi) {
            break; // success
        }
        new_ll_node->next_node_llbi = found;
    }
}

template <typename T>
__global__ void kernel_fill_ht(T* input, hashtable<T> ht)
{
    size_t stride = blockDim.x * gridDim.x;
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tid; i < ht.element_count; i += stride) {
        hashtable_insert(&ht, input[i], i);
    }
}

template <typename T>
void conflictdetect_hashtable(T* d_input, hashtable<T>* ht)
{
    kernel_fill_ht<<<1024, 256>>>(d_input, *ht);
}
