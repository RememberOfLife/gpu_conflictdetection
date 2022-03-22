#pragma once

#include <assert.h>
#include <stdio.h>

#include "cuda_try.cuh"
#include "utils.cuh"
// macro for timing gpu operations
// sadly, you can't use code blocks as the arguments for these
// i don't know why, and i don't want to know why

#define CUDA_TIME_FORCE_ENABLED(ce_start, ce_stop, stream, time, ...)          \
    do {                                                                       \
        CUDA_TRY(cudaStreamSynchronize((stream)));                             \
        CUDA_TRY(cudaEventRecord((ce_start)));                                 \
        {                                                                      \
            __VA_ARGS__;                                                       \
        }                                                                      \
        CUDA_TRY(cudaEventRecord((ce_stop)));                                  \
        CUDA_TRY(cudaEventSynchronize((ce_stop)));                             \
        CUDA_TRY(cudaEventElapsedTime((time), (ce_start), (ce_stop)));         \
    } while (0)

#ifdef DISABLE_CUDA_TIME

#define CUDA_TIME(ce_start, ce_stop, stream, time, ...)                        \
    do {                                                                       \
        CUDA_TRY(cudaStreamSynchronize((stream)));                             \
        {                                                                      \
            __VA_ARGS__;                                                       \
        }                                                                      \
        *(time) = 0;                                                           \
        UNUSED(ce_start);                                                      \
        UNUSED(ce_stop);                                                       \
    } while (0)

#else

#define CUDA_TIME(ce_start, ce_stop, stream, time, ...)                        \
    CUDA_TIME_FORCE_ENABLED(ce_start, ce_stop, stream, time, __VA_ARGS__)

#endif

#define CUDA_TIME_PRINT(fmt_str, ...)                                          \
    do {                                                                       \
        cudaEvent_t _cuda_time_print_start, _cuda_time_print_stop;             \
        float _cuda_time_print_elapsed_time;                                   \
        CUDA_TRY(cudaEventCreate((&_cuda_time_print_start)));                  \
        CUDA_TRY(cudaEventCreate((&_cuda_time_print_stop)));                   \
                                                                               \
        CUDA_TRY(cudaEventRecord((_cuda_time_print_start)));                   \
        {                                                                      \
            __VA_ARGS__;                                                       \
        }                                                                      \
        CUDA_TRY(cudaEventRecord((_cuda_time_print_stop)));                    \
                                                                               \
        CUDA_TRY(cudaEventSynchronize((_cuda_time_print_stop)));               \
        CUDA_TRY(cudaEventElapsedTime(                                         \
            (&_cuda_time_print_elapsed_time),                                  \
            (_cuda_time_print_start),                                          \
            (_cuda_time_print_stop)));                                         \
        CUDA_TRY(cudaEventDestroy((_cuda_time_print_start)));                  \
        CUDA_TRY(cudaEventDestroy((_cuda_time_print_stop)));                   \
        printf(fmt_str, _cuda_time_print_elapsed_time);                        \
        fflush(stdout);                                                        \
    } while (0)
