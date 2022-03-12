# convenient cuda clang support was added in 3.19
cmake_minimum_required(VERSION 3.19)
option(USE_CLANG "use clang instead of nvcc" ON)
set(CLANG_VERSION "" CACHE STRING "specify clang version")

if ("${CLANG_VERSION}" STREQUAL "")
set(CLANG_VERSION_POSTFIX "")
else()
set(CLANG_VERSION_POSTFIX "-${CLANG_VERSION}")
endif()
if(${USE_CLANG})
    set(CMAKE_CUDA_COMPILER "clang++${CLANG_VERSION_POSTFIX}")
    set(CMAKE_CXX_COMPILER "clang++${CLANG_VERSION_POSTFIX}")

    # force c++20
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
else()
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# don't enable gnu extensions
set(CMAKE_CXX_EXTENSIONS OFF)


project(gpu_conflictdetection LANGUAGES CXX CUDA)

set(INCLUDES
    src
    .
    deps
)

set(SOURCES
    src/main.cu
)

add_executable(gpu_conflictdetection ${SOURCES})

add_custom_target(
    cub_fix 
    sed -i "${CMAKE_CURRENT_LIST_DIR}/deps/cub/cub/iterator/tex_ref_input_iterator.cuh" -e "s|\\(^[^/]*\\)words\\[i\\] = tex1Dfetch(ref, (tex_offset \\* TEXTURE_MULTIPLE) + i);|\\1//words[i] = tex1Dfetch(ref, (tex_offset * TEXTURE_MULTIPLE) + i);|"
    VERBATIM
)

set_property(TARGET gpu_conflictdetection PROPERTY CUDA_ARCHITECTURES 75 61)

target_include_directories(gpu_conflictdetection PRIVATE ${INCLUDES})

set_target_properties(gpu_conflictdetection PROPERTIES EXPORT_COMPILE_COMMANDS true)
add_dependencies(gpu_conflictdetection cub_fix)
