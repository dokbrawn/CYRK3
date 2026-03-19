cmake_minimum_required(VERSION 3.20)
project(trpo_syrk C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)


find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    add_compile_definitions(HAS_OPENMP)
endif()


set(OPENBLAS_ROOT "C:/OpenBLAS")
include_directories(${OPENBLAS_ROOT}/include)
link_directories(${OPENBLAS_ROOT}/lib)


add_library(syrk_lib STATIC src/syrk.c)

if(MSVC)
target_compile_options(syrk_lib PRIVATE /O2 /W4 /openmp /openmp:experimental /arch:AVX2)
    target_link_libraries(syrk_lib PUBLIC OpenMP::OpenMP_C)
else()
    target_compile_options(syrk_lib PRIVATE -O3 -Wall -mavx2 -fopenmp)
endif()


add_executable(test_syrk src/test_syrk.c)
target_link_libraries(test_syrk PRIVATE syrk_lib)


add_executable(benchmark_syrk src/benchmark_syrk.c)
target_link_libraries(benchmark_syrk PRIVATE syrk_lib libopenblas)

if(MSVC)
    target_compile_options(test_syrk PRIVATE /O2 /W4)
    target_compile_options(benchmark_syrk PRIVATE /O2 /W4 /openmp)
endif()