cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/workPlace/tools/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

include(./cmake/func.cmake)

set(TensorRTPath ${CMAKE_CURRENT_SOURCE_DIR}/ThirdLibrary/TensorRT-10.9.0.34)
set(TensorRT_include ${TensorRTPath}/include)

find_all_libs(${TensorRTPath}/lib TensorRT_libs)
find_all_dlls(${TensorRTPath}/lib TensorRT_dlls)

enable_testing()

project(RW_UL VERSION 0.1 LANGUAGES CXX CUDA)

find_package(gtest REQUIRED CONFIG)

add_subdirectory(ime)
add_subdirectory(oso)
add_subdirectory(hoe)