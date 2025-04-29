cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/workPlace/tools/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

enable_testing()

project(RW_UL VERSION 0.1 LANGUAGES CXX CUDA)

find_package(gtest REQUIRED CONFIG)

add_subdirectory(ime)