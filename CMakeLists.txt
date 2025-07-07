cmake_minimum_required(VERSION 3.10)
# 检查当前 CMakeLists 是否为根 CMakeLists
#如果是根 CMakeLists，则设置不构建测试
set(CONFIG_TEST false)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	set(CONFIG_TEST true)
	
	set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

	find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # 如果是根 CMakeLists，则执行以下逻辑
    if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    endif()

    set(SerializedCodeGeneratorCmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SerializedCodeGenerator)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SerializedCodeGenerator/cmake")
    include(SerializedCodeGenerator)

    include(./cmake/func.cmake)
    include(./cmake/EnvironmetIni.cmake)
	include(./cmake/PackageFind.cmake)

    set(INSTALL_PATH ${CMAKE_SOURCE_DIR}/install/${CMAKE_BUILD_TYPE})
endif()

project(RW_UL VERSION 0.1 LANGUAGES CXX CUDA)
	include(./cmake/PackageIni.cmake)
	include(./cmake/ComplierIni.cmake)

find_package(gtest REQUIRED CONFIG)

add_subdirectory(RW_UL)
add_subdirectory(tools)

 enable_testing()