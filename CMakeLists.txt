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
    include(./cmake/iniEnvironmet.cmake)

    set(TensorRTPath ${CMAKE_CURRENT_SOURCE_DIR}/ThirdLibrary/TensorRT-10.9.0.34)
    set(TensorRT_include ${TensorRTPath}/include)

    find_all_libs(${TensorRTPath}/lib TensorRT_libs)
    find_all_dlls(${TensorRTPath}/lib TensorRT_dlls)
    find_all_exes(${TensorRTPath}/bin TensorRT_exes)

    set(OnnxRuntimePath ${CMAKE_CURRENT_SOURCE_DIR}/ThirdLibrary/onnxruntime-win-x64-gpu-1.21.0)
    set(OnnxRuntime_include ${OnnxRuntimePath}/include)

    find_all_libs(${OnnxRuntimePath}/lib OnnxRuntime_libs)
    find_all_dlls(${OnnxRuntimePath}/lib OnnxRuntime_dlls)

    set(ZMotionRutimePath ${CMAKE_CURRENT_SOURCE_DIR}/ThirdLibrary/ZMotion)
    set(ZMotionRutime_include ${ZMotionRutimePath}/include)

    find_all_libs(${ZMotionRutimePath}/lib ZMotionRutime_libs)
    find_all_dlls(${ZMotionRutimePath}/dll ZMotionRutime_dlls)

    set(INSTALL_PATH ${CMAKE_SOURCE_DIR}/install/${CMAKE_BUILD_TYPE})

    enable_testing()
endif()

project(RW_UL VERSION 0.1 LANGUAGES CXX CUDA)

if(MSVC)
    # 对于 MSVC 编译器，启用 UTF-8 编码
    add_compile_options("/utf-8")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # 对于 GCC 或 Clang 编译器，设置源文件编码为 UTF-8
    add_compile_options("-finput-charset=UTF-8")
endif()

find_package(gtest REQUIRED CONFIG)

add_subdirectory(RW_UL)
#add_subdirectory(Project)
add_subdirectory(tools)