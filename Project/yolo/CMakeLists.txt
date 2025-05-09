cmake_minimum_required(VERSION 3.12)
project(yolov11-tensorrt)

find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
add_definitions(-DAPI_EXPORTS)

# Add source files
set(SOURCES
    main.cpp
    src/YOLOv11.cpp
)

# Add headers
set(HEADERS
    src/YOLOv11.h
    src/common.h
)

# Find and include OpenCV
set(OpenCV_DIR C:/Users/zfkj/Desktop/opencv/build)

# Set TensorRT path if not set in environment variables
set(TENSORRT_DIR C:/TensorRT-10.9.0.34)
set(ONNXRUNTIME_DIR C:/Users/zfkj/Desktop/onnxruntime-win-x64-gpu-1.21.0)

# Include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include TensorRT
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)

# Include onnxruntime
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)
set(ONNXRUNTIME_LIBS onnxruntime onnxruntime_providers_cuda onnxruntime_providers_shared)

# Create an executable
cuda_add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
    ${ONNXRUNTIME_LIBS}
)