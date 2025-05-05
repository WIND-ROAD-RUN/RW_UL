project(imeo VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(${PROJECT_NAME} "include/imeo_ModelEngine_yolov11_obb.hpp" 
"src/imeo_ModelEngine_yolov11_obb.cpp" ${UTILTY_SRC_LIST})

target_include_directories(${PROJECT_NAME} 
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${UTILTY_INCLUDES}
)

target_link_libraries(${PROJECT_NAME} PUBLIC  ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME} 
	PUBLIC ${OpenCV_INCLUDE_DIRS}
	PUBLIC ${OnnxRuntime_include}
    PUBLIC ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart               # CUDA Runtime Library
    CUDA::cublas               # cuBLAS Library
    CUDA::cufft                # cuFFT Library
    CUDA::curand               # cuRAND Library
    CUDA::cusolver             # cuSolver Library
    CUDA::cusparse             # cuSPARSE Library
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OnnxRuntime_libs})

add_subdirectory(test)
