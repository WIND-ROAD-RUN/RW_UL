project(imet VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_library(${PROJECT_NAME} "include/imet_ModelEngine.h" "src/imet_ModelEngine.cpp")

target_link_libraries(${PROJECT_NAME} PUBLIC  ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME} 
	PUBLIC ${OpenCV_INCLUDE_DIRS}
	PRIVATE ${TensorRT_include})

target_link_libraries(${PROJECT_NAME} PRIVATE ${TensorRT_libs})

# 添加自定义命令，复制 .dll 文件到运行时目录
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TensorRT_dlls}
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# 暴露 .dll 文件路径给依赖模块
set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERFACE_RUNTIME_DLLS "${TensorRT_dlls}"
)

add_subdirectory(test)
