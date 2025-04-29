project(ime VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_library(${PROJECT_NAME} ./include/ime.h ./src/ime.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC  ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

add_subdirectory(test)
