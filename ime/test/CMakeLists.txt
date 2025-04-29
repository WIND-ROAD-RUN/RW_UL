project(ime_test VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ime)

