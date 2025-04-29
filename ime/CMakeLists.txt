project(ime VERSION 0.1 LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

add_executable(ime main.cpp)

target_link_libraries(ime PRIVATE  ${OpenCV_LIBS})
target_include_directories(ime PRIVATE ${OpenCV_INCLUDE_DIRS})
