cmake_minimum_required(VERSION 3.27)

project(hoe_utilty VERSION 0.1 LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(HWLOC REQUIRED hwloc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FIND_SOURCES_AND_HEADERS(SRC_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

group_interface_sources_and_headers(
	${CMAKE_CURRENT_SOURCE_DIR}/include 
	${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_library(${PROJECT_NAME} STATIC ${SRC_LIST})
target_include_directories(${PROJECT_NAME} 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(${HWLOC_LIBRARY_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${HWLOC_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${HWLOC_INCLUDE_DIRS})

add_subdirectory(./test)








