cmake_minimum_required(VERSION 3.27)

project(hoei VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)

FIND_SOURCES_AND_HEADERS(SRC_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(WIN32)
    FIND_SOURCES_AND_HEADERS(SRC_LIST_PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/private/windows/include 
        ${CMAKE_CURRENT_SOURCE_DIR}/private/windows/src
    )
elseif(UNIX AND NOT APPLE)
    FIND_SOURCES_AND_HEADERS(SRC_LIST_PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/private/linux/include 
        ${CMAKE_CURRENT_SOURCE_DIR}/private/linux/src
    )
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

add_library(${PROJECT_NAME} STATIC ${SRC_LIST} ${SRC_LIST_PRIVATE})

target_link_directories(${PROJECT_NAME} PUBLIC ${HWLOC_LIBRARY_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${HWLOC_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${HWLOC_INCLUDE_DIRS})

target_include_directories(${PROJECT_NAME} 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/private/include
)

if(CONFIG_TEST)
    add_subdirectory(./test)
endif()