project(ime VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(UTILTY_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/utilty/include )

FIND_SOURCES_AND_HEADERS(UTILTY_SRC_LIST
	${CMAKE_CURRENT_SOURCE_DIR}/utilty/include 
	${CMAKE_CURRENT_SOURCE_DIR}/utilty/src
)

add_subdirectory(imet)
add_subdirectory(imeo)