project(ime VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

FIND_SOURCES_AND_HEADERS(SRC_LIST
	${CMAKE_CURRENT_SOURCE_DIR}/include 
	${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_library(${PROJECT_NAME}  ${SRC_LIST} ${UTILTY_SRC_LIST})

target_include_directories(${PROJECT_NAME} 
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${UTILTY_INCLUDES}
)

target_link_libraries(${PROJECT_NAME} PUBLIC  ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} 
	PUBLIC imet
	PUBLIC imeo
)

add_subdirectory(test)
