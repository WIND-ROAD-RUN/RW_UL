set(UTILTY_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/utilty/include )

FIND_SOURCES_AND_HEADERS(UTILTY_SRC_LIST
	${CMAKE_CURRENT_SOURCE_DIR}/utilty/include 
	${CMAKE_CURRENT_SOURCE_DIR}/utilty/src
)

add_subdirectory(imet)
add_subdirectory(imeo)
add_subdirectory(ime)