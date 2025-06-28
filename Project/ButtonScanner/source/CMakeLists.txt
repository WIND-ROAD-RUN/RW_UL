cmake_minimum_required(VERSION 3.27)

project(ButtonScanner VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets Concurrent)

file(GLOB UI_FILES "${CMAKE_CURRENT_SOURCE_DIR}/ui/*.ui")
qt_wrap_ui(GENERATED_UI_HEADERS ${UI_FILES})

file(GLOB QRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/qrc/*.qrc")

FIND_SOURCES_AND_HEADERS(INCLUDE_LIST
	${CMAKE_CURRENT_SOURCE_DIR}/include 
	${CMAKE_CURRENT_SOURCE_DIR}/include/data
	${CMAKE_CURRENT_SOURCE_DIR}/include/DetachThread
	${CMAKE_CURRENT_SOURCE_DIR}/include/func
	${CMAKE_CURRENT_SOURCE_DIR}/include/ui
)

FIND_SOURCES_AND_HEADERS(SRC_LIST
	${CMAKE_CURRENT_SOURCE_DIR}/src
	${CMAKE_CURRENT_SOURCE_DIR}/src/data
	${CMAKE_CURRENT_SOURCE_DIR}/src/DetachThread
	${CMAKE_CURRENT_SOURCE_DIR}/src/func
	${CMAKE_CURRENT_SOURCE_DIR}/src/ui
)

qt_add_executable(${PROJECT_NAME} 
	MANUAL_FINALIZATION
	${INCLUDE_LIST}
	${SRC_LIST}
	${GENERATED_UI_HEADERS}
	${QRC_FILES}
)


target_include_directories(${PROJECT_NAME} 
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/data
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/DetachThread
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/func
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/ui
	PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets Qt6::Concurrent)
target_link_libraries(${PROJECT_NAME} 
	PRIVATE ${OpenCV_LIBS}
)

target_link_libraries(${PROJECT_NAME} 
	PRIVATE hoec
	PRIVATE scc
	PRIVATE rqwm
	PRIVATE rqwime
	PRIVATE dsl
	PRIVATE hoec
	PRIVATE oso
	PRIVATE ime
	PRIVATE utilty
	PRIVATE cdm
	PRIVATE rqw
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${QT_PATH}/bin/windeployqt.exe --no-compiler-runtime --no-opengl-sw $<TARGET_FILE:${PROJECT_NAME}>
    COMMENT "Running windeployqt to package Qt dependencies"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TensorRT_dlls} ${TensorRT_exes} ${OnnxRuntime_dlls}
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${ZMotionRutime_dlls}
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

#获取python脚本
file(GLOB PYTHON_FILES "${CMAKE_CURRENT_SOURCE_DIR}/py/*.py")
file(GLOB YAML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/py/*.yaml")
file(GLOB BAT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/bat/*.bat")
#拷贝python脚本
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${PYTHON_FILES} ${BAT_FILES} ${YAML_FILES}
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# 安装可执行文件
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${INSTALL_PATH}/${PROJECT_NAME} # 可执行文件安装到 bin 目录
)

# 安装依赖的 DLL 文件
install(FILES ${TensorRT_dlls} ${ZMotionRutime_dlls} ${OnnxRuntime_dlls}
    DESTINATION ${INSTALL_PATH}/${PROJECT_NAME} 
)

string(REPLACE "\\" "/" QT_PATH_FIXED ${QT_PATH})
install(CODE "
    execute_process(
        COMMAND ${QT_PATH_FIXED}/bin/windeployqt.exe --no-compiler-runtime --no-opengl-sw ${INSTALL_PATH}/${PROJECT_NAME}/${PROJECT_NAME}.exe
        WORKING_DIRECTORY ${INSTALL_PATH}/${PROJECT_NAME}
    )
    message(\"Running windeployqt during install phase\")
")

file(GLOB VERSION_TXT "${CMAKE_CURRENT_SOURCE_DIR}/sourceFiles/*.txt")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${VERSION_TXT}
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)








