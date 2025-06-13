# 定义文本文件路径
set(PROJECTS_FILE "${CMAKE_SOURCE_DIR}/projects_to_build.txt")

# 默认构建所有项目
set(ALL_PROJECTS "ButtonScanner;SmartCroppingOfBags;ZipperScanner")
set(PROJECTS_TO_BUILD "${ALL_PROJECTS}")

# 检查文件是否存在
if(EXISTS "${PROJECTS_FILE}")
    # 读取文件内容
    file(READ "${PROJECTS_FILE}" FILE_CONTENT)
    
    # 去除文件内容中的空格和换行符
    string(STRIP "${FILE_CONTENT}" FILE_CONTENT)
    
    # 如果文件内容非空，则使用文件中的项目名
    if(NOT "${FILE_CONTENT}" STREQUAL "")
        set(PROJECTS_TO_BUILD "${FILE_CONTENT}")
    endif()
endif()

# 分割项目名并逐个构建
foreach(PROJECT ${PROJECTS_TO_BUILD})
    add_subdirectory(${PROJECT})
endforeach()