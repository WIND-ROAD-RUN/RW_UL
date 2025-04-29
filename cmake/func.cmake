# 查找所有 .lib 文件
function(find_all_libs dir output_var)
    file(GLOB_RECURSE libs "${dir}/*.lib")
    set(${output_var} ${libs} PARENT_SCOPE)
endfunction()

# 查找所有 .dll 文件
function(find_all_dlls dir output_var)
    file(GLOB_RECURSE dlls "${dir}/*.dll")
    set(${output_var} ${dlls} PARENT_SCOPE)
endfunction()

# Utility function to find all sources and headers in a list of directories
function(find_sources_and_headers result_var)
    set(sources_and_headers)
    foreach(path IN LISTS ARGN)
        file(GLOB_RECURSE headers "${path}/*.h" "${path}/*.hpp")
        file(GLOB_RECURSE sources "${path}/*.cpp")
        list(APPEND sources_and_headers ${headers} ${sources})
    endforeach()
    set(${result_var} ${sources_and_headers} PARENT_SCOPE)
endfunction()

# Utility function to find all sources and headers in a list of directories and group them
function(group_sources_and_headers group_name)
    set(SOURCESFILES)
    set(HEADERSFILES)
    foreach(path IN LISTS ARGN)
        file(GLOB_RECURSE headers "${path}/*.h")
        file(GLOB_RECURSE sources "${path}/*.cpp")
        list(APPEND HEADERSFILES ${headers})
        list(APPEND SOURCESFILES ${sources})
    endforeach()
    
    # Create source group
    source_group("${group_name}/include" FILES ${HEADERSFILES})
    source_group("${group_name}/src" FILES ${SOURCESFILES})
endfunction()

# Utility function to find all sources and headers in a list of directories and group them
function(group_interface_sources_and_headers)
    set(SOURCESFILES)
    set(HEADERSFILES)
    foreach(path IN LISTS ARGN)
        file(GLOB_RECURSE headers "${path}/*.h")
        file(GLOB_RECURSE sources "${path}/*.cpp")
        list(APPEND HEADERSFILES ${headers})
        list(APPEND SOURCESFILES ${sources})
    endforeach()
    
    # Create source group
    source_group("Header Files" FILES ${HEADERSFILES})
    source_group("Source Files" FILES ${SOURCESFILES})
endfunction()

# Utility function to check if a specified path exists in a specified system environment variable
function(check_path_in_system_env_var env_var_name search_path result_var)
    message("Check if the system environment variable ${env_var_name} contains ${search_path}" )
    string(TOUPPER "${env_var_name}" env_var_name_upper)
    if (DEFINED ENV{${env_var_name_upper}})
        # Replace backslashes with forward slashes in the environment variable value and search path
        string(REPLACE "\\" "/" env_var_value "$ENV{${env_var_name_upper}}")
        string(REPLACE "\\" "/" search_path_normalized "${search_path}")
        
        string(FIND "${env_var_value}" "${search_path_normalized}" path_index)
        if (path_index GREATER -1)
            set(${result_var} TRUE PARENT_SCOPE)
            message("The system environment variable ${env_var_name} contains ${search_path}" )
        else()
            set(${result_var} FALSE PARENT_SCOPE)
            Message ("The system environment variable ${env_var_name} does not contain ${search_path}")
        endif()
    else()
        set(${result_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

