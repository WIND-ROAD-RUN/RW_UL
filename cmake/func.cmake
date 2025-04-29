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