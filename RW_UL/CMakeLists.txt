add_subdirectory(ime)
add_subdirectory(oso)
add_subdirectory(hoe)
add_subdirectory(dsl)
add_subdirectory(utilty)
add_subdirectory(rqw)
add_subdirectory(scc)
add_subdirectory(cla)

# 检查是否为Windows操作系统
if(WIN32)
    if(INSTALL_HALCON)
		message("-------------${Halcon_libs}")
		add_subdirectory(halcon)
    endif()
else()

endif()

