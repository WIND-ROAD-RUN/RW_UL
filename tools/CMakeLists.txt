add_subdirectory(AutomaticAnnotation)
add_subdirectory(ModelConverter)

if(CONFIG_TEST)
    add_subdirectory(demo)
	add_subdirectory(demoNoQt)
endif()
