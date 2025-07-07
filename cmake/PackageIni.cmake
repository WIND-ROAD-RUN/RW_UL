	find_package(PkgConfig REQUIRED)
	pkg_check_modules(HWLOC REQUIRED hwloc)
	link_directories( ${HWLOC_LIBRARY_DIRS})