# Halcon
if(WIN32)
    if(INSTALL_HALCON)
		find_all_libs(${HALCON_ROOT}/lib/x64-win64 Halcon_libs)
        set(Halcon_include ${HALCON_ROOT}/include)
    endif()
else()
endif()

# TensorRT
set(TensorRTPath ${CMAKE_CURRENT_SOURCE_DIR}/ThirdLibrary/TensorRT-10.9.0.34)
set(TensorRT_include ${TensorRTPath}/include)
find_all_libs(${TensorRTPath}/lib TensorRT_libs)
find_all_dlls(${TensorRTPath}/lib TensorRT_dlls)
find_all_exes(${TensorRTPath}/bin TensorRT_exes)

# OnnxRuntime
set(OnnxRuntimePath ${CMAKE_CURRENT_SOURCE_DIR}/ThirdLibrary/onnxruntime-win-x64-gpu-1.21.0)
set(OnnxRuntime_include ${OnnxRuntimePath}/include)
find_all_libs(${OnnxRuntimePath}/lib OnnxRuntime_libs)
find_all_dlls(${OnnxRuntimePath}/lib OnnxRuntime_dlls)

# ZMotion
set(ZMotionRutimePath ${CMAKE_CURRENT_SOURCE_DIR}/ThirdLibrary/ZMotion)
set(ZMotionRutime_include ${ZMotionRutimePath}/include)
find_all_libs(${ZMotionRutimePath}/lib ZMotionRutime_libs)
find_all_dlls(${ZMotionRutimePath}/dll ZMotionRutime_dlls)