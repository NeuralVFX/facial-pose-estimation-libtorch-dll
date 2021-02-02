cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(facial-pose-estimation-libtorch)

set(Torch_DIR C:/libtorch1.7n/share/cmake/Torch)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set( OpenCV_FOUND 1 )
set( OpenCV_DIR C:/OpenCV/build)
set( OpenCV_INC_DIR C:/OpenCV/build/install/include)
set (OpenCV_LIB_DIR  C:/OpenCV/build/lib/Release)
include_directories( ${OpenCV_INC_DIR} )

add_library(facial-pose-estimation-libtorch SHARED export_wrapper.cpp pose_estimate.cpp pose_estimate.h )

target_link_libraries(facial-pose-estimation-libtorch ${TORCH_LIBRARIES} 
${OpenCV_LIBS}
${OpenCV_LIB_DIR}/opencv_core412.lib
${OpenCV_LIB_DIR}/opencv_highgui412.lib
${OpenCV_LIB_DIR}/opencv_videoio412.lib
${OpenCV_LIB_DIR}/opencv_calib3d412.lib
${OpenCV_LIB_DIR}/opencv_imgcodecs412.lib
${OpenCV_LIB_DIR}/opencv_objdetect412.lib
${OpenCV_LIB_DIR}/opencv_dnn412.lib
${OpenCV_LIB_DIR}/opencv_imgproc412.lib)

set_property(TARGET facial-pose-estimation-libtorch PROPERTY CXX_STANDARD 14)

if (MSVC)
	file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	add_custom_command(TARGET facial-pose-estimation-libtorch
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${TORCH_DLLS}
		$<TARGET_FILE_DIR:x64)
endif (MSVC)