# Validation-only bridge; the production FlagCV library does not link OpenCV.
cmake_minimum_required(VERSION 3.23)
project(flagcv_opencv_cuda LANGUAGES CXX)
find_package(OpenCV REQUIRED COMPONENTS core cudaarithm)
find_package(CUDAToolkit REQUIRED)
add_library(flagcv_opencv_cuda SHARED bridge.cpp)
target_compile_features(flagcv_opencv_cuda PRIVATE cxx_std_17)
target_include_directories(flagcv_opencv_cuda PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(flagcv_opencv_cuda PRIVATE ${OpenCV_LIBS} CUDA::cudart)
set_target_properties(flagcv_opencv_cuda PROPERTIES
  BUILD_RPATH "${OpenCV_LIB_DIR}" INSTALL_RPATH_USE_LINK_PATH TRUE)
