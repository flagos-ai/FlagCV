# NVIDIA plugin: a dlopen'd module linking only the CUDA driver API. The core
# library finds it next to itself under flagcv/, in both the build tree and
# the install tree.
find_package(CUDAToolkit REQUIRED)

add_library(flagcv_nvidia MODULE backend.cpp)
target_include_directories(flagcv_nvidia PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(flagcv_nvidia PRIVATE CUDA::cuda_driver)
target_compile_features(flagcv_nvidia PRIVATE cxx_std_17)
set_target_properties(flagcv_nvidia PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY "${FLAGCV_PLUGIN_OUTPUT_DIR}")
flagcv_enable_warnings(flagcv_nvidia)

install(TARGETS flagcv_nvidia
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/flagcv")
install(FILES compiler.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/nvidia")
install(FILES kernels/registry.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/nvidia/kernels")
install(DIRECTORY kernels/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/nvidia/kernels"
  FILES_MATCHING PATTERN "*.py")
