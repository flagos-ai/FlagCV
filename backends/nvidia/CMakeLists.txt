# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Nvidia plugin: a dlopen'd module that links only this platform's SDK.
# The core never links it (docs/architecture.md, "Plugin boundary").

add_library(flagcv_nvidia MODULE backend.cpp)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(flagcv_nvidia PRIVATE CUDA::cuda_driver)
target_include_directories(flagcv_nvidia PRIVATE ${PROJECT_SOURCE_DIR})
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
