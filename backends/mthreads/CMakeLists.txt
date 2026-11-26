# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Mthreads plugin: a dlopen'd module that links only this platform's SDK.
# The core never links it (docs/architecture.md, "Plugin boundary").

add_library(flagcv_mthreads MODULE backend.cpp)
find_library(MUSA_DRIVER musa_driver REQUIRED)
target_link_libraries(flagcv_mthreads PRIVATE ${MUSA_DRIVER})
target_include_directories(flagcv_mthreads PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(flagcv_mthreads PRIVATE cxx_std_17)
set_target_properties(flagcv_mthreads PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY "${FLAGCV_PLUGIN_OUTPUT_DIR}")
flagcv_enable_warnings(flagcv_mthreads)

install(TARGETS flagcv_mthreads
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/flagcv")
install(FILES compiler.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/mthreads")
install(FILES kernels/registry.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/mthreads/kernels")
install(DIRECTORY kernels/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/mthreads/kernels"
  FILES_MATCHING PATTERN "*.py")
