# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Ascend plugin: a dlopen'd module that links only this platform's SDK.
# The core never links it (docs/architecture.md, "Plugin boundary").

add_library(flagcv_ascend MODULE backend.cpp)
find_package(Ascend REQUIRED)
target_link_libraries(flagcv_ascend PRIVATE Ascend::ascendcl Ascend::runtime)
target_include_directories(flagcv_ascend PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(flagcv_ascend PRIVATE cxx_std_17)
set_target_properties(flagcv_ascend PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY "${FLAGCV_PLUGIN_OUTPUT_DIR}")
flagcv_enable_warnings(flagcv_ascend)

install(TARGETS flagcv_ascend
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/flagcv")
install(FILES compiler.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/ascend")
install(FILES kernels/registry.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/ascend/kernels")
install(DIRECTORY kernels/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/ascend/kernels"
  FILES_MATCHING PATTERN "*.py")
