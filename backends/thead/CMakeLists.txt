# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Thead plugin: a dlopen'd module that links only this platform's SDK.
# The core never links it (docs/architecture.md, "Plugin boundary").

add_library(flagcv_thead MODULE backend.cpp)
find_library(HGGC_DRIVER hggc REQUIRED)
target_link_libraries(flagcv_thead PRIVATE ${HGGC_DRIVER})
target_include_directories(flagcv_thead PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(flagcv_thead PRIVATE cxx_std_17)
set_target_properties(flagcv_thead PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY "${FLAGCV_PLUGIN_OUTPUT_DIR}")
flagcv_enable_warnings(flagcv_thead)

install(TARGETS flagcv_thead
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/flagcv")
install(FILES compiler.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/thead")
install(FILES kernels/registry.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/thead/kernels")
install(DIRECTORY kernels/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/thead/kernels"
  FILES_MATCHING PATTERN "*.py")
