# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Hygon plugin: a dlopen'd module that links only this platform's SDK.
# The core never links it (docs/architecture.md, "Plugin boundary").

add_library(flagcv_hygon MODULE backend.cpp)
find_package(hip REQUIRED)
target_link_libraries(flagcv_hygon PRIVATE hip::host)
target_include_directories(flagcv_hygon PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(flagcv_hygon PRIVATE cxx_std_17)
set_target_properties(flagcv_hygon PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY "${FLAGCV_PLUGIN_OUTPUT_DIR}")
flagcv_enable_warnings(flagcv_hygon)

install(TARGETS flagcv_hygon
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/flagcv")
install(FILES compiler.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/hygon")
install(FILES kernels/registry.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/hygon/kernels")
install(DIRECTORY kernels/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/hygon/kernels"
  FILES_MATCHING PATTERN "*.py")
