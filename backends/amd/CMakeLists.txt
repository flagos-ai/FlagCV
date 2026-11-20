# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Host-only module API: neither hipcc nor GPU architecture flags are needed.
find_package(hip CONFIG REQUIRED)
add_library(flagcv_backend_amd MODULE backend.cpp)
target_link_libraries(flagcv_backend_amd PRIVATE hip::host)
target_include_directories(flagcv_backend_amd PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(flagcv_backend_amd PRIVATE cxx_std_17)
flagcv_enable_warnings(flagcv_backend_amd)
install(TARGETS flagcv_backend_amd
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/flagcv/backends")
install(FILES compiler.py DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/amd")
install(DIRECTORY kernels/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/amd/kernels"
  FILES_MATCHING PATTERN "*.json" PATTERN "*.py")
