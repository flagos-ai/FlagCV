# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Iluvatar plugin: a dlopen'd module that links only this platform's SDK.
# The core never links it (docs/architecture.md, "Plugin boundary").
#
# Iluvatar's driver API is CUDA-compatible, so it is found as CUDAToolkit --
# but it must be Iluvatar's CoreX, not NVIDIA's. Point CUDAToolkit_ROOT at the
# CoreX install (e.g. -DCMAKE_PREFIX_PATH=/usr/local/corex-4.4.0); building
# this plugin against an NVIDIA toolkit would produce a plugin that cannot
# bind an Iluvatar device while looking perfectly healthy at build time.
if(NOT CUDAToolkit_ROOT AND NOT DEFINED ENV{COREX_ROOT})
  message(FATAL_ERROR
    "the iluvatar backend needs Iluvatar's CoreX toolkit as CUDAToolkit_ROOT "
    "(e.g. -DCUDAToolkit_ROOT=/usr/local/corex-<version>), not the NVIDIA one")
endif()
find_package(CUDAToolkit REQUIRED)

add_library(flagcv_iluvatar MODULE backend.cpp)
target_link_libraries(flagcv_iluvatar PRIVATE CUDA::cuda_driver)
target_include_directories(flagcv_iluvatar PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(flagcv_iluvatar PRIVATE cxx_std_17)
set_target_properties(flagcv_iluvatar PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY "${FLAGCV_PLUGIN_OUTPUT_DIR}")
flagcv_enable_warnings(flagcv_iluvatar)

install(TARGETS flagcv_iluvatar
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/flagcv")
install(FILES compiler.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/iluvatar")
install(FILES kernels/registry.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/iluvatar/kernels")
install(DIRECTORY kernels/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/backends/iluvatar/kernels"
  FILES_MATCHING PATTERN "*.py")
