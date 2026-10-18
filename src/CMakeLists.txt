# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

add_library(flagcv
  image/spec.cpp
  pipeline/lowering/dispatch.cpp
  pipeline/lowering/bitwise.cpp
  runtime/kernel_resolution.cpp
  runtime/immediate_cache.cpp
  imgproc/bitwise.cpp)

target_include_directories(flagcv
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${PROJECT_SOURCE_DIR})

target_compile_features(flagcv PUBLIC cxx_std_20)
flagcv_enable_warnings(flagcv)

install(TARGETS flagcv EXPORT FlagCVTargets)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" TYPE INCLUDE)
install(DIRECTORY "${PROJECT_BINARY_DIR}/generated/include/" TYPE INCLUDE)
