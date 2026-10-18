# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)

project(
  FlagCV
  VERSION 0.1.0
  DESCRIPTION "Multi-backend image processing runtime with Triton kernels"
  LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(CTest)

option(FLAGCV_BUILD_TESTS "Build FlagCV tests" ${BUILD_TESTING})
option(FLAGCV_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

set(FLAGCV_BACKENDS "nvidia"
    CACHE STRING "Semicolon-separated backend plugins to build")

function(flagcv_enable_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>)
  if(FLAGCV_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror>)
  endif()
endfunction()

add_subdirectory(src)
add_subdirectory(backends)

# Kernels and the external compiler are installed as data, not compiled here.
install(FILES kernels/registry.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/kernels")
install(DIRECTORY kernels/common/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/kernels/common"
  FILES_MATCHING PATTERN "*.py")
install(DIRECTORY compiler/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/compiler"
  FILES_MATCHING PATTERN "*.py")

configure_package_config_file(
  cmake/FlagCVConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/FlagCVConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlagCV")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/FlagCVConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion)
install(EXPORT FlagCVTargets
  FILE FlagCVTargets.cmake
  NAMESPACE FlagCV::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlagCV")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/FlagCVConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/FlagCVConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlagCV")

if(FLAGCV_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
