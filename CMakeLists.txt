cmake_minimum_required(VERSION 3.23)

project(
  FlagCV
  VERSION 0.1.0
  DESCRIPTION "Native C/C++ image processing runtime for FlagCV"
  LANGUAGES C CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(CTest)

math(EXPR FLAGCV_VERSION_NUMBER
  "${PROJECT_VERSION_MAJOR} * 10000 + ${PROJECT_VERSION_MINOR} * 100 + ${PROJECT_VERSION_PATCH}")
configure_file(
  cmake/flagcv_version.h.in
  "${CMAKE_CURRENT_BINARY_DIR}/generated/include/flagcv/version.h"
  @ONLY)

option(FLAGCV_BUILD_TESTS "Build FlagCV native tests" ${BUILD_TESTING})
option(FLAGCV_WARNINGS_AS_ERRORS "Treat native compiler warnings as errors" OFF)

set(FLAGCV_BACKENDS "nvidia"
    CACHE STRING "Semicolon-separated backend plugins to build")
set(FLAGCV_EXECUTION_ENGINE "libtriton_jit"
    CACHE STRING "Default execution engine")

find_package(Python3 COMPONENTS Interpreter QUIET)
set(FLAGCV_CODEGEN_PYTHON "${Python3_EXECUTABLE}"
    CACHE FILEPATH "Python executable providing the Triton toolchain")

function(flagcv_enable_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>)
  if(FLAGCV_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror>)
  endif()
endfunction()

set(FLAGCV_RESOURCE_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/flagcv")

add_subdirectory(src)

install(FILES kernels/registry.json
  DESTINATION "${FLAGCV_RESOURCE_INSTALL_DIR}/kernels")
install(DIRECTORY kernels/common/
  DESTINATION "${FLAGCV_RESOURCE_INSTALL_DIR}/kernels/common"
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE)

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
