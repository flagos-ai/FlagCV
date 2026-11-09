# Operator sources are added here as they land: one file per family under
# core/ or imgproc/, plus its lowering under lowering/.
add_library(flagcv
  core/flip.cpp
  image/gpumat.cpp
  lowering/flip.cpp
  runtime/compiler_client.cpp
  runtime/json.cpp
  runtime/paths.cpp
  runtime/runtime.cpp)

target_include_directories(flagcv
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${PROJECT_SOURCE_DIR})

# Build-tree fallbacks for locating kernels, the compiler and plugins; the
# install tree is found relative to the library itself (src/runtime/paths.hpp).
target_compile_definitions(flagcv PRIVATE
  FLAGCV_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
  FLAGCV_PLUGIN_BUILD_DIR="${FLAGCV_PLUGIN_OUTPUT_DIR}")

# The core links no vendor SDK and no Python: only the dynamic loader, for
# backend plugins.
target_link_libraries(flagcv PRIVATE ${CMAKE_DL_LIBS})
target_compile_features(flagcv PUBLIC cxx_std_17)
flagcv_enable_warnings(flagcv)

install(TARGETS flagcv EXPORT FlagCVTargets)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" TYPE INCLUDE)
