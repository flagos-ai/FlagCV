# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Each platform in FLAGCV_BACKENDS is built as a separate plugin. A platform
# owns its SDK discovery inside its own directory, so adding one does not
# touch this file beyond the list in the root CMakeLists.txt.
set(_flagcv_backends ${FLAGCV_BACKENDS})
list(REMOVE_DUPLICATES _flagcv_backends)

foreach(backend IN LISTS _flagcv_backends)
  if(NOT backend MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "invalid FLAGCV_BACKENDS entry: ${backend}")
  endif()
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${backend}/CMakeLists.txt")
    message(FATAL_ERROR
      "backend ${backend} has no implementation at backends/${backend}")
  endif()
  add_subdirectory("${backend}")
endforeach()
