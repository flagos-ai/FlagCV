# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# One test executable per operator family, named after the family:
#
#   flagcv_add_test(bitwise)   # builds tests/test_bitwise.cpp
#
# Operator tests use the helper below; backend tests are registered separately.
function(flagcv_add_test family)
  add_executable(test_${family} test_${family}.cpp)
  target_link_libraries(test_${family} PRIVATE flagcv)
  target_include_directories(test_${family} PRIVATE ${PROJECT_SOURCE_DIR})
  flagcv_enable_warnings(test_${family})
  add_test(NAME ${family} COMMAND test_${family})
endfunction()

if(TARGET flagcv_backend_amd)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME amd_provider
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_amd.py)

  option(FLAGCV_TEST_AMD_GPU "Run AMD tests requiring a GPU, Triton and PyTorch" OFF)
  if(FLAGCV_TEST_AMD_GPU)
    find_package(hip CONFIG REQUIRED)
    flagcv_add_test(amd_handle)
    target_link_libraries(test_amd_handle PRIVATE hip::host)
    add_dependencies(test_amd_handle flagcv_backend_amd)
    set_tests_properties(amd_handle PROPERTIES
      ENVIRONMENT "FLAGCV_BACKEND_PATH=$<TARGET_FILE_DIR:flagcv_backend_amd>")
    add_test(NAME amd_gpu
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_amd.py
        --plugin $<TARGET_FILE:flagcv_backend_amd> GpuTests)
    set_tests_properties(amd_handle amd_gpu PROPERTIES LABELS "amd;gpu" RUN_SERIAL TRUE)
  endif()
endif()
