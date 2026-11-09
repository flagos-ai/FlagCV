# One test executable per operator family, named after the family:
#
#   flagcv_add_test(bitwise)   # builds tests/test_bitwise.cpp
function(flagcv_add_test family)
  add_executable(test_${family} test_${family}.cpp)
  target_link_libraries(test_${family} PRIVATE flagcv)
  target_include_directories(test_${family} PRIVATE ${PROJECT_SOURCE_DIR})
  flagcv_enable_warnings(test_${family})
  add_test(NAME ${family} COMMAND test_${family})
endfunction()

# The operator tests allocate device memory themselves (FlagCV never does),
# which ties them to a device runtime. Only the CUDA runtime is wired up so
# far, so they build when the nvidia backend does.
if("nvidia" IN_LIST FLAGCV_BACKENDS)
  find_package(CUDAToolkit REQUIRED)
  flagcv_add_test(flip)
  target_link_libraries(test_flip PRIVATE CUDA::cudart)
  # Each test needs the plugin it loads at runtime.
  add_dependencies(test_flip flagcv_nvidia)
endif()
