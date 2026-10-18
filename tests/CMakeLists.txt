# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# One test executable per operator family, named after the family:
#
#   flagcv_add_test(bitwise)   # builds tests/test_bitwise.cpp
#
# No operators exist yet, so nothing is registered here.
function(flagcv_add_test family)
  add_executable(test_${family} test_${family}.cpp)
  target_link_libraries(test_${family} PRIVATE flagcv)
  target_include_directories(test_${family} PRIVATE ${PROJECT_SOURCE_DIR})
  flagcv_enable_warnings(test_${family})
  add_test(NAME ${family} COMMAND test_${family})
endfunction()
