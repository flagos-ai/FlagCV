# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

add_executable(test_bitwise_and test_bitwise_and.cpp)
target_link_libraries(test_bitwise_and PRIVATE flagcv)
target_include_directories(test_bitwise_and PRIVATE ${PROJECT_SOURCE_DIR})
add_test(NAME bitwise_and COMMAND test_bitwise_and)
