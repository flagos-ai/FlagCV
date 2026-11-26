/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_TESTS_COMMON_TOLERANCE_HPP_
#define FLAGCV_TESTS_COMMON_TOLERANCE_HPP_

#include <cstddef>
#include <cstdint>
#include <string>

namespace flagcv {
namespace testing {

/*
 * Comparators for docs/numerical-tolerance.md. Only class A (bit-exact) is
 * implemented so far; B, C and D land with the first operator that needs them.
 */

/* Class A: byte-identical. Returns an empty string on success, otherwise a
 * description of the first mismatching byte. */
inline std::string bit_exact(const void* actual, const void* expected,
                             std::size_t bytes) {
  const auto* a = static_cast<const std::uint8_t*>(actual);
  const auto* e = static_cast<const std::uint8_t*>(expected);
  for (std::size_t i = 0; i < bytes; ++i) {
    if (a[i] != e[i]) {
      return "byte " + std::to_string(i) + ": got " + std::to_string(a[i]) +
             ", expected " + std::to_string(e[i]);
    }
  }
  return {};
}

}  // namespace testing
}  // namespace flagcv

#endif  // FLAGCV_TESTS_COMMON_TOLERANCE_HPP_
