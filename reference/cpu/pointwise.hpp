/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGCV_REFERENCE_CPU_POINTWISE_HPP_
#define FLAGCV_REFERENCE_CPU_POINTWISE_HPP_

#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace flagcv::reference {
// Scalar oracle: widen arithmetic before saturating integer outputs.
template <class T>
T pointwise(int mode, T a, T b) {
  if (mode < 3) {
    const double x = a, y = b;
    const double result = mode == 0 ? x + y : (mode == 1 ? x - y : x * y);
    if constexpr (std::is_integral_v<T>)
      return static_cast<T>(std::clamp(result, 0.0, 255.0));
    else
      return static_cast<T>(result);
  }
  if constexpr (std::is_integral_v<T>)
    return mode == 3 ? static_cast<T>(a & b) : static_cast<T>(~a);
  throw std::runtime_error("invalid oracle mode");
}

}  // namespace flagcv::reference

#endif
