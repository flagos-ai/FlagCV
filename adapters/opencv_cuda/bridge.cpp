/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <exception>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
struct Operation {
  int mode;
  int flip_code;
  cv::cuda::GpuMat a, b, dst;
  cv::cuda::Stream stream;

  Operation(int mode_, int rows, int cols, int depth, void* a_, void* b_,
            void* dst_, int flip_code_, void* stream_)
      : mode(mode_),
        flip_code(flip_code_),
        a(rows, cols, CV_MAKETYPE(depth, 1), a_),
        dst(rows, cols, CV_MAKETYPE(depth, 1), dst_),
        stream(cv::cuda::StreamAccessor::wrapStream(
            static_cast<cudaStream_t>(stream_))) {
    if (b_) b = cv::cuda::GpuMat(rows, cols, CV_MAKETYPE(depth, 1), b_);
  }

  void run() {
    auto* original = dst.data;
    switch (mode) {
      case 0:
        cv::cuda::add(a, b, dst, cv::noArray(), -1, stream);
        break;
      case 1:
        cv::cuda::subtract(a, b, dst, cv::noArray(), -1, stream);
        break;
      case 2:
        cv::cuda::multiply(a, b, dst, 1.0, -1, stream);
        break;
      case 3:
        cv::cuda::bitwise_and(a, b, dst, cv::noArray(), stream);
        break;
      case 4:
        cv::cuda::bitwise_not(a, dst, cv::noArray(), stream);
        break;
      case 5:
        cv::cuda::flip(a, dst, flip_code, stream);
        break;
      default:
        throw std::invalid_argument("unknown OpenCV operation");
    }
    if (dst.data != original)
      throw std::runtime_error("OpenCV replaced the preallocated destination");
  }
};
}  // namespace

extern "C" {
const char* flagcv_opencv_error() { return error.c_str(); }
const char* flagcv_opencv_version() { return CV_VERSION; }
void* flagcv_opencv_prepare(int mode, int rows, int cols, int depth, void* a,
                            void* b, void* dst, int flip_code, void* stream,
                            int device) {
  try {
    if (mode < 0 || mode > 5 || rows <= 0 || cols <= 0 ||
        (depth != CV_8U && depth != CV_32F) || !a || !dst || (mode <= 3 && !b))
      throw std::invalid_argument("invalid OpenCV CUDA operands");
    cv::cuda::setDevice(device);
    return new Operation(mode, rows, cols, depth, a, b, dst, flip_code, stream);
  } catch (const std::exception& e) {
    error = e.what();
    return nullptr;
  }
}
int flagcv_opencv_run(void* operation) {
  try {
    static_cast<Operation*>(operation)->run();
    return 0;
  } catch (const std::exception& e) {
    error = e.what();
    return 1;
  }
}
void flagcv_opencv_destroy(void* operation) {
  delete static_cast<Operation*>(operation);
}
}
