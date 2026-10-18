/*
 * Copyright (c) 2025-2026 BAAI. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLAGCV_TYPES_H_
#define FLAGCV_TYPES_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Planar and semi-planar formats need per-plane storage metadata. */
#define FLAGCV_MAX_PLANES 4

typedef enum flagcvStatus {
  FLAGCV_STATUS_SUCCESS = 0,
  FLAGCV_STATUS_INVALID_VALUE = 1,
  FLAGCV_STATUS_NOT_INITIALIZED = 2,
  FLAGCV_STATUS_ALLOC_FAILED = 3,
  FLAGCV_STATUS_NOT_SUPPORTED = 4,
  FLAGCV_STATUS_COMPILATION_FAILED = 5,
  FLAGCV_STATUS_BACKEND_ERROR = 6,
  FLAGCV_STATUS_INTERNAL_ERROR = 7
} flagcvStatus_t;

typedef enum flagcvBackend {
  FLAGCV_BACKEND_AUTO = 0,
  FLAGCV_BACKEND_NVIDIA = 1,
  FLAGCV_BACKEND_CUDA = FLAGCV_BACKEND_NVIDIA
} flagcvBackend_t;

/*
 * Per-channel storage type. Numeric values are owned by FlagCV and must never
 * be passed directly to a platform library.
 */
typedef enum flagcvPixelDepth {
  FLAGCV_DEPTH_8U = 0,
  FLAGCV_DEPTH_8S = 1,
  FLAGCV_DEPTH_16U = 2,
  FLAGCV_DEPTH_16S = 3,
  FLAGCV_DEPTH_32S = 4,
  FLAGCV_DEPTH_16F = 5,
  FLAGCV_DEPTH_32F = 6,
  FLAGCV_DEPTH_64F = 7
} flagcvPixelDepth_t;

/*
 * Colour format carries channel meaning and plane topology, which row strides
 * alone cannot express. Chroma-subsampled formats derive per-plane extents
 * from the luma extent; see src/image/format.cpp for the single authority on
 * that derivation.
 */
typedef enum flagcvColorFormat {
  FLAGCV_FORMAT_GRAY = 0,
  FLAGCV_FORMAT_BGR = 1,
  FLAGCV_FORMAT_RGB = 2,
  FLAGCV_FORMAT_BGRA = 3,
  FLAGCV_FORMAT_RGBA = 4,
  FLAGCV_FORMAT_HSV = 5,
  FLAGCV_FORMAT_LAB = 6,
  FLAGCV_FORMAT_YCRCB = 7,
  /* Two planes: luma followed by interleaved chroma at half resolution. */
  FLAGCV_FORMAT_NV12 = 8,
  FLAGCV_FORMAT_NV21 = 9,
  /* Three planes: luma, then both chroma planes at half resolution. */
  FLAGCV_FORMAT_YUV420P = 10,
  /* Arbitrary channel count with no colour semantics. */
  FLAGCV_FORMAT_GENERIC = 11
} flagcvColorFormat_t;

typedef enum flagcvInterpolation {
  FLAGCV_INTERP_NEAREST = 0,
  FLAGCV_INTERP_LINEAR = 1,
  FLAGCV_INTERP_CUBIC = 2,
  FLAGCV_INTERP_AREA = 3
} flagcvInterpolation_t;

typedef enum flagcvBorderMode {
  FLAGCV_BORDER_CONSTANT = 0,
  FLAGCV_BORDER_REPLICATE = 1,
  FLAGCV_BORDER_REFLECT = 2,
  /* Reflect without repeating the edge pixel. */
  FLAGCV_BORDER_REFLECT_101 = 3,
  FLAGCV_BORDER_WRAP = 4
} flagcvBorderMode_t;

typedef struct flagcvSize {
  int32_t width;
  int32_t height;
} flagcvSize_t;

typedef struct flagcvRect {
  int32_t x;
  int32_t y;
  int32_t width;
  int32_t height;
} flagcvRect_t;

typedef struct flagcvScalar {
  double value[4];
} flagcvScalar_t;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FLAGCV_TYPES_H_ */
