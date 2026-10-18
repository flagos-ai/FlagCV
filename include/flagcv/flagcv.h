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

#ifndef FLAGCV_FLAGCV_H_
#define FLAGCV_FLAGCV_H_

#include <stddef.h>
#include <stdint.h>

#include <flagcv/types.h>
#include <flagcv/version.h>

/*
 * Revision 1 permits bounded, reclaimable execution-engine temporaries while
 * retaining the no-compile, no-autotune and no-synchronization guarantees of
 * flagcvExecuteAsync(). Binaries can query the linked runtime below.
 */
#define FLAGCV_EXECUTION_CONTRACT_VERSION 1U

#if defined(_WIN32)
#if defined(FLAGCV_BUILD_SHARED)
#define FLAGCV_API __declspec(dllexport)
#elif defined(FLAGCV_USE_SHARED)
#define FLAGCV_API __declspec(dllimport)
#else
#define FLAGCV_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define FLAGCV_API __attribute__((visibility("default")))
#else
#define FLAGCV_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum flagcvOperation {
  /* Generic elementwise descriptor covering arithmetic and logic. */
  FLAGCV_OPERATION_ELEMENTWISE = 0,
  FLAGCV_OPERATION_CVT_COLOR = 1,
  FLAGCV_OPERATION_RESIZE = 2,
  FLAGCV_OPERATION_FILTER = 3,
  FLAGCV_OPERATION_MORPHOLOGY = 4,
  FLAGCV_OPERATION_THRESHOLD = 5,
  /* Descriptor created by flagcvCreateOperationDescriptorByName. */
  FLAGCV_OPERATION_CUSTOM = 6
} flagcvOperation_t;

/*
 * Backend-neutral elementwise modes. Numeric values are owned by FlagCV and
 * must never be passed directly to a platform library.
 */
typedef enum flagcvElementwiseMode {
  FLAGCV_ELEMENTWISE_NOT_SET = 0,
  FLAGCV_ELEMENTWISE_ADD = 1,
  FLAGCV_ELEMENTWISE_SUBTRACT = 2,
  FLAGCV_ELEMENTWISE_MULTIPLY = 3,
  FLAGCV_ELEMENTWISE_DIVIDE = 4,
  FLAGCV_ELEMENTWISE_BITWISE_AND = 5,
  FLAGCV_ELEMENTWISE_BITWISE_OR = 6,
  FLAGCV_ELEMENTWISE_BITWISE_XOR = 7,
  FLAGCV_ELEMENTWISE_BITWISE_NOT = 8
} flagcvElementwiseMode_t;

typedef struct flagcvContext* flagcvHandle_t;
typedef struct flagcvImageDescriptor* flagcvImageDescriptor_t;
typedef struct flagcvOperationDescriptor* flagcvOperationDescriptor_t;
typedef struct flagcvPipeline* flagcvPipeline_t;
typedef struct flagcvExecutable* flagcvExecutable_t;

/*
 * A stream is an opaque native backend stream. NVIDIA interprets it as a
 * cudaStream_t/CUstream and Ascend interprets it as an aclrtStream. A null
 * value denotes the selected backend's default stream. The caller owns the
 * stream and must keep it alive until all enqueued work has completed.
 */
typedef void* flagcvStream_t;

/*
 * One device allocation per bound image. Multi-plane images bind a single
 * base pointer; per-plane offsets come from the image descriptor so that a
 * decoder's single allocation stays bindable without being split.
 */
typedef struct flagcvBinding {
  int64_t uid;
  void* device_pointer;
} flagcvBinding_t;

#define FLAGCV_BUILD_OPTIONS_VERSION 1U

#define FLAGCV_BUILD_OPTION_HEURISTIC_MODE_A (UINT64_C(1) << 0)
#define FLAGCV_BUILD_OPTION_AUTOTUNE (UINT64_C(1) << 1)
/* Disables adjacent-operation fusion; intended for differential debugging. */
#define FLAGCV_BUILD_OPTION_DISABLE_FUSION (UINT64_C(1) << 2)
#define FLAGCV_BUILD_OPTION_FLAGS_ALL            \
  (FLAGCV_BUILD_OPTION_HEURISTIC_MODE_A |        \
   FLAGCV_BUILD_OPTION_AUTOTUNE |                \
   FLAGCV_BUILD_OPTION_DISABLE_FUSION)

typedef struct flagcvBuildOptions {
  size_t struct_size;
  uint32_t version;
  uint64_t flags;
} flagcvBuildOptions_t;

#define FLAGCV_BUILD_OPTIONS_INITIALIZER                           \
  {                                                                \
    sizeof(flagcvBuildOptions_t), FLAGCV_BUILD_OPTIONS_VERSION, 0U \
  }

FLAGCV_API size_t flagcvGetVersion(void);
FLAGCV_API const char* flagcvGetVersionString(void);
FLAGCV_API uint32_t flagcvGetExecutionContractVersion(void);
FLAGCV_API const char* flagcvGetErrorString(flagcvStatus_t status);

/*
 * Returns the current thread's diagnostic for the most recent failed API
 * call. The returned pointer remains valid until the next FlagCV API call
 * on the same thread.
 */
FLAGCV_API const char* flagcvGetLastErrorString(void);

FLAGCV_API flagcvStatus_t flagcvCreate(flagcvHandle_t* handle);
FLAGCV_API flagcvStatus_t flagcvCreateWithBackendName(
    const char* backend_name,
    int32_t device_ordinal,
    flagcvHandle_t* handle);
FLAGCV_API flagcvStatus_t flagcvDestroy(flagcvHandle_t handle);

/* Returned strings remain valid for the lifetime of handle. */
FLAGCV_API flagcvStatus_t flagcvGetBackendName(
    flagcvHandle_t handle,
    const char** backend_name);
FLAGCV_API flagcvStatus_t flagcvGetTargetFingerprint(
    flagcvHandle_t handle,
    const char** target_fingerprint);

/*
 * Configures the external compiler used on an executable-cache miss. Strings
 * are copied by FlagCV. The compiler executable may be Python today, but the
 * public contract is backend-neutral and never embeds it in the caller.
 */
FLAGCV_API flagcvStatus_t flagcvSetCompilerConfig(
    flagcvHandle_t handle,
    const char* compiler_executable,
    const char* compiler_path,
    const char* cache_directory);

/*
 * Image descriptors
 *
 * Row pitch is expressed in bytes and is deliberately independent of
 * width * channels * element_size, because DMA-aligned padding is the norm
 * for imaging hardware. Pass a zero pitch to request the densely packed
 * value derived from the format.
 */
FLAGCV_API flagcvStatus_t flagcvCreateImageDescriptor(
    flagcvImageDescriptor_t* descriptor);
FLAGCV_API flagcvStatus_t flagcvDestroyImageDescriptor(
    flagcvImageDescriptor_t descriptor);

FLAGCV_API flagcvStatus_t flagcvSetImageDescriptor(
    flagcvImageDescriptor_t descriptor,
    int64_t uid,
    int32_t width,
    int32_t height,
    flagcvPixelDepth_t depth,
    int32_t channels,
    flagcvColorFormat_t format,
    int64_t row_pitch_in_bytes);

FLAGCV_API flagcvStatus_t flagcvGetImageDescriptor(
    flagcvImageDescriptor_t descriptor,
    int64_t* uid,
    int32_t* width,
    int32_t* height,
    flagcvPixelDepth_t* depth,
    int32_t* channels,
    flagcvColorFormat_t* format);
FLAGCV_API flagcvStatus_t flagcvGetImageSizeInBytes(
    flagcvImageDescriptor_t descriptor,
    size_t* size_in_bytes);

/*
 * Marks storage as executable-owned pipeline workspace. Virtual images may be
 * shared between operations by UID and must not appear in execute bindings.
 * The default after a set call is non-virtual.
 */
FLAGCV_API flagcvStatus_t flagcvSetImageDescriptorVirtual(
    flagcvImageDescriptor_t descriptor,
    int32_t is_virtual);
FLAGCV_API flagcvStatus_t flagcvGetImageDescriptorVirtual(
    flagcvImageDescriptor_t descriptor,
    int32_t* is_virtual);

/*
 * Operation descriptors
 *
 * The extensible by-name path mirrors the typed setters below. operation_kind,
 * port names and attribute names use lower_snake_case and are copied.
 * Finalize before adding the operation to a pipeline.
 */
FLAGCV_API flagcvStatus_t flagcvCreateOperationDescriptor(
    flagcvOperation_t operation,
    flagcvOperationDescriptor_t* descriptor);
FLAGCV_API flagcvStatus_t flagcvCreateOperationDescriptorByName(
    const char* operation_kind,
    flagcvOperationDescriptor_t* descriptor);
FLAGCV_API flagcvStatus_t flagcvSetOperationDescriptorInput(
    flagcvOperationDescriptor_t descriptor,
    const char* port_name,
    flagcvImageDescriptor_t image,
    int32_t is_optional);
FLAGCV_API flagcvStatus_t flagcvSetOperationDescriptorOutput(
    flagcvOperationDescriptor_t descriptor,
    const char* port_name,
    flagcvImageDescriptor_t image,
    int32_t is_optional);
FLAGCV_API flagcvStatus_t flagcvSetOperationDescriptorAttributeInt64(
    flagcvOperationDescriptor_t descriptor,
    const char* attribute_name,
    int64_t value);
FLAGCV_API flagcvStatus_t flagcvSetOperationDescriptorAttributeDouble(
    flagcvOperationDescriptor_t descriptor,
    const char* attribute_name,
    double value);
FLAGCV_API flagcvStatus_t flagcvSetOperationDescriptorAttributeBoolean(
    flagcvOperationDescriptor_t descriptor,
    const char* attribute_name,
    int32_t value);
FLAGCV_API flagcvStatus_t flagcvFinalizeOperationDescriptor(
    flagcvOperationDescriptor_t descriptor);
FLAGCV_API flagcvStatus_t flagcvDestroyOperationDescriptor(
    flagcvOperationDescriptor_t descriptor);

/*
 * Typed operation setters. These only describe computation and never access
 * a device. Each is a convenience layer over the by-name path above.
 *
 * The optional mask is an 8-bit single-channel image. Destination pixels whose
 * mask value is zero are left unmodified, so a masked operation must bind a
 * destination the caller has already initialized.
 */
FLAGCV_API flagcvStatus_t flagcvSetElementwiseBinaryOperationDescriptor(
    flagcvOperationDescriptor_t descriptor,
    flagcvImageDescriptor_t src1,
    flagcvImageDescriptor_t src2,
    flagcvElementwiseMode_t mode,
    flagcvImageDescriptor_t mask,
    flagcvImageDescriptor_t destination);

FLAGCV_API flagcvStatus_t flagcvSetElementwiseUnaryOperationDescriptor(
    flagcvOperationDescriptor_t descriptor,
    flagcvImageDescriptor_t source,
    flagcvElementwiseMode_t mode,
    flagcvImageDescriptor_t mask,
    flagcvImageDescriptor_t destination);

/*
 * Pipelines
 *
 * Operations are built in stable dependency order; callers need not add
 * producer nodes before consumers. A virtual output may feed another
 * operation through the same UID, and its storage is part of executable
 * workspace. Non-virtual images are supplied through execute bindings.
 */
FLAGCV_API flagcvStatus_t flagcvCreatePipeline(flagcvPipeline_t* pipeline);
FLAGCV_API flagcvStatus_t flagcvDestroyPipeline(flagcvPipeline_t pipeline);
FLAGCV_API flagcvStatus_t flagcvSetPipelineName(
    flagcvPipeline_t pipeline,
    const char* name);
FLAGCV_API flagcvStatus_t flagcvPipelineAddOperation(
    flagcvPipeline_t pipeline,
    flagcvOperationDescriptor_t operation);
/* Validates pipeline topology and operation semantics without freezing it. */
FLAGCV_API flagcvStatus_t flagcvValidatePipeline(flagcvPipeline_t pipeline);
/* Validates and freezes the pipeline. */
FLAGCV_API flagcvStatus_t flagcvFinalizePipeline(flagcvPipeline_t pipeline);
FLAGCV_API flagcvStatus_t flagcvGetPipelineOperationCount(
    flagcvPipeline_t pipeline,
    size_t* operation_count);

FLAGCV_API flagcvStatus_t flagcvBuildExecutable(
    flagcvHandle_t handle,
    flagcvPipeline_t pipeline,
    const flagcvBuildOptions_t* options,
    flagcvExecutable_t* executable);
FLAGCV_API flagcvStatus_t flagcvDestroyExecutable(
    flagcvExecutable_t executable);
FLAGCV_API flagcvStatus_t flagcvGetExecutableWorkspaceSize(
    flagcvExecutable_t executable,
    size_t* workspace_size);

/*
 * Enqueues work on caller_stream. This function does not compile, autotune,
 * or actively synchronize the stream/device. An execution engine may create
 * bounded host temporaries that are released before return and reclaimable
 * device temporaries whose cleanup is enqueued on caller_stream. Repeated
 * execution must not leak or grow resident resources without bound.
 *
 * Device bindings, pipeline workspace and stream remain caller-owned and must
 * stay alive until the enqueued work completes. When the executable requires
 * nonzero workspace, workspace must be non-null and its base address must be
 * at least 256-byte aligned.
 */
FLAGCV_API flagcvStatus_t flagcvExecuteAsync(
    flagcvExecutable_t executable,
    const flagcvBinding_t bindings[],
    size_t binding_count,
    void* workspace,
    size_t workspace_size,
    flagcvStream_t caller_stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FLAGCV_FLAGCV_H_ */
