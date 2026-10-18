/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGCV_BACKENDS_AUTOTUNE_POLICY_HPP_
#define FLAGCV_BACKENDS_AUTOTUNE_POLICY_HPP_

/*
 * TODO: cross-platform autotune candidate policy.
 *
 * Shared by every plugin so that timing methodology does not vary per
 * platform. A single viable candidate skips timing entirely; multiple
 * candidates are warmed up and then timed with device events.
 *
 * A persisted selection may only be reused when the policy, candidate set,
 * measurement method and device all match, otherwise a stale choice silently
 * degrades performance.
 */

#endif  // FLAGCV_BACKENDS_AUTOTUNE_POLICY_HPP_
