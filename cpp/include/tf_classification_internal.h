// Internal declarations for the Phase-E classification kernels
// (cpp/src/classification.cpp). NOT part of the C ABI: these symbols stay
// in ``namespace tf`` with hidden visibility, so Python never reaches
// them — the dependency-free CTests link the translation unit directly to
// exercise the compute in isolation from the guarded C ABI wrapper, the
// same split conv2d/pooling use (tf_conv2d_internal.h,
// tf_pooling_internal.h).
#pragma once

#include <cstdint>

namespace tf {

// Numerically stable softmax over one axis of a row-major CONTIGUOUS
// float64 tensor, written as the (outer, axis_length, inner)
// decomposition: element (o, k, i) of a slice lives at
// ``o * axis_length * inner + k * inner + i``, so one reduction slice is
// a fixed (o, i) pair walked over k with stride ``inner``.
//
// Per slice: m = max_k x, then out = exp(x - m), then out /= sum(out) —
// the maximum shift keeps every exponent <= 0, so a common offset can
// never overflow. ``src`` already points at the first element (the
// caller adds any storage offset); ``dst`` is caller-allocated with at
// least outer*axis_length*inner elements and is written in full.
//
// noexcept and allocation-free: the guarded exported wrapper does all
// validation, and this reads ``src`` without mutating it. Deterministic
// traversal; no NaN/inf special-casing beyond plain IEEE arithmetic.
void softmax_forward_contiguous(
    const double* src, double* dst,
    int64_t outer, int64_t axis_length, int64_t inner) noexcept;

}  // namespace tf
