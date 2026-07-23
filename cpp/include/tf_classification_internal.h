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

// The reduction codes the fused cross-entropy exports carry across the C
// ABI (docs/native_classification_design.md §9.2). A small integer, never
// a string: the ABI stays free of allocation and encoding concerns, and
// both sides validate the code. Phase E supports exactly these two.
constexpr int64_t kCrossEntropyReductionMean = 0;
constexpr int64_t kCrossEntropyReductionSum = 1;

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

// Numerically stable log-softmax over one axis of a row-major CONTIGUOUS
// float64 tensor, using the identical (outer, axis_length, inner)
// decomposition as ``softmax_forward_contiguous`` above.
//
// Per slice: m = max_k x, then out = (x - m), accumulating
// sum_exp = Σ_k exp(x_k - m) as it goes, then out -= log(sum_exp) — the
// fused maximum-shift / log-sum-exp form. It is emphatically NOT
// log(softmax(x)): no probability buffer is formed and no division
// happens, which is exactly the precision loss the operation exists to
// avoid in the small-probability regime. ``src`` already points at the
// first element (the caller adds any storage offset); ``dst`` is
// caller-allocated with at least outer*axis_length*inner elements and is
// written in full.
//
// noexcept and allocation-free: the guarded exported wrapper does all
// validation, and this reads ``src`` without mutating it. Deterministic
// traversal; no NaN/inf special-casing beyond plain IEEE arithmetic.
void log_softmax_forward_contiguous(
    const double* src, double* dst,
    int64_t outer, int64_t axis_length, int64_t inner) noexcept;

// Fused multi-class cross-entropy forward over a row-major CONTIGUOUS
// float64 ``(batch_size, num_classes)`` logits block, with the class axis
// fixed at axis 1. Element (n, c) lives at ``n * num_classes + c``.
//
// Per row: m = max_c x, then, in ONE pass, exp(x - m) is written straight
// into ``probabilities`` while sum_exp = Σ_c exp(x_c - m) accumulates;
// the row is normalized in place, and the per-example loss is
// ``log(sum_exp) - (x[target] - m)``. That is the fused maximum-shift /
// log-sum-exp form: it never computes ``-log(probabilities[target])``
// (which loses the whole point once a probability underflows to 0),
// never forms a public softmax or log-softmax first, clamps nothing, and
// inserts no epsilon. The per-example losses accumulate in deterministic
// batch order; ``*loss`` receives that total for
// ``kCrossEntropyReductionSum`` and the total divided ONCE by
// ``batch_size`` for ``kCrossEntropyReductionMean`` (never by
// ``num_classes``).
//
// ``logits`` already points at the first element (the caller adds any
// storage offset); ``targets`` is a caller-owned span of ``batch_size``
// int64 class labels, every one already proved to be in
// ``[0, num_classes)`` by the guarded wrapper; ``loss`` is one caller
// -allocated double; ``probabilities`` is caller-allocated with at least
// ``batch_size * num_classes`` elements and is written in full.
//
// noexcept and allocation-free: the guarded exported wrapper does all
// validation, and this reads its inputs without mutating them.
// Deterministic traversal; no NaN/inf special-casing beyond plain IEEE
// arithmetic. ALLOCATES NOTHING and knows nothing about autograd.
void cross_entropy_forward_contiguous(
    const double* logits, const int64_t* targets,
    double* loss, double* probabilities,
    int64_t batch_size, int64_t num_classes,
    int64_t reduction_code) noexcept;

// Gradient of the fused cross-entropy with respect to its logits, from
// the SAVED probabilities alone — the logits are never reread and the
// softmax is never recomputed.
//
// For each (n, c): base = probabilities[n, c] - (c == targets[n] ? 1 : 0),
// divided once by ``batch_size`` for ``kCrossEntropyReductionMean``, then
// scaled by the single upstream value ``*upstream``. The result goes into
// the caller-allocated contiguous ``(batch_size, num_classes)``
// ``grad_logits`` block, which is written in full.
//
// ``probabilities`` and ``upstream`` already point at their first
// elements (the caller adds any storage offset) and are read without
// mutation; ``targets`` is the same validated int64 span the forward
// used. noexcept, allocation-free, deterministic, autograd-unaware.
void cross_entropy_backward_contiguous(
    const double* probabilities, const int64_t* targets,
    const double* upstream, double* grad_logits,
    int64_t batch_size, int64_t num_classes,
    int64_t reduction_code) noexcept;

}  // namespace tf
