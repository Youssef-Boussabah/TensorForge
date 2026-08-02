// Internal declarations for the Phase-E classification kernels. NOT part
// of the C ABI: these symbols stay in ``namespace tf`` with hidden
// visibility, so Python never reaches them — the dependency-free CTests
// link the translation unit directly to exercise the compute in isolation
// from the guarded C ABI wrapper, the same split conv2d/pooling use
// (tf_conv2d_internal.h, tf_pooling_internal.h).
//
// ---------------------------------------------------------------------------
// Phase I, milestone I6: the four compute kernels carry a scalar type
// parameter, and their definitions moved here from cpp/src/classification.cpp.
// ---------------------------------------------------------------------------
//
// ``T`` is **deduced from the pointer arguments**, so every pre-Phase-I call
// site — all of which pass ``double*`` — instantiates ``T = double`` and is
// the pre-I6 kernel statement for statement. ``T = float`` is the same
// source at binary32.
//
// The definitions live in this header rather than in the .cpp for exactly
// I4's and I5's reason: a template must be *visible* where it is
// instantiated, and both instantiations have to reach the exported wrappers
// in classification.cpp **and** the CTests that compile that file directly.
// Nothing about the source organization of docs/native_classification_design.md
// §9.1 changes — the classification compute is still one unit, still
// deliberately apart from elementwise.cpp, and the guarded C ABI wrappers
// still live beside it in classification.cpp.
//
// **Everything numerical happens in ``T``** (design §10.1, §10.5): the
// per-slice maximum is a ``T`` comparison scan, the shift is a ``T``
// subtraction, ``std::exp``/``std::log`` receive a ``T`` and therefore
// select the ``float`` overload for a float32 tensor rather than computing
// in double and narrowing, the normalizing sum accumulates in ``T``, and the
// batch loss accumulates in ``T``. There is **no hidden float64
// accumulator** at any width and no dtype branch below this point — not per
// slice, not per class, not per element.
//
// The stability of these kernels comes from the **maximum shift** and the
// **fused log-sum-exp**, not from the width: every exponent is <= 0 for
// finite input, so ``exp`` cannot overflow at either width, and the fused
// form never computes ``log(softmax(x))`` or ``-log(probability[target])``.
// What *is* width-dependent, and is stated rather than glossed: the shifted
// value ``x - m`` is itself a ``T``, so when the slice's spread ``m - min``
// exceeds ``T``'s largest finite value the shift overflows to ``-inf``.
// That is the correctly-rounded IEEE-754 result for a quantity with no
// representation at that width, it happens at binary64 too (past ~1.8e308),
// and it leaves softmax exact — the affected class gets exactly the
// mathematically correct probability 0 — while log-softmax reports ``-inf``
// and cross-entropy ``+inf`` for a true value below/above the representable
// range. See docs/native_dtype_float32_design.md §10.5.
#pragma once

#include <cmath>
#include <cstdint>

namespace tf {

// The reduction codes the fused cross-entropy exports carry across the C
// ABI (docs/native_classification_design.md §9.2). A small integer, never
// a string: the ABI stays free of allocation and encoding concerns, and
// both sides validate the code. Phase E supports exactly these two.
constexpr int64_t kCrossEntropyReductionMean = 0;
constexpr int64_t kCrossEntropyReductionSum = 1;

// Numerically stable softmax over one axis of a row-major CONTIGUOUS
// tensor of element type ``T``, written as the (outer, axis_length, inner)
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
template <class T>
inline void softmax_forward_contiguous(
    const T* src, T* dst,
    int64_t outer, int64_t axis_length, int64_t inner) noexcept {
    // One reduction slice per (outer, inner) pair; the axis is walked
    // with stride ``inner``. Loop order is fixed, so results are
    // reproducible run to run.
    for (int64_t o = 0; o < outer; ++o) {
        const int64_t plane = o * axis_length * inner;
        for (int64_t i = 0; i < inner; ++i) {
            const int64_t base = plane + i;

            // Pass 1 — the slice maximum. A strict `>` comparison means a
            // NaN never becomes the maximum (NaN > m is false), matching
            // the pooling kernel's convention; a NaN still poisons the
            // slice through the shift below, which is the honest IEEE
            // outcome rather than a special case. The comparison is in
            // ``T``, and IEEE-754 defines it identically at both widths.
            T maximum = src[base];
            for (int64_t k = 1; k < axis_length; ++k) {
                const T value = src[base + k * inner];
                if (value > maximum) {
                    maximum = value;
                }
            }

            // Pass 2 — shifted exponentials, written straight into the
            // destination (no second buffer is allocated) and accumulated
            // in the element type. Every exponent is <= 0 for finite
            // input, so a large common offset cannot overflow. ``std::exp``
            // is called on a ``T``, so a float32 slice takes the ``float``
            // overload rather than widening to double and narrowing back.
            T total = T(0);
            for (int64_t k = 0; k < axis_length; ++k) {
                const T shifted = std::exp(src[base + k * inner] - maximum);
                dst[base + k * inner] = shifted;
                total += shifted;
            }

            // Pass 3 — normalize in place. For finite input ``total`` is
            // >= 1 (the maximum contributes exp(0) == 1), so this never
            // divides by zero; for a NaN/inf slice the division simply
            // propagates the IEEE result.
            for (int64_t k = 0; k < axis_length; ++k) {
                dst[base + k * inner] /= total;
            }
        }
    }
}

// Numerically stable log-softmax over one axis of a row-major CONTIGUOUS
// tensor of element type ``T``, using the identical (outer, axis_length,
// inner) decomposition as ``softmax_forward_contiguous`` above.
//
// Per slice: m = max_k x, then out = (x - m), accumulating
// sum_exp = Σ_k exp(x_k - m) as it goes, then out -= log(sum_exp) — the
// fused maximum-shift / log-sum-exp form. It is emphatically NOT
// log(softmax(x)): no probability buffer is formed and no division
// happens, which is exactly the precision loss the operation exists to
// avoid in the small-probability regime — a reason that only gets stronger
// at float32, where the smallest normal probability is ~1.18e-38 and
// underflow therefore arrives far sooner. ``src`` already points at the
// first element (the caller adds any storage offset); ``dst`` is
// caller-allocated with at least outer*axis_length*inner elements and is
// written in full.
//
// noexcept and allocation-free: the guarded exported wrapper does all
// validation, and this reads ``src`` without mutating it. Deterministic
// traversal; no NaN/inf special-casing beyond plain IEEE arithmetic.
template <class T>
inline void log_softmax_forward_contiguous(
    const T* src, T* dst,
    int64_t outer, int64_t axis_length, int64_t inner) noexcept {
    // The same slice decomposition softmax uses; only the arithmetic per
    // slice differs. Deliberately NOT softmax followed by a logarithm:
    // no probability is ever formed, so a tiny probability never rounds
    // to 0 before its logarithm is taken (log(0) == -inf), which is the
    // whole reason this operation exists as its own kernel.
    for (int64_t o = 0; o < outer; ++o) {
        const int64_t plane = o * axis_length * inner;
        for (int64_t i = 0; i < inner; ++i) {
            const int64_t base = plane + i;

            // Pass 1 — the slice maximum, with the same strict `>` that
            // keeps a NaN from becoming the maximum. A NaN still poisons
            // the slice through the accumulation below (the honest IEEE
            // outcome), so nothing is special-cased here either.
            T maximum = src[base];
            for (int64_t k = 1; k < axis_length; ++k) {
                const T value = src[base + k * inner];
                if (value > maximum) {
                    maximum = value;
                }
            }

            // Pass 2 — the shifted logits go straight into the
            // destination while their exponentials are accumulated in the
            // element type. Every exponent is <= 0 for finite input, so a
            // large common offset cannot overflow, and the maximum
            // itself contributes exp(0) == 1, so ``sum_exp`` is >= 1 and
            // its logarithm is well defined. ``sum_exp`` is a ``T``: there
            // is deliberately no widened accumulator here (design §10.1).
            T sum_exp = T(0);
            for (int64_t k = 0; k < axis_length; ++k) {
                const T shifted = src[base + k * inner] - maximum;
                dst[base + k * inner] = shifted;
                sum_exp += std::exp(shifted);
            }

            // Pass 3 — subtract the log-normalizer in place. For a
            // length-1 slice this is exactly 0 - log(1) == 0; for equal
            // logits it is exactly -log(axis_length).
            const T log_denominator = std::log(sum_exp);
            for (int64_t k = 0; k < axis_length; ++k) {
                dst[base + k * inner] -= log_denominator;
            }
        }
    }
}

// Fused multi-class cross-entropy forward over a row-major CONTIGUOUS
// ``(batch_size, num_classes)`` logits block of element type ``T``, with
// the class axis fixed at axis 1. Element (n, c) lives at
// ``n * num_classes + c``.
//
// Per row: m = max_c x, then, in ONE pass, exp(x - m) is written straight
// into ``probabilities`` while sum_exp = Σ_c exp(x_c - m) accumulates;
// the row is normalized in place, and the per-example loss is
// ``log(sum_exp) - (x[target] - m)``. That is the fused maximum-shift /
// log-sum-exp form: it never computes ``-log(probabilities[target])``
// (which loses the whole point once a probability underflows to 0),
// never forms a public softmax or log-softmax first, clamps nothing, and
// inserts no epsilon. The per-example losses accumulate in deterministic
// batch order, **in the element type**; ``*loss`` receives that total for
// ``kCrossEntropyReductionSum`` and the total divided ONCE by
// ``batch_size`` for ``kCrossEntropyReductionMean`` (never by
// ``num_classes``), with the batch size converted to ``T`` once, outside
// the accumulation loop, at the single point the division happens.
//
// ``logits`` already points at the first element (the caller adds any
// storage offset); ``targets`` is a caller-owned span of ``batch_size``
// int64 class labels, every one already proved to be in
// ``[0, num_classes)`` by the guarded wrapper — targets are host int64
// metadata at every value dtype and take no part in the dtype dispatch;
// ``loss`` is one caller-allocated ``T``; ``probabilities`` is
// caller-allocated with at least ``batch_size * num_classes`` elements and
// is written in full.
//
// noexcept and allocation-free: the guarded exported wrapper does all
// validation, and this reads its inputs without mutating them.
// Deterministic traversal; no NaN/inf special-casing beyond plain IEEE
// arithmetic. ALLOCATES NOTHING and knows nothing about autograd.
template <class T>
inline void cross_entropy_forward_contiguous(
    const T* logits, const int64_t* targets,
    T* loss, T* probabilities,
    int64_t batch_size, int64_t num_classes,
    int64_t reduction_code) noexcept {
    // One row per example; the class axis is fixed at axis 1, so a row is
    // a contiguous run of ``num_classes`` elements. Fixed traversal order,
    // so the accumulation is reproducible run to run.
    T total = T(0);
    for (int64_t n = 0; n < batch_size; ++n) {
        const int64_t base = n * num_classes;

        // Pass 1 — the row maximum, with the same strict `>` the softmax
        // and log-softmax kernels use, so a NaN never becomes the
        // maximum. A NaN still poisons its row through the shift below,
        // which is the honest IEEE outcome rather than a special case.
        T maximum = logits[base];
        for (int64_t c = 1; c < num_classes; ++c) {
            const T value = logits[base + c];
            if (value > maximum) {
                maximum = value;
            }
        }

        // Pass 2 — shifted exponentials, written straight into the saved
        // probability destination (no second buffer is allocated) and
        // accumulated in the element type. Every exponent is <= 0 for
        // finite input, so a large common offset cannot overflow.
        T sum_exp = T(0);
        for (int64_t c = 0; c < num_classes; ++c) {
            const T shifted = std::exp(logits[base + c] - maximum);
            probabilities[base + c] = shifted;
            sum_exp += shifted;
        }

        // Pass 3 — normalize the row in place. For finite input
        // ``sum_exp`` is >= 1 (the maximum contributes exp(0) == 1), so
        // this never divides by zero.
        for (int64_t c = 0; c < num_classes; ++c) {
            probabilities[base + c] /= sum_exp;
        }

        // The per-example loss comes from the log-sum-exp directly:
        //     log(Σ exp(x - m)) - (x[target] - m)
        // NOT -log(probabilities[target]), which would first round a tiny
        // probability to 0 and then report an infinite loss. The target
        // index was proved in [0, num_classes) by the guarded wrapper.
        const T log_denominator = std::log(sum_exp);
        const T shifted_target = logits[base + targets[n]] - maximum;
        total += log_denominator - shifted_target;
    }

    // "sum" takes the accumulated total unscaled; "mean" divides ONCE by
    // the batch size — never by num_classes. The count is converted to the
    // element type at this single point, outside the loop above (§7.4's
    // "narrow once, before the loop" rule applied to a value that is not
    // even an ABI scalar): with T = double this is character for character
    // the pre-I6 expression.
    *loss = (reduction_code == kCrossEntropyReductionMean)
                ? total / static_cast<T>(batch_size)
                : total;
}

// Gradient of the fused cross-entropy with respect to its logits, from
// the SAVED probabilities alone — the logits are never reread and the
// softmax is never recomputed. The logits are not a parameter of this
// function, which is the structural half of that guarantee.
//
// For each (n, c): base = probabilities[n, c] - (c == targets[n] ? 1 : 0),
// divided once by ``batch_size`` for ``kCrossEntropyReductionMean``, then
// scaled by the single upstream value ``*upstream``. Every one of those
// steps happens in ``T``, in that order — the order is part of the
// contract, not an incidental detail, because reassociating it would
// change float32 rounding and could change float64 bits. The result goes
// into the caller-allocated contiguous ``(batch_size, num_classes)``
// ``grad_logits`` block, which is written in full.
//
// ``probabilities`` and ``upstream`` already point at their first
// elements (the caller adds any storage offset) and are read without
// mutation; ``targets`` is the same validated int64 span the forward
// used, host metadata at every value dtype. noexcept, allocation-free,
// deterministic, autograd-unaware.
template <class T>
inline void cross_entropy_backward_contiguous(
    const T* probabilities, const int64_t* targets,
    const T* upstream, T* grad_logits,
    int64_t batch_size, int64_t num_classes,
    int64_t reduction_code) noexcept {
    // The whole gradient comes from the SAVED probabilities and the
    // copied targets: no logit is reread, no maximum is recomputed, and
    // no exponential is evaluated here.
    const T scale = *upstream;
    const bool mean = (reduction_code == kCrossEntropyReductionMean);
    // Converted once, outside both loops, exactly as the forward converts
    // its mean divisor once.
    const T count = static_cast<T>(batch_size);
    for (int64_t n = 0; n < batch_size; ++n) {
        const int64_t base = n * num_classes;
        const int64_t target = targets[n];
        for (int64_t c = 0; c < num_classes; ++c) {
            // d(loss_n)/d(x[n, c]) = p[n, c] - [c == target]
            T contribution = probabilities[base + c];
            if (c == target) {
                contribution -= T(1);
            }
            if (mean) {
                contribution /= count;
            }
            grad_logits[base + c] = scale * contribution;
        }
    }
}

}  // namespace tf
