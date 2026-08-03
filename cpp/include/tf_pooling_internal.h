// Internal (non-ABI) definitions of the MaxPool2d compute kernels
// (Phase D, milestones D8/D9). See docs/native_cnn_design.md §10-§12, §14.
//
// Like the Conv2d compute kernels these are deliberately NOT part of the
// public C ABI: plain C++ functions in ``namespace tf`` with hidden
// visibility, holding only the contiguous CPU window-maximum arithmetic,
// the winner recording, and the backward scatter. They perform no
// allocation, no error reporting, and no mutation of their inputs. The
// exported, guarded ``tf_core_maxpool2d_*`` wrappers live in
// cpp/src/pooling.cpp alongside them; the Python/Core validation and the
// output/winner allocation live in backends/cpp.py.
//
// Because the shared library exports only TF_EXPORT symbols, these
// internal functions are not reachable from a separately linked test
// binary — so the pooling tests (cpp/tests/test_maxpool2d_*.cpp and
// cpp/tests/test_dtype_cnn.cpp) compile cpp/src/pooling.cpp directly
// rather than linking the shared library.
//
// ---------------------------------------------------------------------------
// Phase I, milestone I5: the *value* path carries a scalar type parameter;
// the winner buffer does not.
// ---------------------------------------------------------------------------
//
// ``T`` is **deduced from the value pointer arguments** (the input and
// output in the forward; the upstream gradient and input gradient in the
// backward), so every pre-Phase-I call site — all of which pass ``double*``
// — instantiates ``T = double`` and is the pre-I5 kernel statement for
// statement. ``T = float`` is the same source at binary32: the window scan,
// the seeding, the strict ``>`` comparison, the NaN policy, the signed-zero
// behavior, and the first-winner tie rule are all comparison-and-select
// semantics that IEEE-754 defines identically at both widths.
//
// **The winner buffer stays ``double*`` at every value dtype.** This is the
// locked §13.3 decision of docs/native_dtype_float32_design.md: winners are
// flat plane offsets encoded as floating-point values, exact only while the
// plane fits the mantissa — 2**53 for binary64 against only 2**24 for
// binary32 — so a winner buffer that followed the tensor's dtype would
// silently cut the largest poolable plane from ~9.0e15 elements to
// 16,777,216. The winner is private index *metadata*, never a numeric
// operand, so a float64 winner beside float32 values is not a mixed-dtype
// operation; the exported wrappers validate the value handles for
// agreement and the winner handle for float64, separately, and dispatch on
// the value dtype alone. There is no winner dtype dispatch anywhere.
#pragma once

#include <cstdint>
#include <limits>

namespace tf {

// Row-major flat offset into a 4-D array of extents (·, dim1, dim2, dim3)
// at index (i0, i1, i2, i3). Moved here from pooling.cpp's anonymous
// namespace with the kernels themselves (I5); metadata arithmetic over
// ``int64`` with no dtype. Prefixed ``pool_`` so it cannot collide with
// another compute unit's helper in a test binary that compiles several
// kernel sources together. Both the NCHW input and the NCHW output/winner
// buffers share this layout.
inline int64_t pool_index4d(
    int64_t i0, int64_t i1, int64_t i2, int64_t i3,
    int64_t dim1, int64_t dim2, int64_t dim3) noexcept {
    return ((i0 * dim1 + i1) * dim2 + i2) * dim3 + i3;
}

// Two-dimensional max pooling, CPU float32/float64, direct nested loops.
// Writes the pooled output *and* the parallel winner buffer in a single
// pass, so the forward value and the saved winner always come from the
// same selected candidate (docs/native_cnn_design.md §10).
//
//   out[n, c, i, j]     = max over the kh x kw window of the padded plane
//   winners[n, c, i, j] = flat offset (ih * input_width + iw) of the
//                         selected input cell, or -1.0 when the selected
//                         cell is padding
//
// Window scan order is row-major over the kernel (``p`` outer, ``q``
// inner). Out-of-bounds (padded) positions conceptually hold -inf and
// *participate* in the selection — they are not skipped — so a padded
// cell can win only when every in-bounds candidate is -inf too and the
// padded position comes first in row-major order. Updates use a strict
// ``>`` comparison, so the FIRST occurrence of the maximum wins every tie
// (matching the stable framework's ``argmax``). ``-inf`` here is the
// element type's own infinity, so a padded candidate loses to every finite
// value at either width.
//
// NaN policy (deliberate, documented, outside the supported contract —
// docs/native_cnn_design.md §10): a NaN candidate never wins, so the
// first *non-NaN* candidate seeds the scan and later NaNs cannot displace
// it. If a window has no non-NaN candidate at all (only possible with an
// unpadded window whose every input value is NaN, since padding is -inf),
// the deterministic fallback is the FIRST candidate of the window — its
// value and its winner — so output and winner still agree. No parity with
// NumPy's NaN-propagating argmax is claimed. The policy is the same
// comparison sequence at both widths, so a float32 pool and a float64 pool
// of the same values select the same winner.
//
// Layouts (all row-major contiguous; input/output in the element type,
// winners **always float64**):
//   input   : NCHW  (batch, channels, input_height, input_width)
//   output  : NCHW  (batch, channels, output_height, output_width) — written
//   winners : NCHW  (batch, channels, output_height, output_width) — written
//
// Every output and winner element is fully defined by this routine (each
// window always contains at least one candidate), so the caller need not
// pre-initialize either buffer.
//
// Winner value domain: -1.0, or an exactly representable non-negative
// integral offset in [0, input_height*input_width - 1]. Exactness is the
// caller's precondition (``input_height * input_width <= 2^53``, validated
// in Python and re-checked at the ABI boundary); no rounded or fractional
// winner is ever written. The bound is float64's and does not shrink when
// the values are float32, because the winner buffer does not follow the
// value dtype.
//
// Preconditions (guaranteed by the exported wrapper and the Core layer;
// NOT re-validated here — this routine is the inner math, not a
// validation boundary):
//   * input / output / winners are non-null and each point to contiguous
//     storage of exactly batch*channels*input_height*input_width and
//     batch*channels*output_height*output_width elements respectively;
//   * every dimension is positive; kernel/stride >= 1; padding >= 0;
//   * output_height / output_width equal the floor formula
//       floor((input_dim + 2*pad - kernel) / stride) + 1;
//   * all integer products and offsets are representable in int64.
//
// The routine allocates no heap memory and cannot throw (noexcept): it is
// pure arithmetic over caller-owned buffers. It reads the input without
// modifying it and writes only inside the output and winner spans.
template <class T>
inline void maxpool2d_forward_contiguous(
    const T* input,
    T* output,
    double* winners,
    int64_t batch,
    int64_t channels,
    int64_t input_height,
    int64_t input_width,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) noexcept {
    // Direct nested loops, correctness first: n, c, i, j outer; p, q inner
    // (row-major over the kernel window). Padded coordinates are computed
    // in *signed* int64 so an out-of-bounds position is genuinely negative,
    // never an unsigned wrap; no padded copy of the input is materialized.
    const T negative_infinity = -std::numeric_limits<T>::infinity();
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t i = 0; i < output_height; ++i) {
                for (int64_t j = 0; j < output_width; ++j) {
                    // best_* holds the current selection. ``seen_any`` marks
                    // the fallback anchor (the window's first candidate,
                    // used only when every candidate is NaN); ``seen_number``
                    // marks the first non-NaN candidate, which seeds the
                    // real scan. Afterwards only a strictly greater value
                    // replaces the selection, so ties keep the first
                    // occurrence in row-major window order.
                    T best_value = T(0);
                    double best_winner = -1.0;
                    bool seen_any = false;
                    bool seen_number = false;
                    for (int64_t p = 0; p < kernel_height; ++p) {
                        const int64_t ih = i * stride_height + p - pad_height;
                        const bool row_inside =
                            (ih >= 0 && ih < input_height);
                        for (int64_t q = 0; q < kernel_width; ++q) {
                            const int64_t iw = j * stride_width + q - pad_width;
                            const bool inside =
                                row_inside && iw >= 0 && iw < input_width;
                            // A padded position conceptually holds -inf and
                            // participates in the selection with winner -1.
                            const T candidate =
                                inside
                                    ? input[pool_index4d(n, c, ih, iw, channels,
                                                         input_height,
                                                         input_width)]
                                    : negative_infinity;
                            const double candidate_winner =
                                inside
                                    ? static_cast<double>(ih * input_width + iw)
                                    : -1.0;
                            if (!seen_any) {
                                // Deterministic all-NaN fallback anchor.
                                seen_any = true;
                                best_value = candidate;
                                best_winner = candidate_winner;
                            }
                            if (candidate != candidate) {
                                continue;  // NaN never wins (see above)
                            }
                            if (!seen_number) {
                                seen_number = true;
                                best_value = candidate;
                                best_winner = candidate_winner;
                            } else if (candidate > best_value) {
                                // Strict >: an equal later candidate never
                                // displaces the earlier winner.
                                best_value = candidate;
                                best_winner = candidate_winner;
                            }
                        }
                    }
                    const int64_t out_index = pool_index4d(
                        n, c, i, j, channels, output_height, output_width);
                    output[out_index] = best_value;
                    winners[out_index] = best_winner;
                }
            }
        }
    }
}

// Gradient of the MaxPool2d forward with respect to its input (Phase D,
// milestone D9). The scatter-add that routes each upstream value to the
// exact input element its window selected in the forward; pure arithmetic
// only. The exported, guarded ``tf_core_maxpool2d_backward`` wrapper
// (which is where winner values are validated), its ctypes registration,
// the NativeTensorCore backward method, and the ``NativeTensor.maxpool2d``
// autograd node live elsewhere.
//
// For every output position, using ONLY the saved winner (the input value
// is never reread and the window maximum is never recomputed):
//
//   winner = winners[n, c, oh, ow]
//   if winner == -1:  the window's maximum was padding -> drop the gradient
//   else:             ih = winner / input_width, iw = winner % input_width
//                     grad_input[n, c, ih, iw] += grad_output[n, c, oh, ow]
//
// Overlapping windows (stride < kernel) can select the same input element
// from several output positions, so the accumulation is a genuine ``+=``
// **in the element type**: a float32 gradient accumulates in binary32 with
// no widening intermediate (design §10.1), in the same deterministic
// n -> c -> oh -> ow order at both widths. Ties were already resolved at
// forward time: the single recorded winner receives the whole window's
// gradient and the equal-valued cells receive nothing, matching the stable
// framework.
//
// Layouts (all row-major contiguous; gradients in the element type,
// winners **always float64**):
//   grad_output : NCHW  (batch, channels, output_height, output_width)
//   winners     : NCHW  (batch, channels, output_height, output_width)
//   grad_input  : NCHW  (batch, channels, input_height, input_width) — written
//
// Output initialization: the routine **zero-initializes the entire
// grad_input span itself** before accumulating, so the caller need NOT
// pre-zero it; any prior contents are fully overwritten/defined. Input
// elements that won no window keep their zero.
//
// Preconditions (guaranteed by the exported wrapper; NOT re-validated here
// — this routine is the inner math, not a validation boundary):
//   * grad_output / winners / grad_input are non-null and each point to
//     contiguous storage of exactly
//     batch*channels*output_height*output_width (the first two) and
//     batch*channels*input_height*input_width elements;
//   * every dimension is positive;
//   * every winner is either exactly -1.0 or an exact integral value in
//     [0, input_height*input_width - 1] — the wrapper checks each one, so
//     this routine converts without rounding and never scatters out of
//     range;
//   * all integer products and offsets are representable in int64.
//
// The routine allocates no heap memory and cannot throw (noexcept): it is
// pure arithmetic over caller-owned buffers. It reads grad_output and
// winners without modifying them and writes only inside the grad_input
// span, in deterministic n -> c -> oh -> ow order.
template <class T>
inline void maxpool2d_backward_contiguous(
    const T* grad_output,
    const double* winners,
    T* grad_input,
    int64_t batch,
    int64_t channels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) noexcept {
    // Scatter-add adjoint of the forward window maximum
    // (docs/native_cnn_design.md §11). The output is fully defined here:
    // zero the entire grad_input span first, so the caller need not
    // pre-zero it and overlapping windows can accumulate with a plain +=.
    const int64_t plane = input_height * input_width;
    const int64_t input_count = batch * channels * plane;
    for (int64_t idx = 0; idx < input_count; ++idx) {
        grad_input[idx] = T(0);
    }
    // Deterministic n -> c -> oh -> ow order. Backward reads ONLY the saved
    // winners and the upstream: no input value is reread and no window
    // maximum is recomputed, so forward and backward can never disagree.
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            // Base of this (n, c) plane in both buffers.
            const int64_t input_base = (n * channels + c) * plane;
            for (int64_t i = 0; i < output_height; ++i) {
                for (int64_t j = 0; j < output_width; ++j) {
                    const int64_t out_index = pool_index4d(
                        n, c, i, j, channels, output_height, output_width);
                    const double winner = winners[out_index];
                    if (winner < 0.0) {
                        continue;  // -1 sentinel: padding won, drop the grad
                    }
                    // The wrapper proved this is an exact integer in
                    // [0, plane - 1], so the conversion truncates nothing.
                    const int64_t offset = static_cast<int64_t>(winner);
                    grad_input[input_base + offset] += grad_output[out_index];
                }
            }
        }
    }
}

}  // namespace tf
