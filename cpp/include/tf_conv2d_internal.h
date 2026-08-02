// Internal (non-ABI) definitions of the Conv2d compute kernels
// (Phase D, milestones D2/D4/D5). See docs/native_cnn_design.md §6 and §14.
//
// These are deliberately NOT part of the public C ABI: plain C++ functions
// in ``namespace tf`` with hidden visibility, holding only the contiguous
// CPU cross-correlation arithmetic. They perform no allocation, no error
// reporting, and no mutation of their inputs. The exported
// ``tf_core_conv2d_*`` wrappers (TF_GUARD error contract + Storage
// handles) live in cpp/src/conv2d.cpp alongside the Python/Core validation
// and output allocation; none of that lives here.
//
// Because the shared library exports only TF_EXPORT symbols, these
// internal functions are not reachable from a separately linked test
// binary — so the Conv2d tests (cpp/tests/test_conv2d_*.cpp and
// cpp/tests/test_dtype_cnn.cpp) compile cpp/src/conv2d.cpp directly rather
// than linking the shared library.
//
// ---------------------------------------------------------------------------
// Phase H, milestone H9 — two compute paths per direction, one export each.
//
// Each of the three ``*_contiguous`` entry points below is a *dispatcher*
// that picks between the retained Phase-D direct loop (the generic reference
// path, still reachable for every supported geometry) and one H9 optimized
// traversal, from the integer geometry it already receives. This reuses the
// dispatch shape H2 (matmul), H5 (copy), H6 (reduction), and H8 (elementwise)
// each proved: no new exported C ABI symbol, no new translation unit, and no
// public control of any kind — no path selector, block-size setter, dispatch
// tracer, profiling counter, or environment-variable mode.
//
// The predicates are total, pure, allocation-free, and functions of the
// integer geometry alone — never of a pointer value, an alignment, a clock,
// an environment variable, or a CPU-feature probe. A false answer selects the
// generic path and is **never an error**.
//
// **Per-destination accumulation order is preserved exactly** on both paths,
// for all three directions; see the proof comments on each optimized kernel.
//
// ---------------------------------------------------------------------------
// Phase I, milestone I5: every kernel carries a scalar type parameter
// ---------------------------------------------------------------------------
//
// ``T`` is **deduced from the pointer arguments**, so every pre-Phase-I call
// site — all of which pass ``double*`` — instantiates ``T = double`` and is
// the pre-I5 kernel statement for statement. ``T = float`` is the same
// source at binary32.
//
// One traversal, two instantiations, is the whole point: the dtypes cannot
// drift apart because there is nothing separate to drift, and float64 keeps
// running the code Phase H measured (H9's traversals, predicates, and
// per-destination accumulation orders are untouched). The definitions moved
// into this header for the ordinary reason a template must — so both
// instantiations are available to the exported wrappers in conv2d.cpp *and*
// to the CTests that compile that file directly — and no loop nest, no tap
// range, no seed, and no accumulator changed in the move.
//
// **The accumulator type is exactly ``T``** (design §10.1). A float32
// convolution loads ``float``, multiplies in ``float``, accumulates every
// partial sum as binary32, and stores ``float``; there is no widening
// intermediate, no double accumulator, no FMA, and no reassociation. A sum
// of kernel taps is genuinely observable against "accumulate in binary64
// and round once" — witnessed directly by test in all three directions.
#pragma once

#include <cstdint>

namespace tf {

// The minimum swept extent an H9 optimized traversal requires. Below this the
// swept inner loop is too short to pay for its own setup and the retained
// generic path is genuinely faster — measured, not assumed: at a swept extent
// of 1 the optimized forms ran 0.57x-0.93x, at 2 they ran 1.04x-1.38x, and at
// 4 they ran 1.91x-2.40x. Four is the smallest extent at which all three
// directions are clearly ahead. It is a structural minimum on the loop, not a
// tuned constant: it does not vary with the machine, the cache, or the build.
inline constexpr int64_t kConv2dMinSweptExtent = 4;

// All three optimized paths replace a kernel-tap inner loop with a sweep
// along one spatial row, and in every direction that sweep is bounded by
// **both** widths at once: the run of output columns it can cover is limited
// by output_width, and the run of input columns those map to is limited by
// input_width. So ``min(input_width, output_width)`` is the one honest bound
// on all three inner loops, and one shared rule keeps the three predicates
// from drifting apart. (Keying the input gradient on input_width alone was
// measured wrong: a 5-wide input with a 1-wide output sweeps a single element
// and ran 0.73x.)
bool conv2d_sweep_extent_is_worthwhile(
    int64_t input_width, int64_t output_width) noexcept;

// Does the forward take the H9 row sweep?
bool conv2d_forward_prefers_row_sweep(
    int64_t input_width, int64_t output_width) noexcept;

// Does the input gradient take the H9 gather? Besides the shared extent rule
// it requires **unit stride in both axes**: it visits the taps contributing to
// one destination by walking the kernel offsets downward, which reproduces
// the reference's ascending output-position order only when each output
// position maps to a source position one-for-one (see the kernel comment).
bool conv2d_input_backward_prefers_gather(
    int64_t stride_height, int64_t stride_width,
    int64_t input_width, int64_t output_width) noexcept;

// Does the weight gradient take the H9 gather?
bool conv2d_weight_backward_prefers_gather(
    int64_t input_width, int64_t output_width) noexcept;

// ---------------------------------------------------------------------------
// Shared indexing helpers. These moved here from conv2d.cpp's anonymous
// namespace with the kernels themselves (I5); they are metadata arithmetic
// over ``int64`` and have no dtype. Prefixed ``conv2d_`` so no compute
// translation unit's own file-local helper can collide with them.
// ---------------------------------------------------------------------------

// Row-major flat offset into a 4-D array of extents (·, dim1, dim2, dim3)
// at index (i0, i1, i2, i3). Never part of the ABI; the leading extent is
// not needed for the offset, so it is not a parameter. One helper serves
// all the arrays the kernels touch, which share this layout: the NCHW
// input (dim1=in_channels, dim2=input_height, dim3=input_width), the OIHW
// weight (dim1=in_channels, dim2=kernel_height, dim3=kernel_width), and
// the NCHW output (dim1=out_channels, dim2=output_height, dim3=output_width).
// int64_t arithmetic throughout, matching the rest of the native runtime.
inline int64_t conv2d_index4d(
    int64_t i0, int64_t i1, int64_t i2, int64_t i3,
    int64_t dim1, int64_t dim2, int64_t dim3) noexcept {
    return ((i0 * dim1 + i1) * dim2 + i2) * dim3 + i3;
}

// -- H9 tap-range helpers ---------------------------------------------------
//
// The taps a padded convolution skips are exactly the ones whose source
// coordinate leaves the real input, and because that coordinate advances by a
// fixed +1 per kernel step, the *kept* taps always form one contiguous run.
// These two helpers compute that run's half-open bounds instead of testing
// each candidate — the generic path's ``continue`` and the optimized paths'
// loop bounds therefore skip **the identical set of taps in the identical
// order**, which is what makes the optimized traversals order-preserving
// rather than merely equivalent.
//
// Both take non-negative extents and a stride >= 1, return a half-open range
// clamped into [0, limit], and return an empty range (lo >= hi) when no
// position is valid. All arithmetic is signed int64 so an out-of-bounds
// position is genuinely negative, never an unsigned wrap.

// First index t >= 0 with ``t * stride + offset - pad >= 0``, i.e. the ceiling
// of (pad - offset) / stride, clamped at 0.
inline int64_t conv2d_tap_begin(
    int64_t offset, int64_t pad, int64_t stride) noexcept {
    const int64_t need = pad - offset;
    if (need <= 0) {
        return 0;
    }
    return (need + stride - 1) / stride;  // ceil, both operands positive
}

// One past the last index t with ``t * stride + offset - pad <= size - 1``,
// clamped into [0, limit]. A negative numerator means no position is valid at
// all; it is turned into 0 explicitly rather than left to C++ division's
// truncation-toward-zero, so the empty case is stated rather than implied.
inline int64_t conv2d_tap_end(
    int64_t offset, int64_t pad, int64_t stride,
    int64_t size, int64_t limit) noexcept {
    const int64_t room = size - 1 + pad - offset;
    if (room < 0) {
        return 0;
    }
    const int64_t end = room / stride + 1;
    return (end < limit) ? end : limit;
}

// ---------------------------------------------------------------------------
// The retained generic reference paths (Phase D, kept verbatim by H9 and
// templated by I5). These are what every geometry that fails a predicate
// runs, and they remain the oracle every optimized result is compared
// against — per dtype.
// ---------------------------------------------------------------------------

#define TF_CONV2D_KERNEL_PARAMS                                              \
    int64_t batch, int64_t in_channels, int64_t input_height,                \
    int64_t input_width, int64_t out_channels, int64_t kernel_height,        \
    int64_t kernel_width, int64_t stride_height, int64_t stride_width,       \
    int64_t pad_height, int64_t pad_width, int64_t output_height,            \
    int64_t output_width

// Two-dimensional cross-correlation (NOT flipped convolution), CPU
// float32/float64, direct nested loops. Matches the stable
// tensorforge.nn.Conv2d numerically (to a floating-point tolerance; the
// summation order is deterministic but not guaranteed bit-identical to
// NumPy's einsum).
//
//   out[n, o, i, j] = (bias ? bias[o] : 0)
//                   + sum_{c, p, q} in[n, c, i*sh + p - ph, j*sw + q - pw]
//                                     * weight[o, c, p, q]
//
// with symmetric zero padding applied by skipping source coordinates that
// fall outside the real input (a padded cell contributes 0). Bias, when
// present, is added exactly once per output element. Accumulation runs in
// deterministic c -> p -> q order into an accumulator of the element type.
//
// Layouts (all row-major contiguous, every buffer the same element type):
//   input  : NCHW  (batch, in_channels, input_height, input_width)
//   weight : OIHW  (out_channels, in_channels, kernel_height, kernel_width)
//   bias   : (out_channels,), or nullptr for no bias
//   output : NCHW  (batch, out_channels, output_height, output_width) — written
//
// Preconditions (guaranteed by the exported wrapper; NOT re-validated
// here — this routine is the inner math, not a validation boundary):
//   * input / weight / output are non-null and each point to contiguous
//     storage of exactly batch*in_channels*input_height*input_width,
//     out_channels*in_channels*kernel_height*kernel_width, and
//     batch*out_channels*output_height*output_width elements;
//   * bias is nullptr or points to out_channels elements;
//   * every dimension is positive; stride/kernel >= 1; padding >= 0;
//   * output_height / output_width equal the floor formula
//       floor((input_dim + 2*pad - kernel) / stride) + 1
//     so every output coordinate maps inside the padded grid.
//
// The routine allocates no heap memory and cannot throw (noexcept): it is
// pure arithmetic over caller-owned buffers.
template <class T>
inline void conv2d_forward_generic(
    const T* input, const T* weight, const T* bias, T* output,
    TF_CONV2D_KERNEL_PARAMS) noexcept {
    // Direct nested loops, correctness first (docs/native_cnn_design.md §6):
    // n, o, i, j outer; c, p, q inner; deterministic c -> p -> q sum order.
    // Padded coordinates are computed in *signed* int64 so an out-of-bounds
    // position is genuinely negative (a skip), never an unsigned wrap.
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t o = 0; o < out_channels; ++o) {
            // Bias is added exactly once per output element, as the seed of
            // the accumulator (T(0) when no bias pointer was supplied).
            const T bias_o = (bias != nullptr) ? bias[o] : T(0);
            for (int64_t i = 0; i < output_height; ++i) {
                for (int64_t j = 0; j < output_width; ++j) {
                    T acc = bias_o;
                    for (int64_t c = 0; c < in_channels; ++c) {
                        for (int64_t p = 0; p < kernel_height; ++p) {
                            const int64_t ih =
                                i * stride_height + p - pad_height;
                            if (ih < 0 || ih >= input_height) {
                                continue;  // whole kernel row is padding
                            }
                            for (int64_t q = 0; q < kernel_width; ++q) {
                                const int64_t iw =
                                    j * stride_width + q - pad_width;
                                if (iw < 0 || iw >= input_width) {
                                    continue;  // padded column: contributes 0
                                }
                                acc += input[conv2d_index4d(
                                           n, c, ih, iw,
                                           in_channels, input_height,
                                           input_width)]
                                     * weight[conv2d_index4d(
                                           o, c, p, q,
                                           in_channels, kernel_height,
                                           kernel_width)];
                            }
                        }
                    }
                    output[conv2d_index4d(
                        n, o, i, j,
                        out_channels, output_height, output_width)] = acc;
                }
            }
        }
    }
}

// Gradient of the Conv2d forward with respect to its input (Phase D, D4).
// The scatter-add adjoint of the cross-correlation above; pure arithmetic
// only. Given the forward relation
//   out[n,o,i,j] = bias[o] + sum_{c,p,q} in[n,c, i*sh+p-ph, j*sw+q-pw]
//                                          * weight[o,c,p,q],
// the input gradient scatters each upstream value back through the same
// coordinate map:
//   grad_input[n,c, i*sh+p-ph, j*sw+q-pw] += grad_output[n,o,i,j]
//                                            * weight[o,c,p,q]
// summed over o,i,j (and the kernel taps c,p,q), with padded source
// coordinates that fall outside the real input skipped (the gradient that
// would land on the zero pad border is discarded, matching the stable
// framework). Overlapping windows accumulate through ``+=``. Bias does not
// affect the input gradient, so this kernel neither receives nor reads a
// bias. Accumulation runs in deterministic n -> o -> i -> j -> c -> p -> q
// order into the output span, in the element type.
//
// Layouts (all row-major contiguous, every buffer the same element type):
//   grad_output : NCHW  (batch, out_channels, output_height, output_width)
//   weight      : OIHW  (out_channels, in_channels, kernel_height, kernel_width)
//   grad_input  : NCHW  (batch, in_channels, input_height, input_width) — written
//
// Output initialization: the routine **zero-initializes the entire
// grad_input span itself** before accumulating, so the caller need NOT
// pre-zero it; any prior contents are fully overwritten/defined.
//
// Preconditions: as the forward kernel's, with the operand roles above.
// Allocates nothing and cannot throw (noexcept); reads grad_output and
// weight without modifying them and writes only inside the grad_input span.
template <class T>
inline void conv2d_input_backward_generic(
    const T* grad_output, const T* weight, T* grad_input,
    TF_CONV2D_KERNEL_PARAMS) noexcept {
    // Scatter-add adjoint of the forward cross-correlation
    // (docs/native_cnn_design.md §7.1). The output is fully defined here:
    // zero the entire grad_input span first, so the caller need not
    // pre-zero it and overlapping windows can accumulate with a plain +=.
    const int64_t input_count =
        batch * in_channels * input_height * input_width;
    for (int64_t idx = 0; idx < input_count; ++idx) {
        grad_input[idx] = T(0);
    }
    // Deterministic n -> o -> i -> j -> c -> p -> q order. The upstream
    // value g at (n, o, i, j) is hoisted out of the kernel-tap loops; the
    // padded source coordinate is computed in *signed* int64 so an
    // out-of-bounds (pad) position is genuinely negative and skipped, never
    // an unsigned wrap.
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t o = 0; o < out_channels; ++o) {
            for (int64_t i = 0; i < output_height; ++i) {
                for (int64_t j = 0; j < output_width; ++j) {
                    const T g = grad_output[conv2d_index4d(
                        n, o, i, j,
                        out_channels, output_height, output_width)];
                    for (int64_t c = 0; c < in_channels; ++c) {
                        for (int64_t p = 0; p < kernel_height; ++p) {
                            const int64_t ih =
                                i * stride_height + p - pad_height;
                            if (ih < 0 || ih >= input_height) {
                                continue;  // whole kernel row is padding
                            }
                            for (int64_t q = 0; q < kernel_width; ++q) {
                                const int64_t iw =
                                    j * stride_width + q - pad_width;
                                if (iw < 0 || iw >= input_width) {
                                    continue;  // padded column: no input cell
                                }
                                grad_input[conv2d_index4d(
                                    n, c, ih, iw,
                                    in_channels, input_height, input_width)]
                                    += g * weight[conv2d_index4d(
                                           o, c, p, q,
                                           in_channels, kernel_height,
                                           kernel_width)];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Gradient of the Conv2d forward with respect to its weight (Phase D, D5).
// Pairs each upstream value with the input pixel it multiplied in the
// forward; pure arithmetic only. Given the forward relation above, the
// weight gradient accumulates, for every valid forward contribution,
//   grad_weight[o,c,p,q] += grad_output[n,o,i,j]
//                           * in[n,c, i*sh+p-ph, j*sw+q-pw]
// summed over n, i, j (matching the stable einsum "no,nckl->ockl"), with
// padded source coordinates that fall outside the real input skipped (they
// contributed 0 in the forward, so they contribute 0 to the gradient — the
// pad border never accumulates into grad_weight). Accumulation runs in
// deterministic n -> o -> i -> j -> c -> p -> q order into the output span,
// in the element type.
//
// Layouts (all row-major contiguous, every buffer the same element type):
//   grad_output : NCHW  (batch, out_channels, output_height, output_width)
//   input       : NCHW  (batch, in_channels, input_height, input_width)
//   grad_weight : OIHW  (out_channels, in_channels, kernel_height, kernel_width) — written
//
// Output initialization: the routine **zero-initializes the entire
// grad_weight span itself** before accumulating, so the caller need NOT
// pre-zero it; any prior contents are fully overwritten/defined.
//
// Bias does not appear in the weight gradient, so this kernel neither
// receives nor reads a bias, and grad_weight is independent of whether the
// forward used a bias.
//
// Preconditions: as the forward kernel's, with the operand roles above.
// Allocates nothing and cannot throw (noexcept); reads grad_output and
// input without modifying them and writes only inside the grad_weight span.
template <class T>
inline void conv2d_weight_backward_generic(
    const T* grad_output, const T* input, T* grad_weight,
    TF_CONV2D_KERNEL_PARAMS) noexcept {
    // Pairs each upstream value with the input pixel it multiplied in the
    // forward (docs/native_cnn_design.md §7.2). The output is fully defined
    // here: zero the entire grad_weight span first, so the caller need not
    // pre-zero it and every (n, i, j) accumulates with a plain +=.
    const int64_t weight_count =
        out_channels * in_channels * kernel_height * kernel_width;
    for (int64_t idx = 0; idx < weight_count; ++idx) {
        grad_weight[idx] = T(0);
    }
    // Deterministic n -> o -> i -> j -> c -> p -> q order. The upstream value
    // g at (n, o, i, j) is hoisted out of the kernel-tap loops; the padded
    // source coordinate is computed in *signed* int64 so an out-of-bounds
    // (pad) position is genuinely negative and skipped, never an unsigned
    // wrap — a padded tap contributed 0 in the forward, so it adds 0 here.
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t o = 0; o < out_channels; ++o) {
            for (int64_t i = 0; i < output_height; ++i) {
                for (int64_t j = 0; j < output_width; ++j) {
                    const T g = grad_output[conv2d_index4d(
                        n, o, i, j,
                        out_channels, output_height, output_width)];
                    for (int64_t c = 0; c < in_channels; ++c) {
                        for (int64_t p = 0; p < kernel_height; ++p) {
                            const int64_t ih =
                                i * stride_height + p - pad_height;
                            if (ih < 0 || ih >= input_height) {
                                continue;  // whole kernel row is padding
                            }
                            for (int64_t q = 0; q < kernel_width; ++q) {
                                const int64_t iw =
                                    j * stride_width + q - pad_width;
                                if (iw < 0 || iw >= input_width) {
                                    continue;  // padded column: no input cell
                                }
                                grad_weight[conv2d_index4d(
                                    o, c, p, q,
                                    in_channels, kernel_height, kernel_width)]
                                    += g * input[conv2d_index4d(
                                           n, c, ih, iw,
                                           in_channels, input_height,
                                           input_width)];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// H9 optimized traversals.
//
// All three replace a short inner loop over kernel taps with a long inner
// loop over one contiguous *spatial row*, and all three keep every
// destination's contribution sequence exactly as the generic path produces
// it. Nothing is reassociated, no partial sums are combined, no accumulator
// width changes, and there is no FMA, fast-math, tree/pairwise reduction,
// SIMD intrinsic, threading, or parallel accumulation anywhere — at either
// element width.
// ---------------------------------------------------------------------------

template <class T>
inline void conv2d_forward_row_sweep(
    const T* input, const T* weight, const T* bias, T* output,
    TF_CONV2D_KERNEL_PARAMS) noexcept {
    // Loop interchange: the generic path is n, o, i, j | c, p, q with a
    // register accumulator; this is n, o, i | c, p, q | j accumulating into
    // the output row. j moves from the innermost-but-three position to the
    // innermost one, so the inner loop walks a whole output row (and, at unit
    // stride, a whole input row) instead of kernel_width kernel taps.
    //
    // ACCUMULATION-ORDER PROOF. Fix one destination (n, o, i, j). The generic
    // path seeds it with the bias, then adds the taps in ascending c, then p,
    // then q, skipping a tap exactly when its source coordinate leaves the
    // real input. Here the same destination is seeded with the same bias in
    // the priming loop below, and thereafter c, p, q are *outer* to j — so the
    // contributions reaching that one destination still arrive in ascending
    // c, then p, then q. A tap is skipped exactly when j falls outside
    // [tap_begin, tap_end) for its q, which is precisely the condition
    // ``0 <= j*stride_width + q - pad_width < input_width`` the generic path
    // tests. Same seed, same taps, same order — bit-identical by
    // construction, on every finite value and every special value alike, at
    // either element width.
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t o = 0; o < out_channels; ++o) {
            const T bias_o = (bias != nullptr) ? bias[o] : T(0);
            for (int64_t i = 0; i < output_height; ++i) {
                T* out_row = output + conv2d_index4d(
                    n, o, i, 0, out_channels, output_height, output_width);
                // Prime the whole row with the bias seed. This both starts
                // every accumulation from the generic path's initial value
                // and writes every element of the row, so the destination is
                // never read before it is written (H1: the caller may hand
                // this kernel uninitialized storage).
                for (int64_t j = 0; j < output_width; ++j) {
                    out_row[j] = bias_o;
                }
                // Kernel rows whose source row lies inside the real input.
                const int64_t p_begin =
                    conv2d_tap_begin(i * stride_height, pad_height, 1);
                const int64_t p_end = conv2d_tap_end(
                    i * stride_height, pad_height, 1, input_height,
                    kernel_height);
                for (int64_t c = 0; c < in_channels; ++c) {
                    for (int64_t p = p_begin; p < p_end; ++p) {
                        const int64_t ih = i * stride_height + p - pad_height;
                        const T* in_row = input + conv2d_index4d(
                            n, c, ih, 0, in_channels, input_height,
                            input_width);
                        const T* w_row = weight + conv2d_index4d(
                            o, c, p, 0, in_channels, kernel_height,
                            kernel_width);
                        for (int64_t q = 0; q < kernel_width; ++q) {
                            // Output columns whose source column lies inside
                            // the real input, for this kernel column.
                            const int64_t j_begin =
                                conv2d_tap_begin(q, pad_width, stride_width);
                            const int64_t j_end = conv2d_tap_end(
                                q, pad_width, stride_width, input_width,
                                output_width);
                            if (j_begin >= j_end) {
                                continue;  // this kernel column is all padding
                            }
                            const T w_value = w_row[q];
                            // Bases taken at the first *valid* position, so no
                            // pointer is ever formed outside its array even
                            // when the padding would place one there.
                            T* dst = out_row + j_begin;
                            const T* src = in_row
                                + j_begin * stride_width + q - pad_width;
                            const int64_t span = j_end - j_begin;
                            if (stride_width == 1) {
                                for (int64_t t = 0; t < span; ++t) {
                                    dst[t] += src[t] * w_value;
                                }
                            } else {
                                for (int64_t t = 0; t < span; ++t) {
                                    dst[t] += src[t * stride_width] * w_value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <class T>
inline void conv2d_input_backward_gather(
    const T* grad_output, const T* weight, T* grad_input,
    TF_CONV2D_KERNEL_PARAMS) noexcept {
    // The generic path *scatters*: it walks upstream positions and adds into
    // whichever grad_input cell each tap lands on, so one destination is
    // revisited from far-apart iterations. This path *gathers*: it walks
    // grad_input rows and pulls in every contribution that belongs to them,
    // so a destination row is written by one contiguous inner loop.
    //
    // Preconditions beyond the shared ones: unit stride in both axes, which
    // the dispatch predicate guarantees.
    //
    // ACCUMULATION-ORDER PROOF. Fix one destination (n, c, ih, iw). The
    // generic path visits its contributions in ascending o, then i, then j
    // (its n, o, i, j loops are outer to the c, p, q taps, and for a fixed
    // destination each (o, i, j) supplies at most one tap once p and q are
    // pinned by ih and iw). At unit stride the tap indices satisfy
    // ``p = ih + pad_height - i`` and ``q = iw + pad_width - j``, so p falls
    // as i rises and q falls as j rises — the correspondence is one-for-one,
    // which is exactly what unit stride buys and why a strided geometry
    // cannot use this path. Walking o ascending, p **descending**, and q
    // **descending** therefore enumerates the identical contributions in the
    // identical ascending-o, ascending-i, ascending-j order. Same start value
    // (+0.0), same taps, same order — bit-identical by construction.
    //
    // The stride parameters are deliberately unread: this path exists only
    // for unit stride, the dispatch predicate is what enforces that, and
    // silently multiplying by a stride the proof above does not hold for
    // would be worse than not reading it. They stay in the signature so all
    // six compute paths share one argument list.
    (void)stride_height;
    (void)stride_width;
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < in_channels; ++c) {
            for (int64_t ih = 0; ih < input_height; ++ih) {
                T* grad_row = grad_input + conv2d_index4d(
                    n, c, ih, 0, in_channels, input_height, input_width);
                // Zero this destination row. Every (n, c, ih) is visited, so
                // the whole grad_input span is written exactly as the generic
                // path's leading zero-fill writes it, and a cell that receives
                // no contribution keeps the same +0.0.
                for (int64_t t = 0; t < input_width; ++t) {
                    grad_row[t] = T(0);
                }
                for (int64_t o = 0; o < out_channels; ++o) {
                    for (int64_t p = kernel_height - 1; p >= 0; --p) {
                        const int64_t i = ih + pad_height - p;
                        if (i < 0 || i >= output_height) {
                            continue;  // no upstream row supplies this tap
                        }
                        const T* g_row = grad_output + conv2d_index4d(
                            n, o, i, 0, out_channels, output_height,
                            output_width);
                        const T* w_row = weight + conv2d_index4d(
                            o, c, p, 0, in_channels, kernel_height,
                            kernel_width);
                        for (int64_t q = kernel_width - 1; q >= 0; --q) {
                            // j = iw + pad_width - q must lie in [0,
                            // output_width), so iw runs over one contiguous
                            // run clamped into [0, input_width).
                            int64_t lo = q - pad_width;
                            if (lo < 0) {
                                lo = 0;
                            }
                            int64_t hi = output_width - 1 + q - pad_width;
                            if (hi > input_width - 1) {
                                hi = input_width - 1;
                            }
                            if (lo > hi) {
                                continue;
                            }
                            const T w_value = w_row[q];
                            T* dst = grad_row + lo;
                            const T* src = g_row + (lo + pad_width - q);
                            const int64_t span = hi - lo + 1;
                            for (int64_t t = 0; t < span; ++t) {
                                dst[t] += src[t] * w_value;
                            }
                        }
                    }
                }
            }
        }
    }
}

template <class T>
inline void conv2d_weight_backward_gather(
    const T* grad_output, const T* input, T* grad_weight,
    TF_CONV2D_KERNEL_PARAMS) noexcept {
    // The generic path scatters into grad_weight from every upstream
    // position; this path owns one destination at a time and gathers the
    // whole sum for it into a register, so the destination is written once
    // instead of being read-modify-written batch*output_height*output_width
    // times.
    //
    // ACCUMULATION-ORDER PROOF. Fix one destination (o, c, p, q). The generic
    // path's outer loops are n, o, i, j and its taps c, p, q, so that
    // destination's contributions arrive in ascending n, then i, then j —
    // which is exactly this nest's n, i, j order. The generic path skips a
    // contribution when its source coordinate leaves the real input; here the
    // same contributions are skipped by the [i_begin, i_end) and [j_begin,
    // j_end) ranges, which are that same condition solved for i and j. The
    // register accumulator starts at T(0), the same value the generic path's
    // zero-filled destination starts from, and is stored once at the end.
    // Same start value, same taps, same order — bit-identical by
    // construction. Because every destination is assigned, this path needs no
    // separate zero-fill and never reads its destination (H1).
    for (int64_t o = 0; o < out_channels; ++o) {
        for (int64_t c = 0; c < in_channels; ++c) {
            for (int64_t p = 0; p < kernel_height; ++p) {
                // Upstream rows whose source row lies inside the real input.
                // Depends only on p, so it is hoisted above the batch loop.
                const int64_t i_begin =
                    conv2d_tap_begin(p, pad_height, stride_height);
                const int64_t i_end = conv2d_tap_end(
                    p, pad_height, stride_height, input_height, output_height);
                for (int64_t q = 0; q < kernel_width; ++q) {
                    const int64_t j_begin =
                        conv2d_tap_begin(q, pad_width, stride_width);
                    const int64_t j_end = conv2d_tap_end(
                        q, pad_width, stride_width, input_width, output_width);
                    T acc = T(0);
                    if (j_begin < j_end) {
                        const int64_t span = j_end - j_begin;
                        for (int64_t n = 0; n < batch; ++n) {
                            for (int64_t i = i_begin; i < i_end; ++i) {
                                const int64_t ih =
                                    i * stride_height + p - pad_height;
                                const T* g_row = grad_output + conv2d_index4d(
                                    n, o, i, j_begin, out_channels,
                                    output_height, output_width);
                                const T* in_row = input + conv2d_index4d(
                                    n, c, ih,
                                    j_begin * stride_width + q - pad_width,
                                    in_channels, input_height, input_width);
                                if (stride_width == 1) {
                                    for (int64_t t = 0; t < span; ++t) {
                                        acc += g_row[t] * in_row[t];
                                    }
                                } else {
                                    for (int64_t t = 0; t < span; ++t) {
                                        acc += g_row[t]
                                             * in_row[t * stride_width];
                                    }
                                }
                            }
                        }
                    }
                    grad_weight[conv2d_index4d(
                        o, c, p, q, in_channels, kernel_height,
                        kernel_width)] = acc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatchers. These are the entry points the exported wrappers call; each
// picks a compute path from the integer geometry it already holds and calls
// it. No state is consulted or kept, and a rejected predicate is a fallback,
// never an error. The geometry predicates read ``int64`` metadata only, so
// the same layout chooses the same traversal at both element widths.
// ---------------------------------------------------------------------------

template <class T>
inline void conv2d_forward_contiguous(
    const T* input, const T* weight, const T* bias, T* output,
    TF_CONV2D_KERNEL_PARAMS) noexcept {
    if (conv2d_forward_prefers_row_sweep(input_width, output_width)) {
        conv2d_forward_row_sweep(
            input, weight, bias, output, batch, in_channels, input_height,
            input_width, out_channels, kernel_height, kernel_width,
            stride_height, stride_width, pad_height, pad_width, output_height,
            output_width);
        return;
    }
    conv2d_forward_generic(
        input, weight, bias, output, batch, in_channels, input_height,
        input_width, out_channels, kernel_height, kernel_width, stride_height,
        stride_width, pad_height, pad_width, output_height, output_width);
}

template <class T>
inline void conv2d_input_backward_contiguous(
    const T* grad_output, const T* weight, T* grad_input,
    TF_CONV2D_KERNEL_PARAMS) noexcept {
    if (conv2d_input_backward_prefers_gather(
            stride_height, stride_width, input_width, output_width)) {
        conv2d_input_backward_gather(
            grad_output, weight, grad_input, batch, in_channels, input_height,
            input_width, out_channels, kernel_height, kernel_width,
            stride_height, stride_width, pad_height, pad_width, output_height,
            output_width);
        return;
    }
    conv2d_input_backward_generic(
        grad_output, weight, grad_input, batch, in_channels, input_height,
        input_width, out_channels, kernel_height, kernel_width, stride_height,
        stride_width, pad_height, pad_width, output_height, output_width);
}

template <class T>
inline void conv2d_weight_backward_contiguous(
    const T* grad_output, const T* input, T* grad_weight,
    TF_CONV2D_KERNEL_PARAMS) noexcept {
    if (conv2d_weight_backward_prefers_gather(input_width, output_width)) {
        conv2d_weight_backward_gather(
            grad_output, input, grad_weight, batch, in_channels, input_height,
            input_width, out_channels, kernel_height, kernel_width,
            stride_height, stride_width, pad_height, pad_width, output_height,
            output_width);
        return;
    }
    conv2d_weight_backward_generic(
        grad_output, input, grad_weight, batch, in_channels, input_height,
        input_width, out_channels, kernel_height, kernel_width, stride_height,
        stride_width, pad_height, pad_width, output_height, output_width);
}

}  // namespace tf
