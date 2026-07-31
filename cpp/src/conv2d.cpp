// Conv2d forward (Phase D, milestone D2): the internal CPU float64
// cross-correlation compute kernel — pure arithmetic only.
//
// This file holds the math and nothing else: no C ABI export, no
// TF_GUARD, no Storage handles, no allocation, no validation. The D3
// milestone will add the exported ``tf_core_conv2d_forward`` wrapper
// (Storage handles + the thread-local error contract) alongside the
// Python/Core validation and output allocation; the input-, weight-, and
// bias-gradient kernels (D4/D5) will join this same file later.
//
// Cross-correlation (NOT flipped convolution), matching the reference
// stable tensorforge.nn.Conv2d:
//   out[n,o,i,j] = bias[o] + sum_{c,p,q} x_pad[n,c,i*sh+p,j*sw+q] * w[o,c,p,q]
// Symmetric zero padding is realized by skipping out-of-bounds source
// coordinates (a padded cell contributes 0*w = 0), so no padded copy of
// the input is materialized. See docs/native_cnn_design.md §6.

#include "tf_conv2d_internal.h"
#include "tf_internal.h"  // export macro, Storage/as_storage, TF_GUARD, set_error

namespace {

// Row-major flat offset into a 4-D array of extents (·, dim1, dim2, dim3)
// at index (i0, i1, i2, i3). File-local (never part of the ABI). The
// leading extent is not needed for the offset, so it is not a parameter.
// One helper serves all three arrays, which share this layout: the NCHW
// input (dim1=in_channels, dim2=input_height, dim3=input_width), the OIHW
// weight (dim1=in_channels, dim2=kernel_height, dim3=kernel_width), and
// the NCHW output (dim1=out_channels, dim2=output_height, dim3=output_width).
// int64_t arithmetic throughout, matching the rest of the native runtime.
inline int64_t index4d(
    int64_t i0, int64_t i1, int64_t i2, int64_t i3,
    int64_t dim1, int64_t dim2, int64_t dim3) {
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
inline int64_t tap_begin(int64_t offset, int64_t pad, int64_t stride) {
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
inline int64_t tap_end(int64_t offset, int64_t pad, int64_t stride,
                       int64_t size, int64_t limit) {
    const int64_t room = size - 1 + pad - offset;
    if (room < 0) {
        return 0;
    }
    const int64_t end = room / stride + 1;
    return (end < limit) ? end : limit;
}

}  // namespace

namespace tf {

// ---------------------------------------------------------------------------
// H9 dispatch predicates. Total, pure, allocation-free, and functions of the
// integer geometry alone; a false answer selects the retained generic path and
// is never an error. See tf_conv2d_internal.h for the measured justification
// of kConv2dMinSweptExtent.
// ---------------------------------------------------------------------------

bool conv2d_sweep_extent_is_worthwhile(
    int64_t input_width, int64_t output_width) noexcept {
    const int64_t swept =
        (input_width < output_width) ? input_width : output_width;
    return swept >= kConv2dMinSweptExtent;
}

bool conv2d_forward_prefers_row_sweep(
    int64_t input_width, int64_t output_width) noexcept {
    return conv2d_sweep_extent_is_worthwhile(input_width, output_width);
}

bool conv2d_input_backward_prefers_gather(
    int64_t stride_height, int64_t stride_width,
    int64_t input_width, int64_t output_width) noexcept {
    return stride_height == 1 && stride_width == 1 &&
           conv2d_sweep_extent_is_worthwhile(input_width, output_width);
}

bool conv2d_weight_backward_prefers_gather(
    int64_t input_width, int64_t output_width) noexcept {
    return conv2d_sweep_extent_is_worthwhile(input_width, output_width);
}

void conv2d_forward_generic(
    const double* input,
    const double* weight,
    const double* bias,
    double* output,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) noexcept {
    // Direct nested loops, correctness first (docs/native_cnn_design.md §6):
    // n, o, i, j outer; c, p, q inner; deterministic c -> p -> q sum order.
    // Padded coordinates are computed in *signed* int64 so an out-of-bounds
    // position is genuinely negative (a skip), never an unsigned wrap.
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t o = 0; o < out_channels; ++o) {
            // Bias is added exactly once per output element, as the seed of
            // the accumulator (0.0 when no bias pointer was supplied).
            const double bias_o = (bias != nullptr) ? bias[o] : 0.0;
            for (int64_t i = 0; i < output_height; ++i) {
                for (int64_t j = 0; j < output_width; ++j) {
                    double acc = bias_o;
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
                                acc += input[index4d(
                                           n, c, ih, iw,
                                           in_channels, input_height,
                                           input_width)]
                                     * weight[index4d(
                                           o, c, p, q,
                                           in_channels, kernel_height,
                                           kernel_width)];
                            }
                        }
                    }
                    output[index4d(
                        n, o, i, j,
                        out_channels, output_height, output_width)] = acc;
                }
            }
        }
    }
}

void conv2d_input_backward_generic(
    const double* grad_output,
    const double* weight,
    double* grad_input,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) noexcept {
    // Scatter-add adjoint of the forward cross-correlation
    // (docs/native_cnn_design.md §7.1). The output is fully defined here:
    // zero the entire grad_input span first, so the caller need not
    // pre-zero it and overlapping windows can accumulate with a plain +=.
    const int64_t input_count =
        batch * in_channels * input_height * input_width;
    for (int64_t idx = 0; idx < input_count; ++idx) {
        grad_input[idx] = 0.0;
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
                    const double g = grad_output[index4d(
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
                                grad_input[index4d(
                                    n, c, ih, iw,
                                    in_channels, input_height, input_width)]
                                    += g * weight[index4d(
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

void conv2d_weight_backward_generic(
    const double* grad_output,
    const double* input,
    double* grad_weight,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) noexcept {
    // Pairs each upstream value with the input pixel it multiplied in the
    // forward (docs/native_cnn_design.md §7.2). The output is fully defined
    // here: zero the entire grad_weight span first, so the caller need not
    // pre-zero it and every (n, i, j) accumulates with a plain +=.
    const int64_t weight_count =
        out_channels * in_channels * kernel_height * kernel_width;
    for (int64_t idx = 0; idx < weight_count; ++idx) {
        grad_weight[idx] = 0.0;
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
                    const double g = grad_output[index4d(
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
                                grad_weight[index4d(
                                    o, c, p, q,
                                    in_channels, kernel_height, kernel_width)]
                                    += g * input[index4d(
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
// SIMD intrinsic, threading, or parallel accumulation anywhere.
// ---------------------------------------------------------------------------

void conv2d_forward_row_sweep(
    const double* input,
    const double* weight,
    const double* bias,
    double* output,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) noexcept {
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
    // construction, on every finite value and every special value alike.
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t o = 0; o < out_channels; ++o) {
            const double bias_o = (bias != nullptr) ? bias[o] : 0.0;
            for (int64_t i = 0; i < output_height; ++i) {
                double* out_row = output + index4d(
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
                    tap_begin(i * stride_height, pad_height, 1);
                const int64_t p_end = tap_end(
                    i * stride_height, pad_height, 1, input_height,
                    kernel_height);
                for (int64_t c = 0; c < in_channels; ++c) {
                    for (int64_t p = p_begin; p < p_end; ++p) {
                        const int64_t ih = i * stride_height + p - pad_height;
                        const double* in_row = input + index4d(
                            n, c, ih, 0, in_channels, input_height,
                            input_width);
                        const double* w_row = weight + index4d(
                            o, c, p, 0, in_channels, kernel_height,
                            kernel_width);
                        for (int64_t q = 0; q < kernel_width; ++q) {
                            // Output columns whose source column lies inside
                            // the real input, for this kernel column.
                            const int64_t j_begin =
                                tap_begin(q, pad_width, stride_width);
                            const int64_t j_end = tap_end(
                                q, pad_width, stride_width, input_width,
                                output_width);
                            if (j_begin >= j_end) {
                                continue;  // this kernel column is all padding
                            }
                            const double w_value = w_row[q];
                            // Bases taken at the first *valid* position, so no
                            // pointer is ever formed outside its array even
                            // when the padding would place one there.
                            double* dst = out_row + j_begin;
                            const double* src = in_row
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

void conv2d_input_backward_gather(
    const double* grad_output,
    const double* weight,
    double* grad_input,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) noexcept {
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
                double* grad_row = grad_input + index4d(
                    n, c, ih, 0, in_channels, input_height, input_width);
                // Zero this destination row. Every (n, c, ih) is visited, so
                // the whole grad_input span is written exactly as the generic
                // path's leading zero-fill writes it, and a cell that receives
                // no contribution keeps the same +0.0.
                for (int64_t t = 0; t < input_width; ++t) {
                    grad_row[t] = 0.0;
                }
                for (int64_t o = 0; o < out_channels; ++o) {
                    for (int64_t p = kernel_height - 1; p >= 0; --p) {
                        const int64_t i = ih + pad_height - p;
                        if (i < 0 || i >= output_height) {
                            continue;  // no upstream row supplies this tap
                        }
                        const double* g_row = grad_output + index4d(
                            n, o, i, 0, out_channels, output_height,
                            output_width);
                        const double* w_row = weight + index4d(
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
                            const double w_value = w_row[q];
                            double* dst = grad_row + lo;
                            const double* src = g_row + (lo + pad_width - q);
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

void conv2d_weight_backward_gather(
    const double* grad_output,
    const double* input,
    double* grad_weight,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) noexcept {
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
    // register accumulator starts at 0.0, the same value the generic path's
    // zero-filled destination starts from, and is stored once at the end.
    // Same start value, same taps, same order — bit-identical by
    // construction. Because every destination is assigned, this path needs no
    // separate zero-fill and never reads its destination (H1).
    for (int64_t o = 0; o < out_channels; ++o) {
        for (int64_t c = 0; c < in_channels; ++c) {
            for (int64_t p = 0; p < kernel_height; ++p) {
                // Upstream rows whose source row lies inside the real input.
                // Depends only on p, so it is hoisted above the batch loop.
                const int64_t i_begin = tap_begin(p, pad_height, stride_height);
                const int64_t i_end = tap_end(
                    p, pad_height, stride_height, input_height, output_height);
                for (int64_t q = 0; q < kernel_width; ++q) {
                    const int64_t j_begin =
                        tap_begin(q, pad_width, stride_width);
                    const int64_t j_end = tap_end(
                        q, pad_width, stride_width, input_width, output_width);
                    double acc = 0.0;
                    if (j_begin < j_end) {
                        const int64_t span = j_end - j_begin;
                        for (int64_t n = 0; n < batch; ++n) {
                            for (int64_t i = i_begin; i < i_end; ++i) {
                                const int64_t ih =
                                    i * stride_height + p - pad_height;
                                const double* g_row = grad_output + index4d(
                                    n, o, i, j_begin, out_channels,
                                    output_height, output_width);
                                const double* in_row = input + index4d(
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
                    grad_weight[index4d(
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
// never an error.
// ---------------------------------------------------------------------------

void conv2d_forward_contiguous(
    const double* input,
    const double* weight,
    const double* bias,
    double* output,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) noexcept {
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

void conv2d_input_backward_contiguous(
    const double* grad_output,
    const double* weight,
    double* grad_input,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) noexcept {
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

void conv2d_weight_backward_contiguous(
    const double* grad_output,
    const double* input,
    double* grad_weight,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) noexcept {
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

// ---------------------------------------------------------------------------
// Exported C ABI wrapper (Phase D, milestone D3).
//
// ``tf_core_conv2d_forward`` is the exception-guarded boundary between the
// Python/Core layer and the internal noexcept arithmetic above. The Core
// wrapper (backends/cpp.py) owns the *shape/argument* validation and, by
// Policy B (docs/native_cnn_design.md §5), guarantees every operand it
// passes is **row-major contiguous** — so the raw ABI takes only storage
// handles + per-operand offsets + the integer dimensions, never stride
// arrays, and computes row-major offsets itself.
//
// Even though the Core layer validates first, the boundary still defends
// itself against a direct (mis)call: null required handles, non-positive
// dimensions, negative padding/offsets, an output shape that disagrees
// with the locked floor formula, integer overflow in the shape/span
// products, and any storage span that would read or write outside its
// allocation are all rejected with TF_ERROR_INVALID *before* the internal
// kernel runs. The internal kernel neither allocates nor throws, so the
// only failure this wrapper can surface is invalid metadata; the guard is
// present regardless, per the ABI error contract.
// ---------------------------------------------------------------------------

using tf::as_storage;

namespace {

// Checked int64 multiply/add for non-negative operands: return false on
// overflow instead of wrapping, so a bogus dimension can never turn into a
// small (passing) span through silent int64 overflow.
bool checked_mul(int64_t a, int64_t b, int64_t& out) {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > INT64_MAX / b) {
        return false;
    }
    out = a * b;
    return true;
}

bool checked_add(int64_t a, int64_t b, int64_t& out) {
    if (a > INT64_MAX - b) {
        return false;
    }
    out = a + b;
    return true;
}

// A contiguous operand of ``count`` elements beginning at ``offset`` must
// fit inside a storage holding ``size`` doubles. ``offset``/``count`` are
// already known non-negative here.
bool span_within(int64_t offset, int64_t count, int64_t size) {
    int64_t end;
    if (!checked_add(offset, count, end)) {
        return false;  // offset + count overflowed -> cannot fit
    }
    return end <= size;
}

// Product of four non-negative extents with overflow detection.
bool numel4(int64_t a, int64_t b, int64_t c, int64_t d, int64_t& out) {
    return checked_mul(a, b, out) && checked_mul(out, c, out) &&
           checked_mul(out, d, out);
}

// Recompute one spatial output extent from the locked floor formula and
// confirm the caller's claimed value agrees. ``num`` (padded - kernel) is
// kept non-negative: a negative value means the kernel does not fit the
// padded input, which would make the output extent < 1. With ``num >= 0``
// and ``stride >= 1``, C++ integer division equals the floor the Python
// side computes with ``//``. Returns nullptr on success, else an error
// message.
const char* check_output_dim(
    int64_t size, int64_t kernel, int64_t stride, int64_t pad, int64_t claimed) {
    int64_t two_pad, padded;
    if (!checked_mul(2, pad, two_pad) || !checked_add(size, two_pad, padded)) {
        return "conv2d_forward: output-shape arithmetic overflow";
    }
    const int64_t num = padded - kernel;
    if (num < 0) {
        return "conv2d_forward: kernel does not fit the padded input";
    }
    const int64_t expected = num / stride + 1;  // floor: num >= 0, stride >= 1
    if (claimed != expected) {
        return "conv2d_forward: output dimension disagrees with the computed "
               "shape";
    }
    return nullptr;
}

}  // namespace

TF_EXPORT void tf_core_conv2d_forward(
    const void* input_handle, int64_t input_offset,
    const void* weight_handle, int64_t weight_offset,
    const void* bias_handle, int64_t bias_offset,
    void* output_handle,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) {
    TF_GUARD_BEGIN
    // -- required handles (bias is the one nullable handle: null = no bias) --
    if (input_handle == nullptr || weight_handle == nullptr ||
        output_handle == nullptr) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: null required storage handle");
        return;
    }
    // -- dimension / kernel / stride bounds --
    if (batch < 1 || in_channels < 1 || input_height < 1 || input_width < 1 ||
        out_channels < 1 || kernel_height < 1 || kernel_width < 1 ||
        stride_height < 1 || stride_width < 1) {
        tf::set_error(
            TF_ERROR_INVALID,
            "conv2d_forward: batch, channels, spatial, kernel, and stride "
            "extents must each be >= 1");
        return;
    }
    if (pad_height < 0 || pad_width < 0) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: padding must be >= 0");
        return;
    }
    if (input_offset < 0 || weight_offset < 0 || bias_offset < 0 ||
        output_height < 1 || output_width < 1) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: negative offset or non-positive output "
                      "extent");
        return;
    }
    // -- output shape must equal the locked floor formula --
    if (const char* err = check_output_dim(
            input_height, kernel_height, stride_height, pad_height,
            output_height)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    if (const char* err = check_output_dim(
            input_width, kernel_width, stride_width, pad_width,
            output_width)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    // -- storage spans must fit their allocations (overflow-checked) --
    int64_t input_count, weight_count, output_count;
    if (!numel4(batch, in_channels, input_height, input_width, input_count) ||
        !numel4(out_channels, in_channels, kernel_height, kernel_width,
                weight_count) ||
        !numel4(batch, out_channels, output_height, output_width,
                output_count)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: shape product overflows int64");
        return;
    }
    if (!span_within(input_offset, input_count,
                     as_storage(input_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: input span exceeds its storage");
        return;
    }
    if (!span_within(weight_offset, weight_count,
                     as_storage(weight_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: weight span exceeds its storage");
        return;
    }
    if (!span_within(0, output_count, as_storage(output_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: output span exceeds its storage");
        return;
    }
    const double* bias = nullptr;
    if (bias_handle != nullptr) {
        if (!span_within(bias_offset, out_channels,
                         as_storage(bias_handle)->size)) {
            tf::set_error(TF_ERROR_INVALID,
                          "conv2d_forward: bias span exceeds its storage");
            return;
        }
        bias = as_storage(bias_handle)->data + bias_offset;
    }
    // -- validated: run the internal noexcept cross-correlation --
    const double* input = as_storage(input_handle)->data + input_offset;
    const double* weight = as_storage(weight_handle)->data + weight_offset;
    double* output = as_storage(output_handle)->data;
    tf::conv2d_forward_contiguous(
        input, weight, bias, output,
        batch, in_channels, input_height, input_width,
        out_channels, kernel_height, kernel_width,
        stride_height, stride_width, pad_height, pad_width,
        output_height, output_width);
    TF_GUARD_END_VOID()
}

// ---------------------------------------------------------------------------
// Exported C ABI backward wrappers (Phase D, milestone D6).
//
// ``tf_core_conv2d_input_backward`` and ``tf_core_conv2d_weight_backward``
// are the exception-guarded boundaries over the D4/D5 internal noexcept
// gradient kernels. They share the forward wrapper's contract exactly (same
// file-local checked-arithmetic helpers, same validation classes, same
// error style): the raw ABI takes storage handles + per-operand offsets +
// the integer dimensions, **never stride arrays**, so it interprets each
// span as canonical contiguous data and bounds-checks it — logical
// contiguity remains the caller precondition the NativeTensorCore layer
// guarantees by Policy-B copy-then-compute. Neither allocates or frees
// caller-owned storage, and neither mutates its read-only operands. There
// is **no** bias-gradient C ABI symbol — that gradient composes from the
// existing native ``sum`` reduction (docs/native_cnn_design.md §7.3).
// ---------------------------------------------------------------------------

// Shared validation for both backward wrappers: the dimension/kernel/stride/
// padding/offset bounds and the output-shape agreement identical to the
// forward wrapper. Returns nullptr on success, else an error message. The
// caller checks required handles and storage spans separately (the operand
// roles differ between the two backward directions).
namespace {

const char* validate_conv2d_backward_dims(
    int64_t batch, int64_t in_channels,
    int64_t input_height, int64_t input_width,
    int64_t out_channels, int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t output_height, int64_t output_width,
    int64_t grad_output_offset, int64_t other_offset) {
    if (batch < 1 || in_channels < 1 || input_height < 1 || input_width < 1 ||
        out_channels < 1 || kernel_height < 1 || kernel_width < 1 ||
        stride_height < 1 || stride_width < 1) {
        return "conv2d backward: batch, channels, spatial, kernel, and stride "
               "extents must each be >= 1";
    }
    if (pad_height < 0 || pad_width < 0) {
        return "conv2d backward: padding must be >= 0";
    }
    if (grad_output_offset < 0 || other_offset < 0 ||
        output_height < 1 || output_width < 1) {
        return "conv2d backward: negative offset or non-positive output extent";
    }
    if (const char* err = check_output_dim(
            input_height, kernel_height, stride_height, pad_height,
            output_height)) {
        return err;
    }
    if (const char* err = check_output_dim(
            input_width, kernel_width, stride_width, pad_width,
            output_width)) {
        return err;
    }
    return nullptr;
}

}  // namespace

// Gradient w.r.t. the Conv2d input. grad_output (NCHW) + weight (OIHW) ->
// grad_input (NCHW, caller-allocated, offset 0).
TF_EXPORT void tf_core_conv2d_input_backward(
    const void* grad_output_handle, int64_t grad_output_offset,
    const void* weight_handle, int64_t weight_offset,
    void* grad_input_handle,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) {
    TF_GUARD_BEGIN
    if (grad_output_handle == nullptr || weight_handle == nullptr ||
        grad_input_handle == nullptr) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_input_backward: null required storage handle");
        return;
    }
    if (const char* err = validate_conv2d_backward_dims(
            batch, in_channels, input_height, input_width,
            out_channels, kernel_height, kernel_width,
            stride_height, stride_width, pad_height, pad_width,
            output_height, output_width, grad_output_offset, weight_offset)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    int64_t grad_output_count, weight_count, grad_input_count;
    if (!numel4(batch, out_channels, output_height, output_width,
                grad_output_count) ||
        !numel4(out_channels, in_channels, kernel_height, kernel_width,
                weight_count) ||
        !numel4(batch, in_channels, input_height, input_width,
                grad_input_count)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_input_backward: shape product overflows int64");
        return;
    }
    if (!span_within(grad_output_offset, grad_output_count,
                     as_storage(grad_output_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_input_backward: grad_output span exceeds storage");
        return;
    }
    if (!span_within(weight_offset, weight_count,
                     as_storage(weight_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_input_backward: weight span exceeds storage");
        return;
    }
    if (!span_within(0, grad_input_count,
                     as_storage(grad_input_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_input_backward: grad_input span exceeds storage");
        return;
    }
    const double* grad_output =
        as_storage(grad_output_handle)->data + grad_output_offset;
    const double* weight = as_storage(weight_handle)->data + weight_offset;
    double* grad_input = as_storage(grad_input_handle)->data;
    tf::conv2d_input_backward_contiguous(
        grad_output, weight, grad_input,
        batch, in_channels, input_height, input_width,
        out_channels, kernel_height, kernel_width,
        stride_height, stride_width, pad_height, pad_width,
        output_height, output_width);
    TF_GUARD_END_VOID()
}

// Gradient w.r.t. the Conv2d weight. grad_output (NCHW) + input (NCHW) ->
// grad_weight (OIHW, caller-allocated, offset 0).
TF_EXPORT void tf_core_conv2d_weight_backward(
    const void* grad_output_handle, int64_t grad_output_offset,
    const void* input_handle, int64_t input_offset,
    void* grad_weight_handle,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) {
    TF_GUARD_BEGIN
    if (grad_output_handle == nullptr || input_handle == nullptr ||
        grad_weight_handle == nullptr) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_weight_backward: null required storage handle");
        return;
    }
    if (const char* err = validate_conv2d_backward_dims(
            batch, in_channels, input_height, input_width,
            out_channels, kernel_height, kernel_width,
            stride_height, stride_width, pad_height, pad_width,
            output_height, output_width, grad_output_offset, input_offset)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    int64_t grad_output_count, input_count, grad_weight_count;
    if (!numel4(batch, out_channels, output_height, output_width,
                grad_output_count) ||
        !numel4(batch, in_channels, input_height, input_width,
                input_count) ||
        !numel4(out_channels, in_channels, kernel_height, kernel_width,
                grad_weight_count)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_weight_backward: shape product overflows int64");
        return;
    }
    if (!span_within(grad_output_offset, grad_output_count,
                     as_storage(grad_output_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_weight_backward: grad_output span exceeds storage");
        return;
    }
    if (!span_within(input_offset, input_count,
                     as_storage(input_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_weight_backward: input span exceeds storage");
        return;
    }
    if (!span_within(0, grad_weight_count,
                     as_storage(grad_weight_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_weight_backward: grad_weight span exceeds storage");
        return;
    }
    const double* grad_output =
        as_storage(grad_output_handle)->data + grad_output_offset;
    const double* input = as_storage(input_handle)->data + input_offset;
    double* grad_weight = as_storage(grad_weight_handle)->data;
    tf::conv2d_weight_backward_contiguous(
        grad_output, input, grad_weight,
        batch, in_channels, input_height, input_width,
        out_channels, kernel_height, kernel_width,
        stride_height, stride_width, pad_height, pad_width,
        output_height, output_width);
    TF_GUARD_END_VOID()
}
