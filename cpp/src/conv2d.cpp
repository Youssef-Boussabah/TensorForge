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

}  // namespace

namespace tf {

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
