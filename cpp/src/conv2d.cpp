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

}  // namespace tf
