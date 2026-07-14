// Internal (non-ABI) declaration of the Conv2d forward compute kernel
// (Phase D, milestone D2). See docs/native_cnn_design.md §6 and §14.
//
// This is deliberately NOT part of the public C ABI: it is a plain C++
// function in ``namespace tf`` with hidden visibility, holding only the
// contiguous CPU float64 cross-correlation arithmetic. It performs no
// allocation, no error reporting, and no mutation of its inputs. The D3
// milestone will add the exported ``tf_core_conv2d_forward`` wrapper
// (TF_GUARD error contract + Storage handles) plus the Python/Core
// validation and output allocation; none of that lives here.
//
// Because the shared library exports only TF_EXPORT symbols, this
// internal function is not reachable from a separately linked test
// binary — so the D2 test (cpp/tests/test_conv2d_forward.cpp) compiles
// cpp/src/conv2d.cpp directly rather than linking the shared library.
#pragma once

#include <cstdint>

namespace tf {

// Two-dimensional cross-correlation (NOT flipped convolution), CPU
// float64, direct nested loops. Matches the stable tensorforge.nn.Conv2d
// numerically (to a floating-point tolerance; the summation order is
// deterministic but not guaranteed bit-identical to NumPy's einsum).
//
//   out[n, o, i, j] = (bias ? bias[o] : 0)
//                   + sum_{c, p, q} in[n, c, i*sh + p - ph, j*sw + q - pw]
//                                     * weight[o, c, p, q]
//
// with symmetric zero padding applied by skipping source coordinates that
// fall outside the real input (a padded cell contributes 0). Bias, when
// present, is added exactly once per output element. Accumulation runs in
// deterministic c -> p -> q order into a double accumulator.
//
// Layouts (all row-major contiguous float64):
//   input  : NCHW  (batch, in_channels, input_height, input_width)
//   weight : OIHW  (out_channels, in_channels, kernel_height, kernel_width)
//   bias   : (out_channels,), or nullptr for no bias
//   output : NCHW  (batch, out_channels, output_height, output_width) — written
//
// Preconditions (guaranteed by the future D3 wrapper; NOT re-validated
// here — this routine is the inner math, not a validation boundary):
//   * input / weight / output are non-null and each point to contiguous
//     storage of exactly batch*in_channels*input_height*input_width,
//     out_channels*in_channels*kernel_height*kernel_width, and
//     batch*out_channels*output_height*output_width doubles;
//   * bias is nullptr or points to out_channels doubles;
//   * every dimension is positive; stride/kernel >= 1; padding >= 0;
//   * output_height / output_width equal the floor formula
//       floor((input_dim + 2*pad - kernel) / stride) + 1
//     so every output coordinate maps inside the padded grid.
//
// The routine allocates no heap memory and cannot throw (noexcept): it is
// pure arithmetic over caller-owned buffers.
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
    int64_t output_width) noexcept;

}  // namespace tf
