// Internal (non-ABI) declaration of the MaxPool2d forward compute kernel
// (Phase D, milestone D8). See docs/native_cnn_design.md §10, §12, §14.
//
// Like the Conv2d compute kernels this is deliberately NOT part of the
// public C ABI: a plain C++ function in ``namespace tf`` with hidden
// visibility, holding only the contiguous CPU float64 window-maximum
// arithmetic and the winner recording. It performs no allocation, no
// error reporting, and no mutation of its input. The exported, guarded
// ``tf_core_maxpool2d_forward`` wrapper lives in cpp/src/pooling.cpp
// alongside it; the Python/Core validation and the output/winner
// allocation live in backends/cpp.py. The pooling *backward* scatter
// kernel and the NativeTensor autograd integration are D9 — neither is
// declared here.
//
// Because the shared library exports only TF_EXPORT symbols, this
// internal function is not reachable from a separately linked test
// binary — so the D8 test (cpp/tests/test_maxpool2d_forward.cpp)
// compiles cpp/src/pooling.cpp directly rather than linking the shared
// library.
#pragma once

#include <cstdint>

namespace tf {

// Two-dimensional max pooling, CPU float64, direct nested loops. Writes
// the pooled output *and* the parallel winner buffer in a single pass, so
// the forward value and the saved winner always come from the same
// selected candidate (docs/native_cnn_design.md §10).
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
// (matching the stable framework's ``argmax``).
//
// NaN policy (deliberate, documented, outside the supported contract —
// docs/native_cnn_design.md §10): a NaN candidate never wins, so the
// first *non-NaN* candidate seeds the scan and later NaNs cannot displace
// it. If a window has no non-NaN candidate at all (only possible with an
// unpadded window whose every input value is NaN, since padding is -inf),
// the deterministic fallback is the FIRST candidate of the window — its
// value and its winner — so output and winner still agree. No parity with
// NumPy's NaN-propagating argmax is claimed.
//
// Layouts (all row-major contiguous float64):
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
// winner is ever written.
//
// Preconditions (guaranteed by the exported wrapper and the Core layer;
// NOT re-validated here — this routine is the inner math, not a
// validation boundary):
//   * input / output / winners are non-null and each point to contiguous
//     storage of exactly batch*channels*input_height*input_width and
//     batch*channels*output_height*output_width doubles respectively;
//   * every dimension is positive; kernel/stride >= 1; padding >= 0;
//   * output_height / output_width equal the floor formula
//       floor((input_dim + 2*pad - kernel) / stride) + 1;
//   * all integer products and offsets are representable in int64.
//
// The routine allocates no heap memory and cannot throw (noexcept): it is
// pure arithmetic over caller-owned buffers. It reads the input without
// modifying it and writes only inside the output and winner spans.
void maxpool2d_forward_contiguous(
    const double* input,
    double* output,
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
    int64_t output_width) noexcept;

}  // namespace tf
