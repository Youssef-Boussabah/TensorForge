// Internal (non-ABI) declarations of the two matrix-multiplication
// compute paths the production tensor-core matmul dispatches between,
// and the metadata predicate that chooses (Phase H, milestone H2). See
// docs/native_cpu_performance_design.md §16.2.
//
// Like the Conv2d and pooling compute kernels these are deliberately NOT
// part of the public C ABI: plain C++ in ``namespace tf`` with
// hidden visibility, holding only contiguous/strided CPU
// arithmetic at the element type. They allocate nothing, report no error, and mutate no
// input. The exported, guarded ``tf_core_matmul`` wrapper lives in
// cpp/src/matmul.cpp alongside them, and the shape validation and the
// output allocation live in backends/cpp.py.
//
// **H2 added no exported symbol.** The choice between the two paths is
// made inside ``tf_core_matmul`` from the stride metadata it already
// receives, so there is no kernel selector, block-size setter, dispatch
// tracer, or CPU-feature control anywhere in the ABI — and none may be
// added. The dispatch is observable to tests through
// ``matmul_prefers_row_sweep`` (compiled directly into
// cpp/tests/test_matmul.cpp, exactly as the D8/D9 pooling tests reach
// their internal kernels) and, from Python, by the fact that the two
// paths are proved to agree on the same logical operands under the
// four-part numerical contract stated below.
//
// ---------------------------------------------------------------------------
// Phase I, milestone I4: both kernels carry a scalar type parameter
// ---------------------------------------------------------------------------
//
// ``T`` is **deduced from the pointer arguments**, so every pre-Phase-I call
// site — all of which pass ``double*`` — instantiates ``T = double`` and is
// the pre-I4 kernel statement for statement. ``T = float`` is the same source
// at binary32: the same loop nest, the same ``k`` order, the same row
// grouping, the same ``MATMUL_ROW_BLOCK``, the same predicate.
//
// **The accumulator type is exactly ``T``** (design §10.1): a float32 matmul
// loads ``float``, multiplies as ``float``, accumulates as ``float``, and
// stores ``float``. There is no widening accumulator, no FMA, no
// reassociation, and no fast-math — so the per-output ``k`` order really is
// the value's definition at both widths, and a float32 result is a genuinely
// different number from "accumulate in binary64 and round once", which is
// observable and is witnessed by test rather than argued.
//
// The literal ``0.0`` on the row sweep's assigning pass and the generic
// kernel's ``double sum = 0.0`` became ``T(0)``. At ``T = double`` that *is*
// the old literal; at ``T = float`` it keeps the addition in binary32 instead
// of promoting the whole expression around a binary64 zero. It is still
// written out rather than folded away, for H2's original reason:
// ``0.0f + (-0.0f)`` is ``+0.0f`` while ``-0.0f`` alone is not.
//
// The two definitions moved into this header for the ordinary reason a
// template must — so both instantiations are available to the exported
// wrapper in matmul.cpp *and* to the CTests that compile it directly.
#pragma once

#include <cstdint>

namespace tf {

// ---------------------------------------------------------------------------
// The fixed, documented policy constants. Compile-time values, not a
// runtime probe, an environment variable, an autotuner, or a stored
// machine-specific measurement: the same source produces the same
// dispatch decision on every machine, which is what keeps every
// bit-exact resume proof reproducible.
// ---------------------------------------------------------------------------

// How many destination rows the row sweep keeps hot at once. Each of the
// group's rows is swept with the same ``b`` row segment, so one segment
// load serves MATMUL_ROW_BLOCK outputs. Measured across 4x4 to 384x384x384
// plus rectangular, tall-skinny, short-wide, and Linear/MLP shapes: 4 was
// the best or statistically tied-best fixed choice, and never the worst.
// The working set it must keep in L1 is (MATMUL_ROW_BLOCK + 1) * p
// doubles, so a small value is also the portable one.
constexpr int64_t MATMUL_ROW_BLOCK = 4;

// The narrowest destination the row sweep is used for. Below this the
// inner ``j`` loop is too short to pay for its per-``k`` setup, and the
// generic i-j-k kernel — whose inner ``k`` loop is long in exactly that
// regime — measured strictly better (down to 0.27x at p == 1). At p >= 8
// the row sweep measured level or ahead everywhere. A threshold, not a
// heuristic: it is a fixed function of one metadata value.
constexpr int64_t MATMUL_MIN_COLUMNS = 8;

// ---------------------------------------------------------------------------
// The dispatch predicate
// ---------------------------------------------------------------------------

// True when ``matmul_row_sweep`` may be used for this problem, false when
// ``matmul_generic_strided`` must be. Total (every input answers), pure
// (no allocation, no mutation, no error state, no global read), and a
// function of metadata alone — never of a pointer value, an alignment, a
// wall-clock reading, an environment variable, or a CPU-feature probe.
//
// All three conditions must hold:
//
//   1. ``b_stride1 == 1`` — consecutive columns of the right operand are
//      consecutive in memory, which is what makes the row sweep's inner
//      loop a contiguous read. A right operand whose *rows* are the unit
//      stride instead (a transposed view: ``b_stride0 == 1``) is exactly
//      the case the generic kernel already handles well, since its inner
//      ``k`` loop is then the contiguous one.
//   2. ``n >= 1`` — the row sweep's ``k == 0`` pass is what assigns every
//      destination element before anything accumulates into it. With no
//      ``k`` at all there is no assigning pass, so an empty inner
//      dimension goes to the generic kernel, which writes a plain 0.0.
//   3. ``p >= MATMUL_MIN_COLUMNS`` — see the constant above.
//
// A false answer is a fallback, never an error.
bool matmul_prefers_row_sweep(int64_t m, int64_t n, int64_t p,
                              int64_t b_stride1) noexcept;

// ---------------------------------------------------------------------------
// The retained generic reference path (docs/native_cpu_performance_design.md
// §8.3). This is the pre-H2 kernel, unchanged: the i-j-k triple loop that
// addresses both operands through their own strides and offsets, so a
// transposed, narrowed, or offset view multiplies without being
// materialized first. It is shipped, reachable through ordinary
// production dispatch, and is the oracle every optimized result is
// compared against.
// ---------------------------------------------------------------------------
//
//   dst[i, j] = sum over k of a[i, k] * b[k, j]
//
// with a[i, k] at ``a_offset + i*a_stride0 + k*a_stride1`` and b[k, j] at
// ``b_offset + k*b_stride0 + j*b_stride1``, for any strides those
// expressions stay in bounds for.
//
// Accumulation: one local ``double sum``, initialized to 0.0, taking k in
// ascending order 0 .. n-1, stored once. **The destination is never read**
// — which is the H1 property this path has always had.
//
// Layout: ``dst`` is row-major contiguous (m, p) at offset 0.
//
// Preconditions (guaranteed by the exported wrapper and the Core layer;
// NOT re-validated here — this is the inner math, not a validation
// boundary): all three pointers are non-null; every addressed element
// lies inside its storage; ``dst`` aliases neither operand; every integer
// product and offset is representable in int64.
//
// Allocates nothing and cannot throw (noexcept).
template <class T>
inline void matmul_generic_strided(
    const T* a, const T* b, T* dst,
    int64_t m, int64_t n, int64_t p,
    int64_t a_stride0, int64_t a_stride1,
    int64_t b_stride0, int64_t b_stride1,
    int64_t a_offset, int64_t b_offset
) noexcept {
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < p; ++j) {
            // One local accumulator, in the element type. ``T(0)`` is
            // ``0.0`` at T = double, so float64 is the pre-I4 expression.
            T sum = T(0);
            for (int64_t k = 0; k < n; ++k) {
                sum += a[a_offset + i * a_stride0 + k * a_stride1]
                     * b[b_offset + k * b_stride0 + j * b_stride1];
            }
            dst[i * p + j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// The optimized path: an i-k-j row sweep over MATMUL_ROW_BLOCK
// destination rows at a time. Requires ``matmul_prefers_row_sweep`` to
// have returned true, i.e. b's column stride is 1, n >= 1, and
// p >= MATMUL_MIN_COLUMNS.
// ---------------------------------------------------------------------------
//
// Why it is faster: the generic kernel's innermost loop walks ``b`` down a
// column, stepping ``b_stride0`` doubles per element, so a row-major right
// operand touches a new cache line on every step. Swapping the ``k`` and
// ``j`` loops makes the innermost loop walk a **row** of ``b`` and a row of
// ``dst`` sequentially, which is the access pattern the hardware
// prefetcher is built for. Nothing about the arithmetic changes.
//
// How it relates numerically to the generic path — stated in the four
// parts the contract actually has, because "bit-identical" on its own
// would be an overclaim (docs/native_cpu_performance_design.md §16.2.3):
//
//   1. **Accumulation order is preserved exactly.** For a fixed output
//      (i, j) the products are accumulated starting from 0.0 and taking k
//      in ascending order 0, 1, ... n-1 — the same values, the same
//      operations, the same order as the generic kernel. No addition is
//      reassociated, no partial sums are combined, no wider or narrower
//      accumulator is used, no fused multiply-add is requested, and no
//      parallel or vector reduction exists. The ``0.0 +`` on the k == 0
//      pass is written out rather than folded away, because
//      ``0.0 + (-0.0)`` is ``+0.0`` and dropping it would change the sign
//      of a zero result.
//   2. **Finite-result bit identity.** Whenever the result is not a NaN,
//      the two paths agree bit for bit — including +0.0 versus -0.0,
//      ±infinity, denormals, the smallest normal, and the largest finite
//      magnitudes.
//   3. **NaN-class equivalence.** Whenever either path produces a NaN,
//      both do, in exactly the same positions, and both are *quiet*.
//      Neither path can produce a signaling NaN.
//   4. **NaN payload bits are NOT part of TensorForge's numerical
//      contract**, and the two paths may differ in them.
//
// Part 4 is a measured property of this code, not a hedge. When two NaN
// operands meet, x86-64's ADDSD returns the *destination* operand's NaN,
// and which addend the compiler places in the destination is an
// instruction-selection decision C++ cannot express. In the i-j-k kernel
// the compiler puts the freshly computed product there, so the **last**
// NaN in k order survives; in every i-k-j structure it puts the
// accumulator there, so the **first** survives. Ten source-level
// formulations were measured — compound versus explicit assignment, named
// locals for the accumulator and for the product, ``__restrict``,
// disabling inner-loop vectorization, and both a 4x64 and a 4x4 stack
// accumulator tile — and all ten i-k-j spellings produced identical
// payloads to one another and different payloads from the reference. The
// only structure that reproduces the reference's payloads is the i-j-k
// order itself, which is the arrangement this kernel exists to replace.
// Payload parity is therefore not available at any price short of
// abandoning the optimization, and it is not claimed.
//
// Why it is safe against an uninitialized destination (H1): the k == 0
// pass **assigns** ``0.0 + a[i,0]*b[0,j]`` to every element of every row in
// the group, and only the k >= 1 passes accumulate. So every destination
// element is written before it is read, and no output value can depend on
// what the buffer held beforehand. The ``n >= 1`` precondition is what
// guarantees the assigning pass runs at all.
//
// Layout: ``dst`` is row-major contiguous (m, p) at offset 0; ``a`` is read
// through its own strides and offset (any layout); ``b`` is read through
// ``b_stride0`` and its offset with a column stride of exactly 1.
//
// Preconditions: as ``matmul_generic_strided``, plus
// ``matmul_prefers_row_sweep(m, n, p, 1)``. Allocates nothing (the row
// group is addressed in place; there is no scratch buffer, workspace, or
// pool) and cannot throw (noexcept).
template <class T>
inline void matmul_row_sweep(
    const T* a, const T* b, T* dst,
    int64_t m, int64_t n, int64_t p,
    int64_t a_stride0, int64_t a_stride1,
    int64_t b_stride0,
    int64_t a_offset, int64_t b_offset
) noexcept {
    for (int64_t i0 = 0; i0 < m; i0 += MATMUL_ROW_BLOCK) {
        const int64_t rows =
            (m - i0 < MATMUL_ROW_BLOCK) ? (m - i0) : MATMUL_ROW_BLOCK;
        // k == 0 — the assigning pass. Every element of every row in the
        // group is written here, before any accumulation reads it, which
        // is what makes an uninitialized destination safe (H1). The
        // explicit `T(0) +` reproduces the generic kernel's
        // `T sum = T(0); sum += ...` exactly: it is not redundant,
        // because 0 + (-0) is +0 while -0 alone is not — at either width.
        {
            const T* b_row = b + b_offset;
            for (int64_t r = 0; r < rows; ++r) {
                T* out = dst + (i0 + r) * p;
                const T a_ik = a[a_offset + (i0 + r) * a_stride0];
                for (int64_t j = 0; j < p; ++j) {
                    out[j] = T(0) + a_ik * b_row[j];
                }
            }
        }
        // k >= 1 — the accumulating passes, ascending, so every output
        // takes its products in the generic kernel's order.
        for (int64_t k = 1; k < n; ++k) {
            const T* b_row = b + b_offset + k * b_stride0;
            for (int64_t r = 0; r < rows; ++r) {
                T* out = dst + (i0 + r) * p;
                const T a_ik =
                    a[a_offset + (i0 + r) * a_stride0 + k * a_stride1];
                for (int64_t j = 0; j < p; ++j) {
                    out[j] += a_ik * b_row[j];
                }
            }
        }
    }
}

}  // namespace tf
