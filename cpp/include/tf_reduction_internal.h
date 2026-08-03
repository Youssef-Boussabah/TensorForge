// Internal (non-ABI) definitions of the two sum-reduction traversals the
// production reduction kernel dispatches between, the narrow-backward
// scatter, and the metadata predicate that chooses between the two
// traversals (Phase H, milestone H6). See
// docs/native_cpu_performance_design.md §16.6.
//
// Like the H2 matmul paths, the H5 copy predicate, and the Conv2d/pooling
// compute kernels these are deliberately NOT part of the public C ABI:
// plain C++ in ``namespace tf`` with hidden visibility, holding only
// traversal and arithmetic at the element type. They allocate nothing, report no
// error, and mutate no input. The exported, guarded ``tf_core_sum``
// wrapper lives in cpp/src/reduction.cpp alongside them, and the axis
// normalization, the output-shape construction, the write-stride
// construction, and the output allocation all live in backends/cpp.py.
//
// **H6 added no exported symbol.** ``tf_core_sum`` keeps the signature,
// the accumulate-into-``dst`` contract, and the ownership contract it has
// had since v1.19; only the loop it runs inside changes. There is no
// reduction-path selector, block-size setter, threshold setter, traversal
// tracer, profiling counter, environment variable, or CPU-feature probe
// anywhere in the ABI — and none may be added. The dispatch is observable
// to tests through ``reduce_prefers_contiguous_blocks`` (compiled directly
// into cpp/tests/test_sum_reduction.cpp, exactly as the H2 matmul test
// reaches ``matmul_prefers_row_sweep`` and the H5 copy test reaches
// ``copy_prefers_contiguous``) and, from Python, by the fact that the two
// traversals are proved to write identical bits on the layouts both can
// address.
//
// ``tf_core_narrow_backward`` — the odometer *dual* of this reduction, a
// scatter rather than an accumulate — is deliberately **out of H6's
// scope** and keeps the generic odometer unchanged.
//
// ---------------------------------------------------------------------------
// Phase I, milestone I4: both traversals carry a scalar type parameter
// ---------------------------------------------------------------------------
//
// ``T`` is **deduced from the pointer arguments**, so every pre-Phase-I call
// site — all of which pass ``double*`` — instantiates ``T = double`` and is
// the pre-I4 traversal statement for statement. ``T = float`` is the same
// source at binary32.
//
// One traversal, two instantiations, is the whole point: the dtypes cannot
// drift apart because there is nothing separate to drift, and float64 keeps
// running the code Phase H measured. The two definitions moved into this
// header for the ordinary reason a template must — so both instantiations
// are available to the exported wrapper in reduction.cpp *and* to the CTests
// that compile it directly — and neither loop, neither carry, and neither
// accumulator changed in the move.
//
// **The accumulator type is exactly ``T``** (design §10.1). A float32
// reduction loads ``float``, accumulates in ``float``, rounds every partial
// sum as binary32, and stores ``float``; there is no widening intermediate,
// no double accumulator, no Kahan compensation, and no reassociation. That
// is what makes the float32 result a genuinely different value from
// "accumulate in binary64 and round once at the end" — which, unlike the
// single correctly-rounded operations of I3, is **observable**, and is
// witnessed directly by test rather than argued.
#pragma once

#include <cstdint>

namespace tf {

// ---------------------------------------------------------------------------
// The dispatch predicate
// ---------------------------------------------------------------------------

// True when this reduction's layout factorizes into three block extents a
// flat traversal can address, false when the generic odometer must be
// used. Total (every input answers), pure (no allocation, no mutation, no
// error state, no global read), and a function of the layout metadata
// alone — never of a pointer value, an alignment, a wall-clock reading, an
// environment variable, or a CPU-feature probe.
//
// On true, ``*outer``, ``*mid``, and ``*inner`` receive the factorization
// (each >= 1, with ``outer * mid * inner`` equal to the element count), and
// the reduction is exactly:
//
//   dst[o * inner + i]  accumulates  src[offset + (o * mid + m) * inner + i]
//
// for o in [0, outer), m in [0, mid), i in [0, inner). On false they are
// left untouched.
//
// All three conditions must hold:
//
//   1. ``in_strides`` is exactly the row-major stride vector implied by
//      ``shape`` — ``in_strides[ndim-1] == 1`` and
//      ``in_strides[d] == in_strides[d+1] * shape[d+1]``. That is the same
//      definition ``NativeTensorView`` uses in backends/cpp.py, so the two
//      layers agree by construction rather than by coincidence. This is
//      what makes the source's logical row-major order a flat ascending
//      walk of memory, and therefore what makes the factorization above
//      address the same elements the odometer visits, in the same order.
//      A transposed, narrowed-on-a-leading-axis, non-unit-strided, or
//      broadcast (stride-0) source fails here.
//   2. The **reduced** axes — those with a write stride of exactly 0, which
//      is how the kernel has always identified them — form **one
//      contiguous run** of axis indices. At least one axis must be
//      reduced. A reduction over non-adjacent axes (which the Python layer
//      cannot currently express, since it reduces a single axis or all of
//      them) fails here and keeps the odometer, rather than H6 growing a
//      general layout compiler.
//   3. The **kept** axes carry exactly the row-major strides of the output
//      formed by dropping that run — checked right to left over the kept
//      axes only. This is what makes ``o * inner + i`` the correct flat
//      destination index. ``keepdims`` is invisible here and needs no
//      special case: a retained reduced axis has write stride 0 either
//      way, and a size-1 output axis contributes nothing to the product.
//
// Adjacent axes are collapsed implicitly rather than by a separate pass:
// conditions 1 and 3 are precisely the statement that adjacent axes of the
// same class have identical address progressions, so the leading kept
// axes, the reduced run, and the trailing kept axes each multiply into a
// single extent. Nothing is cached, interned, or stored anywhere — the
// factorization is recomputed from the arguments on every call.
//
// A false answer is a fallback to the always-correct odometer, never an
// error.
bool reduce_prefers_contiguous_blocks(
    const int64_t* shape, const int64_t* in_strides,
    const int64_t* out_strides, int64_t ndim,
    int64_t* outer, int64_t* mid, int64_t* inner) noexcept;

// ---------------------------------------------------------------------------
// The retained generic reference path
// (docs/native_cpu_performance_design.md §8.3). This is the pre-H6 kernel,
// unchanged: the standard odometer that addresses the source through its
// own shape/strides/offset while an output position advances through
// ``out_strides`` — the row-major stride of each KEPT axis, or 0 for each
// REDUCED axis, so reduced axes accumulate into the same cell. It is
// shipped, reachable through ordinary production dispatch, and is the
// oracle every optimized result is compared against. It is also the only
// path that can address a transposed, narrowed, non-unit-strided, or
// broadcast source at all.
// ---------------------------------------------------------------------------
//
// Accumulation: ``dst[out_pos] += src[in_pos]`` in row-major **source**
// order, so for each destination cell the contributions arrive in
// ascending source order starting from whatever ``dst`` already holds
// (the caller supplies zero-initialized storage, i.e. the additive
// identity). No SIMD horizontal reduction, no FMA, no Kahan
// compensation, no pairwise or tree summation, no wider accumulator.
//
// ``counter`` is a caller-owned scratch array of at least ``ndim``
// ``int64_t`` slots; this function zeroes and uses it, and never allocates
// (the RAII allocation stays in the exported wrapper, where the guard can
// map a failure onto the C ABI error contract).
//
// Preconditions (guaranteed by the exported wrapper and the Core layer;
// NOT re-validated here — this is the inner traversal, not a validation
// boundary): both pointers are non-null; ``ndim >= 1``; every dimension is
// positive; every addressed element lies inside its storage; ``dst``
// aliases no part of ``src``; every integer product and offset is
// representable in int64.
//
// Allocates nothing and cannot throw (noexcept).
template <class T>
inline void sum_generic_strided(
    const T* src, T* dst,
    const int64_t* shape, const int64_t* in_strides,
    const int64_t* out_strides, int64_t offset, int64_t ndim,
    int64_t* counter
) noexcept {
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
        counter[d] = 0;
    }
    int64_t in_pos = offset;
    int64_t out_pos = 0;
    for (int64_t i = 0; i < total; ++i) {
        // The accumulation, in the element type: ``dst`` is a ``T*`` and
        // ``src`` is a ``const T*``, so no operand is widened on the way in
        // and no result is narrowed on the way out.
        dst[out_pos] += src[in_pos];
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            in_pos += in_strides[d];
            out_pos += out_strides[d];
            if (counter[d] < shape[d]) {
                break;
            }
            counter[d] = 0;
            in_pos -= shape[d] * in_strides[d];
            out_pos -= shape[d] * out_strides[d];
        }
    }
}

// ---------------------------------------------------------------------------
// The optimized path: a flat block traversal over the
// outer x mid x inner factorization. Requires
// ``reduce_prefers_contiguous_blocks`` to have returned true and filled
// those three extents.
// ---------------------------------------------------------------------------
//
// Why it is faster: the generic path pays a per-element carry loop — up to
// ``ndim`` increments, comparisons, and two multiply-subtracts — to derive
// the next source and destination addresses, for a body that is one load
// and one add. Here both addresses are affine in the loop counters, so the
// per-element work really is a load and an add, and both the source run
// and the destination run are walked sequentially, which is the access
// pattern the hardware prefetcher is built for. Nothing about the
// arithmetic changes.
//
// The two shapes it takes, and why the split exists:
//
//   * ``inner == 1`` (a full reduction, or one where the reduced run is a
//     suffix): each destination cell is fed by one contiguous ascending
//     source run, so the run is accumulated in a **local accumulator**
//     seeded from ``dst[o]`` and stored once. That removes a
//     store-to-load dependency on memory from every element of the
//     hottest reduction shape there is.
//   * ``inner > 1``: for each ``(o, m)`` a contiguous source row is added
//     elementwise into a contiguous destination row. Each destination
//     element is independent of its neighbours, so this loop may be
//     vectorized by the compiler — that is a vectorization *across
//     distinct outputs*, never a horizontal reduction within one, so no
//     addition is reassociated.
//
// How it relates numerically to the generic path — and unlike H2's matmul
// row sweep this needs **no carve-out at all**:
//
//   1. **Per-output accumulation order is preserved exactly.** For a fixed
//      destination cell the two paths add the same source values, in the
//      same ascending order, starting from the same initial value — the
//      value ``dst`` already holds. The loop nest ``o``, ``m``, ``i`` is
//      the lexicographic order of the source's own row-major index, which
//      is exactly the order the odometer walks, so the *outer* traversal
//      is not even reordered; and every destination cell is touched by
//      exactly one ``(o, i)`` pair, so the cells are independent. No
//      addition is reassociated, no partial sums are combined, no
//      accumulator width changes, no fused multiply-add is requested, and
//      no parallel or vector reduction exists.
//   2. **Every result is bit-identical**, exceptional values included:
//      +0.0 versus -0.0, +/-infinity, denormals, the smallest normal, the
//      largest finite magnitudes, and NaNs — **including their payload and
//      sign bits**, and including a NaN manufactured by the arithmetic
//      itself (``+inf + -inf``). H2's matmul NaN-payload carve-out exists
//      because two NaN operands meet inside an accumulation and x86-64's
//      ADDSD returns the destination operand's NaN, which the loop order
//      influences through instruction selection; here the operand *roles*
//      are unchanged — the running accumulator is always the destination
//      and the freshly loaded source element always the addend, in both
//      paths — so there is nothing for the loop order to change.
//   3. The local accumulator in the ``inner == 1`` branch is seeded from
//      ``dst[o]`` rather than from a literal ``0.0``, which is what keeps
//      the export's documented **accumulate-into** semantics identical on
//      both paths even if a caller ever supplied a non-zero destination.
//      It also means the sum of a run of ``-0.0`` values yields ``+0.0``
//      on both paths, because both start from the destination's ``+0.0``
//      and ``+0.0 + -0.0`` is ``+0.0``.
//
// Why it is safe with respect to H1's output-allocation contract: it is
// not affected by it. This kernel **reads** its destination — that is what
// accumulation means — so ``tf_core_sum`` requires a zero-initialized
// destination on both paths, exactly as it always has, and H6 changes no
// allocation policy. The zeroed buffer is the additive identity, not a
// redundant write.
//
// Layout: ``dst`` is row-major contiguous ``outer x inner`` at offset 0;
// ``src`` is row-major contiguous from ``offset``.
//
// Preconditions: as ``sum_generic_strided``, plus a true answer from
// ``reduce_prefers_contiguous_blocks`` with these extents. Allocates
// nothing (there is no scratch buffer, workspace, or pool) and cannot
// throw (noexcept).
template <class T>
inline void sum_contiguous_blocks(
    const T* src, T* dst,
    int64_t outer, int64_t mid, int64_t inner, int64_t offset
) noexcept {
    if (inner == 1) {
        // One contiguous ascending source run per destination cell. The
        // accumulator is seeded from the destination, so the
        // accumulate-into contract and the signed-zero behavior are
        // exactly the generic path's, and the running total never makes a
        // round trip through memory.
        //
        // It is declared ``T``, not ``double``: at ``T = float`` a
        // ``double`` accumulator here would silently make every float32
        // reduction a mixed-precision one (design §10.1), and it would do
        // so *only* on the optimized path, so the two traversals would
        // stop agreeing. Following the element type is what keeps them
        // the same reduction.
        const T* run = src + offset;
        for (int64_t o = 0; o < outer; ++o) {
            T accumulator = dst[o];
            for (int64_t m = 0; m < mid; ++m) {
                accumulator += run[m];
            }
            dst[o] = accumulator;
            run += mid;
        }
        return;
    }
    // A contiguous source row added elementwise into a contiguous
    // destination row, ``mid`` times per outer block. Distinct ``i`` are
    // distinct destination cells, so nothing here is a horizontal
    // reduction and no addition is reassociated.
    const T* in_row = src + offset;
    T* out_row = dst;
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t m = 0; m < mid; ++m) {
            for (int64_t i = 0; i < inner; ++i) {
                out_row[i] += in_row[i];
            }
            in_row += inner;
        }
        out_row += inner;
    }
}

// ---------------------------------------------------------------------------
// Narrow's backward scatter — the odometer *dual* of the reduction above
// (Phase I, milestone I4 gave it the same scalar type parameter).
// ---------------------------------------------------------------------------
//
// It walks the (smaller) narrowed shape and **assigns** each upstream
// element into its own destination cell: the write position advances by
// ``out_strides`` — the row-major strides of the FULL parent shape, none
// reduced — from a base ``out_offset`` that skips the leading ``start``
// slabs along the narrowed dimension. Narrow regions never overlap, so each
// upstream element maps to exactly one distinct cell and a plain assignment
// is correct.
//
// **This is a scatter, not a reduction**, and the difference is
// load-bearing rather than terminological: its destination stride vector
// has no zeros, so it has no accumulation to preserve and no reduced run to
// factorize — which is why H6 left it alone and why it has one traversal
// rather than two. It is also **not** an identity copy: it writes only the
// narrowed region and every un-narrowed cell keeps the zero the allocation
// gave it, and that zero *is* the gradient there. The caller must therefore
// supply zero-initialized storage at both dtypes (H1 rejected this
// destination explicitly).
//
// Assignment performs no arithmetic, so at both widths it reproduces the
// upstream's object representation exactly — signed zeros and NaN payloads
// included — for the cells it touches.
//
// ``counter`` is a caller-owned scratch array of at least ``ndim``
// ``int64_t`` slots, for the same reason ``sum_generic_strided`` takes one:
// the RAII allocation stays in the exported wrapper, where the guard can map
// a failure onto the C ABI error contract.
//
// Preconditions (guaranteed by the exported wrapper and the Core layer; NOT
// re-validated here): both pointers are non-null; ``ndim >= 1``; every
// dimension is positive; every addressed element lies inside its storage;
// ``dst`` aliases no part of ``upstream``; every integer product and offset
// is representable in int64. Allocates nothing and cannot throw (noexcept).
template <class T>
inline void narrow_backward_scatter(
    const T* upstream, T* dst,
    const int64_t* shape, const int64_t* u_strides,
    const int64_t* out_strides, int64_t u_offset, int64_t out_offset,
    int64_t ndim, int64_t* counter
) noexcept {
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
        counter[d] = 0;
    }
    int64_t u_pos = u_offset;
    int64_t out_pos = out_offset;
    for (int64_t i = 0; i < total; ++i) {
        dst[out_pos] = upstream[u_pos];
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            u_pos += u_strides[d];
            out_pos += out_strides[d];
            if (counter[d] < shape[d]) {
                break;
            }
            counter[d] = 0;
            u_pos -= shape[d] * u_strides[d];
            out_pos -= shape[d] * out_strides[d];
        }
    }
}

}  // namespace tf
