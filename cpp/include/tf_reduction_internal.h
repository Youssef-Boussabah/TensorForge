// Internal (non-ABI) declarations of the two sum-reduction traversals the
// production reduction kernel dispatches between, and the metadata
// predicate that chooses (Phase H, milestone H6). See
// docs/native_cpu_performance_design.md §16.6.
//
// Like the H2 matmul paths, the H5 copy predicate, and the Conv2d/pooling
// compute kernels these are deliberately NOT part of the public C ABI:
// plain C++ functions in ``namespace tf`` with hidden visibility, holding
// only float64 traversal and arithmetic. They allocate nothing, report no
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
void sum_generic_strided(
    const double* src, double* dst,
    const int64_t* shape, const int64_t* in_strides,
    const int64_t* out_strides, int64_t offset, int64_t ndim,
    int64_t* counter) noexcept;

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
void sum_contiguous_blocks(
    const double* src, double* dst,
    int64_t outer, int64_t mid, int64_t inner, int64_t offset) noexcept;

}  // namespace tf
