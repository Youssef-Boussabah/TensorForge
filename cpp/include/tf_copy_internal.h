// Internal (non-ABI) declaration of the metadata predicate the production
// strided-to-contiguous gather uses to choose its traversal (Phase H,
// milestone H5). See docs/native_cpu_performance_design.md §17.
//
// Like the H2 matmul predicate and the Conv2d/pooling compute kernels this
// is deliberately NOT part of the public C ABI: a plain C++ function in
// ``namespace tf`` with hidden visibility, reading only the layout
// metadata the exported wrapper already receives. It allocates nothing,
// reports no error, reads no global, and mutates nothing. The exported,
// guarded ``tf_core_contiguous_copy`` wrapper lives in
// cpp/src/elementwise.cpp alongside the two traversals it picks between.
//
// **H5 added no exported symbol.** ``tf_core_contiguous_copy`` keeps the
// signature, the validation, and the ownership contract it has had since
// E3.1; only the loop it runs inside changes. There is no copy-mode
// selector, overlap-mode flag, traversal tracer, environment variable, or
// CPU-feature probe anywhere in the ABI — and none may be added. The
// dispatch is observable to tests through ``copy_prefers_contiguous``
// (compiled directly into cpp/tests/test_contiguous_copy.cpp, exactly as
// the H2 matmul test reaches ``matmul_prefers_row_sweep``) and, from
// Python, by the fact that the two traversals are proved to write
// identical bits.
//
// Why the two paths need no numerical carve-out of the kind H2's row
// sweep needed: the operation is the **identity map**. Both traversals
// evaluate ``dst[out] = op_identity(src[pos])`` on exactly the same
// logical elements in exactly the same row-major destination order; they
// differ only in how ``pos`` is computed. No arithmetic is performed on
// the value at all, so there is no accumulation order to preserve, no
// operand position for an FPU to choose a NaN from, and nothing that can
// quiet a signaling NaN or normalize a signed zero. The results are
// bit-identical for every representable double — including -0.0, both NaN
// signs, every NaN payload, signaling NaNs, denormals, and the largest
// finite magnitudes — and that is a property of the construction, not a
// measured coincidence.
#pragma once

#include <cstdint>

namespace tf {

// True when the logical view described by ``shape``/``strides`` is exactly
// row-major contiguous, so the gather may walk it with a flat pointer loop
// instead of the odometer. Total (every input answers), pure (no
// allocation, no mutation, no error state, no global read), and a function
// of the layout metadata alone — never of a pointer value, an alignment, a
// wall-clock reading, an environment variable, or a CPU-feature probe.
//
// The test is exact equality against the row-major strides implied by
// ``shape``, computed right to left: ``strides[ndim-1] == 1`` and
// ``strides[d] == strides[d+1] * shape[d+1]``. That is precisely the
// definition ``NativeTensorView`` uses in backends/cpp.py, so the two
// layers agree by construction rather than by coincidence. ``ndim == 0``
// (a scalar view) is contiguous: it reads the single element at
// ``offset``, which is what the flat loop with ``numel == 1`` does.
//
// The offset is deliberately **not** consulted: a nonzero offset only
// moves where the flat loop starts, and the exported wrapper has already
// bounds-checked the whole reachable span. A non-positive dimension or a
// stride product that would overflow cannot reach here either — the same
// validation rejects those before any traversal is chosen.
//
// A false answer is a fallback to the always-correct odometer, never an
// error.
bool copy_prefers_contiguous(const int64_t* shape, const int64_t* strides,
                             int64_t ndim) noexcept;

}  // namespace tf
