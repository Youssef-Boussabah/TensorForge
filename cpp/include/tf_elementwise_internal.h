// Internal (non-ABI) elementwise traversal contract: the operation-local
// collapsed layout descriptor the unary and binary kernels build from the
// metadata their exported wrappers already receive, and the templated
// traversals that walk it (Phase H, milestone H8). See
// docs/native_cpu_performance_design.md §19.
//
// Like the H2 matmul predicate, the H5 copy predicate, and the H6 reduction
// predicate this is deliberately NOT part of the public C ABI: plain C++ in
// ``namespace tf`` with hidden visibility, reading only layout metadata. It
// allocates nothing, reports no error, reads no global, and mutates nothing.
// The exported, guarded wrappers live in cpp/src/elementwise.cpp alongside
// the retained generic odometers they fall back to.
//
// **H8 added no exported symbol.** Every wrapper keeps the signature, the
// validation, and the ownership contract it already had; only the loop it
// runs inside changes. There is no traversal selector, plan inspector,
// collapse-mode flag, threshold setter, dispatch tracer, environment
// variable, or CPU-feature probe anywhere in the ABI — and none may be
// added. The dispatch is observable to tests through ``build_unary_plan`` /
// ``build_binary_plan`` (compiled directly into
// cpp/tests/test_elementwise_traversal.cpp, exactly as the H2, H5, and H6
// tests reach their predicates) and, from Python, by the fact that the
// traversals are proved to write identical bits.
//
// ---------------------------------------------------------------------------
// What a plan is, and what it deliberately is not
// ---------------------------------------------------------------------------
//
// A plan is an **operation-local normalized descriptor**: the same logical
// element sequence the odometer would walk, re-expressed over as few axes as
// the address progressions allow. It is built on the stack, used by exactly
// one call, and dropped. Nothing is cached, interned, memoized, or shared
// between calls, and no plan outlives the wrapper that built it. This is not
// a layout compiler and not a general strided-iteration subsystem: it merges
// adjacent axes and drops unit axes, and that is all it does.
//
// Two transformations, both of which preserve the logical element sequence
// exactly:
//
//   * **Unit axes are dropped.** An axis of extent 1 is visited once and
//     contributes nothing to any read or write address.
//   * **Adjacent axes are merged** when, for *every* operand at once, one
//     step of the outer axis is exactly ``extent(inner)`` steps of the inner
//     axis — ``stride[outer] == stride[inner] * extent(inner)``. That is
//     precisely the statement that the two axes' address progressions form
//     one arithmetic run, so walking the merged axis of extent
//     ``extent(outer) * extent(inner)`` with the inner stride visits the same
//     addresses in the same order.
//
// Axes are never reordered, split, or transposed, and the destination is
// untouched by all of this: it is always the freshly allocated row-major
// contiguous output the wrapper allocated, written at ``0, 1, 2, ...``. Since
// merging and dropping preserve the *order* of the remaining axes, the linear
// destination index stays exactly what it was.
//
// **Elementwise outputs are independent.** Every destination element is a
// function of exactly one element of each source; no value is accumulated,
// combined, reassociated, or produced by more than one expression. Reaching a
// given output through a merged axis instead of a carry chain changes only
// *how the address was computed*, never which operands meet in which
// operation or in which order.
//
// ---------------------------------------------------------------------------
// The numerical contract, in four parts, measured rather than assumed
// ---------------------------------------------------------------------------
//
// (1) **Every result in which at most one operand is a NaN is bit-identical**
//     to the pre-H8 kernel's, on every path — signed zeros, infinities,
//     denormals, the smallest normal, the largest finite magnitudes, and a
//     lone NaN of either sign with any payload included. Measured over every
//     ordered pair of 14 IEEE-754 representatives times three operations
//     times five layouts: **zero differing results**.
//
// (2) **NaN positions are identical**, and every NaN any path produces is
//     **quiet**, a signaling-NaN operand included.
//
// (3) **Subtraction is bit-identical everywhere**, two-NaN pairs included, on
//     every layout and against the pre-H8 kernel. It is not commutative, so
//     the compiler has no freedom to choose which operand reaches the
//     destination register.
//
// (4) **For addition and multiplication, when *both* operands are NaN, the
//     payload bits are outside the contract** and are asserted in neither
//     direction. Which of the two NaNs survives is an instruction-selection
//     decision — x86-64's ADDSD/MULSD return the destination operand's NaN,
//     and a commutative operation lets the compiler put either addend there.
//
//     This was **already true before H8** and is not something the collapsed
//     traversal introduced: measured on the pre-H8 library, its flat
//     contiguous kernel and its odometer disagreed on **30 of 196** ordered
//     pairs. H8 *narrows* it. Post-H8 the contiguous, same-shape strided, and
//     row-broadcast paths all agree with each other exactly, and only a
//     transposed operand still differs, on **5 of 196**. The direction of
//     travel is toward agreement, and no path is ever wrong — every result is
//     a quiet NaN in the right position.
//
// This is a weaker claim than "bit-identical, no exceptions", and it is
// stated that way because that is what the measurement showed. It is a
// *different* qualification from H2's and H6's, which concerned NaNs meeting
// inside an accumulation; here there is no accumulation at all, only operand
// order in a single commutative operation.
//
// ---------------------------------------------------------------------------
// Why the traversals are templates and not function-pointer walkers
// ---------------------------------------------------------------------------
//
// The retained odometers take a ``double (*)(double, double)`` and call it
// once per element. That indirect call is not merely a call: it stops the
// compiler from proving anything about the loop body, so the loop cannot be
// unrolled or vectorized. Measured on this machine, over 65,536 contiguous
// elements of ``add``, removing *only* the indirection took the odometer from
// 123.5 us to 81.3 us and removing *only* the carry took it to 63.6 us, while
// removing both took it to 11.5 us — the combination is worth far more than
// either part, because it is what lets the compiler emit a vector loop at
// all.
//
// A vector loop over an elementwise map computes each lane's value with the
// same correctly-rounded operation on the same operands, so every value
// IEEE-754 specifies is reproduced exactly. What it does not pin down is
// which of two NaN operands reaches a commutative instruction's destination
// register — see part (4) of the contract above, which is measured rather
// than assumed and which the pre-H8 kernels already exhibited more widely
// than these do.
//
// That argument holds only for operations IEEE-754 actually specifies, so the
// templated traversals are used **only** for those: addition, subtraction,
// multiplication, square root, ``1.0 / x``, the ReLU comparison-select, its
// backward, and the identity map. ``exp`` and ``log`` are library functions
// with no correctly-rounded guarantee, and a toolchain that vectorized them
// through a vector-math library would be free to return different bits; they
// therefore keep the retained function-pointer paths exactly as they were.
// Nothing is lost by excluding them — measured, the templated traversal is
// 1.05x on both, inside this machine's noise, because a transcendental's own
// cost dominates the traversal completely.
#pragma once

#include <cmath>
#include <cstdint>

namespace tf {

// The most axes a plan holds. Rank 4 is every tensor this runtime can
// construct (NCHW is the widest), and a plan that cannot fit is simply
// rejected — a rejection is a fallback to the retained odometer, never an
// error. A fixed bound is what keeps a plan a stack object with no
// allocation and no failure mode of its own.
inline constexpr std::int64_t ELEMENTWISE_PLAN_AXES = 4;

// The collapsed descriptor for a one-source elementwise operation.
struct ElementwiseUnaryPlan {
    std::int64_t ndim;
    std::int64_t shape[ELEMENTWISE_PLAN_AXES];
    std::int64_t stride[ELEMENTWISE_PLAN_AXES];
};

// The collapsed descriptor for a two-source elementwise operation. Both
// operands are described over the *same* axes, which is what makes a merge
// legal only when it is legal for both at once.
struct ElementwiseBinaryPlan {
    std::int64_t ndim;
    std::int64_t shape[ELEMENTWISE_PLAN_AXES];
    std::int64_t a_stride[ELEMENTWISE_PLAN_AXES];
    std::int64_t b_stride[ELEMENTWISE_PLAN_AXES];
};

// Build the collapsed descriptor, or report that this call keeps the generic
// odometer. Total (every input answers), pure (no allocation, no mutation, no
// error state, no global read), and a function of the layout metadata alone —
// never of a pointer value, an alignment, a wall-clock reading, an
// environment variable, or a CPU-feature probe.
//
// Returns false — meaning "use the retained odometer", never "this call is
// invalid" — when:
//
//   * ``ndim <= 0``. A rank-0 view is one element at ``offset``, which the
//     odometer's own rank-0 branch already handles; folding it in here would
//     duplicate that branch for no gain.
//   * any extent is below 1. The odometer treats such metadata exactly as it
//     always has; the plan declines to reinterpret it.
//   * the logical element count is not representable in int64. The retained
//     odometer multiplies the extents unchecked, so this is metadata neither
//     traversal could honor; declining leaves such a call behaving exactly
//     as it does today rather than guessing at an answer.
//   * the collapsed rank still exceeds ELEMENTWISE_PLAN_AXES.
//   * a product needed to *test* a merge would overflow int64. (A product
//     needed to *perform* one simply leaves the two axes unmerged, which is
//     always a valid description — collapsing is an optimization, never a
//     correctness requirement.)
//
// On false the plan's contents are unspecified and no caller may read them.
bool build_unary_plan(const std::int64_t* shape, const std::int64_t* strides,
                      std::int64_t ndim, ElementwiseUnaryPlan& plan) noexcept;
bool build_binary_plan(const std::int64_t* shape,
                       const std::int64_t* a_strides,
                       const std::int64_t* b_strides, std::int64_t ndim,
                       ElementwiseBinaryPlan& plan) noexcept;

// ---------------------------------------------------------------------------
// The traversals.
//
// ``Op`` is a stateless struct with a ``static`` ``apply(...)``, so the
// operation is a compile-time constant of the loop body and the whole call
// inlines. One instantiation per operation; the operation is never a runtime
// value here.
//
// Each row kernel spells the three layouts that actually occur separately
// rather than folding them into one strided expression, because ``s[i]`` and
// ``s[0]`` are what the compiler can reason about while ``s[i * stride]``
// with a runtime stride is not. The final branch is the fully general one and
// is correct for every stride, negative strides included.
//
// **Every traversal carries a scalar type parameter.** ``T`` is deduced
// from the pointer arguments, so every pre-Phase-I call site — all of which
// pass ``double*`` — instantiates ``T = double`` and compiles unchanged,
// character for character. The unary traversals gained it at milestone I2,
// so the **identity map** could be walked over ``float`` by
// ``tf_core_contiguous_copy``; the binary traversals gained it at milestone
// I3, when ``add``, ``subtract``, ``multiply``, and the ReLU backward
// became dtype-general.
//
// One traversal, two instantiations, is the whole point: the dtypes take
// the *same source*, so they cannot drift apart, and float64 keeps running
// the code Phase H measured.
// ---------------------------------------------------------------------------

template <class Op, class T>
inline void unary_row(const T* src, T* dst, std::int64_t n,
                      std::int64_t stride) {
    if (stride == 1) {
        for (std::int64_t i = 0; i < n; ++i) dst[i] = Op::apply(src[i]);
        return;
    }
    if (stride == 0) {
        for (std::int64_t i = 0; i < n; ++i) dst[i] = Op::apply(src[0]);
        return;
    }
    for (std::int64_t i = 0; i < n; ++i) dst[i] = Op::apply(src[i * stride]);
}

template <class Op, class T>
inline void binary_row(const T* a, const T* b, T* dst,
                       std::int64_t n, std::int64_t a_stride,
                       std::int64_t b_stride) {
    if (a_stride == 1) {
        if (b_stride == 1) {
            for (std::int64_t i = 0; i < n; ++i)
                dst[i] = Op::apply(a[i], b[i]);
            return;
        }
        if (b_stride == 0) {
            for (std::int64_t i = 0; i < n; ++i)
                dst[i] = Op::apply(a[i], b[0]);
            return;
        }
    } else if (a_stride == 0) {
        if (b_stride == 1) {
            for (std::int64_t i = 0; i < n; ++i)
                dst[i] = Op::apply(a[0], b[i]);
            return;
        }
        if (b_stride == 0) {
            for (std::int64_t i = 0; i < n; ++i)
                dst[i] = Op::apply(a[0], b[0]);
            return;
        }
    }
    for (std::int64_t i = 0; i < n; ++i)
        dst[i] = Op::apply(a[i * a_stride], b[i * b_stride]);
}

// Walk a built plan. The destination is written strictly left to right, one
// element per logical position, which is the same order and the same count
// the odometer produces — so H1's "every destination element is written
// exactly once" proof carries over unchanged.
template <class Op, class T>
inline void unary_plan_walk(const T* src, T* dst,
                            const ElementwiseUnaryPlan& plan,
                            std::int64_t offset) {
    src += offset;
    if (plan.ndim == 0) {  // every axis had extent 1
        dst[0] = Op::apply(src[0]);
        return;
    }
    if (plan.ndim == 1) {
        unary_row<Op>(src, dst, plan.shape[0], plan.stride[0]);
        return;
    }
    if (plan.ndim == 2) {
        for (std::int64_t o = 0; o < plan.shape[0]; ++o)
            unary_row<Op>(src + o * plan.stride[0], dst + o * plan.shape[1],
                          plan.shape[1], plan.stride[1]);
        return;
    }
    if (plan.ndim == 3) {
        std::int64_t k = 0;
        for (std::int64_t o = 0; o < plan.shape[0]; ++o)
            for (std::int64_t m = 0; m < plan.shape[1]; ++m, k += plan.shape[2])
                unary_row<Op>(src + o * plan.stride[0] + m * plan.stride[1],
                              dst + k, plan.shape[2], plan.stride[2]);
        return;
    }
    std::int64_t k = 0;
    for (std::int64_t o = 0; o < plan.shape[0]; ++o)
        for (std::int64_t m = 0; m < plan.shape[1]; ++m)
            for (std::int64_t q = 0; q < plan.shape[2]; ++q, k += plan.shape[3])
                unary_row<Op>(src + o * plan.stride[0] + m * plan.stride[1]
                                  + q * plan.stride[2],
                              dst + k, plan.shape[3], plan.stride[3]);
}

template <class Op, class T>
inline void binary_plan_walk(const T* a, const T* b, T* dst,
                             const ElementwiseBinaryPlan& plan,
                             std::int64_t a_offset, std::int64_t b_offset) {
    a += a_offset;
    b += b_offset;
    if (plan.ndim == 0) {
        dst[0] = Op::apply(a[0], b[0]);
        return;
    }
    if (plan.ndim == 1) {
        binary_row<Op>(a, b, dst, plan.shape[0], plan.a_stride[0],
                       plan.b_stride[0]);
        return;
    }
    if (plan.ndim == 2) {
        for (std::int64_t o = 0; o < plan.shape[0]; ++o)
            binary_row<Op>(a + o * plan.a_stride[0], b + o * plan.b_stride[0],
                           dst + o * plan.shape[1], plan.shape[1],
                           plan.a_stride[1], plan.b_stride[1]);
        return;
    }
    if (plan.ndim == 3) {
        std::int64_t k = 0;
        for (std::int64_t o = 0; o < plan.shape[0]; ++o)
            for (std::int64_t m = 0; m < plan.shape[1]; ++m, k += plan.shape[2])
                binary_row<Op>(
                    a + o * plan.a_stride[0] + m * plan.a_stride[1],
                    b + o * plan.b_stride[0] + m * plan.b_stride[1],
                    dst + k, plan.shape[2], plan.a_stride[2],
                    plan.b_stride[2]);
        return;
    }
    std::int64_t k = 0;
    for (std::int64_t o = 0; o < plan.shape[0]; ++o)
        for (std::int64_t m = 0; m < plan.shape[1]; ++m)
            for (std::int64_t q = 0; q < plan.shape[2]; ++q, k += plan.shape[3])
                binary_row<Op>(
                    a + o * plan.a_stride[0] + m * plan.a_stride[1]
                      + q * plan.a_stride[2],
                    b + o * plan.b_stride[0] + m * plan.b_stride[1]
                      + q * plan.b_stride[2],
                    dst + k, plan.shape[3], plan.a_stride[3],
                    plan.b_stride[3]);
}

// The operation functors — the **single** source of every per-element
// expression in this family (Phase I, milestone I3). They live here rather
// than in the .cpp so the CTest can instantiate the traversals with exactly
// the operations production uses, and elementwise.cpp's retained odometers
// take ``&Op::apply<T>`` as their function pointer rather than re-spelling
// each expression beside them: with one definition, the templated traversal
// and the retained reference path *cannot* drift apart, where before they
// merely happened to agree character for character.
//
// **Every ``apply`` is templated on the element type**, so a float32
// operand is loaded as ``float``, combined as ``float``, and stored as
// ``float``, with no double temporary anywhere. That is the whole of
// design §10.1's "float32 accumulates in float32" at this level: a fixed
// ``double apply(double, double)`` reached with float operands would widen,
// compute in binary64, and narrow the result once — a different value from
// the binary32 operation IEEE-754 specifies, and mixed precision by the
// back door. ``T`` is deduced from the arguments, so every pre-Phase-I call
// site instantiates ``T = double`` and is the pre-Phase-I expression
// statement for statement.
//
// Every constant is written ``T(...)`` for the same reason. At ``T =
// double`` it *is* the old literal (``T(0)`` is ``0.0``, ``T(1)`` is
// ``1.0``); at ``T = float`` it keeps the comparison and the division in
// binary32 rather than promoting the whole expression to binary64 around a
// binary64 literal.
struct AddOp {
    template <class T> static inline T apply(T x, T y) { return x + y; }
};
struct SubtractOp {
    template <class T> static inline T apply(T x, T y) { return x - y; }
};
struct MultiplyOp {
    template <class T> static inline T apply(T x, T y) { return x * y; }
};
// x == 0 blocks, matching the Python Tensor's (x > 0) * grad convention.
struct ReluBackwardOp {
    template <class T> static inline T apply(T x, T u) {
        return x > T(0) ? u : T(0);
    }
};
struct ReluOp {
    template <class T> static inline T apply(T x) { return x > T(0) ? x : T(0); }
};
// ``std::sqrt`` is overloaded on the element type, so ``T = float`` selects
// the ``float`` overload and returns ``float`` — no float64 intermediate is
// created and none is narrowed away afterwards.
struct SqrtOp {
    template <class T> static inline T apply(T x) { return std::sqrt(x); }
};
struct ReciprocalOp {
    template <class T> static inline T apply(T x) { return T(1) / x; }
};
// The identity map was the first functor to be templated (Phase I,
// milestone I2), because it was the first instantiated at a second width.
// The template is load-bearing rather than stylistic there in a way it is
// not for the arithmetic above: a fixed ``double apply(double)`` reached
// with a ``float`` operand would convert float -> double -> float around
// the "copy", and a conversion is not a copy. It is exact for every finite
// value and every quiet NaN payload, but it **quiets a signalling NaN** and
// so would break the value-transfer contract (design §10.3) in exactly the
// case the contract is written for. Deducing ``T`` keeps the assignment an
// assignment.
struct IdentityOp {
    template <class T> static inline T apply(T x) { return x; }
};

}  // namespace tf
