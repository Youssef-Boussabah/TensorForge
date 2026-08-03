// Elementwise kernels: the unary and binary walkers over native tensor
// cores (odometer + contiguous fast paths), ReLU and its backward, the
// v3.11 optimizer-math primitives (sqrt / reciprocal), the Phase-E
// stable math (exponential — milestone E1; logarithm — milestone E2),
// and the legacy raw-buffer elementwise kernels the v0.x benchmarks
// still call.
//
// The walkers read strided views directly from Storage and write into
// fresh contiguous output, so contiguous and non-contiguous inputs
// (transposes, narrows) work identically. Ownership boundary: the caller
// (Python) owns every Storage handle and the fresh output; these kernels
// only compute.

// Phase H, milestone H8 gave every export below whose per-element function
// IEEE-754 actually specifies a second **traversal**: the operation-local
// collapsed plan in tf_elementwise_internal.h, walked by a templated loop
// with no odometer carry and no indirect call. The odometers in this file
// are **retained verbatim as the shipped generic reference path** — they
// remain the only traversal that can address an arbitrary layout, they run
// whenever a plan is rejected, and ``exp``/``log`` still use them
// exclusively. A rejected plan is a fallback, never an error.

#include <cmath>

#include "tf_copy_internal.h"
#include "tf_elementwise_internal.h"
#include "tf_internal.h"

using tf::Storage;
using tf::as_storage;

namespace {

// The retained odometers below still call their operation through a plain
// function pointer, exactly as they always have. What changed at milestone
// I3 is only where that pointer comes from: every operation the templated
// traversals also run now has **one** definition — the functor in
// tf_elementwise_internal.h — and the odometer is handed
// ``&Op::apply<T>``. Before I3 each expression was written twice, here and
// in the header, and kept identical by hand; now there is nothing to keep
// identical. The pointed-to expression is unchanged in both cases.
//
// ``exp`` and ``log`` are the deliberate exception. H8 excluded them from
// the templated traversals — they are library functions with no
// correctly-rounded guarantee, and a toolchain that vectorized them through
// a vector-math library would be free to return different bits — so they
// keep the retained odometer *exclusively*. They therefore have no functor
// in the shared header and cannot be plan-walked even by accident; these
// two file-local function templates are what the odometer calls, and there
// is no ``ExpOp``/``LogOp`` type for a later edit to hand to a plan walk.
//
// Both are templated on the element type so a float32 tensor selects the
// ``float`` overload of ``std::exp`` / ``std::log`` and produces a
// ``float`` — rather than computing in binary64 and narrowing, which would
// be a hidden float64 intermediate (design §8.2, §10.1). ``T = double`` is
// the pre-Phase-I function statement for statement.
//
// Phase E, milestone E1. Plain std::exp: no clamping, no inserted bound,
// no fast approximation. IEEE semantics at the element type: exp(0) == 1,
// a large positive argument overflows to +inf, a large negative one
// underflows toward +0, +inf maps to +inf, -inf maps to +0, and NaN
// propagates — the same values NumPy's exp produces at the same dtype
// (NumPy additionally warns on overflow; this kernel does not).
template <class T> T op_exp(T x) { return std::exp(x); }
// Phase E, milestone E2. Plain std::log: the natural logarithm, with no
// clamping, no inserted epsilon, no absolute value, and no domain
// rejection. IEEE semantics at the element type: log(1) == 0,
// log(±0) == -inf, log(negative) == NaN, log(+inf) == +inf, and NaN
// propagates. The zero and negative cases raise IEEE divide-by-zero /
// invalid *flags*, not C++ exceptions, so they stay numerical results and
// never become ABI errors — the same values NumPy produces (NumPy also
// warns; this kernel does not).
template <class T> T op_log(T x) { return std::log(x); }

// Walk one strided source with the standard odometer and write row-major
// contiguous output.
//
// **The retained generic reference path**, now written over an explicit
// scalar type (Phase I, milestone I2). The loop body, the traversal order,
// the carry, and the indirect call through ``op`` are exactly what they
// were — ``T = double`` reproduces the pre-Phase-I kernel statement for
// statement — and the handle-based wrapper below keeps every existing
// caller's signature. The parameter exists so the identity map, which is
// the one operation I2 generalizes, has the *same* retained fallback at
// both widths rather than a second copy that could drift from this one.
template <class T>
void core_unary_typed(
    const T* src, T* dst,
    const int64_t* shape, const int64_t* strides,
    int64_t offset, int64_t ndim, T (*op)(T)
) {
    if (ndim == 0) {
        dst[0] = op(src[offset]);
        return;
    }
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
    }
    std::vector<int64_t> counter = tf::make_counter(ndim);
    int64_t src_pos = offset;
    for (int64_t out = 0; out < total; ++out) {
        dst[out] = op(src[src_pos]);
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            src_pos += strides[d];
            if (counter[d] < shape[d]) {
                break;
            }
            counter[d] = 0;
            src_pos -= shape[d] * strides[d];
        }
    }
}

// The handle wrapper every caller uses: it resolves the two handles at the
// element type its caller has already dispatched on and runs the retained
// odometer above. ``T = double`` resolves through the same static_cast
// ``tf::storage_f64`` performs and reproduces the pre-Phase-I wrapper
// exactly; ``T = float`` is the same wrapper over a float32 buffer.
//
// The typed accessor is sound because the caller established the tag first:
// a generalized export reads it once through ``tf::dispatch_dtype``, after
// ``tf::require_matching_dtype`` has proved every operand agrees, so the
// recovered pointer addresses a genuine ``T[]`` array (design §4.1).
template <class T>
void core_unary(
    const void* src_handle, void* dst_handle,
    const int64_t* shape, const int64_t* strides,
    int64_t offset, int64_t ndim, T (*op)(T)
) {
    core_unary_typed<T>(tf::storage_typed<T>(src_handle),
                        tf::storage_typed<T>(dst_handle),
                        shape, strides, offset, ndim, op);
}

// Contiguous fast path: flat, index-free loop. Scalars fall out as
// numel == 1; a nonzero offset starts from data + offset.
template <class T>
void core_unary_contiguous(
    const void* src_handle, void* dst_handle,
    int64_t numel, int64_t offset, T (*op)(T)
) {
    const T* src = tf::storage_typed<T>(src_handle) + offset;
    T* dst = tf::storage_typed<T>(dst_handle);
    for (int64_t i = 0; i < numel; ++i) {
        dst[i] = op(src[i]);
    }
}

// Walk two strided sources in lockstep (same logical shape, separate
// strides/offsets) and write row-major contiguous output.
//
// **The retained generic reference path for two-source operations**, over
// an explicit scalar type since milestone I3. The loop body, the traversal
// order, the carry, and the indirect call through ``op`` are exactly what
// they were — ``T = double`` reproduces the pre-Phase-I kernel statement
// for statement — and it remains the only traversal that can address an
// arbitrary layout, so it is still what every rejected plan falls back to,
// now at both widths from one source.
template <class T>
void core_binary_typed(
    const T* a, const T* b, T* dst,
    const int64_t* shape, const int64_t* a_strides, const int64_t* b_strides,
    int64_t a_offset, int64_t b_offset, int64_t ndim, T (*op)(T, T)
) {
    if (ndim == 0) {
        dst[0] = op(a[a_offset], b[b_offset]);
        return;
    }
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
    }
    std::vector<int64_t> counter = tf::make_counter(ndim);
    int64_t a_pos = a_offset;
    int64_t b_pos = b_offset;
    for (int64_t out = 0; out < total; ++out) {
        dst[out] = op(a[a_pos], b[b_pos]);
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            a_pos += a_strides[d];
            b_pos += b_strides[d];
            if (counter[d] < shape[d]) {
                break;
            }
            counter[d] = 0;
            a_pos -= shape[d] * a_strides[d];
            b_pos -= shape[d] * b_strides[d];
        }
    }
}

// The handle wrapper, the two-source twin of ``core_unary`` above and sound
// for the same reason: the caller dispatched on the tag first.
template <class T>
void core_binary(
    const void* a_handle, const void* b_handle, void* dst_handle,
    const int64_t* shape, const int64_t* a_strides, const int64_t* b_strides,
    int64_t a_offset, int64_t b_offset, int64_t ndim, T (*op)(T, T)
) {
    core_binary_typed<T>(tf::storage_typed<T>(a_handle),
                         tf::storage_typed<T>(b_handle),
                         tf::storage_typed<T>(dst_handle),
                         shape, a_strides, b_strides, a_offset, b_offset, ndim,
                         op);
}

// (The pre-H8 flat binary loop that used to live here — the same
// ``dst[i] = op(a[i], b[i])`` walk behind a function pointer — is now
// ``tf::binary_row<Op>`` with both strides 1, in
// tf_elementwise_internal.h. It is the identical loop with the operation
// as a compile-time constant instead of an indirect call, so it reads the
// same elements in the same order and writes the same bits; nothing was
// kept here as a second copy that could drift. The *generic* pre-H8
// reference path, ``core_binary`` above, is retained verbatim and is what
// every rejected plan still falls back to.)

// -- trust-boundary validation for the guarded unary exports ----------------
//
// Added with the Phase-E exponential (E1) and used by its two exports. The
// older unary exports (relu/sqrt/reciprocal) predate the self-validating
// convention the Phase-C/D exports established and are deliberately left
// exactly as they are — E1 tightens only what it adds
// (docs/native_classification_design.md §9.3: "raw ABI validation is
// self-contained", because these symbols are reachable by any ctypes
// caller, not only by the Python wrapper that already validates).
//
// Each helper returns nullptr when the call is safe, otherwise a static
// message (no allocation, so validation itself can never fail). Every
// product and sum is overflow-checked, so a bogus dimension cannot wrap
// int64 into a small span that would pass a naive bounds test.

// Checked int64 multiply/add for non-negative operands. File-local, like
// the equivalents in conv2d.cpp and pooling.cpp (each compute unit keeps
// its own rather than growing a premature shared surface).
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

// The destination is always freshly allocated row-major contiguous
// storage of exactly ``numel`` elements starting at index 0.
const char* check_destination(void* dst_handle, int64_t numel) {
    if (as_storage(dst_handle)->size < numel) {
        return "unary kernel: output storage smaller than the element count";
    }
    return nullptr;
}

// A strided source view: validate the layout metadata, then prove that
// every element the odometer will touch lies inside the source storage.
// Strides may in principle be negative, so both the lowest and the
// highest reachable index are bounded (rather than assuming the walk only
// moves forward from ``offset``).
const char* unary_strided_error(
    const void* src_handle, void* dst_handle,
    const int64_t* shape, const int64_t* strides,
    int64_t offset, int64_t ndim
) {
    if (src_handle == nullptr || dst_handle == nullptr) {
        return "unary kernel: null storage handle";
    }
    if (ndim < 0) {
        return "unary kernel: negative ndim";
    }
    if (ndim > 0 && (shape == nullptr || strides == nullptr)) {
        return "unary kernel: null shape or stride array";
    }
    if (offset < 0) {
        return "unary kernel: negative offset";
    }
    const int64_t src_size = as_storage(src_handle)->size;
    // Scalars (ndim == 0) read exactly one element, at ``offset``.
    if (ndim == 0) {
        if (offset >= src_size) {
            return "unary kernel: offset outside the input storage";
        }
        return check_destination(dst_handle, 1);
    }
    int64_t numel = 1;
    // Total travel away from ``offset``, split by direction: axis d moves
    // the read position by strides[d] * k for k in [0, shape[d] - 1].
    int64_t forward_travel = 0;
    int64_t backward_travel = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        if (shape[d] < 1) {
            return "unary kernel: non-positive dimension";
        }
        if (!checked_mul(numel, shape[d], numel)) {
            return "unary kernel: element count overflows int64";
        }
        const int64_t stride = strides[d];
        if (stride == INT64_MIN) {
            // Negating INT64_MIN is undefined behavior, so reject it
            // before taking the magnitude below.
            return "unary kernel: stride arithmetic overflows int64";
        }
        int64_t magnitude;
        if (!checked_mul(stride < 0 ? -stride : stride, shape[d] - 1,
                         magnitude)) {
            return "unary kernel: stride arithmetic overflows int64";
        }
        const bool ok = stride < 0
            ? checked_add(backward_travel, magnitude, backward_travel)
            : checked_add(forward_travel, magnitude, forward_travel);
        if (!ok) {
            return "unary kernel: stride arithmetic overflows int64";
        }
    }
    int64_t highest;  // offset + forward_travel, the last index touched
    if (!checked_add(offset, forward_travel, highest)) {
        return "unary kernel: stride arithmetic overflows int64";
    }
    // lowest = offset - backward_travel; both are non-negative, so the
    // subtraction cannot overflow.
    if (backward_travel > offset || highest >= src_size) {
        return "unary kernel: input span exceeds its storage";
    }
    return check_destination(dst_handle, numel);
}

// The contiguous fast path takes ``numel`` and an ``offset`` instead of
// layout arrays: the source run is [offset, offset + numel).
const char* unary_contiguous_error(
    const void* src_handle, void* dst_handle, int64_t numel, int64_t offset
) {
    if (src_handle == nullptr || dst_handle == nullptr) {
        return "unary kernel: null storage handle";
    }
    if (numel < 0 || offset < 0) {
        return "unary kernel: negative element count or offset";
    }
    int64_t end;
    if (!checked_add(offset, numel, end) ||
        end > as_storage(src_handle)->size) {
        return "unary kernel: input span exceeds its storage";
    }
    return check_destination(dst_handle, numel);
}

// -- Phase H, milestone H8: the collapsed-plan traversal --------------------
//
// Two helpers keep every dispatch site below to one shape: resolve the
// handles, build the plan, and either walk it with the templated traversal
// or hand the call to the retained odometer. The plan is a stack object of
// this call and nothing outlives the helper.

// Checked signed multiply. Nothing above this file has proved that an
// arbitrary ctypes caller's strides and extents multiply without wrapping —
// the older binary exports validate nothing at all — and signed overflow is
// undefined behavior, so every product the collapse needs is guarded and an
// unprovable one simply declines the plan.
bool plan_checked_mul(int64_t x, int64_t y, int64_t& out) {
    if (x == 0 || y == 0) {
        out = 0;
        return true;
    }
    if (x > 0) {
        if (y > 0) {
            if (x > INT64_MAX / y) return false;
        } else if (y < INT64_MIN / x) {
            return false;
        }
    } else if (y > 0) {
        if (x < INT64_MIN / y) return false;
    } else if (x < INT64_MAX / y) {
        return false;
    }
    out = x * y;
    return true;
}

template <class Op, class T>
void unary_dispatch(
    const void* src_handle, void* dst_handle,
    const int64_t* shape, const int64_t* strides,
    int64_t offset, int64_t ndim
) {
    tf::ElementwiseUnaryPlan plan;
    if (tf::build_unary_plan(shape, strides, ndim, plan)) {
        tf::unary_plan_walk<Op>(tf::storage_typed<T>(src_handle),
                                tf::storage_typed<T>(dst_handle), plan, offset);
        return;
    }
    core_unary<T>(src_handle, dst_handle, shape, strides, offset, ndim,
                  &Op::template apply<T>);
}

template <class Op, class T>
void binary_dispatch(
    const void* a_handle, const void* b_handle, void* dst_handle,
    const int64_t* shape, const int64_t* a_strides, const int64_t* b_strides,
    int64_t a_offset, int64_t b_offset, int64_t ndim
) {
    tf::ElementwiseBinaryPlan plan;
    if (tf::build_binary_plan(shape, a_strides, b_strides, ndim, plan)) {
        tf::binary_plan_walk<Op>(tf::storage_typed<T>(a_handle),
                                 tf::storage_typed<T>(b_handle),
                                 tf::storage_typed<T>(dst_handle), plan,
                                 a_offset, b_offset);
        return;
    }
    core_binary<T>(a_handle, b_handle, dst_handle, shape, a_strides, b_strides,
                   a_offset, b_offset, ndim, &Op::template apply<T>);
}

// The contiguous exports already state "one flat run of ``numel`` elements
// from ``offset``", which is a rank-1 plan with stride 1 — so they need no
// plan at all, only the templated row.
template <class Op, class T>
void unary_contiguous_dispatch(
    const void* src_handle, void* dst_handle, int64_t numel, int64_t offset
) {
    tf::unary_row<Op>(tf::storage_typed<T>(src_handle) + offset,
                      tf::storage_typed<T>(dst_handle), numel, 1);
}

template <class Op, class T>
void binary_contiguous_dispatch(
    const void* a_handle, const void* b_handle, void* dst_handle,
    int64_t numel, int64_t a_offset, int64_t b_offset
) {
    tf::binary_row<Op>(tf::storage_typed<T>(a_handle) + a_offset,
                       tf::storage_typed<T>(b_handle) + b_offset,
                       tf::storage_typed<T>(dst_handle), numel, 1, 1);
}

// -- Phase I, milestone I3: the one dispatch per exported call --------------
//
// Four helpers, one per traversal family, each holding the **single**
// ``switch`` its exports perform (design §8.1). They sit above the
// traversals and below the exports, so the dtype decision is made exactly
// once — after the export's validation and before any compute — and nothing
// beneath them branches on dtype again: not the plan builder, not the row
// kernel, not the odometer carry, not the operation functor, and
// emphatically not any element.
//
// Both arms take the *same source*. ``T = double`` is the pre-I3 call
// statement for statement, so Phase H's measured float64 traversal is
// preserved rather than re-derived, and ``T = float`` cannot drift from it
// because there is nothing separate to drift.
//
// The dtype comes from ``tf::dispatch_dtype``, which reads the storage tag —
// layout and operand metadata only, never a pointer value, an alignment, a
// clock, an environment variable, or a CPU-feature probe. The ``switch`` has
// no ``default:`` label, so a future dtype without an instantiation is a
// compile-time warning rather than a silent misread.

template <class Op>
void unary_by_dtype(
    const void* src_handle, void* dst_handle,
    const int64_t* shape, const int64_t* strides,
    int64_t offset, int64_t ndim
) {
    switch (tf::dispatch_dtype({src_handle, dst_handle})) {
        case tf::Dtype::Float32:
            unary_dispatch<Op, float>(src_handle, dst_handle, shape, strides,
                                      offset, ndim);
            return;
        case tf::Dtype::Float64:
            break;
    }
    unary_dispatch<Op, double>(src_handle, dst_handle, shape, strides, offset,
                               ndim);
}

template <class Op>
void unary_contiguous_by_dtype(
    const void* src_handle, void* dst_handle, int64_t numel, int64_t offset
) {
    switch (tf::dispatch_dtype({src_handle, dst_handle})) {
        case tf::Dtype::Float32:
            unary_contiguous_dispatch<Op, float>(src_handle, dst_handle, numel,
                                                 offset);
            return;
        case tf::Dtype::Float64:
            break;
    }
    unary_contiguous_dispatch<Op, double>(src_handle, dst_handle, numel,
                                          offset);
}

template <class Op>
void binary_by_dtype(
    const void* a_handle, const void* b_handle, void* dst_handle,
    const int64_t* shape, const int64_t* a_strides, const int64_t* b_strides,
    int64_t a_offset, int64_t b_offset, int64_t ndim
) {
    switch (tf::dispatch_dtype({a_handle, b_handle, dst_handle})) {
        case tf::Dtype::Float32:
            binary_dispatch<Op, float>(a_handle, b_handle, dst_handle, shape,
                                       a_strides, b_strides, a_offset, b_offset,
                                       ndim);
            return;
        case tf::Dtype::Float64:
            break;
    }
    binary_dispatch<Op, double>(a_handle, b_handle, dst_handle, shape, a_strides,
                                b_strides, a_offset, b_offset, ndim);
}

template <class Op>
void binary_contiguous_by_dtype(
    const void* a_handle, const void* b_handle, void* dst_handle,
    int64_t numel, int64_t a_offset, int64_t b_offset
) {
    switch (tf::dispatch_dtype({a_handle, b_handle, dst_handle})) {
        case tf::Dtype::Float32:
            binary_contiguous_dispatch<Op, float>(a_handle, b_handle, dst_handle,
                                                  numel, a_offset, b_offset);
            return;
        case tf::Dtype::Float64:
            break;
    }
    binary_contiguous_dispatch<Op, double>(a_handle, b_handle, dst_handle, numel,
                                           a_offset, b_offset);
}

}  // namespace

// The plan builders. Hidden-visibility C++ in ``namespace tf``; see
// cpp/include/tf_elementwise_internal.h for the full contract.
namespace tf {

bool build_unary_plan(const int64_t* shape, const int64_t* strides,
                      int64_t ndim, ElementwiseUnaryPlan& plan) noexcept {
    if (ndim <= 0) {
        return false;
    }
    int64_t n = 0;
    int64_t count = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (shape[d] < 1) {
            return false;
        }
        // The logical element count must be representable. The retained
        // odometer multiplies the extents unchecked, so metadata this
        // rejects is metadata neither traversal could honor — declining it
        // here simply leaves such a call behaving exactly as it does today.
        if (!plan_checked_mul(count, shape[d], count)) {
            return false;
        }
        if (shape[d] == 1) {
            // Visited once, contributing nothing to any address.
            continue;
        }
        int64_t promoted, merged;
        if (n > 0 && plan_checked_mul(strides[d], shape[d], promoted)
                && plan.stride[n - 1] == promoted
                && plan_checked_mul(plan.shape[n - 1], shape[d], merged)) {
            plan.shape[n - 1] = merged;
            plan.stride[n - 1] = strides[d];
            continue;
        }
        if (n == ELEMENTWISE_PLAN_AXES) {
            return false;
        }
        plan.shape[n] = shape[d];
        plan.stride[n] = strides[d];
        ++n;
    }
    plan.ndim = n;
    return true;
}

bool build_binary_plan(const int64_t* shape, const int64_t* a_strides,
                       const int64_t* b_strides, int64_t ndim,
                       ElementwiseBinaryPlan& plan) noexcept {
    if (ndim <= 0) {
        return false;
    }
    int64_t n = 0;
    int64_t count = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (shape[d] < 1) {
            return false;
        }
        if (!plan_checked_mul(count, shape[d], count)) {
            return false;
        }
        if (shape[d] == 1) {
            continue;
        }
        int64_t a_promoted, b_promoted, merged;
        // A merge must be legal for *both* operands at once: one step of the
        // outer axis has to be exactly ``shape[d]`` steps of this axis in
        // each of them, or the merged axis would visit different addresses.
        if (n > 0 && plan_checked_mul(a_strides[d], shape[d], a_promoted)
                && plan_checked_mul(b_strides[d], shape[d], b_promoted)
                && plan.a_stride[n - 1] == a_promoted
                && plan.b_stride[n - 1] == b_promoted
                && plan_checked_mul(plan.shape[n - 1], shape[d], merged)) {
            plan.shape[n - 1] = merged;
            plan.a_stride[n - 1] = a_strides[d];
            plan.b_stride[n - 1] = b_strides[d];
            continue;
        }
        if (n == ELEMENTWISE_PLAN_AXES) {
            return false;
        }
        plan.shape[n] = shape[d];
        plan.a_stride[n] = a_strides[d];
        plan.b_stride[n] = b_strides[d];
        ++n;
    }
    plan.ndim = n;
    return true;
}

}  // namespace tf

// -- ReLU over tensor cores --------------------------------------------------
//
// Phase I, milestone I3: this export and every other in the elementwise
// family below became dtype-general. Each keeps the symbol, the argument
// list, the calling convention, the traversal tiers, and the ownership
// contract it already had — the only change is that
// ``tf::require_float64`` ("this operation has not been generalized")
// became ``tf::require_matching_dtype`` ("it has been, and its operands
// must agree"), and that one ``switch`` now selects the instantiation.
//
// Operand dtypes must **match**: there is no casting, no promotion, and no
// mixed-dtype arithmetic anywhere in the runtime (design §9), so float32
// with float64 is an invalid *request* rather than a conversion
// opportunity. The rejection is recorded before anything is read or
// written, so a rejected call leaves every destination byte-for-byte
// unchanged.

TF_EXPORT void tf_core_relu(
    const void* src, void* dst,
    const int64_t* shape, const int64_t* strides,
    int64_t offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_relu", {src, dst})) {
        return;
    }
    unary_by_dtype<tf::ReluOp>(src, dst, shape, strides, offset, ndim);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_relu_contiguous(
    const void* src, void* dst, int64_t numel, int64_t offset
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_relu_contiguous", {src, dst})) {
        return;
    }
    unary_contiguous_by_dtype<tf::ReluOp>(src, dst, numel, offset);
    TF_GUARD_END_VOID()
}

// -- optimizer math primitives (v3.11) --------------------------------------

TF_EXPORT void tf_core_sqrt(
    const void* src, void* dst,
    const int64_t* shape, const int64_t* strides, int64_t offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_sqrt", {src, dst})) {
        return;
    }
    unary_by_dtype<tf::SqrtOp>(src, dst, shape, strides, offset, ndim);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_sqrt_contiguous(
    const void* src, void* dst, int64_t numel, int64_t offset
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_sqrt_contiguous", {src, dst})) {
        return;
    }
    unary_contiguous_by_dtype<tf::SqrtOp>(src, dst, numel, offset);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_reciprocal(
    const void* src, void* dst,
    const int64_t* shape, const int64_t* strides, int64_t offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_reciprocal", {src, dst})) {
        return;
    }
    unary_by_dtype<tf::ReciprocalOp>(src, dst, shape, strides, offset, ndim);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_reciprocal_contiguous(
    const void* src, void* dst, int64_t numel, int64_t offset
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_reciprocal_contiguous",
                                    {src, dst})) {
        return;
    }
    unary_contiguous_by_dtype<tf::ReciprocalOp>(src, dst, numel, offset);
    TF_GUARD_END_VOID()
}

// -- strided-to-contiguous storage gather (E3.1) ----------------------------
//
// The shared native materialization path: read any strided/offset view of
// one storage and write its logical elements, in row-major order, into a
// second storage. This is the storage-to-storage twin of
// ``tf_storage_materialize`` (storage.cpp), which gathers into a raw
// caller-supplied double* — i.e. a NumPy buffer — and therefore cannot
// serve a Core-to-Core copy without exporting tensor data to the host.
//
// ``NativeTensorCore.contiguous_copy`` is built on this, so every
// Policy-B copy-then-compute path (softmax, conv2d, maxpool2d), the
// differentiable ``contiguous_copy`` operation, ``NativeFlatten``, and
// ``NativeParameter`` construction now keep tensor values in native
// memory for the whole copy.
//
// It reuses this file's existing pieces rather than re-deriving them: the
// same ``core_unary`` odometer that every strided unary op walks with,
// and the same ``unary_strided_error`` trust-boundary validation the E1/E2
// exports use (handles, layout metadata, spans in both stride directions,
// overflow, destination capacity) — so the copy inherits validation that
// is already exercised by the exp and log CTests. The operation is the
// identity map; only the traversal matters.

namespace {
// Templated for the same reason ``tf::IdentityOp::apply`` is (Phase I,
// milestone I2): a fixed ``double(double)`` reached with a float operand
// would convert float -> double -> float, and a conversion quiets a
// signalling NaN. Deducing the type keeps the identity an identity.
template <class T>
T op_identity(T x) { return x; }
}  // namespace

// Phase H, milestone H5: the traversal predicate. See
// cpp/include/tf_copy_internal.h for the full contract — total, pure, a
// function of layout metadata alone, and a false answer is a fallback
// rather than an error.
namespace tf {
bool copy_prefers_contiguous(const int64_t* shape, const int64_t* strides,
                             int64_t ndim) noexcept {
    if (ndim <= 0) {
        // A scalar view reads exactly one element at ``offset``, which is
        // what the flat loop does with numel == 1. (ndim < 0 cannot reach
        // here: the wrapper's validation rejects it first. Folding it in
        // keeps the predicate total.)
        return ndim == 0;
    }
    // Row-major strides, right to left, compared for exact equality —
    // the same definition NativeTensorView uses in backends/cpp.py. The
    // running product cannot overflow: the wrapper has already proved
    // that this shape's element count fits in int64.
    int64_t expected = 1;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (strides[d] != expected) {
            return false;
        }
        expected *= shape[d];
    }
    return true;
}
}  // namespace tf

// The three-tier traversal, over an explicit scalar type (Phase I,
// milestone I2). ``T = double`` is the pre-I2 body statement for
// statement — the same predicate, the same plan builder, the same flat
// row, the same retained odometer, in the same order — so Phase H's
// measured float64 behavior is preserved rather than re-derived.
//
// H5: pick the traversal from the metadata already in hand. A row-major
// source is swept with the flat pointer loop every other unary op's
// contiguous path already uses; anything else keeps the generic odometer,
// which is the retained reference traversal and the only one that can
// address a transposed, narrowed, or negatively strided view at all. Both
// write ``dst[out] = src[pos]`` over the same logical elements in the same
// row-major destination order, so they are bit-identical by construction —
// the identity map performs no arithmetic, so no signed zero can be
// normalized and no NaN can be quieted or have its payload chosen on any
// path.
//
// H8 keeps both of H5's tiers and adds a middle one. The row-major tier
// still runs the flat loop, now with the identity as a compile-time
// constant of the loop body rather than an indirect call; a source H5's
// predicate rejects gets the collapsed plan; and a plan the builder rejects
// still gets the odometer, which remains the only traversal that can
// address an arbitrary layout. The bit-identity argument is unchanged and
// covers all three, at **both** dtypes: the identity map performs no
// arithmetic at all, which is a property of the operation rather than of
// the element width.
//
// The plan builder and the predicate are untouched by dtype: both read
// ``int64`` layout metadata only, which is exactly why they carry over.
namespace {
template <class T>
void copy_dispatch(
    const void* src_handle, void* dst_handle,
    const int64_t* shape, const int64_t* strides, int64_t offset, int64_t ndim
) {
    const T* src = tf::storage_typed<T>(src_handle);
    T* dst = tf::storage_typed<T>(dst_handle);
    if (tf::copy_prefers_contiguous(shape, strides, ndim)) {
        int64_t numel = 1;
        for (int64_t d = 0; d < ndim; ++d) {
            numel *= shape[d];
        }
        tf::unary_row<tf::IdentityOp>(src + offset, dst, numel, 1);
        return;
    }
    tf::ElementwiseUnaryPlan plan;
    if (tf::build_unary_plan(shape, strides, ndim, plan)) {
        tf::unary_plan_walk<tf::IdentityOp>(src, dst, plan, offset);
        return;
    }
    core_unary_typed<T>(src, dst, shape, strides, offset, ndim, op_identity<T>);
}
}  // namespace

// Phase I, milestone I2: the one handle-based compute export the milestone
// generalizes, and the only one — every other kernel keeps
// ``tf::require_float64`` until its own milestone.
//
// It is generalized here because it is not really arithmetic: it is the
// runtime's **value-transfer primitive** (H5), the storage-to-storage twin
// of ``tf_storage_materialize``, and the thing every Policy-B
// copy-then-compute path, ``NativeFlatten``, ``NativeParameter``
// construction, and the differentiable ``contiguous_copy`` are built on.
// Transfer is what I2 is about.
//
// Source and destination dtypes must **agree**: there is no casting and no
// promotion anywhere in the runtime, so float32 -> float64 is an invalid
// request rather than a conversion. The rejection is recorded before the
// span validation runs and long before any element is written, so a
// rejected call leaves the destination byte-for-byte unchanged.
TF_EXPORT void tf_core_contiguous_copy(
    const void* src, void* dst,
    const int64_t* shape, const int64_t* strides, int64_t offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_contiguous_copy", src, dst)) {
        return;
    }
    if (const char* err =
            unary_strided_error(src, dst, shape, strides, offset, ndim)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    // The one dispatch: after validation, before any compute, reading the
    // dtype from the handle the caller already passed. Nothing below this
    // point branches on dtype — not the predicate, not the plan builder,
    // not the row kernel, not the odometer carry, and emphatically not any
    // element.
    switch (tf::storage_dtype(src)) {
        case tf::Dtype::Float32:
            copy_dispatch<float>(src, dst, shape, strides, offset, ndim);
            return;
        case tf::Dtype::Float64:
            break;
    }
    copy_dispatch<double>(src, dst, shape, strides, offset, ndim);
    TF_GUARD_END_VOID()
}

// -- Phase E stable math: exponential (E1) and logarithm (E2) ---------------
//
// Same two-path shape as every other unary core op — a generic odometer
// export for strided views and a flat contiguous export — so
// NativeTensorCore.exp()/log() dispatch exactly like relu/sqrt/reciprocal
// and both paths produce bit-for-bit identical values.
//
// Unlike the older unary exports these four **validate their own
// arguments** before touching memory, through the shared helpers above
// (E2 reuses E1's validators unchanged — the messages were written
// op-agnostic for exactly this reason). Validation runs inside the guard,
// so a rejected call records TF_ERROR_INVALID in the thread-local slot,
// writes nothing, allocates nothing, leaves every caller-owned object
// untouched, and surfaces in Python as ValueError.
//
// IEEE domain results (NaN from log of a negative, ±inf, ±0) are
// **values**, not failures: they flow to the destination and leave the
// error slot clear.
//
// Phase I, milestone I3 generalized these four to both dtypes, and they are
// the family's one structural exception: they hold H8's exclusion from the
// templated traversals, so they dispatch **straight into the retained
// odometer** rather than through a plan, at either width. The dtype
// ``switch`` is therefore written out here instead of going through the
// ``*_by_dtype`` helpers above — there is no functor to hand them, which is
// exactly the point (a later edit cannot plan-walk an operation that has no
// functor). The validation order is unchanged apart from the dtype guard
// running first, so a mixed-dtype call reports the dtype rather than having
// its rejection overwritten by a span error.

TF_EXPORT void tf_core_exp(
    const void* src, void* dst,
    const int64_t* shape, const int64_t* strides, int64_t offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_exp", {src, dst})) {
        return;
    }
    if (const char* err =
            unary_strided_error(src, dst, shape, strides, offset, ndim)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    switch (tf::dispatch_dtype({src, dst})) {
        case tf::Dtype::Float32:
            core_unary<float>(src, dst, shape, strides, offset, ndim,
                              op_exp<float>);
            return;
        case tf::Dtype::Float64:
            break;
    }
    core_unary<double>(src, dst, shape, strides, offset, ndim, op_exp<double>);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_exp_contiguous(
    const void* src, void* dst, int64_t numel, int64_t offset
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_exp_contiguous", {src, dst})) {
        return;
    }
    if (const char* err = unary_contiguous_error(src, dst, numel, offset)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    switch (tf::dispatch_dtype({src, dst})) {
        case tf::Dtype::Float32:
            core_unary_contiguous<float>(src, dst, numel, offset,
                                         op_exp<float>);
            return;
        case tf::Dtype::Float64:
            break;
    }
    core_unary_contiguous<double>(src, dst, numel, offset, op_exp<double>);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_log(
    const void* src, void* dst,
    const int64_t* shape, const int64_t* strides, int64_t offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_log", {src, dst})) {
        return;
    }
    if (const char* err =
            unary_strided_error(src, dst, shape, strides, offset, ndim)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    switch (tf::dispatch_dtype({src, dst})) {
        case tf::Dtype::Float32:
            core_unary<float>(src, dst, shape, strides, offset, ndim,
                              op_log<float>);
            return;
        case tf::Dtype::Float64:
            break;
    }
    core_unary<double>(src, dst, shape, strides, offset, ndim, op_log<double>);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_log_contiguous(
    const void* src, void* dst, int64_t numel, int64_t offset
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_log_contiguous", {src, dst})) {
        return;
    }
    if (const char* err = unary_contiguous_error(src, dst, numel, offset)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    switch (tf::dispatch_dtype({src, dst})) {
        case tf::Dtype::Float32:
            core_unary_contiguous<float>(src, dst, numel, offset,
                                         op_log<float>);
            return;
        case tf::Dtype::Float64:
            break;
    }
    core_unary_contiguous<double>(src, dst, numel, offset, op_log<double>);
    TF_GUARD_END_VOID()
}

// -- binary ops over tensor cores -------------------------------------------

TF_EXPORT void tf_core_add(
    const void* a, const void* b, void* dst,
    const int64_t* shape, const int64_t* a_strides, const int64_t* b_strides,
    int64_t a_offset, int64_t b_offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_add", {a, b, dst})) {
        return;
    }
    binary_by_dtype<tf::AddOp>(a, b, dst, shape, a_strides, b_strides,
                               a_offset, b_offset, ndim);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_subtract(
    const void* a, const void* b, void* dst,
    const int64_t* shape, const int64_t* a_strides, const int64_t* b_strides,
    int64_t a_offset, int64_t b_offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_subtract", {a, b, dst})) {
        return;
    }
    binary_by_dtype<tf::SubtractOp>(a, b, dst, shape, a_strides, b_strides,
                                    a_offset, b_offset, ndim);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_multiply(
    const void* a, const void* b, void* dst,
    const int64_t* shape, const int64_t* a_strides, const int64_t* b_strides,
    int64_t a_offset, int64_t b_offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_multiply", {a, b, dst})) {
        return;
    }
    binary_by_dtype<tf::MultiplyOp>(a, b, dst, shape, a_strides, b_strides,
                                    a_offset, b_offset, ndim);
    TF_GUARD_END_VOID()
}

// ReLU backward over tensor cores: dst = upstream where x > 0, else 0.
// The one genuinely new kernel native autograd's first scope needed (the
// runtime has no compare/where to compose it from): the generic binary
// odometer walking the forward input x and the upstream gradient in
// lockstep, each through its own strides/offset.
TF_EXPORT void tf_core_relu_backward(
    const void* x, const void* upstream, void* dst,
    const int64_t* shape, const int64_t* x_strides, const int64_t* u_strides,
    int64_t x_offset, int64_t u_offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_relu_backward",
                                    {x, upstream, dst})) {
        return;
    }
    binary_by_dtype<tf::ReluBackwardOp>(x, upstream, dst, shape, x_strides,
                                        u_strides, x_offset, u_offset, ndim);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_add_contiguous(
    const void* a, const void* b, void* dst,
    int64_t numel, int64_t a_offset, int64_t b_offset
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_add_contiguous", {a, b, dst})) {
        return;
    }
    binary_contiguous_by_dtype<tf::AddOp>(a, b, dst, numel, a_offset, b_offset);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_subtract_contiguous(
    const void* a, const void* b, void* dst,
    int64_t numel, int64_t a_offset, int64_t b_offset
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_subtract_contiguous",
                                    {a, b, dst})) {
        return;
    }
    binary_contiguous_by_dtype<tf::SubtractOp>(a, b, dst, numel, a_offset,
                                               b_offset);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_multiply_contiguous(
    const void* a, const void* b, void* dst,
    int64_t numel, int64_t a_offset, int64_t b_offset
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_multiply_contiguous",
                                    {a, b, dst})) {
        return;
    }
    binary_contiguous_by_dtype<tf::MultiplyOp>(a, b, dst, numel, a_offset,
                                               b_offset);
    TF_GUARD_END_VOID()
}

// -- legacy raw-buffer kernels (the v0.x benchmark reference set) ------------
//
// These operate over plain float64 arrays Python passes directly (no
// Storage handle) and cannot allocate, so they need no guard.

TF_EXPORT void tf_elementwise_add(
    const double* a, const double* b, double* out, int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

TF_EXPORT void tf_elementwise_subtract(
    const double* a, const double* b, double* out, int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}

TF_EXPORT void tf_elementwise_multiply(
    const double* a, const double* b, double* out, int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

// IEEE float64 division: x/0 gives +-inf and 0/0 gives NaN, the same
// values NumPy produces (NumPy additionally warns; this kernel does not).
TF_EXPORT void tf_elementwise_divide(
    const double* a, const double* b, double* out, int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
}

TF_EXPORT void tf_relu(const double* a, double* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] > 0.0 ? a[i] : 0.0;
    }
}
