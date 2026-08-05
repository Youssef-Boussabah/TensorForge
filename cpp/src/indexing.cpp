// Phase-K indexing kernels (milestone K3: the `argmax` forward).
//
// The source unit locked by docs/native_integer_tensors_design.md §32: the
// phase's index-producing and index-consuming operations live here,
// deliberately NOT in reduction.cpp. `argmax` is not a reduction in this
// runtime's sense — `tf_core_sum` accumulates *values* into a destination
// through per-input-axis write strides and has no notion of a position,
// while this searches for a position and writes an index of a different
// dtype — so folding it into that unit would have meant either two meanings
// for one traversal or a general reduction framework the phase does not
// build.
//
// Two layers, the Phase-D/E split (conv2d.cpp, classification.cpp):
//   1. tf::argmax_contiguous — the internal, hidden, noexcept compute
//      kernel, which assumes fully validated contiguous arguments and
//      allocates nothing. It is a **template over the source element type**,
//      so its definition lives in tf_indexing_internal.h (a template must be
//      visible where it is instantiated, and both instantiations must reach
//      the wrapper below *and* the CTest that compiles this file directly).
//   2. tf_core_argmax — the exported, exception-guarded C ABI wrapper that
//      revalidates every trust-boundary argument itself (handles, both dtype
//      roles, dimensions, overflow, the source span, the destination's exact
//      capacity, and aliasing) before a single destination element is
//      written.
//
// The ABI is **contiguous-only** by design: the Python NativeTensorCore
// layer applies the existing Policy-B copy-then-compute rule, so a strided
// view is materialized into a private contiguous copy before the call and no
// stride metadata ever crosses the boundary. The decomposition it takes —
// (outer, axis_length, inner) — is exactly the one
// tf_core_softmax_forward and tf_core_log_softmax_forward already take, and
// a full reduction is expressed as (1, numel, 1), so one symbol covers both
// the full and the per-axis case.
//
// **Dtype discipline, and the one place it differs from every other export**
// (design §22.8): the source and the destination deliberately carry
// *different* dtypes. The source must be floating and the destination must
// be exactly int64, so this export applies tf::require_floating to the
// source and tf::require_index to the destination, and applies **neither**
// tf::require_floating **nor** tf::require_matching_dtype to the
// destination — either would reject every valid call. The single dispatch is
// therefore on the **source** dtype alone; nothing below it branches on
// dtype again, not per run and not per element.
//
// Numerical contract: none, in the usual sense. The kernel performs no
// arithmetic on values — it compares them and returns a position — so there
// is nothing to reassociate, no accumulator width to state, and no
// exceptional-value normalization. The NaN, tie, and signed-zero rules are
// the normative algorithm in tf_indexing_internal.h, and its results are
// bit-exact in the only sense that applies to an integer: they are equal.

#include <cstdio>

#include "tf_indexing_internal.h"
#include "tf_internal.h"  // export macro, Storage/as_storage, TF_GUARD, set_error

namespace {

// -- the dtype dispatch arm (one per export) ---------------------------
//
// Recovers the typed source pointer through the single
// ``tf::storage_typed<T>`` accessor and calls the one templated kernel. It
// exists so the export's ``switch`` stays short branches and the pointer
// recovery is written once per operation rather than once per dtype. It is
// reached only after ``tf::require_floating`` has proved the source really
// carries ``T``'s dtype, which is what makes the recovery sound.
//
// The destination pointer is **not** a template parameter: an index is
// ``std::int64_t`` at every source width, so it is recovered once, outside
// the dtype question entirely.
template <class T>
void argmax_dispatch(
    const void* src_handle, int64_t src_offset, void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    tf::argmax_contiguous(
        tf::storage_typed<T>(src_handle) + src_offset,
        tf::storage_typed<std::int64_t>(dst_handle),
        outer, axis_length, inner);
}

// Checked int64 multiply/add for non-negative operands. File-local, like the
// equivalents in classification.cpp, conv2d.cpp, pooling.cpp, and
// elementwise.cpp — each compute unit keeps its own rather than growing a
// premature shared surface.
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

// A contiguous run of ``count`` elements starting at ``offset`` must fit
// inside a storage holding ``size`` elements. Spans, offsets, and capacities
// are counted in **logical elements** at every dtype, which is what lets one
// span check serve a float32 source, a float64 source, and an int64
// destination without ever mentioning a byte.
bool span_within(int64_t offset, int64_t count, int64_t size) {
    int64_t end;
    if (!checked_add(offset, count, end)) {
        return false;
    }
    return end <= size;
}

// Trust-boundary validation for the layout half of the call, in a fixed
// order. Returns nullptr when every argument is sound, or a short
// description of the FIRST failure. The caller has already proved both
// handles are non-null and that each carries its required dtype role, so
// what is left is arithmetic and capacity.
//
// The destination capacity is checked as an **exact equality** rather than a
// lower bound, which is where this differs from the fused forwards: those
// write into a destination whose shape the caller chose, while this one
// writes exactly one index per output position, so a destination larger than
// ``outer * inner`` is a malformed request rather than a roomy one.
const char* argument_error(
    const void* src_handle, int64_t src_offset, const void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    // -- dimensional factors: every extent is a real, positive count --
    if (outer < 1 || axis_length < 1 || inner < 1) {
        return "outer, axis_length, and inner must each be >= 1";
    }
    // -- element counts, overflow-checked so a bogus factor cannot wrap
    //    int64 into a small span that would pass the bounds tests below.
    //    Both products are proved: the source's, and the destination's --
    int64_t source_count;
    if (!checked_mul(outer, axis_length, source_count) ||
        !checked_mul(source_count, inner, source_count)) {
        return "shape product overflows int64";
    }
    int64_t index_count;
    if (!checked_mul(outer, inner, index_count)) {
        return "output index count overflows int64";
    }
    if (src_offset < 0) {
        return "negative source offset";
    }
    // -- spans must fit their allocations --
    if (!span_within(src_offset, source_count,
                     tf::as_storage(src_handle)->size)) {
        return "source span exceeds its storage";
    }
    if (tf::as_storage(dst_handle)->size != index_count) {
        return "destination storage does not hold exactly outer * inner "
               "indices";
    }
    // -- the destination is written while the source is read, so the two
    //    must not be the same storage. The dtype roles above already make a
    //    genuine alias unreachable — one storage carries one dtype, and this
    //    call needs a floating source and an int64 destination — so this is
    //    **defense in depth for tf_core_argmax**, not a check some other
    //    export leans on: the C ABI validates independently of what another
    //    check happens to imply, and no-write-on-rejection is a property
    //    this export owns rather than infers (design §22.10).
    //
    //    It is deliberately not described as groundwork for a later
    //    milestone. This validator is file-local to K3's export and its
    //    argument list; the committed contract gives ``tf_core_index_select``
    //    its **own** operand set and its **own** validation order (§22.9,
    //    which includes a complete index bounds scan this call has no
    //    operand for), so K4 will validate its ABI independently rather than
    //    inheriting anything from here --
    if (src_handle == dst_handle) {
        return "destination storage aliases the source";
    }
    return nullptr;
}

// Validate, and on failure record ``op: reason`` in the thread-local slot
// and report that the call must be rejected. Nothing has touched the
// destination by this point, so a rejected call leaves it byte-for-byte
// unchanged.
bool reject(const char* op, const char* reason) {
    char message[192];
    std::snprintf(message, sizeof(message), "%s: %s", op, reason);
    tf::set_error(TF_ERROR_INVALID, message);
    return true;
}

}  // namespace

// The position of a maximum along one axis (Phase K, K3). Contiguous-only
// for the tensor data, like the fused classification forwards: the Core
// layer applies Policy-B copy-then-compute to a strided view, so no stride
// metadata crosses the boundary. Only the source carries an offset — the
// destination is caller-allocated at offset 0 — and the three trailing
// int64s are the (outer, axis_length, inner) decomposition of the searched
// axis, with a full reduction expressed as (1, numel, 1).
//
// The destination is **int64 index storage**, not a value buffer of the
// source's dtype, and that asymmetry is the operation. It is caller
// allocated, holds exactly ``outer * inner`` elements, and every one of them
// is written.
//
// There is deliberately no second output handle and no maximum *value*
// returned: a kernel that finds the position of a maximum necessarily knows
// the maximum, and Phase K does not expose it (design §17.10).
TF_EXPORT void tf_core_argmax(
    const void* src_handle, int64_t src_offset,
    void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    TF_GUARD_BEGIN
    const char* op = "argmax";
    // 1. Null handles first, so the dtype guards below — which deliberately
    //    let a null handle pass — never pre-empt this export's own null
    //    report.
    if (src_handle == nullptr || dst_handle == nullptr) {
        reject(op, "null storage handle");
        return;
    }
    // 2. The source computes nothing but is compared, and comparison is a
    //    floating-role capability here: an int64 source is a *role* error,
    //    never a promotion opportunity.
    if (!tf::require_floating("tf_core_argmax", {src_handle})) {
        return;
    }
    // 3. The destination is an **index** buffer. Neither tf::require_floating
    //    nor tf::require_matching_dtype is applied to it — either would
    //    reject every valid call, which is precisely why this export has its
    //    own role check rather than reusing the value-operand pair
    //    (design §22.8, §22.10).
    if (!tf::require_index("tf_core_argmax", "destination", dst_handle)) {
        return;
    }
    // 4-9. Dimensions, checked products, offset sign, spans, the exact
    //      destination capacity, and aliasing.
    const char* reason = argument_error(src_handle, src_offset, dst_handle,
                                        outer, axis_length, inner);
    if (reason != nullptr) {
        reject(op, reason);
        return;
    }
    // 10. Validated: one dtype dispatch, on the **source** alone, into the
    //     internal noexcept kernel. Nothing above this point writes to the
    //     destination, so a rejected call leaves it byte-for-byte unchanged.
    //     A NaN or an infinity among the source values is a *value* and
    //     never an ABI failure — the algorithm has an exact answer for each,
    //     and the error slot the guard cleared on entry stays TF_OK.
    //
    //     The switch has no ``default:`` label, so a future dtype without an
    //     instantiation is a compile-time diagnostic rather than a silent
    //     misread. The Int64 arm is unreachable — step 2 rejected it — and
    //     is written out rather than defaulted for exactly that reason.
    switch (tf::dispatch_dtype({src_handle})) {
        case tf::Dtype::Float32:
            argmax_dispatch<float>(src_handle, src_offset, dst_handle,
                                   outer, axis_length, inner);
            break;
        case tf::Dtype::Float64:
            argmax_dispatch<double>(src_handle, src_offset, dst_handle,
                                    outer, axis_length, inner);
            break;
        case tf::Dtype::Int64:
            break;  // unreachable: rejected by the source role check above
    }
    TF_GUARD_END_VOID()
}
