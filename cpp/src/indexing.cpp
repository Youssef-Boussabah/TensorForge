// Phase-K indexing kernels (milestone K3: the `argmax` forward; milestone
// K4: the `index_select` forward).
//
// The source unit locked by docs/native_integer_tensors_design.md §32: the
// phase's index-producing and index-consuming operations live here,
// deliberately NOT in reduction.cpp. `argmax` is not a reduction in this
// runtime's sense — `tf_core_sum` accumulates *values* into a destination
// through per-input-axis write strides and has no notion of a position,
// while this searches for a position and writes an index of a different
// dtype — so folding it into that unit would have meant either two meanings
// for one traversal or a general reduction framework the phase does not
// build. `index_select` is not a reduction at all: it copies whole slices.
//
// Two layers per operation, the Phase-D/E split (conv2d.cpp,
// classification.cpp):
//   1. tf::argmax_contiguous / tf::index_select_contiguous — the internal,
//      hidden, noexcept compute kernels, which assume fully validated
//      contiguous arguments and allocate nothing. Each is a **template over
//      the value element type**, so both definitions live in
//      tf_indexing_internal.h (a template must be visible where it is
//      instantiated, and every instantiation must reach the wrappers below
//      *and* the CTests that compile this file directly).
//   2. tf_core_argmax / tf_core_index_select — the exported,
//      exception-guarded C ABI wrappers that revalidate every
//      trust-boundary argument themselves (handles, every dtype role,
//      dimensions, overflow, spans, the destination's exact capacity,
//      aliasing, and — for index_select — the complete index bounds scan)
//      before a single destination element is written.
//
// **Two exports, two validation lists, and deliberately no blanket
// validator** (design §22.10). They share the file-local arithmetic
// primitives below — checked multiplication, checked addition, span
// containment, and the error-recording helper — because those are
// operation-independent facts about int64 and about storage. They do NOT
// share an argument validator: `argmax` has no index handle to scan and an
// exactly-`outer * inner` destination, while `index_select` has three
// handles, two offsets, an exactly-`outer * index_count * inner`
// destination, and a bounds scan; one function covering both would need a
// mode flag, and a mode flag is how two contracts quietly become one.
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
// **Dtype discipline, and the two exports differ in it precisely**
// (design §22.8, §22.9, §22.10):
//
//   * tf_core_argmax — the source and the destination deliberately carry
//     *different* dtypes. The source must be floating and the destination
//     must be exactly int64, so it applies tf::require_floating to the
//     source and tf::require_index to the destination, and applies
//     **neither** tf::require_floating **nor** tf::require_matching_dtype to
//     the destination — either would reject every valid call.
//   * tf_core_index_select — the source and the destination are both
//     floating and must **agree**, so it applies tf::require_floating to
//     each and then tf::require_matching_dtype across the pair; the separate
//     index handle takes tf::require_index. require_matching_dtype is used
//     here and only here, and never across a floating/index role boundary:
//     an index operand is a *role*, not an arithmetic operand, so a role
//     mismatch is a role error rather than a promotion opportunity (§12.4).
//
// Each export therefore dispatches once, on the **source** dtype alone;
// nothing below that branches on dtype again, not per run, not per slice,
// and not per element.
//
// Numerical contract: none, in the usual sense, for either kernel. `argmax`
// performs no arithmetic on values — it compares them and returns a
// position — so there is nothing to reassociate, no accumulator width to
// state, and no exceptional-value normalization; the NaN, tie, and
// signed-zero rules are the normative algorithm in tf_indexing_internal.h.
// `index_select` performs no arithmetic and does not even *read* a value: it
// copies object representations, so every bit of every element it moves —
// both signed zeros, both infinities, subnormals, every NaN payload and
// signalling bit — arrives unchanged. Both results are bit-exact in the only
// senses that apply: an equal integer, and an identical bit pattern.

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
    //    operand for), so K4 validates its ABI independently rather than
    //    inheriting anything from here -- which is what shipped:
    //    ``index_select_argument_error`` below is its own function, with its
    //    own extents, its own spans, and its own two aliasing checks --
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

// -- index_select (Phase K, milestone K4) ------------------------------

// The dtype dispatch arm for the gather. One template parameter covers the
// **source and the destination together**, which is sound precisely because
// tf::require_matching_dtype has already proved they agree; the index
// pointer is recovered outside the dtype question entirely, because an index
// is std::int64_t at every value width.
template <class T>
void index_select_dispatch(
    const void* src_handle, int64_t src_offset,
    const void* idx_handle, int64_t idx_offset, void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t index_count, int64_t inner) {
    tf::index_select_contiguous(
        tf::storage_typed<T>(src_handle) + src_offset,
        tf::storage_typed<std::int64_t>(idx_handle) + idx_offset,
        tf::storage_typed<T>(dst_handle),
        outer, axis_length, index_count, inner);
}

// Trust-boundary validation for the layout half of tf_core_index_select, in
// the fixed order design §22.9 gives it. Returns nullptr when every argument
// is sound, or a short description of the FIRST failure. The caller has
// already proved all three handles are non-null and that each carries the
// dtype its role requires, so what is left is arithmetic, capacity, and
// aliasing.
//
// **Its own function, not argmax's with an extra argument** (§22.10): this
// call has three handles rather than two, two offsets rather than one, four
// extents rather than three, a destination sized ``outer * index_count *
// inner`` rather than ``outer * inner``, and two aliasing pairs rather than
// one. The shared pieces below it — checked_mul, checked_add, span_within —
// are facts about int64 and about storage, not about either operation.
const char* index_select_argument_error(
    const void* src_handle, int64_t src_offset,
    const void* idx_handle, int64_t idx_offset, const void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t index_count, int64_t inner) {
    // -- dimensional factors: every extent is a real, positive count.
    //    ``index_count`` joins the three argmax takes: an empty selection is
    //    not a special case here, it is a malformed request, because the
    //    runtime cannot represent a zero-element result to return --
    if (outer < 1 || axis_length < 1 || index_count < 1 || inner < 1) {
        return "outer, axis_length, index_count, and inner must each be >= 1";
    }
    // -- element counts, overflow-checked so a bogus factor cannot wrap
    //    int64 into a small span that would pass the bounds tests below.
    //    Both are proved: the source's and the destination's, which differ
    //    in this operation (index_count replaces axis_length) --
    int64_t source_count;
    if (!checked_mul(outer, axis_length, source_count) ||
        !checked_mul(source_count, inner, source_count)) {
        return "source shape product overflows int64";
    }
    int64_t destination_count;
    if (!checked_mul(outer, index_count, destination_count) ||
        !checked_mul(destination_count, inner, destination_count)) {
        return "destination shape product overflows int64";
    }
    if (src_offset < 0) {
        return "negative source offset";
    }
    if (idx_offset < 0) {
        return "negative index offset";
    }
    // -- spans must fit their allocations. Three storages, three questions:
    //    the source block, exactly ``index_count`` index values from their
    //    own offset, and a destination whose capacity is an **exact**
    //    equality for argmax's reason -- this call writes every destination
    //    element, so a larger destination is a malformed request rather than
    //    a roomy one --
    if (!span_within(src_offset, source_count,
                     tf::as_storage(src_handle)->size)) {
        return "source span exceeds its storage";
    }
    if (!span_within(idx_offset, index_count,
                     tf::as_storage(idx_handle)->size)) {
        return "index span exceeds its storage";
    }
    if (tf::as_storage(dst_handle)->size != destination_count) {
        return "destination storage does not hold exactly "
               "outer * index_count * inner values";
    }
    // -- the byte count one slice copy moves. Counted through the single
    //    checked element-to-byte conversion, so a platform with a size_t
    //    narrower than int64 rejects here instead of truncating the length
    //    handed to memcpy. ``inner`` is positive and no larger than the
    //    destination capacity just proved, so this cannot fail on the
    //    64-bit platforms TensorForge supports -- it is the trust boundary
    //    stating its assumption rather than inheriting it --
    std::size_t slice_bytes = 0;
    if (!tf::dtype_checked_bytes(inner, tf::as_storage(src_handle)->dtype,
                                 slice_bytes)) {
        return "the per-slice byte count is not representable";
    }
    // -- the destination is written while both operands are read, so it must
    //    be neither of them. Two checks, because there are two operands: the
    //    first is the same defense in depth argmax keeps (the dtype roles
    //    already make it unreachable), the second is genuinely reachable in
    //    principle only for a caller that has manufactured an int64
    //    destination, which the floating destination role has already
    //    refused. Both are retained because the C ABI validates
    //    independently of what another check happens to imply.
    //
    //    Source/index aliasing is deliberately NOT checked: both are
    //    read-only, one storage carries one dtype, and their roles require
    //    different dtypes, so an alias is structurally impossible and there
    //    is nothing a check could protect --
    if (dst_handle == src_handle) {
        return "destination storage aliases the source";
    }
    if (dst_handle == idx_handle) {
        return "destination storage aliases the index operand";
    }
    return nullptr;
}

// Every index the kernel will use must name a real position along the
// selected axis. This runs over the WHOLE span before the first destination
// element is written (design §14.4, §22.9 step 9), which is the one thing
// that must not be folded into the copy loop: checking each index as it is
// used leaves a partly written destination behind when it rejects.
//
// Negative values are rejected rather than wrapped (§14.2). An int64 tensor
// may legitimately *hold* negative values; using one as an index position is
// a different question, and the answer is no.
//
// The report names the offending value, the position it was found at, and
// the valid interval, because "an index was out of range" is not something a
// caller can act on. This is a **second** authority, not a restatement of
// Python's identical scan: neither may be removed because the other exists,
// the rule docs/native_classification_design.md §9.3 already states for the
// cross-entropy target indices.
bool reject_index_range(const char* op, const std::int64_t* indices,
                        int64_t index_count, int64_t axis_length) {
    for (int64_t position = 0; position < index_count; ++position) {
        const std::int64_t value = indices[position];
        if (value < 0 || value >= axis_length) {
            char message[192];
            std::snprintf(
                message, sizeof(message),
                "%s: index %lld at position %lld is outside the selectable "
                "range [0, %lld)", op, static_cast<long long>(value),
                static_cast<long long>(position),
                static_cast<long long>(axis_length));
            tf::set_error(TF_ERROR_INVALID, message);
            return true;
        }
    }
    return false;
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

// The slices an int64 index tensor names, along one axis (Phase K, K4).
// Contiguous-only for all three storages, like every fused export since
// Phase D: the Core layer applies Policy-B copy-then-compute to a strided
// source or a strided index view, so no stride metadata crosses the
// boundary. The source and the index each carry an offset — either may be a
// narrowed view — while the caller-allocated destination is always at offset
// 0, and the four trailing int64s are the (outer, axis_length, index_count,
// inner) decomposition of the selected axis.
//
// The destination is **floating storage of exactly the source's dtype**, so
// this export's role table is the mirror image of tf_core_argmax's: both
// floating guards and require_matching_dtype apply here, to the
// source/destination pair, and the separate int64 index handle takes
// require_index instead of either (design §22.9, §22.10).
//
// It performs no arithmetic on values and never reads one: every element it
// moves crosses by object representation, so both signed zeros, both
// infinities, subnormals, and every NaN payload survive exactly. Duplicate
// indices are copied again rather than shared, and the caller's order is the
// destination's order — no sorting, no deduplication, no wrapping.
//
// There is deliberately no backward export. index_select's gradient is a
// scatter-add with its own accumulation-order and duplicate-index contract,
// and Phase K's ABI budget does not spend a third symbol on it (§18.9).
TF_EXPORT void tf_core_index_select(
    const void* src_handle, int64_t src_offset,
    const void* idx_handle, int64_t idx_offset,
    void* dst_handle,
    int64_t outer, int64_t axis_length,
    int64_t index_count, int64_t inner) {
    TF_GUARD_BEGIN
    const char* op = "index_select";
    // 1. Null handles first, so the dtype guards below — which deliberately
    //    let a null handle pass — never pre-empt this export's own null
    //    report. All three are checked, not just the two argmax has.
    if (src_handle == nullptr || idx_handle == nullptr ||
        dst_handle == nullptr) {
        reject(op, "null storage handle");
        return;
    }
    // 2. The source holds the values being selected, and holding values is a
    //    floating-role capability here: an int64 source is a *role* error,
    //    never a promotion opportunity, and never an "integer gather".
    if (!tf::require_floating("tf_core_index_select", {src_handle})) {
        return;
    }
    // 3. The destination holds the selected values, so it is floating too —
    //    which is exactly where this export diverges from tf_core_argmax,
    //    whose destination is an index buffer and would be rejected by this
    //    very check.
    if (!tf::require_floating("tf_core_index_select", {dst_handle})) {
        return;
    }
    // 4. ...and the pair must agree. Applied **after** both floating guards,
    //    so a mixed float/integer call is reported as "this operation is
    //    floating-only" rather than as two tags that disagree. This is the
    //    one export in the phase that asks this question, and it asks it
    //    only across the floating pair (design §22.9).
    if (!tf::require_matching_dtype("tf_core_index_select",
                                    src_handle, dst_handle)) {
        return;
    }
    // 5. The index operand is an **index**: exactly int64, at both value
    //    widths. require_matching_dtype is never applied across this
    //    boundary — it would reject every valid call (§12.4).
    if (!tf::require_index("tf_core_index_select", "index operand",
                           idx_handle)) {
        return;
    }
    // 6-13. Dimensions, checked products, offset signs, all three spans, the
    //       exact destination capacity, the slice byte count, and both
    //       aliasing pairs.
    const char* reason = index_select_argument_error(
        src_handle, src_offset, idx_handle, idx_offset, dst_handle,
        outer, axis_length, index_count, inner);
    if (reason != nullptr) {
        reject(op, reason);
        return;
    }
    // 14. The complete index bounds scan, over every one of the
    //     ``index_count`` values, **before** the first destination element is
    //     written. The typed recovery is sound because step 5 proved the tag,
    //     and the span it walks is the one step 6-13 proved lies inside the
    //     index storage.
    if (reject_index_range(
            op, tf::storage_typed<std::int64_t>(idx_handle) + idx_offset,
            index_count, axis_length)) {
        return;
    }
    // 15. Validated: one dtype dispatch, on the **source** alone — the
    //     destination is already proved to match it and the index is int64
    //     by construction — into the internal noexcept kernel. Nothing above
    //     this point writes to the destination, so a rejected call leaves it
    //     byte-for-byte unchanged.
    //
    //     The switch has no ``default:`` label, so a future dtype without an
    //     instantiation is a compile-time diagnostic rather than a silent
    //     misread. The Int64 arm is unreachable — step 2 rejected it — and
    //     is written out rather than defaulted for exactly that reason.
    switch (tf::dispatch_dtype({src_handle})) {
        case tf::Dtype::Float32:
            index_select_dispatch<float>(
                src_handle, src_offset, idx_handle, idx_offset, dst_handle,
                outer, axis_length, index_count, inner);
            break;
        case tf::Dtype::Float64:
            index_select_dispatch<double>(
                src_handle, src_offset, idx_handle, idx_offset, dst_handle,
                outer, axis_length, index_count, inner);
            break;
        case tf::Dtype::Int64:
            break;  // unreachable: rejected by the source role check above
    }
    TF_GUARD_END_VOID()
}
