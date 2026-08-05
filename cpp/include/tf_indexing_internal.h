// Internal declarations for the Phase-K indexing kernels. NOT part of the
// C ABI: everything here stays in ``namespace tf`` with hidden visibility,
// so Python never reaches it — the dependency-free CTest links the
// translation unit directly to exercise the compute in isolation from the
// guarded C ABI wrapper, the same split conv2d, pooling, and classification
// use (tf_conv2d_internal.h, tf_pooling_internal.h,
// tf_classification_internal.h).
//
// **No exported declaration belongs in this header.** ``tf_core_argmax`` is
// defined, with its TF_EXPORT marker, in cpp/src/indexing.cpp; a header
// declaration would be a second place the ABI is written down.
//
// ---------------------------------------------------------------------------
// Phase K, milestone K3: `argmax`, the phase's one index-*producing*
// operation (docs/native_integer_tensors_design.md §17).
// ---------------------------------------------------------------------------
//
// Two things live here, and they have **different** futures — which is
// worth stating, because "shared header" is not the same claim as "shared
// code":
//
//   * ``tf::argmax_contiguous`` — the traversal, as one template over the
//     source element type. ``T = float`` and ``T = double`` are the same
//     source; there is no dtype branch below the single dispatch in
//     indexing.cpp, and emphatically none per element. It is **K3's
//     specific index-producing traversal**: it searches a run for the
//     position of a maximum and writes one index per output position.
//     Nothing about it generalizes to selecting *by* index, and no later
//     milestone is expected to reuse it — K4's ``index_select`` gathers,
//     which is a different traversal over a different operand set with a
//     different destination dtype, and it will define and validate its own.
//   * ``tf::require_index`` — the **index-role** dtype guard. It is a
//     genuinely different question from ``tf::require_floating``, and
//     applying either in the other's place would reject every valid call:
//     ``argmax`` consumes a floating source and produces an ``int64``
//     destination *by design* (design §22.8). Unlike the traversal, this
//     one asks a question any index operand raises — "is this handle
//     exactly ``int64``?" — so it is the piece expected to stay useful to
//     K4, whose index operand needs the same answer about a different
//     handle in a different role.
//
// What the file and this header therefore are is the **common
// architectural home** for the phase's indexing operations, in the sense
// §32 assigns: one translation unit and one internal header where such
// work lives, rather than a promise that any particular routine is shared.
// A later milestone adding an operation here still owns its own traversal,
// its own ABI argument list, and its own validation.
//
// The output element type is **not** a template parameter and never becomes
// one. An index is an ``std::int64_t`` at every source width — that is the
// whole point of the operation — so widening the template would invent a
// degree of freedom the contract does not have.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "tf_internal.h"  // tf::Dtype, tf::Storage, dtype_name, set_error

namespace tf {

// The **index-role** guard: this handle must carry exactly ``int64``.
//
// The division of labour, completing the family tf_internal.h documents:
//
//   * ``require_float64``   — "this operation has not been generalized".
//   * ``require_floating``  — "this operation computes, and computing is a
//     floating-only capability".
//   * ``require_matching_dtype`` — "its operands must agree".
//   * ``require_index`` (here) — "this operand is an **index**, and an index
//     is exactly ``int64``". Used for ``argmax``'s destination, and never
//     for a value operand.
//
// It is applied **instead of**, never beside, the two floating guards on the
// handle it governs. ``require_floating`` on an ``argmax`` destination, or
// ``require_matching_dtype`` across the source/destination pair, would
// reject every valid call — the mistake §22.8 exists to name.
//
// Same contract as the guards it joins: total, ``noexcept``,
// allocation-free, and a function of the storage tag alone — never of a
// pointer value, an alignment, a clock, an environment variable, or a
// CPU-feature probe. A **null handle passes**, so each export keeps its own
// null validation, its own message, and its own ordering. On rejection it
// has already recorded TF_ERROR_INVALID naming the operation, the operand's
// role, and the offending dtype, and the caller must return **without
// touching any destination**.
//
// ``role`` names the handle's part in the call ("destination" here) because
// "this storage is float64" and "this *destination* is float64" are
// different reports, and the second is the one a caller can act on.
inline bool require_index(const char* operation, const char* role,
                          const void* handle) noexcept {
    if (handle == nullptr) {
        return true;
    }
    const Dtype dtype = as_storage(handle)->dtype;
    if (dtype != Dtype::Int64) {
        // Bounded stack buffer: recording a rejection must not itself
        // allocate, and snprintf truncates rather than overflowing.
        char message[192];
        std::snprintf(message, sizeof message,
                      "%s: the %s must be int64 index storage; got %s "
                      "(an index is exactly int64 at every source dtype)",
                      operation, role, dtype_name(dtype));
        set_error(TF_ERROR_INVALID, message);
        return false;
    }
    return true;
}

// The position of a maximum along one axis of a row-major CONTIGUOUS block
// of element type ``T``, written as the (outer, axis_length, inner)
// decomposition the fused classification forwards already use: element
// (o, k, i) lives at ``o * axis_length * inner + k * inner + i``, so one run
// is a fixed (o, i) pair walked over k with stride ``inner``, and the result
// for that run lands at ``dst[o * inner + i]``.
//
// A **full** reduction is expressed as outer = 1, axis_length = numel,
// inner = 1, so one kernel covers both the full and the per-axis case: no
// second symbol, no mode flag, and no branch a caller controls.
//
// ``src`` already points at the first element (the caller adds any storage
// offset); ``dst`` is caller-allocated ``int64`` storage holding exactly
// ``outer * inner`` elements, every one of which is written.
//
// **The value rule, which is normative** (design §17.5). Per run, left to
// right:
//
//     best_index = 0
//     best       = run[0]
//     for i in 1 .. len(run) - 1:
//         v = run[i]
//         if isnan(best):  continue      // nothing displaces an incumbent NaN
//         if isnan(v) or v > best:       // a NaN displaces a non-NaN; else strict >
//             best = v; best_index = i
//
// Its consequences are answers this loop produces rather than special
// cases: equal maxima keep the **lowest** index because ``>`` is strict;
// ``+0.0`` and ``-0.0`` tie because IEEE comparison does not order them; an
// all-``-inf`` run returns 0 because everything ties; the **first** NaN wins
// and no later NaN displaces it, because ``v > NaN`` is false and the second
// clause requires the incumbent not to be NaN; and a length-1 run returns 0.
//
// **Initialization is load-bearing.** ``best`` starts at ``run[0]``, never
// at a sentinel such as the type's lowest representable value: a sentinel
// start makes an all-``-inf`` run and an all-NaN run return 0 *by accident*,
// and this start makes them return 0 *by construction*.
//
// It never inspects a NaN's payload, its signalling bit, or its sign — the
// question is asked with ``std::isnan`` at the source element type rather
// than with a comparison whose result could vary — so the integers this
// produces on Windows and on Linux are equal (design §29.5).
//
// noexcept, allocation-free, and it performs **no arithmetic on values**:
// there is nothing to reassociate, no accumulator, and no accumulation-order
// contract to state. It reads each source element exactly once per run and
// writes only inside ``[0, outer * inner)``.
template <class T>
inline void argmax_contiguous(
    const T* src, std::int64_t* dst,
    std::int64_t outer, std::int64_t axis_length, std::int64_t inner) noexcept {
    for (std::int64_t o = 0; o < outer; ++o) {
        const std::int64_t plane = o * axis_length * inner;
        for (std::int64_t i = 0; i < inner; ++i) {
            const std::int64_t base = plane + i;
            std::int64_t best_index = 0;
            T best = src[base];
            for (std::int64_t k = 1; k < axis_length; ++k) {
                const T value = src[base + k * inner];
                if (std::isnan(best)) {
                    continue;  // nothing displaces an incumbent NaN
                }
                if (std::isnan(value) || value > best) {
                    best = value;
                    best_index = k;
                }
            }
            dst[o * inner + i] = best_index;
        }
    }
}

}  // namespace tf
