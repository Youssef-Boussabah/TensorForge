// Stateless native random derivation and the Dropout forward compute
// kernel (Phase G, milestone G2): the internal "tensorforge.splitmix64"
// bit path, the inverted-Dropout CPU kernel, and their exported,
// exception-guarded C ABI wrapper.
//
// Phase I, milestone I7 made the compute dtype-general — one templated
// kernel in tf_random_internal.h, one ``tf::require_matching_dtype`` over
// the three participating handles, and one ``switch`` per exported call —
// **without touching the random derivation at all**. The algorithm
// identifier, the algorithm version, the finalizer, the key derivation, the
// element-index derivation, and the 53-bit bits-to-uniform conversion are
// exactly what G2 locked, so one ``(seed, call_index, element count)`` key
// drops the same elements at float32 and at float64 (design §14.2).
//
// The organizing rule of the phase (docs/native_rng_dropout_design.md §1)
// is that **random state is Python-managed and native kernels are
// stateless**. This translation unit is handed the complete random key
// for one operation — an explicit ``(seed, call_index)`` pair — and
// computes every element's draw from it. It never reads, holds, or
// advances a generator; there is no generator type, no counter, no seed
// storage, no static or thread-local random state, and no lazy
// initialization anywhere in the file. Nothing here consults <random>,
// std::random_device, mt19937, the clock, the process id, an address, or
// the allocation history.
//
// The forward writes the dropped/kept output **and** the parallel private
// multiplier mask in one pass, so the forward value and the saved
// multiplier always come from the same decision — the same shape the
// MaxPool2d forward uses for its winner buffer (§7.4). The mask is
// internal state: it never becomes a public tensor, never enters a
// state_dict or a checkpoint, and introduces no new dtype — it carries the
// input's, which is the only dtype in the call.
//
// There is deliberately **no backward kernel** (§7.5): the gradient of
// inverted Dropout is ``upstream * mask``, which the existing
// ``NativeTensorCore.multiply`` already computes.

#include <cmath>
#include <cstdio>

#include "tf_internal.h"  // export macro, Storage/as_storage, TF_GUARD, set_error
#include "tf_random_internal.h"

namespace tf {

std::uint64_t splitmix64_mix(std::uint64_t x) noexcept {
    // Unsigned overflow is intentional and defined: modulo 2**64. The
    // constants, the shift amounts, and the order of the two multiplies
    // are the locked specification (design §4.2) — changing any of them
    // changes the stream and requires a new algorithm_version.
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

std::uint64_t dropout_stream_key(
    std::uint64_t seed, std::uint64_t call_index) noexcept {
    // Two full finalizer applications separate the per-call stream from
    // the per-element draw (this one, then dropout_element_bits), so two
    // different call indices cannot produce overlapping element
    // sequences by a simple offset.
    return splitmix64_mix(seed + kSplitMix64Golden * (call_index + 1ULL));
}

std::uint64_t dropout_element_bits(
    std::uint64_t stream_key, std::uint64_t element_index) noexcept {
    return splitmix64_mix(
        stream_key + kSplitMix64Golden * (element_index + 1ULL));
}

double dropout_uniform(std::uint64_t bits) noexcept {
    // The top 53 bits are exactly representable in float64 and 2**-53 is
    // a power of two, so both the conversion and the scaling are exact.
    // 0x1p-53 is the hexadecimal float literal for 2**-53: written this
    // way rather than as a decimal so no parse can round it.
    //
    // Phase I, milestone I7: this stays **binary64 at every tensor dtype**
    // (design §14.2). It is not an oversight and it is not a place to save
    // a narrowing — deriving a 24-bit uniform for float32 would make the
    // keep/drop pattern depend on the element type for the same random key.
    return static_cast<double>(bits >> 11) * 0x1p-53;
}

// ``dropout_forward_contiguous`` is a template and lives in
// tf_random_internal.h (Phase I, milestone I7), for the reason recorded
// there: both instantiations have to be visible to the exported wrapper
// below and to the CTests that compile this file directly.

}  // namespace tf

// ---------------------------------------------------------------------------
// Exported C ABI wrapper (Phase G, milestone G2).
//
// ``tf_core_dropout_forward`` is the exception-guarded boundary between
// the Python/Core layer and the internal noexcept arithmetic above. It
// follows the Phase-E self-validating export precedent: even though the
// Core wrapper (backends/cpp.py) validates first, this boundary is
// reachable by any ctypes caller, so it re-proves every trust-boundary
// argument — non-null handles, a non-negative offset and count, spans
// that fit their allocations under overflow-checked int64 arithmetic, a
// finite ``p`` in [0.0, 1.0), and no aliasing between the input and
// either destination — and writes **nothing** to either destination when
// it rejects.
//
// **Contiguous storage only.** By Policy B (docs/native_cnn_design.md §5)
// the Core layer materializes a non-contiguous input into a private
// owning contiguous copy first, so no stride metadata crosses this
// boundary and the ABI interprets the (handle, offset, count) span as
// canonical contiguous data.
//
// ``seed`` and ``call_index`` are the complete random key. This function
// holds no state between calls and has no way to obtain any: it never
// touches a generator, and no C++ code in this project does.
// ---------------------------------------------------------------------------

namespace {

// -- the dtype dispatch arm (Phase I, milestone I7) --------------------
//
// One arm, mirroring the I5/I6 conv2d, pooling, and classification arms:
// recover the typed pointers through the single ``tf::storage_typed<T>``
// accessor and call the one templated kernel. It exists so the export's
// ``switch`` stays two short branches and the pointer recovery is written
// once for the operation rather than once per dtype. It is reached only
// after ``tf::require_matching_dtype`` has proved the three participating
// handles agree, which is what makes the recovery sound.
//
// ``seed``, ``call_index``, and ``p`` cross it unchanged: the random key is
// not storage, has no dtype to dispatch on, and nothing is ever inferred
// from it.
template <class T>
void dropout_forward_dispatch(
    const void* input_handle, std::int64_t input_offset,
    void* output_handle, void* mask_handle,
    std::int64_t count, std::uint64_t seed, std::uint64_t call_index,
    double p) {
    tf::dropout_forward_contiguous(
        tf::storage_typed<T>(input_handle) + input_offset,
        tf::storage_typed<T>(output_handle),
        tf::storage_typed<T>(mask_handle),
        count, seed, call_index, p);
}

// Checked int64 add for non-negative operands: return false on overflow
// instead of wrapping, so a bogus offset/count pair can never turn into a
// small (passing) span. File-local, matching the equivalent helpers in
// conv2d.cpp / pooling.cpp / classification.cpp — each compute unit keeps
// its own rather than sharing a premature header surface.
bool checked_add(std::int64_t a, std::int64_t b, std::int64_t& out) {
    if (a > INT64_MAX - b) {
        return false;
    }
    out = a + b;
    return true;
}

// A contiguous operand of ``count`` elements beginning at ``offset`` must
// fit inside a storage holding ``size`` elements. ``offset``/``count`` are
// already known non-negative here. Spans are measured in **logical
// elements** at every dtype (design §4.3), so this is unchanged by I7.
bool span_within(std::int64_t offset, std::int64_t count, std::int64_t size) {
    std::int64_t end;
    if (!checked_add(offset, count, end)) {
        return false;  // offset + count overflowed -> cannot fit
    }
    return end <= size;
}

// Record ``dropout_forward: reason`` in the thread-local slot and report
// rejection, matching the classification exports' style.
bool reject(const char* reason) {
    char message[192];
    std::snprintf(message, sizeof(message), "dropout_forward: %s", reason);
    tf::set_error(TF_ERROR_INVALID, message);
    return true;
}

}  // namespace

TF_EXPORT void tf_core_dropout_forward(
    const void* input_handle, std::int64_t input_offset,
    void* output_handle,
    void* mask_handle,
    std::int64_t count,
    std::uint64_t seed,
    std::uint64_t call_index,
    double p) {
    TF_GUARD_BEGIN
    // The dtype guard runs first (I7, matching the I3/I4/I5/I6 exports):
    // the input, the output, and the multiplier mask must all agree, and a
    // call that is both mixed-dtype and otherwise malformed reports the
    // dtype. There is no promotion and no narrowing, so a float32 input
    // with a float64 mask is an invalid request, not a conversion
    // opportunity — and reading a 4-byte-per-element buffer through a
    // ``double*`` would overrun it by a factor of two. Nothing below writes
    // to either destination before the dispatch, so a rejected call leaves
    // both byte-for-byte unchanged.
    // K1: the dtype-role guard runs first — an int64 operand is
    // a role error, never a promotion opportunity (§22.4).
    if (!tf::require_floating(
            "tf_core_dropout_forward",
            {input_handle, output_handle, mask_handle})) {
        return;
    }
    if (!tf::require_matching_dtype(
            "tf_core_dropout_forward",
            {input_handle, output_handle, mask_handle})) {
        return;
    }
    // -- required handles (none is nullable) --
    if (input_handle == nullptr || output_handle == nullptr ||
        mask_handle == nullptr) {
        reject("null required storage handle");
        return;
    }
    // -- element count and offset --
    if (count < 0) {
        reject("element count must be >= 0");
        return;
    }
    if (input_offset < 0) {
        reject("negative input offset");
        return;
    }
    // -- the probability: finite and in [0.0, 1.0). p == 1 would make the
    //    inverted multiplier a division by zero, so it is rejected here
    //    exactly as it is in Python (design §6.3), and NaN fails every
    //    comparison, so the explicit isfinite check is what catches it. --
    if (!std::isfinite(p)) {
        reject("probability must be finite (NaN and infinity rejected)");
        return;
    }
    if (!(p >= 0.0) || !(p < 1.0)) {
        reject("probability must satisfy 0.0 <= p < 1.0");
        return;
    }
    // -- storage spans must fit their allocations (overflow-checked) --
    if (!span_within(input_offset, count,
                     tf::as_storage(input_handle)->size)) {
        reject("input span exceeds its storage");
        return;
    }
    if (tf::as_storage(output_handle)->size < count) {
        reject("output storage smaller than the element count");
        return;
    }
    if (tf::as_storage(mask_handle)->size < count) {
        reject("mask storage smaller than the element count");
        return;
    }
    // -- neither destination may alias the input or the other: the mask
    //    and the output are both written while the input is read, so an
    //    aliasing call would silently compute wrong values rather than
    //    fail (the cross-entropy export's rule, for the same reason) --
    if (output_handle == input_handle || mask_handle == input_handle ||
        output_handle == mask_handle) {
        reject("destination storage aliases another operand");
        return;
    }
    // -- validated: one dtype dispatch into the internal noexcept kernel.
    //    Nothing above this point writes to either destination, so a
    //    rejected call leaves both byte-for-byte unchanged. --
    switch (tf::dispatch_dtype({input_handle, output_handle, mask_handle})) {
        case tf::Dtype::Float32:
            dropout_forward_dispatch<float>(
                input_handle, input_offset, output_handle, mask_handle,
                count, seed, call_index, p);
            break;
        case tf::Dtype::Float64:
            dropout_forward_dispatch<double>(
                input_handle, input_offset, output_handle, mask_handle,
                count, seed, call_index, p);
            break;
    }
    TF_GUARD_END_VOID()
}
