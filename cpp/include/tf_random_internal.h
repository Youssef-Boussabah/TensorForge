// Internal (non-ABI) declarations for the Phase-G stateless random
// derivation and the Dropout forward compute kernel (milestone G2). See
// docs/native_rng_dropout_design.md §4 and §7.
//
// NOT part of the public C ABI: these symbols stay in ``namespace tf``
// with hidden visibility, exactly like the conv2d, pooling, and
// classification internals, so Python never reaches them. The
// dependency-free CTest (cpp/tests/test_dropout_forward.cpp) links
// cpp/src/random.cpp directly to exercise the derivation and the compute
// in isolation from the guarded C ABI wrapper.
//
// The whole file is **stateless by construction**. There is no generator
// object, no counter, no seed storage, no static or thread-local random
// state, and no lazy initialization anywhere in the translation unit:
// every value is a pure function of the arguments handed in. The
// generator that issues ``(seed, call_index)`` lives in Python
// (``tensorforge.experimental.NativeGenerator``) and is never visible
// here (design §7.6).
#pragma once

#include <cstdint>

namespace tf {

// ---------------------------------------------------------------------------
// The locked algorithm: "tensorforge.splitmix64", algorithm_version 1
// (docs/native_rng_dropout_design.md §4.1-§4.4).
//
// All arithmetic is on std::uint64_t with WRAPPING (modulo 2**64)
// semantics. C++ guarantees wraparound for unsigned types, so nothing in
// the bit path is implementation-defined or undefined; the explicit
// std::uint64_t spelling (never ``unsigned long`` or ``size_t``) is what
// keeps MSVC and Clang/GCC producing the identical bit sequence.
//
// A change to any constant, shift, multiplication order, the derivation,
// the bits-to-uniform conversion, or the comparison direction MUST
// introduce a new (algorithm, algorithm_version) pair (design §4.8) —
// the committed known-answer vectors in the CTest and in
// tests/test_native_dropout_core.py are the specification, not a
// regression convenience.
// ---------------------------------------------------------------------------

// The golden-ratio odd constant used to separate successive keys.
constexpr std::uint64_t kSplitMix64Golden = 0x9E3779B97F4A7C15ULL;

// The SplitMix64 finalizer: xor-shift, multiply, xor-shift, multiply,
// xor-shift. Pure, total, and reversible; noexcept and allocation-free.
//
//   x ^= x >> 30;  x *= 0xBF58476D1CE4E5B9
//   x ^= x >> 27;  x *= 0x94D049BB133111EB
//   x ^= x >> 31
std::uint64_t splitmix64_mix(std::uint64_t x) noexcept;

// The per-call stream key: mix64(seed + GOLDEN * (call_index + 1)).
//
// The ``+ 1`` keeps call index 0 from degenerating to mix64(seed), and
// the full finalizer (rather than a bare addition) is what decorrelates
// two call streams — the defect design §2.2 identifies in the reference
// implementation this phase was compared against.
std::uint64_t dropout_stream_key(
    std::uint64_t seed, std::uint64_t call_index) noexcept;

// One element's 64 random bits: mix64(stream + GOLDEN * (element + 1)).
//
// ``element`` is the **logical row-major element index** over the logical
// shape — never a physical storage position. The Core layer materializes
// a non-contiguous input into row-major contiguous storage before the
// kernel runs (Policy B), so the kernel's flat traversal index *is* the
// logical index and the draw cannot depend on strides, offsets, view
// history, or allocation order (design §7.3).
std::uint64_t dropout_element_bits(
    std::uint64_t stream_key, std::uint64_t element_index) noexcept;

// Bits to a uniform value on [0, 1): (bits >> 11) * 2**-53.
//
// The top 53 bits are at most 2**53 - 1, which float64 represents
// exactly, and 2**-53 is a power of two, so the product is exact and the
// result has 2**-53 granularity with no rounding surprise.
double dropout_uniform(std::uint64_t bits) noexcept;

// Inverted Dropout forward over ONE row-major CONTIGUOUS span of element
// type ``T``.
//
//   u          = dropout_uniform(dropout_element_bits(stream, i))
//   drop       = u < p                     (strict <, so p == 0 drops nothing)
//   mask[i]    = drop ? T(0) : static_cast<T>(1.0 / (1.0 - p))
//   output[i]  = input[i] * mask[i]
//
// ---------------------------------------------------------------------------
// Phase I, milestone I7: the compute kernel carries a scalar type parameter,
// and its definition moved here from cpp/src/random.cpp.
// ---------------------------------------------------------------------------
//
// ``T`` is **deduced from the pointer arguments**, so every pre-Phase-I call
// site — all of which pass ``double*`` — instantiates ``T = double`` and is
// the pre-I7 kernel statement for statement. ``T = float`` is the same
// source at binary32. The definition lives in this header rather than in the
// .cpp for exactly I4's, I5's, and I6's reason: a template must be *visible*
// where it is instantiated, and both instantiations have to reach the
// exported wrapper in random.cpp **and** the CTests that compile that file
// directly. Nothing about the Phase-G source organization changes.
//
// **The draw itself is dtype-independent, deliberately and permanently**
// (design §14.2). ``dropout_uniform`` still produces a binary64 value from
// the top 53 bits, and the keep/drop comparison is still that binary64 value
// against the binary64 ``p``. So for one ``(seed, call_index, element
// count)`` key a float32 Dropout and a float64 Dropout drop **exactly the
// same elements**; only the values written differ. A 24-bit float32 uniform
// would have made the drop pattern dtype-dependent for the same key — a
// structural divergence between two runs of the same model, and a second
// unrelated form for every committed known-answer vector — and is rejected.
//
// ``1.0 / (1.0 - p)`` is computed ONCE per call **in binary64** and narrowed
// ONCE to ``T``, so the mask holds exactly two distinct values and every
// kept element carries the identical multiplier (design §4.4, §7.4). At
// ``T = float`` that value is exactly ``float(1.0 / (1.0 - p))``.
//
// The keep/drop decision is a deterministic function of
// ``(seed, call_index, element_index, p)`` and NOTHING else — in
// particular it never depends on an input value, an address, the thread,
// the traversal partition, the element type, or any prior call.
//
// Layouts: ``input``, ``output``, and ``mask`` are each row-major
// contiguous ``T`` spans of exactly ``count`` elements; ``input``
// already points at its first element (the caller adds any storage
// offset). Both destinations are written in full, so the caller need not
// pre-initialize either. ``count == 0`` is legal and writes nothing.
//
// Preconditions (guaranteed by the exported wrapper and the Core layer;
// NOT re-validated here — this routine is the inner math, not a
// validation boundary):
//   * input / output / mask are non-null when ``count > 0`` and each span
//     ``count`` elements of type ``T``;
//   * all three have the SAME element type — the exported wrapper proves
//     that with ``tf::require_matching_dtype`` before it dispatches, and
//     there is no casting or promotion anywhere in the runtime;
//   * output and mask alias neither the input nor each other;
//   * ``count >= 0``;
//   * ``p`` is finite and in [0.0, 1.0), so ``1.0 - p`` is strictly
//     positive and the reciprocal cannot divide by zero.
//
// Allocates nothing and cannot throw (noexcept): pure arithmetic over
// caller-owned buffers. It reads the input without modifying it, writes
// only inside the output and mask spans, and holds no state between
// calls — two calls with the same arguments produce byte-identical
// results.
template <class T>
inline void dropout_forward_contiguous(
    const T* input,
    T* output,
    T* mask,
    std::int64_t count,
    std::uint64_t seed,
    std::uint64_t call_index,
    double p) noexcept {
    // One reciprocal for the whole call, in binary64, narrowed once — so
    // every kept element carries the identical multiplier and the mask
    // holds exactly two distinct values (design §4.4; the narrow-once rule
    // is §7.4's, the same one tf_storage_fill and tf_storage_scale follow).
    // The wrapper proved 0.0 <= p < 1.0, so the denominator is strictly
    // positive.
    const T scale = static_cast<T>(1.0 / (1.0 - p));
    const std::uint64_t stream_key = dropout_stream_key(seed, call_index);
    for (std::int64_t index = 0; index < count; ++index) {
        // The element index IS the logical row-major index: the Core
        // layer materializes a non-contiguous input before calling, so a
        // transposed view and its contiguous copy receive the same mask.
        const std::uint64_t bits = dropout_element_bits(
            stream_key, static_cast<std::uint64_t>(index));
        // Strict <, matching the design: at p == 0 nothing is dropped. The
        // comparison is binary64 on both sides at **both** element types,
        // which is what makes the drop pattern dtype-independent.
        const T multiplier =
            (dropout_uniform(bits) < p) ? T(0) : scale;
        mask[index] = multiplier;
        output[index] = input[index] * multiplier;
    }
}

}  // namespace tf
