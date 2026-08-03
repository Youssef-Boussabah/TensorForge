// Dependency-free C++ test for the dtype-general Dropout forward kernel
// (Phase I, milestone I7). No GoogleTest / Catch2 — a plain executable that
// prints failures and returns a nonzero exit code if any check fails, so
// CTest reports pass/fail.
//
// This binary compiles cpp/src/random.cpp, cpp/src/storage.cpp, and
// cpp/src/error.cpp directly, so it reaches the hidden templated
// ``tf::dropout_forward_contiguous`` at **both** instantiations, alongside
// the exported wrapper it lives inside — at the layer where the properties
// are actually decided, with no Python wrapper, no ctypes boundary, and no
// NumPy anywhere.
//
// It is deliberately complementary to ``test_dropout_forward`` (G2) rather
// than a superset of it. That target owns the **algorithm**: the committed
// mix64, stream-key, and element-bit known-answer vectors, the
// bits-to-uniform conversion, the strict `<` equality-threshold vector, and
// the full float64 validation matrix. Every one of those is untouched here
// and untouched by this milestone — I7 changed no constant, no shift, no
// multiplication order, no key derivation, and no comparison.
//
// This one proves the set that opens at I7:
//
//   1. **The drop pattern is dtype-independent for one random key.** The
//      *same committed keep patterns* G2 locked at float64 are asserted at
//      float32, from the same seeds and call indices — not a second vector
//      set, which would have made the two dtypes' streams separately
//      assertable and therefore separately breakable. That is the whole
//      point of design §14.2's decision to keep ``dropout_uniform``
//      binary64: only the *values written* are dtype-dependent, never which
//      elements are written. The two runs are also compared directly,
//      element by element, so the property is asserted rather than inferred
//      from two independent agreements with a table.
//
//   2. **The kept multiplier is the binary64 reciprocal narrowed once.**
//      Not recomputed at the element type, not accumulated, not
//      re-derived: exactly ``static_cast<T>(1.0 / (1.0 - p))``, asserted by
//      raw bit pattern. At ``T = float`` this is *observable*, and the
//      witness is proved non-vacuous first: at p = 0.025 the narrow-once
//      value and the all-binary32 value ``1.0f / (1.0f - 0.025f)`` are
//      checked to genuinely differ, and the kernel is then asserted to
//      equal the first and to differ from the second. A structural claim
//      alone would not have distinguished them.
//
//   3. **float64 is byte-for-byte what Phase G produced.** Every float64
//      assertion here is the pre-I7 value, so the templating cannot have
//      moved the shipped dtype by a bit.
//
//   4. **Exceptional values follow plain IEEE arithmetic**, identically in
//      shape at both widths: ``-0.0`` (kept and dropped), ±inf, NaN,
//      subnormals, and the ``0 * inf`` NaN a dropped infinity produces —
//      every one against a reference computed here at the same width, by
//      raw bit pattern, because ``==`` cannot see ``-0.0`` and calls every
//      NaN unequal to itself.
//
//   5. **Mixed dtype is rejected before anything is written**, in each of
//      the three participating handle positions independently, with
//      TF_ERROR_INVALID and byte-for-byte unchanged destinations — and the
//      error slot clears on the next successful call, so a rejection is not
//      sticky.
//
//   6. **The validation matrix is unchanged at both widths.** The
//      probability rules (finite, 0 <= p < 1, p == 1 rejected), the null
//      handles, the negative count and offset, the span checks, and the
//      aliasing rules all behave exactly as G2 defined them — I7 added one
//      dtype guard *above* them and reordered nothing.
//
//   7. **No generator state exists in C++.** Two identical calls produce
//      byte-identical results and two different call indices do not, at
//      both widths, which is the observable half of "the kernel is a pure
//      function of its arguments".
//
// The exported symbol is the same one, with the same argument count, order,
// and types. I7 adds no export.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "tf_internal.h"  // TF_EXPORT, tf::Storage, TfDtype, error accessors
#include "tf_random_internal.h"

TF_EXPORT void* tf_storage_create_typed(std::int64_t size,
                                        std::int32_t dtype_code);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT std::int64_t tf_storage_size(const void* handle);
TF_EXPORT void tf_storage_copy_from(void* handle, const void* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, void* dst);

TF_EXPORT void tf_core_dropout_forward(
    const void* input_handle, std::int64_t input_offset,
    void* output_handle,
    void* mask_handle,
    std::int64_t count,
    std::uint64_t seed,
    std::uint64_t call_index,
    double p);

TF_EXPORT int tf_last_error_code();
TF_EXPORT const char* tf_last_error_message();
TF_EXPORT void tf_clear_error();

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    }
}

// ---------------------------------------------------------------------------
// Bit-pattern plumbing and per-dtype traits, in the shape the earlier dtype
// targets established so the files read alike.
// ---------------------------------------------------------------------------

template <class T> struct BitsOf;
template <> struct BitsOf<double> { using type = std::uint64_t; };
template <> struct BitsOf<float> { using type = std::uint32_t; };

template <class T>
typename BitsOf<T>::type bits(T value) {
    typename BitsOf<T>::type out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

template <class T>
T from_bits(typename BitsOf<T>::type pattern) {
    T out;
    std::memcpy(&out, &pattern, sizeof(out));
    return out;
}

template <class T> struct DtypeTraits;
template <> struct DtypeTraits<double> {
    static constexpr std::int32_t code = TF_DTYPE_FLOAT64;
    static const char* name() { return "float64"; }
    static double sentinel() { return from_bits<double>(0x4B2D000000000000ull); }
    static double subnormal() { return from_bits<double>(0x0000000000000001ull); }
};
template <> struct DtypeTraits<float> {
    static constexpr std::int32_t code = TF_DTYPE_FLOAT32;
    static const char* name() { return "float32"; }
    static float sentinel() { return from_bits<float>(0x4B2D0000u); }
    static float subnormal() { return from_bits<float>(0x00000001u); }
};

template <class T>
void* storage_of(const std::vector<T>& values) {
    void* handle = tf_storage_create_typed(
        static_cast<std::int64_t>(values.size()), DtypeTraits<T>::code);
    if (handle != nullptr) {
        tf_storage_copy_from(handle, values.data());
    }
    return handle;
}

// A destination pre-filled with a recognizable sentinel, so any element the
// kernel fails to write is left holding a locatable pattern (the H1 poison
// discipline, applied entirely by test infrastructure — there is no poison
// control anywhere in the library).
template <class T>
void* poisoned_storage(std::int64_t count) {
    const std::vector<T> poison(static_cast<std::size_t>(count),
                                DtypeTraits<T>::sentinel());
    return storage_of(poison);
}

template <class T>
std::vector<T> read_back(const void* handle) {
    std::vector<T> out(static_cast<std::size_t>(tf_storage_size(handle)));
    tf_storage_copy_to(handle, out.data());
    return out;
}

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b,
               const char* what) {
    if (a.size() != b.size()) {
        char message[256];
        std::snprintf(message, sizeof message, "%.180s [%.16s]: size mismatch",
                      what, DtypeTraits<T>::name());
        check(false, message);
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (bits<T>(a[i]) != bits<T>(b[i])) {
            char message[256];
            std::snprintf(message, sizeof message,
                          "%.150s [%.16s]: element %d differs",
                          what, DtypeTraits<T>::name(), static_cast<int>(i));
            check(false, message);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// The committed keep patterns, re-used verbatim from test_dropout_forward
// (docs/native_rng_dropout_design.md §4.7).
//
// These are the SAME vectors the float64 target asserts, and asserting them
// at float32 too is the point: the drop pattern is a property of the random
// key, not of the element type, so a second table would be a second thing to
// break rather than a second proof.
// ---------------------------------------------------------------------------

struct DropoutVector {
    const char* name;
    std::uint64_t seed;
    std::uint64_t call_index;
    double p;
    const char* keep;  // twelve characters; '1' kept, '0' dropped
};

const DropoutVector kDropoutVectors[] = {
    {"zero_seed_call0", 0x0000000000000000ULL, 0ULL, 0.25, "111110111110"},
    {"zero_seed_call1", 0x0000000000000000ULL, 1ULL, 0.25, "101011111011"},
    {"mixed_seed_call0", 0x0123456789ABCDEFULL, 0ULL, 0.25, "011011111010"},
    {"mixed_seed_call7", 0x0123456789ABCDEFULL, 7ULL, 0.75, "000010000000"},
    {"high_bit_seed_call3", 0x8000000000000000ULL, 3ULL, 0.75, "011000100000"},
    {"max_seed_call0", 0xFFFFFFFFFFFFFFFFULL, 0ULL, 0.25, "111110110110"},
    {"zero_seed_max_call", 0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFEULL, 0.75,
     "000001100001"},
};

const std::int64_t kVectorCount = 12;

// ---------------------------------------------------------------------------
// 1. The committed patterns hold at both widths, and the values written are
//    exactly input * multiplier at the element type.
// ---------------------------------------------------------------------------

template <class T>
void test_kernel_known_answers() {
    for (const DropoutVector& vector : kDropoutVectors) {
        std::vector<T> input(static_cast<std::size_t>(kVectorCount));
        for (std::int64_t i = 0; i < kVectorCount; ++i) {
            input[static_cast<std::size_t>(i)] = static_cast<T>(i) + T(0.5);
        }
        std::vector<T> output(static_cast<std::size_t>(kVectorCount),
                              DtypeTraits<T>::sentinel());
        std::vector<T> mask(static_cast<std::size_t>(kVectorCount),
                            DtypeTraits<T>::sentinel());
        tf::dropout_forward_contiguous(input.data(), output.data(),
                                       mask.data(), kVectorCount, vector.seed,
                                       vector.call_index, vector.p);

        const T scale = static_cast<T>(1.0 / (1.0 - vector.p));
        for (std::int64_t i = 0; i < kVectorCount; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            const T want = vector.keep[i] == '1' ? scale : T(0);
            if (bits<T>(mask[k]) != bits<T>(want)) {
                char message[256];
                std::snprintf(message, sizeof message,
                              "committed mask multiplier [%.16s] %.40s[%d]",
                              DtypeTraits<T>::name(), vector.name,
                              static_cast<int>(i));
                check(false, message);
            }
            if (bits<T>(output[k]) != bits<T>(input[k] * want)) {
                char message[256];
                std::snprintf(message, sizeof message,
                              "output is input * mask [%.16s] %.40s[%d]",
                              DtypeTraits<T>::name(), vector.name,
                              static_cast<int>(i));
                check(false, message);
            }
            // Exactly two distinct values in the mask, and nothing else.
            check(bits<T>(mask[k]) == bits<T>(T(0))
                      || bits<T>(mask[k]) == bits<T>(scale),
                  "mask holds exactly 0 or the inverted scale");
        }
    }
}

// ---------------------------------------------------------------------------
// 2. The two widths drop exactly the same elements for one key.
//
// Asserted directly rather than inferred from two table agreements: the
// keep/drop decision is recovered from each run's own mask (multiplier != 0)
// and the two boolean sequences are compared.
// ---------------------------------------------------------------------------

void test_the_drop_pattern_is_identical_across_dtypes() {
    const std::int64_t count = 257;      // deliberately not a power of two
    const double probabilities[] = {0.01, 0.25, 0.5, 0.75, 0.9375, 0.999};
    const std::uint64_t seeds[] = {0ULL, 0x0123456789ABCDEFULL,
                                   0xFFFFFFFFFFFFFFFFULL};
    const std::uint64_t calls[] = {0ULL, 1ULL, 7ULL, 0xFFFFFFFFFFFFFFFEULL};

    for (double p : probabilities) {
        for (std::uint64_t seed : seeds) {
            for (std::uint64_t call : calls) {
                std::vector<double> in64(static_cast<std::size_t>(count), 1.0);
                std::vector<float> in32(static_cast<std::size_t>(count), 1.0f);
                std::vector<double> out64(static_cast<std::size_t>(count));
                std::vector<double> mask64(static_cast<std::size_t>(count));
                std::vector<float> out32(static_cast<std::size_t>(count));
                std::vector<float> mask32(static_cast<std::size_t>(count));
                tf::dropout_forward_contiguous(in64.data(), out64.data(),
                                               mask64.data(), count, seed,
                                               call, p);
                tf::dropout_forward_contiguous(in32.data(), out32.data(),
                                               mask32.data(), count, seed,
                                               call, p);
                for (std::int64_t i = 0; i < count; ++i) {
                    const std::size_t k = static_cast<std::size_t>(i);
                    const bool kept64 = mask64[k] != 0.0;
                    const bool kept32 = mask32[k] != 0.0f;
                    if (kept64 != kept32) {
                        check(false,
                              "float32 and float64 dropped different elements "
                              "for one random key");
                        return;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. The kept multiplier is computed once in binary64 and narrowed once.
//
// At float32 this is observable, and the witness is proved non-vacuous
// before it is used: p = 0.025 is a probability at which the narrow-once
// value and the all-binary32 value genuinely differ, so the assertion below
// distinguishes the two policies rather than restating one of them.
// ---------------------------------------------------------------------------

const double kWitnessP = 0.025;

void test_the_float32_scale_is_narrowed_once_not_recomputed() {
    const float narrow_once = static_cast<float>(1.0 / (1.0 - kWitnessP));
    const float all_binary32 =
        1.0f / (1.0f - static_cast<float>(kWitnessP));
    // Non-vacuity first: if these agreed, everything below would pass under
    // either policy and would prove nothing.
    check(bits<float>(narrow_once) != bits<float>(all_binary32),
          "the p = 0.025 witness does not distinguish the two scale policies");

    const std::int64_t count = 64;
    std::vector<float> input(static_cast<std::size_t>(count), 1.0f);
    std::vector<float> output(static_cast<std::size_t>(count));
    std::vector<float> mask(static_cast<std::size_t>(count));
    tf::dropout_forward_contiguous(input.data(), output.data(), mask.data(),
                                   count, 4242ULL, 3ULL, kWitnessP);
    bool saw_kept = false;
    for (std::int64_t i = 0; i < count; ++i) {
        const float value = mask[static_cast<std::size_t>(i)];
        if (bits<float>(value) == bits<float>(0.0f)) {
            continue;
        }
        saw_kept = true;
        check(bits<float>(value) == bits<float>(narrow_once),
              "the float32 kept multiplier is not the narrowed binary64 scale");
        check(bits<float>(value) != bits<float>(all_binary32),
              "the float32 kept multiplier was recomputed in binary32");
    }
    // At p = 0.025 over 64 elements a run with no kept element is
    // astronomically unlikely, but the check must not pass vacuously.
    check(saw_kept, "the scale witness saw no kept element");
}

// The same rule, stated over a sweep at both widths: whatever the
// probability, the kept multiplier is exactly the narrowed binary64
// reciprocal — never a per-element recomputation.
template <class T>
void test_the_kept_multiplier_matches_the_specified_scale() {
    const double probabilities[] = {0.0, 1e-9, 0.021, 0.025, 0.1, 0.25, 0.5,
                                    0.75, 0.9, 0.999, 1.0 - 1e-9};
    for (double p : probabilities) {
        const std::int64_t count = 96;
        std::vector<T> input(static_cast<std::size_t>(count), T(1));
        std::vector<T> output(static_cast<std::size_t>(count));
        std::vector<T> mask(static_cast<std::size_t>(count));
        tf::dropout_forward_contiguous(input.data(), output.data(),
                                       mask.data(), count, 99ULL, 5ULL, p);
        const T scale = static_cast<T>(1.0 / (1.0 - p));
        for (std::int64_t i = 0; i < count; ++i) {
            const T value = mask[static_cast<std::size_t>(i)];
            check(bits<T>(value) == bits<T>(T(0))
                      || bits<T>(value) == bits<T>(scale),
                  "a mask value is neither zero nor the specified scale");
        }
        if (p == 0.0) {
            // Strict `<` means p == 0 drops nothing, at either width, and
            // the multiplier is exactly one.
            for (std::int64_t i = 0; i < count; ++i) {
                check(bits<T>(mask[static_cast<std::size_t>(i)])
                          == bits<T>(T(1)),
                      "p == 0 must keep every element with multiplier 1");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Exceptional values, against a reference computed here at the same width.
// ---------------------------------------------------------------------------

template <class T>
void test_exceptional_values() {
    const T inf = std::numeric_limits<T>::infinity();
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const std::vector<T> input = {
        T(0.0), -T(0.0), inf, -inf, nan, DtypeTraits<T>::subnormal(),
        -DtypeTraits<T>::subnormal(), std::numeric_limits<T>::max(),
        -std::numeric_limits<T>::max(), std::numeric_limits<T>::min(),
        T(1.5), T(-1.5),
    };
    const std::int64_t count = static_cast<std::int64_t>(input.size());
    for (double p : {0.25, 0.5, 0.75}) {
        std::vector<T> output(static_cast<std::size_t>(count));
        std::vector<T> mask(static_cast<std::size_t>(count));
        tf::dropout_forward_contiguous(input.data(), output.data(),
                                       mask.data(), count, 7ULL, 2ULL, p);
        // The reference is the documented formula, evaluated here in T:
        // plain IEEE multiplication, no special case. A dropped +inf
        // therefore becomes 0 * inf == NaN, and a kept -0.0 stays signed.
        std::vector<T> expected(static_cast<std::size_t>(count));
        for (std::int64_t i = 0; i < count; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            expected[k] = input[k] * mask[k];
        }
        same_bits(output, expected, "exceptional-value output is input * mask");
        // ...and the mask itself is never NaN, whatever the input was: it
        // is a function of the key alone and never of a value.
        for (std::int64_t i = 0; i < count; ++i) {
            const T value = mask[static_cast<std::size_t>(i)];
            check(value == value, "a mask value became NaN");
        }
    }
    // The signed zero and the dropped infinity, called out explicitly
    // rather than left to the loop: they are the two cases a "helpful"
    // special case would silently change.
    std::vector<T> pair = {-T(0.0), inf};
    std::vector<T> out(2);
    std::vector<T> msk(2);
    // p just below 1 with this key drops both; the values are then exactly
    // -0.0 * 0 == -0.0 and inf * 0 == NaN.
    tf::dropout_forward_contiguous(pair.data(), out.data(), msk.data(), 2,
                                   1ULL, 1ULL, 0.999999);
    if (bits<T>(msk[0]) == bits<T>(T(0))) {
        check(bits<T>(out[0]) == bits<T>(-T(0.0)),
              "a dropped -0.0 must stay -0.0");
    }
    if (bits<T>(msk[1]) == bits<T>(T(0))) {
        check(out[1] != out[1], "a dropped +inf must produce NaN");
    }
}

// ---------------------------------------------------------------------------
// 5-7. The exported wrapper: dispatch, mixed-dtype rejection, the unchanged
//      validation matrix, and statelessness.
// ---------------------------------------------------------------------------

template <class T>
void test_export_matches_the_kernel() {
    const std::int64_t count = kVectorCount;
    const std::int64_t offset = 3;      // a nonzero input offset
    std::vector<T> source(static_cast<std::size_t>(count + offset));
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<T>(i) - T(4.25);
    }
    void* input = storage_of(source);
    void* output = poisoned_storage<T>(count);
    void* mask = poisoned_storage<T>(count);
    check(input != nullptr && output != nullptr && mask != nullptr,
          "typed storage allocation failed");

    tf_clear_error();
    tf_core_dropout_forward(input, offset, output, mask, count, 5ULL, 9ULL,
                            0.375);
    check(tf_last_error_code() == TF_OK, "the export rejected a valid call");

    std::vector<T> expected_out(static_cast<std::size_t>(count));
    std::vector<T> expected_mask(static_cast<std::size_t>(count));
    tf::dropout_forward_contiguous(source.data() + offset, expected_out.data(),
                                   expected_mask.data(), count, 5ULL, 9ULL,
                                   0.375);
    same_bits(read_back<T>(output), expected_out,
              "export output matches the internal kernel");
    same_bits(read_back<T>(mask), expected_mask,
              "export mask matches the internal kernel");

    // count == 0 is legal and writes nothing: both destinations keep the
    // sentinel they were poisoned with.
    void* untouched_out = poisoned_storage<T>(count);
    void* untouched_mask = poisoned_storage<T>(count);
    const std::vector<T> before = read_back<T>(untouched_out);
    tf_clear_error();
    tf_core_dropout_forward(input, 0, untouched_out, untouched_mask, 0, 1ULL,
                            0ULL, 0.5);
    check(tf_last_error_code() == TF_OK, "count == 0 must be accepted");
    same_bits(read_back<T>(untouched_out), before,
              "count == 0 wrote to the output");
    same_bits(read_back<T>(untouched_mask), before,
              "count == 0 wrote to the mask");

    tf_storage_destroy(untouched_mask);
    tf_storage_destroy(untouched_out);
    tf_storage_destroy(mask);
    tf_storage_destroy(output);
    tf_storage_destroy(input);
}

// One rejection case: run the call, prove it recorded TF_ERROR_INVALID, and
// prove **both** destinations are byte-for-byte what they were.
template <class T>
void expect_rejected(void* input, std::int64_t offset, void* output,
                     void* mask, std::int64_t count, std::uint64_t seed,
                     std::uint64_t call_index, double p, const char* what) {
    const std::vector<T> out_before = read_back<T>(output);
    const std::vector<T> mask_before = read_back<T>(mask);
    tf_clear_error();
    tf_core_dropout_forward(input, offset, output, mask, count, seed,
                            call_index, p);
    if (tf_last_error_code() != TF_ERROR_INVALID) {
        char message[256];
        std::snprintf(message, sizeof message,
                      "%.180s [%.16s]: not rejected with TF_ERROR_INVALID",
                      what, DtypeTraits<T>::name());
        check(false, message);
    }
    same_bits(out_before, read_back<T>(output), what);
    same_bits(mask_before, read_back<T>(mask), what);
}

// The dtype guard, in each of the three participating handle positions
// independently. The "other" dtype's storage is the odd one out each time.
template <class T>
void test_mixed_dtype_is_rejected() {
    using Other = typename std::conditional<std::is_same<T, double>::value,
                                            float, double>::type;
    const std::int64_t count = 8;
    std::vector<T> values(static_cast<std::size_t>(count), T(2.5));
    void* input = storage_of(values);
    void* output = poisoned_storage<T>(count);
    void* mask = poisoned_storage<T>(count);
    void* other_input = poisoned_storage<Other>(count);
    void* other_output = poisoned_storage<Other>(count);
    void* other_mask = poisoned_storage<Other>(count);
    check(other_input != nullptr && other_output != nullptr
              && other_mask != nullptr,
          "typed storage allocation failed");

    // Position 1: the input disagrees.
    {
        const std::vector<Other> in_before = read_back<Other>(other_input);
        expect_rejected<T>(other_input, 0, output, mask, count, 1ULL, 0ULL,
                           0.5, "mixed dtype in the input position");
        same_bits(in_before, read_back<Other>(other_input),
                  "a rejected call wrote to the input");
    }
    // Position 2: the output disagrees.
    {
        const std::vector<Other> out_before = read_back<Other>(other_output);
        tf_clear_error();
        tf_core_dropout_forward(input, 0, other_output, mask, count, 1ULL,
                                0ULL, 0.5);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "mixed dtype in the output position was not rejected");
        same_bits(out_before, read_back<Other>(other_output),
                  "a rejected call wrote to the output");
    }
    // Position 3: the mask disagrees.
    {
        const std::vector<Other> mask_before = read_back<Other>(other_mask);
        const std::vector<T> out_before = read_back<T>(output);
        tf_clear_error();
        tf_core_dropout_forward(input, 0, output, other_mask, count, 1ULL,
                                0ULL, 0.5);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "mixed dtype in the mask position was not rejected");
        same_bits(mask_before, read_back<Other>(other_mask),
                  "a rejected call wrote to the mask");
        same_bits(out_before, read_back<T>(output),
                  "a mask-dtype rejection wrote to the output");
    }
    // The message names both dtypes, so a caller can see what disagreed.
    tf_clear_error();
    tf_core_dropout_forward(input, 0, other_output, mask, count, 1ULL, 0ULL,
                            0.5);
    {
        const char* message = tf_last_error_message();
        check(message != nullptr && std::strstr(message, "same dtype") != nullptr,
              "the dtype rejection message does not name the rule");
        check(message != nullptr && std::strstr(message, "float32") != nullptr
                  && std::strstr(message, "float64") != nullptr,
              "the dtype rejection message does not name both dtypes");
    }
    // ...and the slot clears on the next good call: a rejection is not
    // sticky, and a later success is not misread as a failure.
    tf_clear_error();
    tf_core_dropout_forward(input, 0, output, mask, count, 1ULL, 0ULL, 0.5);
    check(tf_last_error_code() == TF_OK,
          "a valid call after a rejection did not clear the error slot");

    tf_storage_destroy(other_mask);
    tf_storage_destroy(other_output);
    tf_storage_destroy(other_input);
    tf_storage_destroy(mask);
    tf_storage_destroy(output);
    tf_storage_destroy(input);
}

template <class T>
void test_the_validation_matrix_is_unchanged() {
    const std::int64_t count = 8;
    std::vector<T> values(static_cast<std::size_t>(count), T(3.25));
    void* input = storage_of(values);
    void* output = poisoned_storage<T>(count);
    void* mask = poisoned_storage<T>(count);
    void* small = poisoned_storage<T>(count - 1);

    // -- probability: finite, 0 <= p < 1. p == 1 is the division by zero.
    const double bad_probabilities[] = {
        -0.0000001, 1.0, 1.5, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (double p : bad_probabilities) {
        expect_rejected<T>(input, 0, output, mask, count, 1ULL, 0ULL, p,
                           "invalid probability");
    }
    // -- null handles, in each required position.
    expect_rejected<T>(nullptr, 0, output, mask, count, 1ULL, 0ULL, 0.5,
                       "null input handle");
    {
        const std::vector<T> mask_before = read_back<T>(mask);
        tf_clear_error();
        tf_core_dropout_forward(input, 0, nullptr, mask, count, 1ULL, 0ULL,
                                0.5);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "null output handle was not rejected");
        same_bits(mask_before, read_back<T>(mask),
                  "a null-output rejection wrote to the mask");
    }
    {
        const std::vector<T> out_before = read_back<T>(output);
        tf_clear_error();
        tf_core_dropout_forward(input, 0, output, nullptr, count, 1ULL, 0ULL,
                                0.5);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "null mask handle was not rejected");
        same_bits(out_before, read_back<T>(output),
                  "a null-mask rejection wrote to the output");
    }
    // -- counts, offsets, spans, and the int64 overflow guard.
    expect_rejected<T>(input, 0, output, mask, -1, 1ULL, 0ULL, 0.5,
                       "negative element count");
    expect_rejected<T>(input, -1, output, mask, count, 1ULL, 0ULL, 0.5,
                       "negative input offset");
    expect_rejected<T>(input, 1, output, mask, count, 1ULL, 0ULL, 0.5,
                       "input span exceeds its storage");
    expect_rejected<T>(input, INT64_MAX, output, mask, count, 1ULL, 0ULL, 0.5,
                       "offset + count overflows int64");
    {
        // A too-small destination, in each position.
        const std::vector<T> small_before = read_back<T>(small);
        tf_clear_error();
        tf_core_dropout_forward(input, 0, small, mask, count, 1ULL, 0ULL, 0.5);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "a too-small output was not rejected");
        tf_clear_error();
        tf_core_dropout_forward(input, 0, output, small, count, 1ULL, 0ULL,
                                0.5);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "a too-small mask was not rejected");
        same_bits(small_before, read_back<T>(small),
                  "a span rejection wrote to the small storage");
    }
    // -- aliasing: neither destination may be the input or the other.
    expect_rejected<T>(input, 0, input, mask, count, 1ULL, 0ULL, 0.5,
                       "output aliases the input");
    expect_rejected<T>(input, 0, output, input, count, 1ULL, 0ULL, 0.5,
                       "mask aliases the input");
    expect_rejected<T>(input, 0, output, output, count, 1ULL, 0ULL, 0.5,
                       "output aliases the mask");
    // -- the input itself was never modified by any rejection above.
    same_bits(values, read_back<T>(input),
              "a rejected call modified the input");

    tf_storage_destroy(small);
    tf_storage_destroy(mask);
    tf_storage_destroy(output);
    tf_storage_destroy(input);
}

// The seed and call-index edges, at both widths: the whole uint64 range is
// legal key material and nothing in the path treats an edge specially.
template <class T>
void test_seed_and_call_index_boundaries() {
    const std::int64_t count = 16;
    std::vector<T> values(static_cast<std::size_t>(count), T(1));
    void* input = storage_of(values);
    void* output = poisoned_storage<T>(count);
    void* mask = poisoned_storage<T>(count);
    const std::uint64_t edges[] = {0ULL, 1ULL, 0x8000000000000000ULL,
                                   0xFFFFFFFFFFFFFFFEULL,
                                   0xFFFFFFFFFFFFFFFFULL};
    for (std::uint64_t seed : edges) {
        for (std::uint64_t call : edges) {
            tf_clear_error();
            tf_core_dropout_forward(input, 0, output, mask, count, seed, call,
                                    0.5);
            check(tf_last_error_code() == TF_OK,
                  "a seed/call-index edge was rejected");
            std::vector<T> expected_out(static_cast<std::size_t>(count));
            std::vector<T> expected_mask(static_cast<std::size_t>(count));
            tf::dropout_forward_contiguous(values.data(), expected_out.data(),
                                           expected_mask.data(), count, seed,
                                           call, 0.5);
            same_bits(read_back<T>(mask), expected_mask,
                      "a seed/call-index edge disagreed with the kernel");
        }
    }
    tf_storage_destroy(mask);
    tf_storage_destroy(output);
    tf_storage_destroy(input);
}

// No state between calls, at either width: same arguments, same bits; a
// different call index, different bits.
template <class T>
void test_the_kernel_holds_no_state() {
    const std::int64_t count = 128;
    std::vector<T> values(static_cast<std::size_t>(count), T(1));
    std::vector<T> out_a(static_cast<std::size_t>(count));
    std::vector<T> mask_a(static_cast<std::size_t>(count));
    std::vector<T> out_b(static_cast<std::size_t>(count));
    std::vector<T> mask_b(static_cast<std::size_t>(count));
    tf::dropout_forward_contiguous(values.data(), out_a.data(), mask_a.data(),
                                   count, 31ULL, 4ULL, 0.5);
    tf::dropout_forward_contiguous(values.data(), out_b.data(), mask_b.data(),
                                   count, 31ULL, 4ULL, 0.5);
    same_bits(mask_a, mask_b, "two identical calls disagreed");
    same_bits(out_a, out_b, "two identical calls disagreed");

    tf::dropout_forward_contiguous(values.data(), out_b.data(), mask_b.data(),
                                   count, 31ULL, 5ULL, 0.5);
    bool differs = false;
    for (std::int64_t i = 0; i < count; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        if (bits<T>(mask_a[k]) != bits<T>(mask_b[k])) {
            differs = true;
            break;
        }
    }
    check(differs, "two different call indices produced the same mask");
}

}  // namespace

int main() {
    test_kernel_known_answers<double>();
    test_kernel_known_answers<float>();

    test_the_drop_pattern_is_identical_across_dtypes();

    test_the_float32_scale_is_narrowed_once_not_recomputed();
    test_the_kept_multiplier_matches_the_specified_scale<double>();
    test_the_kept_multiplier_matches_the_specified_scale<float>();

    test_exceptional_values<double>();
    test_exceptional_values<float>();

    test_export_matches_the_kernel<double>();
    test_export_matches_the_kernel<float>();

    test_mixed_dtype_is_rejected<double>();
    test_mixed_dtype_is_rejected<float>();

    test_the_validation_matrix_is_unchanged<double>();
    test_the_validation_matrix_is_unchanged<float>();

    test_seed_and_call_index_boundaries<double>();
    test_seed_and_call_index_boundaries<float>();

    test_the_kernel_holds_no_state<double>();
    test_the_kernel_holds_no_state<float>();

    if (g_failures == 0) {
        std::printf("test_dtype_dropout: all checks passed\n");
        return 0;
    }
    std::printf("test_dtype_dropout: %d check(s) failed\n", g_failures);
    return 1;
}
