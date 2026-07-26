// Dependency-free C++ test for the stateless native random derivation and
// the Dropout forward kernel (Phase G, milestone G2). No GoogleTest /
// Catch2 — a plain executable that prints failures and returns a nonzero
// exit code if any check fails, so CTest reports pass/fail.
//
// This binary compiles cpp/src/random.cpp (and error.cpp) directly, so it
// exercises **both** layers of that translation unit, the way
// test_cross_entropy.cpp does:
//
//   * tf::splitmix64_mix / tf::dropout_stream_key /
//     tf::dropout_element_bits / tf::dropout_uniform /
//     tf::dropout_forward_contiguous — the internal, hidden derivation and
//     compute (valid, pre-validated arguments only: they are the math, not
//     a validation boundary); and
//   * tf_core_dropout_forward — the exported guarded C ABI wrapper, where
//     every trust-boundary argument must be rejected, driven here through
//     plain tf::Storage handles and the thread-local error slot.
//
// The reference values are **committed known-answer vectors**, not a
// second implementation of the same formula: the expected mix64 outputs,
// stream keys, element bits, and keep/drop patterns below are literal
// constants. That is the point — a Python reference that repeats the
// algorithm would agree with any consistent mistake, while these
// constants catch a changed multiplier, a changed shift, a reordered
// multiply, a changed key derivation, a changed bits-to-uniform
// conversion, a flipped comparison, or a dropped call/element index.
// The identical vectors are asserted from Python in
// tests/test_native_dropout_core.py, so both sides pin the same stream.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "tf_internal.h"  // TF_EXPORT, tf::Storage, error accessors, TfStatus
#include "tf_random_internal.h"

TF_EXPORT void tf_core_dropout_forward(
    const void* input_handle, std::int64_t input_offset,
    void* output_handle,
    void* mask_handle,
    std::int64_t count,
    std::uint64_t seed,
    std::uint64_t call_index,
    double p);

namespace {

int g_failures = 0;

const double kPosInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kPoison = 7777.5;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

void check_u64(std::uint64_t got, std::uint64_t want, const char* what) {
    if (got != want) {
        std::printf("FAIL: %s (got 0x%016llX, want 0x%016llX)\n", what,
                    static_cast<unsigned long long>(got),
                    static_cast<unsigned long long>(want));
        ++g_failures;
    }
}

void check_eq(double got, double want, const char* what) {
    if (got != want) {
        std::printf("FAIL: %s (got %.17g, want %.17g)\n", what, got, want);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// The committed known-answer vectors (docs/native_rng_dropout_design.md
// §4.7). Computed once from the locked specification and thereafter
// treated AS the specification.
// ---------------------------------------------------------------------------

struct MixVector {
    std::uint64_t input;
    std::uint64_t expected;
};

// mix64 over the edges of the 64-bit range: zero, the two smallest
// non-zero inputs (which pin the low-bit behavior a dropped xor-shift
// would change), the golden constant itself, the high-bit-only value, and
// all ones.
const MixVector kMixVectors[] = {
    {0x0000000000000000ULL, 0x0000000000000000ULL},
    {0x0000000000000001ULL, 0x5692161D100B05E5ULL},
    {0x0000000000000002ULL, 0xDBD238973A2B148AULL},
    {0x9E3779B97F4A7C15ULL, 0xE220A8397B1DCDAFULL},
    {0x8000000000000000ULL, 0x25C26EA579CEA98AULL},
    {0xFFFFFFFFFFFFFFFFULL, 0xB4D055FCF2CBBD7BULL},
};

struct StreamVector {
    std::uint64_t seed;
    std::uint64_t call_index;
    std::uint64_t expected;
};

// stream(seed, call_index) across seeds (zero, a mixed pattern, high-bit,
// all ones) and call indices (0, 1, 2, 7, and the highest index the
// generator can ever issue, 2**64 - 2 — see design §4.6).
const StreamVector kStreamVectors[] = {
    {0x0000000000000000ULL, 0ULL, 0xE220A8397B1DCDAFULL},
    {0x0000000000000000ULL, 1ULL, 0x6E789E6AA1B965F4ULL},
    {0x0000000000000000ULL, 2ULL, 0x06C45D188009454FULL},
    {0x0123456789ABCDEFULL, 0ULL, 0x157A3807A48FAA9DULL},
    {0x0123456789ABCDEFULL, 7ULL, 0x8931545F4F9EA651ULL},
    {0x8000000000000000ULL, 0ULL, 0x481EC0A212A9F3DBULL},
    {0xFFFFFFFFFFFFFFFFULL, 0ULL, 0xE4D971771B652C20ULL},
    {0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFEULL, 0x336503C6B835BEC0ULL},
    {0x0123456789ABCDEFULL, 0xFFFFFFFFFFFFFFFEULL, 0x20BEC7299668A13FULL},
};

// Seven full cases: the first four raw element-bit words, and the
// keep/drop pattern the kernel must produce over twelve logical elements
// at the stated probability. '1' means kept (multiplier = 1/(1-p)), '0'
// means dropped (multiplier = 0.0). Every pattern below is distinct, so a
// case cannot pass by accidentally matching a neighbour.
struct DropoutVector {
    const char* name;
    std::uint64_t seed;
    std::uint64_t call_index;
    double p;
    std::uint64_t bits[4];
    const char* keep;  // twelve characters
};

const DropoutVector kDropoutVectors[] = {
    {"zero_seed_call0", 0x0000000000000000ULL, 0ULL, 0.25,
     {0xA706DD2F4D197E6FULL, 0xB382A305F4414F5EULL,
      0x631A9154FBABF717ULL, 0xA80ABA8C86640906ULL},
     "111110111110"},
    {"zero_seed_call1", 0x0000000000000000ULL, 1ULL, 0.25,
     {0x46B73E79F0C37C00ULL, 0x374327C63D0CC8A6ULL,
      0xE10CF86AE3079278ULL, 0x26A223C360B54F32ULL},
     "101011111011"},
    {"mixed_seed_call0", 0x0123456789ABCDEFULL, 0ULL, 0.25,
     {0x021C88D0A3FD73B6ULL, 0x498D3E51E781CDE0ULL,
      0xA2A1796FEB7EF314ULL, 0x1A2D33D4F57B4CD4ULL},
     "011011111010"},
    {"mixed_seed_call7", 0x0123456789ABCDEFULL, 7ULL, 0.75,
     {0x0184F08818982A99ULL, 0x99E0A20D1E1F1641ULL,
      0x3E9AD5FC011194F1ULL, 0x52E464BC2FB3BF83ULL},
     "000010000000"},
    {"high_bit_seed_call3", 0x8000000000000000ULL, 3ULL, 0.75,
     {0x94E05B24F614999EULL, 0xD58EE1DBADEF970DULL,
      0xE932E5239EC1F7C9ULL, 0xB01B43DD212F69A7ULL},
     "011000100000"},
    {"max_seed_call0", 0xFFFFFFFFFFFFFFFFULL, 0ULL, 0.25,
     {0x5DC20AA7B2A27137ULL, 0xBDA5668A01D7049CULL,
      0x82B43276ABB80226ULL, 0xED4D5ED4A6EA59B4ULL},
     "111110110110"},
    // The highest call index a NativeGenerator can ever issue (§4.6).
    {"zero_seed_max_call", 0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFEULL, 0.75,
     {0x53531EEB39C4C095ULL, 0x1EACB2A4329B0259ULL,
      0x2402CC7044E8B298ULL, 0xAAB3D73BF633B046ULL},
     "000001100001"},
};

const std::int64_t kVectorCount = 12;

// ---------------------------------------------------------------------------
// The equality-threshold vector (docs/native_rng_dropout_design.md §4.7).
//
// Everything above pins the bit path; this pins the COMPARISON. The locked
// rule is `drop = u < p`, strictly — so an element whose uniform value is
// exactly `p` is KEPT, and the very next representable probability drops
// it. The p = 0.25 / p = 0.75 vectors cannot see that: no committed word
// converts to either value, so replacing `<` with `<=` would reproduce
// every one of those patterns unchanged.
//
// This vector is the one place the two rules disagree. The word below is
// already committed as kDropoutVectors["mixed_seed_call0"].bits[2] — the
// same seed, call index, and logical element index — so nothing new is
// introduced into the stream; only the probability is chosen to land
// exactly on it.
const std::uint64_t kEqualitySeed = 0x0123456789ABCDEFULL;
const std::uint64_t kEqualityCall = 0ULL;
const std::int64_t kEqualityIndex = 2;
const std::uint64_t kEqualityWord = 0xA2A1796FEB7EF314ULL;
// (0xA2A1796FEB7EF314 >> 11) * 2**-53, written as a hexadecimal float
// literal so no decimal parse can round it. Decimal value 0.635276403259464.
const double kEqualityUniform = 0x1.4542f2dfd6fdep-1;
const std::int64_t kEqualityCount = 4;
// Keep patterns over the first four elements of that stream:
//   * at p == u        the strict `<` rule KEEPS element 2 ....... "0010"
//   * at p == u        the rejected `<=` rule would drop it ...... "0000"
//   * at nextafter(u)  the strict `<` rule drops it too .......... "0000"
const char* const kEqualityKeepAtEqual = "0010";
const char* const kEqualityKeepAtNext = "0000";

// ---------------------------------------------------------------------------
// Internal-layer checks
// ---------------------------------------------------------------------------

void test_mix64_known_answers() {
    for (const MixVector& vector : kMixVectors) {
        check_u64(tf::splitmix64_mix(vector.input), vector.expected,
                  "mix64 known-answer vector");
    }
    // The finalizer is a pure function: the same input twice, and no
    // dependence on call order.
    check_u64(tf::splitmix64_mix(1ULL), tf::splitmix64_mix(1ULL),
              "mix64 is deterministic");
    // Two adjacent inputs must not produce adjacent outputs (a missing
    // multiply would leave them correlated).
    check(tf::splitmix64_mix(1ULL) != tf::splitmix64_mix(2ULL) + 1ULL,
          "mix64 decorrelates adjacent inputs");
}

void test_stream_known_answers() {
    for (const StreamVector& vector : kStreamVectors) {
        check_u64(tf::dropout_stream_key(vector.seed, vector.call_index),
                  vector.expected, "stream key known-answer vector");
    }
    // stream(seed, 0) is mix64(seed + GOLDEN), never mix64(seed): the
    // "+ 1" in the derivation is what keeps call 0 from degenerating.
    check_u64(tf::dropout_stream_key(0ULL, 0ULL),
              tf::splitmix64_mix(tf::kSplitMix64Golden),
              "stream(seed, 0) applies the +1 offset");
    check(tf::dropout_stream_key(0ULL, 0ULL) != tf::splitmix64_mix(0ULL),
          "stream(0, 0) is not mix64(0)");
    // Different call indices give different streams.
    check(tf::dropout_stream_key(5ULL, 0ULL) !=
              tf::dropout_stream_key(5ULL, 1ULL),
          "different call indices give different streams");
    // Different seeds give different streams.
    check(tf::dropout_stream_key(5ULL, 0ULL) !=
              tf::dropout_stream_key(6ULL, 0ULL),
          "different seeds give different streams");
}

void test_element_bits_and_uniform() {
    for (const DropoutVector& vector : kDropoutVectors) {
        const std::uint64_t stream_key =
            tf::dropout_stream_key(vector.seed, vector.call_index);
        for (int index = 0; index < 4; ++index) {
            check_u64(
                tf::dropout_element_bits(stream_key,
                                         static_cast<std::uint64_t>(index)),
                vector.bits[index], "element-bits known-answer vector");
        }
    }
    // The bits-to-uniform conversion: exact endpoints and exact scaling.
    check_eq(tf::dropout_uniform(0ULL), 0.0, "u(0) == 0.0");
    // All ones -> (2**53 - 1) * 2**-53, the largest value below 1.
    check_eq(tf::dropout_uniform(0xFFFFFFFFFFFFFFFFULL),
             9007199254740991.0 / 9007199254740992.0,
             "u(all ones) is the largest value below 1");
    check(tf::dropout_uniform(0xFFFFFFFFFFFFFFFFULL) < 1.0,
          "u is strictly below 1");
    // The low 11 bits are discarded, so they cannot move u at all.
    check_eq(tf::dropout_uniform(0x0000000000000800ULL - 1ULL), 0.0,
             "u ignores the low 11 bits");
    check_eq(tf::dropout_uniform(0x0000000000000800ULL), 0x1p-53,
             "u of the 12th bit is exactly 2**-53");
}

// The keep/drop pattern the committed vector expects, expanded into the
// multiplier the kernel must write.
double expected_multiplier(const DropoutVector& vector, std::int64_t index) {
    return vector.keep[index] == '1' ? 1.0 / (1.0 - vector.p) : 0.0;
}

void test_forward_known_answers() {
    for (const DropoutVector& vector : kDropoutVectors) {
        std::vector<double> input(static_cast<size_t>(kVectorCount));
        for (std::int64_t i = 0; i < kVectorCount; ++i) {
            input[static_cast<size_t>(i)] = static_cast<double>(i) + 0.5;
        }
        std::vector<double> output(static_cast<size_t>(kVectorCount), kPoison);
        std::vector<double> mask(static_cast<size_t>(kVectorCount), kPoison);
        tf::dropout_forward_contiguous(input.data(), output.data(), mask.data(),
                                       kVectorCount, vector.seed,
                                       vector.call_index, vector.p);
        for (std::int64_t i = 0; i < kVectorCount; ++i) {
            const double want = expected_multiplier(vector, i);
            check_eq(mask[static_cast<size_t>(i)], want,
                     "committed mask multiplier");
            check_eq(output[static_cast<size_t>(i)],
                     input[static_cast<size_t>(i)] * want,
                     "output is input * mask");
        }
        // The mask holds exactly the two locked values and nothing else.
        for (std::int64_t i = 0; i < kVectorCount; ++i) {
            const double value = mask[static_cast<size_t>(i)];
            check(value == 0.0 || value == 1.0 / (1.0 - vector.p),
                  "mask holds exactly 0.0 or the inverted scale");
        }
    }
}

void test_forward_is_repeatable_and_stream_separated() {
    // Deliberately wide. A thresholded mask is one bit per element, so
    // two *different* streams can still agree over a short tensor by
    // chance — at p = 0.5 an eight-element agreement happens once in 256
    // key pairs, and (seed 11, calls 4 and 5) is one such pair. The
    // per-element BITS are what differ; the committed bit vectors in
    // test_element_bits_and_uniform pin those exactly, and this check
    // uses enough elements that an accidental agreement is not a
    // realistic outcome.
    const size_t n = 32;
    std::vector<double> input(n, 3.0);
    std::vector<double> out_a(n, kPoison), mask_a(n, kPoison);
    std::vector<double> out_b(n, kPoison), mask_b(n, kPoison);
    tf::dropout_forward_contiguous(input.data(), out_a.data(), mask_a.data(),
                                   static_cast<std::int64_t>(n), 11ULL, 4ULL,
                                   0.5);
    tf::dropout_forward_contiguous(input.data(), out_b.data(), mask_b.data(),
                                   static_cast<std::int64_t>(n), 11ULL, 4ULL,
                                   0.5);
    for (size_t i = 0; i < n; ++i) {
        check_eq(mask_b[i], mask_a[i], "repeated call reproduces the mask");
        check_eq(out_b[i], out_a[i], "repeated call reproduces the output");
    }
    // A different call index must use a different stream.
    std::vector<double> mask_c(n, kPoison), out_c(n, kPoison);
    tf::dropout_forward_contiguous(input.data(), out_c.data(), mask_c.data(),
                                   static_cast<std::int64_t>(n), 11ULL, 5ULL,
                                   0.5);
    bool differs = false;
    for (size_t i = 0; i < n; ++i) {
        differs = differs || (mask_c[i] != mask_a[i]);
    }
    check(differs, "a different call index uses a different stream");
    // A different seed likewise.
    std::vector<double> mask_d(n, kPoison), out_d(n, kPoison);
    tf::dropout_forward_contiguous(input.data(), out_d.data(), mask_d.data(),
                                   static_cast<std::int64_t>(n), 12ULL, 4ULL,
                                   0.5);
    differs = false;
    for (size_t i = 0; i < n; ++i) {
        differs = differs || (mask_d[i] != mask_a[i]);
    }
    check(differs, "a different seed uses a different stream");
    // The underlying streams differ regardless of the threshold: the raw
    // element bits are the strong statement, and they never collide here.
    check(tf::dropout_element_bits(tf::dropout_stream_key(11ULL, 4ULL), 0ULL) !=
              tf::dropout_element_bits(tf::dropout_stream_key(11ULL, 5ULL),
                                       0ULL),
          "adjacent call indices produce different element bits");
    // The mask does not depend on the input VALUES: the same key over
    // different data produces the identical mask.
    std::vector<double> other(n);
    for (size_t i = 0; i < n; ++i) {
        other[i] = -1000.0 * static_cast<double>(i + 1);
    }
    std::vector<double> mask_e(n, kPoison), out_e(n, kPoison);
    tf::dropout_forward_contiguous(other.data(), out_e.data(), mask_e.data(),
                                   static_cast<std::int64_t>(n), 11ULL, 4ULL,
                                   0.5);
    for (size_t i = 0; i < n; ++i) {
        check_eq(mask_e[i], mask_a[i], "the mask ignores the input values");
    }
}

void test_forward_edge_shapes_and_probabilities() {
    // A single element (the scalar case at the Core layer).
    double scalar_in = 2.5;
    double scalar_out = kPoison, scalar_mask = kPoison;
    tf::dropout_forward_contiguous(&scalar_in, &scalar_out, &scalar_mask, 1,
                                   0ULL, 0ULL, 0.25);
    check_eq(scalar_mask, 1.0 / 0.75, "scalar element kept per the vector");
    check_eq(scalar_out, 2.5 * (1.0 / 0.75), "scalar output is input * mask");
    check_eq(scalar_in, 2.5, "scalar input unchanged");

    // count == 0: no draw, no write. (The Python tensor representation
    // cannot build a zero-element core today, so this is the layer where
    // the empty case is provable — see the Core method's docstring.)
    double untouched_out = kPoison, untouched_mask = kPoison;
    tf::dropout_forward_contiguous(nullptr, &untouched_out, &untouched_mask, 0,
                                   1ULL, 1ULL, 0.5);
    check_eq(untouched_out, kPoison, "count == 0 writes no output");
    check_eq(untouched_mask, kPoison, "count == 0 writes no mask");

    // p == 0: the kernel is still asked to compute (the identity bypass
    // belongs to the Python operation layer, design §6.2), and it keeps
    // every element with multiplier exactly 1.0.
    std::vector<double> input(6);
    for (size_t i = 0; i < 6; ++i) {
        input[i] = static_cast<double>(i) - 2.0;
    }
    std::vector<double> output(6, kPoison), mask(6, kPoison);
    tf::dropout_forward_contiguous(input.data(), output.data(), mask.data(), 6,
                                   99ULL, 3ULL, 0.0);
    for (size_t i = 0; i < 6; ++i) {
        check_eq(mask[i], 1.0, "p == 0 keeps every element with scale 1");
        check_eq(output[i], input[i], "p == 0 reproduces the input");
    }
    // The input is never mutated by any of the above.
    for (size_t i = 0; i < 6; ++i) {
        check_eq(input[i], static_cast<double>(i) - 2.0, "input unchanged");
    }
}

// ---------------------------------------------------------------------------
// Exported C ABI wrapper checks
// ---------------------------------------------------------------------------

// Run the export over plain tf::Storage nodes and report the error code.
int run_export(std::vector<double>& input, std::int64_t input_offset,
               std::vector<double>& output, std::vector<double>& mask,
               std::int64_t count, std::uint64_t seed,
               std::uint64_t call_index, double p) {
    tf::Storage in{input.data(), static_cast<std::int64_t>(input.size())};
    tf::Storage out{output.data(), static_cast<std::int64_t>(output.size())};
    tf::Storage msk{mask.data(), static_cast<std::int64_t>(mask.size())};
    tf_core_dropout_forward(&in, input_offset, &out, &msk, count, seed,
                            call_index, p);
    return tf::last_error_code();
}

// Render a produced mask as a keep/drop pattern: '1' where the element
// carries the inverted scale, '0' where it is exactly 0.0. Anything else
// is a failure, reported as '?' so the comparison message shows it.
void mask_pattern(const std::vector<double>& mask, double scale, char* out) {
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] == 0.0) {
            out[i] = '0';
        } else if (mask[i] == scale) {
            out[i] = '1';
        } else {
            out[i] = '?';
        }
    }
    out[mask.size()] = '\0';
}

void check_pattern(const char* got, const char* want, const char* what) {
    if (std::strcmp(got, want) != 0) {
        std::printf("FAIL: %s (got %s, want %s)\n", what, got, want);
        ++g_failures;
    }
}

// The four inputs the boundary cases run over: distinct, nonzero, and of
// mixed sign, so `output == input * mask` is a real check at every
// position rather than an accident of zeros.
std::vector<double> equality_inputs() {
    return std::vector<double>{1.5, -2.25, 3.75, -4.5};
}

void test_equality_threshold_boundary() {
    // -- the vector is self-consistent with the committed derivation --
    const std::uint64_t stream_key =
        tf::dropout_stream_key(kEqualitySeed, kEqualityCall);
    check_u64(tf::dropout_element_bits(
                  stream_key, static_cast<std::uint64_t>(kEqualityIndex)),
              kEqualityWord, "equality vector reproduces its committed word");
    check_eq(tf::dropout_uniform(kEqualityWord), kEqualityUniform,
             "equality vector reproduces its committed uniform value");
    check(kEqualityUniform > 0.0 && kEqualityUniform < 1.0,
          "the equality uniform lies strictly inside (0, 1)");

    const std::vector<double> input = equality_inputs();
    char pattern[8];

    // -- case 1: p == u. Strict `<` KEEPS the element. --
    {
        const double p = kEqualityUniform;
        const double scale = 1.0 / (1.0 - p);
        std::vector<double> output(4, kPoison), mask(4, kPoison);
        tf::dropout_forward_contiguous(input.data(), output.data(),
                                       mask.data(), kEqualityCount,
                                       kEqualitySeed, kEqualityCall, p);
        mask_pattern(mask, scale, pattern);
        check_pattern(pattern, kEqualityKeepAtEqual,
                      "at p == u the element on the threshold is KEPT");
        check_eq(mask[static_cast<size_t>(kEqualityIndex)], scale,
                 "the kept threshold element carries 1 / (1 - p)");
        check_eq(output[static_cast<size_t>(kEqualityIndex)],
                 input[static_cast<size_t>(kEqualityIndex)] * scale,
                 "the kept threshold element's output is input * scale");
        for (size_t i = 0; i < 4; ++i) {
            const double want = (static_cast<std::int64_t>(i) == kEqualityIndex)
                                    ? scale
                                    : 0.0;
            check_eq(mask[i], want, "p == u mask element");
            check_eq(output[i], input[i] * want, "p == u output is in * mask");
        }
    }

    // -- case 2: p == nextafter(u, 1.0). The element is now DROPPED. --
    {
        const double p = std::nextafter(kEqualityUniform, 1.0);
        check(p > kEqualityUniform && p < 1.0,
              "nextafter(u, 1.0) is a larger legal probability");
        const double scale = 1.0 / (1.0 - p);
        std::vector<double> output(4, kPoison), mask(4, kPoison);
        tf::dropout_forward_contiguous(input.data(), output.data(),
                                       mask.data(), kEqualityCount,
                                       kEqualitySeed, kEqualityCall, p);
        mask_pattern(mask, scale, pattern);
        check_pattern(pattern, kEqualityKeepAtNext,
                      "at nextafter(u) the threshold element is DROPPED");
        check_eq(mask[static_cast<size_t>(kEqualityIndex)], 0.0,
                 "the dropped threshold element's multiplier is exactly 0.0");
        for (size_t i = 0; i < 4; ++i) {
            check_eq(mask[i], 0.0, "nextafter(u) mask element");
            check_eq(output[i], input[i] * 0.0, "nextafter(u) output");
        }
    }

    // -- the same two cases through the EXPORTED path, so the boundary is
    //    proved at the C ABI the Python Core actually calls, not only at
    //    the internal kernel. --
    {
        std::vector<double> source = equality_inputs();
        const double p = kEqualityUniform;
        const double scale = 1.0 / (1.0 - p);
        std::vector<double> output(4, kPoison), mask(4, kPoison);
        const int code = run_export(source, 0, output, mask, kEqualityCount,
                                    kEqualitySeed, kEqualityCall, p);
        check(code == TF_OK, "the export accepts p == u");
        mask_pattern(mask, scale, pattern);
        check_pattern(pattern, kEqualityKeepAtEqual,
                      "exported path keeps the threshold element at p == u");
        check_eq(output[static_cast<size_t>(kEqualityIndex)],
                 source[static_cast<size_t>(kEqualityIndex)] * scale,
                 "exported path output at p == u");

        const double next = std::nextafter(kEqualityUniform, 1.0);
        std::vector<double> next_output(4, kPoison), next_mask(4, kPoison);
        check(run_export(source, 0, next_output, next_mask, kEqualityCount,
                         kEqualitySeed, kEqualityCall, next) == TF_OK,
              "the export accepts nextafter(u)");
        mask_pattern(next_mask, 1.0 / (1.0 - next), pattern);
        check_pattern(pattern, kEqualityKeepAtNext,
                      "exported path drops the threshold element at "
                      "nextafter(u)");
    }
}

// Negative control for the boundary vector.
//
// The point of an equality vector is that it DISCRIMINATES: it must give a
// different answer under the rejected `<=` rule than under the locked `<`
// rule. This computes what a `<=` kernel would have produced from the same
// derivation — only the comparison differs — and proves it disagrees with
// what the production kernel actually produced. If the production
// comparison were ever changed to `<=`, the case-1 checks above would fail;
// this check proves those assertions are not vacuous.
void test_equality_vector_discriminates_the_comparison() {
    const std::uint64_t stream_key =
        tf::dropout_stream_key(kEqualitySeed, kEqualityCall);
    const double p = kEqualityUniform;

    char rejected[8];
    for (std::int64_t i = 0; i < kEqualityCount; ++i) {
        const double u = tf::dropout_uniform(
            tf::dropout_element_bits(stream_key,
                                     static_cast<std::uint64_t>(i)));
        // The ONLY difference from the production rule is `<=` for `<`.
        rejected[i] = (u <= p) ? '0' : '1';
    }
    rejected[kEqualityCount] = '\0';

    // A `<=` kernel drops the threshold element, so it produces the
    // all-dropped pattern...
    check_pattern(rejected, kEqualityKeepAtNext,
                  "the rejected <= rule drops the threshold element");
    // ...which is NOT what the production kernel produced at p == u.
    check(std::strcmp(rejected, kEqualityKeepAtEqual) != 0,
          "the equality vector discriminates < from <= (a <= kernel would "
          "fail test_equality_threshold_boundary)");

    // And the production kernel really is on the `<` side of that split.
    const double scale = 1.0 / (1.0 - p);
    const std::vector<double> input = equality_inputs();
    std::vector<double> output(4, kPoison), mask(4, kPoison);
    tf::dropout_forward_contiguous(input.data(), output.data(), mask.data(),
                                   kEqualityCount, kEqualitySeed,
                                   kEqualityCall, p);
    char produced[8];
    mask_pattern(mask, scale, produced);
    check(std::strcmp(produced, rejected) != 0,
          "the production kernel disagrees with the rejected <= rule at the "
          "equality vector");
    check_pattern(produced, kEqualityKeepAtEqual,
                  "the production kernel implements the strict < rule");
}

void test_export_matches_the_internal_kernel() {
    std::vector<double> input(kVectorCount + 3, 0.0);
    for (std::int64_t i = 0; i < kVectorCount; ++i) {
        input[static_cast<size_t>(i + 3)] = static_cast<double>(i) + 0.5;
    }
    const DropoutVector& vector = kDropoutVectors[2];  // mixed seed, call 0
    std::vector<double> output(static_cast<size_t>(kVectorCount), kPoison);
    std::vector<double> mask(static_cast<size_t>(kVectorCount), kPoison);
    const int code = run_export(input, 3, output, mask, kVectorCount,
                                vector.seed, vector.call_index, vector.p);
    check(code == TF_OK, "a valid export call succeeds");
    for (std::int64_t i = 0; i < kVectorCount; ++i) {
        const double want = expected_multiplier(vector, i);
        check_eq(mask[static_cast<size_t>(i)], want, "export mask matches");
        check_eq(output[static_cast<size_t>(i)],
                 (static_cast<double>(i) + 0.5) * want,
                 "export output matches");
    }
    // The nonzero storage offset shifted the *input* span but not the
    // logical element indices: the mask is the same one the vector pins.
    check_eq(input[0], 0.0, "export did not write before the input span");
}

// Every rejection must leave both destinations byte-for-byte untouched.
void expect_rejected(const char* what, std::vector<double>& input,
                     std::int64_t input_offset, std::int64_t count,
                     double p, std::int64_t output_size,
                     std::int64_t mask_size) {
    std::vector<double> output(static_cast<size_t>(output_size), kPoison);
    std::vector<double> mask(static_cast<size_t>(mask_size), kPoison);
    const int code =
        run_export(input, input_offset, output, mask, count, 7ULL, 2ULL, p);
    check(code == TF_ERROR_INVALID, what);
    for (double value : output) {
        check_eq(value, kPoison, "rejected call left the output untouched");
    }
    for (double value : mask) {
        check_eq(value, kPoison, "rejected call left the mask untouched");
    }
    tf::clear_error();
}

void test_export_rejects_invalid_arguments() {
    std::vector<double> input(4, 1.0);

    // Null handles, each position.
    {
        std::vector<double> output(4, kPoison), mask(4, kPoison);
        tf::Storage out{output.data(), 4};
        tf::Storage msk{mask.data(), 4};
        tf_core_dropout_forward(nullptr, 0, &out, &msk, 4, 1ULL, 0ULL, 0.5);
        check(tf::last_error_code() == TF_ERROR_INVALID,
              "null input handle rejected");
        tf::clear_error();
        tf::Storage in{input.data(), 4};
        tf_core_dropout_forward(&in, 0, nullptr, &msk, 4, 1ULL, 0ULL, 0.5);
        check(tf::last_error_code() == TF_ERROR_INVALID,
              "null output handle rejected");
        tf::clear_error();
        tf_core_dropout_forward(&in, 0, &out, nullptr, 4, 1ULL, 0ULL, 0.5);
        check(tf::last_error_code() == TF_ERROR_INVALID,
              "null mask handle rejected");
        tf::clear_error();
        for (double value : output) {
            check_eq(value, kPoison, "null-handle rejection wrote nothing");
        }
        for (double value : mask) {
            check_eq(value, kPoison, "null-handle rejection wrote nothing");
        }
    }

    // Metadata.
    expect_rejected("negative count rejected", input, 0, -1, 0.5, 4, 4);
    expect_rejected("negative offset rejected", input, -1, 4, 0.5, 4, 4);
    expect_rejected("input span exceeding storage rejected", input, 2, 4, 0.5,
                    4, 4);
    expect_rejected("small output storage rejected", input, 0, 4, 0.5, 3, 4);
    expect_rejected("small mask storage rejected", input, 0, 4, 0.5, 4, 3);

    // Probability.
    expect_rejected("p == 1 rejected", input, 0, 4, 1.0, 4, 4);
    expect_rejected("p > 1 rejected", input, 0, 4, 1.5, 4, 4);
    expect_rejected("negative p rejected", input, 0, 4, -0.25, 4, 4);
    expect_rejected("NaN p rejected", input, 0, 4, kNaN, 4, 4);
    expect_rejected("+inf p rejected", input, 0, 4, kPosInf, 4, 4);
    expect_rejected("-inf p rejected", input, 0, 4, -kPosInf, 4, 4);

    // Aliasing: the same storage node used twice.
    {
        std::vector<double> shared(4, kPoison);
        tf::Storage node{shared.data(), 4};
        tf::Storage other{shared.data(), 4};
        tf_core_dropout_forward(&node, 0, &node, &other, 4, 1ULL, 0ULL, 0.5);
        check(tf::last_error_code() == TF_ERROR_INVALID,
              "output aliasing the input rejected");
        tf::clear_error();
        tf_core_dropout_forward(&node, 0, &other, &node, 4, 1ULL, 0ULL, 0.5);
        check(tf::last_error_code() == TF_ERROR_INVALID,
              "mask aliasing the input rejected");
        tf::clear_error();
        tf::Storage in{input.data(), 4};
        tf_core_dropout_forward(&in, 0, &node, &node, 4, 1ULL, 0ULL, 0.5);
        check(tf::last_error_code() == TF_ERROR_INVALID,
              "output aliasing the mask rejected");
        tf::clear_error();
        for (double value : shared) {
            check_eq(value, kPoison, "aliasing rejection wrote nothing");
        }
    }

    // p == 0 and count == 0 are ACCEPTED at this boundary (the identity
    // bypass is a Python-layer decision, design §6.2).
    {
        std::vector<double> output(4, kPoison), mask(4, kPoison);
        check(run_export(input, 0, output, mask, 4, 1ULL, 0ULL, 0.0) == TF_OK,
              "p == 0 accepted by the export");
        for (size_t i = 0; i < 4; ++i) {
            check_eq(mask[i], 1.0, "p == 0 export mask is all ones");
        }
    }
    {
        std::vector<double> output(4, kPoison), mask(4, kPoison);
        check(run_export(input, 0, output, mask, 0, 1ULL, 0ULL, 0.5) == TF_OK,
              "count == 0 accepted by the export");
        for (size_t i = 0; i < 4; ++i) {
            check_eq(output[i], kPoison, "count == 0 wrote no output");
            check_eq(mask[i], kPoison, "count == 0 wrote no mask");
        }
    }

    // A successful call after a rejected one starts from a clear slot.
    {
        std::vector<double> output(4, kPoison), mask(4, kPoison);
        check(run_export(input, 0, output, mask, 4, 1ULL, 0ULL, 0.5) == TF_OK,
              "a valid call after a rejection reports no error");
    }
}

}  // namespace

int main() {
    test_mix64_known_answers();
    test_stream_known_answers();
    test_element_bits_and_uniform();
    test_forward_known_answers();
    test_forward_is_repeatable_and_stream_separated();
    test_forward_edge_shapes_and_probabilities();
    test_equality_threshold_boundary();
    test_equality_vector_discriminates_the_comparison();
    test_export_matches_the_internal_kernel();
    test_export_rejects_invalid_arguments();

    if (g_failures == 0) {
        std::printf("OK: dropout_forward checks passed\n");
        return 0;
    }
    std::printf("FAILED: %d dropout_forward check(s)\n", g_failures);
    return 1;
}
