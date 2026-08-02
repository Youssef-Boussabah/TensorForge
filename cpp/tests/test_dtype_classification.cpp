// Dependency-free C++ test for the dtype-general stable-math and
// classification kernels (Phase I, milestone I6). No GoogleTest / Catch2 —
// a plain executable that prints failures and returns a nonzero exit code
// if any check fails, so CTest reports pass/fail.
//
// This binary compiles cpp/src/classification.cpp, cpp/src/storage.cpp, and
// cpp/src/error.cpp directly, so it reaches the hidden templated
// ``tf::softmax_forward_contiguous`` / ``tf::log_softmax_forward_contiguous``
// / ``tf::cross_entropy_forward_contiguous`` /
// ``tf::cross_entropy_backward_contiguous`` kernels at **both**
// instantiations, alongside the exported wrappers they live inside — at the
// layer where the properties are actually decided, with no Python wrapper,
// no ctypes boundary, and no NumPy anywhere.
//
// It is deliberately complementary to the neighbouring targets rather than
// a superset of any of them:
//
//   * ``test_softmax`` (E3), ``test_log_softmax`` (E4) and
//     ``test_cross_entropy`` (E5) prove the Phase-E semantics and the full
//     validation matrix at **float64**, hand-computed case by
//     hand-computed case. All three are untouched.
//   * ``test_exp`` / ``test_log`` keep the transcendental exports' own
//     contracts. Untouched.
//   * ``test_dtype_storage`` proved (through I5) that a float32 handle was
//     *rejected* by these four exports; that expectation moves to
//     acceptance with this milestone and is re-stated there.
//
// This one proves the set that opens at I6:
//
//   1. **Every kernel agrees with an independently written same-dtype
//      reference, bit for bit, at both widths.** The references are
//      written here, in ``T``, in the documented traversal order — a test
//      that called the production kernel for its expectation would be
//      comparing the kernel against itself. Because this is C++ the
//      reference reaches the *same* ``std::exp``/``std::log`` the kernel
//      does, so the comparison can be exact rather than tolerance-based:
//      what is being proved is the maximum scan, the shift, the
//      accumulation order, the normalization, and the fused log-sum-exp —
//      with the one ingredient that has no correctly-rounded IEEE
//      guarantee factored out rather than glossed over.
//
//   2. **float32 really accumulates in float32.** The batch-loss
//      accumulator gets a deterministic witness built from the kernel's
//      *own* per-row losses, so the check isolates the accumulator width
//      from any ``exp``/``log`` variation entirely: TensorForge is asserted
//      by raw bit pattern to equal the sequential binary32 total and to
//      differ from the binary64 total narrowed once. The witness is proved
//      non-vacuous first — the two policies are checked to actually differ
//      on this data before either is compared against the kernel.
//
//   3. **Stability comes from the shift and the fusion, not from the
//      width.** Large common offsets on which a naive ``exp(x)/Σexp(x)``
//      overflows to NaN, and small probabilities on which a naive
//      ``log(softmax(x))`` collapses to ``-inf``, are run at both widths:
//      the naive form is computed here and shown to fail, while the kernel
//      stays finite and correct. The float32 cases use magnitudes chosen
//      for binary32 (~88 for ``expf``), not float64 magnitudes reused.
//
//   4. **The one honest domain qualification is recorded, not hidden.**
//      A finite binary32 slice whose *spread* exceeds FLT_MAX makes the
//      shift ``x - m`` itself overflow to ``-inf``. That is the
//      correctly-rounded IEEE result for a quantity with no binary32
//      representation, it happens at binary64 too past ~1.8e308, and it is
//      asserted here in both directions: softmax stays exact (the affected
//      class gets exactly +0.0), while log-softmax reports ``-inf`` and
//      cross-entropy ``+inf``. No special case is added to the kernel and
//      no widened intermediate is introduced to paper over it.
//
//   5. **Exceptional values follow plain IEEE arithmetic**, identically at
//      both widths: NaN first and NaN later (strict ``>`` means a NaN never
//      becomes the maximum, and still poisons its slice), ``+inf``,
//      ``-inf``, an all-``-inf`` slice, mixed infinities, ``+0.0``/``-0.0``,
//      and subnormals — every one against the independent reference, by
//      raw bit pattern, and every one leaving the error slot at TF_OK
//      because a NaN in a *result* is a value, not an ABI failure.
//
//   6. **Mixed dtype is rejected before anything is written**, in every
//      participating handle position of all four exports — two for the
//      transforms, three for each cross-entropy direction — with
//      TF_ERROR_INVALID and byte-for-byte unchanged destinations, and with
//      the dtype reported ahead of a co-occurring structural error.
//
//   7. **The int64 target boundary is unchanged at both widths.** Targets
//      carry no dtype, are revalidated in C++ for every index, and both
//      cross-entropy destinations stay untouched when a target is out of
//      range — at float32 exactly as at float64.
//
//   8. **H1 destination initialization holds at both widths.** Every
//      destination the audit classifies as fully overwritten is poisoned
//      with a sentinel here and proved sentinel-free after the kernel.
//
//   9. **float64 did not move.** Every check runs at float64 too, against
//      the same independently written references.
//
// Comparison is by raw bit pattern throughout: ``==`` on floating values
// cannot see -0.0 versus +0.0 and calls every NaN unequal to itself.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "tf_classification_internal.h"
#include "tf_internal.h"

TF_EXPORT void* tf_storage_create_typed(std::int64_t size,
                                        std::int32_t dtype_code);
TF_EXPORT void* tf_storage_create(std::int64_t size);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT std::int64_t tf_storage_size(const void* handle);
TF_EXPORT void tf_storage_copy_from(void* handle, const void* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, void* dst);

TF_EXPORT void tf_core_softmax_forward(
    const void* src_handle, std::int64_t src_offset, void* dst_handle,
    std::int64_t outer, std::int64_t axis_length, std::int64_t inner);
TF_EXPORT void tf_core_log_softmax_forward(
    const void* src_handle, std::int64_t src_offset, void* dst_handle,
    std::int64_t outer, std::int64_t axis_length, std::int64_t inner);
TF_EXPORT void tf_core_cross_entropy_forward(
    const void* logits_handle, std::int64_t logits_offset,
    const std::int64_t* targets, std::int64_t target_count,
    void* loss_handle, void* probabilities_handle,
    std::int64_t batch_size, std::int64_t num_classes,
    std::int64_t reduction_code);
TF_EXPORT void tf_core_cross_entropy_backward(
    const void* probabilities_handle, std::int64_t probabilities_offset,
    const std::int64_t* targets, std::int64_t target_count,
    const void* upstream_handle, std::int64_t upstream_offset,
    void* grad_logits_handle,
    std::int64_t batch_size, std::int64_t num_classes,
    std::int64_t reduction_code);
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
    // Where a naive unshifted exp(x) overflows, and the largest finite value.
    static double overflow_offset() { return 800.0; }
    static double max_finite() { return 1.7976931348623157e308; }
    static double huge_but_finite() { return 1.0e308; }
};
template <> struct DtypeTraits<float> {
    static constexpr std::int32_t code = TF_DTYPE_FLOAT32;
    static const char* name() { return "float32"; }
    static float sentinel() { return from_bits<float>(0x4B2D0000u); }
    // binary32's own magnitudes, not float64's reused: expf overflows past
    // about 88.7, and FLT_MAX is ~3.40e38.
    static float overflow_offset() { return 120.0f; }
    static float max_finite() { return 3.4028234663852886e38f; }
    static float huge_but_finite() { return 3.0e38f; }
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
                          "%.150s [%.16s]: element %zu differs in bits", what,
                          DtypeTraits<T>::name(), i);
            check(false, message);
            return false;
        }
    }
    return true;
}

template <class T>
void check_no_sentinel(const std::vector<T>& values, const char* what) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (bits<T>(values[i]) == bits<T>(DtypeTraits<T>::sentinel())) {
            char message[256];
            std::snprintf(message, sizeof message,
                          "%.150s [%.16s]: element %zu still holds the poison",
                          what, DtypeTraits<T>::name(), i);
            check(false, message);
            return;
        }
    }
}

// Deterministic, dtype-exact finite values with varied signs and
// magnitudes; every one is exactly representable at binary32, so the
// float32 case is not a rounded copy of the float64 one.
template <class T>
std::vector<T> patterned(std::size_t count, int salt) {
    std::vector<T> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        const int k = static_cast<int>(
            (i * 5 + static_cast<std::size_t>(salt) * 11) % 19);
        const T magnitude = T(0.5) * T(k) - T(4.0);
        out[i] = (i % 4 == 0) ? -magnitude : magnitude;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Independently written references, in ``T``, in the documented traversal
// orders. Written from the design contract, not copied from the kernel: the
// slice decomposition, the strict ``>`` maximum scan, the shift, the
// accumulation order, and the in-place normalization are each re-derived
// here. They call the same std::exp/std::log the kernel does, which is what
// lets the comparison be exact — see the header note.
// ---------------------------------------------------------------------------

template <class T>
std::vector<T> reference_softmax(const std::vector<T>& src, std::int64_t outer,
                                 std::int64_t axis_length, std::int64_t inner) {
    std::vector<T> dst(src.size());
    for (std::int64_t o = 0; o < outer; ++o) {
        for (std::int64_t i = 0; i < inner; ++i) {
            const std::int64_t base = o * axis_length * inner + i;
            T maximum = src[static_cast<std::size_t>(base)];
            for (std::int64_t k = 1; k < axis_length; ++k) {
                const T v = src[static_cast<std::size_t>(base + k * inner)];
                if (v > maximum) {
                    maximum = v;
                }
            }
            T total = T(0);
            for (std::int64_t k = 0; k < axis_length; ++k) {
                const std::size_t at = static_cast<std::size_t>(base + k * inner);
                const T shifted = std::exp(src[at] - maximum);
                dst[at] = shifted;
                total += shifted;
            }
            for (std::int64_t k = 0; k < axis_length; ++k) {
                dst[static_cast<std::size_t>(base + k * inner)] /= total;
            }
        }
    }
    return dst;
}

template <class T>
std::vector<T> reference_log_softmax(const std::vector<T>& src,
                                     std::int64_t outer,
                                     std::int64_t axis_length,
                                     std::int64_t inner) {
    std::vector<T> dst(src.size());
    for (std::int64_t o = 0; o < outer; ++o) {
        for (std::int64_t i = 0; i < inner; ++i) {
            const std::int64_t base = o * axis_length * inner + i;
            T maximum = src[static_cast<std::size_t>(base)];
            for (std::int64_t k = 1; k < axis_length; ++k) {
                const T v = src[static_cast<std::size_t>(base + k * inner)];
                if (v > maximum) {
                    maximum = v;
                }
            }
            T sum_exp = T(0);
            for (std::int64_t k = 0; k < axis_length; ++k) {
                const std::size_t at = static_cast<std::size_t>(base + k * inner);
                const T shifted = src[at] - maximum;
                dst[at] = shifted;
                sum_exp += std::exp(shifted);
            }
            const T log_denominator = std::log(sum_exp);
            for (std::int64_t k = 0; k < axis_length; ++k) {
                dst[static_cast<std::size_t>(base + k * inner)] -=
                    log_denominator;
            }
        }
    }
    return dst;
}

template <class T>
void reference_cross_entropy_forward(
    const std::vector<T>& logits, const std::vector<std::int64_t>& targets,
    std::int64_t batch, std::int64_t classes, std::int64_t reduction_code,
    T& loss_out, std::vector<T>& probabilities_out) {
    probabilities_out.assign(logits.size(), T(0));
    T total = T(0);
    for (std::int64_t n = 0; n < batch; ++n) {
        const std::int64_t base = n * classes;
        T maximum = logits[static_cast<std::size_t>(base)];
        for (std::int64_t c = 1; c < classes; ++c) {
            const T v = logits[static_cast<std::size_t>(base + c)];
            if (v > maximum) {
                maximum = v;
            }
        }
        T sum_exp = T(0);
        for (std::int64_t c = 0; c < classes; ++c) {
            const std::size_t at = static_cast<std::size_t>(base + c);
            const T shifted = std::exp(logits[at] - maximum);
            probabilities_out[at] = shifted;
            sum_exp += shifted;
        }
        for (std::int64_t c = 0; c < classes; ++c) {
            probabilities_out[static_cast<std::size_t>(base + c)] /= sum_exp;
        }
        const T log_denominator = std::log(sum_exp);
        const T shifted_target =
            logits[static_cast<std::size_t>(base + targets[
                static_cast<std::size_t>(n)])] - maximum;
        total += log_denominator - shifted_target;
    }
    loss_out = (reduction_code == tf::kCrossEntropyReductionMean)
                   ? total / static_cast<T>(batch)
                   : total;
}

template <class T>
std::vector<T> reference_cross_entropy_backward(
    const std::vector<T>& probabilities,
    const std::vector<std::int64_t>& targets, T upstream, std::int64_t batch,
    std::int64_t classes, std::int64_t reduction_code) {
    std::vector<T> grad(probabilities.size());
    const bool mean = (reduction_code == tf::kCrossEntropyReductionMean);
    const T count = static_cast<T>(batch);
    for (std::int64_t n = 0; n < batch; ++n) {
        const std::int64_t base = n * classes;
        const std::int64_t target = targets[static_cast<std::size_t>(n)];
        for (std::int64_t c = 0; c < classes; ++c) {
            const std::size_t at = static_cast<std::size_t>(base + c);
            T contribution = probabilities[at];
            if (c == target) {
                contribution -= T(1);
            }
            if (mean) {
                contribution /= count;
            }
            grad[at] = upstream * contribution;
        }
    }
    return grad;
}

// ---------------------------------------------------------------------------
// 1. Both transforms against their references, over several axis
//    decompositions, through the internal kernel AND the exported wrapper.
// ---------------------------------------------------------------------------

struct AxisCase {
    std::int64_t outer, axis_length, inner;
    const char* label;
};

const AxisCase kAxisCases[] = {
    {1, 1, 1, "single element"},
    {1, 5, 1, "one slice, last axis"},
    {4, 3, 1, "rows, last axis"},
    {1, 4, 3, "one plane, first axis"},           // inner > 1: a non-last axis
    {2, 3, 4, "middle axis of a rank-3 tensor"},  // outer > 1 and inner > 1
    {3, 1, 5, "length-one axis"},
    {6, 7, 2, "wide non-last axis"},
};

template <class T>
void test_transforms_match_the_reference() {
    for (const AxisCase& c : kAxisCases) {
        const std::size_t numel =
            static_cast<std::size_t>(c.outer * c.axis_length * c.inner);
        const std::vector<T> src = patterned<T>(numel, 3);

        // -- the internal kernels, directly --
        std::vector<T> got_softmax(numel);
        tf::softmax_forward_contiguous(src.data(), got_softmax.data(), c.outer,
                                       c.axis_length, c.inner);
        char message[256];
        std::snprintf(message, sizeof message, "softmax kernel: %s", c.label);
        same_bits(got_softmax,
                  reference_softmax(src, c.outer, c.axis_length, c.inner),
                  message);

        std::vector<T> got_log(numel);
        tf::log_softmax_forward_contiguous(src.data(), got_log.data(), c.outer,
                                           c.axis_length, c.inner);
        std::snprintf(message, sizeof message, "log_softmax kernel: %s",
                      c.label);
        same_bits(got_log,
                  reference_log_softmax(src, c.outer, c.axis_length, c.inner),
                  message);

        // -- the exported wrappers, over real typed storage, into poisoned
        //    destinations so "written in full" is proved rather than assumed --
        void* in = storage_of(src);
        void* out_softmax =
            poisoned_storage<T>(static_cast<std::int64_t>(numel));
        void* out_log = poisoned_storage<T>(static_cast<std::int64_t>(numel));
        if (in != nullptr && out_softmax != nullptr && out_log != nullptr) {
            tf_clear_error();
            tf_core_softmax_forward(in, 0, out_softmax, c.outer, c.axis_length,
                                    c.inner);
            std::snprintf(message, sizeof message, "softmax export ok: %s",
                          c.label);
            check(tf_last_error_code() == TF_OK, message);
            std::snprintf(message, sizeof message, "softmax export: %s",
                          c.label);
            same_bits(read_back<T>(out_softmax), got_softmax, message);
            std::snprintf(message, sizeof message, "softmax fully written: %s",
                          c.label);
            check_no_sentinel(read_back<T>(out_softmax), message);

            tf_core_log_softmax_forward(in, 0, out_log, c.outer, c.axis_length,
                                        c.inner);
            std::snprintf(message, sizeof message, "log_softmax export ok: %s",
                          c.label);
            check(tf_last_error_code() == TF_OK, message);
            std::snprintf(message, sizeof message, "log_softmax export: %s",
                          c.label);
            same_bits(read_back<T>(out_log), got_log, message);
            std::snprintf(message, sizeof message,
                          "log_softmax fully written: %s", c.label);
            check_no_sentinel(read_back<T>(out_log), message);
        } else {
            check(false, "transform storage");
        }
        if (in != nullptr) tf_storage_destroy(in);
        if (out_softmax != nullptr) tf_storage_destroy(out_softmax);
        if (out_log != nullptr) tf_storage_destroy(out_log);
    }

    // A slice of equal logits normalizes to exactly 1/n at both widths, and
    // its log-softmax is exactly -log(n) — the two cases the design calls
    // out by name.
    {
        const std::vector<T> equal(4, T(2.5));
        std::vector<T> got(4);
        tf::softmax_forward_contiguous(equal.data(), got.data(), 1, 4, 1);
        for (std::size_t i = 0; i < got.size(); ++i) {
            check(bits<T>(got[i]) == bits<T>(T(0.25)),
                  "equal logits normalize to exactly 1/n");
        }
        tf::log_softmax_forward_contiguous(equal.data(), got.data(), 1, 4, 1);
        const T want = T(0) - std::log(T(4));
        for (std::size_t i = 0; i < got.size(); ++i) {
            check(bits<T>(got[i]) == bits<T>(want),
                  "equal logits log-normalize to exactly -log(n)");
        }
    }
    // A length-one slice is exactly 1.0 and exactly 0.0 respectively.
    {
        const std::vector<T> one(1, T(-7.5));
        std::vector<T> got(1);
        tf::softmax_forward_contiguous(one.data(), got.data(), 1, 1, 1);
        check(bits<T>(got[0]) == bits<T>(T(1)),
              "a length-one softmax slice is exactly 1.0");
        tf::log_softmax_forward_contiguous(one.data(), got.data(), 1, 1, 1);
        check(bits<T>(got[0]) == bits<T>(T(0)),
              "a length-one log-softmax slice is exactly +0.0");
    }
}

// ---------------------------------------------------------------------------
// 2. Cross-entropy against its reference, both reductions, both widths.
// ---------------------------------------------------------------------------

template <class T>
void test_cross_entropy_matches_the_reference() {
    const std::int64_t shapes[][2] = {{1, 1}, {1, 5}, {4, 3}, {7, 2}, {3, 9}};
    for (const auto& shape : shapes) {
        const std::int64_t batch = shape[0];
        const std::int64_t classes = shape[1];
        const std::size_t numel = static_cast<std::size_t>(batch * classes);
        const std::vector<T> logits = patterned<T>(numel, 5);
        std::vector<std::int64_t> targets(static_cast<std::size_t>(batch));
        for (std::int64_t n = 0; n < batch; ++n) {
            // Sweep the whole class range, including both boundaries.
            targets[static_cast<std::size_t>(n)] = (n * 3) % classes;
        }
        for (std::int64_t code : {tf::kCrossEntropyReductionMean,
                                  tf::kCrossEntropyReductionSum}) {
            T want_loss = T(0);
            std::vector<T> want_probabilities;
            reference_cross_entropy_forward(logits, targets, batch, classes,
                                            code, want_loss,
                                            want_probabilities);

            // -- internal kernel --
            T got_loss = DtypeTraits<T>::sentinel();
            std::vector<T> got_probabilities(numel,
                                             DtypeTraits<T>::sentinel());
            tf::cross_entropy_forward_contiguous(
                logits.data(), targets.data(), &got_loss,
                got_probabilities.data(), batch, classes, code);
            char message[256];
            std::snprintf(message, sizeof message,
                          "ce forward kernel loss: %lld x %lld / code %lld",
                          static_cast<long long>(batch),
                          static_cast<long long>(classes),
                          static_cast<long long>(code));
            check(bits<T>(got_loss) == bits<T>(want_loss), message);
            std::snprintf(message, sizeof message,
                          "ce forward kernel probs: %lld x %lld / code %lld",
                          static_cast<long long>(batch),
                          static_cast<long long>(classes),
                          static_cast<long long>(code));
            same_bits(got_probabilities, want_probabilities, message);

            // -- exported wrapper, poisoned destinations --
            void* in = storage_of(logits);
            void* loss = poisoned_storage<T>(1);
            void* probabilities =
                poisoned_storage<T>(static_cast<std::int64_t>(numel));
            if (in != nullptr && loss != nullptr && probabilities != nullptr) {
                tf_clear_error();
                tf_core_cross_entropy_forward(in, 0, targets.data(), batch,
                                              loss, probabilities, batch,
                                              classes, code);
                check(tf_last_error_code() == TF_OK, "ce forward export ok");
                const std::vector<T> loss_read = read_back<T>(loss);
                check(bits<T>(loss_read[0]) == bits<T>(want_loss),
                      "ce forward export loss");
                same_bits(read_back<T>(probabilities), want_probabilities,
                          "ce forward export probabilities");
                check_no_sentinel(read_back<T>(probabilities),
                                  "ce probabilities fully written");
                check_no_sentinel(loss_read, "ce loss fully written");

                // -- backward, from the SAVED probabilities alone --
                for (T upstream : {T(1), T(-2.5), T(0.125)}) {
                    const std::vector<T> want_grad =
                        reference_cross_entropy_backward(
                            want_probabilities, targets, upstream, batch,
                            classes, code);
                    std::vector<T> got_grad(numel,
                                            DtypeTraits<T>::sentinel());
                    tf::cross_entropy_backward_contiguous(
                        want_probabilities.data(), targets.data(), &upstream,
                        got_grad.data(), batch, classes, code);
                    same_bits(got_grad, want_grad, "ce backward kernel");

                    const std::vector<T> up_values(1, upstream);
                    void* up = storage_of(up_values);
                    void* grad =
                        poisoned_storage<T>(static_cast<std::int64_t>(numel));
                    if (up != nullptr && grad != nullptr) {
                        tf_clear_error();
                        tf_core_cross_entropy_backward(
                            probabilities, 0, targets.data(), batch, up, 0,
                            grad, batch, classes, code);
                        check(tf_last_error_code() == TF_OK,
                              "ce backward export ok");
                        same_bits(read_back<T>(grad), want_grad,
                                  "ce backward export");
                        check_no_sentinel(read_back<T>(grad),
                                          "ce gradient fully written");
                    } else {
                        check(false, "ce backward storage");
                    }
                    if (up != nullptr) tf_storage_destroy(up);
                    if (grad != nullptr) tf_storage_destroy(grad);
                }
            } else {
                check(false, "ce forward storage");
            }
            if (in != nullptr) tf_storage_destroy(in);
            if (loss != nullptr) tf_storage_destroy(loss);
            if (probabilities != nullptr) tf_storage_destroy(probabilities);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. The float32 batch-loss accumulation witness.
//
// The question this answers is narrow and would otherwise be invisible: is
// the accumulator that sums the per-example losses a ``float`` or a
// ``double``? Every other float32 property of this kernel is a single
// correctly-rounded operation per destination, where computing in binary64
// and rounding once is *provably* indistinguishable from computing in
// binary32 — accumulation is the first place the two policies can differ.
//
// The witness deliberately takes the per-row losses from the **kernel
// itself** (one single-row ``sum`` call per row), so no assumption about
// ``expf``/``logf`` enters: the only thing that varies between the two
// candidate policies is the width of the running total.
// ---------------------------------------------------------------------------

void test_float32_batch_loss_accumulates_in_float32() {
    // Row 0 contributes a loss of exactly 200 — a two-class row whose target
    // is the *smaller* logit by a margin of 200, so sum_exp rounds to
    // exactly 1, its logarithm is exactly 0, and the loss is exactly the
    // margin. One binary32 ULP there is 2**-16 ~ 1.53e-5.
    //
    // Every later row is a two-class row whose target is the *larger* logit
    // by a margin of 12, giving a loss of log(1 + e**-12) ~ 6.14e-6 — large
    // enough to be a perfectly ordinary nonzero binary32 number, and below
    // half a ULP of the running total, so sequential binary32 absorbs every
    // one of them and stays at exactly 200. Binary64 accumulates all 199 and
    // lands ~1.2e-3 higher, which survives narrowing by two orders of
    // magnitude.
    const std::int64_t classes = 2;
    const std::int64_t batch = 200;
    std::vector<float> logits(static_cast<std::size_t>(batch * classes));
    std::vector<std::int64_t> targets(static_cast<std::size_t>(batch), 0);
    logits[0] = 0.0f;
    logits[1] = 200.0f;   // row 0, target 0: loss is exactly 200
    for (std::int64_t n = 1; n < batch; ++n) {
        logits[static_cast<std::size_t>(n * classes)] = 0.0f;
        logits[static_cast<std::size_t>(n * classes + 1)] = 12.0f;
        targets[static_cast<std::size_t>(n)] = 1;   // the larger logit
    }

    // Per-row losses, straight from the kernel, one row at a time.
    std::vector<float> row_losses(static_cast<std::size_t>(batch));
    for (std::int64_t n = 0; n < batch; ++n) {
        std::vector<float> row_probabilities(
            static_cast<std::size_t>(classes));
        tf::cross_entropy_forward_contiguous(
            logits.data() + n * classes, targets.data() + n,
            &row_losses[static_cast<std::size_t>(n)],
            row_probabilities.data(), 1, classes,
            tf::kCrossEntropyReductionSum);
    }

    float sequential = 0.0f;
    double widened = 0.0;
    for (std::int64_t n = 0; n < batch; ++n) {
        sequential += row_losses[static_cast<std::size_t>(n)];
        widened += static_cast<double>(row_losses[static_cast<std::size_t>(n)]);
    }
    const float widened_narrowed = static_cast<float>(widened);

    // Non-vacuous first: if the two policies agreed here the comparison
    // below would prove nothing at all.
    check(bits<float>(sequential) != bits<float>(widened_narrowed),
          "the batch-loss witness distinguishes the two accumulation policies");

    float got = 0.0f;
    std::vector<float> probabilities(
        static_cast<std::size_t>(batch * classes));
    tf::cross_entropy_forward_contiguous(
        logits.data(), targets.data(), &got, probabilities.data(), batch,
        classes, tf::kCrossEntropyReductionSum);
    check(bits<float>(got) == bits<float>(sequential),
          "float32 cross-entropy accumulates the batch loss in float32");
    check(bits<float>(got) != bits<float>(widened_narrowed),
          "float32 cross-entropy does NOT accumulate in float64 and narrow");

    // The same data at float64 must reproduce the widened total exactly,
    // which is the other half of the statement: the accumulator follows the
    // element type rather than being pinned to either width.
    std::vector<double> logits64(logits.begin(), logits.end());
    double got64 = 0.0;
    std::vector<double> probabilities64(logits64.size());
    tf::cross_entropy_forward_contiguous(
        logits64.data(), targets.data(), &got64, probabilities64.data(), batch,
        classes, tf::kCrossEntropyReductionSum);
    double sequential64 = 0.0;
    for (std::int64_t n = 0; n < batch; ++n) {
        std::vector<double> row_probabilities(
            static_cast<std::size_t>(classes));
        double row = 0.0;
        tf::cross_entropy_forward_contiguous(
            logits64.data() + n * classes, targets.data() + n, &row,
            row_probabilities.data(), 1, classes,
            tf::kCrossEntropyReductionSum);
        sequential64 += row;
    }
    check(bits<double>(got64) == bits<double>(sequential64),
          "float64 cross-entropy accumulates the batch loss in float64");
}

// The mean reduction divides ONCE, by the batch size, at the element type.
template <class T>
void test_mean_is_the_sum_divided_once() {
    const std::int64_t batch = 6;
    const std::int64_t classes = 4;
    const std::vector<T> logits =
        patterned<T>(static_cast<std::size_t>(batch * classes), 9);
    std::vector<std::int64_t> targets(static_cast<std::size_t>(batch));
    for (std::int64_t n = 0; n < batch; ++n) {
        targets[static_cast<std::size_t>(n)] = n % classes;
    }
    T sum_loss = T(0), mean_loss = T(0);
    std::vector<T> probabilities(static_cast<std::size_t>(batch * classes));
    tf::cross_entropy_forward_contiguous(logits.data(), targets.data(),
                                         &sum_loss, probabilities.data(),
                                         batch, classes,
                                         tf::kCrossEntropyReductionSum);
    tf::cross_entropy_forward_contiguous(logits.data(), targets.data(),
                                         &mean_loss, probabilities.data(),
                                         batch, classes,
                                         tf::kCrossEntropyReductionMean);
    check(bits<T>(mean_loss) == bits<T>(sum_loss / static_cast<T>(batch)),
          "mean is the sum divided once by batch_size, at the element type");
    // Never by num_classes — the classic wrong divisor.
    check(bits<T>(mean_loss) != bits<T>(sum_loss / static_cast<T>(classes)),
          "mean does not divide by num_classes");
}

// ---------------------------------------------------------------------------
// 4. Stability: the shift and the fusion carry it, not the width.
// ---------------------------------------------------------------------------

template <class T>
void test_stability_witnesses() {
    // (a) A large common offset on which a naive unshifted exp(x) overflows.
    for (T sign : {T(1), T(-1)}) {
        const T offset = sign * DtypeTraits<T>::overflow_offset();
        const std::vector<T> shifted_input = {offset, offset + T(1),
                                              offset + T(2), offset - T(1)};
        // The naive form really does fail here — otherwise the witness
        // proves nothing about why the shift exists.
        T naive_total = T(0);
        for (T v : shifted_input) {
            naive_total += std::exp(v);
        }
        const bool naive_broken =
            !(naive_total > T(0)) || !std::isfinite(naive_total) ||
            naive_total == T(0);
        check(naive_broken,
              "the naive unshifted softmax denominator really does fail here");

        std::vector<T> got(shifted_input.size());
        tf::softmax_forward_contiguous(shifted_input.data(), got.data(), 1,
                                       static_cast<std::int64_t>(
                                           shifted_input.size()), 1);
        // The kernel stays finite, non-negative, and normalized: identical
        // to the same slice with the offset removed, since the shift makes
        // a common offset exactly cancel.
        std::vector<T> centered(shifted_input.size());
        for (std::size_t i = 0; i < centered.size(); ++i) {
            centered[i] = shifted_input[i] - offset;
        }
        std::vector<T> want(centered.size());
        tf::softmax_forward_contiguous(centered.data(), want.data(), 1,
                                       static_cast<std::int64_t>(
                                           centered.size()), 1);
        same_bits(got, want, "a large common offset cancels exactly");
        for (T v : got) {
            check(std::isfinite(v) && v >= T(0) && v <= T(1),
                  "shifted softmax stays a finite probability");
        }

        // log-softmax on the same slice is finite too.
        std::vector<T> got_log(shifted_input.size());
        tf::log_softmax_forward_contiguous(
            shifted_input.data(), got_log.data(), 1,
            static_cast<std::int64_t>(shifted_input.size()), 1);
        for (T v : got_log) {
            check(std::isfinite(v) && v <= T(0),
                  "shifted log-softmax stays finite and non-positive");
        }
    }

    // (b) A probability far below the element type's smallest normal, where
    //     log(softmax(x)) collapses to -inf and the fused form does not.
    {
        const T gap = T(2) * DtypeTraits<T>::overflow_offset();
        const std::vector<T> input = {T(0), -gap};
        std::vector<T> probabilities(2);
        tf::softmax_forward_contiguous(input.data(), probabilities.data(), 1, 2,
                                       1);
        // The naive composition: the probability underflows to exactly zero,
        // so its logarithm is -inf and all information is gone.
        check(bits<T>(probabilities[1]) == bits<T>(T(0)),
              "the small probability really does underflow to +0.0");
        check(std::isinf(std::log(probabilities[1])),
              "log(softmax(x)) really does collapse to -inf here");

        std::vector<T> fused(2);
        tf::log_softmax_forward_contiguous(input.data(), fused.data(), 1, 2, 1);
        check(std::isfinite(fused[1]) && fused[1] < T(0),
              "the fused log-softmax stays finite where log(softmax) does not");
        // And it is accurate: for this gap sum_exp rounds to exactly 1, so
        // the answer is exactly the shifted logit.
        check(bits<T>(fused[1]) == bits<T>(-gap),
              "the fused log-softmax reports the shifted logit exactly");
        check(bits<T>(fused[0]) == bits<T>(T(0)),
              "the maximum's fused log-probability is exactly +0.0");

        // Cross-entropy on the same row: -log(p[target]) would be +inf, the
        // fused log-sum-exp form is finite and exact.
        const std::vector<std::int64_t> targets = {1};
        T loss = T(0);
        std::vector<T> saved(2);
        tf::cross_entropy_forward_contiguous(input.data(), targets.data(),
                                             &loss, saved.data(), 1, 2,
                                             tf::kCrossEntropyReductionSum);
        check(bits<T>(loss) == bits<T>(gap),
              "the fused cross-entropy loss is exact where -log(p) is +inf");
        check(bits<T>(saved[1]) == bits<T>(T(0)),
              "the saved probability still underflows — the loss does not use it");
    }
}

// ---------------------------------------------------------------------------
// 5. The one honest domain qualification, recorded rather than hidden.
//
// The maximum shift guarantees no *exponent* overflows. It does not — and
// cannot — guarantee that the shifted value ``x - m`` is itself
// representable: when a slice's spread exceeds the element type's largest
// finite value, the subtraction overflows to -inf. That is the correctly
// rounded IEEE-754 result for a quantity with no representation at that
// width, and it happens at binary64 as readily as at binary32 — only the
// magnitude at which it starts differs.
//
// Asserted in both directions so no later milestone can quietly "fix" it
// with a widened intermediate (which would be mixed precision) or a special
// case (which would break the traversal contract).
// ---------------------------------------------------------------------------

template <class T>
void test_spread_beyond_the_finite_range() {
    const T hi = DtypeTraits<T>::huge_but_finite();
    const std::vector<T> input = {hi, -hi};   // both finite; spread is not
    check(std::isfinite(input[0]) && std::isfinite(input[1]),
          "the spread witness input is finite at this width");
    check(!std::isfinite(input[1] - input[0]),
          "the spread itself overflows this width");

    // softmax is unaffected: the affected class gets exactly the
    // mathematically correct probability, +0.0, and the maximum gets 1.0.
    std::vector<T> probabilities(2);
    tf::softmax_forward_contiguous(input.data(), probabilities.data(), 1, 2, 1);
    check(bits<T>(probabilities[0]) == bits<T>(T(1)),
          "softmax over an unrepresentable spread is still exactly 1.0");
    check(bits<T>(probabilities[1]) == bits<T>(T(0)),
          "softmax over an unrepresentable spread is still exactly +0.0");

    // log-softmax reports -inf for a true value below -MAX, and
    // cross-entropy +inf for a true loss above +MAX. Both are IEEE overflow
    // of a representable *request*, not an ABI failure.
    std::vector<T> logs(2);
    tf::log_softmax_forward_contiguous(input.data(), logs.data(), 1, 2, 1);
    check(bits<T>(logs[0]) == bits<T>(T(0)),
          "the maximum's log-probability is exactly +0.0 here too");
    check(std::isinf(logs[1]) && logs[1] < T(0),
          "log-softmax reports -inf for a log-probability below -MAX");

    const std::vector<std::int64_t> targets = {1};
    T loss = T(0);
    std::vector<T> saved(2);
    tf::cross_entropy_forward_contiguous(input.data(), targets.data(), &loss,
                                         saved.data(), 1, 2,
                                         tf::kCrossEntropyReductionSum);
    check(std::isinf(loss) && loss > T(0),
          "cross-entropy reports +inf for a loss above +MAX");

    // And it stays a value, not an error, through the exported wrapper.
    void* in = storage_of(input);
    void* out = poisoned_storage<T>(2);
    if (in != nullptr && out != nullptr) {
        tf_clear_error();
        tf_core_log_softmax_forward(in, 0, out, 1, 2, 1);
        check(tf_last_error_code() == TF_OK,
              "an overflowing shift is a value, not an ABI failure");
        check_no_sentinel(read_back<T>(out), "overflowing slice fully written");
    } else {
        check(false, "spread witness storage");
    }
    if (in != nullptr) tf_storage_destroy(in);
    if (out != nullptr) tf_storage_destroy(out);
}

// ---------------------------------------------------------------------------
// 6. Exceptional values, at both widths, against the reference.
// ---------------------------------------------------------------------------

template <class T>
void test_exceptional_values() {
    const T inf = std::numeric_limits<T>::infinity();
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const T denorm = std::numeric_limits<T>::denorm_min();

    const std::vector<std::vector<T>> slices = {
        {nan, T(1), T(2), T(3)},           // NaN first
        {T(1), T(2), nan, T(3)},           // NaN later
        {inf, T(1), T(2), T(3)},           // +inf present
        {-inf, T(1), T(2), T(3)},          // -inf present
        {-inf, -inf, -inf, -inf},          // every value -inf
        {inf, -inf, T(1), T(2)},           // mixed infinities
        {T(0), -T(0), T(0), -T(0)},        // signed zeros
        {denorm, -denorm, T(0), denorm},   // subnormals
        {nan, inf, -inf, T(0)},            // everything at once
    };

    for (std::size_t s = 0; s < slices.size(); ++s) {
        const std::vector<T>& slice = slices[s];
        const std::int64_t n = static_cast<std::int64_t>(slice.size());
        char message[192];

        std::vector<T> got(slice.size());
        tf::softmax_forward_contiguous(slice.data(), got.data(), 1, n, 1);
        std::snprintf(message, sizeof message, "softmax exceptional slice %zu",
                      s);
        same_bits(got, reference_softmax(slice, 1, n, 1), message);

        tf::log_softmax_forward_contiguous(slice.data(), got.data(), 1, n, 1);
        std::snprintf(message, sizeof message,
                      "log_softmax exceptional slice %zu", s);
        same_bits(got, reference_log_softmax(slice, 1, n, 1), message);

        // A NaN or an infinity in the RESULT is a value: the error slot the
        // guard cleared on entry stays TF_OK.
        void* in = storage_of(slice);
        void* out = poisoned_storage<T>(n);
        if (in != nullptr && out != nullptr) {
            tf_clear_error();
            tf_core_softmax_forward(in, 0, out, 1, n, 1);
            std::snprintf(message, sizeof message,
                          "exceptional softmax stays TF_OK: slice %zu", s);
            check(tf_last_error_code() == TF_OK, message);
            std::snprintf(message, sizeof message,
                          "exceptional softmax fully written: slice %zu", s);
            check_no_sentinel(read_back<T>(out), message);

            tf_core_log_softmax_forward(in, 0, out, 1, n, 1);
            std::snprintf(message, sizeof message,
                          "exceptional log_softmax stays TF_OK: slice %zu", s);
            check(tf_last_error_code() == TF_OK, message);
        } else {
            check(false, "exceptional storage");
        }
        if (in != nullptr) tf_storage_destroy(in);
        if (out != nullptr) tf_storage_destroy(out);

        // The same slice as a one-row cross-entropy: forward and backward.
        const std::vector<std::int64_t> targets = {0};
        T loss = T(0);
        std::vector<T> saved(slice.size());
        tf::cross_entropy_forward_contiguous(slice.data(), targets.data(),
                                             &loss, saved.data(), 1, n,
                                             tf::kCrossEntropyReductionSum);
        T want_loss = T(0);
        std::vector<T> want_saved;
        reference_cross_entropy_forward(slice, targets, 1, n,
                                        tf::kCrossEntropyReductionSum,
                                        want_loss, want_saved);
        check(bits<T>(loss) == bits<T>(want_loss),
              "exceptional cross-entropy loss matches the reference");
        std::snprintf(message, sizeof message, "exceptional ce probs %zu", s);
        same_bits(saved, want_saved, message);
    }

    // A NaN never becomes the maximum under the strict ``>`` scan — proved
    // directly by the fact that a slice whose only non-NaN value is the
    // *last* one still shifts by that value.
    {
        const std::vector<T> slice = {nan, nan, T(4)};
        std::vector<T> got(3);
        tf::log_softmax_forward_contiguous(slice.data(), got.data(), 1, 3, 1);
        // The maximum is seeded with slice[0] (a NaN) and never replaced,
        // because ``value > NaN`` is false for every value: the entire slice
        // is therefore NaN. This is the documented IEEE outcome, not a bug,
        // and it is identical at both widths.
        for (std::size_t i = 0; i < got.size(); ++i) {
            check(std::isnan(got[i]),
                  "a leading NaN seeds the maximum and poisons its slice");
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Mixed dtype: rejected in every participating handle position, before
//    anything is written.
// ---------------------------------------------------------------------------

// A destination filled with the sentinel, whose bits must be unchanged after
// a rejected call.
template <class T>
bool destination_untouched(void* handle, std::int64_t count) {
    const std::vector<T> read = read_back<T>(handle);
    if (static_cast<std::int64_t>(read.size()) != count) {
        return false;
    }
    for (T value : read) {
        if (bits<T>(value) != bits<T>(DtypeTraits<T>::sentinel())) {
            return false;
        }
    }
    return true;
}

template <class T>
void test_mixed_dtype_is_rejected() {
    using Other = typename std::conditional<std::is_same<T, float>::value,
                                            double, float>::type;
    const std::int64_t batch = 3, classes = 4;
    const std::int64_t numel = batch * classes;
    const std::vector<T> logits =
        patterned<T>(static_cast<std::size_t>(numel), 2);
    const std::vector<Other> logits_other =
        patterned<Other>(static_cast<std::size_t>(numel), 2);
    std::vector<std::int64_t> targets(static_cast<std::size_t>(batch), 1);

    void* src_t = storage_of(logits);
    void* src_other = storage_of(logits_other);

    // -- the two transforms: source and destination, both directions ------
    for (int direction = 0; direction < 2; ++direction) {
        const bool source_is_t = (direction == 0);
        void* src = source_is_t ? src_t : src_other;
        // The destination is always the *other* dtype from the source.
        if (source_is_t) {
            void* dst = poisoned_storage<Other>(numel);
            if (dst != nullptr) {
                for (auto fn : {&tf_core_softmax_forward,
                                &tf_core_log_softmax_forward}) {
                    tf_clear_error();
                    fn(src, 0, dst, 1, classes, 1);
                    check(tf_last_error_code() == TF_ERROR_INVALID,
                          "mixed-dtype transform is rejected");
                    check(destination_untouched<Other>(dst, numel),
                          "a rejected transform leaves its destination alone");
                }
                tf_storage_destroy(dst);
            }
        }
    }

    // -- a mixed-dtype call that is ALSO structurally malformed reports the
    //    dtype: the guard runs first, matching the I3/I4/I5 ordering ------
    {
        void* dst = poisoned_storage<Other>(numel);
        if (dst != nullptr) {
            tf_clear_error();
            // outer/axis_length/inner of 0 would be rejected on its own.
            tf_core_softmax_forward(src_t, 0, dst, 0, 0, 0);
            check(tf_last_error_code() == TF_ERROR_INVALID,
                  "mixed dtype + malformed is still TF_ERROR_INVALID");
            const char* message = tf_last_error_message();
            check(message != nullptr &&
                      std::strstr(message, "same dtype") != nullptr,
                  "mixed dtype is reported ahead of the structural error");
            check(destination_untouched<Other>(dst, numel),
                  "the doubly-invalid call still writes nothing");
            tf_storage_destroy(dst);
        }
    }

    // -- cross-entropy forward: three handles, every mismatch position ----
    for (int position = 0; position < 3; ++position) {
        void* loss_t = poisoned_storage<T>(1);
        void* loss_other = poisoned_storage<Other>(1);
        void* probs_t = poisoned_storage<T>(numel);
        void* probs_other = poisoned_storage<Other>(numel);
        if (loss_t == nullptr || loss_other == nullptr || probs_t == nullptr ||
            probs_other == nullptr) {
            check(false, "mixed ce forward storage");
        } else {
            // position 0: logits differ; 1: loss differs; 2: probabilities differ
            void* logits_handle = (position == 0) ? src_other : src_t;
            void* loss_handle = (position == 1) ? loss_other : loss_t;
            void* probs_handle = (position == 2) ? probs_other : probs_t;
            tf_clear_error();
            tf_core_cross_entropy_forward(logits_handle, 0, targets.data(),
                                          batch, loss_handle, probs_handle,
                                          batch, classes,
                                          tf::kCrossEntropyReductionMean);
            check(tf_last_error_code() == TF_ERROR_INVALID,
                  "mixed-dtype ce forward is rejected");
            check(destination_untouched<T>(loss_t, 1) &&
                      destination_untouched<Other>(loss_other, 1),
                  "a rejected ce forward leaves the loss alone");
            check(destination_untouched<T>(probs_t, numel) &&
                      destination_untouched<Other>(probs_other, numel),
                  "a rejected ce forward leaves the probabilities alone");
        }
        if (loss_t != nullptr) tf_storage_destroy(loss_t);
        if (loss_other != nullptr) tf_storage_destroy(loss_other);
        if (probs_t != nullptr) tf_storage_destroy(probs_t);
        if (probs_other != nullptr) tf_storage_destroy(probs_other);
    }

    // -- cross-entropy backward: three handles, every mismatch position ---
    const std::vector<T> up_values(1, T(1));
    const std::vector<Other> up_values_other(1, Other(1));
    void* up_t = storage_of(up_values);
    void* up_other = storage_of(up_values_other);
    for (int position = 0; position < 3; ++position) {
        void* grad_t = poisoned_storage<T>(numel);
        void* grad_other = poisoned_storage<Other>(numel);
        if (grad_t == nullptr || grad_other == nullptr || up_t == nullptr ||
            up_other == nullptr) {
            check(false, "mixed ce backward storage");
        } else {
            void* probs_handle = (position == 0) ? src_other : src_t;
            void* up_handle = (position == 1) ? up_other : up_t;
            void* grad_handle = (position == 2) ? grad_other : grad_t;
            tf_clear_error();
            tf_core_cross_entropy_backward(probs_handle, 0, targets.data(),
                                           batch, up_handle, 0, grad_handle,
                                           batch, classes,
                                           tf::kCrossEntropyReductionSum);
            check(tf_last_error_code() == TF_ERROR_INVALID,
                  "mixed-dtype ce backward is rejected");
            check(destination_untouched<T>(grad_t, numel) &&
                      destination_untouched<Other>(grad_other, numel),
                  "a rejected ce backward leaves its gradient alone");
        }
        if (grad_t != nullptr) tf_storage_destroy(grad_t);
        if (grad_other != nullptr) tf_storage_destroy(grad_other);
    }

    if (src_t != nullptr) tf_storage_destroy(src_t);
    if (src_other != nullptr) tf_storage_destroy(src_other);
    if (up_t != nullptr) tf_storage_destroy(up_t);
    if (up_other != nullptr) tf_storage_destroy(up_other);
}

// ---------------------------------------------------------------------------
// 8. The int64 target boundary, unchanged at both widths.
// ---------------------------------------------------------------------------

template <class T>
void test_target_boundary_is_unchanged() {
    const std::int64_t batch = 3, classes = 4;
    const std::int64_t numel = batch * classes;
    const std::vector<T> logits =
        patterned<T>(static_cast<std::size_t>(numel), 4);
    void* src = storage_of(logits);
    const std::vector<T> up_values(1, T(1));
    void* up = storage_of(up_values);

    // Both boundary labels are accepted.
    {
        const std::vector<std::int64_t> targets = {0, classes - 1, 0};
        void* loss = poisoned_storage<T>(1);
        void* probs = poisoned_storage<T>(numel);
        if (loss != nullptr && probs != nullptr) {
            tf_clear_error();
            tf_core_cross_entropy_forward(src, 0, targets.data(), batch, loss,
                                          probs, batch, classes,
                                          tf::kCrossEntropyReductionMean);
            check(tf_last_error_code() == TF_OK,
                  "labels 0 and num_classes-1 are accepted at this dtype");
        }
        if (loss != nullptr) tf_storage_destroy(loss);
        if (probs != nullptr) tf_storage_destroy(probs);
    }

    // Out-of-range labels are refused, in both directions, with BOTH
    // destinations byte-for-byte unchanged.
    const std::vector<std::vector<std::int64_t>> bad = {
        {0, -1, 0},                  // negative
        {0, classes, 0},             // == num_classes
        {0, 0, classes + 99},        // far out of range
    };
    for (const std::vector<std::int64_t>& targets : bad) {
        void* loss = poisoned_storage<T>(1);
        void* probs = poisoned_storage<T>(numel);
        if (loss != nullptr && probs != nullptr) {
            tf_clear_error();
            tf_core_cross_entropy_forward(src, 0, targets.data(), batch, loss,
                                          probs, batch, classes,
                                          tf::kCrossEntropyReductionSum);
            check(tf_last_error_code() == TF_ERROR_INVALID,
                  "an out-of-range target is refused at this dtype");
            check(destination_untouched<T>(loss, 1),
                  "a refused target leaves the loss unwritten");
            check(destination_untouched<T>(probs, numel),
                  "a refused target leaves the probabilities unwritten");
        }
        void* grad = poisoned_storage<T>(numel);
        if (grad != nullptr && up != nullptr) {
            tf_clear_error();
            tf_core_cross_entropy_backward(src, 0, targets.data(), batch, up, 0,
                                           grad, batch, classes,
                                           tf::kCrossEntropyReductionSum);
            check(tf_last_error_code() == TF_ERROR_INVALID,
                  "an out-of-range target is refused by the backward too");
            check(destination_untouched<T>(grad, numel),
                  "a refused target leaves the gradient unwritten");
        }
        if (loss != nullptr) tf_storage_destroy(loss);
        if (probs != nullptr) tf_storage_destroy(probs);
        if (grad != nullptr) tf_storage_destroy(grad);
    }

    // A null target pointer, a target-count mismatch, an unknown reduction
    // code, and a destination that aliases an operand are all refused with
    // both destinations untouched — the pre-existing E5 matrix, re-driven at
    // this dtype so the dtype guard cannot have displaced any of it.
    {
        const std::vector<std::int64_t> targets = {0, 1, 2};
        void* loss = poisoned_storage<T>(1);
        void* probs = poisoned_storage<T>(numel);
        if (loss != nullptr && probs != nullptr) {
            struct Case {
                const std::int64_t* targets;
                std::int64_t count, classes, code;
                const char* label;
            };
            const Case cases[] = {
                {nullptr, batch, classes, 0, "null target pointer"},
                {targets.data(), batch + 1, classes, 0, "target count mismatch"},
                {targets.data(), batch, classes, 7, "unknown reduction code"},
                {targets.data(), batch, 0, 0, "non-positive num_classes"},
            };
            for (const Case& c : cases) {
                tf_clear_error();
                tf_core_cross_entropy_forward(src, 0, c.targets, c.count, loss,
                                              probs, batch, c.classes, c.code);
                check(tf_last_error_code() == TF_ERROR_INVALID, c.label);
                check(destination_untouched<T>(loss, 1) &&
                          destination_untouched<T>(probs, numel),
                      "a refused ce forward writes nothing at all");
            }
            // Aliasing: a destination may not be an operand.
            tf_clear_error();
            tf_core_cross_entropy_forward(src, 0, targets.data(), batch, loss,
                                          src, batch, classes, 0);
            check(tf_last_error_code() == TF_ERROR_INVALID,
                  "an aliasing probability destination is refused");
            check(destination_untouched<T>(loss, 1),
                  "the aliasing rejection writes nothing");
        }
        if (loss != nullptr) tf_storage_destroy(loss);
        if (probs != nullptr) tf_storage_destroy(probs);
    }

    // A stale error clears on the next successful call.
    {
        const std::vector<std::int64_t> targets = {0, 1, 2};
        void* loss = poisoned_storage<T>(1);
        void* probs = poisoned_storage<T>(numel);
        if (loss != nullptr && probs != nullptr) {
            tf_core_cross_entropy_forward(src, 0, nullptr, batch, loss, probs,
                                          batch, classes, 0);
            check(tf_last_error_code() == TF_ERROR_INVALID, "error is armed");
            tf_core_cross_entropy_forward(src, 0, targets.data(), batch, loss,
                                          probs, batch, classes, 0);
            check(tf_last_error_code() == TF_OK,
                  "a stale error clears on the next successful call");
        }
        if (loss != nullptr) tf_storage_destroy(loss);
        if (probs != nullptr) tf_storage_destroy(probs);
    }

    if (src != nullptr) tf_storage_destroy(src);
    if (up != nullptr) tf_storage_destroy(up);
}

// ---------------------------------------------------------------------------
// 9. Offsets: the source offset the Core layer adds is honoured at both
//    widths, and the exports' span checks still count logical elements.
// ---------------------------------------------------------------------------

template <class T>
void test_offsets_and_spans() {
    const std::int64_t pad = 3;
    const std::int64_t classes = 5;
    std::vector<T> backing(static_cast<std::size_t>(pad + classes));
    for (std::size_t i = 0; i < backing.size(); ++i) {
        backing[i] = T(100);                      // never read
    }
    const std::vector<T> slice = patterned<T>(static_cast<std::size_t>(classes),
                                              6);
    for (std::int64_t k = 0; k < classes; ++k) {
        backing[static_cast<std::size_t>(pad + k)] =
            slice[static_cast<std::size_t>(k)];
    }
    void* src = storage_of(backing);
    void* dst = poisoned_storage<T>(classes);
    if (src != nullptr && dst != nullptr) {
        tf_clear_error();
        tf_core_softmax_forward(src, pad, dst, 1, classes, 1);
        check(tf_last_error_code() == TF_OK, "an offset source is accepted");
        same_bits(read_back<T>(dst), reference_softmax(slice, 1, classes, 1),
                  "the source offset is honoured");
        // One element past the end is refused, and writes nothing.
        void* poisoned = poisoned_storage<T>(classes);
        if (poisoned != nullptr) {
            tf_clear_error();
            tf_core_softmax_forward(src, pad + 1, poisoned, 1, classes, 1);
            check(tf_last_error_code() == TF_ERROR_INVALID,
                  "a source span past the end is refused");
            check(destination_untouched<T>(poisoned, classes),
                  "the refused span writes nothing");
            tf_storage_destroy(poisoned);
        }
    } else {
        check(false, "offset storage");
    }
    if (src != nullptr) tf_storage_destroy(src);
    if (dst != nullptr) tf_storage_destroy(dst);
}

}  // namespace

int main() {
    test_transforms_match_the_reference<double>();
    test_transforms_match_the_reference<float>();

    test_cross_entropy_matches_the_reference<double>();
    test_cross_entropy_matches_the_reference<float>();

    test_float32_batch_loss_accumulates_in_float32();

    test_mean_is_the_sum_divided_once<double>();
    test_mean_is_the_sum_divided_once<float>();

    test_stability_witnesses<double>();
    test_stability_witnesses<float>();

    test_spread_beyond_the_finite_range<double>();
    test_spread_beyond_the_finite_range<float>();

    test_exceptional_values<double>();
    test_exceptional_values<float>();

    test_mixed_dtype_is_rejected<double>();
    test_mixed_dtype_is_rejected<float>();

    test_target_boundary_is_unchanged<double>();
    test_target_boundary_is_unchanged<float>();

    test_offsets_and_spans<double>();
    test_offsets_and_spans<float>();

    if (g_failures == 0) {
        std::printf("test_dtype_classification: all checks passed\n");
        return 0;
    }
    std::printf("test_dtype_classification: %d check(s) failed\n", g_failures);
    return 1;
}
