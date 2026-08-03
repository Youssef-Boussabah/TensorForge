// Dependency-free C++ test for the typed array transfer, view
// materialization, and identity-copy boundaries (Phase I, milestone I2).
// No GoogleTest / Catch2 — a plain executable that prints failures and
// returns a nonzero exit code if any check fails, so CTest reports
// pass/fail.
//
// This binary compiles cpp/src/storage.cpp, cpp/src/elementwise.cpp, and
// cpp/src/error.cpp directly, so it reaches the hidden
// ``tf::copy_prefers_contiguous`` predicate alongside the four exports the
// milestone generalizes, at the layer where the properties are actually
// decided — no Python wrapper, no ctypes boundary, no NumPy anywhere.
//
// What it proves:
//
//   1. **Contiguous host transfer at both dtypes.** ``tf_storage_copy_from``
//      and ``tf_storage_copy_to`` move exactly ``size`` elements of the
//      storage's own element type, in order, preserving every object
//      representation: signed zeros, both infinities, subnormals, quiet
//      NaNs with distinct payloads, and signalling NaNs of both signs. The
//      dtype tag and the logical size are unchanged by a transfer.
//   2. **Strided materialization at both dtypes.** Scalar, 1-D, 2-D
//      row-major, transposed, narrowed-with-offset, non-unit stride,
//      negative stride, broadcast (stride 0), and chained-view layouts all
//      materialize in row-major destination order with exact bits, against
//      an independently walked reference odometer.
//   3. **The identity copy is dtype-preserving and dtype-strict.**
//      ``tf_core_contiguous_copy`` copies float32 to float32 and float64
//      to float64 across all three H5/H8 traversal tiers with identical
//      bits, and **rejects a mixed-dtype pair before writing anything** —
//      float32-to-float64 is not a conversion opportunity, it is an
//      invalid request.
//   4. **Failure behavior is unchanged and total.** An invalid span, a
//      negative ndim or offset, a non-positive extent, an element-count
//      overflow, a stride-arithmetic overflow, and an undersized
//      destination are all rejected with TF_ERROR_INVALID and leave the
//      destination byte-for-byte unchanged, at both dtypes; a later
//      success clears the slot.
//
// Bit comparison, never a tolerance: the entire question is whether object
// representations survive, and ``==`` on floating values cannot see -0.0
// versus +0.0 and calls every NaN unequal to itself.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "tf_copy_internal.h"
#include "tf_internal.h"

TF_EXPORT void* tf_storage_create(std::int64_t size);
TF_EXPORT void* tf_storage_create_typed(std::int64_t size,
                                        std::int32_t dtype_code);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT std::int64_t tf_storage_size(const void* handle);
TF_EXPORT void tf_storage_fill(void* handle, double value);
// The three retyped boundaries. ``void*`` host positions; the storage
// handle's immutable dtype tag decides how they are read.
TF_EXPORT void tf_storage_copy_from(void* handle, const void* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, void* dst);
TF_EXPORT void tf_storage_materialize(
    const void* handle, void* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);
TF_EXPORT void tf_core_contiguous_copy(
    const void* src, void* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);
TF_EXPORT int tf_last_error_code();
TF_EXPORT const char* tf_last_error_message();
TF_EXPORT void tf_clear_error();
TF_EXPORT void tf_test_arm_alloc_failure(std::int64_t nth);

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    }
}

// ---------------------------------------------------------------------------
// Bit-pattern plumbing.
//
// Every value in this file is *built from* and *compared as* raw bits. The
// patterns are never produced by arithmetic and never compared with ``==``,
// because the three cases the transfer contract exists for — negative zero,
// NaN payloads, and signalling NaNs — are exactly the ones an arithmetic
// build-or-compare would silently launder.
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

// The binary64 sweep, in the order the Python suite and the H5 CTest use.
const std::uint64_t kF64Patterns[] = {
    0x0000000000000000ull,  // +0.0
    0x8000000000000000ull,  // -0.0                   <- arithmetic normalizes
    0x3FF0000000000000ull,  // 1.0
    0xBFF0000000000000ull,  // -1.0
    0x7FF0000000000000ull,  // +inf
    0xFFF0000000000000ull,  // -inf
    0x7FF8000000000001ull,  // quiet NaN, payload 1
    0x7FF800000000000Aull,  // quiet NaN, payload A
    0xFFF8000000000001ull,  // negative quiet NaN
    0x7FF0000000000001ull,  // signalling NaN         <- arithmetic quiets
    0xFFF0000000000001ull,  // negative signalling NaN
    0x0000000000000001ull,  // smallest positive subnormal
    0x8000000000000001ull,  // -smallest subnormal
    0x000FFFFFFFFFFFFFull,  // largest subnormal
    0x0010000000000000ull,  // smallest positive normal
    0x7FEFFFFFFFFFFFFFull,  // largest finite
    0xFFEFFFFFFFFFFFFFull,  // -largest finite
};

// The binary32 sweep — the same seventeen classes at the narrower width,
// so the two dtypes are held to one contract rather than two.
const std::uint32_t kF32Patterns[] = {
    0x00000000u,  // +0.0
    0x80000000u,  // -0.0                   <- arithmetic normalizes
    0x3F800000u,  // 1.0
    0xBF800000u,  // -1.0
    0x7F800000u,  // +inf
    0xFF800000u,  // -inf
    0x7FC00001u,  // quiet NaN, payload 1
    0x7FC0000Au,  // quiet NaN, payload A
    0xFFC00001u,  // negative quiet NaN
    0x7F800001u,  // signalling NaN         <- arithmetic quiets
    0xFF800001u,  // negative signalling NaN
    0x00000001u,  // smallest positive subnormal
    0x80000001u,  // -smallest subnormal
    0x007FFFFFu,  // largest subnormal
    0x00800000u,  // smallest positive normal
    0x7F7FFFFFu,  // largest finite
    0xFF7FFFFFu,  // -largest finite
};

const std::int64_t kPatternCount =
    static_cast<std::int64_t>(sizeof(kF64Patterns) / sizeof(kF64Patterns[0]));
static_assert(sizeof(kF32Patterns) / sizeof(kF32Patterns[0]) == 17,
              "the two sweeps must cover the same seventeen classes");

// One traits struct so every test below is written once and instantiated
// twice. ``code`` is the frozen ABI dtype code; ``name`` is only for
// failure messages.
template <class T> struct DtypeTraits;
template <> struct DtypeTraits<double> {
    static constexpr std::int32_t code = TF_DTYPE_FLOAT64;
    static constexpr tf::Dtype dtype = tf::Dtype::Float64;
    // A finite value that appears nowhere in the sweep, used to pre-fill a
    // destination so an unwritten element is detectable rather than
    // accidentally right.
    static constexpr std::uint64_t sentinel = 0x4B2D000000000000ull;
    static const char* name() { return "float64"; }
    static const std::uint64_t* patterns() { return kF64Patterns; }
};
template <> struct DtypeTraits<float> {
    static constexpr std::int32_t code = TF_DTYPE_FLOAT32;
    static constexpr tf::Dtype dtype = tf::Dtype::Float32;
    static constexpr std::uint32_t sentinel = 0x4B2D0000u;
    static const char* name() { return "float32"; }
    static const std::uint32_t* patterns() { return kF32Patterns; }
};

// A vector of ``n`` values drawn from the dtype's sweep, cycling so that a
// size larger than the sweep still exercises every class and a size smaller
// than it still starts at +0.0 and -0.0.
template <class T>
std::vector<T> pattern_values(std::int64_t n) {
    std::vector<T> out(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] =
            from_bits<T>(DtypeTraits<T>::patterns()[i % kPatternCount]);
    }
    return out;
}

// Typed storage holding ``values``, or null. The caller destroys it.
template <class T>
void* storage_of(const std::vector<T>& values) {
    void* handle = tf_storage_create_typed(
        static_cast<std::int64_t>(values.size()), DtypeTraits<T>::code);
    if (handle == nullptr) {
        return nullptr;
    }
    tf_storage_copy_from(handle, values.data());
    return handle;
}

// Compare two element runs as raw bits and report the first difference.
template <class T>
bool same_bits(const T* a, const T* b, std::int64_t n, const char* what) {
    for (std::int64_t i = 0; i < n; ++i) {
        if (bits<T>(a[static_cast<std::size_t>(i)])
                != bits<T>(b[static_cast<std::size_t>(i)])) {
            char message[256];
            // Bounded conversions throughout this file: the caller-supplied
            // labels are runtime pointers, so an unbounded %s leaves the
            // compiler unable to prove the buffer suffices and it warns.
            // A precision states the bound instead of relying on truncation.
            std::snprintf(message, sizeof message,
                          "%.150s [%.16s]: element %lld differs in bits", what,
                          DtypeTraits<T>::name(), static_cast<long long>(i));
            check(false, message);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// 1. Contiguous host transfer: copy_from / copy_to at both dtypes
// ---------------------------------------------------------------------------

template <class T>
void test_host_round_trip_preserves_every_bit() {
    char message[256];
    // Several sizes, including one below the sweep, one exactly the sweep,
    // and two above it, so a size-dependent bug in the loop bound shows up.
    const std::int64_t sizes[] = {1, 2, kPatternCount, kPatternCount + 1, 1000};
    for (std::size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
        const std::int64_t n = sizes[s];
        std::vector<T> source = pattern_values<T>(n);
        void* handle = storage_of(source);
        std::snprintf(message, sizeof message, "%s: storage of %lld elements",
                      DtypeTraits<T>::name(), static_cast<long long>(n));
        check(handle != nullptr, message);
        if (handle == nullptr) {
            continue;
        }
        // The logical size is an element count at both widths, and a
        // transfer does not move it.
        std::snprintf(message, sizeof message,
                      "%s: tf_storage_size changed across a transfer",
                      DtypeTraits<T>::name());
        check(tf_storage_size(handle) == n, message);
        // ...and the dtype tag is likewise untouched by moving data.
        std::snprintf(message, sizeof message,
                      "%s: the dtype tag changed across a transfer",
                      DtypeTraits<T>::name());
        check(tf::storage_dtype(handle) == DtypeTraits<T>::dtype, message);

        std::vector<T> out(static_cast<std::size_t>(n),
                           from_bits<T>(DtypeTraits<T>::patterns()[2]));
        tf_storage_copy_to(handle, out.data());
        same_bits(source.data(), out.data(), n, "host round trip");

        // Repeated round trips must not erode anything — a quieting or a
        // normalization would show up on the second pass if not the first.
        for (int pass = 0; pass < 3; ++pass) {
            tf_storage_copy_from(handle, out.data());
            tf_storage_copy_to(handle, out.data());
        }
        same_bits(source.data(), out.data(), n, "repeated round trips");

        // First, middle, and last elements named explicitly, so a failure
        // localizes rather than merely reporting "something differs".
        std::snprintf(message, sizeof message, "%s: first element",
                      DtypeTraits<T>::name());
        check(bits<T>(out[0]) == bits<T>(source[0]), message);
        std::snprintf(message, sizeof message, "%s: middle element",
                      DtypeTraits<T>::name());
        check(bits<T>(out[static_cast<std::size_t>(n / 2)])
                  == bits<T>(source[static_cast<std::size_t>(n / 2)]),
              message);
        std::snprintf(message, sizeof message, "%s: last element",
                      DtypeTraits<T>::name());
        check(bits<T>(out[static_cast<std::size_t>(n - 1)])
                  == bits<T>(source[static_cast<std::size_t>(n - 1)]),
              message);

        tf_storage_destroy(handle);
    }
}

// The transfer touches exactly ``size * itemsize`` bytes and not one more.
// A float32 buffer walked as ``double`` would run off the end by exactly a
// factor of two, which is the failure this guards; the guard band is
// checked as raw bytes because its contents are not floating values.
template <class T>
void test_a_transfer_stays_inside_its_own_buffer() {
    const std::int64_t n = 64;
    const std::size_t guard = 64;
    const std::size_t span = static_cast<std::size_t>(n) * sizeof(T);
    // One allocation: [guard][n elements][guard], with the element region
    // suitably aligned because it begins at a multiple of the guard size.
    std::vector<unsigned char> arena(2 * guard + span, 0xA5);
    void* handle = tf_storage_create_typed(n, DtypeTraits<T>::code);
    check(handle != nullptr, "guard-band storage");
    if (handle == nullptr) {
        return;
    }
    T* region = reinterpret_cast<T*>(arena.data() + guard);
    std::vector<T> values = pattern_values<T>(n);
    std::memcpy(region, values.data(), span);

    tf_storage_copy_from(handle, region);
    std::memset(arena.data() + guard, 0, span);
    tf_storage_copy_to(handle, region);

    char message[256];
    std::snprintf(message, sizeof message,
                  "%s: a transfer wrote outside its host buffer",
                  DtypeTraits<T>::name());
    bool intact = true;
    for (std::size_t i = 0; i < guard; ++i) {
        intact = intact && arena[i] == 0xA5
                 && arena[guard + span + i] == 0xA5;
    }
    check(intact, message);
    same_bits(values.data(), region, n, "guarded round trip");
    tf_storage_destroy(handle);
}

// ---------------------------------------------------------------------------
// 2. Strided materialization
// ---------------------------------------------------------------------------

// The reference odometer, written independently of the kernel so the two
// can disagree. Row-major destination order, element-valued strides.
template <class T>
void reference_materialize(const std::vector<T>& storage, std::vector<T>& out,
                           const std::int64_t* shape,
                           const std::int64_t* strides, std::int64_t offset,
                           std::int64_t ndim) {
    if (ndim == 0) {
        out.assign(1, storage[static_cast<std::size_t>(offset)]);
        return;
    }
    std::int64_t total = 1;
    for (std::int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
    }
    out.assign(static_cast<std::size_t>(total), T());
    std::vector<std::int64_t> counter(static_cast<std::size_t>(ndim), 0);
    std::int64_t pos = offset;
    for (std::int64_t i = 0; i < total; ++i) {
        out[static_cast<std::size_t>(i)] = storage[static_cast<std::size_t>(pos)];
        for (std::int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[static_cast<std::size_t>(d)];
            pos += strides[d];
            if (counter[static_cast<std::size_t>(d)] < shape[d]) {
                break;
            }
            counter[static_cast<std::size_t>(d)] = 0;
            pos -= shape[d] * strides[d];
        }
    }
}

struct Layout {
    const char* name;
    std::int64_t ndim;
    std::int64_t shape[3];
    std::int64_t strides[3];
    std::int64_t offset;
};

// Every layout class the metadata contract permits over a 4x5 storage of
// 20 elements (plus the rank-3 chained case over the same buffer).
const Layout kLayouts[] = {
    {"scalar", 0, {0, 0, 0}, {0, 0, 0}, 7},
    {"scalar at the last element", 0, {0, 0, 0}, {0, 0, 0}, 19},
    {"1-D contiguous", 1, {20, 0, 0}, {1, 0, 0}, 0},
    {"1-D non-unit stride", 1, {5, 0, 0}, {4, 0, 0}, 0},
    {"1-D reversed", 1, {5, 0, 0}, {-4, 0, 0}, 16},
    {"2-D contiguous", 2, {4, 5, 0}, {5, 1, 0}, 0},
    {"2-D transposed", 2, {5, 4, 0}, {1, 5, 0}, 0},
    {"2-D narrowed with offset", 2, {2, 3, 0}, {5, 1, 0}, 6},
    {"2-D broadcast row (stride 0)", 2, {3, 5, 0}, {0, 1, 0}, 5},
    {"2-D broadcast column (stride 0)", 2, {4, 3, 0}, {5, 0, 0}, 0},
    {"2-D unit extent", 2, {1, 20, 0}, {20, 1, 0}, 0},
    {"3-D chained view", 3, {2, 2, 5}, {10, 5, 1}, 0},
    {"3-D transposed chain", 3, {5, 2, 2}, {1, 10, 5}, 0},
};

template <class T>
void test_materialization_matches_the_reference_bit_for_bit() {
    char message[256];
    std::vector<T> values = pattern_values<T>(20);
    void* handle = storage_of(values);
    check(handle != nullptr, "materialization storage");
    if (handle == nullptr) {
        return;
    }
    for (std::size_t i = 0; i < sizeof(kLayouts) / sizeof(kLayouts[0]); ++i) {
        const Layout& layout = kLayouts[i];
        std::vector<T> expected;
        reference_materialize(values, expected, layout.shape, layout.strides,
                              layout.offset, layout.ndim);
        // The destination is pre-filled with a value that appears nowhere
        // in the source, so an element the kernel failed to write is
        // detectable rather than accidentally right.
        std::vector<T> got(expected.size(),
                           from_bits<T>(DtypeTraits<T>::sentinel));
        tf_clear_error();
        tf_storage_materialize(handle, got.data(), layout.shape,
                               layout.strides, layout.offset, layout.ndim);
        std::snprintf(message, sizeof message, "%s [%s]: materialize errored",
                      layout.name, DtypeTraits<T>::name());
        check(tf_last_error_code() == TF_OK, message);
        std::snprintf(message, sizeof message, "%s: materialize", layout.name);
        same_bits(expected.data(), got.data(),
                  static_cast<std::int64_t>(expected.size()), message);
    }
    tf_storage_destroy(handle);
}

// ---------------------------------------------------------------------------
// 3. The identity copy: dtype-preserving, dtype-strict, and unchanged for
//    float64
// ---------------------------------------------------------------------------

template <class T>
void test_identity_copy_preserves_bits_on_every_traversal() {
    char message[256];
    std::vector<T> values = pattern_values<T>(20);
    void* src = storage_of(values);
    check(src != nullptr, "identity-copy source");
    if (src == nullptr) {
        return;
    }
    for (std::size_t i = 0; i < sizeof(kLayouts) / sizeof(kLayouts[0]); ++i) {
        const Layout& layout = kLayouts[i];
        std::vector<T> expected;
        reference_materialize(values, expected, layout.shape, layout.strides,
                              layout.offset, layout.ndim);
        void* dst = tf_storage_create_typed(
            static_cast<std::int64_t>(expected.size()), DtypeTraits<T>::code);
        check(dst != nullptr, "identity-copy destination");
        if (dst == nullptr) {
            continue;
        }
        tf_clear_error();
        tf_core_contiguous_copy(src, dst, layout.shape, layout.strides,
                                layout.offset, layout.ndim);
        std::snprintf(message, sizeof message, "%s [%s]: identity copy errored",
                      layout.name, DtypeTraits<T>::name());
        check(tf_last_error_code() == TF_OK, message);
        std::vector<T> got(expected.size());
        tf_storage_copy_to(dst, got.data());
        std::snprintf(message, sizeof message, "%s: identity copy",
                      layout.name);
        same_bits(expected.data(), got.data(),
                  static_cast<std::int64_t>(expected.size()), message);
        // ...and the traversal the predicate chose is recorded beside the
        // result, so a layout silently moving between tiers is visible.
        const bool flat = tf::copy_prefers_contiguous(
            layout.shape, layout.strides, layout.ndim);
        std::snprintf(message, sizeof message,
                      "%s: predicate disagrees with the layout's contiguity",
                      layout.name);
        check(flat == (i == 0 || i == 1 || i == 2 || i == 5 || i == 10
                       || i == 11),
              message);
        tf_storage_destroy(dst);
    }
    tf_storage_destroy(src);
}

// A rejected call writes **nothing**. The destination is filled with a
// sentinel and compared as raw bytes afterwards.
void expect_rejected(const char* what, void* dst, std::int64_t dst_bytes,
                     const std::vector<unsigned char>& before,
                     const char* must_name) {
    char message[256];
    std::snprintf(message, sizeof message, "%.150s: was not rejected", what);
    check(tf_last_error_code() == TF_ERROR_INVALID, message);
    if (must_name != nullptr) {
        std::snprintf(message, sizeof message,
                      "%.150s: the message does not name %.32s", what,
                      must_name);
        check(std::strstr(tf_last_error_message(), must_name) != nullptr,
              message);
    }
    std::snprintf(message, sizeof message, "%.150s: mutated its destination",
                  what);
    check(std::memcmp(before.data(),
                      reinterpret_cast<const unsigned char*>(
                          tf::as_storage(dst)->data),
                      static_cast<std::size_t>(dst_bytes)) == 0,
          message);
    tf_clear_error();
}

std::vector<unsigned char> snapshot(const void* handle, std::int64_t bytes) {
    const unsigned char* raw =
        reinterpret_cast<const unsigned char*>(tf::as_storage(handle)->data);
    return std::vector<unsigned char>(raw, raw + bytes);
}

void test_mixed_dtype_identity_copy_is_rejected_before_any_write() {
    const std::int64_t shape[2] = {4, 5};
    const std::int64_t strides[2] = {5, 1};

    void* f32 = tf_storage_create_typed(20, TF_DTYPE_FLOAT32);
    void* f64 = tf_storage_create_typed(20, TF_DTYPE_FLOAT64);
    check(f32 != nullptr && f64 != nullptr, "mixed-dtype fixtures");
    if (f32 == nullptr || f64 == nullptr) {
        if (f32 != nullptr) tf_storage_destroy(f32);
        if (f64 != nullptr) tf_storage_destroy(f64);
        return;
    }
    tf_storage_fill(f64, -1234.5);  // a float64-only primitive: legal here
    tf_clear_error();

    {   // float32 source into a float64 destination.
        std::vector<unsigned char> before = snapshot(f64, 20 * 8);
        tf_core_contiguous_copy(f32, f64, shape, strides, 0, 2);
        expect_rejected("mixed dtype (f32 -> f64)", f64, 20 * 8, before,
                        "float32");
    }
    {   // ...and the reverse, where the float32 buffer is the destination
        // and a float64 walk of 20 elements would overrun it twofold.
        std::vector<unsigned char> before = snapshot(f32, 20 * 4);
        tf_core_contiguous_copy(f64, f32, shape, strides, 0, 2);
        expect_rejected("mixed dtype (f64 -> f32)", f32, 20 * 4, before,
                        "float32");
    }
    // The rejection names both dtypes and says why, rather than merely
    // failing.
    tf_core_contiguous_copy(f32, f64, shape, strides, 0, 2);
    check(std::strstr(tf_last_error_message(), "float64") != nullptr,
          "the mixed-dtype message does not name float64");
    check(std::strstr(tf_last_error_message(), "same dtype") != nullptr,
          "the mixed-dtype message does not state the rule");
    tf_clear_error();

    // ...and a same-dtype call through the very same export still works,
    // which is what makes the rejection a dtype rule rather than a broken
    // kernel.
    void* out32 = tf_storage_create_typed(20, TF_DTYPE_FLOAT32);
    check(out32 != nullptr, "same-dtype destination");
    if (out32 != nullptr) {
        tf_core_contiguous_copy(f32, out32, shape, strides, 0, 2);
        check(tf_last_error_code() == TF_OK,
              "a same-dtype float32 copy was rejected");
        tf_storage_destroy(out32);
    }
    tf_clear_error();
    tf_storage_destroy(f32);
    tf_storage_destroy(f64);
}

// ---------------------------------------------------------------------------
// 4. Failure behavior
// ---------------------------------------------------------------------------

template <class T>
void test_invalid_metadata_is_rejected_and_writes_nothing() {
    const std::int64_t size = 20;
    void* src = tf_storage_create_typed(size, DtypeTraits<T>::code);
    void* dst = tf_storage_create_typed(size, DtypeTraits<T>::code);
    check(src != nullptr && dst != nullptr, "failure-path fixtures");
    if (src == nullptr || dst == nullptr) {
        if (src != nullptr) tf_storage_destroy(src);
        if (dst != nullptr) tf_storage_destroy(dst);
        return;
    }
    std::vector<T> values = pattern_values<T>(size);
    tf_storage_copy_from(src, values.data());
    tf_storage_copy_from(dst, values.data());
    const std::int64_t dst_bytes = size * static_cast<std::int64_t>(sizeof(T));

    struct Bad {
        const char* what;
        std::int64_t ndim;
        std::int64_t shape[2];
        std::int64_t strides[2];
        std::int64_t offset;
    };
    const Bad cases[] = {
        {"negative ndim", -1, {0, 0}, {0, 0}, 0},
        {"negative offset", 2, {4, 5}, {5, 1}, -1},
        {"non-positive dimension", 2, {4, 0}, {5, 1}, 0},
        {"span exceeds the storage", 2, {4, 6}, {6, 1}, 0},
        {"offset pushes the span past the end", 2, {4, 5}, {5, 1}, 1},
        {"element count overflows int64", 2,
         {4000000000ll, 4000000000ll}, {1, 1}, 0},
        {"stride arithmetic overflows int64", 2, {4, 5},
         {4611686018427387904ll, 1}, 0},
        {"negative span reaches below zero", 1, {5}, {-4}, 0},
    };
    char message[256];
    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const Bad& bad = cases[i];
        std::vector<unsigned char> before = snapshot(dst, dst_bytes);
        tf_clear_error();
        tf_core_contiguous_copy(src, dst, bad.shape, bad.strides, bad.offset,
                                bad.ndim);
        std::snprintf(message, sizeof message, "%s [%s]", bad.what,
                      DtypeTraits<T>::name());
        expect_rejected(message, dst, dst_bytes, before, nullptr);
    }

    // An undersized destination is rejected too, and again writes nothing.
    {
        void* small = tf_storage_create_typed(4, DtypeTraits<T>::code);
        check(small != nullptr, "undersized destination fixture");
        if (small != nullptr) {
            const std::int64_t shape[2] = {4, 5};
            const std::int64_t strides[2] = {5, 1};
            std::vector<unsigned char> before =
                snapshot(small, 4 * static_cast<std::int64_t>(sizeof(T)));
            tf_clear_error();
            tf_core_contiguous_copy(src, small, shape, strides, 0, 2);
            std::snprintf(message, sizeof message, "undersized destination [%s]",
                          DtypeTraits<T>::name());
            expect_rejected(message, small,
                            4 * static_cast<std::int64_t>(sizeof(T)), before,
                            nullptr);
            tf_storage_destroy(small);
        }
    }

    // ...and the slot clears on the next success, so a rejection can never
    // be misattributed to a later call.
    {
        const std::int64_t shape[2] = {4, 5};
        const std::int64_t strides[2] = {5, 1};
        tf_core_contiguous_copy(src, dst, shape, strides, 0, 2);
        std::snprintf(message, sizeof message,
                      "a stale error survived a later success [%s]",
                      DtypeTraits<T>::name());
        check(tf_last_error_code() == TF_OK, message);
        std::vector<T> got(static_cast<std::size_t>(size));
        tf_storage_copy_to(dst, got.data());
        same_bits(values.data(), got.data(), size, "post-failure copy");
    }

    tf_storage_destroy(src);
    tf_storage_destroy(dst);
}

// The odometer's counter allocation is the one allocation the strided
// paths make, and it honors the deterministic fault-injection hook. A
// failure there must surface as TF_ERROR_ALLOC, leak nothing, and leave
// the destination untouched — at both dtypes.
template <class T>
void test_an_injected_allocation_failure_is_clean() {
    const std::int64_t size = 20;
    void* src = tf_storage_create_typed(size, DtypeTraits<T>::code);
    void* dst = tf_storage_create_typed(size, DtypeTraits<T>::code);
    check(src != nullptr && dst != nullptr, "alloc-failure fixtures");
    if (src == nullptr || dst == nullptr) {
        if (src != nullptr) tf_storage_destroy(src);
        if (dst != nullptr) tf_storage_destroy(dst);
        return;
    }
    std::vector<T> values = pattern_values<T>(size);
    tf_storage_copy_from(src, values.data());
    tf_storage_copy_from(dst, values.data());
    const std::int64_t dst_bytes = size * static_cast<std::int64_t>(sizeof(T));

    // A rank-4 non-contiguous layout, so the walk reaches the odometer and
    // therefore the counter allocation — the one allocation the strided
    // materialization path makes.
    const std::int64_t shape[4] = {2, 2, 5, 1};
    const std::int64_t strides[4] = {1, 10, 2, 1};
    std::vector<T> host(static_cast<std::size_t>(size),
                        from_bits<T>(DtypeTraits<T>::sentinel));
    const std::vector<T> host_before = host;
    char message[256];

    tf_clear_error();
    tf_test_arm_alloc_failure(1);  // the very next attempt fails
    tf_storage_materialize(src, host.data(), shape, strides, 0, 4);
    tf_test_arm_alloc_failure(0);  // disarm
    std::snprintf(message, sizeof message,
                  "an injected counter-allocation failure was not reported "
                  "as TF_ERROR_ALLOC [%s]", DtypeTraits<T>::name());
    check(tf_last_error_code() == TF_ERROR_ALLOC, message);
    // The counter is allocated before a single destination element is
    // written, so a failure there leaves the host buffer exactly as it was.
    std::snprintf(message, sizeof message,
                  "an injected allocation failure mutated its destination [%s]",
                  DtypeTraits<T>::name());
    check(std::memcmp(host_before.data(), host.data(),
                      static_cast<std::size_t>(dst_bytes)) == 0, message);
    tf_clear_error();

    // ...and the very next call succeeds, so nothing latched.
    const std::int64_t flat_shape[1] = {size};
    const std::int64_t flat_strides[1] = {1};
    tf_core_contiguous_copy(src, dst, flat_shape, flat_strides, 0, 1);
    std::snprintf(message, sizeof message,
                  "a later call failed after an injected failure [%s]",
                  DtypeTraits<T>::name());
    check(tf_last_error_code() == TF_OK, message);

    tf_storage_destroy(src);
    tf_storage_destroy(dst);
}

}  // namespace

int main() {
    test_host_round_trip_preserves_every_bit<double>();
    test_host_round_trip_preserves_every_bit<float>();
    test_a_transfer_stays_inside_its_own_buffer<double>();
    test_a_transfer_stays_inside_its_own_buffer<float>();

    test_materialization_matches_the_reference_bit_for_bit<double>();
    test_materialization_matches_the_reference_bit_for_bit<float>();

    test_identity_copy_preserves_bits_on_every_traversal<double>();
    test_identity_copy_preserves_bits_on_every_traversal<float>();
    test_mixed_dtype_identity_copy_is_rejected_before_any_write();

    test_invalid_metadata_is_rejected_and_writes_nothing<double>();
    test_invalid_metadata_is_rejected_and_writes_nothing<float>();
    test_an_injected_allocation_failure_is_clean<double>();
    test_an_injected_allocation_failure_is_clean<float>();

    if (g_failures == 0) {
        std::printf("typed transfer: all checks passed\n");
        return 0;
    }
    std::printf("typed transfer: %d check(s) failed\n", g_failures);
    return 1;
}
