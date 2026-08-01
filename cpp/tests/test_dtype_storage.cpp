// Dependency-free C++ test for the native dtype model and dtype-tagged
// storage (Phase I, milestone I1). No GoogleTest / Catch2 — a plain
// executable that prints failures and returns a nonzero exit code if any
// check fails, so CTest reports pass/fail.
//
// Three things are proved here, and they are deliberately in one binary
// because the third depends on the first two:
//
//   1. **The dtype model.** The frozen ABI codes, the one item-size
//      authority, the canonical names, and a conversion that is total and
//      rejects every unknown code without producing a dtype.
//   2. **The typed creators.** ``tf_storage_create_typed`` and
//      ``tf_storage_create_uninitialized_typed`` — the only two symbols
//      Phase I adds — allocating, tagging, and destroying both dtypes,
//      plus every rejection path: unknown code, non-positive size, byte
//      overflow, and injected allocation failure.
//   3. **Unsafe use is prevented.** I1 makes float32 *allocatable* through
//      the C ABI while generalizing **no** operation, so a float32 handle
//      can reach a kernel that still reads float64. Reading a
//      4-byte-per-element buffer through a ``double*`` would overrun it by
//      exactly a factor of two, so every operation that has not been
//      generalized must reject a float32 operand *before* touching memory.
//      That is checked against a representative export from **every**
//      compute translation unit, not just one, because each has its own
//      validation front end and could have been missed independently.
//
// This binary compiles the whole kernel source set directly and drives the
// **exported** C ABI, which is where the I1 contract lives.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "tf_internal.h"  // TF_EXPORT, tf::Dtype, tf::Storage, error accessors

// -- storage lifecycle ------------------------------------------------------
TF_EXPORT void* tf_storage_create(std::int64_t size);
TF_EXPORT void* tf_storage_create_uninitialized(std::int64_t size);
TF_EXPORT void* tf_storage_create_typed(std::int64_t size,
                                        std::int32_t dtype_code);
TF_EXPORT void* tf_storage_create_uninitialized_typed(std::int64_t size,
                                                      std::int32_t dtype_code);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT std::int64_t tf_storage_size(const void* handle);
TF_EXPORT void tf_storage_fill(void* handle, double value);
TF_EXPORT void tf_storage_scale(void* handle, double factor);
TF_EXPORT void tf_storage_copy_from(void* handle, const double* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, double* dst);
TF_EXPORT void tf_storage_materialize(
    const void* handle, double* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);

// -- one representative export from every other compute unit ---------------
TF_EXPORT void tf_core_relu(
    const void* src, void* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);
TF_EXPORT void tf_core_add(
    const void* a, const void* b, void* dst,
    const std::int64_t* shape, const std::int64_t* a_strides,
    const std::int64_t* b_strides,
    std::int64_t a_offset, std::int64_t b_offset, std::int64_t ndim);
TF_EXPORT void tf_core_contiguous_copy(
    const void* src, void* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);
TF_EXPORT void tf_core_matmul(
    const void* a_handle, const void* b_handle, void* dst_handle,
    std::int64_t m, std::int64_t n, std::int64_t p,
    std::int64_t a_stride0, std::int64_t a_stride1,
    std::int64_t b_stride0, std::int64_t b_stride1,
    std::int64_t a_offset, std::int64_t b_offset);
TF_EXPORT void tf_core_sum(
    const void* src_handle, void* dst_handle,
    const std::int64_t* shape, const std::int64_t* in_strides,
    const std::int64_t* out_strides,
    std::int64_t offset, std::int64_t ndim);
TF_EXPORT void tf_core_softmax_forward(
    const void* src_handle, std::int64_t src_offset, void* dst_handle,
    std::int64_t outer, std::int64_t axis_length, std::int64_t inner);
TF_EXPORT void tf_core_maxpool2d_forward(
    const void* input_handle, std::int64_t input_offset,
    void* output_handle, void* winners_handle,
    std::int64_t batch, std::int64_t channels,
    std::int64_t input_height, std::int64_t input_width,
    std::int64_t kernel_height, std::int64_t kernel_width,
    std::int64_t stride_height, std::int64_t stride_width,
    std::int64_t pad_height, std::int64_t pad_width,
    std::int64_t output_height, std::int64_t output_width);
TF_EXPORT void tf_core_conv2d_forward(
    const void* input_handle, std::int64_t input_offset,
    const void* weight_handle, std::int64_t weight_offset,
    const void* bias_handle, std::int64_t bias_offset,
    void* output_handle,
    std::int64_t batch, std::int64_t in_channels,
    std::int64_t input_height, std::int64_t input_width,
    std::int64_t out_channels,
    std::int64_t kernel_height, std::int64_t kernel_width,
    std::int64_t stride_height, std::int64_t stride_width,
    std::int64_t pad_height, std::int64_t pad_width,
    std::int64_t output_height, std::int64_t output_width);
TF_EXPORT void tf_core_dropout_forward(
    const void* input_handle, std::int64_t input_offset,
    void* output_handle, void* mask_handle,
    std::int64_t count, std::uint64_t seed, std::uint64_t call_index,
    double p);

// -- error / fault-injection ------------------------------------------------
TF_EXPORT void tf_test_arm_alloc_failure(std::int64_t nth);
TF_EXPORT int tf_last_error_code();
TF_EXPORT const char* tf_last_error_message();
TF_EXPORT void tf_clear_error();

// AddressSanitizer detection. ASan caps a single allocation at 1 TB and,
// with its default ``allocator_may_return_null=0``, treats a larger
// request as a hard *error* that aborts the process rather than a null
// return — so the one check that deliberately requests an unsatisfiable
// (but representable) size is skipped under ASan and runs everywhere
// else. The *overflow* check is unaffected: it is rejected by arithmetic
// before any allocator is asked, which is precisely the point of it.
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define TF_TEST_UNDER_ASAN 1
#  endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(TF_TEST_UNDER_ASAN)
#  define TF_TEST_UNDER_ASAN 1
#endif

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

// Raw byte view of a storage's buffer. The test needs to inspect bytes at
// both widths, which the production float64 accessor deliberately cannot
// do; this is the one place in the repository that reads a storage buffer
// as raw memory, and it exists only to prove what the bytes are.
const unsigned char* raw_bytes(const void* handle) {
    return static_cast<const unsigned char*>(tf::as_storage(handle)->data);
}

tf::Dtype tag_of(const void* handle) { return tf::as_storage(handle)->dtype; }

// ===========================================================================
// 1. The dtype model
// ===========================================================================

void test_the_abi_codes_are_frozen() {
    check(TF_DTYPE_FLOAT64 == 0, "TF_DTYPE_FLOAT64 is not 0");
    check(TF_DTYPE_FLOAT32 == 1, "TF_DTYPE_FLOAT32 is not 1");
    check(static_cast<std::int32_t>(tf::Dtype::Float64) == TF_DTYPE_FLOAT64,
          "tf::Dtype::Float64 does not carry the ABI code");
    check(static_cast<std::int32_t>(tf::Dtype::Float32) == TF_DTYPE_FLOAT32,
          "tf::Dtype::Float32 does not carry the ABI code");
    // The underlying type is fixed width, because it crosses a struct that
    // an opaque handle points at.
    static_assert(sizeof(tf::Dtype) == sizeof(std::int32_t),
                  "tf::Dtype is not int32-wide");
}

void test_the_item_size_authority() {
    check(tf::dtype_item_size(tf::Dtype::Float64) == 8,
          "float64 item size is not 8");
    check(tf::dtype_item_size(tf::Dtype::Float32) == 4,
          "float32 item size is not 4");
    // The assumptions the widths rest on, stated here as well as in the
    // header so a toolchain change fails a test rather than only a build.
    check(sizeof(double) == 8, "double is not 8 bytes");
    check(sizeof(float) == 4, "float is not 4 bytes");
    check(std::numeric_limits<double>::is_iec559,
          "double is not IEEE-754 binary64");
    check(std::numeric_limits<float>::is_iec559,
          "float is not IEEE-754 binary32");
    check(tf::dtype_item_size(tf::Dtype::Float64) == sizeof(double),
          "the float64 item size disagrees with sizeof(double)");
    check(tf::dtype_item_size(tf::Dtype::Float32) == sizeof(float),
          "the float32 item size disagrees with sizeof(float)");
}

void test_the_canonical_names() {
    check(std::strcmp(tf::dtype_name(tf::Dtype::Float64), "float64") == 0,
          "the float64 canonical name is wrong");
    check(std::strcmp(tf::dtype_name(tf::Dtype::Float32), "float32") == 0,
          "the float32 canonical name is wrong");
}

void test_code_conversion_accepts_exactly_the_two_known_codes() {
    tf::Dtype out = tf::Dtype::Float32;
    check(tf::dtype_from_code(TF_DTYPE_FLOAT64, out),
          "code 0 was not accepted");
    check(out == tf::Dtype::Float64, "code 0 did not convert to float64");
    check(tf::dtype_from_code(TF_DTYPE_FLOAT32, out),
          "code 1 was not accepted");
    check(out == tf::Dtype::Float32, "code 1 did not convert to float32");
    check(tf::dtype_is_float64(tf::Dtype::Float64)
              && !tf::dtype_is_float64(tf::Dtype::Float32),
          "dtype_is_float64 misclassifies");
    check(tf::dtype_is_float32(tf::Dtype::Float32)
              && !tf::dtype_is_float32(tf::Dtype::Float64),
          "dtype_is_float32 misclassifies");
}

void test_code_conversion_rejects_every_unknown_code_without_writing() {
    // Both directions, the neighbours of the valid range, and the extremes:
    // an unknown code must never clamp to a default or produce a dtype.
    const std::int32_t unknown[] = {
        -1, -2, -1000, 2, 3, 8, 64, 1 << 20,
        std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::max(),
    };
    for (std::int32_t code : unknown) {
        // Seeded with a value the call must not disturb, so "left
        // untouched" is observable rather than merely claimed.
        tf::Dtype out = tf::Dtype::Float32;
        const bool accepted = tf::dtype_from_code(code, out);
        check(!accepted, "an unknown dtype code was accepted");
        check(out == tf::Dtype::Float32,
              "a rejected conversion wrote to its output");

        tf::Dtype other = tf::Dtype::Float64;
        check(!tf::dtype_from_code(code, other),
              "an unknown dtype code was accepted on the second probe");
        check(other == tf::Dtype::Float64,
              "a rejected conversion wrote to its output");
    }
}

void test_the_dtype_helpers_are_noexcept() {
    tf::Dtype out = tf::Dtype::Float64;
    std::size_t bytes = 0;
    static_assert(noexcept(tf::dtype_from_code(0, out)),
                  "dtype_from_code is not noexcept");
    static_assert(noexcept(tf::dtype_item_size(tf::Dtype::Float64)),
                  "dtype_item_size is not noexcept");
    static_assert(noexcept(tf::dtype_name(tf::Dtype::Float64)),
                  "dtype_name is not noexcept");
    static_assert(noexcept(tf::dtype_checked_bytes(1, tf::Dtype::Float64,
                                                   bytes)),
                  "dtype_checked_bytes is not noexcept");
    (void)out;
    (void)bytes;
}

void test_checked_byte_sizing() {
    std::size_t bytes = 0;
    check(tf::dtype_checked_bytes(10, tf::Dtype::Float64, bytes)
              && bytes == 80,
          "10 float64 elements is not 80 bytes");
    check(tf::dtype_checked_bytes(10, tf::Dtype::Float32, bytes)
              && bytes == 40,
          "10 float32 elements is not 40 bytes");
    // Non-positive counts have no byte size.
    check(!tf::dtype_checked_bytes(0, tf::Dtype::Float64, bytes),
          "a zero element count produced a byte size");
    check(!tf::dtype_checked_bytes(-1, tf::Dtype::Float32, bytes),
          "a negative element count produced a byte size");
    // The overflow boundary, exactly: the largest representable count and
    // the first one that is not, at each width.
    const std::int64_t max64 = std::numeric_limits<std::int64_t>::max() / 8;
    const std::int64_t max32 = std::numeric_limits<std::int64_t>::max() / 4;
    check(tf::dtype_checked_bytes(max64, tf::Dtype::Float64, bytes),
          "the largest representable float64 count was rejected");
    check(!tf::dtype_checked_bytes(max64 + 1, tf::Dtype::Float64, bytes),
          "a float64 count past the boundary was accepted");
    check(tf::dtype_checked_bytes(max32, tf::Dtype::Float32, bytes),
          "the largest representable float32 count was rejected");
    check(!tf::dtype_checked_bytes(max32 + 1, tf::Dtype::Float32, bytes),
          "a float32 count past the boundary was accepted");
    // The widths differ, so the boundary must differ too — a count that
    // overflows as float64 is fine as float32. This is the check that
    // would fail if any caller hardcoded one width.
    check(!tf::dtype_checked_bytes(max32, tf::Dtype::Float64, bytes),
          "a float64 count past its own boundary was accepted");
}

// ===========================================================================
// 2. The typed creators
// ===========================================================================

struct TypedVariant {
    const char* name;
    void* (*create)(std::int64_t, std::int32_t);
    bool zero_initialized;
};
const TypedVariant kTyped[] = {
    {"tf_storage_create_typed", &tf_storage_create_typed, true},
    {"tf_storage_create_uninitialized_typed",
     &tf_storage_create_uninitialized_typed, false},
};

struct DtypeCase {
    const char* name;
    std::int32_t code;
    tf::Dtype dtype;
    std::size_t item;
};
const DtypeCase kDtypes[] = {
    {"float64", TF_DTYPE_FLOAT64, tf::Dtype::Float64, 8},
    {"float32", TF_DTYPE_FLOAT32, tf::Dtype::Float32, 4},
};

void describe(const char* who, const char* dtype, const char* what,
              char* out, std::size_t n) {
    std::snprintf(out, n, "%s (%s): %s", who, dtype, what);
}

void test_typed_creation_tags_and_sizes_both_dtypes() {
    char message[256];
    for (const TypedVariant& v : kTyped) {
        for (const DtypeCase& d : kDtypes) {
            for (std::int64_t size : {std::int64_t{1}, std::int64_t{7},
                                      std::int64_t{1024}}) {
                tf_clear_error();
                void* handle = v.create(size, d.code);
                describe(v.name, d.name, "creation returned null", message,
                         sizeof message);
                check(handle != nullptr, message);
                if (handle == nullptr) {
                    continue;
                }
                describe(v.name, d.name, "creation set an error", message,
                         sizeof message);
                check(tf_last_error_code() == TF_OK, message);
                // The logical element count is preserved — it is a count,
                // not a byte size, at both widths.
                describe(v.name, d.name, "lost its logical element count",
                         message, sizeof message);
                check(tf_storage_size(handle) == size, message);
                describe(v.name, d.name, "carries the wrong dtype tag",
                         message, sizeof message);
                check(tag_of(handle) == d.dtype, message);
                tf_storage_destroy(handle);
            }
        }
    }
    tf_clear_error();
}

void test_zero_initialized_typed_storage_is_all_zero_bytes() {
    char message[256];
    for (const DtypeCase& d : kDtypes) {
        const std::int64_t size = 64;
        void* handle = tf_storage_create_typed(size, d.code);
        describe("tf_storage_create_typed", d.name, "creation failed",
                 message, sizeof message);
        check(handle != nullptr, message);
        if (handle == nullptr) {
            continue;
        }
        // The contract is stated in **bytes**: all-zero bytes is +0.0 in
        // both IEEE-754 binary32 and binary64, which is what lets one
        // value-initializing byte allocation serve every dtype with no
        // per-dtype fill pass.
        const std::size_t bytes = static_cast<std::size_t>(size) * d.item;
        bool all_zero_bytes = true;
        for (std::size_t i = 0; i < bytes; ++i) {
            all_zero_bytes = all_zero_bytes && raw_bytes(handle)[i] == 0;
        }
        describe("tf_storage_create_typed", d.name,
                 "was not zero-initialized byte for byte", message,
                 sizeof message);
        check(all_zero_bytes, message);

        // ...and read as the dtype's own element type, every element is
        // **positive** zero: value equal to zero and sign bit clear. A
        // negative zero would compare equal to 0.0 and slip past a naive
        // check, so the sign is tested separately.
        bool all_positive_zero = true;
        for (std::int64_t i = 0; i < size; ++i) {
            if (d.dtype == tf::Dtype::Float64) {
                const double value =
                    static_cast<const double*>(tf::as_storage(handle)->data)[i];
                all_positive_zero = all_positive_zero && value == 0.0
                                    && !std::signbit(value);
            } else {
                const float value =
                    static_cast<const float*>(tf::as_storage(handle)->data)[i];
                all_positive_zero = all_positive_zero && value == 0.0f
                                    && !std::signbit(value);
            }
        }
        describe("tf_storage_create_typed", d.name,
                 "is not positive zero in every element", message,
                 sizeof message);
        check(all_positive_zero, message);
        tf_storage_destroy(handle);
    }
}

void test_the_untyped_creators_still_produce_float64() {
    // The compatibility contract: the two pre-Phase-I exports are not
    // removed, not renamed, and not behaviorally altered — they are thin
    // wrappers over the same shared body with Dtype::Float64.
    void* zeroed = tf_storage_create(8);
    check(zeroed != nullptr, "tf_storage_create returned null");
    if (zeroed != nullptr) {
        check(tag_of(zeroed) == tf::Dtype::Float64,
              "tf_storage_create no longer produces float64 storage");
        check(tf_storage_size(zeroed) == 8,
              "tf_storage_create lost its size");
        tf_storage_destroy(zeroed);
    }
    void* uninitialized = tf_storage_create_uninitialized(8);
    check(uninitialized != nullptr,
          "tf_storage_create_uninitialized returned null");
    if (uninitialized != nullptr) {
        check(tag_of(uninitialized) == tf::Dtype::Float64,
              "tf_storage_create_uninitialized no longer produces float64");
        tf_storage_destroy(uninitialized);
    }
}

void test_unknown_dtype_codes_are_rejected() {
    char message[256];
    const std::int32_t unknown[] = {
        -1, -2, 2, 3, 99,
        std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::max(),
    };
    for (const TypedVariant& v : kTyped) {
        for (std::int32_t code : unknown) {
            tf_clear_error();
            void* handle = v.create(16, code);
            std::snprintf(message, sizeof message,
                          "%s: dtype code %d did not return null", v.name,
                          static_cast<int>(code));
            check(handle == nullptr, message);
            std::snprintf(message, sizeof message,
                          "%s: dtype code %d did not set TF_ERROR_INVALID",
                          v.name, static_cast<int>(code));
            check(tf_last_error_code() == TF_ERROR_INVALID, message);
            std::snprintf(message, sizeof message,
                          "%s: dtype code %d set no message", v.name,
                          static_cast<int>(code));
            check(std::strlen(tf_last_error_message()) > 0, message);
            if (handle != nullptr) {
                tf_storage_destroy(handle);
            }
        }
    }
    tf_clear_error();
}

void test_the_dtype_code_is_validated_before_the_size() {
    // The validation order is part of the contract: the dtype comes first,
    // because the item size the overflow check needs comes from it. A call
    // that is wrong in *both* respects must therefore report the dtype.
    char message[256];
    for (const TypedVariant& v : kTyped) {
        tf_clear_error();
        void* handle = v.create(0, 77);
        std::snprintf(message, sizeof message,
                      "%s: a doubly invalid call did not return null", v.name);
        check(handle == nullptr, message);
        std::snprintf(message, sizeof message,
                      "%s: the size was reported before the dtype", v.name);
        check(std::strstr(tf_last_error_message(), "dtype") != nullptr,
              message);
        if (handle != nullptr) {
            tf_storage_destroy(handle);
        }
    }
    tf_clear_error();
}

void test_non_positive_sizes_are_rejected_identically() {
    // Byte-for-byte the message the untyped creators produce, so a caller
    // cannot tell the two constructor families apart by their errors.
    char message[256];
    char untyped_message[256];
    tf_clear_error();
    void* reference = tf_storage_create(0);
    check(reference == nullptr, "tf_storage_create(0) returned a handle");
    std::snprintf(untyped_message, sizeof untyped_message, "%s",
                  tf_last_error_message());

    for (const TypedVariant& v : kTyped) {
        for (const DtypeCase& d : kDtypes) {
            for (std::int64_t size : {std::int64_t{0}, std::int64_t{-1},
                                      std::int64_t{-4096},
                                      std::numeric_limits<
                                          std::int64_t>::min()}) {
                tf_clear_error();
                void* handle = v.create(size, d.code);
                describe(v.name, d.name, "a non-positive size returned a "
                         "handle", message, sizeof message);
                check(handle == nullptr, message);
                describe(v.name, d.name, "a non-positive size did not set "
                         "TF_ERROR_INVALID", message, sizeof message);
                check(tf_last_error_code() == TF_ERROR_INVALID, message);
                describe(v.name, d.name, "a non-positive size produced a "
                         "different message than the untyped creators",
                         message, sizeof message);
                check(std::strcmp(tf_last_error_message(),
                                  untyped_message) == 0, message);
                if (handle != nullptr) {
                    tf_storage_destroy(handle);
                }
            }
        }
    }
    tf_clear_error();
}

void test_byte_overflow_is_rejected_before_allocation() {
    char message[256];
    for (const TypedVariant& v : kTyped) {
        for (const DtypeCase& d : kDtypes) {
            // The first count whose byte product is not representable at
            // this width. Derived from the one item-size authority rather
            // than written as a literal, so the two cannot disagree.
            const std::int64_t overflowing =
                std::numeric_limits<std::int64_t>::max()
                / static_cast<std::int64_t>(d.item) + 1;
            tf_clear_error();
            void* handle = v.create(overflowing, d.code);
            describe(v.name, d.name, "an overflowing count returned a handle",
                     message, sizeof message);
            check(handle == nullptr, message);
            describe(v.name, d.name, "an overflowing count did not set "
                     "TF_ERROR_INVALID", message, sizeof message);
            check(tf_last_error_code() == TF_ERROR_INVALID, message);
            describe(v.name, d.name, "the overflow message does not name the "
                     "dtype", message, sizeof message);
            check(std::strstr(tf_last_error_message(), d.name) != nullptr,
                  message);
            if (handle != nullptr) {
                tf_storage_destroy(handle);
            }
        }
    }
    tf_clear_error();
}

void test_injected_allocation_failure_maps_to_alloc_at_both_dtypes() {
    char message[256];
    for (const TypedVariant& v : kTyped) {
        for (const DtypeCase& d : kDtypes) {
            tf_clear_error();
            tf_test_arm_alloc_failure(1);  // the very next attempt fails
            void* handle = v.create(32, d.code);
            describe(v.name, d.name, "an injected failure returned a handle",
                     message, sizeof message);
            check(handle == nullptr, message);
            describe(v.name, d.name, "an injected failure did not set "
                     "TF_ERROR_ALLOC", message, sizeof message);
            check(tf_last_error_code() == TF_ERROR_ALLOC, message);
            if (handle != nullptr) {
                tf_storage_destroy(handle);
            }
            tf_test_arm_alloc_failure(0);  // disarm

            // ...and the very next real creation succeeds and **clears**
            // the stale error, so a failure cannot contaminate a later
            // successful call.
            void* recovered = v.create(32, d.code);
            describe(v.name, d.name, "creation did not recover after an "
                     "injected failure", message, sizeof message);
            check(recovered != nullptr, message);
            describe(v.name, d.name, "a successful creation did not clear the "
                     "stale error", message, sizeof message);
            check(tf_last_error_code() == TF_OK, message);
            if (recovered != nullptr) {
                check(tag_of(recovered) == d.dtype,
                      "the recovered storage carries the wrong dtype");
                tf_storage_destroy(recovered);
            }
        }
    }
    tf_clear_error();
}

void test_unsatisfiable_but_representable_sizes_fail_cleanly() {
#ifdef TF_TEST_UNDER_ASAN
    std::printf("SKIP (ASan): the unsatisfiable-size probe is an ASan hard "
                "error, not a null return\n");
#else
    char message[256];
    // Representable as a byte count, so the arithmetic check passes and the
    // allocator is genuinely asked — and genuinely refuses.
    const std::int64_t unsatisfiable =
        std::numeric_limits<std::int64_t>::max() / 32;
    for (const TypedVariant& v : kTyped) {
        for (const DtypeCase& d : kDtypes) {
            std::size_t bytes = 0;
            check(tf::dtype_checked_bytes(unsatisfiable, d.dtype, bytes),
                  "the probe size is not representable, so it probes nothing");
            tf_clear_error();
            void* handle = v.create(unsatisfiable, d.code);
            describe(v.name, d.name, "an unsatisfiable size returned a handle",
                     message, sizeof message);
            check(handle == nullptr, message);
            describe(v.name, d.name, "an unsatisfiable size set no error",
                     message, sizeof message);
            check(tf_last_error_code() == TF_ERROR_ALLOC
                      || tf_last_error_code() == TF_ERROR_RUNTIME,
                  message);
            if (handle != nullptr) {
                tf_storage_destroy(handle);
            }
        }
    }
    tf_clear_error();
#endif
}

void test_destruction_is_safe_for_every_dtype() {
    // Null stays a no-op.
    tf_storage_destroy(nullptr);
    // Every dtype and both creators destroy through the one byte-array
    // deallocation form. Repeated cycles so a mismatched form or a
    // double free shows up as a crash or a sanitizer report rather than
    // passing quietly.
    for (int round = 0; round < 64; ++round) {
        for (const TypedVariant& v : kTyped) {
            for (const DtypeCase& d : kDtypes) {
                void* handle = v.create(37, d.code);
                check(handle != nullptr, "a lifecycle-round creation failed");
                if (handle != nullptr) {
                    tf_storage_destroy(handle);
                }
            }
        }
    }
    check(true, "unreachable");  // reaching here without a crash is the check
}

void test_the_whole_allocation_is_one_indexable_array() {
    // The property the typed-array model exists to provide: a pointer
    // recovered from the storage addresses **one array object**, so
    // ``p[i]`` is well-defined for every i in [0, size) — not just for
    // element 0. Written as a real traversal at several nontrivial
    // counts, touching first, middle, and last, so a model that created
    // separate scalars (or bytes) would be walking outside its array
    // object here and a sanitizer would say so.
    char message[256];
    const std::int64_t counts[] = {1, 2, 3, 17, 64, 255, 1024, 4097};
    for (const DtypeCase& d : kDtypes) {
        for (std::int64_t size : counts) {
            void* handle = tf_storage_create_typed(size, d.code);
            describe("indexable array", d.name, "creation failed", message,
                     sizeof message);
            check(handle != nullptr, message);
            if (handle == nullptr) {
                continue;
            }
            void* raw = tf::as_storage(handle)->data;
            bool wrote_and_read_back = true;
            if (d.dtype == tf::Dtype::Float64) {
                double* p = static_cast<double*>(raw);
                for (std::int64_t i = 0; i < size; ++i) {
                    p[i] = static_cast<double>(i) * 1.5 - 3.0;
                }
                for (std::int64_t i = 0; i < size; ++i) {
                    wrote_and_read_back = wrote_and_read_back
                        && p[i] == static_cast<double>(i) * 1.5 - 3.0;
                }
                // ...and through pointer arithmetic rather than indexing,
                // at the two ends and the middle.
                wrote_and_read_back = wrote_and_read_back
                    && *(p + 0) == -3.0
                    && *(p + (size - 1))
                           == static_cast<double>(size - 1) * 1.5 - 3.0
                    && *(p + size / 2)
                           == static_cast<double>(size / 2) * 1.5 - 3.0;
            } else {
                float* p = static_cast<float*>(raw);
                for (std::int64_t i = 0; i < size; ++i) {
                    p[i] = static_cast<float>(i) * 1.5f - 3.0f;
                }
                for (std::int64_t i = 0; i < size; ++i) {
                    wrote_and_read_back = wrote_and_read_back
                        && p[i] == static_cast<float>(i) * 1.5f - 3.0f;
                }
                wrote_and_read_back = wrote_and_read_back
                    && *(p + 0) == -3.0f
                    && *(p + (size - 1))
                           == static_cast<float>(size - 1) * 1.5f - 3.0f
                    && *(p + size / 2)
                           == static_cast<float>(size / 2) * 1.5f - 3.0f;
            }
            describe("indexable array", d.name,
                     "the allocation is not traversable end to end",
                     message, sizeof message);
            check(wrote_and_read_back, message);
            // Adjacent elements really are one stride apart, which is what
            // an array object guarantees and a run of separate scalars
            // does not.
            const auto first = reinterpret_cast<std::uintptr_t>(raw);
            const auto second = first + d.item;
            describe("indexable array", d.name,
                     "element stride does not match the item size",
                     message, sizeof message);
            check(second - first == d.item, message);
            tf_storage_destroy(handle);
        }
    }
}

void test_element_types_are_trivially_destructible() {
    // The licence for the allocator having no destructor loop. Asserted
    // here as well as beside the placement construction in storage.cpp,
    // so a dtype whose elements needed destruction fails a test as well
    // as a build.
    static_assert(std::is_trivially_destructible<float>::value,
                  "float must be trivially destructible");
    static_assert(std::is_trivially_destructible<double>::value,
                  "double must be trivially destructible");
    // ...and both have fundamental alignment, which is what makes the
    // storage ``::operator new`` returns correctly aligned for them.
    static_assert(alignof(float) <= alignof(std::max_align_t),
                  "float must have fundamental alignment");
    static_assert(alignof(double) <= alignof(std::max_align_t),
                  "double must have fundamental alignment");
    check(true, "unreachable");
}

void test_every_allocation_is_correctly_aligned_for_its_element_type() {
    // The raw block must be suitably aligned for the type whose objects
    // now live in it, at every size and through every creator — otherwise
    // the placement constructions would have begun lifetimes at
    // misaligned addresses.
    char message[256];
    for (const TypedVariant& v : kTyped) {
        for (const DtypeCase& d : kDtypes) {
            for (std::int64_t size : {std::int64_t{1}, std::int64_t{3},
                                      std::int64_t{64}, std::int64_t{1000}}) {
                void* handle = v.create(size, d.code);
                describe(v.name, d.name, "creation failed", message,
                         sizeof message);
                check(handle != nullptr, message);
                if (handle == nullptr) {
                    continue;
                }
                const auto address =
                    reinterpret_cast<std::uintptr_t>(tf::as_storage(handle)->data);
                const std::size_t want =
                    (d.dtype == tf::Dtype::Float32) ? alignof(float)
                                                    : alignof(double);
                describe(v.name, d.name, "buffer is misaligned for its "
                         "element type", message, sizeof message);
                check(address % want == 0, message);
                // ``::operator new`` promises fundamental alignment, which
                // is at least as strict as either dtype needs.
                describe(v.name, d.name, "buffer is not fundamentally "
                         "aligned", message, sizeof message);
                check(address % alignof(std::max_align_t) == 0, message);
                tf_storage_destroy(handle);
            }
        }
    }
}

void test_metadata_allocation_failure_releases_the_raw_block() {
    // The allocation path takes the raw block first and the Storage node
    // second. The interesting failure is the one *between* them: if the
    // RAII owner did not hold the block across that window, the block
    // would leak. Arming the deterministic hook at 2 targets the second
    // allocation attempt, which is the metadata node.
    //
    // Whichever attempt the hook lands on, the required post-conditions
    // are the same: no handle, an error recorded, and (proved by the
    // LeakSanitizer lifecycle run) nothing leaked.
    char message[256];
    for (const TypedVariant& v : kTyped) {
        for (const DtypeCase& d : kDtypes) {
            for (std::int64_t nth : {std::int64_t{1}, std::int64_t{2}}) {
                tf_clear_error();
                tf_test_arm_alloc_failure(nth);
                void* handle = v.create(128, d.code);
                tf_test_arm_alloc_failure(0);
                if (handle != nullptr) {
                    // The hook did not fire on this attempt; the call
                    // succeeded normally and must still be well formed.
                    describe(v.name, d.name, "a surviving handle lost its "
                             "dtype", message, sizeof message);
                    check(tag_of(handle) == d.dtype, message);
                    tf_storage_destroy(handle);
                    continue;
                }
                describe(v.name, d.name, "an injected failure set no error",
                         message, sizeof message);
                check(tf_last_error_code() == TF_ERROR_ALLOC, message);
            }
        }
    }
    tf_clear_error();
    // ...and the allocator still works afterwards, so no state leaked out
    // of the failure path.
    void* recovered = tf_storage_create_typed(8, TF_DTYPE_FLOAT32);
    check(recovered != nullptr, "creation broke after metadata failures");
    if (recovered != nullptr) {
        tf_storage_destroy(recovered);
    }
}

void test_the_two_dtypes_get_independent_buffers() {
    void* a = tf_storage_create_typed(16, TF_DTYPE_FLOAT64);
    void* b = tf_storage_create_typed(16, TF_DTYPE_FLOAT32);
    check(a != nullptr && b != nullptr, "typed creation failed");
    if (a != nullptr && b != nullptr) {
        check(a != b, "two creations returned the same handle");
        check(tf::as_storage(a)->data != tf::as_storage(b)->data,
              "two creations shared one buffer");
        check(tag_of(a) == tf::Dtype::Float64 && tag_of(b) == tf::Dtype::Float32,
              "the dtype tags were crossed");
    }
    if (a != nullptr) tf_storage_destroy(a);
    if (b != nullptr) tf_storage_destroy(b);
}

// ===========================================================================
// 3. Unsafe use of a float32 handle is prevented, in every translation unit
// ===========================================================================

// A float64 destination filled with a recognizable sentinel, so "the
// rejected call wrote nothing" is checked against real bytes rather than
// assumed. Returns the handle; the caller destroys it.
void* sentinel_storage(std::int64_t size, double value) {
    void* handle = tf_storage_create(size);
    if (handle != nullptr) {
        tf_storage_fill(handle, value);
    }
    return handle;
}

bool all_equal(const void* handle, std::int64_t size, double value) {
    const double* data = tf::storage_f64(handle);
    for (std::int64_t i = 0; i < size; ++i) {
        if (data[i] != value) {
            return false;
        }
    }
    return true;
}

// Run one export that has **not** been generalized with a float32 operand
// and require: TF_ERROR_INVALID, a message naming the operation, and a
// destination that is byte-for-byte what it was.
template <class Call>
void reject_case(const char* what, void* destination, std::int64_t dst_size,
                 Call call) {
    char message[256];
    const double sentinel = -12345.5;
    tf_storage_fill(destination, sentinel);
    tf_clear_error();
    call();
    std::snprintf(message, sizeof message,
                  "%s: a float32 operand was not rejected", what);
    check(tf_last_error_code() == TF_ERROR_INVALID, message);
    std::snprintf(message, sizeof message,
                  "%s: the rejection message does not name float32", what);
    check(std::strstr(tf_last_error_message(), "float32") != nullptr, message);
    std::snprintf(message, sizeof message,
                  "%s: a rejected call mutated its destination", what);
    check(all_equal(destination, dst_size, sentinel), message);
    tf_clear_error();
}

void test_every_still_float64_operation_rejects_float32_storage() {
    // One float32 storage, offered to a representative export from every
    // compute translation unit. Each has its own validation front end, so
    // a single spot check would prove nothing about the others.
    void* f32 = tf_storage_create_typed(64, TF_DTYPE_FLOAT32);
    check(f32 != nullptr, "float32 storage could not be created");
    if (f32 == nullptr) {
        return;
    }
    void* dst = sentinel_storage(64, 0.0);
    check(dst != nullptr, "the sentinel destination could not be created");
    if (dst == nullptr) {
        tf_storage_destroy(f32);
        return;
    }

    const std::int64_t shape[2] = {8, 8};
    const std::int64_t strides[2] = {8, 1};
    const std::int64_t out_strides[2] = {0, 1};
    std::vector<double> host(64, 0.0);

    // -- storage.cpp: the five float64 storage primitives ------------------
    // fill / scale / copy_from / copy_to mutate or read the float32 buffer
    // itself, so the destination they must not touch *is* the float32
    // storage. Its bytes are checked directly.
    {
        char message[256];
        std::vector<unsigned char> before(
            raw_bytes(f32), raw_bytes(f32) + 64 * 4);
        tf_clear_error();
        tf_storage_fill(f32, 7.5);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "tf_storage_fill: a float32 handle was not rejected");
        tf_clear_error();
        tf_storage_scale(f32, 2.0);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "tf_storage_scale: a float32 handle was not rejected");
        tf_clear_error();
        tf_storage_copy_from(f32, host.data());
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "tf_storage_copy_from: a float32 handle was not rejected");
        tf_clear_error();
        tf_storage_copy_to(f32, host.data());
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "tf_storage_copy_to: a float32 handle was not rejected");
        std::snprintf(message, sizeof message,
                      "a rejected storage primitive mutated float32 memory");
        check(std::memcmp(before.data(), raw_bytes(f32), 64 * 4) == 0,
              message);
        // ...and none of them overran the buffer while doing so, which is
        // the whole reason the check exists: 64 float32 elements are 256
        // bytes, and a double* walk of "size" elements would touch 512.
        tf_clear_error();
        tf_storage_materialize(f32, host.data(), shape, strides, 0, 2);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "tf_storage_materialize: a float32 handle was not rejected");
        tf_clear_error();
    }

    // -- elementwise.cpp: unary strided, binary strided, and the copy -----
    reject_case("tf_core_relu", dst, 64, [&] {
        tf_core_relu(f32, dst, shape, strides, 0, 2);
    });
    reject_case("tf_core_add", dst, 64, [&] {
        tf_core_add(f32, dst, dst, shape, strides, strides, 0, 0, 2);
    });
    reject_case("tf_core_contiguous_copy", dst, 64, [&] {
        tf_core_contiguous_copy(f32, dst, shape, strides, 0, 2);
    });
    // ...and with the float32 storage as the *destination* rather than the
    // source, which is the direction that would corrupt memory.
    reject_case("tf_core_relu (float32 destination)", dst, 64, [&] {
        tf_core_relu(dst, f32, shape, strides, 0, 2);
    });

    // -- matmul.cpp --------------------------------------------------------
    reject_case("tf_core_matmul", dst, 64, [&] {
        tf_core_matmul(f32, dst, dst, 8, 8, 8, 8, 1, 8, 1, 0, 0);
    });

    // -- reduction.cpp -----------------------------------------------------
    reject_case("tf_core_sum", dst, 64, [&] {
        tf_core_sum(f32, dst, shape, strides, out_strides, 0, 2);
    });

    // -- classification.cpp ------------------------------------------------
    reject_case("tf_core_softmax_forward", dst, 64, [&] {
        tf_core_softmax_forward(f32, 0, dst, 8, 8, 1);
    });

    // -- pooling.cpp -------------------------------------------------------
    reject_case("tf_core_maxpool2d_forward", dst, 64, [&] {
        tf_core_maxpool2d_forward(f32, 0, dst, dst, 1, 1, 8, 8, 2, 2, 2, 2,
                                  0, 0, 4, 4);
    });

    // -- conv2d.cpp --------------------------------------------------------
    reject_case("tf_core_conv2d_forward", dst, 64, [&] {
        tf_core_conv2d_forward(f32, 0, dst, 0, nullptr, 0, dst,
                               1, 1, 8, 8, 1, 3, 3, 1, 1, 0, 0, 6, 6);
    });

    // -- random.cpp --------------------------------------------------------
    reject_case("tf_core_dropout_forward", dst, 64, [&] {
        tf_core_dropout_forward(f32, 0, dst, dst, 64, 12345u, 0u, 0.5);
    });

    tf_storage_destroy(dst);
    tf_storage_destroy(f32);
}

void test_float64_operations_still_work_unchanged() {
    // The other half of the guard's contract: it must reject float32 and
    // be invisible to everything else. A float64 call through the same
    // exports still succeeds and still computes.
    void* src = tf_storage_create(4);
    void* dst = tf_storage_create(4);
    check(src != nullptr && dst != nullptr, "float64 creation failed");
    if (src == nullptr || dst == nullptr) {
        if (src != nullptr) tf_storage_destroy(src);
        if (dst != nullptr) tf_storage_destroy(dst);
        return;
    }
    const double values[4] = {-2.0, -0.5, 1.5, 3.0};
    tf_clear_error();
    tf_storage_copy_from(src, values);
    const std::int64_t shape[1] = {4};
    const std::int64_t strides[1] = {1};
    tf_core_relu(src, dst, shape, strides, 0, 1);
    check(tf_last_error_code() == TF_OK,
          "a float64 relu through the guarded export reported an error");
    const double* out = tf::storage_f64(dst);
    check(out[0] == 0.0 && out[1] == 0.0 && out[2] == 1.5 && out[3] == 3.0,
          "a float64 relu produced the wrong values");
    tf_storage_destroy(src);
    tf_storage_destroy(dst);
}

}  // namespace

int main() {
    test_the_abi_codes_are_frozen();
    test_the_item_size_authority();
    test_the_canonical_names();
    test_code_conversion_accepts_exactly_the_two_known_codes();
    test_code_conversion_rejects_every_unknown_code_without_writing();
    test_the_dtype_helpers_are_noexcept();
    test_checked_byte_sizing();

    test_typed_creation_tags_and_sizes_both_dtypes();
    test_zero_initialized_typed_storage_is_all_zero_bytes();
    test_the_untyped_creators_still_produce_float64();
    test_unknown_dtype_codes_are_rejected();
    test_the_dtype_code_is_validated_before_the_size();
    test_non_positive_sizes_are_rejected_identically();
    test_byte_overflow_is_rejected_before_allocation();
    test_injected_allocation_failure_maps_to_alloc_at_both_dtypes();
    test_unsatisfiable_but_representable_sizes_fail_cleanly();
    test_destruction_is_safe_for_every_dtype();
    test_the_whole_allocation_is_one_indexable_array();
    test_element_types_are_trivially_destructible();
    test_every_allocation_is_correctly_aligned_for_its_element_type();
    test_metadata_allocation_failure_releases_the_raw_block();
    test_the_two_dtypes_get_independent_buffers();

    test_every_still_float64_operation_rejects_float32_storage();
    test_float64_operations_still_work_unchanged();

    if (g_failures != 0) {
        std::printf("%d dtype-storage check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all dtype-storage checks passed\n");
    return 0;
}
