// Dependency-free C++ test for the native int64 representation and the
// floating-role barriers (Phase K, milestone K1). No GoogleTest / Catch2 —
// a plain executable that prints failures and returns a nonzero exit code
// if any check fails, so CTest reports pass/fail.
//
// Three claims live here, and they are deliberately in one binary because
// the third is only meaningful given the first two:
//
//   1. **The representation.** ``TF_DTYPE_INT64 == 2``, ``tf::Dtype::Int64``
//      carrying that code, an item size of exactly 8, the canonical name,
//      a total code conversion that now accepts 2 and still rejects every
//      other unknown code without producing a dtype, and the two
//      compile-time assumptions the whole model rests on (exact width and
//      two's complement).
//
//   2. **Exact transfer.** ``tf_storage_create_typed(size, 2)`` allocates,
//      tags, zero-initializes, and destroys genuine ``std::int64_t[]``
//      storage, and the three transfer boundaries move values through it
//      **bit for bit** — including ``INT64_MIN``, ``INT64_MAX``, values
//      beyond the 32-bit range, and values beyond 2^53, which is precisely
//      where a float64 detour would start rounding. Nothing casts, widens,
//      narrows, or reinterprets: the bytes that go in are the bytes that
//      come out, and the byte pattern is checked directly rather than
//      inferred from a comparison.
//
//   3. **Unsafe use is prevented.** K1 makes int64 *allocatable* through
//      the raw C ABI while generalizing exactly one compute-family export
//      to it (``tf_core_contiguous_copy``, which is a transfer). Every
//      other handle-based export computes, and computing is a
//      floating-only capability — reading an 8-byte integer through a
//      ``double*`` would produce arithmetic on a bit pattern that means
//      nothing. So every one of them must reject an int64 operand
//      **before** it reads or writes anything.
//
//      That claim is checked as a **table**, one row per audited export,
//      covering every compute translation unit rather than a representative
//      of each: elementwise, matmul, reduction, classification, conv2d,
//      pooling, random, and the two scalar storage primitives. Each row
//      asserts the same four things — the call returns, it records
//      ``TF_ERROR_INVALID``, the message names the export and the offending
//      dtype, and **the destination is byte-for-byte unchanged** — and the
//      table's own size is asserted against a committed count, so an export
//      cannot be added to the library and silently left out of the audit.
//
//      The negative control is beside it: the same destination, the same
//      arguments, a **float64** source, and the call succeeds with the
//      error slot clear and the destination written. Without that, "every
//      row rejected" would also be satisfied by a table whose calls all
//      failed for some unrelated reason.
//
// This binary compiles the whole kernel source set directly and drives the
// **exported** C ABI, which is where the K1 contract lives.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include "tf_internal.h"  // TF_EXPORT, tf::Dtype, tf::Storage, error accessors

// -- storage lifecycle and transfer ----------------------------------------
TF_EXPORT void* tf_storage_create_typed(std::int64_t size,
                                        std::int32_t dtype_code);
TF_EXPORT void* tf_storage_create_uninitialized_typed(std::int64_t size,
                                                      std::int32_t dtype_code);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT std::int64_t tf_storage_size(const void* handle);
TF_EXPORT void tf_storage_fill(void* handle, double value);
TF_EXPORT void tf_storage_scale(void* handle, double factor);
TF_EXPORT void tf_storage_copy_from(void* handle, const void* src);
TF_EXPORT void tf_storage_copy_to(const void* handle, void* dst);
TF_EXPORT void tf_storage_materialize(
    const void* handle, void* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);

// -- the elementwise family -------------------------------------------------
TF_EXPORT void tf_core_relu(
    const void* src, void* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);
TF_EXPORT void tf_core_relu_contiguous(
    const void* src, void* dst, std::int64_t numel, std::int64_t offset);
TF_EXPORT void tf_core_sqrt(
    const void* src, void* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);
TF_EXPORT void tf_core_sqrt_contiguous(
    const void* src, void* dst, std::int64_t numel, std::int64_t offset);
TF_EXPORT void tf_core_reciprocal(
    const void* src, void* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);
TF_EXPORT void tf_core_reciprocal_contiguous(
    const void* src, void* dst, std::int64_t numel, std::int64_t offset);
TF_EXPORT void tf_core_exp(
    const void* src, void* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);
TF_EXPORT void tf_core_exp_contiguous(
    const void* src, void* dst, std::int64_t numel, std::int64_t offset);
TF_EXPORT void tf_core_log(
    const void* src, void* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);
TF_EXPORT void tf_core_log_contiguous(
    const void* src, void* dst, std::int64_t numel, std::int64_t offset);
TF_EXPORT void tf_core_contiguous_copy(
    const void* src, void* dst,
    const std::int64_t* shape, const std::int64_t* strides,
    std::int64_t offset, std::int64_t ndim);
TF_EXPORT void tf_core_add(
    const void* a, const void* b, void* dst,
    const std::int64_t* shape, const std::int64_t* a_strides,
    const std::int64_t* b_strides,
    std::int64_t a_offset, std::int64_t b_offset, std::int64_t ndim);
TF_EXPORT void tf_core_subtract(
    const void* a, const void* b, void* dst,
    const std::int64_t* shape, const std::int64_t* a_strides,
    const std::int64_t* b_strides,
    std::int64_t a_offset, std::int64_t b_offset, std::int64_t ndim);
TF_EXPORT void tf_core_multiply(
    const void* a, const void* b, void* dst,
    const std::int64_t* shape, const std::int64_t* a_strides,
    const std::int64_t* b_strides,
    std::int64_t a_offset, std::int64_t b_offset, std::int64_t ndim);
TF_EXPORT void tf_core_relu_backward(
    const void* x, const void* upstream, void* dst,
    const std::int64_t* shape, const std::int64_t* x_strides,
    const std::int64_t* u_strides,
    std::int64_t x_offset, std::int64_t u_offset, std::int64_t ndim);
TF_EXPORT void tf_core_add_contiguous(
    const void* a, const void* b, void* dst,
    std::int64_t numel, std::int64_t a_offset, std::int64_t b_offset);
TF_EXPORT void tf_core_subtract_contiguous(
    const void* a, const void* b, void* dst,
    std::int64_t numel, std::int64_t a_offset, std::int64_t b_offset);
TF_EXPORT void tf_core_multiply_contiguous(
    const void* a, const void* b, void* dst,
    std::int64_t numel, std::int64_t a_offset, std::int64_t b_offset);

// -- matmul, reduction ------------------------------------------------------
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
TF_EXPORT void tf_core_narrow_backward(
    const void* upstream_handle, void* dst_handle,
    const std::int64_t* shape, const std::int64_t* u_strides,
    const std::int64_t* out_strides,
    std::int64_t u_offset, std::int64_t out_offset, std::int64_t ndim);

// -- classification ---------------------------------------------------------
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

// -- conv2d, pooling, random ------------------------------------------------
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
TF_EXPORT void tf_core_conv2d_input_backward(
    const void* grad_output_handle, std::int64_t grad_output_offset,
    const void* weight_handle, std::int64_t weight_offset,
    void* grad_input_handle,
    std::int64_t batch, std::int64_t in_channels,
    std::int64_t input_height, std::int64_t input_width,
    std::int64_t out_channels,
    std::int64_t kernel_height, std::int64_t kernel_width,
    std::int64_t stride_height, std::int64_t stride_width,
    std::int64_t pad_height, std::int64_t pad_width,
    std::int64_t output_height, std::int64_t output_width);
TF_EXPORT void tf_core_conv2d_weight_backward(
    const void* grad_output_handle, std::int64_t grad_output_offset,
    const void* input_handle, std::int64_t input_offset,
    void* grad_weight_handle,
    std::int64_t batch, std::int64_t in_channels,
    std::int64_t input_height, std::int64_t input_width,
    std::int64_t out_channels,
    std::int64_t kernel_height, std::int64_t kernel_width,
    std::int64_t stride_height, std::int64_t stride_width,
    std::int64_t pad_height, std::int64_t pad_width,
    std::int64_t output_height, std::int64_t output_width);
TF_EXPORT void tf_core_maxpool2d_forward(
    const void* input_handle, std::int64_t input_offset,
    void* output_handle, void* winners_handle,
    std::int64_t batch, std::int64_t channels,
    std::int64_t input_height, std::int64_t input_width,
    std::int64_t kernel_height, std::int64_t kernel_width,
    std::int64_t stride_height, std::int64_t stride_width,
    std::int64_t pad_height, std::int64_t pad_width,
    std::int64_t output_height, std::int64_t output_width);
TF_EXPORT void tf_core_maxpool2d_backward(
    const void* grad_output_handle, std::int64_t grad_output_offset,
    const void* winners_handle, std::int64_t winners_offset,
    void* grad_input_handle,
    std::int64_t batch, std::int64_t channels,
    std::int64_t input_height, std::int64_t input_width,
    std::int64_t output_height, std::int64_t output_width);
TF_EXPORT void tf_core_dropout_forward(
    const void* input_handle, std::int64_t input_offset,
    void* output_handle, void* mask_handle,
    std::int64_t count, std::uint64_t seed, std::uint64_t call_index,
    double p);

// -- error accessors --------------------------------------------------------
TF_EXPORT int tf_last_error_code();
TF_EXPORT const char* tf_last_error_message();
TF_EXPORT void tf_clear_error();

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

void check_named(bool condition, const char* what, const char* who) {
    if (!condition) {
        std::printf("FAIL: %s (%s)\n", what, who);
        ++g_failures;
    }
}

// Raw byte view of a storage's buffer. The test needs to inspect bytes,
// which the production typed accessors deliberately cannot do generically;
// this exists only to prove what the bytes are.
const unsigned char* raw_bytes(const void* handle) {
    return static_cast<const unsigned char*>(tf::as_storage(handle)->data);
}

tf::Dtype tag_of(const void* handle) { return tf::as_storage(handle)->dtype; }

std::vector<unsigned char> snapshot(const void* handle) {
    const std::size_t bytes =
        static_cast<std::size_t>(tf_storage_size(handle))
        * tf::dtype_item_size(tag_of(handle));
    const unsigned char* data = raw_bytes(handle);
    return std::vector<unsigned char>(data, data + bytes);
}

bool unchanged(const void* handle, const std::vector<unsigned char>& before) {
    return snapshot(handle) == before;
}

// ===========================================================================
// 1. The representation
// ===========================================================================

void test_the_int64_abi_code_is_two_and_carries_through() {
    check(TF_DTYPE_INT64 == 2, "TF_DTYPE_INT64 is not 2");
    check(static_cast<std::int32_t>(tf::Dtype::Int64) == TF_DTYPE_INT64,
          "tf::Dtype::Int64 does not carry the ABI code");
    // The two floating codes are untouched — a new enumerator must never
    // renumber an existing one.
    check(TF_DTYPE_FLOAT64 == 0, "TF_DTYPE_FLOAT64 moved");
    check(TF_DTYPE_FLOAT32 == 1, "TF_DTYPE_FLOAT32 moved");
    static_assert(sizeof(tf::Dtype) == sizeof(std::int32_t),
                  "tf::Dtype is not int32-wide");
}

void test_the_int64_item_size_and_name() {
    check(tf::dtype_item_size(tf::Dtype::Int64) == 8,
          "int64 item size is not 8");
    check(tf::dtype_item_size(tf::Dtype::Int64) == sizeof(std::int64_t),
          "the int64 item size disagrees with sizeof(std::int64_t)");
    check(std::strcmp(tf::dtype_name(tf::Dtype::Int64), "int64") == 0,
          "the int64 canonical name is wrong");
    // The floating widths and names are unmoved.
    check(tf::dtype_item_size(tf::Dtype::Float64) == 8, "float64 width moved");
    check(tf::dtype_item_size(tf::Dtype::Float32) == 4, "float32 width moved");
    check(std::strcmp(tf::dtype_name(tf::Dtype::Float64), "float64") == 0,
          "the float64 name moved");
    check(std::strcmp(tf::dtype_name(tf::Dtype::Float32), "float32") == 0,
          "the float32 name moved");
}

void test_the_platform_assumptions_hold_at_runtime_too() {
    // Stated in the header as static_asserts and re-checked here, so a
    // toolchain change fails a test rather than only a build.
    check(sizeof(std::int64_t) == 8, "std::int64_t is not 8 bytes");
    check(static_cast<std::int64_t>(-1) == ~static_cast<std::int64_t>(0),
          "std::int64_t is not two's complement");
    check(std::numeric_limits<std::int64_t>::min()
              == -std::numeric_limits<std::int64_t>::max() - 1,
          "std::int64_t does not have the two's-complement range");
    check(std::numeric_limits<std::int64_t>::is_signed,
          "std::int64_t is not signed");
}

void test_code_conversion_accepts_two_and_still_rejects_the_rest() {
    tf::Dtype out = tf::Dtype::Float32;
    check(tf::dtype_from_code(TF_DTYPE_INT64, out), "code 2 was not accepted");
    check(out == tf::Dtype::Int64, "code 2 did not convert to int64");
    // Every other unknown code is still rejected without writing ``out``.
    const std::int32_t unknown[] = {
        -1, -2, -1000, 3, 4, 8, 64, 1 << 20,
        std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::max(),
    };
    for (std::int32_t code : unknown) {
        tf::Dtype probe = tf::Dtype::Int64;
        check(!tf::dtype_from_code(code, probe),
              "an unknown dtype code was accepted");
        check(probe == tf::Dtype::Int64,
              "a rejected conversion wrote to its output");
    }
}

void test_the_floating_role_predicate() {
    check(tf::dtype_is_floating(tf::Dtype::Float64),
          "float64 is not classified as floating");
    check(tf::dtype_is_floating(tf::Dtype::Float32),
          "float32 is not classified as floating");
    check(!tf::dtype_is_floating(tf::Dtype::Int64),
          "int64 is classified as floating");
    static_assert(noexcept(tf::dtype_is_floating(tf::Dtype::Int64)),
                  "dtype_is_floating is not noexcept");
}

void test_checked_byte_sizing_at_int64() {
    std::size_t bytes = 0;
    check(tf::dtype_checked_bytes(10, tf::Dtype::Int64, bytes)
              && bytes == 80,
          "10 int64 elements is not 80 bytes");
    check(!tf::dtype_checked_bytes(0, tf::Dtype::Int64, bytes),
          "a zero element count was accepted");
    check(!tf::dtype_checked_bytes(-1, tf::Dtype::Int64, bytes),
          "a negative element count was accepted");
    // The overflow guard uses the int64 item size, so the threshold is the
    // same as float64's and is rejected by arithmetic, before any allocator
    // is asked.
    check(!tf::dtype_checked_bytes(std::numeric_limits<std::int64_t>::max(),
                                   tf::Dtype::Int64, bytes),
          "an unrepresentable int64 byte count was accepted");
}

// ===========================================================================
// 2. Allocation, exact transfer, and cleanup
// ===========================================================================

void test_int64_storage_is_allocated_tagged_zeroed_and_destroyed() {
    for (bool zero : {true, false}) {
        void* handle = zero
            ? tf_storage_create_typed(6, TF_DTYPE_INT64)
            : tf_storage_create_uninitialized_typed(6, TF_DTYPE_INT64);
        check(handle != nullptr, "int64 storage could not be created");
        if (handle == nullptr) {
            continue;
        }
        // The size is a **logical element count**, as at every dtype.
        check(tf_storage_size(handle) == 6,
              "int64 storage reports the wrong element count");
        check(tag_of(handle) == tf::Dtype::Int64,
              "int64 storage does not carry the int64 tag");
        if (zero) {
            // Value-initialized array new gives exact integer zeros, and an
            // integer zero has no sign bit to get wrong: every byte is 0.
            const std::vector<unsigned char> bytes = snapshot(handle);
            bool all_zero = true;
            for (unsigned char byte : bytes) {
                all_zero = all_zero && byte == 0;
            }
            check(bytes.size() == 48, "int64 storage of 6 is not 48 bytes");
            check(all_zero, "zero-initialized int64 storage is not all zeros");
        }
        tf_storage_destroy(handle);
    }
}

void test_a_non_positive_int64_size_is_rejected() {
    for (std::int64_t size : {static_cast<std::int64_t>(0),
                              static_cast<std::int64_t>(-1),
                              std::numeric_limits<std::int64_t>::min()}) {
        tf_clear_error();
        void* handle = tf_storage_create_typed(size, TF_DTYPE_INT64);
        check(handle == nullptr, "a non-positive int64 size was accepted");
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "a rejected int64 size did not record TF_ERROR_INVALID");
        tf_storage_destroy(handle);
    }
    tf_clear_error();
}

// The values that catch width, sign, and truncation errors. 2^53 + 1 is the
// smallest positive integer a float64 cannot represent, so a value above it
// surviving intact is what distinguishes an exact integer transfer from one
// that took a floating detour.
const std::int64_t kProbeValues[] = {
    0,
    1,
    -1,
    42,
    -42,
    2147483647LL,                 // INT32_MAX
    2147483648LL,                 // one past INT32_MAX
    -2147483648LL,                // INT32_MIN
    -2147483649LL,                // one below INT32_MIN
    4294967296LL,                 // 2^32
    9007199254740993LL,           // 2^53 + 1 — not representable in float64
    -9007199254740993LL,
    std::numeric_limits<std::int64_t>::max(),
    std::numeric_limits<std::int64_t>::min(),
};
const std::size_t kProbeCount =
    sizeof(kProbeValues) / sizeof(kProbeValues[0]);

void test_the_host_round_trip_is_bit_exact() {
    void* handle = tf_storage_create_typed(
        static_cast<std::int64_t>(kProbeCount), TF_DTYPE_INT64);
    check(handle != nullptr, "int64 storage could not be created");
    if (handle == nullptr) {
        return;
    }
    std::vector<std::int64_t> source(kProbeValues, kProbeValues + kProbeCount);
    tf_storage_copy_from(handle, source.data());

    // The bytes in the buffer are the source's object representation,
    // element for element — checked as raw memory rather than by comparing
    // values, so a transfer that happened to round-trip through another
    // type would still be caught.
    const unsigned char* stored = raw_bytes(handle);
    const unsigned char* expected =
        reinterpret_cast<const unsigned char*>(source.data());
    check(std::memcmp(stored, expected, kProbeCount * 8) == 0,
          "int64 host->native transfer did not reproduce the source bytes");

    std::vector<std::int64_t> back(kProbeCount, 0);
    tf_storage_copy_to(handle, back.data());
    for (std::size_t i = 0; i < kProbeCount; ++i) {
        check(back[i] == kProbeValues[i],
              "int64 native->host transfer changed a value");
    }
    check(std::memcmp(back.data(), source.data(), kProbeCount * 8) == 0,
          "the int64 round trip is not byte-for-byte exact");

    // Materialization of a contiguous rank-1 view is the same values in the
    // same order, through the third transfer boundary.
    const std::int64_t shape[] = {static_cast<std::int64_t>(kProbeCount)};
    const std::int64_t strides[] = {1};
    std::vector<std::int64_t> gathered(kProbeCount, 0);
    tf_storage_materialize(handle, gathered.data(), shape, strides, 0, 1);
    check(std::memcmp(gathered.data(), source.data(), kProbeCount * 8) == 0,
          "int64 materialization is not byte-for-byte exact");

    tf_storage_destroy(handle);
    tf_clear_error();
}

void test_a_strided_int64_view_materializes_in_logical_order() {
    // A reversed rank-1 view and a transposed rank-2 view: the odometer
    // walk is dtype-independent, and the values must arrive in logical
    // order with no value changed.
    void* handle = tf_storage_create_typed(6, TF_DTYPE_INT64);
    check(handle != nullptr, "int64 storage could not be created");
    if (handle == nullptr) {
        return;
    }
    const std::int64_t values[] = {-9007199254740993LL, 2, -3, 4,
                                   9007199254740993LL, 6};
    tf_storage_copy_from(handle, values);

    const std::int64_t reversed_shape[] = {6};
    const std::int64_t reversed_strides[] = {-1};
    std::int64_t reversed[6] = {0};
    tf_storage_materialize(handle, reversed, reversed_shape, reversed_strides,
                           5, 1);
    for (int i = 0; i < 6; ++i) {
        check(reversed[i] == values[5 - i],
              "a reversed int64 view did not materialize in logical order");
    }

    const std::int64_t transposed_shape[] = {3, 2};
    const std::int64_t transposed_strides[] = {1, 3};
    std::int64_t transposed[6] = {0};
    tf_storage_materialize(handle, transposed, transposed_shape,
                           transposed_strides, 0, 2);
    const std::int64_t expected[] = {values[0], values[3], values[1],
                                     values[4], values[2], values[5]};
    for (int i = 0; i < 6; ++i) {
        check(transposed[i] == expected[i],
              "a transposed int64 view did not materialize in logical order");
    }

    tf_storage_destroy(handle);
    tf_clear_error();
}

void test_the_identity_copy_carries_int64_exactly() {
    // ``tf_core_contiguous_copy`` is the one compute-family export K1
    // generalizes, because it is a transfer rather than arithmetic.
    void* src = tf_storage_create_typed(
        static_cast<std::int64_t>(kProbeCount), TF_DTYPE_INT64);
    void* dst = tf_storage_create_typed(
        static_cast<std::int64_t>(kProbeCount), TF_DTYPE_INT64);
    check(src != nullptr && dst != nullptr,
          "int64 storage could not be created");
    if (src == nullptr || dst == nullptr) {
        tf_storage_destroy(src);
        tf_storage_destroy(dst);
        return;
    }
    tf_storage_copy_from(src, kProbeValues);
    const std::int64_t shape[] = {static_cast<std::int64_t>(kProbeCount)};
    const std::int64_t strides[] = {1};
    tf_clear_error();
    tf_core_contiguous_copy(src, dst, shape, strides, 0, 1);
    check(tf_last_error_code() == TF_OK,
          "the int64 identity copy reported an error");
    check(std::memcmp(raw_bytes(src), raw_bytes(dst), kProbeCount * 8) == 0,
          "the int64 identity copy did not reproduce the source bytes");

    // ...and a mixed-dtype copy is still refused, in both directions, with
    // the destination untouched. The transfer generalization did not open a
    // conversion.
    void* f64 = tf_storage_create_typed(
        static_cast<std::int64_t>(kProbeCount), TF_DTYPE_FLOAT64);
    if (f64 != nullptr) {
        const std::vector<unsigned char> before = snapshot(f64);
        tf_clear_error();
        tf_core_contiguous_copy(src, f64, shape, strides, 0, 1);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "an int64 -> float64 copy was accepted");
        check(unchanged(f64, before),
              "a rejected int64 -> float64 copy wrote to its destination");
        const std::vector<unsigned char> int_before = snapshot(dst);
        tf_clear_error();
        tf_core_contiguous_copy(f64, dst, shape, strides, 0, 1);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "a float64 -> int64 copy was accepted");
        check(unchanged(dst, int_before),
              "a rejected float64 -> int64 copy wrote to its destination");
        tf_storage_destroy(f64);
    }
    tf_storage_destroy(src);
    tf_storage_destroy(dst);
    tf_clear_error();
}

void test_float32_and_float64_storage_are_unchanged() {
    // The negative control for the whole representation section: adding a
    // third element type moved neither of the two that were already there.
    for (std::int32_t code : {TF_DTYPE_FLOAT64, TF_DTYPE_FLOAT32}) {
        void* handle = tf_storage_create_typed(4, code);
        check(handle != nullptr, "floating storage could not be created");
        if (handle == nullptr) {
            continue;
        }
        check(tf_storage_size(handle) == 4,
              "floating storage reports the wrong element count");
        const std::size_t width = code == TF_DTYPE_FLOAT32 ? 4u : 8u;
        check(snapshot(handle).size() == 4 * width,
              "floating storage has the wrong byte size");
        // Zero-initialized floating storage is **positive** zero, which is
        // all-zero bytes at both widths.
        bool all_zero = true;
        for (unsigned char byte : snapshot(handle)) {
            all_zero = all_zero && byte == 0;
        }
        check(all_zero, "zero-initialized floating storage is not +0");
        tf_storage_fill(handle, 2.5);
        check(tf_last_error_code() == TF_OK,
              "filling floating storage recorded an error");
        tf_storage_scale(handle, 2.0);
        check(tf_last_error_code() == TF_OK,
              "scaling floating storage recorded an error");
        if (code == TF_DTYPE_FLOAT64) {
            const double* data =
                static_cast<const double*>(tf::as_storage(handle)->data);
            for (int i = 0; i < 4; ++i) {
                check(data[i] == 5.0, "float64 fill/scale changed behavior");
            }
        } else {
            const float* data =
                static_cast<const float*>(tf::as_storage(handle)->data);
            for (int i = 0; i < 4; ++i) {
                check(data[i] == 5.0f, "float32 fill/scale changed behavior");
            }
        }
        tf_storage_destroy(handle);
    }
    tf_clear_error();
}

// ===========================================================================
// 3. The floating-role barrier, as an auditable table
// ===========================================================================

// One row per float-only handle-based export. The count is committed so an
// export added to the library cannot silently skip the audit: K1 audits 32
// exports — every handle-based export except ``tf_core_contiguous_copy``,
// which is a transfer and is generalized instead, and the three transfer
// boundaries, which are generalized for the same reason.
const std::size_t kAuditedExportCount = 32;

struct Case {
    const char* name;
    std::function<void()> call;
};

// Shared operands for the table. ``i64`` stands in for whichever handle the
// row is probing; ``dst`` is the float64 destination every row must leave
// untouched.
struct Operands {
    void* i64;
    void* f64;
    void* dst;
    void* winners;
    const std::int64_t* shape;
    const std::int64_t* strides;
    const std::int64_t* targets;
};

std::vector<Case> build_audit(const Operands& o) {
    const std::int64_t* shape = o.shape;
    const std::int64_t* strides = o.strides;
    void* i64 = o.i64;
    void* f64 = o.f64;
    void* dst = o.dst;
    void* winners = o.winners;
    const std::int64_t* targets = o.targets;
    return {
        // -- elementwise: unary, strided and contiguous ------------------
        {"tf_core_relu", [=] {
            tf_core_relu(i64, dst, shape, strides, 0, 1); }},
        {"tf_core_relu_contiguous", [=] {
            tf_core_relu_contiguous(i64, dst, 4, 0); }},
        {"tf_core_sqrt", [=] {
            tf_core_sqrt(i64, dst, shape, strides, 0, 1); }},
        {"tf_core_sqrt_contiguous", [=] {
            tf_core_sqrt_contiguous(i64, dst, 4, 0); }},
        {"tf_core_reciprocal", [=] {
            tf_core_reciprocal(i64, dst, shape, strides, 0, 1); }},
        {"tf_core_reciprocal_contiguous", [=] {
            tf_core_reciprocal_contiguous(i64, dst, 4, 0); }},
        {"tf_core_exp", [=] {
            tf_core_exp(i64, dst, shape, strides, 0, 1); }},
        {"tf_core_exp_contiguous", [=] {
            tf_core_exp_contiguous(i64, dst, 4, 0); }},
        {"tf_core_log", [=] {
            tf_core_log(i64, dst, shape, strides, 0, 1); }},
        {"tf_core_log_contiguous", [=] {
            tf_core_log_contiguous(i64, dst, 4, 0); }},
        // -- elementwise: binary, strided and contiguous ------------------
        {"tf_core_add", [=] {
            tf_core_add(i64, f64, dst, shape, strides, strides, 0, 0, 1); }},
        {"tf_core_subtract", [=] {
            tf_core_subtract(i64, f64, dst, shape, strides, strides, 0, 0,
                             1); }},
        {"tf_core_multiply", [=] {
            tf_core_multiply(i64, f64, dst, shape, strides, strides, 0, 0,
                             1); }},
        {"tf_core_relu_backward", [=] {
            tf_core_relu_backward(i64, f64, dst, shape, strides, strides, 0,
                                  0, 1); }},
        {"tf_core_add_contiguous", [=] {
            tf_core_add_contiguous(i64, f64, dst, 4, 0, 0); }},
        {"tf_core_subtract_contiguous", [=] {
            tf_core_subtract_contiguous(i64, f64, dst, 4, 0, 0); }},
        {"tf_core_multiply_contiguous", [=] {
            tf_core_multiply_contiguous(i64, f64, dst, 4, 0, 0); }},
        // -- matmul ------------------------------------------------------
        {"tf_core_matmul", [=] {
            tf_core_matmul(i64, f64, dst, 2, 2, 2, 2, 1, 2, 1, 0, 0); }},
        // -- reduction ---------------------------------------------------
        {"tf_core_sum", [=] {
            tf_core_sum(i64, dst, shape, strides, strides, 0, 1); }},
        {"tf_core_narrow_backward", [=] {
            tf_core_narrow_backward(i64, dst, shape, strides, strides, 0, 0,
                                    1); }},
        // -- classification ----------------------------------------------
        {"tf_core_softmax_forward", [=] {
            tf_core_softmax_forward(i64, 0, dst, 1, 4, 1); }},
        {"tf_core_log_softmax_forward", [=] {
            tf_core_log_softmax_forward(i64, 0, dst, 1, 4, 1); }},
        {"tf_core_cross_entropy_forward", [=] {
            tf_core_cross_entropy_forward(i64, 0, targets, 2, dst, f64, 2, 2,
                                          0); }},
        {"tf_core_cross_entropy_backward", [=] {
            tf_core_cross_entropy_backward(i64, 0, targets, 2, f64, 0, dst, 2,
                                           2, 0); }},
        // -- conv2d ------------------------------------------------------
        {"tf_core_conv2d_forward", [=] {
            tf_core_conv2d_forward(i64, 0, f64, 0, nullptr, 0, dst,
                                   1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0, 2, 2); }},
        {"tf_core_conv2d_input_backward", [=] {
            tf_core_conv2d_input_backward(i64, 0, f64, 0, dst,
                                          1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0,
                                          2, 2); }},
        {"tf_core_conv2d_weight_backward", [=] {
            tf_core_conv2d_weight_backward(i64, 0, f64, 0, dst,
                                           1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0,
                                           2, 2); }},
        // -- pooling -----------------------------------------------------
        {"tf_core_maxpool2d_forward", [=] {
            tf_core_maxpool2d_forward(i64, 0, dst, winners,
                                      1, 1, 2, 2, 1, 1, 1, 1, 0, 0, 2, 2); }},
        {"tf_core_maxpool2d_backward", [=] {
            tf_core_maxpool2d_backward(i64, 0, winners, 0, dst,
                                       1, 1, 2, 2, 2, 2); }},
        // -- random ------------------------------------------------------
        {"tf_core_dropout_forward", [=] {
            tf_core_dropout_forward(i64, 0, dst, f64, 4, 1, 1, 0.5); }},
        // -- the two scalar storage primitives ---------------------------
        //    Probed on the *destination* handle, because that is their only
        //    handle: a double scalar is not an exact integer primitive.
        {"tf_storage_fill", [=] { tf_storage_fill(i64, 3.0); }},
        {"tf_storage_scale", [=] { tf_storage_scale(i64, 2.0); }},
    };
}

void test_every_floating_export_rejects_an_int64_operand() {
    void* i64 = tf_storage_create_typed(4, TF_DTYPE_INT64);
    void* f64 = tf_storage_create_typed(4, TF_DTYPE_FLOAT64);
    void* dst = tf_storage_create_typed(4, TF_DTYPE_FLOAT64);
    void* winners = tf_storage_create_typed(4, TF_DTYPE_FLOAT64);
    check(i64 && f64 && dst && winners, "the audit operands could not be made");
    if (!(i64 && f64 && dst && winners)) {
        tf_storage_destroy(i64);
        tf_storage_destroy(f64);
        tf_storage_destroy(dst);
        tf_storage_destroy(winners);
        return;
    }
    // A recognizable non-zero destination, so "unchanged" is observable
    // rather than merely the absence of a write into zeros.
    tf_storage_fill(dst, -7.5);
    tf_storage_fill(f64, 1.25);
    const std::int64_t shape[] = {4};
    const std::int64_t strides[] = {1};
    const std::int64_t targets[] = {0, 1};

    Operands operands{i64, f64, dst, winners, shape, strides, targets};
    const std::vector<Case> audit = build_audit(operands);
    check(audit.size() == kAuditedExportCount,
          "the audit table does not cover the committed export count");

    const std::vector<unsigned char> before = snapshot(dst);
    const std::vector<unsigned char> int_before = snapshot(i64);
    for (const Case& one : audit) {
        tf_clear_error();
        one.call();
        check_named(tf_last_error_code() == TF_ERROR_INVALID,
                    "an int64 operand was not rejected with TF_ERROR_INVALID",
                    one.name);
        const char* message = tf_last_error_message();
        check_named(message != nullptr && std::strstr(message, one.name)
                        != nullptr,
                    "the rejection message does not name the export",
                    one.name);
        check_named(message != nullptr && std::strstr(message, "int64")
                        != nullptr,
                    "the rejection message does not name the offending dtype",
                    one.name);
        check_named(message != nullptr
                        && std::strstr(message, "floating-only") != nullptr,
                    "the rejection message does not state the role rule",
                    one.name);
        check_named(unchanged(dst, before),
                    "a rejected call mutated its destination", one.name);
        check_named(unchanged(i64, int_before),
                    "a rejected call mutated the int64 operand", one.name);
    }
    tf_clear_error();
    tf_storage_destroy(i64);
    tf_storage_destroy(f64);
    tf_storage_destroy(dst);
    tf_storage_destroy(winners);
}

void test_the_audit_can_actually_pass_a_valid_floating_call() {
    // The non-vacuity control for the table above. "Every row rejected" is
    // only evidence if the same operands and the same argument shapes
    // succeed when the probed handle is floating — otherwise the table
    // would pass for a repository whose exports rejected everything.
    void* src = tf_storage_create_typed(4, TF_DTYPE_FLOAT64);
    void* other = tf_storage_create_typed(4, TF_DTYPE_FLOAT64);
    void* dst = tf_storage_create_typed(4, TF_DTYPE_FLOAT64);
    check(src && other && dst, "the control operands could not be made");
    if (!(src && other && dst)) {
        tf_storage_destroy(src);
        tf_storage_destroy(other);
        tf_storage_destroy(dst);
        return;
    }
    tf_storage_fill(src, 3.0);
    tf_storage_fill(other, 4.0);
    tf_storage_fill(dst, -7.5);
    const std::int64_t shape[] = {4};
    const std::int64_t strides[] = {1};

    tf_clear_error();
    tf_core_relu(src, dst, shape, strides, 0, 1);
    check(tf_last_error_code() == TF_OK,
          "a valid float64 relu was rejected");
    const double* out = static_cast<const double*>(tf::as_storage(dst)->data);
    for (int i = 0; i < 4; ++i) {
        check(out[i] == 3.0, "a valid float64 relu did not write its output");
    }

    tf_clear_error();
    tf_core_add(src, other, dst, shape, strides, strides, 0, 0, 1);
    check(tf_last_error_code() == TF_OK, "a valid float64 add was rejected");
    for (int i = 0; i < 4; ++i) {
        check(out[i] == 7.0, "a valid float64 add did not write its output");
    }

    tf_clear_error();
    tf_storage_fill(dst, 2.0);
    check(tf_last_error_code() == TF_OK,
          "a valid float64 fill was rejected");
    for (int i = 0; i < 4; ++i) {
        check(out[i] == 2.0, "a valid float64 fill did not write");
    }
    tf_storage_scale(dst, 3.0);
    check(tf_last_error_code() == TF_OK,
          "a valid float64 scale was rejected");
    for (int i = 0; i < 4; ++i) {
        check(out[i] == 6.0, "a valid float64 scale did not write");
    }

    tf_storage_destroy(src);
    tf_storage_destroy(other);
    tf_storage_destroy(dst);
    tf_clear_error();
}

void test_a_guarded_export_clears_the_slot_on_entry() {
    // The error-contract property the barrier depends on: a code recorded
    // by one call can never be misread as another call's failure, because
    // every guarded export clears the slot before it does anything. The two
    // scalar primitives gained the guard at K1 precisely so this stays true
    // now that they can record a rejection.
    void* i64 = tf_storage_create_typed(4, TF_DTYPE_INT64);
    void* f64 = tf_storage_create_typed(4, TF_DTYPE_FLOAT64);
    check(i64 && f64, "the operands could not be made");
    if (!(i64 && f64)) {
        tf_storage_destroy(i64);
        tf_storage_destroy(f64);
        return;
    }
    tf_clear_error();
    tf_storage_fill(i64, 1.0);            // rejected: records TF_ERROR_INVALID
    check(tf_last_error_code() == TF_ERROR_INVALID,
          "an int64 fill did not record a rejection");
    tf_storage_fill(f64, 1.0);            // valid: must clear on entry
    check(tf_last_error_code() == TF_OK,
          "a valid fill did not clear the error slot on entry");
    const std::int64_t shape[] = {4};
    const std::int64_t strides[] = {1};
    void* dst = tf_storage_create_typed(4, TF_DTYPE_FLOAT64);
    if (dst != nullptr) {
        tf_core_relu(i64, dst, shape, strides, 0, 1);
        check(tf_last_error_code() == TF_ERROR_INVALID,
              "an int64 relu did not record a rejection");
        tf_core_relu(f64, dst, shape, strides, 0, 1);
        check(tf_last_error_code() == TF_OK,
              "a valid relu did not clear the error slot on entry");
        tf_storage_destroy(dst);
    }
    tf_storage_destroy(i64);
    tf_storage_destroy(f64);
    tf_clear_error();
}

}  // namespace

int main() {
    test_the_int64_abi_code_is_two_and_carries_through();
    test_the_int64_item_size_and_name();
    test_the_platform_assumptions_hold_at_runtime_too();
    test_code_conversion_accepts_two_and_still_rejects_the_rest();
    test_the_floating_role_predicate();
    test_checked_byte_sizing_at_int64();

    test_int64_storage_is_allocated_tagged_zeroed_and_destroyed();
    test_a_non_positive_int64_size_is_rejected();
    test_the_host_round_trip_is_bit_exact();
    test_a_strided_int64_view_materializes_in_logical_order();
    test_the_identity_copy_carries_int64_exactly();
    test_float32_and_float64_storage_are_unchanged();

    test_every_floating_export_rejects_an_int64_operand();
    test_the_audit_can_actually_pass_a_valid_floating_call();
    test_a_guarded_export_clears_the_slot_on_entry();

    if (g_failures == 0) {
        std::printf("test_dtype_int64_storage: all checks passed\n");
        return 0;
    }
    std::printf("test_dtype_int64_storage: %d check(s) failed\n", g_failures);
    return 1;
}
