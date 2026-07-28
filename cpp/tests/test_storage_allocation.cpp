// Dependency-free C++ test for the storage-creation contract (Phase H,
// milestone H1). No GoogleTest / Catch2 — a plain executable that prints
// failures and returns a nonzero exit code if any check fails, so CTest
// reports pass/fail.
//
// This binary compiles cpp/src/storage.cpp (and error.cpp) directly and
// drives the **exported** C ABI, which is where H1's contract lives.
// H1 added exactly one symbol, and these are the two constructors:
//
//   * tf_storage_create               — zero-initializing, the default,
//                                       unchanged by H1;
//   * tf_storage_create_uninitialized — the H1 sibling, identical in
//                                       every observable respect except
//                                       the buffer's initial contents.
//
// The point of testing both constructors side by side in one file is
// that H1's whole safety argument is that they *cannot drift apart*: the
// two exports share one file-local body, so every check below is run
// against both and the results compared, rather than each being spot
// checked on its own.
//
// What this file does NOT test: whether a given kernel writes every
// destination element. That is a per-kernel property, it needs the real
// production call sites, and it is proved from Python in
// tests/test_native_storage_allocation.py, whose poison wraps the
// private allocation helper and fills the returned storage through
// tf_storage_fill before the real operation runs its real kernel over
// it. There is deliberately **no poison-control export** for this file
// to call — the shipped library must not contain a hook that can change
// what an allocation holds — so the one thing demonstrated here is that
// the technique needs nothing but the production primitives.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "tf_internal.h"  // TF_EXPORT, tf::Storage, error accessors, TfStatus

TF_EXPORT void* tf_storage_create(std::int64_t size);
TF_EXPORT void* tf_storage_create_uninitialized(std::int64_t size);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT std::int64_t tf_storage_size(const void* handle);
TF_EXPORT void tf_storage_fill(void* handle, double value);
TF_EXPORT void tf_test_arm_alloc_failure(std::int64_t nth);
TF_EXPORT int tf_last_error_code();
TF_EXPORT const char* tf_last_error_message();
TF_EXPORT void tf_clear_error();

// AddressSanitizer detection. ASan caps a single allocation at 1 TB and,
// with its default ``allocator_may_return_null=0``, treats a larger
// request as a hard *error* that aborts the process rather than a null
// return. That is ASan behaving correctly and it is not specific to the
// Phase-H allocator — it fires on the ordinary zero-initializing
// constructor just the same — so the one check that deliberately requests
// an unsatisfiable size is skipped under ASan and runs everywhere else.
// Nothing else in this file is conditional.
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

// Both constructors, so every shared-contract check runs against each.
using Creator = void* (*)(std::int64_t);
struct Variant {
    const char* name;
    Creator create;
};
const Variant kVariants[] = {
    {"tf_storage_create", &tf_storage_create},
    {"tf_storage_create_uninitialized", &tf_storage_create_uninitialized},
};

void describe(const Variant& v, const char* what, char* out, std::size_t n) {
    std::snprintf(out, n, "%s: %s", v.name, what);
}

// -- the shared contract ----------------------------------------------------

void test_valid_creation_and_size() {
    char message[256];
    for (const Variant& v : kVariants) {
        for (std::int64_t size : {std::int64_t{1}, std::int64_t{7},
                                  std::int64_t{1024}}) {
            void* handle = v.create(size);
            describe(v, "valid size returns a handle", message, sizeof message);
            check(handle != nullptr, message);
            if (handle == nullptr) {
                continue;
            }
            describe(v, "reports the requested size", message, sizeof message);
            check(tf_storage_size(handle) == size, message);
            describe(v, "leaves no error set", message, sizeof message);
            check(tf_last_error_code() == TF_OK, message);
            // Writable through the ordinary storage primitive, which is
            // the only way either buffer is ever meant to be filled.
            tf_storage_fill(handle, 3.5);
            const tf::Storage* storage = tf::as_storage(handle);
            bool filled = true;
            for (std::int64_t i = 0; i < size; ++i) {
                filled = filled && storage->data[i] == 3.5;
            }
            describe(v, "is writable through tf_storage_fill", message,
                     sizeof message);
            check(filled, message);
            tf_storage_destroy(handle);
        }
    }
}

void test_zero_and_negative_sizes_are_rejected() {
    char message[256];
    for (const Variant& v : kVariants) {
        for (std::int64_t size : {std::int64_t{0}, std::int64_t{-1},
                                  std::int64_t{-4096},
                                  std::numeric_limits<std::int64_t>::min()}) {
            tf_clear_error();
            void* handle = v.create(size);
            describe(v, "non-positive size returns null", message,
                     sizeof message);
            check(handle == nullptr, message);
            describe(v, "non-positive size sets TF_ERROR_INVALID", message,
                     sizeof message);
            check(tf_last_error_code() == TF_ERROR_INVALID, message);
            describe(v, "non-positive size sets a message", message,
                     sizeof message);
            check(std::strlen(tf_last_error_message()) > 0, message);
            if (handle != nullptr) {
                tf_storage_destroy(handle);
            }
        }
    }
    tf_clear_error();
}

void test_overflowing_size_is_rejected_or_fails_cleanly() {
#ifdef TF_TEST_UNDER_ASAN
    // See the note above the anonymous namespace: ASan turns an
    // oversized request into a hard error rather than a null return, on
    // *both* constructors, so probing it here would abort the process
    // without saying anything about the allocator's own contract. Every
    // other check in this file still runs under ASan.
    std::printf("SKIP (ASan): the unsatisfiable-size probe is an ASan "
                "hard error, not a null return\n");
#else
    char message[256];
    // An allocation this large cannot succeed. Either the allocator
    // refuses it (null + TF_ERROR_ALLOC) or new[] throws length_error /
    // bad_alloc, which the guard maps to an error code. What must never
    // happen is a non-null handle or a silent success.
    const std::int64_t huge = std::numeric_limits<std::int64_t>::max() / 4;
    for (const Variant& v : kVariants) {
        tf_clear_error();
        void* handle = v.create(huge);
        describe(v, "an unsatisfiable size returns null", message,
                 sizeof message);
        check(handle == nullptr, message);
        describe(v, "an unsatisfiable size sets an error", message,
                 sizeof message);
        check(tf_last_error_code() == TF_ERROR_ALLOC
                  || tf_last_error_code() == TF_ERROR_RUNTIME,
              message);
        if (handle != nullptr) {
            tf_storage_destroy(handle);
        }
    }
    tf_clear_error();
#endif
}

void test_injected_allocation_failure() {
    char message[256];
    for (const Variant& v : kVariants) {
        tf_clear_error();
        tf_test_arm_alloc_failure(1);   // the very next allocation fails
        void* handle = v.create(64);
        describe(v, "an injected allocation failure returns null", message,
                 sizeof message);
        check(handle == nullptr, message);
        describe(v, "an injected allocation failure sets TF_ERROR_ALLOC",
                 message, sizeof message);
        check(tf_last_error_code() == TF_ERROR_ALLOC, message);
        tf_test_arm_alloc_failure(0);   // disarm
        if (handle != nullptr) {
            tf_storage_destroy(handle);
        }
        // ...and the very next call succeeds, so the hook really is
        // one-shot rather than latching.
        tf_clear_error();
        handle = v.create(64);
        describe(v, "the allocation after an injected failure succeeds",
                 message, sizeof message);
        check(handle != nullptr && tf_last_error_code() == TF_OK, message);
        tf_storage_destroy(handle);
    }
    tf_clear_error();
}

void test_error_is_cleared_on_a_successful_call() {
    char message[256];
    for (const Variant& v : kVariants) {
        tf_clear_error();
        void* rejected = v.create(-1);          // sets TF_ERROR_INVALID
        check(rejected == nullptr, "rejected call returned a handle");
        void* handle = v.create(8);             // must clear the slot
        describe(v, "a successful call clears the previous error", message,
                 sizeof message);
        check(tf_last_error_code() == TF_OK, message);
        describe(v, "a successful call after a failure returns a handle",
                 message, sizeof message);
        check(handle != nullptr, message);
        tf_storage_destroy(handle);
    }
}

void test_destroy_is_null_safe() {
    tf_storage_destroy(nullptr);   // documented as a no-op
    check(true, "tf_storage_destroy(nullptr) returned");
}

void test_repeated_create_destroy_cycles() {
    char message[256];
    for (const Variant& v : kVariants) {
        for (int cycle = 0; cycle < 500; ++cycle) {
            void* handle = v.create(32);
            if (handle == nullptr) {
                describe(v, "a create/destroy cycle failed", message,
                         sizeof message);
                check(false, message);
                break;
            }
            tf_storage_fill(handle, static_cast<double>(cycle));
            tf_storage_destroy(handle);
        }
        describe(v, "500 create/destroy cycles left no error", message,
                 sizeof message);
        check(tf_last_error_code() == TF_OK, message);
    }
}

// -- the one deliberate difference ------------------------------------------

void test_zero_initializing_path_really_zeroes() {
    // The default's contract: every element reads back as +0.0. This is
    // what H1 must not have changed, and what the rejected kernels
    // (tf_core_sum, tf_core_narrow_backward) depend on for correctness.
    for (std::int64_t size : {std::int64_t{1}, std::int64_t{999}}) {
        void* handle = tf_storage_create(size);
        check(handle != nullptr, "zeroed creation returned null");
        if (handle == nullptr) {
            continue;
        }
        const tf::Storage* storage = tf::as_storage(handle);
        bool all_zero = true;
        for (std::int64_t i = 0; i < size; ++i) {
            all_zero = all_zero && storage->data[i] == 0.0;
        }
        check(all_zero, "tf_storage_create did not zero-initialize");
        tf_storage_destroy(handle);
    }
}

void test_uninitialized_storage_is_writable_and_owns_its_buffer() {
    // The production state of the uninitialized path. Its contents are
    // indeterminate by definition, so nothing is asserted about them —
    // reading them here would itself be the uninitialized read this
    // milestone exists to avoid. What is asserted is everything a caller
    // may rely on: the call succeeds, reports its size, owns a distinct
    // buffer, and is writable through the ordinary primitive.
    void* first = tf_storage_create_uninitialized(16);
    void* second = tf_storage_create_uninitialized(16);
    check(first != nullptr && second != nullptr,
          "uninitialized creation failed");
    if (first != nullptr && second != nullptr) {
        check(tf_storage_size(first) == 16, "uninitialized creation lost its size");
        check(first != second, "two creations returned the same handle");
        check(tf::as_storage(first)->data != tf::as_storage(second)->data,
              "two creations shared one buffer");
        tf_storage_fill(first, 2.0);
        tf_storage_fill(second, 5.0);
        check(tf::as_storage(first)->data[15] == 2.0,
              "uninitialized storage was not writable");
        // ...and writing one did not touch the other, so each owns its
        // own allocation rather than aliasing a shared one.
        check(tf::as_storage(second)->data[15] == 5.0,
              "one uninitialized buffer aliased another");
    }
    tf_storage_destroy(first);
    tf_storage_destroy(second);
}

void test_the_poison_technique_needs_no_library_support() {
    // How H1's completeness proof actually works, demonstrated at this
    // layer with **production exports only**: allocate uninitialized,
    // fill the returned buffer with a recognizable pattern, and hand
    // that same buffer to whatever is supposed to write it. There is no
    // poison hook in this library and none is needed — tf_storage_fill,
    // which every caller already has, writes every element.
    //
    // Both patterns the Python suite uses are exercised. A quiet NaN is
    // the sharpest poison for float data (it propagates through
    // arithmetic, so an unwritten element that is later *read*
    // contaminates everything downstream); a large negative finite value
    // catches the opposite mistake, code that special-cases NaN.
    const double finite_pattern = -1.2345678901234567e300;
    const double nan_pattern = std::numeric_limits<double>::quiet_NaN();
    const std::int64_t size = 256;

    void* handle = tf_storage_create_uninitialized(size);
    check(handle != nullptr, "uninitialized creation failed");
    if (handle != nullptr) {
        const tf::Storage* storage = tf::as_storage(handle);

        tf_storage_fill(handle, finite_pattern);
        bool all_poison = true;
        for (std::int64_t i = 0; i < size; ++i) {
            all_poison = all_poison && storage->data[i] == finite_pattern;
        }
        check(all_poison, "the finite poison did not reach every element");

        tf_storage_fill(handle, nan_pattern);
        bool all_nan = true;
        for (std::int64_t i = 0; i < size; ++i) {
            all_nan = all_nan && std::isnan(storage->data[i]);
        }
        check(all_nan, "the NaN poison did not reach every element");

        // A complete write leaves no survivor — the passing shape of
        // every proof in the Python suite.
        tf_storage_fill(handle, 1.5);
        int survivors = 0;
        for (std::int64_t i = 0; i < size; ++i) {
            if (std::isnan(storage->data[i])) {
                ++survivors;
            }
        }
        check(survivors == 0, "a complete write left poison behind");
        tf_storage_destroy(handle);
    }

    // ...and the zero-initializing path is untouched by any of this,
    // which is what lets the rejected kernels keep depending on it.
    void* zeroed = tf_storage_create(size);
    check(zeroed != nullptr, "zeroed creation failed");
    if (zeroed != nullptr) {
        const tf::Storage* storage = tf::as_storage(zeroed);
        bool all_zero = true;
        for (std::int64_t i = 0; i < size; ++i) {
            all_zero = all_zero && storage->data[i] == 0.0;
        }
        check(all_zero, "tf_storage_create stopped zero-initializing");
        tf_storage_destroy(zeroed);
    }
}

void test_many_live_handles_are_independently_accounted() {
    // Live-storage accounting at this layer: many handles of both kinds
    // open at once must be distinct, own distinct buffers, keep their
    // own contents, and all destroy cleanly. This is the C++ counterpart
    // of the Python live-storage baselines, which count the same objects
    // from the wrapper side.
    const int count = 64;
    std::vector<void*> handles;
    handles.reserve(count);
    for (int i = 0; i < count; ++i) {
        void* handle = (i % 2 == 0) ? tf_storage_create(8)
                                    : tf_storage_create_uninitialized(8);
        if (handle == nullptr) {
            check(false, "a handle in the live set failed to allocate");
            break;
        }
        tf_storage_fill(handle, static_cast<double>(i));
        handles.push_back(handle);
    }
    check(static_cast<int>(handles.size()) == count,
          "not every handle in the live set was created");

    bool distinct = true;
    bool intact = true;
    for (std::size_t i = 0; i < handles.size(); ++i) {
        intact = intact
            && tf_storage_size(handles[i]) == 8
            && tf::as_storage(handles[i])->data[7]
                   == static_cast<double>(i);
        for (std::size_t j = i + 1; j < handles.size(); ++j) {
            distinct = distinct
                && handles[i] != handles[j]
                && tf::as_storage(handles[i])->data
                       != tf::as_storage(handles[j])->data;
        }
    }
    check(distinct, "two live handles shared a handle or a buffer");
    check(intact, "a live handle lost its size or its contents");

    for (void* handle : handles) {
        tf_storage_destroy(handle);
    }
    check(tf_last_error_code() == TF_OK,
          "destroying the live set left an error set");

    // ...and the allocator still works afterwards, so nothing latched.
    void* after = tf_storage_create_uninitialized(8);
    check(after != nullptr, "allocation after a full release failed");
    tf_storage_destroy(after);
}

void test_the_two_paths_agree_once_written() {
    // The invariant that makes H1 an allocation change and nothing more:
    // after the same writes, the two buffers are bit-identical.
    const std::int64_t size = 128;
    void* zeroed = tf_storage_create(size);
    void* uninitialized = tf_storage_create_uninitialized(size);
    check(zeroed != nullptr && uninitialized != nullptr,
          "paired creation failed");
    if (zeroed != nullptr && uninitialized != nullptr) {
        tf_storage_fill(zeroed, 7.25);
        tf_storage_fill(uninitialized, 7.25);
        const double* a = tf::as_storage(zeroed)->data;
        const double* b = tf::as_storage(uninitialized)->data;
        check(std::memcmp(a, b, sizeof(double) * size) == 0,
              "the two paths differ after identical writes");
    }
    tf_storage_destroy(zeroed);
    tf_storage_destroy(uninitialized);
}

}  // namespace

int main() {
    test_valid_creation_and_size();
    test_zero_and_negative_sizes_are_rejected();
    test_overflowing_size_is_rejected_or_fails_cleanly();
    test_injected_allocation_failure();
    test_error_is_cleared_on_a_successful_call();
    test_destroy_is_null_safe();
    test_repeated_create_destroy_cycles();
    test_zero_initializing_path_really_zeroes();
    test_uninitialized_storage_is_writable_and_owns_its_buffer();
    test_the_poison_technique_needs_no_library_support();
    test_many_live_handles_are_independently_accounted();
    test_the_two_paths_agree_once_written();

    if (g_failures != 0) {
        std::printf("%d storage-allocation check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all storage-allocation checks passed\n");
    return 0;
}
