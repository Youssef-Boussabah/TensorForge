// TensorForge experimental C++ backend — shared internal header.
//
// This is the one place the internal contract lives: the export macro,
// the C-ABI error codes, the thread-local last-error interface, the
// test-only fault-injection hook, the exception-guard macros that make
// the ``extern "C"`` boundary exception-safe, and the owned-storage
// struct every compute translation unit shares. It is NOT the public
// ABI surface Python sees — Python loads plain exported symbols with
// ctypes and needs no header. Keep genuinely public C functions marked
// with TF_EXPORT in the .cpp files; keep everything else in ``namespace
// tf`` so it never leaks into the ABI.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <limits>
#include <new>
#include <vector>

#ifdef _WIN32
#define TF_EXPORT extern "C" __declspec(dllexport)
#else
#define TF_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ---------------------------------------------------------------------------
// The native error contract (see docs/native_abi_error_contract.md).
//
// No C++ exception may cross ``extern "C"``. Every fallible exported
// function wraps its body in TF_GUARD_BEGIN / TF_GUARD_END: the guard
// clears the thread-local error on entry (so a previous error can never
// contaminate a later successful call), runs the body, and on any
// exception records a status code plus message in thread-local storage
// and returns a benign value instead of unwinding across the boundary.
// Python reads the thread-local error after each call (a ctypes
// ``errcheck`` hook) and raises the matching exception.
// ---------------------------------------------------------------------------

enum TfStatus {
    TF_OK = 0,             // no error
    TF_ERROR_ALLOC = 1,   // allocation failure   -> Python MemoryError
    TF_ERROR_INVALID = 2, // invalid argument     -> Python ValueError
    TF_ERROR_RUNTIME = 3, // any other C++ failure -> Python RuntimeError
};

// ---------------------------------------------------------------------------
// The native dtype model (Phase I, milestone I1; see
// docs/native_dtype_float32_design.md §3).
//
// Exactly two dtype authorities exist in the repository — this one, and
// the private code table in src/tensorforge/backends/cpp.py — and they
// agree by construction because the ABI codes are the same integers.
// There is no third, and no exported dtype-query function: Python knows a
// storage's dtype because Python asked for it at creation.
//
// The codes are **frozen** in exactly the sense the TfStatus codes are:
// their meaning never changes, and a hypothetical future dtype (not in
// Phase I) would take 2. float64 is 0 so that the untyped compatibility
// creators pass a code they could equally have defaulted to.
// ---------------------------------------------------------------------------

enum TfDtype {
    TF_DTYPE_FLOAT64 = 0,
    TF_DTYPE_FLOAT32 = 1,
};

namespace tf {

// The internal form. Scoped, fixed-width, and constructible only through
// ``dtype_from_code`` or one of the two enumerators — so an unvalidated
// integer can never become a Dtype.
enum class Dtype : std::int32_t {
    Float64 = TF_DTYPE_FLOAT64,
    Float32 = TF_DTYPE_FLOAT32,
};

// The platform assumptions the whole dtype model rests on, stated and
// checked at compile time rather than assumed (design §23). A toolchain
// where these do not hold is not supported, and failing here is far
// better than silently computing a wrong element width.
static_assert(sizeof(double) == 8, "TensorForge requires an 8-byte double");
static_assert(sizeof(float) == 4, "TensorForge requires a 4-byte float");
static_assert(std::numeric_limits<double>::is_iec559,
              "TensorForge requires IEEE-754 binary64 double");
static_assert(std::numeric_limits<float>::is_iec559,
              "TensorForge requires IEEE-754 binary32 float");

// ABI code -> internal dtype. Total, noexcept, and allocation-free:
// every unknown code returns false — never a default, never a clamp,
// never an assertion — and leaves ``out`` exactly as the caller left it,
// so a rejected conversion cannot produce an uninitialized dtype.
inline bool dtype_from_code(std::int32_t code, Dtype& out) noexcept {
    switch (code) {
        case TF_DTYPE_FLOAT64: out = Dtype::Float64; return true;
        case TF_DTYPE_FLOAT32: out = Dtype::Float32; return true;
        default: return false;
    }
}

// The **single** place an element width is written down in C++. No
// kernel, export, test helper, or build file may spell ``sizeof(double)``
// or ``8`` as a storage width again; anything that needs a width calls
// this. (Written as a switch with no default so that adding a dtype
// without giving it a size is a compile-time problem, not a runtime one.)
inline std::size_t dtype_item_size(Dtype dtype) noexcept {
    switch (dtype) {
        case Dtype::Float32: return sizeof(float);
        case Dtype::Float64: break;
    }
    return sizeof(double);
}

// The canonical name, matching SUPPORTED_DTYPES exactly. For error
// messages; never parsed, never compared, never dispatched on.
inline const char* dtype_name(Dtype dtype) noexcept {
    switch (dtype) {
        case Dtype::Float32: return "float32";
        case Dtype::Float64: break;
    }
    return "float64";
}

inline bool dtype_is_float32(Dtype dtype) noexcept {
    return dtype == Dtype::Float32;
}

inline bool dtype_is_float64(Dtype dtype) noexcept {
    return dtype == Dtype::Float64;
}

// Checked ``numel x itemsize`` — the **one** element-to-byte conversion
// in the runtime (design §4.3). Everything else — shapes, strides,
// offsets, spans, bounds checks, every ABI layout argument — stays in
// logical elements, which is what lets the whole existing layout and
// bounds-checking apparatus carry over untouched.
//
// Proves the product is representable before anything is allocated, so an
// overflow is a rejection rather than a silent wrap into a small
// allocation a kernel would then overrun. This check is *new*: the
// implicit ``new double[count]`` sizing it replaces had no equivalent.
inline bool dtype_checked_bytes(std::int64_t size, Dtype dtype,
                                std::size_t& out) noexcept {
    if (size <= 0) {
        return false;
    }
    const std::int64_t item = static_cast<std::int64_t>(dtype_item_size(dtype));
    if (size > INT64_MAX / item) {
        return false;
    }
    const std::int64_t bytes = size * item;
    // On a platform whose size_t is narrower than int64 the product must
    // also fit the allocator's own count type. Discarded at compile time
    // on the 64-bit platforms TensorForge supports, so it costs nothing
    // and cannot produce a tautological-comparison warning there.
    if constexpr (sizeof(std::size_t) < sizeof(std::int64_t)) {
        if (static_cast<std::uint64_t>(bytes)
                > static_cast<std::uint64_t>(SIZE_MAX)) {
            return false;
        }
    }
    out = static_cast<std::size_t>(bytes);
    return true;
}

}  // namespace tf

namespace tf {

// Thread-local last-error slot (defined in error.cpp). ``set_error``
// records a code and message for the calling thread; ``clear_error``
// resets it to TF_OK; the readers expose it to Python. All noexcept.
void set_error(int code, const char* message) noexcept;
void clear_error() noexcept;
int last_error_code() noexcept;
const char* last_error_message() noexcept;

// Test-only, deterministic fault injection (defined in error.cpp).
// ``should_fail_alloc`` returns true exactly when a test has armed the
// countdown and this is the targeted allocation attempt; it is inert
// (always false, a single predictable branch) in normal use and never
// changes a successful build's results. Allocation sites call it and
// throw std::bad_alloc when it fires, so the ordinary alloc-failure
// path is exercised without exhausting memory.
bool should_fail_alloc() noexcept;

// A note on how "every destination element is written" is proved for
// ``tf_storage_create_uninitialized`` (Phase H, milestone H1), because
// the answer is deliberately **not** in this library.
//
// That export leaves its buffer unwritten, and real uninitialized heap
// memory is a poor oracle: a fresh page from the OS reads back as zeros,
// so a kernel that forgets to write an element would silently "pass" a
// naive check. Neither ASan nor UBSan detects a read of an uninitialized
// *value* either — that is MemorySanitizer's job, and MSan needs a fully
// instrumented libc and CPython, which this project does not have.
//
// The proof therefore uses a deterministic **poison**: an uninitialized
// buffer is filled with a recognizable pattern (a quiet NaN, or a large
// nontrivial finite value) and the real kernel is then run over it, so
// any element the kernel failed to write is left holding the pattern at
// a locatable index. That poison is applied **entirely by test
// infrastructure**, which wraps the private Python allocation helper and
// writes the pattern through the ordinary ``tf_storage_fill`` primitive
// between the real allocation and the real kernel call. **No
// poison-control hook, thread-local flag, environment variable, or
// global mode exists anywhere in this library**, and none may be added:
// a debugging switch that can alter allocation contents is not something
// the shipped runtime should expose, however carefully it is disarmed.

// The C++-owned, dtype-tagged buffer behind an opaque handle (Phase I,
// milestone I1; design §4.1). Python holds the handle only, moving data
// through copy_from/copy_to.
//
//   * ``data`` is **untyped**. There is no ``double*`` member, no union of
//     typed pointers, and no second pointer: a union would let the tag and
//     the pointer disagree about what the buffer holds, and a ``void*``
//     cannot. It points at a genuine ``float[size]`` or ``double[size]``
//     array object, created by an ordinary array new-expression in
//     storage.cpp and type-erased **after** creation — which is what makes
//     the kernels' ``data[i]`` and ``data + i`` well-defined across the
//     whole allocation under C++17.
//   * ``size`` is a **logical element count**, unchanged in meaning and in
//     name, so ``tf_storage_size`` still returns exactly what it always
//     returned and no caller reinterprets it.
//   * ``dtype`` is the **single** authority for this buffer's element
//     type. It is assigned once, before the handle is published, and no
//     export, method, or loader may change it — changing a tensor's dtype
//     in place would be a cast, and casts do not exist (design §9).
//
// The physical byte size is *derivable* as ``size * dtype_item_size(dtype)``
// and is deliberately **not stored**: a second source of truth is a second
// thing that can be wrong, and the derivation is one multiplication of two
// values the struct already holds.
//
// The default makes every pre-Phase-I aggregate initializer — including
// the ones in the C++ CTests, which build Storage nodes on the stack —
// mean exactly what it meant before: a float64 buffer.
struct Storage {
    void*   data;
    int64_t size;
    Dtype   dtype = Dtype::Float64;
};

inline const Storage* as_storage(const void* handle) {
    return static_cast<const Storage*>(handle);
}
inline Storage* as_storage(void* handle) {
    return static_cast<Storage*>(handle);
}

// The **one** place an untyped storage buffer becomes a typed pointer.
//
// Precondition: the caller has already established that this storage is
// float64 — through ``require_float64`` at the export boundary, or by
// having constructed the node itself. Every float64 kernel in the runtime
// reaches its operands through exactly these two accessors, so there is no
// scattered cast to audit and no site that can reinterpret a buffer
// without having asked.
//
// The dtype tag is what makes the recovery sound: when it says float64,
// ``data`` really points at the first element of a ``double[]`` array
// created as such, so the returned pointer addresses one array object and
// indexing it across ``size`` elements is well-defined.
inline double* storage_f64(void* handle) noexcept {
    return static_cast<double*>(as_storage(handle)->data);
}
inline const double* storage_f64(const void* handle) noexcept {
    return static_cast<const double*>(as_storage(handle)->data);
}

// Phase I, milestone I2: the dtype-general sibling of ``storage_f64``.
//
// Same contract, same soundness argument, one degree of freedom more: the
// caller has already established — through ``storage_dtype`` at a single
// dispatch point — that this storage's tag really is ``T``'s dtype, so the
// recovered pointer addresses the first element of a genuine ``T[]`` array
// and indexing it across ``size`` elements is well-defined.
//
// It exists so that a dtype-general export has exactly **one** way to
// reach a typed buffer, just as every float64 kernel has exactly one. An
// export that has not been generalized keeps ``storage_f64`` and
// ``require_float64``; nothing may cast ``Storage::data`` by hand.
template <class T>
inline T* storage_typed(void* handle) noexcept {
    return static_cast<T*>(as_storage(handle)->data);
}
template <class T>
inline const T* storage_typed(const void* handle) noexcept {
    return static_cast<const T*>(as_storage(handle)->data);
}

// The storage's dtype tag — the single authority a dtype-general export
// dispatches on, read **once** per exported call (design §8.1). There is
// deliberately no exported ``tf_storage_dtype``: Python knows a storage's
// dtype because Python asked for it at creation, and a query symbol would
// be a second authority for a value the wrapper already owns (§6.6).
inline Dtype storage_dtype(const void* handle) noexcept {
    return as_storage(handle)->dtype;
}

// Phase I, milestone I1: the transitional dtype guard.
//
// I1 gives storage a dtype tag and gives the ABI a way to allocate
// float32, but generalizes **no** operation — that is I2 onward. A
// float32 handle can therefore reach a kernel that still reads float64,
// because ``tf_storage_create_typed`` is a public C ABI symbol even while
// float32 is not a publicly supported TensorForge dtype. Reading a
// 4-byte-per-element buffer through a ``double*`` would overrun it by a
// factor of two, so every operation not generalized in I1 rejects a
// float32 operand **before** it reads or writes anything.
//
// The default rule, stated once here rather than re-derived per file: if
// an operation has not been explicitly generalized, a float32 handle is
// invalid for it.
//
// Returns true when every handle may proceed. On rejection it has already
// recorded TF_ERROR_INVALID naming the operation and the offending dtype,
// and the caller must return without touching any destination.
//
// Null handles **pass**: each export keeps its own null validation, with
// its own message and its own ordering, and this guard must not pre-empt
// it. The exports that never validated a null handle are no worse off
// than they were before I1.
inline bool require_float64(
    const char* operation, std::initializer_list<const void*> handles
) noexcept {
    for (const void* handle : handles) {
        if (handle == nullptr) {
            continue;
        }
        const Dtype dtype = as_storage(handle)->dtype;
        if (!dtype_is_float64(dtype)) {
            // Bounded stack buffer: recording a rejection must not itself
            // allocate, and snprintf truncates rather than overflowing.
            char message[192];
            std::snprintf(message, sizeof message,
                          "%s: this operation is float64-only in the current "
                          "runtime; got %s storage",
                          operation, dtype_name(dtype));
            set_error(TF_ERROR_INVALID, message);
            return false;
        }
    }
    return true;
}

// Phase I, milestone I2: the operand-agreement guard for the operations
// that **are** dtype-general.
//
// ``require_float64`` says "this operation has not been generalized"; this
// one says "this operation has been, and its operands must agree". There
// is no promotion, no narrowing, no widening, and no cast anywhere in the
// runtime (design §9), so a float32 source and a float64 destination is an
// invalid *request* rather than a conversion opportunity.
//
// Rejection is TF_ERROR_INVALID naming both dtypes, recorded before the
// caller touches any destination — a rejecting export writes nothing.
//
// Null handles **pass**, for exactly ``require_float64``'s reason: each
// export keeps its own null validation, with its own message and its own
// ordering, and this guard must not pre-empt it.
//
// The list form (Phase I, milestone I3) is the same rule over any number of
// participating handles — a binary kernel's two sources and its
// destination, say — and is the **one** implementation: the two-handle
// overload below delegates to it, so a three-operand export and a
// two-operand one cannot drift apart in what they accept or in what they
// say when they refuse. Every non-null handle is compared against the first
// non-null one, so the message always names the disagreeing pair.
inline bool require_matching_dtype(
    const char* operation, std::initializer_list<const void*> handles
) noexcept {
    const void* reference = nullptr;
    for (const void* handle : handles) {
        if (handle == nullptr) {
            continue;
        }
        if (reference == nullptr) {
            reference = handle;
            continue;
        }
        const Dtype a = as_storage(reference)->dtype;
        const Dtype b = as_storage(handle)->dtype;
        if (a != b) {
            // Bounded stack buffer: recording a rejection must not itself
            // allocate, and snprintf truncates rather than overflowing.
            char message[192];
            std::snprintf(message, sizeof message,
                          "%s: operands must have the same dtype; got %s and "
                          "%s (the native runtime performs no casting or "
                          "promotion)",
                          operation, dtype_name(a), dtype_name(b));
            set_error(TF_ERROR_INVALID, message);
            return false;
        }
    }
    return true;
}

inline bool require_matching_dtype(
    const char* operation, const void* first, const void* second
) noexcept {
    return require_matching_dtype(operation, {first, second});
}

// Phase I, milestone I3: the dtype a dtype-general export dispatches on,
// read **once** per exported call from the handles the caller already
// passed (design §8.1). Total, noexcept, allocation-free, and a function of
// the storage tags alone — never of a pointer value, an alignment, a clock,
// an environment variable, or a CPU-feature probe.
//
// It is called *after* ``require_matching_dtype``, which has already proved
// every non-null handle carries the same tag, so the first non-null one is
// the answer and there is nothing to choose between.
//
// Null handles are skipped for exactly ``require_float64``'s reason: each
// export keeps its own null behavior, and selecting an instantiation must
// not change where a malformed call fails. An all-null call answers
// Float64, which is the instantiation such a call already ran before Phase
// I, so it fails exactly where it always did.
inline Dtype dispatch_dtype(
    std::initializer_list<const void*> handles
) noexcept {
    for (const void* handle : handles) {
        if (handle != nullptr) {
            return as_storage(handle)->dtype;
        }
    }
    return Dtype::Float64;
}

// An owned odometer counter: RAII-managed so a walker that throws (or is
// interrupted by an injected allocation failure) never leaks it. Also
// honors fault injection so tests can force the counter allocation to
// fail deterministically.
inline std::vector<int64_t> make_counter(int64_t ndim) {
    if (should_fail_alloc()) {
        throw std::bad_alloc();
    }
    return std::vector<int64_t>(static_cast<size_t>(ndim), 0);
}

}  // namespace tf

// The exception guard. A guarded body is opened with TF_GUARD_BEGIN and
// closed with one of two endings that differ only in what they return on
// failure:
//   * TF_GUARD_END(failure_return) — value-returning functions; returns
//     ``failure_return`` (e.g. ``nullptr`` for handle constructors).
//   * TF_GUARD_END_VOID()          — ``void`` functions; returns nothing.
// A single ``void`` ending is used instead of TF_GUARD_END() with an
// empty argument because MSVC warns (C4003) about the missing argument.
// Both endings share one catch block, so the exception mapping and error
// messages are defined in exactly one place.
#define TF_GUARD_BEGIN     \
    tf::clear_error();     \
    try {
// Private helper: the shared catch clauses. ``return_stmt`` is the full
// return statement each catch runs after recording the error — ``return``
// for the void ending, ``return failure_return`` for the value ending.
// Portable across MSVC, Clang, and GCC (no empty macro arguments).
#define TF_GUARD_CATCH_(return_stmt)                                        \
    }                                                                      \
    catch (const std::bad_alloc&) {                                        \
        tf::set_error(TF_ERROR_ALLOC, "native allocation failed");         \
        return_stmt;                                                       \
    }                                                                      \
    catch (const std::exception& error) {                                  \
        tf::set_error(TF_ERROR_RUNTIME, error.what());                     \
        return_stmt;                                                       \
    }                                                                      \
    catch (...) {                                                          \
        tf::set_error(TF_ERROR_RUNTIME, "unknown native error");           \
        return_stmt;                                                       \
    }
#define TF_GUARD_END(failure_return) TF_GUARD_CATCH_(return failure_return)
#define TF_GUARD_END_VOID() TF_GUARD_CATCH_(return)
