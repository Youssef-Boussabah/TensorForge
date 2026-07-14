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
#include <exception>
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

// The C++-owned float64 buffer behind an opaque handle. Python holds the
// handle only, moving data through copy_from/copy_to; every compute
// translation unit reads and writes ``.data`` through these accessors.
struct Storage {
    double* data;
    int64_t size;
};

inline const Storage* as_storage(const void* handle) {
    return static_cast<const Storage*>(handle);
}
inline Storage* as_storage(void* handle) {
    return static_cast<Storage*>(handle);
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
