// Thread-local native error state, the exported error accessors, and the
// test-only fault-injection hook. This is the runtime half of the error
// contract declared in tf_internal.h; the compute translation units use
// TF_GUARD_* to feed it and never touch these globals directly.

#include <string>

#include "tf_internal.h"

namespace {

// One error slot per thread: a code and a message that outlive the call
// (a std::string owns its buffer, so the const char* Python reads stays
// valid until the next set/clear on this thread). Thread-local so
// concurrent callers never see each other's errors.
thread_local int g_error_code = TF_OK;
thread_local std::string g_error_message;

// Fault-injection countdown, per thread. 0 means disarmed; when armed to
// N, the N-th subsequent allocation check fires exactly once and then
// disarms. Inert (a single branch that is always false) unless a test
// arms it, so successful builds are unaffected.
thread_local int64_t g_alloc_failure_countdown = 0;

}  // namespace

namespace tf {

void set_error(int code, const char* message) noexcept {
    g_error_code = code;
    try {
        g_error_message = (message != nullptr) ? message : "";
    } catch (...) {
        // Even recording the message must never throw across the ABI;
        // the code alone still signals failure.
        g_error_message.clear();
    }
}

void clear_error() noexcept {
    g_error_code = TF_OK;
    g_error_message.clear();
}

int last_error_code() noexcept { return g_error_code; }

const char* last_error_message() noexcept { return g_error_message.c_str(); }

bool should_fail_alloc() noexcept {
    if (g_alloc_failure_countdown > 0) {
        if (--g_alloc_failure_countdown == 0) {
            return true;
        }
    }
    return false;
}

}  // namespace tf

// ---------------------------------------------------------------------------
// Exported ABI: error introspection. These are the only exported
// functions Python calls *without* the errcheck hook — they read or
// reset the slot rather than performing work, so they never clear the
// error on entry and never set one.
// ---------------------------------------------------------------------------

// The calling thread's last error code (TfStatus), or TF_OK if none.
TF_EXPORT int tf_last_error_code() { return tf::last_error_code(); }

// The calling thread's last error message (never null; "" if none). The
// pointer is valid until the next native call on this thread.
TF_EXPORT const char* tf_last_error_message() {
    return tf::last_error_message();
}

// Clear the calling thread's error slot back to TF_OK.
TF_EXPORT void tf_clear_error() { tf::clear_error(); }

// ---------------------------------------------------------------------------
// Exported ABI: test-only fault injection. Arms the calling thread so the
// ``nth`` subsequent internal allocation attempt fails with a simulated
// std::bad_alloc (nth = 1 targets the very next one); nth <= 0 disarms.
// Deterministic, thread-local, and inert until armed. Present in every
// build but documented as test-only; ``tf_fault_injection_available``
// lets tests confirm the hook is compiled in. This export sits in the
// same translation unit as the countdown, so nothing internal leaks.
// ---------------------------------------------------------------------------

TF_EXPORT void tf_test_arm_alloc_failure(int64_t nth) {
    g_alloc_failure_countdown = nth > 0 ? nth : 0;
}

TF_EXPORT int tf_fault_injection_available() { return 1; }
