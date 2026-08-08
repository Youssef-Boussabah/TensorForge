# Native C ABI error contract

The experimental C++ backend is a set of plain C-ABI functions loaded
with ctypes (see `docs/backend_experiments.md`). C++ exceptions must
never cross that `extern "C"` boundary — an exception unwinding into the
CPython interpreter is undefined behavior and can crash the process. This
document describes the contract that makes every native call safe.

## The rule

**No exported native function may let a C++ exception escape.** Every
function that can fail (today: allocation inside storage creation and the
odometer walkers) wraps its body in the guard macros from
`cpp/include/tf_internal.h`:

```cpp
TF_EXPORT void tf_core_add(/* ... */) {
    TF_GUARD_BEGIN
    core_binary(/* ... */);
    TF_GUARD_END()
}
```

`TF_GUARD_BEGIN` clears the calling thread's error slot and opens a
`try`. `TF_GUARD_END(failure_return)` closes it with three catch blocks —
`std::bad_alloc`, `std::exception`, and `...` — each of which records a
status code and message in thread-local storage and returns a benign
value (nothing for `void`, `nullptr` for handle constructors) instead of
unwinding.

Functions that cannot fail (they neither allocate nor call fallible code
— `tf_storage_copy_from`, `tf_storage_copy_to`, `tf_storage_size`,
`tf_storage_destroy`, the raw-buffer elementwise and matmul kernels, the
error accessors themselves) are deliberately left unguarded: they never
touch the error slot, so nothing can go stale.

**`tf_storage_fill` and `tf_storage_scale` left that list at Phase K,
milestone K1, and the reason is exactly the rule above.** Both carry their
scalar across the ABI as a `double`, which represents every integer in
[-(2^53), 2^53] exactly and no integer outside it, so neither is an exact
integer primitive and both are floating-only permanently — they now ask
`tf::require_floating` and reject a non-floating storage with
`TF_ERROR_INVALID`, writing nothing. A function that can *record* an error
must clear the slot on entry, or a code it recorded could be misread
later; the guard is what gives them that clear-on-entry, so "unguarded
functions never touch the slot" stays true by moving these two out of the
unguarded set rather than by weakening the rule.

They remain **unhooked** on the Python side — H7 kept them hookless so
they cost one native call rather than two, and `_CHECKED_KERNELS` is
unchanged — so a rejection surfaces to a foreign caller through
`tf_last_error_code()` rather than as a Python exception. That is
acceptable because no Python path can reach it: the wrapper's own
narrowed validators refuse a non-floating dtype first, and
`NativeStorage.fill` rejects before the native call. The C-side check is
the **second**, independent authority the contract requires, not a
restatement of Python's.

## Status codes

`TfStatus` (in `tf_internal.h`) is the shared vocabulary:

| Code | Name               | Meaning                     | Python exception |
|------|--------------------|-----------------------------|------------------|
| 0    | `TF_OK`            | no error                    | —                |
| 1    | `TF_ERROR_ALLOC`   | allocation failure          | `MemoryError`    |
| 2    | `TF_ERROR_INVALID` | invalid argument            | `ValueError`     |
| 3    | `TF_ERROR_RUNTIME` | any other C++ failure       | `RuntimeError`   |

## Thread-local last-error slot

`error.cpp` owns a per-thread `(code, message)` pair. It is thread-local
so concurrent callers never see each other's errors, and the message is a
`std::string` whose buffer stays valid until the next `set`/`clear` on the
same thread. The exported accessors are:

- `int tf_last_error_code()` — the current code (`TF_OK` if none).
- `const char* tf_last_error_message()` — the current message (never
  null; `""` if none).
- `void tf_clear_error()` — reset to `TF_OK`.

Because every guarded function clears the slot on entry, a previous
error can never contaminate a later successful call.

## Python side: the errcheck hook

`src/tensorforge/backends/cpp.py` attaches a ctypes `errcheck` hook to the
guarded functions listed in `_CHECKED_KERNELS` — the guarded functions
whose failures are surfaced **automatically**. After each call the hook
reads `tf_last_error_code()`; a nonzero code is translated into the mapped
Python exception, carrying the native message and the failing function's
name for context, and the slot is cleared. Unguarded functions do **not**
get the hook, so a stale code from an earlier call is never misread as
their result.

**The general rule is "guarded ⇒ hooked", and it has exactly two
exceptions: `tf_storage_fill` and `tf_storage_scale`.** Both are guarded,
for the reason given above — they can record an integer-role rejection, and
a function that may write the slot must clear it on entry — and both are
deliberately left out of `_CHECKED_KERNELS`, so `_CHECKED_KERNELS` is
unchanged at **36**. H7 made each of them a single native call rather than
two, and no supported Python path can reach their rejection: the wrappers
above them validate the dtype first (`NativeStorage.fill` through
`cpp._require_floating_dtype`, and `NativeTensorCore.mean` — the only
`tf_storage_scale` caller — through `_require_floating_operand`). A direct
C caller inspects `tf_last_error_code()` instead of receiving a Python
exception. A structural guardrail in
`tests/test_native_integer_barriers.py` pins all four halves of this
arrangement at once, so neither the guard, the absence from
`_CHECKED_KERNELS`, nor the Python pre-validation can drift apart.

The result: a native failure surfaces as an ordinary Python exception at
the call site — `MemoryError` for an allocation failure, `ValueError` for
a bad argument, `RuntimeError` otherwise — never a crash or a silently
wrong answer. Because the fallible kernels allocate their output and
scratch buffers *before* writing any result, a failed call never
partially mutates caller-visible state: the operation simply raises and
the freshly allocated (never-returned) output is discarded.

## Allocation safety (RAII)

`tf_storage_create` allocates the data buffer through a
`std::unique_ptr<double[]>` and only `release()`s it after the `Storage`
metadata node is successfully constructed. If the metadata allocation
fails, the `unique_ptr` frees the buffer — the "data allocated, then
metadata throws, buffer leaks" scenario cannot occur. The odometer
walkers hold their index counter in a `std::vector<int64_t>`
(`tf::make_counter`), which frees itself on any exception.

## Test-only fault injection

To exercise the failure paths deterministically without exhausting
memory, the backend includes an inert-until-armed allocation fault hook:

- `void tf_test_arm_alloc_failure(int64_t nth)` — arm the calling thread
  so the `nth` subsequent internal allocation attempt fails with a
  simulated `std::bad_alloc` (`nth = 1` targets the very next allocation;
  `nth <= 0` disarms). Each `tf_storage_create` and each walker counter
  is one allocation attempt.
- `int tf_fault_injection_available()` — returns 1 (the hook is compiled
  into every build from this repo).

The hook is **inert in normal use**: `tf::should_fail_alloc()` is a single
thread-local branch that is always false until a test arms it, so it never
changes a successful build's results and never alters correct behavior.
Python exposes it as `cpp.fault_injection_available()` and the test-only
`cpp._arm_alloc_failure(nth)`; the failure tests live in
`tests/test_native_abi_errors.py`.

## Source layout

The backend is split into coherent translation units under `cpp/`:

| File                     | Responsibility                                        |
|--------------------------|-------------------------------------------------------|
| `include/tf_internal.h`  | export macro, status codes, error/guard interface, fault hook, `Storage`, RAII counter |
| `src/error.cpp`          | thread-local error slot, exported accessors, fault injection |
| `src/storage.cpp`        | owned storage lifetime, copy, and materialization     |
| `src/elementwise.cpp`    | unary/binary walkers, ReLU (+ backward), sqrt/reciprocal, legacy kernels |
| `src/reduction.cpp`      | sum reduction and narrow-backward scatter             |
| `src/matmul.cpp`         | strided tensor-core matmul and the legacy naive/tiled pair |

Exported symbols keep their historical names and signatures, so the
Python ctypes loader is unchanged apart from the added error contract.
Internal helpers live in `namespace tf` (or file-local anonymous
namespaces) and are never exported, keeping the C ABI narrow. Autograd
stays entirely on the Python side; the C++ kernels remain autograd-unaware.

## Building

The canonical build is CMake (`cpp/CMakeLists.txt`); `cpp/build.py` is a
thin wrapper that uses CMake when available and otherwise compiles the
same sources directly (the path CI and the bundled `ziglang` compiler
use). See `docs/backend_experiments.md` for the full build and sanitizer
instructions.
