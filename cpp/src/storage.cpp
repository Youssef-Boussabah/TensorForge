// Native storage: a C++-owned, dtype-tagged buffer behind an opaque
// handle, plus the contiguous-materialization kernel that reads a strided
// view out of it. Python holds the handle only, moves data in and out
// through copy_from/copy_to, and destroys the handle exactly once (the
// wrapper's close() guarantees that). This is the storage half of the
// tensor runtime; the shape/stride metadata layer lives on the Python
// side.
//
// Phase I, milestone I1 replaced the physically ``double``-only buffer
// with an untyped allocation plus a dtype tag (design §4). The tag is the
// single authority for how the bytes are read; ``size`` still counts
// logical elements; and byte arithmetic happens **only** here, at the
// allocation boundary, through the one checked conversion in
// tf_internal.h.
//
// ---------------------------------------------------------------------------
// Object lifetime: why storage owns a genuine typed *array*
// ---------------------------------------------------------------------------
//
// The project is compiled as **C++17**, and that decides the shape of
// this file.
//
// The kernels do not merely dereference a pointer; they index one:
// ``data[i]`` and ``data + i`` across the whole allocation. In C++17
// pointer arithmetic is only defined *within a single array object*
// ([expr.add]/4), and an object that is not an array element is treated
// as a one-element array. Two things therefore do **not** work, and both
// are tempting:
//
//   * ``new unsigned char[n]`` plus a reinterpreting cast — that begins
//     the lifetime of an array of ``unsigned char`` and of no ``float``
//     or ``double`` at all. C++20 added implicit object creation for
//     exactly this pattern ([intro.object]/10, P0593); C++17 did not, and
//     the project may not rely on a rule it does not compile under.
//   * raw storage plus a per-element placement-new loop — that does begin
//     ``count`` floating-point lifetimes, but they are ``count``
//     *separate scalar objects*. Adjacent scalars do not become elements
//     of one array merely because their storage is contiguous, so
//     ``data[i]`` past the first still walks outside its array object.
//
// The model that actually supports the indexing is the ordinary one: an
// **array new-expression**, which creates a real ``float[count]`` or
// ``double[count]`` object. The element type is a runtime property, so
// the choice is made by one dtype dispatch into a templated allocation
// body; after that the array pointer is type-erased into ``Storage::data``
// as ``void*``.
//
// The immutable dtype tag is what makes the erasure safe in both
// directions: it selects the typed accessor that recovers ``T*``, and it
// selects the matching ``delete[]`` at destruction. Nothing else in the
// runtime is allowed to decide either.
//
// No per-element destruction is needed, and that is a property of the
// types rather than an assumption: ``float`` and ``double`` are trivially
// destructible, which is ``static_assert``ed beside the allocation.

#include <memory>
#include <type_traits>

#include "tf_internal.h"

using tf::Dtype;
using tf::Storage;
using tf::as_storage;

namespace {

// The templated allocation body: creates a genuine ``T[count]`` array
// object, then publishes it behind the metadata node.
//
// This is the **only** thing that differs between the two dtypes. Every
// other part of creation — size validation, byte-count checking, fault
// injection, error reporting, and the ordering that makes failure atomic
// — lives in the single caller below, so the two instantiations cannot
// drift apart on any of it.
//
// ``zero_initialize`` selects between the two array-new forms, and the
// difference is exactly the H1 contract:
//
//   * ``new T[count]()`` — a **value-initialized** array. For a scalar
//     element type this zero-initializes every element, and IEEE-754 zero
//     at both widths is **positive** zero, so every element is +0.0f /
//     +0.0 with the sign bit clear. This is the default everywhere and
//     the only behavior any caller had before Phase H.
//   * ``new T[count]``   — a **default-initialized** array. For a scalar
//     element type this performs no initialization, so the contents are
//     indeterminate and no write pass is carried at all. Legal only for a
//     destination the caller has *proved* is completely written before
//     any read (see docs/native_cpu_performance_design.md §H1).
//
// Both are ordinary array new-expressions, so what they create is one
// array object — which is precisely what makes the kernels' ``data[i]``
// and ``data + i`` well-defined across the whole allocation.
//
// Returns the published handle, or null with the thread-local error set.
template <class T>
Storage* create_typed_storage(int64_t size, Dtype dtype,
                              bool zero_initialize) {
    // The licence for having no per-element destruction, stated where it
    // is used rather than assumed: ``delete[]`` on these arrays runs no
    // destructor, so releasing the array is the whole of destruction.
    static_assert(std::is_trivially_destructible<T>::value,
                  "storage elements must be trivially destructible, or "
                  "destruction would need to run a destructor pass");
    const std::size_t count = static_cast<std::size_t>(size);
    // Type-correct array ownership: ``unique_ptr<T[]>`` releases through
    // ``delete[]`` on a ``T*``, which is the exact partner of the
    // expression that allocated it. It owns the array until the Storage
    // node has successfully adopted it, so the "array allocated, then
    // metadata allocation fails, array leaks" scenario the ABI contract
    // calls out cannot happen.
    std::unique_ptr<T[]> data(
        zero_initialize ? new (std::nothrow) T[count]()
                        : new (std::nothrow) T[count]);
    if (!data) {
        tf::set_error(TF_ERROR_ALLOC, "could not allocate native storage");
        return nullptr;
    }
    // Nothing else happens to the array here. In particular there is no
    // poison hook, debug flag, environment variable, or global mode that
    // could alter what an uninitialized allocation contains: the
    // "every destination element is written" proof is built by test
    // infrastructure *around* this function (it fills the returned
    // storage through tf_storage_fill before running the real kernel),
    // not by a switch inside it. See tf_internal.h.
    //
    // The array pointer is type-erased into ``void*`` only *after* the
    // array exists. The immutable dtype tag recorded beside it is what
    // lets the typed accessors recover the right ``T*`` and what selects
    // the matching ``delete[]`` at destruction.
    Storage* storage = new (std::nothrow) Storage{data.get(), size, dtype};
    if (storage == nullptr) {
        tf::set_error(TF_ERROR_ALLOC, "could not allocate native storage");
        return nullptr;  // array freed by unique_ptr<T[]> — no leak
    }
    data.release();  // ownership transferred to the Storage node
    return storage;
}

// The one creation body every exported constructor shares — the two
// untyped float64 creators and the two typed ones — so no path can drift
// from another on validation, fault injection, allocation-failure
// handling, error state, ownership, or the RAII ordering that makes the
// "buffer allocated, then metadata throws" leak impossible.
//
// ``zero_initialize`` is the *only* behavioral difference, and it is a
// compile-time-fixed argument of the exports rather than any kind of
// runtime policy. It selects the array-new form; see
// ``create_typed_storage`` above.
//
// The sequence, in the order the contract fixes it:
//
//   1. validate the dtype (done by the caller — it arrives as a ``Dtype``,
//      which only ``dtype_from_code`` or an enumerator can produce);
//   2. validate the positive logical size;
//   3. check that ``size x itemsize`` is representable;
//   4. apply deterministic pre-allocation fault injection;
//   5. dispatch **once** on the dtype into the templated body, which
//      allocates the typed array, adopts it under RAII, allocates the
//      metadata, and publishes the handle.
//
// Steps 2-4 are written once here rather than once per dtype, which is
// what keeps the two instantiations of step 5 from drifting apart.
//
// Returns the handle, or null with the thread-local error already set.
// Never throws across the caller's guard boundary by itself.
void* create_storage(int64_t size, Dtype dtype, bool zero_initialize) {
    if (size <= 0) {
        tf::set_error(TF_ERROR_INVALID, "storage size must be positive");
        return nullptr;
    }
    // Checked numel x itemsize, before any allocation is attempted. A
    // product that is not representable is an invalid *request*, not an
    // allocation failure, so it is TF_ERROR_INVALID rather than
    // TF_ERROR_ALLOC and no allocator is ever asked for it.
    //
    // The byte count is a *validation*, not a sizing input: the array
    // new-expression below computes its own size from the element type
    // and count. Checking it here is what turns an unrepresentable
    // request into a clean rejection instead of a throw from the
    // allocator (or, on a sloppier allocator, a wrapped small allocation).
    std::size_t bytes = 0;
    if (!tf::dtype_checked_bytes(size, dtype, bytes)) {
        char message[160];
        std::snprintf(message, sizeof message,
                      "storage of %lld %s elements overflows the addressable "
                      "byte range",
                      static_cast<long long>(size), tf::dtype_name(dtype));
        tf::set_error(TF_ERROR_INVALID, message);
        return nullptr;
    }
    if (tf::should_fail_alloc()) {  // test-only injected failure (one point)
        throw std::bad_alloc();
    }
    // The one dtype dispatch in the allocation path: once per allocation,
    // never per element, choosing which array type to create. Everything
    // below this point is the shared templated body.
    switch (dtype) {
        case Dtype::Float32:
            return create_typed_storage<float>(size, dtype, zero_initialize);
        case Dtype::Float64:
            break;
    }
    return create_typed_storage<double>(size, dtype, zero_initialize);
}

// The **one** authoritative release of a storage's array, and the mirror
// image of the dispatch in ``create_storage``. The dtype tag that chose
// the array type chooses the matching ``delete[]``, applied to the exact
// ``T*`` the array was created as — so allocation and deallocation forms
// cannot disagree.
//
// No ``default:`` label, deliberately. Every enumerator is handled, so a
// dtype added without a deleter is a compile-time warning rather than a
// runtime surprise; and a tag that somehow held neither value would fall
// through **without deleting**, which leaks a block a sanitizer will
// report rather than running a wrong ``delete[]``, which is undefined
// behavior. Declining to guess is the safer failure. (No such tag is
// constructible: it is set once from a validated ``Dtype`` before the
// handle is published and is immutable afterwards.)
void destroy_storage_data(Storage* storage) noexcept {
    switch (storage->dtype) {
        case Dtype::Float32:
            delete[] static_cast<float*>(storage->data);
            return;
        case Dtype::Float64:
            delete[] static_cast<double*>(storage->data);
            return;
    }
}

// The typed creators' shared front half: resolve and validate the ABI
// dtype code, then hand off to the body above.
//
// The validation order is part of the contract (design §6.2): the dtype
// code first, because the item size the overflow check needs comes from
// it; then the size; then the overflow; then fault injection; then the
// allocation. The untyped wrappers know their dtype by construction, so
// **their** first observable failure is still ``size <= 0`` with the
// identical message — existing behavior is preserved to the letter.
void* create_storage_typed(int64_t size, int32_t dtype_code,
                           bool zero_initialize) {
    Dtype dtype;
    if (!tf::dtype_from_code(dtype_code, dtype)) {
        char message[128];
        std::snprintf(message, sizeof message,
                      "unknown dtype code %d; supported codes are 0 (float64) "
                      "and 1 (float32)",
                      static_cast<int>(dtype_code));
        tf::set_error(TF_ERROR_INVALID, message);
        return nullptr;
    }
    return create_storage(size, dtype, zero_initialize);
}

}  // namespace

// Returns an opaque handle, or null (with the thread-local error set) if
// allocation fails. The buffer is zero-initialized for predictable
// behavior. Allocation is RAII-ordered so the failure scenario the ABI
// contract calls out — data allocated, then metadata allocation throws,
// leaking the data — cannot happen: the unique_ptr owns the buffer until
// the Storage node has successfully adopted it.
//
// **This is the default and it did not change in Phase H.** Every caller
// that has not explicitly proved its destination is fully written still
// lands here.
TF_EXPORT void* tf_storage_create(int64_t size) {
    TF_GUARD_BEGIN
    return create_storage(size, Dtype::Float64, /*zero_initialize=*/true);
    TF_GUARD_END(nullptr)
}

// The Phase-H (H1) uninitialized sibling: identical in every observable
// respect — size validation, zero/negative rejection, fault injection,
// allocation-failure handling, error state, handle shape, ownership,
// destruction through ``tf_storage_destroy``, and live-storage
// accounting — except that the buffer's initial contents are
// **indeterminate**.
//
// It exists for exactly one reason: a destination that a kernel
// completely overwrites before reading pays a full extra write pass for
// a zero nobody ever observes. Callers must have proved that property
// per kernel; the audit table lives in
// docs/native_cpu_performance_design.md. It is an internal backend
// detail — no public ``empty`` API is built on it, and the Python
// wrapper exposes it only through a private helper.
//
// This is the **only** C ABI symbol milestone H1 added.
TF_EXPORT void* tf_storage_create_uninitialized(int64_t size) {
    TF_GUARD_BEGIN
    return create_storage(size, Dtype::Float64, /*zero_initialize=*/false);
    TF_GUARD_END(nullptr)
}

// ---------------------------------------------------------------------------
// Phase I, milestone I1: the two typed creators.
//
// These are the **only** two production symbols Phase I adds (52 -> 54),
// and they exist for the one reason the rest of the ABI does not need to
// grow: the dtype travels with the data, and a handle already identifies a
// storage that carries its own tag — so every handle-based export already
// has everything it needs. Construction is the single moment at which the
// dtype is not yet knowable from any argument, and two constructors close
// that gap. Per-operation float32 symbols are rejected outright
// (design §6.5).
//
// Both report failure exactly the way ``tf_storage_create`` reports it —
// a null handle plus a thread-local status code — and both take the
// identical errcheck hook on the Python side, so no second failure
// convention is introduced.
//
// Creating float32 storage through these is **not** the same as float32
// being a supported TensorForge dtype: the public registry still lists
// float32 as unsupported and moves only at milestone I9. Every operation
// that has not been generalized rejects a float32 handle (see
// ``tf::require_float64``).
// ---------------------------------------------------------------------------

TF_EXPORT void* tf_storage_create_typed(int64_t size, int32_t dtype_code) {
    TF_GUARD_BEGIN
    return create_storage_typed(size, dtype_code, /*zero_initialize=*/true);
    TF_GUARD_END(nullptr)
}

TF_EXPORT void* tf_storage_create_uninitialized_typed(int64_t size,
                                                      int32_t dtype_code) {
    TF_GUARD_BEGIN
    return create_storage_typed(size, dtype_code, /*zero_initialize=*/false);
    TF_GUARD_END(nullptr)
}

// Safe to call with null (does nothing), and must be called exactly once
// per successful create — the Python wrapper guarantees that.
//
// Destruction is the mirror of creation: the dtype tag that selected the
// array type selects the matching ``delete[]``. That switch lives in
// exactly one place (``destroy_storage_data``), so no caller duplicates
// it and no future call site can pick a deleter that disagrees with the
// allocation.
//
// **No per-element destruction, and that is a property rather than an
// assumption.** The elements are ``float`` or ``double``, whose
// destructors are trivial — ``create_typed_storage`` ``static_assert``s
// exactly that beside the allocation it licenses — so ``delete[]`` runs
// no destructor pass and releasing the array is the whole of it. A dtype
// whose elements needed destruction could not be added without that
// assertion firing.
TF_EXPORT void tf_storage_destroy(void* handle) {
    if (handle == nullptr) {
        return;
    }
    Storage* storage = as_storage(handle);
    destroy_storage_data(storage);
    delete storage;
}

// Dtype-neutral: still a **logical element count**, for every dtype. It
// does not report bytes, and there is deliberately no size-in-bytes
// export — the byte size is one multiplication away from values the
// caller already has.
TF_EXPORT int64_t tf_storage_size(const void* handle) {
    return as_storage(handle)->size;
}

// ---------------------------------------------------------------------------
// The float64 storage primitives.
//
// I1 does not generalize any of these — that is I2 (transfer) and later —
// so each rejects a float32 handle before reading or writing a single
// element. Without the check, a float32 buffer walked as ``double`` would
// be overrun by exactly a factor of two.
//
// The four unguarded ones below (fill, scale, copy_from, copy_to) keep
// their existing error-contract classification: they do **not** clear the
// thread-local slot on entry and they do **not** carry the Python
// errcheck hook. That is deliberate. H7 kept them hookless precisely so
// they cost one native call rather than two, and no production Python
// path can reach them with a float32 handle — the Python wrapper cannot
// construct float32 storage while the public registry still rejects the
// dtype. A direct C ABI caller, which is the only way to get one, reads
// the rejection through ``tf_last_error_code`` in the ordinary way for an
// unhooked export. A failed call here leaves TF_ERROR_INVALID in the slot;
// the next guarded call clears it on entry, so it can never be
// misattributed to a later checked call.
// ---------------------------------------------------------------------------

TF_EXPORT void tf_storage_fill(void* handle, double value) {
    if (!tf::require_float64("tf_storage_fill", {handle})) {
        return;
    }
    Storage* storage = as_storage(handle);
    double* data = tf::storage_f64(handle);
    for (int64_t i = 0; i < storage->size; ++i) {
        data[i] = value;
    }
}

// Multiply every element by a scalar factor, in place — a small storage
// primitive alongside fill, used by the mean reduction to scale a freshly
// summed output by 1/count in float64.
TF_EXPORT void tf_storage_scale(void* handle, double factor) {
    if (!tf::require_float64("tf_storage_scale", {handle})) {
        return;
    }
    Storage* storage = as_storage(handle);
    double* data = tf::storage_f64(handle);
    for (int64_t i = 0; i < storage->size; ++i) {
        data[i] *= factor;
    }
}

TF_EXPORT void tf_storage_copy_from(void* handle, const double* src) {
    if (!tf::require_float64("tf_storage_copy_from", {handle})) {
        return;
    }
    Storage* storage = as_storage(handle);
    double* data = tf::storage_f64(handle);
    for (int64_t i = 0; i < storage->size; ++i) {
        data[i] = src[i];
    }
}

TF_EXPORT void tf_storage_copy_to(const void* handle, double* dst) {
    if (!tf::require_float64("tf_storage_copy_to", {handle})) {
        return;
    }
    const Storage* storage = as_storage(handle);
    const double* data = tf::storage_f64(handle);
    for (int64_t i = 0; i < storage->size; ++i) {
        dst[i] = data[i];
    }
}

// Materialize a strided view of the storage into a contiguous output
// buffer — the canonical tensor-runtime loop. The logical indices are
// walked like an odometer (last dimension fastest, i.e. row-major output
// order) while the source position is updated incrementally: stepping
// dimension d moves by strides[d], and when a dimension wraps its full
// extent is subtracted before the next dimension steps. Bounds are
// validated on the Python side before this is called.
TF_EXPORT void tf_storage_materialize(
    const void* handle, double* dst,
    const int64_t* shape, const int64_t* strides,
    int64_t offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_float64("tf_storage_materialize", {handle})) {
        return;
    }
    const double* src = tf::storage_f64(handle);
    if (ndim == 0) {  // scalar view: one element at the offset
        dst[0] = src[offset];
        return;
    }
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
    }
    std::vector<int64_t> counter = tf::make_counter(ndim);
    int64_t src_pos = offset;
    for (int64_t out = 0; out < total; ++out) {
        dst[out] = src[src_pos];
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            src_pos += strides[d];
            if (counter[d] < shape[d]) {
                break;  // this dimension advanced; done
            }
            counter[d] = 0;  // wrap and carry into the next dimension
            src_pos -= shape[d] * strides[d];
        }
    }
    TF_GUARD_END_VOID()
}
