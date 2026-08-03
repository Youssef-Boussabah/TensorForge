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
// The scalar storage primitives.
//
// ``fill`` and ``scale`` were float64-only through I3 and became
// dtype-general at **I4**, because that is the milestone that needed them:
// ``scale`` *is* the mean reduction's scaling step (design §7.4), and
// ``fill`` is how a backward materializes the constants its formulas need —
// ``-1`` for a negation, ``1/count`` for mean backward — which must be
// created at the graph's dtype rather than at float64.
//
// **The scalar keeps crossing as ``double`` and its conversion is specified**
// (design §7.4). The ABI signature is unchanged: a scalar is a caller-supplied
// value in Python's only float type, and it crosses as the widest binary
// floating-point type the ABI has. The kernel converts it **once, before its
// loop**, to the storage's element type. Converting a scalar argument is not
// casting a tensor: the no-cast rule of §9 governs native-tensor-to-native-
// tensor conversion, which still never happens.
//
// Narrowing once and outside the loop is the whole numerical statement. For
// ``mean`` the factor is ``1/count``, computed once in binary64 (correctly
// rounded) by the Python layer and narrowed once here, so the result is
// deterministic, identical on every platform, and independent of ``count``'s
// magnitude. Computing ``1.0f / count`` in float32 instead would differ by up
// to one ULP for some counts; the chosen form is written down so no milestone
// can quietly pick the other. It is also why the conversion may not be left
// inside the loop, where a compiler would be free to keep the operand in
// binary64 and make every multiply a mixed-precision one.
//
// Both keep their existing error-contract classification: they do **not**
// clear the thread-local slot on entry and they do **not** carry the Python
// errcheck hook. H7 kept them hookless precisely so they cost one native call
// rather than two, and neither can fail: with every dtype now valid there is
// nothing left for either to reject, so neither writes to the error slot at
// all — which is what an unhooked export should do, and is exactly where
// ``copy_from`` and ``copy_to`` arrived at I2.
// ---------------------------------------------------------------------------

namespace {

// Assign one scalar to every element, in the element type.
template <class T>
void fill_typed(void* handle, double value) {
    const Storage* storage = as_storage(handle);
    T* data = tf::storage_typed<T>(handle);
    const T v = static_cast<T>(value);  // narrowed ONCE, before the loop
    for (int64_t i = 0; i < storage->size; ++i) {
        data[i] = v;
    }
}

// Multiply every element by one scalar, in place, in the element type. At
// ``T = double`` the local is the argument and the loop body is the pre-I4
// ``data[i] *= factor`` statement for statement.
template <class T>
void scale_typed(void* handle, double factor) {
    const Storage* storage = as_storage(handle);
    T* data = tf::storage_typed<T>(handle);
    const T f = static_cast<T>(factor);  // narrowed ONCE, before the loop
    for (int64_t i = 0; i < storage->size; ++i) {
        data[i] *= f;
    }
}

}  // namespace

TF_EXPORT void tf_storage_fill(void* handle, double value) {
    switch (tf::storage_dtype(handle)) {  // ONE dispatch, per call
        case Dtype::Float32:
            fill_typed<float>(handle, value);
            return;
        case Dtype::Float64:
            break;
    }
    fill_typed<double>(handle, value);
}

// Multiply every element by a scalar factor, in place — a small storage
// primitive alongside fill, used by the mean reduction to scale a freshly
// summed output by 1/count at the tensor's own dtype.
TF_EXPORT void tf_storage_scale(void* handle, double factor) {
    switch (tf::storage_dtype(handle)) {  // ONE dispatch, per call
        case Dtype::Float32:
            scale_typed<float>(handle, factor);
            return;
        case Dtype::Float64:
            break;
    }
    scale_typed<double>(handle, factor);
}

// ---------------------------------------------------------------------------
// Phase I, milestone I2: the three dtype-general transfer boundaries.
//
// These are the only exported functions that carry a storage handle **and**
// a raw host buffer, and I2 is the milestone that makes them work at both
// dtypes. The change to each declaration is a source-level retype of the
// host position from ``double*`` to ``void*`` and nothing else: same
// symbol, same argument count, same order, same return type, same calling
// convention, and — since a ``double*`` and a ``void*`` occupy the same
// argument slot on every supported platform and ``extern "C"`` has no
// mangling to change — the same binary interface. The export inventory
// does not grow; the phase still adds exactly the two typed creators.
//
// **The host pointer carries no dtype, and cannot be made to.** The ABI
// receives an address and nothing else, so it is structurally incapable of
// proving that the buffer behind it really holds the element type the
// storage does. The contract is therefore explicit rather than checked
// here: the **storage handle's immutable dtype tag is authoritative**, and
// the caller must supply a contiguous host buffer of exactly that element
// type — ``float`` for float32 storage, ``double`` for float64 storage.
// The Python wrapper enforces it before every call with a per-dtype
// ``numpy.ctypeslib.ndpointer`` check (element type, byte order, and
// C-contiguity), which is the layer that *can* see the buffer's type. A
// direct foreign caller is responsible for satisfying the same contract,
// exactly as it already is for the layout arrays it passes.
//
// Nothing is converted, widened, narrowed, or guessed at this boundary; no
// byte-count or dtype argument was added; the logical ``size`` stays an
// element count; and each function dispatches **once**, on the storage's
// tag, into a template. Below that dispatch there is no dtype branch at
// all — in particular none per element.
//
// **Every transfer is an assignment between two objects of the same type**,
// so it performs no arithmetic and has no operand roles to choose between.
// That is the whole reason value transfer is bit-preserving at both widths
// (design §10.3): positive and negative zero, both infinities, subnormals,
// quiet NaNs with any payload, and signalling NaNs all reproduce their
// source's object representation exactly. A same-type assignment is the
// mechanism the runtime has used for float64 since v0.8 and the one Phase
// H's copy contract rests on; ``memcpy`` is deliberately **not**
// introduced (design §4.3), because byte arithmetic outside the allocation
// boundary is precisely what this phase keeps out of the kernels.
// ---------------------------------------------------------------------------

namespace {

// Host -> storage. ``size`` elements of ``T``, in order, by assignment.
template <class T>
void copy_from_typed(void* handle, const void* src) {
    const Storage* storage = as_storage(handle);
    T* data = tf::storage_typed<T>(handle);
    const T* source = static_cast<const T*>(src);
    for (int64_t i = 0; i < storage->size; ++i) {
        data[i] = source[i];
    }
}

// Storage -> host. The mirror image, same element count, same order.
template <class T>
void copy_to_typed(const void* handle, void* dst) {
    const Storage* storage = as_storage(handle);
    const T* data = tf::storage_typed<T>(handle);
    T* destination = static_cast<T*>(dst);
    for (int64_t i = 0; i < storage->size; ++i) {
        destination[i] = data[i];
    }
}

// Strided storage -> contiguous host. The canonical tensor-runtime loop,
// unchanged in every respect but the element type: the logical indices are
// walked like an odometer (last dimension fastest, i.e. row-major output
// order) while the source position is updated incrementally — stepping
// dimension d moves by strides[d], and when a dimension wraps its full
// extent is subtracted before the next dimension steps.
//
// Shape, strides, and offset remain **logical element** counts at both
// dtypes; no byte offset is computed anywhere in this walk. Bounds are
// validated on the Python side before this is called.
template <class T>
void materialize_typed(
    const void* handle, void* dst_raw,
    const int64_t* shape, const int64_t* strides,
    int64_t offset, int64_t ndim
) {
    const T* src = tf::storage_typed<T>(handle);
    T* dst = static_cast<T*>(dst_raw);
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
}

}  // namespace

TF_EXPORT void tf_storage_copy_from(void* handle, const void* src) {
    switch (tf::storage_dtype(handle)) {  // ONE dispatch, per call
        case Dtype::Float32:
            copy_from_typed<float>(handle, src);
            return;
        case Dtype::Float64:
            break;
    }
    copy_from_typed<double>(handle, src);
}

TF_EXPORT void tf_storage_copy_to(const void* handle, void* dst) {
    switch (tf::storage_dtype(handle)) {  // ONE dispatch, per call
        case Dtype::Float32:
            copy_to_typed<float>(handle, dst);
            return;
        case Dtype::Float64:
            break;
    }
    copy_to_typed<double>(handle, dst);
}

TF_EXPORT void tf_storage_materialize(
    const void* handle, void* dst,
    const int64_t* shape, const int64_t* strides,
    int64_t offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    switch (tf::storage_dtype(handle)) {  // ONE dispatch, outside the walk
        case Dtype::Float32:
            materialize_typed<float>(handle, dst, shape, strides, offset,
                                     ndim);
            return;
        case Dtype::Float64:
            break;
    }
    materialize_typed<double>(handle, dst, shape, strides, offset, ndim);
    TF_GUARD_END_VOID()
}
