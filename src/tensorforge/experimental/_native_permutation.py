"""The deterministic sample-order derivation (Phase J, milestone J2; see
docs/native_data_pipeline_design.md §8).

**Permanently private.** Nothing here is exported, imported by the stable
root package, or added to ``tensorforge.experimental.__all__``. A public
bit-generation surface beside ``NativeGenerator`` is exactly what §20
forbids, and every helper below returns raw 64-bit values, so the whole
module stays behind the underscore — the established place for a private
helper that must not become an API, beside ``_native_dtype.py``,
``_native_state.py``, and ``_native_state_lock.py``.

What this is
------------

**One algorithm, reused, not a new one.** These are the *same*
``tensorforge.splitmix64`` finalizer, the *same* golden constant, the
*same* shifts, the *same* multiplication order, and the *same* wrapping
64-bit arithmetic that ``cpp/src/random.cpp`` and
``cpp/include/tf_random_internal.h`` have shipped since Phase G milestone
G2, written in Python. Only the **key schedule** differs, because this
consumer wants permutation indices rather than a keep/drop mask:
``epoch_key`` adds one domain-separating constant that Dropout's
``dropout_stream_key`` does not (§8.4). ``tests/test_native_sampler.py``
proves the shared half really is shared, by predicting the compiled
Dropout kernel's output from these functions.

What it deliberately is not
---------------------------

There is **no generator here and no stream**. Every function is pure: a
permutation is a function of ``(seed, epoch, length)`` and nothing else,
so calling one twice returns the same tuple, calling it in a different
order changes nothing, and there is no counter, reservation, or partially
consumed sequence anywhere to roll back (§7.7). Nothing consults a clock,
entropy, an environment variable, an address, Python's ``random``,
NumPy's global RNG, a NumPy ``Generator``, the process hash seed, or any
module-level mutable state — and this module imports **no** ``ctypes``,
no ``tensorforge.backends``, no storage class, no ``NativeTensor``, and
no ``NativeGenerator``. It uses ordinary Python integer and sequence
operations, and nothing else.

Why the arithmetic looks the way it does
----------------------------------------

Python ``int`` is arbitrary precision, so **the explicit ``& MASK`` is
what makes the width exactly 64 bits** — not a C type, not a NumPy dtype,
and not platform overflow. That is why the result is bit-identical on
every platform, word size, and Python build **by construction**, which is
the property the whole phase rests on. There is no floating-point
arithmetic anywhere in this file: Dropout's ``dropout_uniform`` step is a
Dropout-only conversion and has no counterpart here.

A change to any constant, shift, multiplication order, bound rule,
rejection rule, or sweep direction below is a change to the algorithm and
requires a new ``(algorithm, algorithm_version)`` pair, exactly as the
Dropout vectors do. The committed known answers in
``tests/test_native_sampler.py`` and design §8.9 are the specification,
not a regression convenience.
"""

# The 64-bit window. Every intermediate is masked back into it explicitly.
MASK = 2**64 - 1

# The golden-ratio odd constant that separates successive keys. This is
# ``tf::kSplitMix64Golden``, to the bit.
GOLDEN = 0x9E3779B97F4A7C15

# The two SplitMix64 finalizer multipliers, in this order. These are the
# constants in ``tf::splitmix64_mix``.
MIX_MUL_1 = 0xBF58476D1CE4E5B9
MIX_MUL_2 = 0x94D049BB133111EB

# The domain separator: the ASCII bytes ``TF_SAMPL``, read as one
# big-endian 64-bit integer.
#
# Without it, ``epoch_key(seed, e)`` would be bit-identical to
# ``tf::dropout_stream_key(seed, e)``, so a caller who passed one seed to a
# sampler and to a generator would drive both from the same 64-bit
# sequence. **It is domain separation, not a cryptographic claim** (§8.4):
# a caller who deliberately chooses ``seed + SAMPLER_DOMAIN`` can still
# align the two streams. The constant prevents the accident, not the
# construction.
SAMPLER_DOMAIN = 0x54465F53414D504C


def splitmix64_mix(x):
    """The SplitMix64 finalizer: ``tf::splitmix64_mix``, in Python.

    xor-shift, multiply, xor-shift, multiply, xor-shift — with the exact
    shift amounts, the exact constants, and the exact multiplication order
    the C++ locked at G2. Pure, total, and reversible.

    The masking is not defensive tidying: C++ gets modulo-2**64 wraparound
    from ``std::uint64_t``, and Python has to ask for it. Every ``&
    MASK`` below is where that happens, and dropping any one of them would
    silently produce a different — and platform-independent but *wrong* —
    value.
    """
    x &= MASK
    x ^= x >> 30
    x = (x * MIX_MUL_1) & MASK
    x ^= x >> 27
    x = (x * MIX_MUL_2) & MASK
    x ^= x >> 31
    return x


def epoch_key(seed, epoch):
    """The per-epoch key: ``mix(seed + SAMPLER_DOMAIN + GOLDEN * (epoch + 1))``.

    One full finalizer application per epoch, matching
    ``tf::dropout_stream_key``'s shape exactly and differing from it by
    the one additive ``SAMPLER_DOMAIN`` constant.

    The ``+ 1`` keeps epoch 0 from degenerating to
    ``mix(seed + SAMPLER_DOMAIN)``, and the full finalizer — rather than a
    bare addition — is what decorrelates two epochs' orders.

    A permutation is indexed by an **epoch**, not by a call, which is the
    structural reason this cannot be a ``NativeGenerator`` call index
    (§8.3): a restored sampler must be able to reproduce epoch 9's order
    without having consumed epochs 0 through 8, and this expression lets
    it.
    """
    return splitmix64_mix((seed + SAMPLER_DOMAIN + GOLDEN * (epoch + 1)) & MASK)


def draw_bits(key, draw_index):
    """One draw's 64 random bits: ``mix(key + GOLDEN * (draw_index + 1))``.

    The second finalizer application, and the exact counterpart of
    ``tf::dropout_element_bits`` — same expression, same constant, same
    ``+ 1``. Two full applications separate the per-epoch key from the
    per-draw value, so two epochs cannot produce overlapping draw
    sequences by a simple offset.
    """
    return splitmix64_mix((key + GOLDEN * (draw_index + 1)) & MASK)


def bounded(key, draw_index, bound):
    """An **unbiased** integer in ``[0, bound)``, and the draw index after it.

    Rejection sampling, not bare modulo. ``bits % bound`` alone is biased
    whenever ``bound`` does not divide ``2**64``: the low residues would
    occur once more often than the high ones. ``limit`` is the largest
    multiple of ``bound`` that fits in ``2**64``, so every accepted value
    covers each residue **exactly** the same number of times and the bias
    is removed exactly rather than approximately.

    For any ``bound <= 2**32`` the rejection probability is below
    ``2**-32``, so in practice the loop runs once — but correctness does
    not rest on that, which is why the branch exists and why
    ``tests/test_native_sampler.py`` forces it directly at a large
    ``bound`` rather than hoping to observe one.

    **A rejected draw still advances ``draw_index``.** That is what keeps
    the whole permutation a pure function of ``(seed, epoch, length)``
    regardless of *where* a rejection lands: a rejection shifts every
    later draw by one, deterministically, on every platform.

    A multiply-shift alternative was rejected: it needs 128-bit
    intermediate reasoning to state precisely and would be a second
    bounded-integer convention beside one that is already exact.

    ``bound >= 1`` is a precondition, not a check. Every production caller
    is ``permutation`` below, which passes ``i + 1`` for ``i >= 1``, so
    the argument is structurally at least 2 and re-validating it here
    would be a private helper second-guessing its only caller.
    """
    limit = (1 << 64) - ((1 << 64) % bound)
    while True:
        bits = draw_bits(key, draw_index)
        draw_index += 1
        if bits < limit:
            return bits % bound, draw_index


def permutation(seed, epoch, length):
    """The shuffled order for one epoch, as a ``tuple`` of ``int``.

    Fisher-Yates, **downward**: ``i`` runs from ``length - 1`` to ``1``,
    ``j`` is uniform in ``[0, i]``, and ``order[i]`` and ``order[j]`` are
    swapped. The direction is part of the specification rather than an
    implementation choice — the upward variant is equally correct and
    produces *different* permutations from the same bits, so committing to
    one is what makes the reference vectors meaningful.

    Exactly ``length - 1`` draws before any rejection, and every one of
    the ``length!`` orders is reachable — including the identity, which is
    why several §8.9 reference rows *are* the identity and must never be
    "fixed" away. Excluding it would bias the sampler.

    ``length < 2`` draws **nothing at all** and returns the identity: at
    length 1 there is one order, and consuming a draw to discover that
    would make the accounting depend on a degenerate case.

    ``length`` is deliberately **not** mixed into the key. It enters
    through the bounds, so two lengths already give different orders from
    one key; and it is the *dataset identity* check (§6) that prevents a
    state from being applied to a differently sized dataset, not a
    redundant mixing step here.
    """
    order = list(range(length))
    if length < 2:
        return tuple(order)
    key = epoch_key(seed, epoch)
    draws = 0
    for i in range(length - 1, 0, -1):
        j, draws = bounded(key, draws, i + 1)
        order[i], order[j] = order[j], order[i]
    return tuple(order)


def sample_order(seed, epoch, length, shuffle):
    """The order one epoch consumes: shuffled, or the identity.

    ``shuffle=False`` is **not** "a shuffle with a fixed seed": it is a
    different, cheaper branch that consumes no derivation at all, at every
    seed and every epoch, and the reference vectors state it that way. It
    never calls ``epoch_key`` or ``draw_bits``, so a sequential sampler
    provably touches no bit path.
    """
    if shuffle:
        return permutation(seed, epoch, length)
    return tuple(range(length))


def batches_per_epoch(length, batch_size, drop_last):
    """How many batches one epoch yields. Integer arithmetic only.

    ``-(-length // batch_size)`` is ``ceil`` without floating point, which
    matters because a float division could round the wrong way for large
    counts and this value bounds the cursor.

    The result is ``0`` only when ``drop_last`` is true and ``batch_size >
    length`` — a configuration both ``NativeBatchSampler.__init__`` and
    its ``load_state_dict`` reject (§7.5), so no caller of this helper
    ever sees a zero and no code anywhere carries a zero-batch branch.
    """
    if drop_last:
        return length // batch_size
    return -(-length // batch_size)


def batch_plan(seed, epoch, length, batch_size, drop_last, shuffle):
    """The whole epoch's batches: a tuple of tuples of indices.

    Batch ``k`` is the slice ``[k * batch_size, (k + 1) * batch_size)`` of
    the epoch's order. With ``drop_last=False`` the final slice may be
    short — Python slicing truncates, which *is* the specified behavior —
    and with ``drop_last=True`` the tail is simply not enumerated.

    No index appears twice: an order is a permutation, so it contains each
    index exactly once, and a batch is a contiguous slice of it.
    """
    order = sample_order(seed, epoch, length, shuffle)
    count = batches_per_epoch(length, batch_size, drop_last)
    return tuple(order[k * batch_size:(k + 1) * batch_size]
                 for k in range(count))
