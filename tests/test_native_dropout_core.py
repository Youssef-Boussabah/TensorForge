"""Deterministic stateless native Dropout-forward Core (Phase G, G2).

G2 ships the internal ``"tensorforge.splitmix64"`` bit derivation, the
inverted-Dropout CPU float64 kernel, the exception-safe C ABI wrapper
(``tf_core_dropout_forward``), the ctypes/errcheck registration, and the
two ``NativeTensorCore`` entry points — the public
``dropout_forward(p, *, seed, call_index)`` and the private
``_dropout_forward_with_mask`` that keeps the multiplier mask a future
backward will consume.

These tests cover the **committed known-answer vectors** (the same
constants the native CTest asserts, so both sides pin one stream), the
probability matrix, the random-key validation, logical-layout
independence, the output/mask ownership and lifetime contract, two-result
failure atomicity in both C++ and Python, the raw-ABI validation surface,
and the strict separation from ``NativeGenerator``: **the Core is
stateless and never touches a generator**.

What G2 deliberately does *not* ship is asserted here too: no
``NativeTensor.dropout``, no ``NativeDropout``, no backward kernel, no
checkpoint version 2, and ``"dropout"`` still in ``UNSUPPORTED``.

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close(); nothing depends on
garbage-collection timing.

Selector: python -m pytest -q -k native_dropout_core
"""

import math
from fractions import Fraction

import numpy as np
import pytest

from tensorforge.backends import cpp

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)


@pytest.fixture(autouse=True)
def _disarm_after_each():
    """Leave the injection hook disarmed and the error slot clear after
    every test, so an armed countdown never leaks into the next one."""
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


@pytest.fixture
def live_storages(monkeypatch):
    """A set of the ids of every NativeStorage currently open.

    Wrapping the storage constructor and ``close()`` gives a real
    live-native-allocation count, so a failure test can prove the count
    returns to its baseline instead of trusting garbage collection."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)  # raises => never recorded
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    monkeypatch.setattr(cpp.NativeStorage, "__init__", tracked_init)
    monkeypatch.setattr(cpp.NativeStorage, "close", tracked_close)
    return open_ids


# --------------------------------------------------------------------------
# The committed known-answer vectors (docs/native_rng_dropout_design.md
# §4.7).
#
# These literals ARE the specification. They were computed once from the
# locked algorithm and are asserted identically here and in
# cpp/tests/test_dropout_forward.cpp, so a change to any constant, shift,
# multiplication order, key derivation, bits-to-uniform conversion,
# comparison direction, or index use fails on both sides rather than
# silently redefining the stream.
#
# A test-only Python reference implementation of §4.2-§4.4 follows them
# (the design explicitly permits one in the test suite and forbids one in
# production). It is the *secondary* oracle: it generates expectations for
# arbitrary shapes, and its own agreement with the hardcoded vectors is
# asserted first, so it can never quietly redefine what "correct" means.
# --------------------------------------------------------------------------

UINT64_MAX = 2 ** 64 - 1
GOLDEN = 0x9E3779B97F4A7C15
# The largest call index a NativeGenerator can ever issue (design §4.6:
# `calls` is a count, so 2**64 - 1 is a reachable count and 2**64 - 2 is
# the last usable index).
MAX_ISSUED_CALL_INDEX = UINT64_MAX - 1

MIX64_VECTORS = (
    (0x0000000000000000, 0x0000000000000000),
    (0x0000000000000001, 0x5692161D100B05E5),
    (0x0000000000000002, 0xDBD238973A2B148A),
    (0x9E3779B97F4A7C15, 0xE220A8397B1DCDAF),
    (0x8000000000000000, 0x25C26EA579CEA98A),
    (0xFFFFFFFFFFFFFFFF, 0xB4D055FCF2CBBD7B),
)

STREAM_VECTORS = (
    (0x0000000000000000, 0, 0xE220A8397B1DCDAF),
    (0x0000000000000000, 1, 0x6E789E6AA1B965F4),
    (0x0000000000000000, 2, 0x06C45D188009454F),
    (0x0123456789ABCDEF, 0, 0x157A3807A48FAA9D),
    (0x0123456789ABCDEF, 7, 0x8931545F4F9EA651),
    (0x8000000000000000, 0, 0x481EC0A212A9F3DB),
    (0xFFFFFFFFFFFFFFFF, 0, 0xE4D971771B652C20),
    (0x0000000000000000, MAX_ISSUED_CALL_INDEX, 0x336503C6B835BEC0),
    (0x0123456789ABCDEF, MAX_ISSUED_CALL_INDEX, 0x20BEC7299668A13F),
)

# name -> (seed, call_index, p, first four element-bit words, keep pattern
# over twelve logical elements). Every keep pattern below is distinct, so
# a case cannot pass by accidentally matching a neighbour.
DROPOUT_VECTORS = {
    "zero_seed_call0": (
        0x0000000000000000, 0, 0.25,
        (0xA706DD2F4D197E6F, 0xB382A305F4414F5E,
         0x631A9154FBABF717, 0xA80ABA8C86640906),
        "111110111110",
    ),
    "zero_seed_call1": (
        0x0000000000000000, 1, 0.25,
        (0x46B73E79F0C37C00, 0x374327C63D0CC8A6,
         0xE10CF86AE3079278, 0x26A223C360B54F32),
        "101011111011",
    ),
    "mixed_seed_call0": (
        0x0123456789ABCDEF, 0, 0.25,
        (0x021C88D0A3FD73B6, 0x498D3E51E781CDE0,
         0xA2A1796FEB7EF314, 0x1A2D33D4F57B4CD4),
        "011011111010",
    ),
    "mixed_seed_call7": (
        0x0123456789ABCDEF, 7, 0.75,
        (0x0184F08818982A99, 0x99E0A20D1E1F1641,
         0x3E9AD5FC011194F1, 0x52E464BC2FB3BF83),
        "000010000000",
    ),
    "high_bit_seed_call3": (
        0x8000000000000000, 3, 0.75,
        (0x94E05B24F614999E, 0xD58EE1DBADEF970D,
         0xE932E5239EC1F7C9, 0xB01B43DD212F69A7),
        "011000100000",
    ),
    "max_seed_call0": (
        0xFFFFFFFFFFFFFFFF, 0, 0.25,
        (0x5DC20AA7B2A27137, 0xBDA5668A01D7049C,
         0x82B43276ABB80226, 0xED4D5ED4A6EA59B4),
        "111110110110",
    ),
    "zero_seed_max_call": (
        0x0000000000000000, MAX_ISSUED_CALL_INDEX, 0.75,
        (0x53531EEB39C4C095, 0x1EACB2A4329B0259,
         0x2402CC7044E8B298, 0xAAB3D73BF633B046),
        "000001100001",
    ),
}

VECTOR_LENGTH = 12

# --------------------------------------------------------------------------
# The equality-threshold vector (design §4.7)
#
# Everything above pins the bit path; this pins the **comparison**. The
# locked rule is ``drop = u < p``, strictly — so an element whose uniform
# value is exactly ``p`` is KEPT, and the very next representable
# probability drops it. The p = 0.25 / p = 0.75 vectors cannot see that:
# no committed word converts to either value, so replacing ``<`` with
# ``<=`` would reproduce every one of those patterns unchanged.
#
# This vector is the one place the two rules disagree. The word is already
# committed as DROPOUT_VECTORS["mixed_seed_call0"][3][2] — the same seed,
# call index, and logical element index — so nothing new enters the
# stream; only the probability is chosen to land exactly on it.
#
# The identical constants are asserted in cpp/tests/test_dropout_forward.cpp.
EQUALITY_SEED = 0x0123456789ABCDEF
EQUALITY_CALL_INDEX = 0
EQUALITY_INDEX = 2
EQUALITY_WORD = 0xA2A1796FEB7EF314
# (EQUALITY_WORD >> 11) * 2**-53, written as a hexadecimal float so no
# decimal parse can round it. Decimal value 0.635276403259464.
EQUALITY_UNIFORM = float.fromhex("0x1.4542f2dfd6fdep-1")
EQUALITY_COUNT = 4
# Keep patterns over the first four elements of that stream:
#   * at p == u        the strict `<` rule KEEPS element 2 ....... "0010"
#   * at p == u        the rejected `<=` rule would drop it ...... "0000"
#   * at nextafter(u)  the strict `<` rule drops it too .......... "0000"
EQUALITY_KEEP_AT_EQUAL = "0010"
EQUALITY_KEEP_AT_NEXT = "0000"
# Distinct, nonzero, mixed-sign values, so `output == input * mask` is a
# real check at every position rather than an accident of zeros.
EQUALITY_INPUT = np.array([1.5, -2.25, 3.75, -4.5])


# -- the test-only reference (design §4.7; never production code) ----------

def reference_mix64(x):
    x &= UINT64_MAX
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & UINT64_MAX
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & UINT64_MAX
    x ^= x >> 31
    return x


def reference_stream(seed, call_index):
    return reference_mix64((seed + GOLDEN * ((call_index + 1) & UINT64_MAX))
                           & UINT64_MAX)


def reference_bits(seed, call_index, element):
    return reference_mix64(
        (reference_stream(seed, call_index)
         + GOLDEN * ((element + 1) & UINT64_MAX)) & UINT64_MAX
    )


def reference_uniform(bits):
    return (bits >> 11) * 2.0 ** -53


def reference_mask(shape, p, seed, call_index):
    """The multiplier mask the kernel must produce, as a NumPy array of
    ``shape`` in logical row-major order."""
    count = int(np.prod(shape, dtype=object)) if shape else 1
    scale = 1.0 / (1.0 - p)
    values = [
        0.0 if reference_uniform(reference_bits(seed, call_index, i)) < p
        else scale
        for i in range(count)
    ]
    return np.array(values, dtype=np.float64).reshape(shape)


# -- helpers ---------------------------------------------------------------

def _core(values):
    """A contiguous NativeTensorCore holding a copy of ``values``.

    ``from_array`` cannot express rank 0 (``np.ascontiguousarray``
    promotes a 0-d array to shape ``(1,)``), so a scalar goes through
    ``zeros(())`` plus a storage copy — the repository's scalar
    convention."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        core = cpp.NativeTensorCore.zeros(())
        core._storage.copy_from(array.reshape(1))
        return core
    return cpp.NativeTensorCore.from_array(array)


def _forward(values, p, seed, call_index):
    """Run the private Core forward over fresh storage and return
    ``(output, mask)`` as NumPy arrays, closing every native object."""
    x = _core(values)
    try:
        out, mask = x._dropout_forward_with_mask(
            p, seed=seed, call_index=call_index
        )
        try:
            return out.to_numpy(), mask.to_numpy()
        finally:
            out.close()
            mask.close()
    finally:
        x.close()


# --------------------------------------------------------------------------
# The algorithm: committed vectors, on both sides
# --------------------------------------------------------------------------

def test_reference_agrees_with_the_committed_mix64_vectors():
    """The secondary oracle is pinned to the primary one first: the
    test-only reference must reproduce the hardcoded finalizer outputs
    before it is allowed to generate any expectation."""
    for value, expected in MIX64_VECTORS:
        assert reference_mix64(value) == expected, hex(value)


def test_reference_agrees_with_the_committed_stream_vectors():
    for seed, call_index, expected in STREAM_VECTORS:
        assert reference_stream(seed, call_index) == expected, (seed,
                                                                call_index)
    # The `+ 1` in the derivation: stream(seed, 0) is mix64(seed + GOLDEN),
    # never mix64(seed).
    assert reference_stream(0, 0) == reference_mix64(GOLDEN)
    assert reference_stream(0, 0) != reference_mix64(0)


def test_reference_agrees_with_the_committed_element_bits():
    for name, (seed, call_index, _p, bits, _keep) in DROPOUT_VECTORS.items():
        for index, expected in enumerate(bits):
            assert reference_bits(seed, call_index, index) == expected, (
                name, index
            )


def test_bits_to_uniform_conversion_is_exact():
    assert reference_uniform(0) == 0.0
    # All ones -> (2**53 - 1) * 2**-53, the largest value strictly below 1.
    assert reference_uniform(UINT64_MAX) == (2 ** 53 - 1) / 2 ** 53
    assert reference_uniform(UINT64_MAX) < 1.0
    # The low 11 bits are discarded and cannot move u at all.
    assert reference_uniform(0x7FF) == 0.0
    assert reference_uniform(0x800) == 2.0 ** -53


@pytest.mark.parametrize("name", sorted(DROPOUT_VECTORS))
def test_kernel_reproduces_the_committed_mask_and_output(name):
    """The load-bearing test of the milestone: the native kernel's mask
    and output must equal the committed keep/drop pattern **exactly**,
    element by element, for every vector — including the zero seed, a
    mixed seed, a high-bit seed, the all-ones seed, call index 0, a
    nonzero call index, and the highest index a generator can issue."""
    seed, call_index, p, _bits, keep = DROPOUT_VECTORS[name]
    values = np.arange(VECTOR_LENGTH, dtype=np.float64) + 0.5
    out, mask = _forward(values, p, seed, call_index)

    scale = 1.0 / (1.0 - p)
    expected_mask = np.array(
        [scale if flag == "1" else 0.0 for flag in keep], dtype=np.float64
    )
    assert np.array_equal(mask, expected_mask)
    assert np.array_equal(out, values * expected_mask)
    # ...and the mask holds exactly the two locked values, nothing else.
    assert set(np.unique(mask)) <= {0.0, scale}


def test_kernel_agrees_with_the_reference_over_a_larger_sample():
    """Beyond the twelve-element vectors, the kernel must agree with the
    reference derivation over a wider sample and a third probability."""
    values = np.linspace(-3.0, 3.0, 97)
    out, mask = _forward(values, 0.4, seed=0xDEADBEEFCAFEF00D, call_index=13)
    expected = reference_mask((97,), 0.4, 0xDEADBEEFCAFEF00D, 13)
    assert np.array_equal(mask, expected)
    assert np.array_equal(out, values * expected)


def _keep_pattern(mask, scale):
    """A produced mask rendered as a keep/drop string: ``"1"`` where the
    element carries the inverted scale, ``"0"`` where it is exactly
    ``0.0``. Anything else becomes ``"?"``, so a wrong multiplier shows up
    in the comparison instead of being rounded away."""
    out = []
    for value in np.asarray(mask).ravel():
        if value == 0.0:
            out.append("0")
        elif value == scale:
            out.append("1")
        else:
            out.append("?")
    return "".join(out)


def test_equality_vector_is_self_consistent():
    """The committed word, its uniform value, and the committed
    twelve-element vector it came from must all agree before the boundary
    test is allowed to mean anything."""
    assert (DROPOUT_VECTORS["mixed_seed_call0"][0] == EQUALITY_SEED
            and DROPOUT_VECTORS["mixed_seed_call0"][1] == EQUALITY_CALL_INDEX)
    assert DROPOUT_VECTORS["mixed_seed_call0"][3][EQUALITY_INDEX] == (
        EQUALITY_WORD
    )
    assert reference_bits(
        EQUALITY_SEED, EQUALITY_CALL_INDEX, EQUALITY_INDEX
    ) == EQUALITY_WORD
    # The uniform value is exactly the locked 53-bit conversion of it...
    assert reference_uniform(EQUALITY_WORD) == EQUALITY_UNIFORM
    assert (EQUALITY_WORD >> 11) * 2.0 ** -53 == EQUALITY_UNIFORM
    # ...and it lies strictly inside (0, 1), so both p == u and the next
    # representable probability above it are legal.
    assert 0.0 < EQUALITY_UNIFORM < 1.0
    assert EQUALITY_UNIFORM.hex() == "0x1.4542f2dfd6fdep-1"


def test_equality_threshold_element_is_kept_by_the_core():
    """Case 1 of the boundary proof: at ``p == u`` the strict ``<`` rule
    **keeps** the element on the threshold.

    This runs the real Core path — the production kernel behind
    ``_dropout_forward_with_mask`` — not a duplicated comparison helper."""
    p = EQUALITY_UNIFORM
    scale = 1.0 / (1.0 - p)
    out, mask = _forward(
        EQUALITY_INPUT, p, EQUALITY_SEED, EQUALITY_CALL_INDEX
    )
    assert _keep_pattern(mask, scale) == EQUALITY_KEEP_AT_EQUAL
    # The threshold element specifically: kept, with the inverted scale.
    assert mask[EQUALITY_INDEX] == scale
    assert out[EQUALITY_INDEX] == EQUALITY_INPUT[EQUALITY_INDEX] * scale
    # ...and the whole result still obeys output == input * mask.
    expected_mask = np.array(
        [scale if index == EQUALITY_INDEX else 0.0
         for index in range(EQUALITY_COUNT)]
    )
    assert np.array_equal(mask, expected_mask)
    assert np.array_equal(out, EQUALITY_INPUT * expected_mask)


def test_next_probability_above_the_threshold_drops_the_element():
    """Case 2 of the boundary proof: at ``nextafter(u, 1.0)`` — the very
    next representable double above the element's uniform value — the same
    element is **dropped**, with a multiplier of exactly ``0.0``."""
    p = math.nextafter(EQUALITY_UNIFORM, 1.0)
    assert p > EQUALITY_UNIFORM and p < 1.0
    scale = 1.0 / (1.0 - p)
    out, mask = _forward(
        EQUALITY_INPUT, p, EQUALITY_SEED, EQUALITY_CALL_INDEX
    )
    assert _keep_pattern(mask, scale) == EQUALITY_KEEP_AT_NEXT
    assert mask[EQUALITY_INDEX] == 0.0
    assert np.array_equal(mask, np.zeros(EQUALITY_COUNT))
    assert np.array_equal(out, EQUALITY_INPUT * mask)


def test_equality_threshold_boundary_through_the_public_core_and_raw_abi():
    """The same two cases through the public Core entry point and through
    the raw C ABI, so the comparison is pinned at every layer the Core
    actually goes through — not only at the private helper."""
    p_equal = EQUALITY_UNIFORM
    p_next = math.nextafter(EQUALITY_UNIFORM, 1.0)
    scale = 1.0 / (1.0 - p_equal)

    # -- the public Core forward (which discards the mask) --
    x = _core(EQUALITY_INPUT)
    try:
        kept = x.dropout_forward(
            p_equal, seed=EQUALITY_SEED, call_index=EQUALITY_CALL_INDEX
        )
        dropped = x.dropout_forward(
            p_next, seed=EQUALITY_SEED, call_index=EQUALITY_CALL_INDEX
        )
        try:
            assert kept.to_numpy()[EQUALITY_INDEX] == (
                EQUALITY_INPUT[EQUALITY_INDEX] * scale
            )
            assert np.array_equal(dropped.to_numpy(),
                                  np.zeros(EQUALITY_COUNT))
        finally:
            kept.close()
            dropped.close()
    finally:
        x.close()

    # -- the raw exported kernel --
    for p, pattern in ((p_equal, EQUALITY_KEEP_AT_EQUAL),
                       (p_next, EQUALITY_KEEP_AT_NEXT)):
        source = cpp.NativeStorage(EQUALITY_COUNT)
        source.copy_from(EQUALITY_INPUT)
        output = cpp.NativeStorage(EQUALITY_COUNT)
        mask = cpp.NativeStorage(EQUALITY_COUNT)
        try:
            _raw_call(source, 0, output, mask, EQUALITY_COUNT,
                      EQUALITY_SEED, EQUALITY_CALL_INDEX, p)
            assert _keep_pattern(mask.to_numpy(), 1.0 / (1.0 - p)) == pattern
            assert np.array_equal(
                output.to_numpy(), EQUALITY_INPUT * mask.to_numpy()
            )
        finally:
            source.close()
            output.close()
            mask.close()


def test_equality_vector_discriminates_strict_less_than_from_less_equal():
    """Negative control for the boundary proof.

    An equality vector is only worth committing if it **discriminates**:
    it must give a different answer under the rejected ``<=`` rule than
    under the locked ``<`` rule. This computes what a ``<=`` kernel would
    have produced from the same derivation — only the comparison differs —
    and proves three things:

    1. the ``<=`` rule drops the threshold element (pattern ``"0000"``),
    2. the production kernel keeps it (pattern ``"0010"``), and
    3. the ``<=`` result fails the very assertion
       ``test_equality_threshold_element_is_kept_by_the_core`` makes.

    So if the production comparison were ever changed to ``<=``, that test
    would fail rather than silently pass — which is exactly what the
    p = 0.25 / p = 0.75 vectors could not guarantee."""
    p = EQUALITY_UNIFORM
    scale = 1.0 / (1.0 - p)

    # What a `<=` kernel would produce: same derivation, one operator
    # changed.
    rejected_mask = np.array([
        0.0 if reference_uniform(
            reference_bits(EQUALITY_SEED, EQUALITY_CALL_INDEX, index)
        ) <= p else scale
        for index in range(EQUALITY_COUNT)
    ])
    assert _keep_pattern(rejected_mask, scale) == EQUALITY_KEEP_AT_NEXT

    # What the production kernel actually produces.
    _out, produced_mask = _forward(
        EQUALITY_INPUT, p, EQUALITY_SEED, EQUALITY_CALL_INDEX
    )
    assert _keep_pattern(produced_mask, scale) == EQUALITY_KEEP_AT_EQUAL

    # They disagree — the vector is discriminating, not vacuous.
    assert not np.array_equal(rejected_mask, produced_mask)
    # ...and the `<=` result fails the boundary assertion the Core passes.
    assert _keep_pattern(rejected_mask, scale) != EQUALITY_KEEP_AT_EQUAL
    assert rejected_mask[EQUALITY_INDEX] != scale
    assert produced_mask[EQUALITY_INDEX] == scale


def test_keep_rate_is_statistically_sane():
    """Not a distribution test — a tripwire. A stream whose keep rate is
    far from ``1 - p`` over ten thousand draws is broken in a way the
    fixed vectors might not localize."""
    values = np.ones(10_000)
    _out, mask = _forward(values, 0.3, seed=12345, call_index=1)
    keep_rate = float((mask != 0.0).mean())
    assert 0.66 < keep_rate < 0.74, keep_rate


# --------------------------------------------------------------------------
# Determinism and stream separation
# --------------------------------------------------------------------------

def test_same_key_reproduces_mask_and_output_exactly():
    values = np.arange(24, dtype=np.float64) - 8.0
    first_out, first_mask = _forward(values, 0.5, seed=7, call_index=3)
    second_out, second_mask = _forward(values, 0.5, seed=7, call_index=3)
    assert np.array_equal(first_mask, second_mask)
    assert np.array_equal(first_out, second_out)


def test_changed_call_index_uses_the_committed_pattern():
    """Two committed vectors share a seed and differ only in the call
    index, so the expected difference is itself a committed constant
    rather than "something changed"."""
    values = np.arange(VECTOR_LENGTH, dtype=np.float64) + 1.0
    _out0, mask0 = _forward(values, 0.25, seed=0, call_index=0)
    _out1, mask1 = _forward(values, 0.25, seed=0, call_index=1)
    scale = 1.0 / 0.75
    assert np.array_equal(
        mask0,
        np.array([scale if f == "1" else 0.0
                  for f in DROPOUT_VECTORS["zero_seed_call0"][4]]),
    )
    assert np.array_equal(
        mask1,
        np.array([scale if f == "1" else 0.0
                  for f in DROPOUT_VECTORS["zero_seed_call1"][4]]),
    )
    assert not np.array_equal(mask0, mask1)


def test_changed_seed_uses_the_committed_pattern():
    """Same structure for the seed: call index 0 at p = 0.25 is committed
    for the zero seed, a mixed seed, and the all-ones seed."""
    values = np.arange(VECTOR_LENGTH, dtype=np.float64) + 1.0
    scale = 1.0 / 0.75
    masks = {}
    for name in ("zero_seed_call0", "mixed_seed_call0", "max_seed_call0"):
        seed, call_index, p, _bits, keep = DROPOUT_VECTORS[name]
        _out, mask = _forward(values, p, seed, call_index)
        assert np.array_equal(
            mask,
            np.array([scale if f == "1" else 0.0 for f in keep]),
        )
        masks[name] = mask
    assert not np.array_equal(masks["zero_seed_call0"],
                              masks["mixed_seed_call0"])
    assert not np.array_equal(masks["zero_seed_call0"],
                              masks["max_seed_call0"])


def test_adjacent_call_indices_use_different_streams():
    """A wide sample, because a thresholded mask is one bit per element:
    two *different* streams can agree over a short tensor by chance (at
    p = 0.5, seed 11 with call indices 4 and 5 agree over eight elements).
    The element bits are what actually differ, and the committed bit
    vectors pin those; this asserts the visible consequence over enough
    elements for an accidental agreement not to be a realistic outcome."""
    values = np.ones(64)
    _a, mask_a = _forward(values, 0.5, seed=11, call_index=4)
    _b, mask_b = _forward(values, 0.5, seed=11, call_index=5)
    assert not np.array_equal(mask_a, mask_b)
    assert reference_bits(11, 4, 0) != reference_bits(11, 5, 0)


def test_the_mask_ignores_the_input_values():
    """The random decision must not depend on the data: two completely
    different inputs of the same logical shape get the identical mask."""
    key = dict(seed=0x0123456789ABCDEF, call_index=7)
    _a, mask_a = _forward(np.zeros(VECTOR_LENGTH), 0.75, **key)
    _b, mask_b = _forward(
        np.array([1e300, -1e-300, np.inf, -np.inf, np.nan, 0.0,
                  -0.0, 1.0, -1.0, 2.5, -2.5, 7.0]),
        0.75, **key,
    )
    assert np.array_equal(mask_a, mask_b)


def test_no_numpy_or_python_rng_is_consulted():
    """The native draw is a pure function of the key. Seeding NumPy and
    Python's ``random`` differently around two identical calls must change
    nothing, and neither global stream may be advanced by the call."""
    import random

    values = np.arange(16, dtype=np.float64)

    np.random.seed(1234)
    random.seed(1234)
    numpy_before = np.random.get_state()[2]
    python_before = random.getstate()
    _out_a, mask_a = _forward(values, 0.5, seed=99, call_index=2)
    assert np.random.get_state()[2] == numpy_before
    assert random.getstate() == python_before

    np.random.seed(4321)
    random.seed(4321)
    _out_b, mask_b = _forward(values, 0.5, seed=99, call_index=2)
    assert np.array_equal(mask_a, mask_b)


def test_public_core_forward_matches_the_private_helper():
    """``dropout_forward`` is exactly ``_dropout_forward_with_mask``
    without the mask — same values, same key, one fewer result."""
    values = np.arange(VECTOR_LENGTH, dtype=np.float64) + 0.5
    x = _core(values)
    try:
        public = x.dropout_forward(0.25, seed=0, call_index=0)
        try:
            private_out, private_mask = x._dropout_forward_with_mask(
                0.25, seed=0, call_index=0
            )
            try:
                assert np.array_equal(public.to_numpy(),
                                      private_out.to_numpy())
            finally:
                private_out.close()
                private_mask.close()
        finally:
            public.close()
    finally:
        x.close()


def test_public_core_forward_releases_the_mask(live_storages):
    """The public entry keeps no private state alive: after it returns,
    exactly one new storage (the output) is open."""
    x = _core(np.arange(8.0))
    baseline = len(live_storages)
    out = x.dropout_forward(0.5, seed=3, call_index=0)
    assert len(live_storages) == baseline + 1
    out.close()
    x.close()
    assert len(live_storages) == baseline - 1


# --------------------------------------------------------------------------
# The probability contract (design §6.1)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.0, 0.001, 0.25, 0.5, 0.75, 0.9999])
def test_valid_probabilities_are_accepted(p):
    values = np.arange(20, dtype=np.float64) + 1.0
    out, mask = _forward(values, p, seed=5, call_index=1)
    scale = 1.0 / (1.0 - p)
    assert set(np.unique(mask)) <= {0.0, scale}
    assert np.array_equal(out, values * mask)


def test_p_zero_keeps_everything_at_the_core_layer():
    """The identity **bypass** (returning the input object, allocating
    nothing, consuming no call) belongs to the operation layer, G3
    (design §6.2). At the Core the kernel is still asked to compute, and
    it keeps every element with a multiplier of exactly 1.0."""
    values = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    out, mask = _forward(values, 0.0, seed=42, call_index=9)
    assert np.array_equal(mask, np.ones(5))
    assert np.array_equal(out, values)


def test_integer_zero_is_accepted_and_normalized():
    values = np.arange(6, dtype=np.float64)
    out, mask = _forward(values, 0, seed=1, call_index=0)
    assert np.array_equal(mask, np.ones(6))
    assert np.array_equal(out, values)


@pytest.mark.parametrize("p", [1, 1.0, 1.5, 2, -0.25, -1, -0.0000001])
def test_out_of_range_probabilities_are_rejected(p):
    x = _core(np.arange(4.0))
    try:
        with pytest.raises(ValueError, match="0 <= p < 1"):
            x._dropout_forward_with_mask(p, seed=0, call_index=0)
    finally:
        x.close()


def test_nan_probability_is_rejected_by_name():
    x = _core(np.arange(4.0))
    try:
        with pytest.raises(ValueError, match="NaN"):
            x._dropout_forward_with_mask(float("nan"), seed=0, call_index=0)
    finally:
        x.close()


@pytest.mark.parametrize("p", [float("inf"), float("-inf")])
def test_infinite_probability_is_rejected(p):
    x = _core(np.arange(4.0))
    try:
        with pytest.raises(ValueError, match="finite"):
            x._dropout_forward_with_mask(p, seed=0, call_index=0)
    finally:
        x.close()


@pytest.mark.parametrize("p", [True, False, np.bool_(True), np.bool_(False)])
def test_bool_probability_is_rejected(p):
    """``True`` must never sail through as ``1.0`` and ``False`` never as
    ``0.0``: a bool is not a probability."""
    x = _core(np.arange(4.0))
    try:
        with pytest.raises(TypeError, match="bool"):
            x._dropout_forward_with_mask(p, seed=0, call_index=0)
    finally:
        x.close()


@pytest.mark.parametrize(
    "p", ["0.5", None, [0.5], (0.5,), {"p": 0.5}, complex(0.5, 0.0), object()]
)
def test_non_real_probability_is_rejected(p):
    x = _core(np.arange(4.0))
    try:
        with pytest.raises(TypeError, match="real number"):
            x._dropout_forward_with_mask(p, seed=0, call_index=0)
    finally:
        x.close()


@pytest.mark.parametrize(
    "p", [np.float64(0.25), np.float32(0.25), Fraction(1, 4)]
)
def test_real_scalar_types_are_accepted_and_normalized(p):
    """``numbers.Real`` is the accepted abstract type, so a NumPy scalar
    or a Fraction is normalized with ``float(p)`` — the same latitude the
    stable Dropout gives — and produces the committed quarter-probability
    pattern."""
    values = np.arange(VECTOR_LENGTH, dtype=np.float64) + 0.5
    _out, mask = _forward(values, p, seed=0, call_index=0)
    scale = 1.0 / 0.75
    expected = np.array(
        [scale if f == "1" else 0.0
         for f in DROPOUT_VECTORS["zero_seed_call0"][4]]
    )
    assert np.array_equal(mask, expected)


def test_rejected_probability_leaves_live_storage_unchanged(live_storages):
    x = _core(np.arange(9.0).reshape(3, 3))
    baseline = len(live_storages)
    for bad in (1.0, -0.5, float("nan"), float("inf"), True, "0.5"):
        with pytest.raises((TypeError, ValueError)):
            x._dropout_forward_with_mask(bad, seed=0, call_index=0)
        assert len(live_storages) == baseline
    x.close()


# --------------------------------------------------------------------------
# The random key
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seed", [0, 1, 2 ** 32, 2 ** 63, UINT64_MAX, MAX_ISSUED_CALL_INDEX]
)
def test_full_uint64_seed_range_is_accepted(seed):
    values = np.arange(5, dtype=np.float64)
    _out, mask = _forward(values, 0.5, seed=seed, call_index=0)
    assert np.array_equal(mask, reference_mask((5,), 0.5, seed, 0))


@pytest.mark.parametrize(
    "call_index", [0, 1, 2 ** 32, MAX_ISSUED_CALL_INDEX, UINT64_MAX]
)
def test_full_uint64_call_index_range_is_accepted(call_index):
    """The key space is the full ``uint64`` the ctypes argument carries. A
    ``NativeGenerator`` never *issues* an index above ``2**64 - 2``, but
    that is its counter rule, not a property of this stateless Core."""
    values = np.arange(5, dtype=np.float64)
    _out, mask = _forward(values, 0.5, seed=17, call_index=call_index)
    assert np.array_equal(mask, reference_mask((5,), 0.5, 17, call_index))


@pytest.mark.parametrize("field", ["seed", "call_index"])
@pytest.mark.parametrize(
    "value", [True, False, 0.0, 1.5, "7", None, np.int64(7), np.uint64(7)]
)
def test_non_exact_int_key_fields_are_rejected(field, value):
    """Exact-``int`` discipline, matching ``NativeGenerator``: a bool is
    not a seed and a NumPy integer scalar is not a Python int."""
    key = {"seed": 0, "call_index": 0}
    key[field] = value
    x = _core(np.arange(4.0))
    try:
        with pytest.raises(TypeError, match=field):
            x._dropout_forward_with_mask(0.5, **key)
    finally:
        x.close()


@pytest.mark.parametrize("field", ["seed", "call_index"])
@pytest.mark.parametrize("value", [-1, 2 ** 64, 2 ** 70])
def test_out_of_range_key_fields_are_rejected(field, value):
    key = {"seed": 0, "call_index": 0}
    key[field] = value
    x = _core(np.arange(4.0))
    try:
        with pytest.raises(ValueError, match=field):
            x._dropout_forward_with_mask(0.5, **key)
    finally:
        x.close()


def test_key_is_keyword_only():
    x = _core(np.arange(4.0))
    try:
        with pytest.raises(TypeError):
            x._dropout_forward_with_mask(0.5, 1, 2)
        with pytest.raises(TypeError):
            x.dropout_forward(0.5)  # seed / call_index are required
    finally:
        x.close()


def test_rejected_key_leaves_live_storage_unchanged(live_storages):
    x = _core(np.arange(6.0))
    baseline = len(live_storages)
    for key in ({"seed": -1, "call_index": 0}, {"seed": 0, "call_index": 2 ** 64},
                {"seed": True, "call_index": 0}, {"seed": 0, "call_index": 1.0}):
        with pytest.raises((TypeError, ValueError)):
            x._dropout_forward_with_mask(0.5, **key)
        assert len(live_storages) == baseline
    x.close()


# --------------------------------------------------------------------------
# Shape, rank, and layout
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "shape",
    [(), (1,), (5,), (2, 3), (2, 3, 4), (2, 1, 3, 2), (2, 2, 2, 2, 2)],
)
def test_arbitrary_rank_including_scalars(shape):
    """Every rank the tensor representation supports, 0-d included: the
    output and mask keep the input's shape, stay contiguous, and match the
    logical row-major reference."""
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    values = (np.arange(count, dtype=np.float64) + 1.0).reshape(shape)
    x = _core(values)
    try:
        out, mask = x._dropout_forward_with_mask(0.4, seed=21, call_index=6)
        try:
            assert out.shape == shape
            assert mask.shape == shape
            assert out.contiguous and mask.contiguous
            assert out.dtype == mask.dtype == "float64"
            assert out.device == mask.device == "cpu"
            expected = reference_mask(shape, 0.4, 21, 6)
            assert np.array_equal(mask.to_numpy(), expected)
            assert np.array_equal(out.to_numpy(), values * expected)
        finally:
            out.close()
            mask.close()
    finally:
        x.close()


def test_scalar_shape_stays_scalar():
    x = _core(np.float64(3.5))
    try:
        assert x.shape == ()
        out, mask = x._dropout_forward_with_mask(0.25, seed=0, call_index=0)
        try:
            assert out.shape == () and mask.shape == ()
            # Element 0 of the committed zero-seed/call-0 vector is kept.
            assert mask.to_numpy().reshape(()) == 1.0 / 0.75
            assert out.to_numpy().reshape(()) == 3.5 * (1.0 / 0.75)
        finally:
            out.close()
            mask.close()
    finally:
        x.close()


def test_the_representation_cannot_express_an_empty_tensor():
    """The design's empty-tensor row (§6.4, §7.3) is not reachable through
    the Core today: the native tensor representation rejects zero-size
    dimensions outright, so no empty core exists to hand in. This pins
    that reality rather than pretending the case was exercised — the
    kernel and the C ABI *do* accept a count of 0, which
    ``test_raw_abi_accepts_a_zero_count`` proves at the layer where it is
    reachable."""
    with pytest.raises(ValueError, match="positive"):
        cpp.NativeTensorCore.zeros((0,))
    with pytest.raises(ValueError, match="positive"):
        cpp.NativeTensorCore.zeros((2, 0, 3))


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_transposed_view_gets_the_same_logical_mask(shape):
    """Logical-layout independence, the locked property of §7.3: a
    transposed view and a contiguous tensor of the **same logical shape**
    receive the same mask, because Policy B materializes the view into
    row-major storage before the kernel runs."""
    count = int(np.prod(shape, dtype=np.int64))
    base = (np.arange(count, dtype=np.float64) + 1.0).reshape(shape)
    transposed_values = base.T
    contiguous = _core(np.ascontiguousarray(transposed_values))
    strided_source = _core(base)
    strided = strided_source.T
    try:
        assert not strided.contiguous
        assert strided.shape == contiguous.shape

        out_c, mask_c = contiguous._dropout_forward_with_mask(
            0.5, seed=8, call_index=2
        )
        out_s, mask_s = strided._dropout_forward_with_mask(
            0.5, seed=8, call_index=2
        )
        try:
            # The mask is identical by logical position...
            assert np.array_equal(mask_s.to_numpy(), mask_c.to_numpy())
            assert mask_s.contiguous and out_s.contiguous
            # ...and each output is its own values times that mask.
            assert np.array_equal(
                out_s.to_numpy(), transposed_values * mask_s.to_numpy()
            )
            assert np.array_equal(
                out_c.to_numpy(),
                np.ascontiguousarray(transposed_values) * mask_c.to_numpy(),
            )
        finally:
            out_c.close()
            mask_c.close()
            out_s.close()
            mask_s.close()
    finally:
        strided.close()
        strided_source.close()
        contiguous.close()


def test_nonzero_offset_view_gets_the_same_logical_mask():
    """A narrowed view sits at a nonzero storage offset; the mask must
    still be keyed by logical position, not by physical offset."""
    base = _core(np.arange(20.0).reshape(4, 5))
    narrowed = base.narrow(0, 1, 2)          # rows 1-2, offset 5
    independent = _core(np.arange(20.0).reshape(4, 5)[1:3])
    try:
        assert narrowed.offset != 0
        assert narrowed.shape == independent.shape == (2, 5)
        out_n, mask_n = narrowed._dropout_forward_with_mask(
            0.5, seed=4, call_index=1
        )
        out_i, mask_i = independent._dropout_forward_with_mask(
            0.5, seed=4, call_index=1
        )
        try:
            assert np.array_equal(mask_n.to_numpy(), mask_i.to_numpy())
            assert np.array_equal(out_n.to_numpy(), out_i.to_numpy())
            assert np.array_equal(
                mask_n.to_numpy(), reference_mask((2, 5), 0.5, 4, 1)
            )
        finally:
            out_n.close()
            mask_n.close()
            out_i.close()
            mask_i.close()
    finally:
        narrowed.close()
        independent.close()
        base.close()


def test_mask_is_independent_of_how_the_view_was_built():
    """Three different construction histories, one logical shape, one
    mask: reshape, transpose-of-transpose, and a plain tensor."""
    values = np.arange(12.0).reshape(3, 4)
    plain = _core(values)
    reshaped_source = _core(values.ravel())
    reshaped = reshaped_source.reshape((3, 4))
    double_t_source = _core(values)
    double_t = double_t_source.T.T
    try:
        masks = []
        for core in (plain, reshaped, double_t):
            out, mask = core._dropout_forward_with_mask(
                0.6, seed=77, call_index=5
            )
            masks.append(mask.to_numpy())
            out.close()
            mask.close()
        assert np.array_equal(masks[0], masks[1])
        assert np.array_equal(masks[0], masks[2])
    finally:
        double_t.close()
        double_t_source.close()
        reshaped.close()
        reshaped_source.close()
        plain.close()


def test_input_and_its_metadata_are_never_mutated():
    values = np.arange(12.0).reshape(3, 4) - 5.0
    x = _core(values)
    before = (x.shape, x.strides, x.offset, x.ndim, x.numel, x.contiguous,
              x.dtype, x.device)
    try:
        out, mask = x._dropout_forward_with_mask(0.5, seed=2, call_index=2)
        out.close()
        mask.close()
        assert np.array_equal(x.to_numpy(), values)
        assert (x.shape, x.strides, x.offset, x.ndim, x.numel, x.contiguous,
                x.dtype, x.device) == before
    finally:
        x.close()


def test_results_alias_neither_the_input_nor_each_other():
    x = _core(np.arange(6.0) + 1.0)
    try:
        out, mask = x._dropout_forward_with_mask(0.5, seed=1, call_index=1)
        try:
            assert out._storage is not x._storage
            assert mask._storage is not x._storage
            assert out._storage is not mask._storage
            assert out._owns_storage and mask._owns_storage
            # Writing through one result cannot be seen through another.
            original_out = out.to_numpy().copy()
            mask._storage.fill(-1.0)
            assert np.array_equal(out.to_numpy(), original_out)
            assert np.array_equal(x.to_numpy(), np.arange(6.0) + 1.0)
        finally:
            out.close()
            mask.close()
    finally:
        x.close()


def test_unsupported_dtype_or_device_is_rejected(live_storages):
    """Metadata is validated before anything is allocated. The runtime has
    exactly one legal dtype/device pair, so this is driven by forcing the
    tags rather than by constructing an unsupported tensor."""
    x = _core(np.arange(4.0))
    baseline = len(live_storages)
    try:
        x._storage._dtype = "float32"
        with pytest.raises(ValueError, match="float64/cpu"):
            x._dropout_forward_with_mask(0.5, seed=0, call_index=0)
        assert len(live_storages) == baseline
        x._storage._dtype = "float64"
        x._storage._device = "cuda"
        with pytest.raises(ValueError, match="float64/cpu"):
            x._dropout_forward_with_mask(0.5, seed=0, call_index=0)
        assert len(live_storages) == baseline
    finally:
        x._storage._dtype = "float64"
        x._storage._device = "cpu"
        x.close()


def test_closed_input_is_rejected(live_storages):
    x = _core(np.arange(4.0))
    x.close()
    baseline = len(live_storages)
    with pytest.raises(RuntimeError):
        x._dropout_forward_with_mask(0.5, seed=0, call_index=0)
    with pytest.raises(RuntimeError):
        x.dropout_forward(0.5, seed=0, call_index=0)
    assert len(live_storages) == baseline


# --------------------------------------------------------------------------
# Ownership, lifetime, and cleanup
# --------------------------------------------------------------------------

def test_results_are_independently_owning_and_close_in_any_order():
    for order in ("output_first", "mask_first"):
        x = _core(np.arange(8.0))
        out, mask = x._dropout_forward_with_mask(0.5, seed=6, call_index=0)
        first, second = (out, mask) if order == "output_first" else (mask, out)
        first.close()
        # The other result is untouched and still readable.
        assert second.to_numpy().shape == (8,)
        second.close()
        # ...and closing twice is safe in either order (no double release).
        first.close()
        second.close()
        out.close()
        mask.close()
        x.close()


def test_closing_the_input_first_leaves_the_results_valid():
    """The results own their own storage, so they outlive the input."""
    x = _core(np.arange(8.0) + 1.0)
    out, mask = x._dropout_forward_with_mask(0.5, seed=6, call_index=0)
    x.close()
    try:
        assert out.to_numpy().shape == (8,)
        assert mask.to_numpy().shape == (8,)
        assert np.array_equal(
            mask.to_numpy(), reference_mask((8,), 0.5, 6, 0)
        )
    finally:
        out.close()
        mask.close()


def test_live_storage_returns_to_baseline_after_success(live_storages):
    x = _core(np.arange(30.0).reshape(5, 6))
    baseline = len(live_storages)
    out, mask = x._dropout_forward_with_mask(0.5, seed=0, call_index=0)
    assert len(live_storages) == baseline + 2
    out.close()
    mask.close()
    assert len(live_storages) == baseline
    x.close()
    assert len(live_storages) == baseline - 1


def test_noncontiguous_policy_b_temporary_is_released(live_storages):
    """The private contiguous copy is closed the moment the native call
    returns, so a strided forward leaves exactly the two results open."""
    source = _core(np.arange(12.0).reshape(3, 4))
    strided = source.T
    baseline = len(live_storages)
    out, mask = strided._dropout_forward_with_mask(0.5, seed=0, call_index=0)
    assert len(live_storages) == baseline + 2   # output + mask, no temp left
    out.close()
    mask.close()
    assert len(live_storages) == baseline
    strided.close()
    source.close()


@needs_fault_injection
def test_output_allocation_failure_leaves_nothing_allocated(live_storages):
    """The first of the two allocations fails: no output, no mask, no
    partial result, and the input untouched."""
    values = np.arange(10.0)
    x = _core(values)
    baseline = len(live_storages)
    with pytest.raises(MemoryError):
        cpp._arm_alloc_failure(1)
        x._dropout_forward_with_mask(0.5, seed=0, call_index=0)
    assert len(live_storages) == baseline
    assert np.array_equal(x.to_numpy(), values)
    # ...and the next call succeeds normally.
    out, mask = x._dropout_forward_with_mask(0.5, seed=0, call_index=0)
    out.close()
    mask.close()
    assert len(live_storages) == baseline
    x.close()


@needs_fault_injection
def test_mask_allocation_failure_closes_the_output(live_storages):
    """The second allocation fails after the first succeeded: the output
    is released, so live storage returns exactly to its baseline and no
    caller can observe a lone output."""
    values = np.arange(10.0)
    x = _core(values)
    baseline = len(live_storages)
    with pytest.raises(MemoryError):
        cpp._arm_alloc_failure(2)
        x._dropout_forward_with_mask(0.5, seed=0, call_index=0)
    assert len(live_storages) == baseline
    assert np.array_equal(x.to_numpy(), values)
    x.close()


@needs_fault_injection
def test_repeated_allocation_failures_do_not_accumulate(live_storages):
    x = _core(np.arange(10.0))
    baseline = len(live_storages)
    for nth in (1, 2) * 5:
        with pytest.raises(MemoryError):
            cpp._arm_alloc_failure(nth)
            x._dropout_forward_with_mask(0.5, seed=0, call_index=0)
        assert len(live_storages) == baseline
    x.close()


def test_wrapper_construction_failure_closes_the_first_result(
    live_storages, monkeypatch
):
    """The Python half of the two-result atomicity rule: if the second
    result object fails to be constructed, the first must still be closed.
    No caller may observe only one successful result."""
    original = cpp.NativeTensorCore.zeros
    calls = {"n": 0}

    def failing_zeros(shape, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected wrapper failure")
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(cpp.NativeTensorCore, "zeros",
                        failing_zeros)
    # H1: the enabled output-allocation sites construct through
    # _uninitialized, so the same probe must watch both
    # constructors for this test to still observe the real path.
    monkeypatch.setattr(cpp.NativeTensorCore, "_uninitialized",
                        failing_zeros)
    x = _core(np.arange(10.0))
    baseline = len(live_storages)
    with pytest.raises(RuntimeError, match="injected wrapper failure"):
        x._dropout_forward_with_mask(0.5, seed=0, call_index=0)
    assert len(live_storages) == baseline
    x.close()


def test_native_call_failure_closes_both_results(live_storages, monkeypatch):
    """A failure *after* both allocations — here the native call itself —
    releases both, so nothing partial escapes and no storage leaks."""
    library = cpp._require_library()
    original = library.tf_core_dropout_forward

    def failing_call(*args, **kwargs):
        raise RuntimeError("injected native failure")

    monkeypatch.setattr(library, "tf_core_dropout_forward", failing_call)
    x = _core(np.arange(10.0))
    baseline = len(live_storages)
    with pytest.raises(RuntimeError, match="injected native failure"):
        x._dropout_forward_with_mask(0.5, seed=0, call_index=0)
    assert len(live_storages) == baseline
    monkeypatch.setattr(library, "tf_core_dropout_forward", original)
    x.close()


def test_public_forward_failure_leaves_nothing_allocated(
    live_storages, monkeypatch
):
    """``dropout_forward`` closes the mask on the way out; a failure
    before that point must still leave the count at baseline."""
    original = cpp.NativeTensorCore.zeros
    calls = {"n": 0}

    def failing_zeros(shape, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected wrapper failure")
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(cpp.NativeTensorCore, "zeros",
                        failing_zeros)
    # H1: the enabled output-allocation sites construct through
    # _uninitialized, so the same probe must watch both
    # constructors for this test to still observe the real path.
    monkeypatch.setattr(cpp.NativeTensorCore, "_uninitialized",
                        failing_zeros)
    x = _core(np.arange(6.0))
    baseline = len(live_storages)
    with pytest.raises(RuntimeError, match="injected wrapper failure"):
        x.dropout_forward(0.5, seed=0, call_index=0)
    assert len(live_storages) == baseline
    x.close()


# --------------------------------------------------------------------------
# The raw C ABI surface
# --------------------------------------------------------------------------

def _raw_call(input_storage, input_offset, output_storage, mask_storage,
              count, seed, call_index, p):
    """Drive the exported kernel directly, bypassing the Core layer."""
    library = cpp._require_library()
    library.tf_core_dropout_forward(
        input_storage._require_open(), input_offset,
        output_storage._require_open(), mask_storage._require_open(),
        count, seed, call_index, p,
    )


def _raw_fixture(size=6):
    source = cpp.NativeStorage(size)
    source.copy_from(np.arange(size, dtype=np.float64) + 1.0)
    output = cpp.NativeStorage(size)
    output.fill(-99.0)
    mask = cpp.NativeStorage(size)
    mask.fill(-99.0)
    return source, output, mask


def test_raw_abi_accepts_a_zero_count():
    """``count == 0`` is legal at the kernel and the ABI: no draw, no
    write. This is the layer where the design's empty-tensor row is
    reachable — the Python tensor representation cannot build a
    zero-element core (see
    ``test_the_representation_cannot_express_an_empty_tensor``)."""
    source, output, mask = _raw_fixture()
    try:
        _raw_call(source, 0, output, mask, 0, 1, 1, 0.5)
        assert np.array_equal(output.to_numpy(), np.full(6, -99.0))
        assert np.array_equal(mask.to_numpy(), np.full(6, -99.0))
        assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    finally:
        source.close()
        output.close()
        mask.close()


def test_raw_abi_matches_the_core_layer():
    source, output, mask = _raw_fixture()
    try:
        _raw_call(source, 0, output, mask, 6, 0, 0, 0.25)
        expected = reference_mask((6,), 0.25, 0, 0)
        assert np.array_equal(mask.to_numpy(), expected)
        assert np.array_equal(
            output.to_numpy(), (np.arange(6.0) + 1.0) * expected
        )
    finally:
        source.close()
        output.close()
        mask.close()


@pytest.mark.parametrize(
    "case",
    [
        ("negative count", dict(count=-1)),
        ("negative offset", dict(input_offset=-1)),
        ("input span overruns", dict(input_offset=3, count=6)),
        ("p == 1", dict(p=1.0)),
        ("p > 1", dict(p=1.5)),
        ("p < 0", dict(p=-0.25)),
        ("p is NaN", dict(p=float("nan"))),
        ("p is +inf", dict(p=float("inf"))),
        ("p is -inf", dict(p=float("-inf"))),
    ],
    ids=lambda case: case[0] if isinstance(case, tuple) else str(case),
)
def test_raw_abi_rejects_invalid_arguments_without_writing(case):
    """A rejecting kernel leaves **both** destinations byte-for-byte
    unchanged — the Phase-E self-validating export contract."""
    _label, overrides = case
    source, output, mask = _raw_fixture()
    arguments = dict(input_offset=0, count=6, seed=1, call_index=1, p=0.5)
    arguments.update(overrides)
    try:
        with pytest.raises(ValueError, match="dropout_forward"):
            _raw_call(source, arguments["input_offset"], output, mask,
                      arguments["count"], arguments["seed"],
                      arguments["call_index"], arguments["p"])
        assert np.array_equal(output.to_numpy(), np.full(6, -99.0))
        assert np.array_equal(mask.to_numpy(), np.full(6, -99.0))
        assert np.array_equal(source.to_numpy(), np.arange(6.0) + 1.0)
    finally:
        source.close()
        output.close()
        mask.close()


def test_raw_abi_rejects_undersized_destinations():
    source = cpp.NativeStorage(6)
    source.copy_from(np.arange(6.0))
    small = cpp.NativeStorage(3)
    small.fill(-99.0)
    big = cpp.NativeStorage(6)
    big.fill(-99.0)
    try:
        with pytest.raises(ValueError, match="output storage"):
            _raw_call(source, 0, small, big, 6, 1, 1, 0.5)
        assert np.array_equal(small.to_numpy(), np.full(3, -99.0))
        with pytest.raises(ValueError, match="mask storage"):
            _raw_call(source, 0, big, small, 6, 1, 1, 0.5)
        assert np.array_equal(big.to_numpy(), np.full(6, -99.0))
    finally:
        source.close()
        small.close()
        big.close()


def test_raw_abi_rejects_null_handles():
    library = cpp._require_library()
    source, output, mask = _raw_fixture()
    try:
        for arguments in (
            (None, 0, output._require_open(), mask._require_open()),
            (source._require_open(), 0, None, mask._require_open()),
            (source._require_open(), 0, output._require_open(), None),
        ):
            with pytest.raises(ValueError, match="null required storage"):
                library.tf_core_dropout_forward(
                    arguments[0], arguments[1], arguments[2], arguments[3],
                    6, 1, 1, 0.5,
                )
        assert np.array_equal(output.to_numpy(), np.full(6, -99.0))
        assert np.array_equal(mask.to_numpy(), np.full(6, -99.0))
    finally:
        source.close()
        output.close()
        mask.close()


def test_raw_abi_rejects_aliasing_destinations():
    source, output, _unused = _raw_fixture()
    try:
        _unused.close()
        with pytest.raises(ValueError, match="aliases"):
            _raw_call(source, 0, source, output, 6, 1, 1, 0.5)
        with pytest.raises(ValueError, match="aliases"):
            _raw_call(source, 0, output, source, 6, 1, 1, 0.5)
        with pytest.raises(ValueError, match="aliases"):
            _raw_call(source, 0, output, output, 6, 1, 1, 0.5)
        assert np.array_equal(source.to_numpy(), np.arange(6.0) + 1.0)
        assert np.array_equal(output.to_numpy(), np.full(6, -99.0))
    finally:
        source.close()
        output.close()


def test_raw_abi_failures_leave_live_storage_unchanged(live_storages):
    source, output, mask = _raw_fixture()
    baseline = len(live_storages)
    try:
        for _ in range(5):
            with pytest.raises(ValueError):
                _raw_call(source, 0, output, mask, 6, 1, 1, 1.0)
            assert len(live_storages) == baseline
    finally:
        source.close()
        output.close()
        mask.close()
    assert len(live_storages) == baseline - 3


def test_raw_abi_clears_the_error_slot_on_a_later_success():
    source, output, mask = _raw_fixture()
    try:
        with pytest.raises(ValueError):
            _raw_call(source, 0, output, mask, 6, 1, 1, 2.0)
        _raw_call(source, 0, output, mask, 6, 1, 1, 0.5)
        assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    finally:
        source.close()
        output.close()
        mask.close()


# --------------------------------------------------------------------------
# Separation: G2 is stateless, and nothing later has shipped
# --------------------------------------------------------------------------

def test_core_calls_never_touch_a_generator():
    """The Core takes explicit integers, not a generator. Running many
    forwards must leave a live ``NativeGenerator`` bit-identical: same
    seed, same ``calls``, and no reservation created."""
    from tensorforge.experimental import NativeGenerator

    generator = NativeGenerator(seed=1234)
    before = generator.state()
    x = _core(np.arange(10.0))
    try:
        for call_index in range(5):
            out, mask = x._dropout_forward_with_mask(
                0.5, seed=generator.seed, call_index=call_index
            )
            out.close()
            mask.close()
            x.dropout_forward(0.5, seed=generator.seed,
                              call_index=call_index).close()
        assert generator.state() == before
        assert generator.calls == 0
        assert generator._has_active_reservation() is False
    finally:
        x.close()


def test_core_rejects_a_generator_object_as_a_key_field():
    """Passing a generator where an integer belongs is a plain TypeError —
    there is no hidden acceptance path that would read its state."""
    from tensorforge.experimental import NativeGenerator

    generator = NativeGenerator(seed=5)
    x = _core(np.arange(4.0))
    try:
        with pytest.raises(TypeError, match="seed"):
            x._dropout_forward_with_mask(0.5, seed=generator, call_index=0)
        assert generator.calls == 0
    finally:
        x.close()


def test_failed_core_calls_leave_a_generator_untouched():
    from tensorforge.experimental import NativeGenerator

    generator = NativeGenerator(seed=99)
    before = generator.state()
    x = _core(np.arange(4.0))
    try:
        for bad in ({"p": 1.0}, {"p": float("nan")}, {"seed": -1},
                    {"call_index": 2 ** 64}):
            arguments = {"p": 0.5, "seed": 0, "call_index": 0}
            arguments.update(bad)
            with pytest.raises((TypeError, ValueError)):
                x._dropout_forward_with_mask(
                    arguments["p"], seed=arguments["seed"],
                    call_index=arguments["call_index"],
                )
        assert generator.state() == before
        assert generator._has_active_reservation() is False
    finally:
        x.close()


def test_g2_ships_the_core_layer_and_nothing_above_it():
    """The Core-layer boundary, from the live tree: the Core forward is
    layer-qualified and stateless, the module does not exist, the
    checkpoint format has not moved, and ``"dropout"`` is still
    unsupported.

    Milestone **G3** since shipped the differentiable ``dropout``
    operation *above* this Core, which is a different capability at a
    different layer — the same Core/operation split conv2d, maxpool2d,
    and cross_entropy already follow. It is covered by
    tests/test_native_dropout_autograd.py; what this file guards is that
    the Core itself gained nothing when that happened."""
    import tensorforge
    import tensorforge.experimental as experimental
    from tensorforge.experimental import NativeTensor, native_checkpoint

    # Shipped at the Core layer, under the layer-qualified name.
    assert "dropout_forward" in cpp.TENSOR_CORE_OPS
    assert hasattr(cpp.NativeTensorCore, "dropout_forward")
    assert "tf_core_dropout_forward" in cpp._CHECKED_KERNELS
    assert cpp.backend_info()["tensor_core_ops"] == cpp.TENSOR_CORE_OPS
    # The Core did not acquire the operation's name, and the operation
    # did not acquire a Core entry: they stay layer-qualified apart.
    assert "dropout" not in cpp.TENSOR_CORE_OPS
    assert not hasattr(cpp.NativeTensorCore, "dropout")
    assert "dropout_forward" not in cpp.AUTOGRAD_OPS
    assert not hasattr(NativeTensor, "dropout_forward")

    # The G4 module exists two layers up and did not reach down into the
    # Core either: it is a module name and nothing else.
    assert hasattr(experimental, "NativeDropout")
    assert "NativeDropout" in cpp.NATIVE_MODULES
    assert not hasattr(tensorforge, "NativeDropout")
    assert "NativeDropout" not in cpp.TENSOR_CORE_OPS
    assert not hasattr(cpp.NativeTensorCore, "NativeDropout")

    # Not shipped: a backward kernel, a generic RNG.
    assert "dropout_backward" not in cpp.TENSOR_CORE_OPS
    assert not hasattr(cpp.NativeTensorCore, "dropout_backward")
    for absent in ("random", "rand", "randn", "bernoulli", "uniform"):
        assert absent not in cpp.TENSOR_CORE_OPS, absent
        assert absent not in cpp.AUTOGRAD_OPS, absent
        assert not hasattr(cpp.NativeTensorCore, absent), absent
    for symbol in ("tf_core_dropout_backward", "tf_core_random",
                   "tf_core_bernoulli", "tf_core_uniform"):
        assert symbol not in cpp._CHECKED_KERNELS, symbol

    # The capability boundary and the format version are untouched.
    assert cpp.UNSUPPORTED == ("float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert native_checkpoint._FORMAT_VERSION == 2
    assert "generator_state" in cpp.STATE_SUPPORT


def test_stable_dropout_is_untouched():
    """Phase G changes no stable file: the stable Dropout still uses the
    NumPy RNG and is unaffected by anything the native Core does."""
    from tensorforge import Dropout, Tensor

    layer = Dropout(0.5)
    np.random.seed(20260725)
    reference = layer(Tensor(np.ones((4, 5)))).data

    # A native Core call in between must change nothing about it.
    _out, _mask = _forward(np.ones(20), 0.5, seed=1, call_index=1)
    np.random.seed(20260725)
    again = layer(Tensor(np.ones((4, 5)))).data
    assert np.array_equal(reference, again)
    assert not hasattr(layer, "generator")


def _code_only(text):
    """``text`` with ``//`` comment tails removed.

    The sources *name* the forbidden constructs in their commentary — that
    is the point of the commentary — so a scan of the raw file would
    always fail. Stripping comments makes the check about what the
    translation unit actually compiles."""
    lines = []
    for line in text.splitlines():
        marker = line.find("//")
        lines.append(line if marker < 0 else line[:marker])
    return "\n".join(lines)


def test_no_global_or_static_random_state_exists_in_the_runtime():
    """Derived from the sources, not from prose: the random translation
    unit declares no static/global mutable state and consults no
    standard-library or system entropy source."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    source = _code_only(
        (repo_root / "cpp" / "src" / "random.cpp").read_text(encoding="utf-8")
    )
    header = _code_only(
        (repo_root / "cpp" / "include" / "tf_random_internal.h").read_text(
            encoding="utf-8"
        )
    )
    for forbidden in ("<random>", "random_device", "mt19937", "thread_local",
                      "static ", "std::time", "getpid", "rand()", "srand"):
        assert forbidden not in source, (forbidden, "random.cpp")
        assert forbidden not in header, (forbidden, "tf_random_internal.h")
    # Every declaration in the header is inside namespace tf (hidden), and
    # the one exported symbol is the guarded wrapper.
    assert source.count("TF_EXPORT") == 1
    assert "tf_core_dropout_forward" in source
    # The kernel signature carries the whole key explicitly.
    assert "std::uint64_t seed" in header
    assert "std::uint64_t call_index" in header
