"""Experimental C++ backend.

A ctypes wrapper around the compiled native kernels (see cpp/ at the
repo root). It is NOT wired into the **stable** ``tensorforge.Tensor``
or its autograd, and ``import tensorforge`` never imports it — the
native line is a separate system. On top of these kernels the
experimental package builds its own strided runtime
(``NativeTensorCore``), a Python-managed native autograd
(``NativeTensor``), and a native training stack (modules, a loss,
optimizers, checkpoints); see ``tensorforge.experimental``.

All kernels are float64/CPU only. The raw buffer kernels exposed here
(``elementwise_add`` … ``matmul_tiled``) require identical shapes, but
the ``NativeTensorCore`` binary ops **do** support NumPy-style
broadcasting (see docs/native_broadcasting_design.md).

The library is loaded lazily and the module always imports: check
``is_available()`` for readiness and ``backend_info()`` for an accurate
capability summary (raw kernels, tensor-core ops, autograd ops, native
modules/loss/optimizers, and state/checkpoint support). Calling a math
kernel while the backend is unbuilt raises ImportError with build
instructions. Native failures surface as ordinary Python exceptions via
the error contract (docs/native_abi_error_contract.md).
"""

import ctypes
import math
import numbers
import platform
from pathlib import Path

import numpy as np

_SUFFIX = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
_LIBRARY_PATH = Path(__file__).with_name("_tensorforge_cpp" + _SUFFIX)

# ---------------------------------------------------------------------------
# Backend capability inventory — the single source of truth backend_info()
# reports. Grouped by layer so introspection can distinguish raw C++
# kernels from the higher-level capabilities composed on top of them. The
# guardrail test (tests/test_cpp_backend_info.py) cross-checks every name
# here against the actual objects, so these tuples cannot silently drift
# out of date.
# ---------------------------------------------------------------------------

# Raw C++ kernels callable directly over NumPy buffers — the reference /
# benchmark set this module exposes as elementwise_add(...) etc. Require
# identical shapes (no broadcasting at this raw level).
RAW_KERNELS = (
    "elementwise_add",
    "elementwise_subtract",
    "elementwise_multiply",
    "elementwise_divide",
    "relu",
    "matmul",
    "matmul_tiled",
)
# Backwards-compatible alias (list_kernels / backend_info["kernels"]).
KERNELS = RAW_KERNELS

# The element types the seven raw kernels above accept — **float64 only,
# and permanently** (Phase I, milestone I2; design §7.2).
#
# This is a genuinely different statement from ``SUPPORTED_DTYPES``, and I2
# is the first milestone at which the difference is observable, which is
# exactly why the registry lands here rather than at I0. The C ABI divides
# cleanly in two:
#
#   * **handle-based paths** — every export that receives a storage handle.
#     The handle carries the dtype, so the operation reads it from an
#     argument it already has. These are the paths dtype generalization
#     travels along, and from I2 the transfer, materialization, and
#     identity-copy members of the group work at float32.
#   * **the seven raw utility kernels** — ``tf_elementwise_add`` through
#     ``tf_matmul_tiled``. They receive *only* ``const double*``,
#     ``double*``, and an element count. There is no handle, therefore no
#     dtype tag, therefore nothing to dispatch on. Making them dtype-general
#     would need either new symbols (rejected — the phase adds exactly two)
#     or a dtype-code parameter, which is an argument-count change and so a
#     real ABI break.
#
# They are also not needed: they are the reference/benchmark set, and no
# ``NativeTensorCore``, ``NativeTensor``, module, loss, optimizer, or
# checkpoint path calls them. The whole native stack runs through the
# handle-based ``tf_core_*`` exports.
#
# Three distinct facts, deliberately reported separately by
# ``backend_info()`` so none of them can be mistaken for another:
# ``supported_dtypes`` (the public promise), ``raw_kernel_dtypes`` (this
# limitation), and the internal ``_DTYPE_CODES`` table (what storage can
# physically be allocated as).
RAW_KERNEL_DTYPES = ("float64",)

_BINARY_KERNELS = (
    "tf_elementwise_add",
    "tf_elementwise_subtract",
    "tf_elementwise_multiply",
    "tf_elementwise_divide",
)

# The historical, deliberately FROZEN tensor-core registry: exactly the
# five originally advertised compute ops. Kept frozen by contract (later
# core ops are not appended — the sum/mean/sqrt precedent), so existing
# consumers that pin this tuple stay stable. The complete, accurate op
# inventory lives in TENSOR_CORE_OPS below.
TENSOR_CORE_KERNELS = ("relu", "add", "subtract", "multiply", "matmul")

# The COMPLETE set of NativeTensorCore operations (the strided native
# runtime) — compute ops (each backed by C ABI kernels; binary ops
# broadcast) plus the metadata-only view ops. This is the accurate,
# non-frozen inventory backend_info() reports as "tensor_core_ops".
TENSOR_CORE_OPS = (
    "relu", "sqrt", "reciprocal",
    # Phase E stable math (E1: exp, E2: log) — unary Core ops with the
    # same odometer/contiguous pair as relu/sqrt/reciprocal. Their guarded
    # exports additionally self-validate at the ABI boundary.
    "exp", "log",
    # Phase E probability transforms (E3: softmax, E4: log_softmax) —
    # axis-wise *fused* Core ops over the contiguous-only
    # tf_core_softmax_forward / tf_core_log_softmax_forward exports, with
    # Policy-B copy-then-compute for non-contiguous inputs. Neither is
    # composed from public max/subtract/exp/sum/divide — the stability
    # transform lives inside each kernel where it cannot be bypassed — and
    # log_softmax is emphatically NOT softmax followed by log: it is its
    # own fused log-sum-exp kernel (design §4.4).
    "softmax", "log_softmax",
    "add", "subtract", "multiply", "matmul",
    "sum", "mean",
    "reshape", "transpose", "T", "narrow", "contiguous_copy",
    # Phase D native Conv2d at the Core layer — layer-qualified forward and
    # (D6) backward wrappers over the exported tf_core_conv2d_* kernels.
    # These are Core operations, distinct from the differentiable
    # "conv2d" NativeTensor autograd op (in AUTOGRAD_OPS) and from the
    # NativeConv2d module (D7, in NATIVE_MODULES). The bias gradient has no
    # Core op — it composes from the existing "sum" reduction.
    "conv2d_forward",          # D3
    "conv2d_input_backward",   # D6
    "conv2d_weight_backward",  # D6
    # Phase D native MaxPool2d at the Core layer — the layer-qualified
    # forward/backward wrappers over the exported tf_core_maxpool2d_*
    # kernels. Forward computes the pooled values and (internally) the
    # private winner buffer; backward scatters an upstream gradient through
    # those saved winners (no window geometry, no input reread). These are
    # Core operations, distinct from the differentiable "maxpool2d"
    # NativeTensor autograd op (in AUTOGRAD_OPS as of D9) and from the
    # NativeMaxPool2d module (D10, in NATIVE_MODULES). The winner buffer
    # stays internal state — never a public tensor, op, or dtype.
    "maxpool2d_forward",       # D8
    "maxpool2d_backward",      # D9
    # Phase E fused cross-entropy at the Core layer (E5) — the
    # layer-qualified, **graph-unaware** forward and backward wrappers over
    # the exported tf_core_cross_entropy_* kernels, exactly the naming the
    # E0 contract locks (design §11) and the conv2d/maxpool2d precedent.
    # The forward returns a scalar loss, the private saved probabilities,
    # the independently copied int64 targets, and the normalized reduction;
    # the backward turns those saved probabilities (never the logits) plus
    # a native one-element upstream into a fresh gradient. There is
    # deliberately **no** bare "cross_entropy" entry here: that name
    # belongs to the differentiable NativeTensor operation, which E6
    # shipped into AUTOGRAD_OPS. Core wrapper and autograd operation are
    # different capabilities with different names (design §11), exactly as
    # for conv2d and maxpool2d.
    "cross_entropy_forward",   # E5
    "cross_entropy_backward",  # E5
    # Phase G stateless Dropout at the Core layer (G2) — the
    # layer-qualified, **graph-unaware** forward wrapper over the exported
    # tf_core_dropout_forward kernel, named exactly as the G0 contract
    # locks (design §7.1) and as the conv2d/maxpool2d/cross_entropy
    # precedent requires. It takes the complete random key as explicit
    # `seed`/`call_index` integers and touches **no** NativeGenerator: the
    # generator, its counter, and the reservation transaction live in
    # Python one layer above (design §5, §7.6). The private
    # `_dropout_forward_with_mask` helper keeps the multiplier mask the
    # future backward needs; this public entry closes it, exactly the
    # maxpool2d_forward / winner-buffer split.
    #
    # There is deliberately **no** bare "dropout" entry here and no
    # "dropout_backward": that name belongs to the differentiable
    # NativeTensor operation, which G3 shipped into AUTOGRAD_OPS, and
    # inverted Dropout's gradient is the existing `multiply` over the
    # saved mask, so no backward kernel was ever written (design §7.5).
    # "dropout" as a *capability* name stayed in UNSUPPORTED through G9
    # and left it at the G10 closure (design §19).
    "dropout_forward",         # G2
)

# Operations the NativeTensor autograd layer (Phase B) differentiates.
# Phase D adds the differentiable "conv2d" fused primitive (D6): its
# backward composes the input/weight-gradient Core ops and the existing
# "sum" reduction (bias); and the differentiable "maxpool2d" primitive
# (D9), whose backward scatters through the winner buffer its own forward
# saved. These are the operations; the modules built on them (NativeConv2d,
# D7; NativeMaxPool2d, D10 — both implemented) are separate entries in
# NATIVE_MODULES. Phase E adds "exp" (E1), whose backward multiplies the
# upstream by the **saved forward output** — so, like sqrt/reciprocal, it
# records no expected parameter version — and "log" (E2), whose backward
# rereads the **live input** (`upstream * reciprocal(x)`) and therefore
# DOES record one for a direct NativeParameter. The pair is the phase's
# deliberate contrast between the two backward archetypes. "softmax" (E3)
# joins exp on the saved-output side: its backward reads only the saved
# probabilities `y` and is *composed* from existing Core ops at the
# graph-unaware level — `y * (upstream - sum(upstream * y, axis,
# keepdims=True))` — so no dedicated backward kernel exists and no
# parameter version is recorded. "log_softmax" (E4) is the same archetype
# again: its own fused log-sum-exp forward (never softmax().log()), and a
# backward composed from existing Core ops out of the saved log
# probabilities alone — `upstream - exp(y) * sum(upstream, axis,
# keepdims=True)` — so it too records no parameter version and needs no
# backward kernel. "cross_entropy" (E6) is the phase's fused loss: one
# graph node over the E5 Core contract, whose backward reads only the
# **saved probabilities** its own forward produced, the independently
# copied int64 targets, the normalized reduction, and the native scalar
# upstream — so it records no expected parameter version either, and the
# probabilities are graph-owned private state released with the graph
# history (the maxpool2d winner-buffer contract, reused unchanged). Its
# layer-qualified Core wrappers stay in TENSOR_CORE_OPS; this entry is the
# differentiable operation.
#
# Phase G adds "dropout" (G3): one graph node over the G2 stateless Core,
# whose backward multiplies the upstream by the **private multiplier mask
# its own forward saved** — the third member of the graph-owned saved-state
# family beside maxpool2d's winners and cross-entropy's probabilities. Like
# them it records no expected parameter version, and like softmax it needed
# no backward kernel: the gradient is the existing `multiply` (design §7.5),
# so nothing joined TENSOR_CORE_OPS, RAW_KERNELS, or _CHECKED_KERNELS with
# it. The operation takes an explicit `NativeGenerator` and owns the
# reserve/commit/abandon call transaction; the Core below it stays
# generator-free.
#
# For the whole of G3-G9 "dropout" was deliberately in this tuple **and**
# in UNSUPPORTED — the one place in the whole registry where a name
# appeared in both. That was never an inconsistency: the two tuples answer
# different questions. This one says "a differentiable native operation by
# that name exists"; UNSUPPORTED says "the user-level Dropout capability is
# not closed". Phase G locked that split explicitly (design §19), because
# "dropout" names a capability whose entire value is exact reproducibility,
# and reproducibility is not a claim that can be made from source.
#
# **The G10 closure ended the overlap**: the committed known-answer vectors
# were reproduced under fresh Windows Release and Debug builds and a Clang
# ASan/UBSan build, the exact stochastic resume ran, and "dropout" left
# UNSUPPORTED as the last act of the phase. No name appears in both tuples
# any more, and the guardrail tests assert exactly that.
AUTOGRAD_OPS = (
    "add", "subtract", "multiply", "relu",
    "sum", "mean", "matmul",
    "reshape", "transpose", "T", "narrow", "contiguous_copy",
    "sqrt", "reciprocal",
    "conv2d",
    "maxpool2d",
    "exp", "log", "softmax", "log_softmax",
    "cross_entropy",
    "dropout",                 # G3
)

# The native training stack composed on the autograd layer (Phase C) and
# the Phase-D CNN modules, reported by name only so this module stays
# decoupled from the experimental package (the guardrail test verifies
# each name imports). "NativeConv2d" (the Conv2d *module*, D7) is the
# trainable layer over the differentiable "conv2d" op; it is distinct from
# that operation (in AUTOGRAD_OPS) and from the Core wrappers (in
# TENSOR_CORE_OPS).
NATIVE_MODULES = (
    "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
    "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
    # "NativeLayerNorm" (Phase F, milestone F2) is the first native
    # normalization module: stateless (no buffers, identical in train and
    # eval), differentiable through the mean and the population variance,
    # and composed entirely from existing native operations (mean,
    # subtract, multiply, add, sqrt, reciprocal). It is a *module*, not an
    # operation — there is no "layer_norm" kernel, C ABI symbol,
    # NativeTensorCore method, or NativeTensor.layer_norm autograd op, so
    # it appears here and nowhere in the op inventories. "layernorm" left
    # UNSUPPORTED when it shipped; "batchnorm" stayed there until F4.
    "NativeLayerNorm",
    # "NativeBatchNorm1d" (Phase F, milestone F3) is the first *stateful*
    # native numerical module: (N, C) batch normalization with
    # differentiable training statistics, persistent native
    # running_mean/running_var buffers advanced by a graph-free atomic
    # two-buffer transaction, and graph-safe immutable snapshots in eval
    # mode. It is composed from the same existing operations, so — like
    # LayerNorm — it adds no "batch_norm" kernel, C ABI symbol,
    # NativeTensorCore method, or NativeTensor.batch_norm autograd op, and
    # appears here and nowhere in the op inventories. "batchnorm" stayed
    # in UNSUPPORTED at F3, because the unqualified name is only honest
    # once the NCHW shape exists too.
    "NativeBatchNorm1d",
    # "NativeBatchNorm2d" (Phase F, milestone F4) is the NCHW
    # (N, C, H, W) shape of the same capability, built on the **same**
    # shared private implementation as NativeBatchNorm1d — it supplies
    # only its rank, its (N, H, W) reduction axes, its (1, C, 1, 1)
    # broadcast layout, and the channels-last permutation the rank-1
    # affine parameters need. It adds no numerical surface of its own, so
    # again nothing joined the op inventories. With both shapes now live,
    # "batchnorm" has finally left UNSUPPORTED.
    "NativeBatchNorm2d",
    # "NativeDropout" (Phase G, milestone G4) is the public module over
    # the G3 differentiable operation: stochastic inverted Dropout in
    # training, the input object itself in evaluation, identity at
    # p == 0, over one registered NativeGenerator it either owns (the
    # default — independent streams) or shares (an explicit generator,
    # stored as the exact object, never a copy). It adds no kernel, C ABI
    # symbol, ctypes declaration, NativeTensorCore method, or NativeTensor
    # operation — the forward is exactly `input.dropout(self.p,
    # generator=self.generator)` — so nothing joined AUTOGRAD_OPS,
    # TENSOR_CORE_OPS, or RAW_KERNELS with it; and the generator is
    # *registered generator state*, not a parameter or buffer, so
    # state_dict() is unchanged too.
    #
    # **"dropout" deliberately did NOT leave UNSUPPORTED here**, and that
    # is the one place Phase G departs from Phase F's precedent (which
    # moved "layernorm" at F2 and "batchnorm" at F4). Dropout's whole
    # value is exact, reproducible randomness: at G4 the module existed,
    # but the checkpoint format was still version 1 and did not persist
    # generator state, so exact stochastic resume did not exist yet. G5
    # moved the format to version 2, G7 demonstrated the resume, and the
    # **G10** closure reproduced the committed known-answer vectors under
    # fresh Release, Debug, and sanitized builds — only then did the name
    # leave UNSUPPORTED (design §19).
    "NativeDropout",
)
# Native loss *modules*. Losses are tracked here and deliberately not in
# NATIVE_MODULES, which lists the model-building layers — the split that
# has held since NativeMSELoss. "NativeCrossEntropyLoss" (E7) is a
# parameter-free wrapper whose forward is exactly
# NativeTensor.cross_entropy: the module is a separate capability from
# the differentiable operation (in AUTOGRAD_OPS) and from the Core
# wrappers (in TENSOR_CORE_OPS), the same three-way split conv2d and
# maxpool2d follow.
NATIVE_LOSSES = ("NativeMSELoss", "NativeCrossEntropyLoss")

# Native **reporting** metrics (E7). A separate inventory on purpose:
# these are neither runtime ops, nor differentiable operations, nor
# modules. `native_accuracy` is a plain Python helper that materializes
# its logits through the explicit public `to_numpy()` boundary and takes
# a NumPy argmax — there is no accuracy kernel, no C ABI export, no Core
# method, and no autograd node, and the native runtime has no integer
# dtype for an index-producing reduction to return. Listing it anywhere
# else would over-claim.
NATIVE_METRICS = ("native_accuracy",)

NATIVE_OPTIMIZERS = ("NativeSGD", "NativeAdam")
# Native state and persistence capabilities.
#
# "persistent_buffers" (added in Phase F, milestone F1) is **capability
# reconciliation, not a new feature**: NativeModule has held
# NativeTensor-backed non-parameter state since the pre-Phase-D hardening
# milestone — `register_buffer(name, tensor, persistent=True)`,
# `buffers()` / `named_buffers()`, persistent buffers included in
# `state_dict()` / `load_state_dict()` and in native checkpoints, and
# non-persistent buffers never serialized — but this tuple never said so,
# so `backend_info()` under-reported an existing capability. Unlike the
# four names beside it, it names a *capability* rather than a single
# callable: the API behind it is the register_buffer/buffers/
# named_buffers trio (see tests/test_cpp_backend_info.py, which proves
# every advertised name maps to something real).
#
# "generator_state" (added in Phase G, milestone G1) is the same kind of
# entry: a *capability* name covering NativeModule's fourth registered
# state category — `register_generator`, `generators()` /
# `named_generators()`, and the `generator_state_dict()` /
# `load_generator_state_dict()` inspection-and-replacement pair. It sits
# beside `state_dict`/`load_state_dict` because that is exactly what it
# is: a second, non-tensor in-memory state surface, deliberately separate
# because `state_dict()` is contractually `{name: NativeTensor}`.
#
# It reports the **in-memory** generator surface, and nothing more; the
# file half is its own name, below.
#
# "checkpoint_generator_state" (added in Phase G, milestone G5) is the
# file half, and it is a separate name precisely because G1's entry was
# explicitly scoped to memory: through G4 a save preserved parameters and
# buffers and silently omitted the random stream. G5's format version 2
# closes that, so this name means exactly what it says — a native
# checkpoint persists and restores every registered generator's
# `(algorithm, algorithm_version, seed, calls)` **and** the
# shared-versus-independent alias topology, in place and by identity,
# with strict both-directions validation against the live model. The API
# behind it is the existing `save_native_checkpoint` /
# `load_native_checkpoint` pair plus the manifest's `"generators"`
# section; there is no third entry point.
#
# What it still does not claim: Python's `random` state, NumPy's global
# RNG, data-loader/shuffle position, or scheduler state — none of which
# the native line has or captures (design §11.1). Reproducibility is
# exact for the state actually captured, and full-program determinism is
# not claimed. It is also not, by itself, a Dropout *capability* claim:
# "dropout" stayed in UNSUPPORTED through G9 and left it only at the G10
# closure, on the strength of the validation matrix (see the note above).
STATE_SUPPORT = (
    "persistent_buffers",
    "state_dict",
    "load_state_dict",
    "generator_state",
    "save_native_checkpoint",
    "load_native_checkpoint",
    "checkpoint_generator_state",
)

# Explicitly NOT implemented — listed so introspection is honest about the
# boundary. These names are layer-qualified where an operation and its
# module diverge, so support is never over- or under-claimed:
#   - the differentiable "conv2d" *operation* IS implemented (D3–D6:
#     forward + input/weight/bias gradients + NativeTensor autograd), so it
#     is NOT listed here — it lives in AUTOGRAD_OPS / TENSOR_CORE_OPS.
#   - "NativeConv2d" (the Conv2d *module*, D7) IS implemented (see
#     NATIVE_MODULES), so it is NOT listed here either — operation support
#     and module support are now both present for Conv2d.
#   - the differentiable "maxpool2d" *operation* IS implemented as of D9
#     (the D8 forward + private winner buffer, the D9 backward scatter, and
#     the NativeTensor autograd node), so it is NOT listed here — it lives
#     in AUTOGRAD_OPS, with its layer-qualified Core ops
#     "maxpool2d_forward"/"maxpool2d_backward" in TENSOR_CORE_OPS.
#   - "NativeMaxPool2d" (the pooling *module*, D10) IS implemented (see
#     NATIVE_MODULES), so it is NOT listed here either — operation support
#     and module support are now both present for MaxPool2d, as they are
#     for Conv2d.
# As of Phase D milestone D1, batch-preserving flatten IS implemented as
# the NativeFlatten module (see NATIVE_MODULES), so "flatten" is not listed.
# Phase D is complete (D0-D12): every CNN operation and module shipped,
# along with the deterministic end-to-end native CNN training +
# checkpoint-resume proof (D11) — a *proof*, not a capability name, so it
# has no entry in any inventory.
# The classification names below are the Phase-E surface contracted in
# docs/native_classification_design.md (milestone E0). A locked contract is
# not an implementation: each stays here until the milestone that
# implements it removes it. Milestones E1-E4 implemented the exponential,
# the logarithm, the softmax, and the log-softmax, so "exp", "log",
# "softmax", and "log_softmax" left this tuple for TENSOR_CORE_OPS and
# AUTOGRAD_OPS. "log_softmax" shipped as its own fused log-sum-exp kernel,
# deliberately NOT composed from the shipped "log" and "softmax"
# (design §4.4).
#
# Cross-entropy left this tuple at **E5**, which shipped its Core layer:
# the fused forward and the saved-probability backward as
# "cross_entropy_forward"/"cross_entropy_backward" in TENSOR_CORE_OPS.
# That is the layer-specific inventory contract (design §11) the Conv2d
# and MaxPool2d milestones already followed — a Core wrapper and a
# differentiable operation are different capabilities with different
# names. **E6** then shipped the differentiable operation itself:
# NativeTensor.cross_entropy builds one graph node over that Core
# contract, so the bare name "cross_entropy" now lives in AUTOGRAD_OPS.
# **E7** completed the public surface: "NativeCrossEntropyLoss" moved to
# NATIVE_LOSSES and "native_accuracy" to the new NATIVE_METRICS, so
# neither is listed here any more. Every classification name has now
# left this tuple, each into the one inventory that describes its actual
# layer — nothing about cross-entropy or accuracy is unsupported.
# What remains below is genuinely absent from the native line.
#
# Phase F milestone F2 shipped "NativeLayerNorm" (see NATIVE_MODULES), so
# "layernorm" has left this tuple — the module is a composition of
# existing operations, not a new kernel or ABI symbol, and there is still
# no "layer_norm" operation, which is why nothing joined AUTOGRAD_OPS /
# TENSOR_CORE_OPS / RAW_KERNELS.
#
# "batchnorm" has now left this tuple too. F3 shipped "NativeBatchNorm1d"
# (the (N, C) shape) and F4 "NativeBatchNorm2d" (NCHW), both in
# NATIVE_MODULES and both modules composed from existing operations — so
# again nothing joined AUTOGRAD_OPS / TENSOR_CORE_OPS / RAW_KERNELS and
# there is still no "batch_norm" operation, kernel, or C ABI symbol. The
# name here is unqualified, which is exactly why it stayed through F3 and
# was removed at F4, the milestone where *both* batch-normalization
# shapes exist. That completed the numerical normalization *module*
# surface, and Phase F has since closed (F0-F9): F5-F9 were hardening, an
# end-to-end proof, a benchmark, integration, and closure — none of them
# a capability — so this tuple is exactly what F4 left.
#
# **"dropout" has now left this tuple**, at the G10 closure and not one
# milestone earlier. It was the one name that ever appeared here *and* in
# an implemented inventory: G3 shipped the differentiable "dropout"
# operation into AUTOGRAD_OPS and G4 added "NativeDropout" to
# NATIVE_MODULES, while this entry stayed put through G9 (design §19). The
# reason was specific to this capability — Dropout's whole value is exact,
# reproducible randomness, and reproducibility is not a claim that can be
# made from source. It has to be demonstrated: against committed
# known-answer vectors under fresh Release, Debug, and ASan/UBSan builds,
# and with the stream surviving a checkpoint. G5 moved the format to
# version 2 so the stream survives, G7 demonstrated the exact stochastic
# resume, and G10 ran the §18 closure matrix — both Windows builds and
# their CTests, the Clang ASan/UBSan/LeakSanitizer validation, the
# sanitized Python suites, the resume example, and the benchmark gates.
# Only then did the name move. This tuple reports what is *closed and
# validated*; the operation inventories report what *exists*, and after
# G10 no name appears in both.
#
# The claim the move makes is deliberately narrow: **native Dropout is
# supported in TensorForge's experimental native float64 CPU backend**. It
# says nothing about the stable framework (which has always had its own
# separate `tensorforge.nn.Dropout`), and float32, CUDA, and AMP stay
# listed below because they remain genuinely absent.
#
# What remains below is genuinely absent from the native line.
UNSUPPORTED = (
    "float32", "cuda", "amp",
)

# Supported native dtype/device metadata (v1.21). The native kernels are
# float64 CPU only, so these are the single legal values today. The tags
# are explicit and validated — a native tensor never claims a dtype/device
# the kernels cannot actually compute, and unsupported values are rejected
# at construction rather than silently coerced (see
# docs/native_dtype_device_metadata_design.md).
SUPPORTED_DTYPES = ("float64",)
SUPPORTED_DEVICES = ("cpu",)

# ---------------------------------------------------------------------------
# The Python half of the native dtype model (Phase I, milestone I1; see
# docs/native_dtype_float32_design.md §3).
#
# Exactly two dtype authorities exist in the repository — the C++ enum in
# cpp/include/tf_internal.h and these tables — and they agree by
# construction because the ABI codes are the same integers. There is no
# third table anywhere: anything that needs a code, a width, or a NumPy
# type reads it from here.
#
# **Private on purpose, and wider than SUPPORTED_DTYPES on purpose.** The
# runtime can *allocate* float32 storage from I1 onward, but float32 is
# not a supported TensorForge dtype and does not become one until milestone
# I9 — so ``"float32"`` appears here (an internal capability) while
# ``normalize_dtype`` still rejects it (the public promise). That gap is
# the deliberate pattern Phase G used for ``dropout``, which existed as an
# operation from G3 and left UNSUPPORTED only at G10. Nothing public reads
# these tables, no public dtype object is built on them, and there is no
# exported dtype-query symbol: Python knows a storage's dtype because
# Python asked for it at creation.
#
# The codes are frozen in the same sense the TfStatus codes are.
_DTYPE_CODES = {"float64": 0, "float32": 1}
_DTYPE_ITEM_SIZES = {"float64": 8, "float32": 4}
_DTYPE_NUMPY = {"float64": np.float64, "float32": np.float32}

# Largest element count the native int64 storage/ABI arithmetic addresses.
_INT64_MAX = 2 ** 63 - 1
# ...and the low end, for the class labels that cross as int64 metadata.
_INT64_MIN = -(2 ** 63)
# IEEE float64 represents every integer in [-(2**53), 2**53] exactly, so a
# flat plane offset stored as a float64 (the internal MaxPool2d winner
# buffer, docs/native_cnn_design.md §12) is exact iff the plane holds at
# most 2**53 elements. Proved in Python arbitrary-precision arithmetic
# before any allocation, and re-proved at the C ABI boundary.
_MAX_EXACT_WINNER_PLANE = 2 ** 53

_lib = None  # loaded lazily by _require_library()

# ---------------------------------------------------------------------------
# The two argument bindings for arrays crossing the C ABI (Phase H, H7)
#
# Every array position in this module's declarations falls into exactly one
# of two categories, and they are bound differently on purpose. Applying one
# blanket policy to both would either leave a real check off a caller-facing
# path or pay a per-call validation cost for an invariant that was already
# established, once, at an immutable construction boundary.
#
# **CHECKED** — ``_CHECKED_F64_ARRAY`` / ``_CHECKED_I64_ARRAY``.
# ``numpy.ctypeslib.ndpointer`` re-verifies, at every call, that the argument
# is a NumPy array, that its dtype matches exactly (byte order included), and
# that it is C-contiguous. That is exactly right where the array is *data* —
# a buffer a caller supplied, or one native code writes into and hands back:
#
#   * the seven raw public kernels (``elementwise_add`` … ``matmul_tiled``),
#     which any caller may reach with any object at all;
#   * ``tf_storage_copy_from`` / ``tf_storage_copy_to`` and the float64
#     destination of ``tf_storage_materialize`` — the explicit host
#     conversion boundary;
#   * the int64 **class labels** of the two cross-entropy exports. Those are
#     int64 like the layout metadata below, and they are deliberately *not*
#     bound as trusted: a label array's required length is the logits'
#     ``batch_size``, which comes from a *different* object than the array
#     does, so a dtype and layout check at the boundary is still doing work
#     the construction site cannot do by itself. There is one such array per
#     cross-entropy call, so the check costs nothing that matters.
#
# **TRUSTED** — ``_LAYOUT_POINTER``.
# The strided C ABI's ``shape`` / ``strides`` / write-stride positions are
# *layout metadata this module built for itself* from a tuple it had already
# validated. Exactly two producers exist and no other object can reach these
# positions:
#
#   * ``NativeTensorView._native_layout_pointers()`` — the per-view pair,
#     derived from the H3 read-only NumPy layout arrays that remain the
#     owning buffers;
#   * ``_layout_vector(values)`` — a fresh, exactly ``len(values)`` long
#     ``c_int64`` vector for the operation-local metadata (broadcast strides,
#     reduction write-strides, ``narrow_backward``'s output strides).
#
# Both establish, **by construction**, every property ``ndpointer`` would
# re-check and one it never checked at all:
#
#   | property        | how it is established                              |
#   |-----------------|----------------------------------------------------|
#   | element type    | the pointer/vector *is* ``c_int64``; ctypes rejects |
#   |                 | every other type at the call (a NumPy array, a      |
#   |                 | differently typed pointer, bytes, a list, an int)   |
#   | contiguity      | a ``c_int64`` vector and a NumPy ``int64`` array    |
#   |                 | are contiguous by construction                      |
#   | byte order      | native by construction — neither carrier has a      |
#   |                 | byte-order concept to get wrong                     |
#   | **length**      | ``len(values)``, or the rank of the immutable view  |
#   |                 | the arrays were built from — **the one invariant    |
#   |                 | ``ndpointer`` never checked**, because the ABI takes |
#   |                 | a pointer and an ``ndim`` and cannot see the        |
#   |                 | Python object's length                              |
#   | owner lifetime  | ``ndarray.ctypes.data_as`` stores the array on the  |
#   |                 | pointer (``ptr._arr``), so the buffer cannot be     |
#   |                 | freed while the pointer exists; a fresh vector owns |
#   |                 | its own buffer                                      |
#
# The one value ``POINTER(c_int64)`` accepts that ``ndpointer`` rejected is
# ``None`` (a null pointer). No production path can produce it: both
# producers are total, and neither can return ``None`` for any constructible
# layout — including rank 0, where both yield a valid zero-length buffer the
# kernels never dereference. ``tests/test_native_abi_boundary.py`` proves
# that, and proves the rank/length agreement these positions depend on.
# ---------------------------------------------------------------------------

# The checked bindings, exactly as they have always been declared. Building
# them at module scope rather than inside ``_load_library`` keeps the two
# policies visible side by side and costs nothing: ``ndpointer`` memoizes its
# generated classes, and neither of these touches the compiled library, so
# importing this module still loads nothing.
_CHECKED_F64_ARRAY = np.ctypeslib.ndpointer(
    dtype=np.float64, flags="C_CONTIGUOUS"
)
# Phase I, milestone I2: the float32 sibling. Same binding, same three
# checks, one dtype narrower — see ``_host_pointer`` below for how one of
# the two is chosen per call, and why the choice is made from data.
_CHECKED_F32_ARRAY = np.ctypeslib.ndpointer(
    dtype=np.float32, flags="C_CONTIGUOUS"
)
_CHECKED_I64_ARRAY = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")

# The checked host-buffer binding, per dtype. Keyed by the canonical dtype
# string so the lookup is the storage's own tag and never a call-site flag.
_CHECKED_HOST_ARRAYS = {
    "float64": _CHECKED_F64_ARRAY,
    "float32": _CHECKED_F32_ARRAY,
}


def _host_pointer(array, dtype):
    """The checked ``void*`` for a host buffer crossing one of the three
    retyped transfer boundaries (Phase I, milestone I2).

    ``tf_storage_copy_from``, ``tf_storage_copy_to``, and
    ``tf_storage_materialize`` are the only exports that carry a storage
    handle *and* a raw host buffer, and at I2 their host positions became
    ``void*`` in C — the storage handle's immutable dtype tag is what says
    how those bytes are read. A single ctypes ``argtypes`` slot cannot hold
    two dtypes, so the declaration is a plain ``c_void_p`` (exactly the C
    parameter) and **the check moves here, where the dtype is known**.

    It is the *same* check, not a reimplementation of one:
    ``ndpointer.from_param`` is precisely what ctypes would have run had the
    binding been in the argtypes slot, so there is one implementation of
    "is this really a NumPy array of exactly this dtype, in native byte
    order, C-contiguous?" and it cannot drift. A wrong host buffer raises
    ``TypeError`` and the native call is never made — nothing is cast,
    widened, narrowed, or guessed at this boundary.

    The returned pointer comes from ``ndarray.ctypes.data_as``, which
    attaches the owning array to it (``ptr._arr``), so the buffer cannot be
    freed while the pointer exists — the same lifetime property the trusted
    layout pointers rely on. ``ctypes.POINTER(...).from_address(...)`` would
    be cheaper and is deliberately not used: it produces a pointer with no
    reference to its owner.

    ``dtype`` is a canonical dtype string that came from a live storage, so
    the lookup is total in practice; an unknown one is a programming error
    and says so rather than silently picking a width.
    """
    try:
        binding = _CHECKED_HOST_ARRAYS[dtype]
    except KeyError:
        raise ValueError(
            f"no host-buffer binding for dtype {dtype!r}; the native "
            f"runtime represents {tuple(_CHECKED_HOST_ARRAYS)}"
        ) from None
    binding.from_param(array)  # the ndpointer check, at every call
    return array.ctypes.data_as(ctypes.c_void_p)

# Layout metadata is always a pointer to int64. Declared once, used by every
# trusted position, so no call site can pick a different (or weaker) type.
_LAYOUT_POINTER = ctypes.POINTER(ctypes.c_int64)


def _layout_vector(values):
    """A fresh ``c_int64`` vector holding ``values``, for the operation-local
    layout metadata the strided C ABI takes (Phase H, milestone H7).

    ``values`` is a sequence of exact Python ints this module derived from an
    already-validated layout — broadcast strides, a reduction's write-strides,
    or ``narrow_backward``'s output strides. Each is a property of *one
    operation* rather than of any tensor, so there is nothing to cache: the
    vector is built, passed, and dropped inside a single call.

    The result carries its own length (``len(vector) == len(values)``), owns
    the buffer the kernel reads, and is a live local of the calling frame for
    the whole native call — so no pointer here can outlive its storage. It
    replaces a ``numpy.asarray(values, dtype=numpy.int64)`` that then had to
    be re-validated by ``ndpointer`` at every call; building the vector
    directly is both cheaper and stricter, because a ``c_int64`` vector
    cannot have the wrong element type or the wrong length.
    """
    return (ctypes.c_int64 * len(values))(*values)


def normalize_dtype(dtype=None):
    """Validate and canonicalize a native dtype tag.

    ``None`` means the default ``"float64"`` (the only supported dtype
    today). A non-string raises TypeError; a string outside
    ``SUPPORTED_DTYPES`` raises ValueError naming the offending value and
    the supported set. Pure Python — never touches the compiled library,
    so it is safe whether or not the backend is built."""
    if dtype is None:
        return "float64"
    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be a string or None, got {dtype!r}")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; the native runtime supports "
            f"{SUPPORTED_DTYPES}"
        )
    return dtype


def _normalize_internal_dtype(dtype):
    """Validate a dtype against the **internal** table rather than the
    public registry (Phase I, milestone I2).

    The private counterpart of ``normalize_dtype``: the same
    canonicalization, the same ``TypeError`` for a non-string, and the same
    shape of ``ValueError`` — measured against ``_DTYPE_CODES``, which is
    deliberately wider than ``SUPPORTED_DTYPES`` between I1 and I9.

    **This is not a public bypass and must not become one.** It is reachable
    only from the private ``_typed`` constructors and from the private
    uninitialized allocators, whose ``dtype`` argument is always either the
    ``"float64"`` default or a canonical tag read off a storage that was
    validated when it was created. Every public constructor — ``NativeStorage``,
    ``NativeTensorCore.zeros`` / ``.full`` / ``.from_array``, and everything
    layered on them — keeps calling ``normalize_dtype`` and keeps rejecting
    ``"float32"`` with the same message, because float32 is not a supported
    TensorForge dtype until milestone I9.

    That gap between internal capability and public promise is the
    deliberate rollout pattern (design §27), the same one Phase G used for
    ``dropout``: the operation existed from G3 and the *name* left
    ``UNSUPPORTED`` only at G10, once it survived a checkpoint.
    """
    if dtype is None:
        return "float64"
    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be a string or None, got {dtype!r}")
    if dtype not in _DTYPE_CODES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; the native runtime represents "
            f"{tuple(_DTYPE_CODES)}"
        )
    return dtype


def normalize_device(device="cpu"):
    """Validate and canonicalize a native device tag.

    ``None`` means the default ``"cpu"`` (the only supported device
    today). A non-string raises TypeError; a string outside
    ``SUPPORTED_DEVICES`` raises ValueError naming the offending value and
    the supported set. Pure Python — never touches the compiled library."""
    if device is None:
        return "cpu"
    if not isinstance(device, str):
        raise TypeError(f"device must be a string or None, got {device!r}")
    if device not in SUPPORTED_DEVICES:
        raise ValueError(
            f"unsupported device {device!r}; the native runtime supports "
            f"{SUPPORTED_DEVICES}"
        )
    return device


def build_instructions():
    """The commands needed to build the experimental backend."""
    return (
        "The C++ backend is experimental. Build it from the repo root:\n"
        "    uv sync --group cpp   # only if you have no C++ compiler\n"
        "    uv run python cpp/build.py"
    )


def _load_library():
    if not _LIBRARY_PATH.exists():
        raise ImportError(
            f"The experimental C++ backend is not built "
            f"(missing {_LIBRARY_PATH.name}).\n" + build_instructions()
        )
    try:
        library = ctypes.CDLL(str(_LIBRARY_PATH))
    except OSError as error:
        raise ImportError(
            f"The experimental C++ backend library exists but failed to "
            f"load ({error}). Try rebuilding it.\n" + build_instructions()
        ) from error
    # The checked binding for caller-facing float64 data buffers, and the
    # trusted binding for this module's own int64 layout metadata. See the
    # contract above ``_LAYOUT_POINTER`` for which positions get which, and
    # why one blanket policy would be wrong.
    f64_array = _CHECKED_F64_ARRAY
    layout = _LAYOUT_POINTER
    for name in _BINARY_KERNELS:
        kernel = getattr(library, name)
        kernel.argtypes = [f64_array, f64_array, f64_array, ctypes.c_int64]
        kernel.restype = None
    library.tf_relu.argtypes = [f64_array, f64_array, ctypes.c_int64]
    library.tf_relu.restype = None
    library.tf_matmul.argtypes = [
        f64_array, f64_array, f64_array,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_matmul.restype = None
    library.tf_matmul_tiled.argtypes = [
        f64_array, f64_array, f64_array,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_matmul_tiled.restype = None
    library.tf_storage_create.argtypes = [ctypes.c_int64]
    library.tf_storage_create.restype = ctypes.c_void_p
    # Phase H (H1): the uninitialized sibling of tf_storage_create. Same
    # signature, same validation, same failure contract; it differs only
    # in leaving the buffer's initial contents indeterminate. Internal
    # backend detail — no public "empty" API is built on it.
    library.tf_storage_create_uninitialized.argtypes = [ctypes.c_int64]
    library.tf_storage_create_uninitialized.restype = ctypes.c_void_p
    # Phase I (I1): the two typed creators — the **only** two symbols the
    # whole phase adds (52 -> 54). Same shape as the untyped pair plus an
    # int32 dtype code, and the identical failure convention: null plus the
    # thread-local error, read by the same errcheck hook. They are the one
    # place the ABI has to grow, because construction is the single moment
    # at which a storage's dtype is not yet knowable from any argument —
    # every other export reads it from a handle it already receives.
    #
    # ``c_int32`` matches the C ABI exactly; the codes come from the one
    # table above and never from a literal at a call site.
    for name in ("tf_storage_create_typed",
                 "tf_storage_create_uninitialized_typed"):
        kernel = getattr(library, name)
        kernel.argtypes = [ctypes.c_int64, ctypes.c_int32]
        kernel.restype = ctypes.c_void_p
    library.tf_storage_destroy.argtypes = [ctypes.c_void_p]
    library.tf_storage_destroy.restype = None
    library.tf_storage_size.argtypes = [ctypes.c_void_p]
    library.tf_storage_size.restype = ctypes.c_int64
    library.tf_storage_fill.argtypes = [ctypes.c_void_p, ctypes.c_double]
    library.tf_storage_fill.restype = None
    # Phase I (I2): the three transfer boundaries take ``void*`` host
    # positions, matching the C declarations exactly. The element type comes
    # from the storage handle's dtype tag, and the per-dtype ``ndpointer``
    # check that used to sit in these slots now runs in ``_host_pointer``
    # at every call — it could not stay here, because one argtypes slot
    # cannot describe two dtypes and the choice must be made from data.
    #
    # No symbol was added, renamed, or reordered: this is the source-level
    # retype §7.3 of the design reserved for I2, and a previously compiled
    # caller would link and run identically.
    library.tf_storage_copy_from.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.tf_storage_copy_from.restype = None
    library.tf_storage_copy_to.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.tf_storage_copy_to.restype = None
    # The class-label binding: int64 like the layout metadata, but bound
    # CHECKED, because a label array's required length comes from the logits
    # rather than from the array itself (see the contract above).
    i64_array = _CHECKED_I64_ARRAY
    # The destination is a ``void*`` for the same reason the two flat
    # transfers are (Phase I, I2) and is checked per dtype at the call site;
    # the shape/stride pair keeps the trusted layout binding, exactly as for
    # every other strided export.
    library.tf_storage_materialize.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, layout, layout,
        ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_storage_materialize.restype = None
    # Unary core kernels share tf_core_relu's signature (one strided
    # source, one contiguous destination); sqrt/reciprocal are the
    # v3.11 optimizer math primitives while exp and log are the Phase-E
    # stable-math primitives (milestones E1 and E2).
    # tf_core_contiguous_copy shares the same signature: it is the
    # identity map over the odometer, gathering a strided view of one
    # storage into a second storage (the native Policy-B copy, E3.1).
    for name in ("tf_core_relu", "tf_core_sqrt", "tf_core_reciprocal",
                 "tf_core_exp", "tf_core_log", "tf_core_contiguous_copy"):
        kernel = getattr(library, name)
        kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, layout, layout,
            ctypes.c_int64, ctypes.c_int64,
        ]
        kernel.restype = None
    # tf_core_relu_backward shares the binary-kernel signature: it walks
    # the forward input and the upstream gradient in lockstep.
    for name in (
        "tf_core_add", "tf_core_subtract", "tf_core_multiply",
        "tf_core_relu_backward",
    ):
        kernel = getattr(library, name)
        kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            layout, layout, layout,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        kernel.restype = None
    # Contiguous fast-path kernels (v1.14): flat, index-free loops that
    # take numel + offsets instead of shape/strides. Selected by
    # NativeTensorCore when the operands are row-major contiguous.
    for name in ("tf_core_relu_contiguous", "tf_core_sqrt_contiguous",
                 "tf_core_reciprocal_contiguous", "tf_core_exp_contiguous",
                 "tf_core_log_contiguous"):
        kernel = getattr(library, name)
        kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
        ]
        kernel.restype = None
    for name in (
        "tf_core_add_contiguous",
        "tf_core_subtract_contiguous",
        "tf_core_multiply_contiguous",
    ):
        kernel = getattr(library, name)
        kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        kernel.restype = None
    library.tf_core_matmul.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ] + [ctypes.c_int64] * 9
    library.tf_core_matmul.restype = None
    # Reduction kernels (v1.19): tf_core_sum scatter-accumulates a strided
    # input into a fresh contiguous output using per-input-axis output
    # write-strides (0 on reduced axes); tf_storage_scale does mean's
    # in-place 1/count scaling.
    #
    # Phase H, milestone H6 gave tf_core_sum a second *traversal* behind
    # this **unchanged** declaration — a flat block walk for row-major
    # sources whose reduced axes form one contiguous run, chosen inside the
    # kernel from these same arguments. The signature, the argument
    # meanings, the accumulate-into contract, and the required
    # zero-initialized destination are all exactly what they were, and no
    # symbol was added; see cpp/include/tf_reduction_internal.h.
    # tf_core_narrow_backward, the scatter dual below, was deliberately
    # left on the generic odometer.
    #
    # Phase H, milestone H7 rebound the three int64 positions from the
    # checked ``ndpointer`` to the trusted ``_LAYOUT_POINTER``. The two
    # source arrays come from the reducing tensor's own immutable per-view
    # cache and the write-strides from ``_layout_vector``; the argument
    # meanings, the order, and the kernel are untouched.
    library.tf_core_sum.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, layout, layout, layout,
        ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_core_sum.restype = None
    library.tf_storage_scale.argtypes = [ctypes.c_void_p, ctypes.c_double]
    library.tf_storage_scale.restype = None
    # Narrow backward (v2.3): scatter the upstream gradient into a fresh
    # zero output of the parent shape. Same odometer as tf_core_sum plus a
    # base output offset (start * row-major stride of the narrowed axis).
    library.tf_core_narrow_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, layout, layout, layout,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_core_narrow_backward.restype = None
    # Conv2d forward (Phase D, D3): the exported wrapper over the internal
    # cross-correlation kernel. Contiguous storage only (Policy B copies at
    # the Core level); a null bias handle means "no bias"; the output is
    # caller-allocated. Handles carry per-operand offsets; the 13 trailing
    # int64s are N, C, H, W, O, kh, kw, sh, sw, ph, pw, out_h, out_w.
    library.tf_core_conv2d_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_int64,   # input handle, offset
        ctypes.c_void_p, ctypes.c_int64,   # weight handle, offset
        ctypes.c_void_p, ctypes.c_int64,   # bias handle (nullable), offset
        ctypes.c_void_p,                   # output handle
    ] + [ctypes.c_int64] * 13
    library.tf_core_conv2d_forward.restype = None
    # Conv2d backward wrappers (Phase D, D6): input- and weight-gradient
    # exports over the D4/D5 internal kernels. Contiguous storage only
    # (Policy B copies at the Core level); caller-allocated output; no bias
    # gradient symbol (that composes from the existing `sum` reduction). The
    # handle pairs differ by direction but both take the same 13 trailing
    # int64 dims as the forward wrapper (N, C, H, W, O, kh, kw, sh, sw, ph,
    # pw, out_h, out_w).
    for name in ("tf_core_conv2d_input_backward",
                 "tf_core_conv2d_weight_backward"):
        kernel = getattr(library, name)
        kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_int64,   # grad_output handle, offset
            ctypes.c_void_p, ctypes.c_int64,   # weight/input handle, offset
            ctypes.c_void_p,                   # grad_input/grad_weight handle
        ] + [ctypes.c_int64] * 13
        kernel.restype = None
    # MaxPool2d forward (Phase D, D8): the exported wrapper over the
    # internal window-maximum kernel. Contiguous storage only (Policy B
    # copies at the Core level); the output *and* the private winner buffer
    # are caller-allocated (offset 0). Only the input carries an offset;
    # the 12 trailing int64s are N, C, H, W, kh, kw, sh, sw, ph, pw, out_h,
    # out_w. There is no backward symbol yet — that is D9.
    library.tf_core_maxpool2d_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_int64,   # input handle, offset
        ctypes.c_void_p,                   # output handle
        ctypes.c_void_p,                   # winner-buffer handle
    ] + [ctypes.c_int64] * 12
    library.tf_core_maxpool2d_forward.restype = None
    # MaxPool2d backward (Phase D, D9): the exported wrapper over the
    # internal scatter-add. It takes the upstream gradient and the private
    # winner buffer (each with an offset) plus the caller-allocated
    # grad_input, and **no kernel/stride/padding metadata** — the saved
    # winners fully determine the routing, so window geometry is never
    # recomputed. The 6 trailing int64s are N, C, H, W, out_h, out_w.
    library.tf_core_maxpool2d_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_int64,   # grad_output handle, offset
        ctypes.c_void_p, ctypes.c_int64,   # winner-buffer handle, offset
        ctypes.c_void_p,                   # grad_input handle
    ] + [ctypes.c_int64] * 6
    library.tf_core_maxpool2d_backward.restype = None
    # Softmax forward (Phase E, E3): the exported wrapper over the fused
    # maximum-shift kernel in classification.cpp. **Contiguous storage
    # only** — the Core layer applies Policy-B copy-then-compute, so no
    # stride metadata crosses the boundary. Only the source carries an
    # offset (the destination is caller-allocated at offset 0); the three
    # trailing int64s are the (outer, axis_length, inner) decomposition of
    # the reduction axis.
    library.tf_core_softmax_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_int64,   # source handle, offset
        ctypes.c_void_p,                   # destination handle
    ] + [ctypes.c_int64] * 3
    library.tf_core_softmax_forward.restype = None
    # Log-softmax forward (Phase E, E4): the exported wrapper over the
    # fused maximum-shift / log-sum-exp kernel in the same translation
    # unit. Identical call shape to the softmax export — contiguous
    # storage only, one source offset, a caller-allocated destination at
    # offset 0, and the (outer, axis_length, inner) decomposition of the
    # reduction axis. It is a *separate* kernel, not softmax composed with
    # log; there is deliberately no log-softmax backward export.
    library.tf_core_log_softmax_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_int64,   # source handle, offset
        ctypes.c_void_p,                   # destination handle
    ] + [ctypes.c_int64] * 3
    library.tf_core_log_softmax_forward.restype = None
    # Fused cross-entropy forward (Phase E, E5): the exported wrapper over
    # the fused maximum-shift / log-sum-exp kernel in the same translation
    # unit. **Contiguous storage only** for the tensor data (Policy-B
    # copy-then-compute happens at the Core layer), with only the logits
    # carrying an offset — the scalar loss and the saved probabilities are
    # both caller-allocated at offset 0. The targets are the one
    # non-tensor operand: a contiguous host int64 array plus its length,
    # marshalled exactly like the shape/stride metadata arrays because
    # that is what they are — host metadata, never native tensor data (the
    # runtime has no integer dtype). The three trailing int64s are
    # batch_size, num_classes, and the reduction code (0 = mean,
    # 1 = sum).
    library.tf_core_cross_entropy_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_int64,   # logits handle, offset
        i64_array, ctypes.c_int64,         # targets, target_count
        ctypes.c_void_p,                   # loss handle (scalar)
        ctypes.c_void_p,                   # probabilities handle
    ] + [ctypes.c_int64] * 3
    library.tf_core_cross_entropy_forward.restype = None
    # Fused cross-entropy backward (E5): reads the **saved probabilities**,
    # the copied targets, and one upstream value; the logits are neither
    # passed nor reachable, which is the structural half of "backward
    # never rereads the logits". Both the probabilities and the upstream
    # carry an offset (the upstream is a single element wherever it sits);
    # the gradient is caller-allocated at offset 0.
    library.tf_core_cross_entropy_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_int64,   # probabilities handle, offset
        i64_array, ctypes.c_int64,         # targets, target_count
        ctypes.c_void_p, ctypes.c_int64,   # upstream handle, offset
        ctypes.c_void_p,                   # grad_logits handle
    ] + [ctypes.c_int64] * 3
    library.tf_core_cross_entropy_backward.restype = None
    # Stateless Dropout forward (Phase G, G2): the exported wrapper over
    # the internal "tensorforge.splitmix64" derivation and the inverted
    # Dropout kernel in random.cpp. **Contiguous storage only** (Policy-B
    # copy-then-compute happens at the Core layer), with only the input
    # carrying an offset — the output and the private multiplier mask are
    # both caller-allocated at offset 0. The random key is the explicit
    # (seed, call_index) pair: two ``c_uint64`` arguments, so every value
    # in [0, 2**64 - 1] crosses exactly, and the kernel holds no state
    # between calls. ``p`` crosses as the ordinary ``c_double``.
    library.tf_core_dropout_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_int64,   # input handle, offset
        ctypes.c_void_p,                   # output handle
        ctypes.c_void_p,                   # multiplier-mask handle
        ctypes.c_int64,                    # element count
        ctypes.c_uint64,                   # seed
        ctypes.c_uint64,                   # call index
        ctypes.c_double,                   # p
    ]
    library.tf_core_dropout_forward.restype = None

    _configure_error_contract(library)
    return library


# ---------------------------------------------------------------------------
# Native error contract (see docs/native_abi_error_contract.md)
#
# No C++ exception may cross the extern "C" boundary. Each fallible
# native function clears a thread-local error slot on entry and, on any
# exception, records a status code plus message there and returns a
# benign value instead of unwinding. A ctypes ``errcheck`` hook on every
# such function reads the slot after the call and raises the matching
# Python exception, so a native failure surfaces as a normal exception at
# the call site with useful context — never a crash or a silently wrong
# result.
# ---------------------------------------------------------------------------

# TfStatus codes (kept in sync with cpp/include/tf_internal.h) mapped to
# the Python exception each becomes.
TF_OK = 0
_STATUS_EXCEPTIONS = {
    1: MemoryError,   # TF_ERROR_ALLOC
    2: ValueError,    # TF_ERROR_INVALID
    3: RuntimeError,  # TF_ERROR_RUNTIME
}

# The exported functions that participate in the error contract — every
# function that clears-on-entry and may set the slot. The unguarded
# storage/legacy kernels (destroy, size, fill, scale, copy, the raw
# elementwise/matmul kernels) never allocate, so they neither clear nor
# set the slot and must NOT carry the hook, or a stale code from an
# earlier call could be misread as their own failure.
_CHECKED_KERNELS = (
    "tf_storage_create",
    # Phase H (H1). The uninitialized constructor reports failure exactly
    # like the zero-initializing one — a null handle with the thread-local
    # error set — so it takes the identical errcheck hook rather than a
    # second, divergent failure convention.
    "tf_storage_create_uninitialized",
    # Phase I (I1). The typed creators report failure exactly as the
    # untyped pair does — a null handle plus the thread-local error — so
    # they join the same hook. An unknown dtype code, a non-positive size,
    # and a byte-count overflow all surface as ValueError; a real or
    # injected allocation failure as MemoryError.
    "tf_storage_create_typed",
    "tf_storage_create_uninitialized_typed",
    "tf_storage_materialize",
    "tf_core_relu", "tf_core_relu_contiguous",
    "tf_core_sqrt", "tf_core_sqrt_contiguous",
    "tf_core_reciprocal", "tf_core_reciprocal_contiguous",
    # Phase E stable math (E1: exp, E2: log). Unlike the older unary
    # exports these validate their own handles/layout/spans before
    # computing, so an invalid call raises ValueError through this hook.
    # IEEE domain results (log of 0/negative) stay values, not errors.
    "tf_core_exp", "tf_core_exp_contiguous",
    "tf_core_log", "tf_core_log_contiguous",
    # The native strided-to-contiguous storage gather behind every
    # contiguous_copy / Policy-B path (E3.1). Self-validating.
    "tf_core_contiguous_copy",
    "tf_core_add", "tf_core_add_contiguous",
    "tf_core_subtract", "tf_core_subtract_contiguous",
    "tf_core_multiply", "tf_core_multiply_contiguous",
    "tf_core_relu_backward",
    "tf_core_matmul",
    "tf_core_sum",
    "tf_core_narrow_backward",
    "tf_core_conv2d_forward",
    "tf_core_conv2d_input_backward",
    "tf_core_conv2d_weight_backward",
    "tf_core_maxpool2d_forward",
    "tf_core_maxpool2d_backward",
    # Phase E, E3/E4: the fused softmax and log-softmax forwards. Both
    # self-validate like the other Phase-E exports (through one shared
    # file-local validator); IEEE NaN/inf results stay values, not errors.
    # Neither has a backward export — those gradients are composed from
    # existing Core ops.
    "tf_core_softmax_forward",
    "tf_core_log_softmax_forward",
    # Phase E, E5: the fused cross-entropy forward and backward. Both
    # self-validate every trust-boundary argument — including the range of
    # every target index they dereference, which Python is never trusted
    # for at this boundary — and write nothing to any destination when
    # they reject. IEEE NaN/inf results stay values, not errors.
    "tf_core_cross_entropy_forward",
    "tf_core_cross_entropy_backward",
    # Phase G, G2: the stateless Dropout forward. Self-validating like the
    # Phase-E exports — null handles, a negative offset or count, a span
    # exceeding its storage, a non-finite or out-of-range probability, and
    # any aliasing between the input and either destination are all
    # rejected with ValueError through this hook, and a rejected call
    # leaves both destinations byte-for-byte unchanged. It touches no
    # generator: the whole random key arrives as two explicit uint64
    # arguments. There is deliberately no backward export — that gradient
    # is the existing `multiply` over the saved mask.
    "tf_core_dropout_forward",
)


def _configure_error_contract(library):
    """Declare the error-introspection ABI and attach the errcheck hook."""
    library.tf_last_error_code.argtypes = []
    library.tf_last_error_code.restype = ctypes.c_int
    library.tf_last_error_message.argtypes = []
    library.tf_last_error_message.restype = ctypes.c_char_p
    library.tf_clear_error.argtypes = []
    library.tf_clear_error.restype = None
    library.tf_test_arm_alloc_failure.argtypes = [ctypes.c_int64]
    library.tf_test_arm_alloc_failure.restype = None
    library.tf_fault_injection_available.argtypes = []
    library.tf_fault_injection_available.restype = ctypes.c_int

    def _errcheck(result, func, arguments):
        # Runs after every checked native call. The callee cleared the
        # slot on entry, so a nonzero code here is genuinely this call's
        # failure; translate it into the right Python exception with the
        # native message for context, then clear the slot.
        code = library.tf_last_error_code()
        if code != TF_OK:
            raw = library.tf_last_error_message()
            message = raw.decode("utf-8", "replace") if raw else ""
            library.tf_clear_error()
            exception = _STATUS_EXCEPTIONS.get(code, RuntimeError)
            raise exception(
                f"native backend {func.__name__} failed"
                + (f": {message}" if message else f" (status {code})")
            )
        return result

    for name in _CHECKED_KERNELS:
        getattr(library, name).errcheck = _errcheck


def _require_library():
    """Load the compiled library on first use; raise helpfully if missing."""
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


def is_available():
    """True if the compiled backend can actually be loaded.

    This attempts a real load (cached after the first success), not
    just a file-existence check. Never raises.
    """
    try:
        _require_library()
    except ImportError:
        return False
    return True


def fault_injection_available():
    """True if the compiled backend includes the test-only allocation
    fault-injection hook (it does for any build from this repo). Never
    raises; returns False when the backend is not built."""
    try:
        library = _require_library()
    except ImportError:
        return False
    return bool(library.tf_fault_injection_available())


def _arm_alloc_failure(nth=1):
    """Test-only: arm the calling thread so the ``nth`` subsequent native
    allocation attempt fails with a simulated ``std::bad_alloc`` (``nth=1``
    targets the very next allocation; ``nth <= 0`` disarms). Deterministic
    and thread-local. The native failure surfaces through the normal error
    contract as ``MemoryError``. Used by the ABI failure tests; inert in
    normal use (see docs/native_abi_error_contract.md)."""
    _require_library().tf_test_arm_alloc_failure(int(nth))


# Deliberately absent: any poison-control helper (Phase H, H1).
#
# H1's "every destination element is written" proof uses a deterministic
# poison, but that poison belongs to the *test infrastructure*, not to
# this runtime. The suite wraps ``NativeStorage._uninitialized``, lets
# the real uninitialized constructor allocate, fills the returned storage
# through the ordinary ``fill`` primitive, and hands that same storage to
# the real operation — so the pattern is in place before the real kernel
# runs, with nothing in the shipped library or this module able to alter
# an allocation's contents. There is no ``_set_uninitialized_poison``, no
# ``_uninitialized_poison`` context manager, no environment variable, and
# no global mode here, and none may be added: a switch that can change
# what production allocations contain is not a debugging convenience
# worth shipping, however carefully it is disarmed by default. The
# corresponding C ABI hook does not exist either — see
# tests/test_native_storage_allocation.py, which asserts its absence
# against the loaded library's real export table.


def list_kernels():
    """The experimental kernels this backend provides, in stable order."""
    return KERNELS


def backend_info():
    """An accurate capability summary of the experimental backend.

    Reports each layer separately so a caller can tell raw C++ kernels
    from the higher-level capabilities composed on them: the raw
    NumPy-buffer kernels, the ``NativeTensorCore`` runtime ops (which
    broadcast), the ``NativeTensor`` autograd ops, and the native
    training stack (modules — including the Phase-D CNN layers — the loss,
    the optimizers, and state/checkpoint support). ``stable_framework_integration`` stays ``False`` — the
    native line is deliberately separate from ``tensorforge.Tensor`` — but
    ``native_autograd`` is ``True`` and the optimizer/state lists are
    populated. Every list is sourced from the module-level inventory
    tuples, so this never drifts from the code. Safe to call whether or
    not the library is built."""
    return {
        "name": "cpp",
        "experimental": True,
        "available": is_available(),
        # dtype / device metadata (v1.21): float64/cpu only.
        "dtype": "float64",
        "device": "cpu",
        "supported_dtypes": SUPPORTED_DTYPES,
        "supported_devices": SUPPORTED_DEVICES,
        # Layered capabilities (single source of truth: the tuples above).
        "raw_kernels": RAW_KERNELS,
        "kernels": RAW_KERNELS,  # backwards-compatible alias
        # The element types the seven raw kernels accept, reported
        # separately from ``supported_dtypes`` because they are separate
        # facts (Phase I, milestone I2). The raw kernels are handle-free and
        # take ``double*`` only, so they are float64-only permanently; a
        # caller must not read overall native dtype support off this row,
        # and must not read this limitation off ``supported_dtypes``.
        "raw_kernel_dtypes": RAW_KERNEL_DTYPES,
        "storage_object": "NativeStorage",
        "tensor_view": "NativeTensorView",
        "tensor_core": "NativeTensorCore",
        "tensor_core_kernels": TENSOR_CORE_KERNELS,  # frozen historical registry
        "tensor_core_ops": TENSOR_CORE_OPS,          # complete, accurate inventory
        "tensor_object": "NativeTensor",
        "autograd_ops": AUTOGRAD_OPS,
        "native_modules": NATIVE_MODULES,
        "native_losses": NATIVE_LOSSES,
        "native_metrics": NATIVE_METRICS,   # reporting helpers, not ops (E7)
        "native_optimizers": NATIVE_OPTIMIZERS,
        "state_support": STATE_SUPPORT,
        "unsupported": UNSUPPORTED,
        # Accurate integration flags (replace the old ambiguous
        # tensor_integration / autograd_integration pair).
        "broadcasting": True,          # NativeTensorCore binary ops broadcast
        "native_autograd": True,       # NativeTensor has reverse-mode autograd
        "stable_framework_integration": False,  # never wired into tensorforge.Tensor
        "build_instructions": build_instructions(),
    }


class NativeStorage:
    """A C++-owned float64 buffer — the storage half of a future
    tensor runtime prototype.

    Not a Tensor: it has a size but no shape, no strides, and no
    connection to Tensor/autograd. Data moves in and out by copy
    (``copy_from`` / ``to_numpy``); the raw native pointer is never
    exposed. Call ``close()`` (or use it as a context manager) to
    release the native memory; operations on a closed storage raise
    RuntimeError, and closing twice is safe.
    """

    def __init__(self, size, dtype=None, device="cpu", *,
                 _zero_initialize=True, _trusted_dtype=False):
        """Allocate ``size`` float64 elements, **zero-initialized**.

        The zero-initializing default did not change in Phase H: every
        existing caller behaves exactly as before.

        ``_zero_initialize`` is a private, keyword-only escape hatch used
        by the ``_uninitialized`` classmethod below and by nothing else.
        It lives on ``__init__`` rather than on a separate construction
        path on purpose: both allocation kinds must pass through the one
        constructor so that **every** live-storage accounting hook in the
        test suite — each of which wraps ``NativeStorage.__init__`` — sees
        an uninitialized allocation exactly as it sees a zeroed one.

        ``_trusted_dtype`` is the second such hatch (Phase I, milestone I2)
        and exists for the same structural reason: the private typed
        constructors must reach the *one* allocation path rather than
        growing a second one that could drift from it. When it is set, the
        dtype is validated against the internal table
        (``_normalize_internal_dtype``) instead of the public registry, so
        an internal caller can allocate float32 while ``"float32"`` is
        still unsupported. It defaults to ``False``, so **every public
        caller is validated by ``normalize_dtype`` exactly as before** and
        gets the identical rejection for ``"float32"``.
        """
        self._handle = None  # so a failed __init__ still __del__s safely
        if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"size must be a positive int, got {size!r}")
        # dtype/device are validated *before* allocation, so an
        # unsupported request never leaks native memory.
        dtype = (_normalize_internal_dtype(dtype) if _trusted_dtype
                 else normalize_dtype(dtype))
        device = normalize_device(device)
        lib = _require_library()
        # Phase I (I1): allocate through the **typed** creators, uniformly.
        # One path is easier to prove correct than two, and the dtype the
        # constructor already validated is the dtype the storage is tagged
        # with, so the Python tag and the C++ tag cannot disagree.
        #
        # ``normalize_dtype`` still admits only ``"float64"``, so the code
        # this passes is always 0 today and every observable behavior —
        # zero-initialization, the size rejection, the MemoryError, the
        # handle, ``close()`` semantics, live-storage accounting — is
        # byte-for-byte what it was. The untyped exports remain part of the
        # ABI and keep their own tests; they did not go away because their
        # primary caller moved.
        create = (lib.tf_storage_create_typed if _zero_initialize
                  else lib.tf_storage_create_uninitialized_typed)
        handle = create(int(size), _DTYPE_CODES[dtype])
        if not handle:
            raise MemoryError(f"could not allocate native storage of size {size}")
        self._lib = lib
        self._handle = handle
        self._size = int(size)
        self._dtype = dtype
        self._device = device

    @classmethod
    def _uninitialized(cls, size, dtype=None, device="cpu"):
        """Allocate ``size`` float64 elements whose **initial contents are
        indeterminate** (Phase H, milestone H1).

        Identical to ``NativeStorage(size, ...)`` in every observable
        respect — argument validation, dtype/device normalization,
        allocation-failure handling, the ``MemoryError`` it raises, the
        handle it owns, ``close()`` semantics, exactly-once destruction,
        and live-storage accounting (it runs through the same
        ``__init__``) — except that the buffer is not written before it is
        returned.

        **Private on purpose.** This is an internal backend detail, not a
        public ``empty`` constructor: the only legitimate caller is a Core
        operation whose kernel has been *proved* to write every
        destination element before reading any of it. That audit is a
        table in ``docs/native_cpu_performance_design.md``, and every row
        of it is backed by a poison test. A caller that cannot point at
        its row uses ``NativeStorage(...)``.

        This is also the **one seam the poison tests wrap**: they replace
        this classmethod with a wrapper that calls it, fills the returned
        storage with a recognizable pattern, and returns that same
        storage — so the poison is in place before the real kernel runs
        without the runtime itself owning any poison control.

        Its ``dtype`` is trusted (Phase I, milestone I2): every caller
        passes either the ``"float64"`` default or a canonical tag read off
        a live storage, and a derived allocation must be able to match its
        source's dtype without asking the public registry for permission
        that source already has.
        """
        return cls(size, dtype=dtype, device=device, _zero_initialize=False,
                   _trusted_dtype=True)

    @classmethod
    def _typed(cls, size, dtype, device="cpu", *, zero_initialize=True):
        """Private: allocate ``size`` elements at an **internally
        representable** dtype (Phase I, milestone I2).

        The one deliberate entry point for float32 storage before milestone
        I9. It differs from ``NativeStorage(size, dtype=...)`` in exactly
        one respect — the dtype is validated against ``_DTYPE_CODES``
        rather than ``SUPPORTED_DTYPES`` — and in no other: the same size
        validation, the same typed C ABI creator, the same ``MemoryError``,
        the same handle, the same ``close()`` semantics, the same
        exactly-once destruction, and the same live-storage accounting,
        because it runs through the same ``__init__``.

        **Private on purpose, and narrowly scoped on purpose.** float32 is
        not a supported TensorForge dtype and no public constructor may
        produce one; this is the private/typed entry point the rollout rule
        (design §27.2) requires intermediate milestones to test through
        while the public boundary stays exactly where it is.
        """
        return cls(size, dtype=dtype, device=device,
                   _zero_initialize=zero_initialize, _trusted_dtype=True)

    @classmethod
    def _typed_from_array(cls, values, dtype, device="cpu"):
        """Private: storage at ``dtype`` holding a copy of ``values``
        (Phase I, milestone I2).

        The typed counterpart of ``from_array``, with the identical
        conversion contract — the host input is converted to the storage's
        element type and flattened in C order — and the identical failure
        behavior: a failed copy closes the storage rather than returning a
        partly written buffer.
        """
        canonical = _normalize_internal_dtype(dtype)
        array = np.ascontiguousarray(
            values, dtype=_DTYPE_NUMPY[canonical]).ravel()
        storage = cls._typed(int(array.size), canonical, device=device,
                             zero_initialize=False)
        try:
            storage.copy_from(array)
        except BaseException:
            storage.close()
            raise
        return storage

    @classmethod
    def from_array(cls, values, dtype=None, device="cpu"):
        """Create storage sized to ``values`` and copy them in.

        The input is converted to contiguous values of the **requested**
        dtype and flattened in C order; ``dtype``/``device`` default to
        ``"float64"``/``"cpu"`` and are validated against the public
        registry, so ``"float32"`` is still rejected here.

        This is the explicit **host-to-native conversion boundary** and it
        has always converted: a Python list or an int64 array becomes
        native storage of the requested dtype. That is not a tensor cast —
        no native tensor changes dtype and none can (design §9.4). Phase I
        changes only *which* target the conversion has, and the dtype is
        never inferred from the input: ``dtype=None`` still means
        ``"float64"``, so handing this a float32 array without asking for
        float32 still produces float64 storage.

        H1: the allocation is uninitialized because ``copy_from`` writes
        **every** element (``tf_storage_copy_from`` loops over the whole
        ``storage->size``) and the size is taken from the same array, so
        no element can survive unwritten. A failed copy closes the
        storage rather than returning a partly written buffer.
        """
        # Normalized first, publicly, so the conversion target is known
        # before any allocation and an unsupported dtype is rejected before
        # NumPy is asked to do any work at all.
        canonical = normalize_dtype(dtype)
        array = np.ascontiguousarray(
            values, dtype=_DTYPE_NUMPY[canonical]).ravel()
        # empty input fails size validation; dtype/device validated too
        storage = cls._uninitialized(int(array.size), dtype=canonical,
                                     device=device)
        try:
            storage.copy_from(array)
        except BaseException:
            storage.close()
            raise
        return storage

    @property
    def size(self):
        """Number of elements the storage holds — a **logical element
        count** at every dtype, never a byte count."""
        return self._size

    @property
    def dtype(self):
        """The element type tag (``"float64"``, or ``"float32"`` for the
        private typed storage milestone I2 introduced). Readable after
        close."""
        return self._dtype

    @property
    def _numpy_dtype(self):
        """The NumPy type this storage's elements physically are, from the
        one private table. Private: it is an implementation detail of the
        transfer boundary, never a public dtype object."""
        return _DTYPE_NUMPY[self._dtype]

    @property
    def device(self):
        """The device tag (``"cpu"``). Readable after close."""
        return self._device

    def _require_open(self):
        if self._handle is None:
            raise RuntimeError("this NativeStorage has been closed")
        return self._handle

    def fill(self, value):
        """Set every element to ``value``."""
        self._lib.tf_storage_fill(self._require_open(), float(value))

    def copy_from(self, values):
        """Copy ``values`` into the storage.

        The input is converted to contiguous values of **this storage's**
        dtype and flattened in C order; it must contain exactly ``size``
        elements.

        The conversion is the host-to-native one ``from_array`` documents:
        a Python list or an array of another type is converted here, once,
        on the way in. Below this line nothing converts — the C ABI receives
        a buffer whose element type is checked against the storage's dtype
        by ``_host_pointer`` and would reject a mismatch rather than
        reinterpret it.
        """
        handle = self._require_open()
        array = np.ascontiguousarray(values, dtype=self._numpy_dtype).ravel()
        if array.size != self._size:
            raise ValueError(
                f"copy_from needs exactly {self._size} values, got {array.size}"
            )
        self._lib.tf_storage_copy_from(
            handle, _host_pointer(array, self._dtype))

    def to_numpy(self):
        """Return a new, independent 1-D copy of the contents, as a NumPy
        array of **exactly this storage's dtype**.

        Never widened on the way out: a float32 storage produces a float32
        array, because a widened result would silently claim precision the
        storage does not have (design §9.4)."""
        handle = self._require_open()
        out = np.empty(self._size, dtype=self._numpy_dtype)
        self._lib.tf_storage_copy_to(handle, _host_pointer(out, self._dtype))
        return out

    def close(self):
        """Release the native memory. Safe to call more than once."""
        if self._handle is not None:
            self._lib.tf_storage_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        # Defensive cleanup only — correctness never depends on when
        # (or whether) the garbage collector runs; use close().
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        state = "closed" if self._handle is None else f"size={self._size}"
        return f"NativeStorage({state})"


class NativeTensorView:
    """A logical view: NativeStorage plus shape/stride/offset metadata.

    This binds the two halves of the backend runtime prototype — the
    v0.7 layout contract and the v0.8 C++-owned buffer — into one
    object that knows which storage element each logical index means.
    Its first (and so far only) operation is contiguous
    materialization. Not a full Tensor: no math ops, no autograd, no
    connection to tensorforge.Tensor.

    Views never own the storage; closing the storage makes the view's
    operations raise. Reachable offsets are bounds-checked at
    construction, so a valid view can never read outside its storage
    (negative strides included).
    """

    def __init__(self, storage, shape, strides=None, offset=0):
        if not isinstance(storage, NativeStorage):
            raise TypeError(
                f"storage must be a NativeStorage, got {type(storage).__name__}"
            )
        storage._require_open()  # a closed storage cannot back a view
        dims, stride_tuple, offset_value, count, contiguous = _normalized_layout(
            shape, strides=strides, offset=offset
        )
        self._bind(storage, dims, stride_tuple, offset_value, count, contiguous)

    @classmethod
    def _from_validated(cls, storage, dims, strides, offset):
        """Private: a view over layout metadata this module already
        normalized (Phase H, milestone H3).

        ``dims`` and ``strides`` must be tuples of exact ``int`` that came
        from ``_as_shape`` or from a live view's own ``shape``/``strides``,
        and ``offset`` must be an exact ``int``. Under that precondition
        the normalization the public constructor performs is a no-op — it
        would re-check a tuple this module produced one construction
        earlier — so it is skipped, and **nothing else is**.

        The element count and the contiguity flag are deliberately
        *derived here* rather than accepted as arguments: a caller that
        cannot pass them cannot pass an inconsistent pair, which is why
        this is a separate constructor rather than a ``validated=True``
        flag on the public one. The storage's open state and the full
        reachable-offset bounds check are performed exactly as the public
        constructor performs them, because neither is a property of the
        metadata alone — a derived layout can still be handed a storage
        that has since been closed, and bounds depend on the storage size.
        """
        view = cls.__new__(cls)
        storage._require_open()  # a closed storage cannot back a view
        view._bind(
            storage, dims, strides, offset,
            _numel_checked(dims),
            strides == _row_major_strides_checked(dims),
        )
        return view

    def _bind(self, storage, dims, strides, offset, count, contiguous):
        """Bounds-check ``dims``/``strides``/``offset`` against ``storage``
        and bind this view's immutable layout.

        Shared by both constructors so the reachable-offset check and the
        field assignment can never drift apart between them. Every field
        assigned here is assigned exactly once in a view's lifetime and is
        never reassigned — not by any operation, and not by ``close()``,
        which is what makes ``_native_layout``'s cache safe.
        """
        # Bounds: each dimension contributes between 0 and
        # (dim - 1) * stride to the offset — negative strides make the
        # low end move. The whole reachable range must fit in storage.
        low = high = offset
        for dim, stride in zip(dims, strides):
            contribution = (dim - 1) * stride
            if contribution >= 0:
                high += contribution
            else:
                low += contribution
        if low < 0 or high > storage.size - 1:
            raise ValueError(
                f"view reaches storage offsets [{low}, {high}], outside "
                f"the valid range [0, {storage.size - 1}]"
            )

        self._storage = storage
        self._shape = dims
        self._strides = strides
        self._offset = offset
        self._numel = count
        self._contiguous = contiguous
        # Lazily built by _native_layout(); see its docstring for why this
        # memoization cannot go stale and why it is not built eagerly.
        self._layout_cache = None
        # Lazily built by _native_layout_pointers() (Phase H, H7): the
        # ``c_int64`` pointers into the arrays above, which is what the
        # strided C ABI actually takes. Same immutability argument, same
        # reason it cannot go stale — it is derived from the same layout
        # that is assigned exactly once, here, and never reassigned.
        self._layout_pointer_cache = None

    def _native_layout(self):
        """This view's ``(shape, strides)`` as the ``int64`` arrays the
        strided C ABI takes — built at most once per view (Phase H, H3).

        **Why this cannot go stale.** The arrays are a pure function of
        ``self._shape`` and ``self._strides``, and a view's layout is
        immutable: both are assigned exactly once, in ``_bind``, and no
        code path anywhere reassigns them. A view never changes shape —
        reshaping, transposing, and narrowing all produce *new* views over
        the same storage — so there is no mutation for an invalidation
        design to guard against. This is memoization of a pure function of
        immutable state, not a cache with a coherence problem.

        **Why the arrays are read-only.** They describe this view's
        metadata, and handing a mutable alias of them to a kernel caller
        would let that caller silently change what every subsequent native
        call believes this view's layout to be. ``writeable = False`` makes
        that a raised ``ValueError`` instead. The C ABI only ever *reads*
        shape and stride arrays, so nothing legitimate is lost, and a
        read-only C-contiguous ``int64`` array still satisfies the
        ``ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")`` argtype.

        **Why it is lazy.** The contiguous fast-path kernels take a flat
        element count and an offset, not shape/stride arrays, so the
        dominant training path never calls this at all. Building the pair
        eagerly in ``_bind`` would put two NumPy objects on every view
        — including the many short-lived ones a training step allocates —
        to serve calls most of them never make.

        The arrays hold plain integers copied out of the layout tuples.
        They contain no native pointer and no reference to the storage, so
        they cannot keep native memory alive.
        """
        arrays = self._layout_cache
        if arrays is None:
            shape_array = np.asarray(self._shape, dtype=np.int64)
            strides_array = np.asarray(self._strides, dtype=np.int64)
            shape_array.flags.writeable = False
            strides_array.flags.writeable = False
            arrays = (shape_array, strides_array)
            self._layout_cache = arrays
        return arrays

    def _native_layout_pointers(self):
        """This view's ``(shape, strides)`` as the **typed int64 pointers**
        the strided C ABI takes — built at most once per view (Phase H, H7).

        H3 gave every view an immutable, read-only ``int64`` NumPy pair and
        stopped rebuilding it. What it could not remove is what happens to
        that pair at each *call*: bound as ``ndpointer``, every position was
        re-verified (is it an ndarray, is its dtype exactly int64, is it
        C-contiguous) and then converted to a pointer, once per argument,
        once per call, forever. This memoizes the conversion the same way H3
        memoized the construction, and for the same reason: it is a pure
        function of state that is assigned exactly once and never reassigned.

        **The NumPy arrays remain the owning buffers, and remain read-only.**
        Nothing is copied and no second description of this view's layout is
        created — a duplicate could in principle disagree with the first,
        while a pointer *into* the one array cannot. ``ndarray.ctypes.data_as``
        stores the array on the pointer it returns (``ptr._arr``), so the
        buffer is reachable for as long as the pointer is and cannot be freed
        underneath a native call. ``ctypes.POINTER(...).from_address(...)``
        would be measurably cheaper to build and is **deliberately not used**:
        it produces a pointer with no reference to its owner, which is exactly
        the dangling-capable object this contract forbids.

        **The rank comes from the same object as the pointers.** Each array
        was built from ``self._shape`` / ``self._strides``, so its length *is*
        ``self.ndim``; a caller that takes the pointers from here and the rank
        from ``self.ndim`` is reading one immutable layout twice, not
        combining two. That is the invariant the C ABI depends on and the one
        ``ndpointer`` never checked — it sees a pointer and an ``ndim`` and
        has no access to the Python object's length.

        Lazy for H3's reason: the contiguous fast-path kernels take a flat
        count and an offset, so the dominant training path never calls this.
        Building it eagerly would put four objects on every short-lived view
        to serve calls most of them never make.
        """
        pointers = self._layout_pointer_cache
        if pointers is None:
            shape_array, strides_array = self._native_layout()
            pointers = (
                shape_array.ctypes.data_as(_LAYOUT_POINTER),
                strides_array.ctypes.data_as(_LAYOUT_POINTER),
            )
            self._layout_pointer_cache = pointers
        return pointers

    @classmethod
    def from_array(cls, values):
        """Create a contiguous view over new storage holding ``values``.

        The array's shape is preserved; its data is copied into a fresh
        NativeStorage.
        """
        array = np.ascontiguousarray(values, dtype=np.float64)
        storage = NativeStorage.from_array(array)
        return cls(storage, array.shape)

    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def offset(self):
        return self._offset

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def numel(self):
        return self._numel

    @property
    def contiguous(self):
        return self._contiguous

    def to_numpy(self):
        """Materialize the logical view into a fresh NumPy array of
        shape ``self.shape`` (row-major), copied element by element by
        the native materialization kernel.

        The result's NumPy dtype is **exactly the storage's** — a view has
        no dtype of its own and never could: it borrows the storage that
        holds the one authority (design §5.1)."""
        handle = self._storage._require_open()
        dtype = self._storage.dtype
        out = np.empty(self._numel, dtype=self._storage._numpy_dtype)
        # ``out`` is still checked at every call — it is real data this call
        # writes into and returns — but by ``_host_pointer`` against the
        # storage's dtype rather than by a fixed argtypes binding (I2). The
        # layout pair is trusted, and both pointers and the rank below come
        # from this one view (H7).
        shape_pointer, strides_pointer = self._native_layout_pointers()
        self._storage._lib.tf_storage_materialize(
            handle,
            _host_pointer(out, dtype),
            shape_pointer,
            strides_pointer,
            self._offset,
            len(self._shape),
        )
        return out.reshape(self._shape)

    def contiguous_copy(self):
        """Materialize into a new NativeStorage in row-major order.

        Always copies, even if the view is already contiguous — the
        result is an independent storage the caller owns (and must
        close).

        **Native storage-to-storage** (E3.1): the elements are gathered
        by ``tf_core_contiguous_copy`` straight from this view's storage
        into the fresh allocation, so tensor data never leaves native
        memory. Only the shape/stride arrays cross as ctypes metadata. A
        failed gather closes the new storage before propagating, so no
        half-filled allocation escapes.
        """
        # A derived allocation at **this view's storage's** dtype (Phase I,
        # milestone I2). The tag is not a caller's request — it was
        # validated when the source storage was created — so it takes the
        # trusted private constructor rather than being re-checked against
        # the public registry, which would refuse to let a float32 view be
        # copied at all. Zero-initialized exactly as before.
        storage = NativeStorage._typed(
            self._numel, self._storage.dtype, device=self._storage.device
        )
        try:
            shape_pointer, strides_pointer = self._native_layout_pointers()
            storage._lib.tf_core_contiguous_copy(
                self._storage._require_open(),
                storage._require_open(),
                shape_pointer,
                strides_pointer,
                self._offset,
                len(self._shape),
            )
        except BaseException:
            storage.close()
            raise
        return storage

    def __repr__(self):
        return (
            f"NativeTensorView(shape={self._shape}, strides={self._strides}, "
            f"offset={self._offset}, contiguous={self._contiguous})"
        )


def _contiguous_view(storage, dims):
    """A row-major contiguous view of ``dims`` at offset 0 over ``storage``.

    The shape of every freshly allocated tensor core (Phase H, H3).
    ``dims`` must already be validated — every caller has just passed it
    through ``_as_shape`` and sized the storage from it — so the strides
    follow from the shape and the bounds check inside ``_from_validated``
    is the one that matters.
    """
    return NativeTensorView._from_validated(
        storage, dims, _row_major_strides_checked(dims), 0
    )


class NativeTensorCore:
    """The first native tensor runtime object: an owned NativeStorage
    plus a NativeTensorView describing its layout, composed into one
    thing you can create, inspect, materialize, and release.

    Still not tensorforge.Tensor: no math operations, no autograd, no
    backend dispatch. It is the foundation those would build on.

    Ownership model: a core created by from_array/zeros/full OWNS its
    storage — its ``close()`` releases the native memory. View
    operations (reshape, transpose, T, narrow) return cores that
    BORROW the same storage: closing a borrowing view closes only that
    view, never the shared memory, so sibling views stay usable.
    Closing the owner releases the memory for every core sharing it,
    after which their data operations raise too. close() is
    idempotent everywhere; metadata properties stay readable after
    close, matching NativeStorage.size.
    """

    def __init__(self, storage, view, owns_storage=True):
        """Advanced constructor; prefer from_array / zeros / full.

        ``view`` must describe ``storage``. With ``owns_storage=True``
        this core's close() releases the storage; view operations pass
        False so shared storage is never freed by a view.
        """
        if not isinstance(storage, NativeStorage):
            raise TypeError(
                f"storage must be a NativeStorage, got {type(storage).__name__}"
            )
        if not isinstance(view, NativeTensorView) or view._storage is not storage:
            raise ValueError("view must be a NativeTensorView over the given storage")
        self._storage = storage
        self._view = view
        self._owns_storage = bool(owns_storage)
        self._closed = False

    # -- constructors -------------------------------------------------

    @classmethod
    def from_array(cls, values, dtype=None, device="cpu"):
        """A contiguous tensor holding a copy of ``values``, with the
        array's shape preserved. ``dtype``/``device`` default to
        ``"float64"``/``"cpu"``; unsupported values are rejected, and the
        host input is converted once to the requested dtype at this
        explicit host-to-native boundary (never inferred from the input —
        see ``NativeStorage.from_array``)."""
        canonical = normalize_dtype(dtype)
        array = np.ascontiguousarray(values, dtype=_DTYPE_NUMPY[canonical])
        # empty input fails here; dtype/device validated in the storage
        storage = NativeStorage.from_array(array, dtype=canonical,
                                           device=device)
        return cls(storage, _contiguous_view(storage, _as_shape(array.shape)))

    @classmethod
    def zeros(cls, shape, dtype="float64", device="cpu", *,
              _trusted_dtype=False):
        """A row-major contiguous tensor of ``shape``, all zeros
        (native storage is zero-initialized, so no fill pass runs).
        ``dtype``/``device`` default to ``"float64"``/``"cpu"``;
        unsupported values are rejected.

        ``_trusted_dtype`` is a private, keyword-only hatch (Phase I,
        milestone I4) and is exactly ``NativeStorage.__init__``'s, one
        layer up and for the same structural reason: the operations that
        must allocate a **zeroed** output at an operand's own dtype —
        ``sum`` and ``narrow_backward``, the two the H1 audit rejected
        from the uninitialized path — have to reach *this* constructor
        rather than growing a second one that could drift from it on the
        shape validation, the storage ownership, the contiguous view, the
        ``MemoryError``, the live-storage accounting, or the failure
        ordering. When it is set the dtype is validated against the
        internal table instead of the public registry.

        Setting it is sound wherever it is set, and the argument is
        ``_uninitialized``'s from I2: the dtype passed is never a caller's
        request but a canonical tag read off a live storage that was
        validated when it was created, and a derived allocation must be
        able to match its operand. It defaults to ``False``, so **every
        public caller is validated exactly as before** and
        ``zeros(..., dtype="float32")`` raises unchanged. Broadening a
        derived allocation is not broadening public construction."""
        dims = _as_shape(shape)  # validates shape by the v0.7 rules
        storage = NativeStorage(
            _numel_checked(dims), dtype=dtype, device=device,
            _trusted_dtype=_trusted_dtype,
        )
        return cls(storage, _contiguous_view(storage, dims))

    @classmethod
    def _uninitialized(cls, shape, dtype="float64", device="cpu"):
        """A row-major contiguous tensor of ``shape`` whose element values
        are **indeterminate** (Phase H, milestone H1).

        The private counterpart of ``zeros``: identical shape validation,
        identical storage ownership, identical ``close()`` semantics — it
        simply skips the zero-fill pass, which is a full extra write over
        the output.

        **Only for a destination a kernel completely overwrites.** Every
        call site is listed, with its proof and its poison test, in the
        H1 audit table in ``docs/native_cpu_performance_design.md``. This
        is not a public ``empty`` constructor and nothing in
        ``tensorforge.experimental`` or the stable framework exposes it;
        an operation that accumulates into its output, scatters into it,
        or leaves any element untouched must keep using ``zeros``.

        Its ``dtype`` is trusted (Phase I, milestone I2), inherited from
        ``NativeStorage._uninitialized``: every one of the call sites in
        the H1 audit table passes ``dtype=<operand>.dtype``, a canonical
        tag read off a storage that was validated when it was created, and
        an operation's freshly allocated output must be able to match its
        operand's dtype. A **public** constructor that reaches this helper
        validates publicly first — see ``full`` below.
        """
        dims = _as_shape(shape)  # validates shape by the v0.7 rules
        storage = NativeStorage._uninitialized(
            _numel_checked(dims), dtype=dtype, device=device
        )
        return cls(storage, _contiguous_view(storage, dims))

    @classmethod
    def _typed(cls, shape, dtype, device="cpu", *, zero_initialize=True):
        """Private: a row-major contiguous tensor of ``shape`` at an
        **internally representable** dtype (Phase I, milestone I2).

        The core-level counterpart of ``NativeStorage._typed``, and private
        for the same reason: it is how the internal float32 paths are
        reached and tested while ``"float32"`` is still unsupported. Shape
        validation, storage ownership, the contiguous view, and ``close()``
        semantics are ``zeros``'s exactly."""
        dims = _as_shape(shape)  # validates shape by the v0.7 rules
        storage = NativeStorage._typed(
            _numel_checked(dims), dtype, device=device,
            zero_initialize=zero_initialize,
        )
        return cls(storage, _contiguous_view(storage, dims))

    @classmethod
    def _typed_from_array(cls, values, dtype, device="cpu"):
        """Private: a contiguous tensor at ``dtype`` holding a copy of
        ``values``, with the array's shape preserved (Phase I, I2).

        The typed counterpart of ``from_array``; see ``_typed`` for why it
        is private."""
        canonical = _normalize_internal_dtype(dtype)
        array = np.ascontiguousarray(values, dtype=_DTYPE_NUMPY[canonical])
        storage = NativeStorage._typed_from_array(array, canonical,
                                                  device=device)
        try:
            return cls(storage, _contiguous_view(storage,
                                                 _as_shape(array.shape)))
        except BaseException:
            storage.close()
            raise

    @classmethod
    def _typed_full(cls, shape, value, dtype, device="cpu"):
        """Private: a row-major contiguous tensor of ``shape`` filled with
        ``value`` at an **internally representable** dtype (Phase I, I4).

        The dtype-preserving counterpart of ``full``, and the **one**
        implementation both share — ``full`` is this method behind the
        public dtype gate. It exists because a backward formula has to
        materialize its constants at the *graph's* dtype: ``-1`` for a
        negation, ``1/count`` for mean backward, and any other literal a
        derivative needs. Building those at float64 and meeting a float32
        operand would be a mixed-dtype request, which the runtime refuses
        (design §9); building them through the public ``full`` is
        impossible while float32 is unsupported. So the constant is built
        here, from the operand's own tag.

        The scalar itself crosses the ABI as a ``double`` and is narrowed
        **once**, before the fill loop, inside ``tf_storage_fill`` (design
        §7.4). Converting a scalar argument is not casting a tensor.

        Private for ``_typed``'s reason, and it does not widen public
        construction one inch: ``full`` still calls ``normalize_dtype``
        first, so ``full(..., dtype="float32")`` raises exactly as it did.

        H1: allocated uninitialized because ``tf_storage_fill`` writes every
        element of the storage, so the zero-fill would be immediately and
        completely overwritten. It reaches that allocation through
        ``_uninitialized`` — the seam the poison tests wrap — rather than
        around it, so the H1 coverage proof for ``full`` is untouched at
        both widths. ``float(value)`` is evaluated **before** the
        allocation, so a bad value allocates nothing; a failed fill closes
        the tensor.
        """
        fill_value = float(value)  # reject a bad value before allocating
        tensor = cls._uninitialized(shape, dtype=dtype, device=device)
        try:
            tensor._storage.fill(fill_value)
        except BaseException:
            tensor.close()
            raise
        return tensor

    @classmethod
    def full(cls, shape, value, dtype="float64", device="cpu"):
        """A row-major contiguous tensor of ``shape`` filled with
        ``value`` (anything float() accepts). ``dtype``/``device`` default
        to ``"float64"``/``"cpu"``; unsupported values are rejected.

        H1: allocated uninitialized because ``tf_storage_fill`` writes
        every element of the storage, so the zero-fill would be
        immediately and completely overwritten. ``float(value)`` is
        evaluated **before** the allocation, so a bad value allocates
        nothing; a failed fill closes the tensor."""
        # Public validation, explicitly, before the private constructor runs
        # (Phase I, milestone I2). ``_uninitialized`` trusts its dtype so a
        # derived allocation can match its operand's; a **public**
        # constructor must not inherit that trust. At I2 the reason was also
        # that ``tf_storage_fill`` was float64-only; I4 generalized that
        # export, so the gate now rests on the only reason that ever
        # mattered — float32 is not a supported TensorForge dtype, and no
        # public constructor may produce one until the registry moves at I9.
        # Broadening a lower primitive does not broaden public construction.
        return cls._typed_full(shape, value, normalize_dtype(dtype),
                               device=device)

    # -- metadata (readable even after close) --------------------------

    @property
    def shape(self):
        return self._view.shape

    @property
    def strides(self):
        return self._view.strides

    @property
    def offset(self):
        return self._view.offset

    @property
    def ndim(self):
        return self._view.ndim

    @property
    def numel(self):
        return self._view.numel

    @property
    def contiguous(self):
        return self._view.contiguous

    @property
    def dtype(self):
        """The element type tag, delegated to this core's storage
        (``"float64"``). A view shares its storage, so it reports the same
        dtype as its owner. Readable after close, like ``shape``."""
        return self._storage.dtype

    @property
    def device(self):
        """The device tag, delegated to this core's storage (``"cpu"``). A
        view shares its storage, so it reports the same device as its
        owner. Readable after close, like ``shape``."""
        return self._storage.device

    @property
    def storage(self):
        """The owned NativeStorage (read-only access to the object)."""
        return self._storage

    @property
    def view(self):
        """The NativeTensorView describing this tensor's layout."""
        return self._view

    # -- data operations ------------------------------------------------

    def _require_open(self):
        if self._closed:
            raise RuntimeError("this NativeTensorCore has been closed")

    def to_numpy(self):
        """Materialize into a fresh, independent NumPy array of
        ``self.shape``."""
        self._require_open()
        return self._view.to_numpy()

    def contiguous_copy(self):
        """A new, independent NativeTensorCore with the same values in
        row-major contiguous storage. Always copies, even when this
        tensor is already contiguous.

        **Native storage-to-storage** (E3.1): the fresh output is
        allocated first and ``tf_core_contiguous_copy`` gathers this
        (possibly strided, possibly offset) view straight into it, so
        **tensor data never round-trips through a NumPy host buffer**.
        Only the shape/stride arrays cross the boundary, as ctypes
        metadata. This is the shared helper behind every Policy-B
        copy-then-compute path (softmax, conv2d, maxpool2d), the
        differentiable ``contiguous_copy`` operation, ``NativeFlatten``,
        and ``NativeParameter`` construction.

        The result owns its storage, is contiguous at offset 0, aliases
        nothing, and leaves this tensor unchanged. A failed gather closes
        the freshly allocated output before propagating, so no partially
        initialized core escapes."""
        self._require_open()
        # H1 uninitialized — contiguous_copy: core_unary identity assigns every one of the numel elements.
        out = NativeTensorCore._uninitialized(
            self.shape, dtype=self.dtype, device=self.device
        )
        try:
            shape_ptr, strides_ptr = self._layout_pointers()
            self._storage._lib.tf_core_contiguous_copy(
                self._storage._require_open(),
                out._storage._require_open(),
                shape_ptr, strides_ptr, self.offset, self.ndim,
            )
        except BaseException:
            out.close()
            raise
        return out

    # -- native compute (arithmetic happens in C++ over storage) ---------

    def _layout_arrays(self):
        """This core's ``(shape, strides)`` int64 arrays for the strided
        C ABI, delegated to its view's per-view cache (Phase H, H3).

        The arrays are **read-only**: every caller only reads them, and
        marking them so is what makes returning the cached pair instead of a
        fresh one incapable of exposing shared mutable state. They are the
        buffers the kernels read; ``_layout_pointers`` below is how a kernel
        addresses them."""
        return self._view._native_layout()

    def _layout_pointers(self):
        """This core's ``(shape, strides)`` as the typed int64 pointers the
        strided C ABI takes, delegated to its view's per-view cache
        (Phase H, H7).

        The pointers address the very arrays ``_layout_arrays`` returns and
        keep them alive, so the rank to pass alongside them is this core's
        own ``ndim`` — one immutable layout, read twice, never two."""
        return self._view._native_layout_pointers()

    def relu(self):
        """max(x, 0) elementwise, computed by the native kernel reading
        this tensor's (possibly strided) view directly. Returns a new
        row-major contiguous NativeTensorCore.

        A contiguous input takes the flat fast-path kernel (a plain
        pointer loop); a strided view takes the generic odometer kernel.
        Both produce bit-for-bit identical results — the fast path is
        purely a traversal choice.

        H1: the output is allocated **uninitialized**. Both kernels write
        `dst[i]` for every `i` in `[0, numel)` — the flat loop directly,
        the odometer once per logical element — and the destination holds
        exactly `numel` elements, so no element survives unwritten. A
        failed kernel closes the output rather than returning it."""
        self._require_open()
        out = NativeTensorCore._uninitialized(
            self.shape, dtype=self.dtype, device=self.device
        )
        try:
            if self.contiguous:
                self._storage._lib.tf_core_relu_contiguous(
                    self._storage._require_open(),
                    out._storage._require_open(),
                    self.numel, self.offset,
                )
                return out
            shape_ptr, strides_ptr = self._layout_pointers()
            self._storage._lib.tf_core_relu(
                self._storage._require_open(),
                out._storage._require_open(),
                shape_ptr, strides_ptr, self.offset, self.ndim,
            )
            return out
        except BaseException:
            out.close()
            raise

    def _unary_compute(self, odometer_name, contiguous_name):
        """Shared plumbing for the unary compute ops (v3.11): require
        open, allocate the fresh contiguous output, then dispatch to
        the contiguous fast-path kernel or the generic odometer kernel
        by this tensor's contiguity — exactly relu's strategy, and the
        two paths are bit-for-bit identical.

        H1: the output is allocated **uninitialized**, on the same proof
        as ``relu`` — every one of these kernels assigns each of the
        destination's ``numel`` elements exactly once. A failed kernel
        closes the output."""
        self._require_open()
        out = NativeTensorCore._uninitialized(
            self.shape, dtype=self.dtype, device=self.device
        )
        try:
            if self.contiguous:
                getattr(self._storage._lib, contiguous_name)(
                    self._storage._require_open(),
                    out._storage._require_open(),
                    self.numel, self.offset,
                )
                return out
            shape_ptr, strides_ptr = self._layout_pointers()
            getattr(self._storage._lib, odometer_name)(
                self._storage._require_open(),
                out._storage._require_open(),
                shape_ptr, strides_ptr, self.offset, self.ndim,
            )
            return out
        except BaseException:
            out.close()
            raise

    def sqrt(self):
        """Elementwise square root, computed by the native kernel
        reading this tensor's (possibly strided) view directly. Returns
        a new row-major contiguous NativeTensorCore. IEEE float64
        semantics: negative inputs give NaN (no exception), signed
        zeros are preserved, +inf gives +inf, NaN propagates."""
        return self._unary_compute("tf_core_sqrt", "tf_core_sqrt_contiguous")

    def reciprocal(self):
        """Elementwise 1/x, computed by the native kernel reading this
        tensor's (possibly strided) view directly. Returns a new
        row-major contiguous NativeTensorCore. IEEE float64 semantics:
        ±0.0 gives ±inf (no exception, no warning), ±inf gives ±0.0,
        NaN propagates — the same values NumPy produces."""
        return self._unary_compute("tf_core_reciprocal",
                                   "tf_core_reciprocal_contiguous")

    def exp(self):
        """Elementwise e**x, computed by the native kernel reading this
        tensor's (possibly strided) view directly (Phase E, milestone
        E1). Returns a new **owning** row-major contiguous
        NativeTensorCore of the same shape; the input is not mutated and
        shares no storage with the result.

        Plain IEEE float64 ``std::exp`` — no clamping and no inserted
        bound: ``exp(0) == 1``, a large positive argument overflows to
        ``+inf``, a large negative one underflows toward ``+0``, ``+inf``
        gives ``+inf``, ``-inf`` gives ``+0``, and NaN propagates (the
        values NumPy produces, without its overflow warning).

        Graph-unaware, like every Core op: the differentiable surface is
        ``NativeTensor.exp()``."""
        return self._unary_compute("tf_core_exp", "tf_core_exp_contiguous")

    def log(self):
        """Elementwise natural logarithm, computed by the native kernel
        reading this tensor's (possibly strided) view directly (Phase E,
        milestone E2). Returns a new **owning** row-major contiguous
        NativeTensorCore of the same shape; the input is not mutated and
        shares no storage with the result.

        Plain IEEE float64 ``std::log`` — **no clamping, no inserted
        epsilon, no absolute value, and no domain rejection**:
        ``log(1) == 0``, values in ``(0, 1)`` give negative finite
        results, ``log(±0) == -inf``, ``log(negative)`` is NaN,
        ``log(+inf) == +inf``, and NaN propagates. Those are *values*, not
        errors — the same ones NumPy produces, without its warnings. A
        caller that needs a guarded logarithm must clamp its own input.

        Graph-unaware, like every Core op: the differentiable surface is
        ``NativeTensor.log()``."""
        return self._unary_compute("tf_core_log", "tf_core_log_contiguous")

    def softmax(self, axis=-1):
        """Numerically stable softmax over one ``axis`` (Phase E,
        milestone E3), computed by the fused native kernel.

        ``axis`` is a plain int (negative allowed, NumPy-style), validated
        by the existing ``_normalize_axis`` rules: a bool, a float, a
        string, ``None``, or an out-of-range value raises, and any axis on
        a rank-0 tensor raises — softmax requires rank >= 1. Validation
        runs **before any allocation**, so a rejected call allocates
        nothing. Returns a fresh **owning** row-major contiguous
        NativeTensorCore (offset 0) of the input's shape; the input is not
        mutated and shares no storage with the result.

        Per slice the kernel computes ``exp(x - max(x)) / sum(exp(x -
        max(x)))`` in one fused pass, all in float64 — the maximum shift
        keeps every exponent <= 0, so a large common offset cannot
        overflow. Exceptional values follow plain IEEE arithmetic with no
        special-casing: a NaN or ``+inf`` anywhere in a slice propagates
        through that slice's shift and sum, so the whole slice becomes
        NaN. Those are values, not errors.

        The C ABI is **contiguous-only**, so a non-contiguous input is
        materialized into a private contiguous copy (Policy B) that is
        closed the moment the native call returns; an already-contiguous
        input is passed through with its offset.

        Graph-unaware, like every Core op: the differentiable surface is
        ``NativeTensor.softmax()``."""
        return self._axis_fused_forward(axis, "tf_core_softmax_forward")

    def log_softmax(self, axis=-1):
        """Numerically stable log-softmax over one ``axis`` (Phase E,
        milestone E4), computed by the fused native kernel.

        Axis rules, validation order, and the output contract are exactly
        ``softmax``'s: a plain int (negative allowed), rejected before any
        allocation if it is a bool, a float, a string, ``None``, or out of
        range (so a rank-0 input is rejected too); the result is a fresh
        **owning** row-major contiguous NativeTensorCore (offset 0) of the
        input's shape, sharing no storage with the input and leaving it
        unmutated.

        Per slice the kernel computes ``(x - max(x)) - log(sum(exp(x -
        max(x))))`` in one fused pass, all in float64. It is **never**
        ``softmax(x).log()`` — no probability buffer is formed and no
        division happens, so a probability too small to represent (which
        would round to 0 and give ``log(0) == -inf``) still gets an
        accurate finite log-probability. Exceptional values follow plain
        IEEE arithmetic with no special-casing: a NaN or ``+inf`` in a
        slice makes that whole slice NaN, while a ``-inf`` gets ``-inf``
        and leaves its finite neighbours governed by the stable
        computation. Those are values, not errors.

        The C ABI is **contiguous-only**, so a non-contiguous input is
        materialized into a private contiguous copy (Policy B) that is
        closed the moment the native call returns; an already-contiguous
        input is passed through with its offset.

        Graph-unaware, like every Core op: the differentiable surface is
        ``NativeTensor.log_softmax()``."""
        return self._axis_fused_forward(axis, "tf_core_log_softmax_forward")

    def _axis_fused_forward(self, axis, kernel_name):
        """Shared plumbing for the fused axis-wise classification
        forwards (E3 ``softmax``, E4 ``log_softmax``).

        The two operations have the identical shape contract and the
        identical **contiguous-only** C ABI — source handle + offset,
        destination handle, and the ``(outer, axis_length, inner)``
        factorization of the reduction axis — so only the exported kernel
        symbol differs, and each caller names its own. Sharing keeps their
        axis normalization, their validate-before-allocate ordering, and
        their Policy-B ownership and failure cleanup provably in step
        rather than as two copies that could drift."""
        self._require_open()
        # Validate/normalize the axis first — nothing is allocated if this
        # raises. _normalize_axis rejects bool/non-int/out-of-range and
        # every axis on a rank-0 shape.
        normalized = _normalize_axis_checked(axis, self.shape)
        shape = self.shape
        # Python ints are arbitrary precision, so these products cannot
        # overflow here; the C ABI re-proves them in int64 anyway.
        outer = 1
        for extent in shape[:normalized]:
            outer *= extent
        axis_length = shape[normalized]
        inner = 1
        for extent in shape[normalized + 1:]:
            inner *= extent

        temporaries = []
        try:
            source = (
                self if self.contiguous else self._contiguous_temp(temporaries)
            )
            # H1 uninitialized — softmax/log_softmax: pass 2 assigns every (outer, k, inner) destination slot.
            out = NativeTensorCore._uninitialized(
                shape, dtype=self.dtype, device=self.device
            )
            try:
                getattr(self._storage._lib, kernel_name)(
                    source._storage._require_open(), source.offset,
                    out._storage._require_open(),
                    outer, axis_length, inner,
                )
            except BaseException:
                # The native call failed (e.g. an injected allocation
                # failure): discard the freshly allocated output so a
                # failed forward returns no half-built tensor.
                out.close()
                raise
            return out
        finally:
            # Close the private contiguous copy exactly once, whether the
            # call succeeded or raised — the caller's input is untouched.
            for temp in temporaries:
                temp.close()

    # -- fused cross-entropy (Phase E, E5: the graph-unaware Core
    #    contract E6 will build its autograd node on) --------------------

    def cross_entropy_forward(self, targets, reduction="mean"):
        """Fused multi-class cross-entropy over this tensor's logits.

        ``self`` is the ``(batch_size, num_classes)`` logits block — rank
        exactly 2, float64/cpu, open. The class axis is fixed at axis 1;
        there is deliberately no ``axis`` argument, no broadcasting, and
        no implicit reshape, so rank-1, rank-3, and flattened logits are
        rejected by shape.

        ``targets`` is a **one-dimensional sequence of integer class
        labels**, one per row — a list or tuple of Python ints, or a 1-D
        NumPy array of signed or unsigned integer dtype. Targets are not
        native tensors (the runtime has no integer dtype, design §6), and
        validation is **strict**: ``bool`` is rejected, floating-point
        values are rejected *including* integral ones like ``1.0``
        (nothing is silently truncated), and so are complex values,
        strings, bytes, nested/ragged sequences, rank-2 arrays, scalars,
        object arrays holding non-integers, values outside the int64
        range, and any label outside ``[0, num_classes)``. Type problems
        raise TypeError; shape, length, and range problems raise
        ValueError naming the offending index and value.

        The accepted labels are **copied into an independently owned,
        contiguous, read-only ``int64`` array** before anything is
        allocated — even when the caller already passed a contiguous
        ``int64`` array. Mutating the caller's list or array afterwards
        therefore cannot affect this forward or the backward built on it.

        ``reduction`` is exactly ``"mean"`` or ``"sum"``, validated by
        exact string match (no case or whitespace normalization, no
        coercion, no ``"none"``) — the ``NativeMSELoss`` precedent — and
        mapped once here to the small integer code the ABI carries.
        ``"sum"`` is the total of the per-example negative log
        likelihoods; ``"mean"`` divides that total once by
        ``batch_size`` (never by ``num_classes``).

        Returns a private ``_CrossEntropyForwardResult`` carrying:

        * ``loss`` — a fresh owning **scalar** core (shape ``()``), the
          repository's scalar convention;
        * ``probabilities`` — a fresh owning contiguous
          ``(batch_size, num_classes)`` core holding the softmax the
          backward needs, aliasing nothing;
        * ``targets`` — the owned ``int64`` copy;
        * ``reduction`` — the normalized name.

        The caller owns both cores and releases them with
        ``result.close()`` (or individually). The forward is **fused**:
        one kernel computes each row's maximum, its log-sum-exp, its
        probabilities, and its loss in a single deterministic pass — it
        never computes ``-log(probabilities[target])``, never forms a
        public softmax or log-softmax first, clamps nothing, and inserts
        no epsilon.

        The C ABI is **contiguous-only** for tensor data, so a
        non-contiguous logits view is materialized into a private
        contiguous copy (Policy B) closed the moment the native call
        returns; an already-contiguous view is passed through with its
        offset. Validation runs entirely before any allocation, and a
        failure at any later stage closes **every** object this method
        allocated, returns no partial result, and leaves the caller's
        logits open and unchanged.

        **Graph-unaware**, like every Core op: no autograd node is built,
        nothing claims graph ownership, and no version is recorded. The
        differentiable ``NativeTensor.cross_entropy`` is milestone E6 and
        does not exist yet."""
        self._require_open()
        if self.ndim != 2:
            raise ValueError(
                f"cross_entropy_forward requires 2-D (batch_size, "
                f"num_classes) logits, got shape {self.shape}"
            )
        if self.dtype != "float64" or self.device != "cpu":
            raise ValueError(
                f"cross_entropy_forward requires float64/cpu logits, got "
                f"{self.dtype}/{self.device}"
            )
        batch_size, num_classes = self.shape
        # Reduction first: it is the cheapest check and the one most
        # likely to be a caller typo, and normalizing it here keeps the
        # string off the ABI entirely.
        reduction_name, reduction_code = _normalize_reduction(
            reduction, "cross_entropy_forward"
        )
        # Then the targets — validated and copied before a single native
        # allocation, so a rejected call allocates nothing at all.
        target_copy = _prepare_class_targets(
            targets, batch_size, num_classes, "cross_entropy_forward"
        )
        # Python ints are arbitrary precision, so this check cannot itself
        # overflow; the C ABI re-proves the product in int64 anyway.
        if batch_size * num_classes > _INT64_MAX:
            raise ValueError(
                f"cross_entropy_forward logits element count "
                f"{batch_size * num_classes} exceeds the int64 range the "
                f"native runtime addresses"
            )

        temporaries = []
        loss = None
        probabilities = None
        try:
            logits = (
                self if self.contiguous else self._contiguous_temp(temporaries)
            )
            # Deterministic allocation order: the scalar loss, then the
            # probability block. If the second allocation fails the first
            # is closed below, so a failed forward leaves nothing
            # half-built and returns no result object.
            # H1 uninitialized — cross_entropy loss: the kernel assigns the single scalar element.
            loss = NativeTensorCore._uninitialized(
                (), dtype=self.dtype, device=self.device
            )
            # H1 uninitialized — cross_entropy probabilities: pass 2 assigns every (batch, class) element.
            probabilities = NativeTensorCore._uninitialized(
                (batch_size, num_classes), dtype=self.dtype, device=self.device
            )
            self._storage._lib.tf_core_cross_entropy_forward(
                logits._storage._require_open(), logits.offset,
                target_copy, int(target_copy.size),
                loss._storage._require_open(),
                probabilities._storage._require_open(),
                batch_size, num_classes, reduction_code,
            )
            return _CrossEntropyForwardResult(
                loss, probabilities, target_copy, reduction_name
            )
        except BaseException:
            # Close whichever outputs were successfully allocated — never
            # rely on garbage collection for native memory, and never let
            # a half-built output escape.
            for allocated in (probabilities, loss):
                if allocated is not None:
                    allocated.close()
            raise
        finally:
            # Close the private contiguous copy (if any) exactly once,
            # whether the call succeeded or raised.
            for temp in temporaries:
                temp.close()

    def cross_entropy_backward(self, targets, upstream, reduction="mean"):
        """Gradient of the fused cross-entropy w.r.t. its logits (E5).

        ``self`` is the **saved probability core** the forward produced —
        rank 2, contiguous, float64/cpu, open. ``targets`` is that
        forward's owned ``int64`` copy (the internal trusted-copy
        contract: an independently owned, C-contiguous 1-D ``int64``
        NumPy array, re-checked here for dtype, layout, length, and class
        range before the ABI is entered, and re-checked *again* in C++
        for every index it dereferences). ``upstream`` is an open
        **one-element** NativeTensorCore — the scalar gradient flowing
        into the loss, whatever its logical shape (``()``, ``(1,)``,
        ``(1, 1)``, or a one-element view at a nonzero offset): its
        storage handle and offset are passed straight to the kernel, so
        the value is **never extracted through NumPy**. ``reduction`` is
        the forward's normalized ``"mean"`` or ``"sum"``.

        Returns a fresh **owning** row-major contiguous
        ``(batch_size, num_classes)`` core:

            grad[n, c] = upstream * (p[n, c] - [c == target_n]) / N

        with the ``/ N`` only for ``"mean"``. The logits are neither
        accepted nor reachable here — **backward never rereads them** and
        never recomputes a softmax, a log-softmax, or the forward loss.
        Neither the probabilities, the targets, nor the upstream is
        mutated, and a failure closes the freshly allocated gradient
        before propagating.

        Graph-unaware: this is a numerical helper, not gradient
        accumulation. The E6 autograd node will call it from its single
        input-gradient callback."""
        self._require_open()
        if self.ndim != 2:
            raise ValueError(
                f"cross_entropy_backward requires a 2-D (batch_size, "
                f"num_classes) probability core, got shape {self.shape}"
            )
        if not self.contiguous:
            raise ValueError(
                f"cross_entropy_backward requires contiguous saved "
                f"probabilities, got strides {self.strides} for shape "
                f"{self.shape}"
            )
        if self.dtype != "float64" or self.device != "cpu":
            raise ValueError(
                f"cross_entropy_backward requires a float64/cpu probability "
                f"core, got {self.dtype}/{self.device}"
            )
        batch_size, num_classes = self.shape
        # Only the ABI code is needed here; the name is the forward's.
        _, reduction_code = _normalize_reduction(
            reduction, "cross_entropy_backward"
        )
        target_copy = _require_target_copy(
            targets, batch_size, num_classes, "cross_entropy_backward"
        )
        if not isinstance(upstream, NativeTensorCore):
            raise TypeError(
                f"cross_entropy_backward requires a NativeTensorCore "
                f"upstream gradient, got {type(upstream).__name__}"
            )
        upstream._require_open()
        self._require_matching_metadata(upstream, "cross_entropy_backward")
        if upstream.numel != 1:
            raise ValueError(
                f"cross_entropy_backward requires a one-element upstream "
                f"gradient (the loss is a scalar), got shape "
                f"{upstream.shape} with {upstream.numel} elements"
            )
        if batch_size * num_classes > _INT64_MAX:
            raise ValueError(
                f"cross_entropy_backward gradient element count "
                f"{batch_size * num_classes} exceeds the int64 range the "
                f"native runtime addresses"
            )

        out = None
        try:
            # H1 uninitialized — cross_entropy backward: assigns every (batch, class) gradient element.
            out = NativeTensorCore._uninitialized(
                (batch_size, num_classes), dtype=self.dtype, device=self.device
            )
            self._storage._lib.tf_core_cross_entropy_backward(
                self._storage._require_open(), self.offset,
                target_copy, int(target_copy.size),
                upstream._storage._require_open(), upstream.offset,
                out._storage._require_open(),
                batch_size, num_classes, reduction_code,
            )
            return out
        except BaseException:
            # A failed backward returns no half-built gradient.
            if out is not None:
                out.close()
            raise

    def relu_backward(self, upstream):
        """The gradient of ``relu`` at this tensor's forward value:
        ``upstream`` where this tensor's element is ``> 0``, else ``0``
        (``x == 0`` blocks the gradient, the Python Tensor convention).

        A forward-shaped numerical kernel, not graph machinery — the core
        stays autograd-unaware; the NativeTensor layer calls this from its
        relu backward closure. Both operands may be strided views (each is
        read through its own strides/offset); shapes must match exactly
        (no broadcasting — the upstream gradient of an op always has the
        op's output shape). Returns a new row-major contiguous
        NativeTensorCore of this tensor's shape."""
        self._require_open()
        if not isinstance(upstream, NativeTensorCore):
            raise TypeError(
                f"relu_backward requires a NativeTensorCore upstream "
                f"gradient, got {type(upstream).__name__}"
            )
        upstream._require_open()
        self._require_matching_metadata(upstream, "relu_backward")
        if self.shape != upstream.shape:
            raise ValueError(
                f"relu_backward requires the upstream gradient shape to "
                f"match the input shape, got {upstream.shape} and {self.shape}"
            )
        # H1: uninitialized — the binary odometer assigns every one of the
        # destination's numel elements exactly once. A failed kernel
        # closes the output.
        out = NativeTensorCore._uninitialized(
            self.shape, dtype=self.dtype, device=self.device
        )
        try:
            shape_ptr, x_strides = self._layout_pointers()
            # Same shape, so only the upstream's *strides* differ; take
            # them from its own view cache rather than rebuilding. The rank
            # passed below is this tensor's, and the shapes were just proved
            # equal, so every pointer describes exactly ``self.ndim`` axes.
            u_strides = upstream._layout_pointers()[1]
            self._storage._lib.tf_core_relu_backward(
                self._storage._require_open(),
                upstream._storage._require_open(),
                out._storage._require_open(),
                shape_ptr, x_strides, u_strides,
                self.offset, upstream.offset, self.ndim,
            )
        except BaseException:
            out.close()
            raise
        return out

    def _require_matching_metadata(self, other, op_name):
        """Both operands of a binary/matmul op must share dtype and
        device; there is no implicit promotion and no automatic device
        move (see docs/native_dtype_device_metadata_design.md §8). Raises
        ValueError naming both dtype/device pairs on a mismatch. With only
        float64/cpu constructible today this guard cannot yet fire, but it
        is the enforced contract native autograd (Phase B) builds on."""
        if self.dtype != other.dtype or self.device != other.device:
            raise ValueError(
                f"{op_name} requires matching dtype and device, got "
                f"{self.dtype}/{self.device} and {other.dtype}/{other.device}"
            )

    def _binary_core_op(self, other, kernel_name, op_name):
        """Shared plumbing for add/subtract/multiply over tensor cores,
        with a three-way traversal dispatch (v1.17):

        A. **Same shape, both contiguous** — the v1.14 flat fast-path
           kernel (``<kernel_name>_contiguous``): a plain pointer loop
           over numel.
        B. **Same shape, either strided** — the generic odometer kernel
           (``<kernel_name>``): walks the shared shape with each
           operand's real strides.
        C. **Differing but broadcast-compatible shapes** — NumPy-style
           broadcasting. The output shape is inferred by
           ``broadcast_shapes`` and each operand is read through
           *broadcast strides* (§ native_broadcasting_design.md): a real
           axis keeps its stride, a stretched or left-padded size-1 axis
           gets stride 0 so the odometer re-reads one element instead of
           advancing. The very same generic odometer kernel from path B
           consumes those strides — a zero stride is broadcasting, no
           expanded operand is materialized. Incompatible shapes raise a
           clear ``ValueError`` naming both shapes.

        Paths A and B are bit-for-bit unchanged from before; only when the
        shapes actually differ does broadcasting engage. The output is
        always freshly allocated row-major contiguous storage.

        H1: every one of the three paths allocates its output
        **uninitialized**. All three write ``dst[i]`` for every ``i`` in
        ``[0, prod(out_shape))`` — the flat loop directly, the odometer
        once per logical output position — and the destination holds
        exactly that many elements. Broadcasting changes only how the
        *operands* are read (a zero stride re-reads one element); it does
        not skip an output position, so the coverage proof is the same.
        A failed kernel closes the output."""
        self._require_open()
        if not isinstance(other, NativeTensorCore):
            raise TypeError(
                f"{op_name} requires a NativeTensorCore operand, "
                f"got {type(other).__name__}"
            )
        other._require_open()
        self._require_matching_metadata(other, op_name)
        lib = self._storage._lib

        # Same-shape paths (A and B) — the exact-shape behavior, unchanged.
        if self.shape == other.shape:
            out = NativeTensorCore._uninitialized(
                self.shape, dtype=self.dtype, device=self.device
            )
            try:
                if self.contiguous and other.contiguous:
                    getattr(lib, kernel_name + "_contiguous")(
                        self._storage._require_open(),
                        other._storage._require_open(),
                        out._storage._require_open(),
                        self.numel, self.offset, other.offset,
                    )
                    return out
                shape_ptr, a_strides = self._layout_pointers()
                # Path B is the same-shape case, so the shared shape array
                # is this operand's; only the other operand's strides are
                # needed, and they come from its own view cache. The shapes
                # were just compared equal, so all three pointers describe
                # exactly the ``self.ndim`` axes passed below.
                b_strides = other._layout_pointers()[1]
                getattr(lib, kernel_name)(
                    self._storage._require_open(),
                    other._storage._require_open(),
                    out._storage._require_open(),
                    shape_ptr, a_strides, b_strides,
                    self.offset, other.offset, self.ndim,
                )
                return out
            except BaseException:
                out.close()
                raise

        # Broadcasting path (C) — differing shapes. broadcast_shapes
        # raises (naming both shapes) if they are incompatible, before
        # any output is allocated.
        out_shape = _broadcast_shapes_checked(self.shape, other.shape)
        out = NativeTensorCore._uninitialized(
            out_shape, dtype=self.dtype, device=self.device
        )
        try:
            out_ndim = len(out_shape)
            # All three descriptions belong to *this broadcast*, not to
            # either tensor, so none of them is cacheable anywhere: they are
            # built here, passed, and dropped (H7). ``_broadcast_strides``
            # returns one entry per output axis, so every vector below is
            # exactly ``out_ndim`` long by construction — the rank passed to
            # the kernel and the length of what it addresses are the same
            # number, read from the same ``out_shape``.
            shape_vector = _layout_vector(out_shape)
            a_strides = _layout_vector(
                _broadcast_strides(self.shape, self.strides, out_shape)
            )
            b_strides = _layout_vector(
                _broadcast_strides(other.shape, other.strides, out_shape)
            )
            getattr(lib, kernel_name)(
                self._storage._require_open(),
                other._storage._require_open(),
                out._storage._require_open(),
                shape_vector, a_strides, b_strides,
                self.offset, other.offset, out_ndim,
            )
        except BaseException:
            out.close()
            raise
        return out

    def add(self, other):
        """self + other elementwise, natively. Identical shapes, or
        NumPy-style broadcasting for compatible shapes (v1.17)."""
        return self._binary_core_op(other, "tf_core_add", "add")

    def subtract(self, other):
        """self - other elementwise, natively. Identical shapes, or
        NumPy-style broadcasting for compatible shapes (v1.17)."""
        return self._binary_core_op(other, "tf_core_subtract", "subtract")

    def multiply(self, other):
        """self * other elementwise, natively. Identical shapes, or
        NumPy-style broadcasting for compatible shapes (v1.17)."""
        return self._binary_core_op(other, "tf_core_multiply", "multiply")

    def matmul(self, other):
        """(m, n) @ (n, p) matrix multiplication over native storage.

        Both operands must be 2-D tensor cores; either may be a
        non-contiguous view (transposed, narrowed) — the kernel
        addresses each source through its own strides, so nothing is
        materialized first. Returns a new (m, p) row-major contiguous
        NativeTensorCore. No broadcasting.

        Phase H (H2): the native side ships **two** compute paths and
        picks between them inside ``tf_core_matmul``, from the stride
        metadata this method already passes down. A right operand whose
        column stride is 1 (any row-major operand, including the
        contiguous weight a Linear layer holds), with ``n >= 1`` and at
        least 8 columns, takes an ``i``-``k``-``j`` row sweep whose inner
        loop walks memory sequentially; everything else — a transposed
        right operand, a narrow result, an empty inner dimension — takes
        the generic ``i``-``j``-``k`` triple loop, which is the retained
        reference path and the case that loop order already suits. The
        choice is deterministic, allocates nothing, and cannot fail; a
        failed precondition is a fallback, never an error. There is no
        selector, environment variable, or public control over it, and
        the two paths are proved to agree bit for bit (see
        docs/native_cpu_performance_design.md §16.2).
        """
        self._require_open()
        if not isinstance(other, NativeTensorCore):
            raise TypeError(
                f"matmul requires a NativeTensorCore operand, "
                f"got {type(other).__name__}"
            )
        other._require_open()
        self._require_matching_metadata(other, "matmul")
        if self.ndim != 2:
            raise ValueError(
                f"matmul requires a 2-D left operand, got shape {self.shape}"
            )
        if other.ndim != 2:
            raise ValueError(
                f"matmul requires a 2-D right operand, got shape {other.shape}"
            )
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"inner dimensions do not match: "
                f"{self.shape} @ {other.shape} (need (m, n) @ (n, p))"
            )
        m, n = self.shape
        p = other.shape[1]
        # H1: uninitialized, and H2 kept it that way through both paths.
        # The generic path accumulates each dot product into a *local*
        # `sum` register and assigns `dst[i * p + j] = sum` for every
        # (i, j), so it never reads the destination at all. The H2 row
        # sweep does accumulate in the destination, but its k == 0 pass
        # *assigns* every element of every row it is about to work on
        # before any accumulation reads one — which is why its dispatch
        # predicate requires `n >= 1`. Either way no output value can
        # depend on what the buffer held beforehand, which the poison
        # tests prove on both paths. A failed kernel closes the output.
        out = NativeTensorCore._uninitialized(
            (m, p), dtype=self.dtype, device=self.device
        )
        try:
            self._storage._lib.tf_core_matmul(
                self._storage._require_open(),
                other._storage._require_open(),
                out._storage._require_open(),
                m, n, p,
                self.strides[0], self.strides[1],
                other.strides[0], other.strides[1],
                self.offset, other.offset,
            )
        except BaseException:
            out.close()
            raise
        return out

    # -- reductions (v1.19) ---------------------------------------------

    def sum(self, axis=None, keepdims=False):
        """Sum over ``axis`` (``None`` = all elements) natively, reading
        this tensor's (possibly strided) view directly. Returns a new
        owning row-major contiguous NativeTensorCore whose shape is
        ``reduce_shape(self.shape, axis, keepdims)``. Single integer or
        negative axis only; no broadcasting of the result, no autograd.

        Deterministic row-major accumulation order over the input; float
        sums are order-sensitive, so results match NumPy to a tolerance,
        not bit-for-bit (see docs/native_reductions_design.md).

        H6: the kernel now chooses between two traversals from the layout
        metadata this method already passes it — a flat block traversal
        when the source is row-major and the reduced axes form one
        contiguous run, the retained generic odometer otherwise. Both
        accumulate the same values into each destination cell in the same
        ascending order, so the choice is invisible here: the output shape,
        the write strides, the validation, and the ownership contract are
        exactly what they were, and there is no path selector to pass."""
        self._require_open()
        # self.shape is already validated; axis and keepdims are the
        # caller-supplied half and are still fully validated here.
        out_shape = _reduce_shape_checked(self.shape, axis, keepdims)
        # H1 REJECTED, and H6 CONFIRMED that rejection rather than revisiting
        # it — this output must stay zero-initialized. tf_core_sum
        # accumulates (`dst[out_pos] += src[in_pos]`) on *both* of its H6
        # traversals, so the zero is the additive identity the reduction
        # starts from, not a redundant write: every reduced axis folds many
        # inputs into one destination cell, and that cell is *read* on every
        # accumulation after the first. Giving this an uninitialized buffer
        # would return garbage. The block traversal's local accumulator is
        # seeded from the destination for exactly this reason, which is also
        # what keeps the export's accumulate-into semantics identical on the
        # two paths. H6 measured the zero-fill it would have removed: a
        # reduction's output is the *reduced* shape, so a (256, 256) axis-0
        # reduction zero-fills 2 KB while reading 512 KB — the fill is under
        # half a percent of the work, against a traversal that was 95 % of
        # it (docs/native_cpu_performance_design.md §16.6.6).
        #
        #
        # Phase I, milestone I4: the **same** zeroed constructor, with the
        # dtype trusted. ``self.dtype`` is not a caller's request but this
        # tensor's own canonical tag, read off a storage that was validated
        # when it was created, and a reduction's output must be able to match
        # its operand — exactly the trust ``_uninitialized`` has carried
        # since I2. Nothing else about the allocation changed, so H1's
        # rejection of the uninitialized path and every seam that observes it
        # are untouched, and public construction is not broadened one inch:
        # ``NativeTensorCore.zeros(..., dtype="float32")`` still raises.
        out = NativeTensorCore.zeros(out_shape, dtype=self.dtype,
                                     device=self.device, _trusted_dtype=True)
        # Everything after the allocation runs inside the same cleanup
        # boundary every other allocating Core op uses (compare
        # contiguous_copy): a failure in the write-stride construction, the
        # layout arrays, or the native call releases the freshly allocated
        # output before propagating, so no partially accumulated core
        # escapes and the release is explicit rather than left to a
        # refcount or the collector.
        try:
            if axis is None:
                reduced = set(range(self.ndim))
            else:
                reduced = {_normalize_axis_checked(axis, self.shape)}
            out_strides = _reduce_out_strides(
                self.shape, reduced, bool(keepdims), out_shape
            )
            # This tensor's own layout comes from its view's cache; the
            # write-strides are a property of *this reduction*, not of any
            # tensor, so they stay an operation-local vector. Both carriers
            # hold one entry per **input** axis — ``_reduce_out_strides``
            # returns ``len(in_shape)`` of them by construction — which is
            # the ``self.ndim`` passed below.
            shape_ptr, strides_ptr = self._layout_pointers()
            self._storage._lib.tf_core_sum(
                self._storage._require_open(),
                out._storage._require_open(),
                shape_ptr,
                strides_ptr,
                _layout_vector(out_strides),
                self.offset, self.ndim,
            )
        except BaseException:
            out.close()
            raise
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean over ``axis`` (``None`` = all elements) natively: the
        native ``sum`` scaled in place by ``1/count``, where ``count`` is
        ``numel`` for ``axis=None`` or ``shape[axis]`` for a single axis.
        Returns a new owning row-major contiguous NativeTensorCore. No
        NumPy touches the data; no autograd.

        **The scale factor is computed once in binary64 and narrowed once**
        (design §7.4). ``1.0 / count`` is a correctly-rounded Python float;
        it crosses the unchanged ``double`` ABI parameter of
        ``tf_storage_scale``; and the kernel converts it to the storage's
        element type **before** its loop. So a float32 mean is
        ``sum`` (accumulated in float32) times ``float(1/count)``, which is
        deterministic, identical on every platform, and independent of
        ``count``'s magnitude — where computing ``1.0f / count`` in float32
        instead would differ by up to one ULP for some counts. The sum
        itself accumulates in the tensor's own dtype, with no widening
        intermediate anywhere."""
        self._require_open()
        result = self.sum(axis=axis, keepdims=keepdims)
        # Same cleanup boundary as ``sum``'s: the summed output is already
        # allocated, so a failure in the count or in the in-place scale must
        # release it explicitly rather than leave it to a refcount.
        try:
            if axis is None:
                count = self.numel
            else:
                count = self.shape[_normalize_axis_checked(axis, self.shape)]
            # In-place native scale of the freshly summed output — no copy,
            # no NumPy round trip. count >= 1 always (dims are positive).
            result._storage._lib.tf_storage_scale(
                result._storage._require_open(), 1.0 / count
            )
        except BaseException:
            result.close()
            raise
        return result

    # -- convolution (Phase D, D3: forward-only Core wrapper) ------------

    def _contiguous_temp(self, temporaries):
        """Materialize this core into a fresh **owning** row-major
        contiguous copy (offset 0) and record it in ``temporaries`` so the
        caller closes it deterministically after the native call — the
        Policy-B copy-then-compute helper (docs/native_cnn_design.md §5)."""
        temp = self.contiguous_copy()
        temporaries.append(temp)
        return temp

    def conv2d_forward(self, weight, bias=None, *, stride=1, padding=0):
        """2-D cross-correlation forward over this NCHW input, natively.

        ``self`` is the ``(N, C, H, W)`` input; ``weight`` is an
        ``(O, C, kh, kw)`` OIHW tensor core; ``bias`` is an optional
        ``(O,)`` tensor core (``None`` = no bias). ``stride`` and
        ``padding`` are each an int or a 2-element ``(height, width)`` pair
        (bools rejected). Returns a fresh **owning** row-major contiguous
        ``(N, O, out_h, out_w)`` NativeTensorCore.

        This is the **forward-only, autograd-unaware** Core wrapper (the
        differentiable ``NativeTensor.conv2d`` primitive is a later
        milestone). It performs the full public validation, computes and
        checks the output shape from the locked floor formula in Python ints
        (so the shape math cannot overflow) *before* allocating anything,
        and — by Policy B (docs/native_cnn_design.md §5) — feeds the raw C
        ABI **contiguous storage only**: any non-contiguous input, weight,
        or bias is materialized into a private contiguous copy that is
        closed the moment the native call returns, while already-contiguous
        operands (even with a non-zero offset) are passed through untouched.
        The caller's tensors are never mutated. A failure at any stage
        allocates no output and leaks no temporary copy.

        No dilation, groups, channels-last, or output padding — those are
        not part of the signature. The weight/bias/input must all be open
        CPU float64 tensor cores."""
        self._require_open()
        if not isinstance(weight, NativeTensorCore):
            raise TypeError(
                f"conv2d_forward requires a NativeTensorCore weight, "
                f"got {type(weight).__name__}"
            )
        weight._require_open()
        self._require_matching_metadata(weight, "conv2d_forward")
        has_bias = bias is not None
        if has_bias:
            if not isinstance(bias, NativeTensorCore):
                raise TypeError(
                    f"conv2d_forward requires a NativeTensorCore bias or None, "
                    f"got {type(bias).__name__}"
                )
            bias._require_open()
            self._require_matching_metadata(bias, "conv2d_forward")

        if self.ndim != 4:
            raise ValueError(
                f"conv2d_forward requires a 4-D NCHW input, got shape {self.shape}"
            )
        if weight.ndim != 4:
            raise ValueError(
                f"conv2d_forward requires a 4-D OIHW weight, got shape {weight.shape}"
            )
        if has_bias and bias.ndim != 1:
            raise ValueError(
                f"conv2d_forward requires a 1-D bias, got shape {bias.shape}"
            )

        n, c, h, w = self.shape
        o, weight_in, kh, kw = weight.shape
        if c != weight_in:
            raise ValueError(
                f"conv2d_forward input channels {c} do not match the weight's "
                f"input channels {weight_in} (input {self.shape}, weight "
                f"{weight.shape})"
            )
        if has_bias and bias.shape[0] != o:
            raise ValueError(
                f"conv2d_forward bias length {bias.shape[0]} does not match the "
                f"number of output channels {o}"
            )

        sh, sw = _spatial_pair(stride, "stride", minimum=1)
        ph, pw = _spatial_pair(padding, "padding", minimum=0)
        # Python-int floor arithmetic; raises before any allocation if the
        # kernel does not fit the padded input.
        out_h, out_w = conv_output_shape((h, w), (kh, kw), (sh, sw), (ph, pw))

        # Policy B: hand the kernel contiguous storage only. A non-contiguous
        # operand is copied into a private owning tensor (offset 0) closed as
        # soon as the call returns; a contiguous operand (possibly with a
        # non-zero offset) is passed straight through with its offset.
        temporaries = []
        try:
            input_core = (
                self if self.contiguous else self._contiguous_temp(temporaries)
            )
            weight_core = (
                weight if weight.contiguous
                else weight._contiguous_temp(temporaries)
            )
            bias_handle = None
            bias_offset = 0
            if has_bias:
                bias_core = (
                    bias if bias.contiguous
                    else bias._contiguous_temp(temporaries)
                )
                bias_handle = bias_core._storage._require_open()
                bias_offset = bias_core.offset

            # H1 uninitialized — conv2d forward: assigns output[n, o, i, j] over the full output extent.
            out = NativeTensorCore._uninitialized(
                (n, o, out_h, out_w), dtype=self.dtype, device=self.device
            )
            try:
                self._storage._lib.tf_core_conv2d_forward(
                    input_core._storage._require_open(), input_core.offset,
                    weight_core._storage._require_open(), weight_core.offset,
                    bias_handle, bias_offset,
                    out._storage._require_open(),
                    n, c, h, w, o, kh, kw, sh, sw, ph, pw, out_h, out_w,
                )
            except BaseException:
                # The native call failed (e.g. an injected allocation
                # failure): discard the freshly allocated output so a failed
                # forward returns no half-built tensor.
                out.close()
                raise
            return out
        finally:
            # Close every private contiguous copy exactly once, whether the
            # call succeeded or raised — the caller's operands are untouched.
            for temp in temporaries:
                temp.close()

    def conv2d_input_backward(self, weight, *, input_shape, stride=1, padding=0):
        """Gradient of Conv2d w.r.t. its input, natively (Phase D, D6).

        ``self`` is the upstream gradient ``grad_output`` with shape
        ``(N, O, out_h, out_w)``; ``weight`` is the ``(O, C, kh, kw)`` OIHW
        tensor core; ``input_shape`` is the parent input's ``(N, C, H, W)``.
        Returns a fresh **owning** row-major contiguous ``(N, C, H, W)``
        NativeTensorCore — the input gradient.

        Forward-only and autograd-unaware (the ``NativeTensor.conv2d``
        node calls this from its input-gradient callback). Validates ranks,
        channel/spatial relationships, and the recomputed grad_output shape
        before allocating; feeds the raw C ABI **contiguous storage only**
        via Policy B (any non-contiguous grad_output/weight is copied into a
        private core closed as soon as the call returns); the caller's
        tensors are never mutated. A failure allocates no output and leaks no
        temporary."""
        self._require_open()
        if not isinstance(weight, NativeTensorCore):
            raise TypeError(
                f"conv2d_input_backward requires a NativeTensorCore weight, "
                f"got {type(weight).__name__}"
            )
        weight._require_open()
        self._require_matching_metadata(weight, "conv2d_input_backward")
        if self.ndim != 4:
            raise ValueError(
                f"conv2d_input_backward requires a 4-D NCHW grad_output, got "
                f"shape {self.shape}"
            )
        if weight.ndim != 4:
            raise ValueError(
                f"conv2d_input_backward requires a 4-D OIHW weight, got shape "
                f"{weight.shape}"
            )
        input_shape = _as_shape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"conv2d_input_backward input_shape must be 4-D NCHW, got "
                f"{input_shape}"
            )
        n, c, h, w = input_shape
        o, weight_in, kh, kw = weight.shape
        if c != weight_in:
            raise ValueError(
                f"conv2d_input_backward input channels {c} do not match the "
                f"weight's input channels {weight_in} (input_shape "
                f"{input_shape}, weight {weight.shape})"
            )
        sh, sw = _spatial_pair(stride, "stride", minimum=1)
        ph, pw = _spatial_pair(padding, "padding", minimum=0)
        out_h, out_w = conv_output_shape((h, w), (kh, kw), (sh, sw), (ph, pw))
        if self.shape != (n, o, out_h, out_w):
            raise ValueError(
                f"conv2d_input_backward grad_output shape {self.shape} does "
                f"not match the expected {(n, o, out_h, out_w)} for input "
                f"{input_shape}, weight {weight.shape}, stride {(sh, sw)}, "
                f"padding {(ph, pw)}"
            )
        temporaries = []
        try:
            go = self if self.contiguous else self._contiguous_temp(temporaries)
            wt = (
                weight if weight.contiguous
                else weight._contiguous_temp(temporaries)
            )
            # H1 uninitialized — conv2d input gradient: the kernel zeroes the whole span itself, then accumulates.
            out = NativeTensorCore._uninitialized(
                (n, c, h, w), dtype=self.dtype, device=self.device
            )
            try:
                self._storage._lib.tf_core_conv2d_input_backward(
                    go._storage._require_open(), go.offset,
                    wt._storage._require_open(), wt.offset,
                    out._storage._require_open(),
                    n, c, h, w, o, kh, kw, sh, sw, ph, pw, out_h, out_w,
                )
            except BaseException:
                out.close()
                raise
            return out
        finally:
            for temp in temporaries:
                temp.close()

    def conv2d_weight_backward(self, input, *, weight_shape, stride=1, padding=0):
        """Gradient of Conv2d w.r.t. its weight, natively (Phase D, D6).

        ``self`` is the upstream gradient ``grad_output`` with shape
        ``(N, O, out_h, out_w)``; ``input`` is the parent's ``(N, C, H, W)``
        NCHW input; ``weight_shape`` is the weight's ``(O, C, kh, kw)`` OIHW
        shape. Returns a fresh **owning** row-major contiguous
        ``(O, C, kh, kw)`` NativeTensorCore — the weight gradient.

        Forward-only and autograd-unaware. Same validation/Policy-B/failure
        contract as ``conv2d_input_backward`` (any non-contiguous
        grad_output/input is copied into a private core closed after the
        call); the caller's tensors are never mutated."""
        self._require_open()
        if not isinstance(input, NativeTensorCore):
            raise TypeError(
                f"conv2d_weight_backward requires a NativeTensorCore input, "
                f"got {type(input).__name__}"
            )
        input._require_open()
        self._require_matching_metadata(input, "conv2d_weight_backward")
        if self.ndim != 4:
            raise ValueError(
                f"conv2d_weight_backward requires a 4-D NCHW grad_output, got "
                f"shape {self.shape}"
            )
        if input.ndim != 4:
            raise ValueError(
                f"conv2d_weight_backward requires a 4-D NCHW input, got shape "
                f"{input.shape}"
            )
        weight_shape = _as_shape(weight_shape)
        if len(weight_shape) != 4:
            raise ValueError(
                f"conv2d_weight_backward weight_shape must be 4-D OIHW, got "
                f"{weight_shape}"
            )
        o, c, kh, kw = weight_shape
        n, input_c, h, w = input.shape
        if input_c != c:
            raise ValueError(
                f"conv2d_weight_backward input channels {input_c} do not match "
                f"the weight's input channels {c} (input {input.shape}, "
                f"weight_shape {weight_shape})"
            )
        sh, sw = _spatial_pair(stride, "stride", minimum=1)
        ph, pw = _spatial_pair(padding, "padding", minimum=0)
        out_h, out_w = conv_output_shape((h, w), (kh, kw), (sh, sw), (ph, pw))
        if self.shape != (n, o, out_h, out_w):
            raise ValueError(
                f"conv2d_weight_backward grad_output shape {self.shape} does "
                f"not match the expected {(n, o, out_h, out_w)} for input "
                f"{input.shape}, weight_shape {weight_shape}, stride "
                f"{(sh, sw)}, padding {(ph, pw)}"
            )
        temporaries = []
        try:
            go = self if self.contiguous else self._contiguous_temp(temporaries)
            inp = (
                input if input.contiguous
                else input._contiguous_temp(temporaries)
            )
            # H1 uninitialized — conv2d weight gradient: the kernel zeroes the whole span itself, then accumulates.
            out = NativeTensorCore._uninitialized(
                (o, c, kh, kw), dtype=self.dtype, device=self.device
            )
            try:
                self._storage._lib.tf_core_conv2d_weight_backward(
                    go._storage._require_open(), go.offset,
                    inp._storage._require_open(), inp.offset,
                    out._storage._require_open(),
                    n, c, h, w, o, kh, kw, sh, sw, ph, pw, out_h, out_w,
                )
            except BaseException:
                out.close()
                raise
            return out
        finally:
            for temp in temporaries:
                temp.close()

    # -- pooling (Phase D, D8: forward-only Core wrapper + winners) -------

    def maxpool2d_forward(self, *, kernel_size, stride=None, padding=0):
        """2-D max pooling forward over this NCHW input, natively.

        ``self`` is the ``(N, C, H, W)`` input. ``kernel_size`` and
        ``stride`` are an int or a 2-element ``(height, width)`` pair of
        ints ≥ 1 (bools rejected); ``stride=None`` means
        ``stride = kernel_size`` (non-overlapping windows, the stable
        convention). ``padding`` is an int or pair ≥ 0, applied
        symmetrically on each spatial axis. Returns a fresh **owning**
        row-major contiguous ``(N, C, out_h, out_w)`` NativeTensorCore.

        Windows see a conceptual ``-inf`` outside the real input, so a
        padded cell loses to any finite value but still *participates* in
        the selection; ties keep the first occurrence in row-major window
        order (docs/native_cnn_design.md §10). This is the **forward-only,
        autograd-unaware** Core wrapper — the differentiable
        ``NativeTensor.maxpool2d`` primitive (D9) and the
        ``NativeMaxPool2d`` module (D10) are separate layers built on it,
        and both are implemented.

        The kernel also produces the private winner buffer backward will
        need; this public method releases it, so the pooled values are all
        that survive. The internal
        ``_maxpool2d_forward_with_winners`` helper is what keeps it (D9).

        No dilation, ceil_mode, return_indices, adaptive/average/global
        pooling, or channels-last — none of those is in the signature."""
        out, winners = self._maxpool2d_forward_with_winners(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        # The public Core forward exposes only the pooled values; the
        # winner buffer is internal state, released deterministically here
        # rather than left to garbage collection.
        winners.close()
        return out

    def _maxpool2d_forward_with_winners(
        self, *, kernel_size, stride=None, padding=0
    ):
        """The pooling forward plus its private saved-winner buffer.

        Returns ``(output, winners)``: two fresh **owning** row-major
        contiguous ``(N, C, out_h, out_w)`` cores. ``winners`` holds, for
        each output cell, the flat offset ``ih * W + iw`` of the selected
        input element inside its ``(n, c)`` plane, or the sentinel ``-1.0``
        when a padding cell won (docs/native_cnn_design.md §12). Every
        stored value is an exact integral float64 — the wrapper proves
        ``H * W <= 2**53`` in Python arbitrary-precision arithmetic
        *before* allocating or calling anything, so no index can round.

        The winner buffer is **internal**: it is never exposed as a public
        NativeTensor, never given a dtype tag of its own, never traversed
        as a parameter or buffer, and never serialized. It exists so the
        D9 backward can scatter without recomputing winners; the caller of
        this private helper owns it and must ``close()`` it.

        Validation runs entirely before any allocation. Per **Policy B**
        (docs/native_cnn_design.md §5) a non-contiguous input is
        materialized into a private owning contiguous copy that is closed
        as soon as the native call returns, while an already-contiguous
        input (even at a non-zero offset) is passed straight through. On
        any failure every object this method allocated is closed, the
        caller's input is untouched, and no partial result is returned."""
        self._require_open()
        if self.ndim != 4:
            raise ValueError(
                f"maxpool2d_forward requires a 4-D NCHW input, got shape "
                f"{self.shape}"
            )
        if self.dtype != "float64" or self.device != "cpu":
            raise ValueError(
                f"maxpool2d_forward requires a float64/cpu input, got "
                f"{self.dtype}/{self.device}"
            )

        kh, kw = _spatial_pair(kernel_size, "kernel_size", minimum=1)
        # Stable convention: no stride means non-overlapping windows.
        if stride is None:
            sh, sw = kh, kw
        else:
            sh, sw = _spatial_pair(stride, "stride", minimum=1)
        ph, pw = _spatial_pair(padding, "padding", minimum=0)

        n, c, h, w = self.shape
        # Winner indices are float64 flat plane offsets, exact only while
        # H*W <= 2**53 (design §12). Python ints, so the product itself
        # cannot overflow while being checked.
        if h * w > _MAX_EXACT_WINNER_PLANE:
            raise ValueError(
                f"maxpool2d_forward input plane {(h, w)} has {h * w} elements, "
                f"more than the {_MAX_EXACT_WINNER_PLANE} float64 can index "
                f"exactly; winner offsets would round"
            )
        # Python-int floor arithmetic; raises before any allocation if the
        # window does not fit the padded input.
        out_h, out_w = conv_output_shape((h, w), (kh, kw), (sh, sw), (ph, pw))
        # The element counts crossing the ABI must be representable as
        # int64 storage sizes (again in Python ints, before allocating).
        for count, what in (
            (n * c * h * w, "input"),
            (n * c * out_h * out_w, "output"),
        ):
            if count > _INT64_MAX:
                raise ValueError(
                    f"maxpool2d_forward {what} element count {count} exceeds "
                    f"the int64 range the native runtime addresses"
                )

        temporaries = []
        out = None
        winners = None
        try:
            input_core = (
                self if self.contiguous else self._contiguous_temp(temporaries)
            )
            # Deterministic allocation order: output first, then winners.
            # If the second allocation fails the first is closed below, so
            # a failed forward leaves nothing half-built.
            # H1 uninitialized — maxpool2d values: assigns output[n, c, i, j] over the full output extent.
            out = NativeTensorCore._uninitialized(
                (n, c, out_h, out_w), dtype=self.dtype, device=self.device
            )
            # H1 uninitialized — maxpool2d winners: assigns winners[n, c, i, j] over the same full extent.
            winners = NativeTensorCore._uninitialized(
                (n, c, out_h, out_w), dtype=self.dtype, device=self.device
            )
            self._storage._lib.tf_core_maxpool2d_forward(
                input_core._storage._require_open(), input_core.offset,
                out._storage._require_open(),
                winners._storage._require_open(),
                n, c, h, w, kh, kw, sh, sw, ph, pw, out_h, out_w,
            )
            return out, winners
        except BaseException:
            # Close whichever result objects were successfully allocated —
            # never rely on garbage collection for native memory.
            for allocated in (winners, out):
                if allocated is not None:
                    allocated.close()
            raise
        finally:
            # Close the private contiguous copy (if any) exactly once,
            # whether the call succeeded or raised.
            for temp in temporaries:
                temp.close()

    def maxpool2d_backward(self, winners, *, input_shape):
        """Gradient of MaxPool2d w.r.t. its input, natively (Phase D, D9).

        ``self`` is the upstream gradient ``grad_output`` with shape
        ``(N, C, out_h, out_w)``; ``winners`` is the **private saved-winner
        core** the D8 forward produced, with exactly that shape;
        ``input_shape`` is the parent input's ``(N, C, H, W)``. Returns a
        fresh **owning** row-major contiguous ``(N, C, H, W)``
        NativeTensorCore — the input gradient.

        The routing comes entirely from the saved winners: each output
        cell's gradient is added to the input element that won its window
        (``ih = winner // W``, ``iw = winner % W``), and a ``-1`` winner
        (padding won) drops that gradient. Overlapping windows accumulate.
        **No input value is reread and no window maximum is recomputed**,
        so no kernel/stride/padding argument is needed or accepted here.

        Autograd-unaware (the ``NativeTensor.maxpool2d`` node calls this
        from its single input-gradient callback). Validation runs before
        any allocation; per Policy B a non-contiguous grad_output or winner
        core is materialized into a private copy closed as soon as the
        native call returns. The checked C ABI additionally validates every
        winner value (``-1`` or an exact in-range integer) before
        scattering, so a corrupted buffer raises instead of writing
        anywhere. Neither operand is mutated, and a failure closes the
        output and leaks no temporary."""
        self._require_open()
        if not isinstance(winners, NativeTensorCore):
            raise TypeError(
                f"maxpool2d_backward requires a NativeTensorCore winner "
                f"buffer, got {type(winners).__name__}"
            )
        winners._require_open()
        self._require_matching_metadata(winners, "maxpool2d_backward")
        if self.dtype != "float64" or self.device != "cpu":
            raise ValueError(
                f"maxpool2d_backward requires float64/cpu operands, got "
                f"{self.dtype}/{self.device}"
            )
        if self.ndim != 4:
            raise ValueError(
                f"maxpool2d_backward requires a 4-D NCHW grad_output, got "
                f"shape {self.shape}"
            )
        if winners.shape != self.shape:
            raise ValueError(
                f"maxpool2d_backward winner shape {winners.shape} does not "
                f"match the grad_output shape {self.shape}"
            )
        input_shape = _as_shape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"maxpool2d_backward input_shape must be 4-D NCHW, got "
                f"{input_shape}"
            )
        n, c, h, w = input_shape
        out_n, out_c, out_h, out_w = self.shape
        if (n, c) != (out_n, out_c):
            raise ValueError(
                f"maxpool2d_backward batch/channels {(n, c)} do not match the "
                f"grad_output's {(out_n, out_c)} (input_shape {input_shape}, "
                f"grad_output {self.shape})"
            )
        # The winner domain is [0, H*W - 1] plus the -1 sentinel, so the
        # same float64 exactness bound the forward proved must hold here.
        if h * w > _MAX_EXACT_WINNER_PLANE:
            raise ValueError(
                f"maxpool2d_backward input plane {(h, w)} has {h * w} "
                f"elements, more than the {_MAX_EXACT_WINNER_PLANE} float64 "
                f"can index exactly"
            )
        if n * c * h * w > _INT64_MAX:
            raise ValueError(
                f"maxpool2d_backward grad_input element count {n * c * h * w} "
                f"exceeds the int64 range the native runtime addresses"
            )

        temporaries = []
        out = None
        try:
            grad_output = (
                self if self.contiguous else self._contiguous_temp(temporaries)
            )
            winner_core = (
                winners if winners.contiguous
                else winners._contiguous_temp(temporaries)
            )
            # H1 uninitialized — maxpool2d backward: the kernel zeroes the whole span itself, then accumulates.
            out = NativeTensorCore._uninitialized(
                (n, c, h, w), dtype=self.dtype, device=self.device
            )
            self._storage._lib.tf_core_maxpool2d_backward(
                grad_output._storage._require_open(), grad_output.offset,
                winner_core._storage._require_open(), winner_core.offset,
                out._storage._require_open(),
                n, c, h, w, out_h, out_w,
            )
            return out
        except BaseException:
            # A failed backward returns no half-built gradient.
            if out is not None:
                out.close()
            raise
        finally:
            for temp in temporaries:
                temp.close()

    # -- stateless Dropout (Phase G, G2: forward-only Core wrapper +
    #    private multiplier mask) ---------------------------------------

    def dropout_forward(self, p, *, seed, call_index):
        """Inverted Dropout forward over this tensor, natively (G2).

        ``self`` is the input of any rank (a 0-d scalar included) and any
        layout; ``p`` is the drop probability in ``[0, 1)``; ``seed`` and
        ``call_index`` are the two halves of the **complete random key**,
        each an exact Python ``int`` in ``[0, 2**64 - 1]``. Returns a fresh
        **owning** row-major contiguous NativeTensorCore of the input's
        shape, where

            output[i] = input[i] * (0.0 if dropped else 1 / (1 - p))

        and the keep/drop decision for logical element ``i`` is a
        deterministic function of ``(seed, call_index, i, p)`` and nothing
        else — never of the input values, the storage address, the
        physical strides, the traversal order, or any earlier call.

        **Stateless.** This method takes no ``NativeGenerator`` and no
        generator is reachable from it: it never reserves, commits,
        cancels, inspects, or mutates one, and no C++ translation unit
        holds random state of any kind (design §7.6). The reservation
        transaction that turns a seed and a counter into a ``call_index``
        is milestone G3's, one layer above.

        This is the **forward-only, autograd-unaware** Core wrapper. The
        kernel also produces the private multiplier mask a backward would
        need; this public method releases it, so the dropped values are all
        that survive. The internal ``_dropout_forward_with_mask`` helper is
        what keeps it — exactly the ``maxpool2d_forward`` / winner-buffer
        split (design §7.1).

        There is no ``dropout_backward`` Core method and no backward
        kernel: the gradient of inverted Dropout is ``upstream * mask``,
        which ``multiply`` already computes (design §7.5). The
        differentiable ``NativeTensor.dropout`` operation (G3) and the
        ``NativeDropout`` module (G4) do not exist yet, and ``"dropout"``
        as a capability name is still in ``UNSUPPORTED``."""
        out, mask = self._dropout_forward_with_mask(
            p, seed=seed, call_index=call_index
        )
        # The public Core forward exposes only the dropped values; the
        # multiplier mask is internal state, released deterministically
        # here rather than left to garbage collection.
        mask.close()
        return out

    def _dropout_forward_with_mask(self, p, *, seed, call_index):
        """The Dropout forward plus its private multiplier mask.

        Returns ``(output, mask)``: two fresh **owning** row-major
        contiguous cores of the input's shape. ``mask`` holds exactly two
        distinct float64 values — ``0.0`` for a dropped element and the
        single ``1 / (1 - p)`` computed once for the whole call — so it is
        a *multiplier* mask, not a boolean one, and a backward is one
        elementwise multiply against it (design §4.4, §7.5).

        The mask is **internal**: it is never exposed as a public
        NativeTensor, never given a dtype tag of its own, never traversed
        as a parameter or buffer, and never serialized. It exists so a
        later backward can multiply without redrawing; the caller of this
        private helper owns it and must ``close()`` it.

        **Logical-layout independence** is a locked property, not an
        accident. Per **Policy B** (docs/native_cnn_design.md §5) a
        non-contiguous input is materialized into a private owning
        contiguous copy that is closed as soon as the native call returns,
        so the kernel's flat traversal index *is* the logical row-major
        index. A transposed view, a narrowed view, a nonzero-offset view,
        and a plain contiguous tensor of the same logical shape therefore
        all receive the **same** mask for the same ``(seed, call_index,
        p)``; only the values they multiply differ.

        Validation runs entirely before any allocation. Allocation order
        is deterministic — **output first, then mask** — and on any
        failure every object this method allocated is closed, any
        Policy-B temporary is closed, the caller's input is left open and
        unchanged, and no partial result is returned. Neither result
        aliases the input or the other.

        **Graph-unaware**, like every Core op: no autograd node is built,
        nothing claims graph ownership, no version is recorded, and no
        generator is touched.

        Zero-element inputs: the kernel and the C ABI both accept a count
        of ``0`` (no draw, no write), but the native tensor representation
        rejects zero-size dimensions outright (``shape`` dimensions must
        be positive ints), so no empty core can be constructed to hand in
        today. The empty case is proved at the kernel and ABI layers,
        where it is reachable."""
        self._require_open()
        if self.dtype != "float64" or self.device != "cpu":
            raise ValueError(
                f"dropout_forward requires a float64/cpu input, got "
                f"{self.dtype}/{self.device}"
            )
        # Both key halves and the probability are validated by the shared
        # helpers, so this surface accepts exactly what the operation and
        # the module will accept (design §6.1).
        probability = _normalize_dropout_probability(p, "dropout_forward")
        seed = _validate_random_key_field(seed, "dropout_forward seed")
        call_index = _validate_random_key_field(
            call_index, "dropout_forward call_index"
        )
        count = self.numel
        # Python ints, so this check cannot itself overflow; the C ABI
        # re-proves every span in int64 anyway.
        if count > _INT64_MAX:
            raise ValueError(
                f"dropout_forward element count {count} exceeds the int64 "
                f"range the native runtime addresses"
            )

        temporaries = []
        out = None
        mask = None
        try:
            input_core = (
                self if self.contiguous else self._contiguous_temp(temporaries)
            )
            # Deterministic allocation order: output first, then the mask.
            # If the second allocation fails the first is closed below, so
            # a failed forward leaves nothing half-built.
            # H1 uninitialized — dropout output: assigns output[i] for every i in [0, count).
            out = NativeTensorCore._uninitialized(
                self.shape, dtype=self.dtype, device=self.device
            )
            # H1 uninitialized — dropout mask: assigns mask[i] for the same full range.
            mask = NativeTensorCore._uninitialized(
                self.shape, dtype=self.dtype, device=self.device
            )
            self._storage._lib.tf_core_dropout_forward(
                input_core._storage._require_open(), input_core.offset,
                out._storage._require_open(),
                mask._storage._require_open(),
                count, seed, call_index, probability,
            )
            return out, mask
        except BaseException:
            # Close whichever result objects were successfully allocated —
            # never rely on garbage collection for native memory, and
            # never let a half-built pair escape.
            for allocated in (mask, out):
                if allocated is not None:
                    allocated.close()
            raise
        finally:
            # Close the private contiguous copy (if any) exactly once,
            # whether the call succeeded or raised.
            for temp in temporaries:
                temp.close()

    # -- view operations (metadata only: no data is copied) --------------

    def _view_core(self, dims, strides, offset):
        """A new core borrowing this core's storage with a new layout.

        Every caller derives ``dims``/``strides``/``offset`` from *this*
        core's already-validated layout — a permutation of it
        (``transpose``), one axis shortened with the offset advanced
        (``narrow``), or a freshly ``_as_shape``-validated shape whose
        element count has been checked to match (``reshape``) — so the
        metadata arrives normalized and takes the private view
        constructor. The storage-bounds check still runs in full."""
        view = NativeTensorView._from_validated(
            self._storage, dims, strides, offset
        )
        return NativeTensorCore(self._storage, view, owns_storage=False)

    def reshape(self, new_shape):
        """A view of the same storage with ``new_shape`` (row-major).

        Metadata only — no copy. Requires a contiguous tensor (a
        non-contiguous layout cannot be reinterpreted by strides
        alone; materialize with contiguous_copy() first) and the same
        total number of elements.
        """
        self._require_open()
        if not self.contiguous:
            raise ValueError(
                "reshape requires a contiguous tensor; call "
                "contiguous_copy() first"
            )
        dims = _as_shape(new_shape)  # validates the shape by the v0.7 rules
        count = _numel_checked(dims)
        if count != self.numel:
            raise ValueError(
                f"cannot reshape {self.shape} ({self.numel} elements) "
                f"into {tuple(new_shape)} ({count} elements)"
            )
        return self._view_core(
            dims, _row_major_strides_checked(dims), self.offset
        )

    def transpose(self, *axes):
        """A view with permuted axes. Metadata only — no copy.

        With no arguments, all axes are reversed (NumPy behavior; a
        no-op for scalars and 1-D tensors). Explicit axes must be a
        complete permutation of range(ndim).
        """
        self._require_open()
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        axes = _as_int_tuple(axes, "axes")
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError(
                f"axes must be a permutation of range({self.ndim}), got {axes}"
            )
        new_shape = tuple(self.shape[axis] for axis in axes)
        new_strides = tuple(self.strides[axis] for axis in axes)
        return self._view_core(new_shape, new_strides, self.offset)

    @property
    def T(self):
        """transpose() with all axes reversed — NumPy's .T semantics,
        so (1, 0) for 2-D and a no-op for scalars and 1-D tensors."""
        return self.transpose()

    def narrow(self, dim, start, length):
        """A view keeping ``length`` positions of dimension ``dim``,
        beginning at ``start``. Metadata only — no copy: the shape
        shrinks in one dimension and the offset advances by
        ``start * strides[dim]``; strides are unchanged.

        ``length`` must be at least 1 (zero-size shapes are not
        supported). No step parameter in v1.1.
        """
        self._require_open()
        for name, value in (("dim", dim), ("start", start), ("length", length)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {value!r}")
        # Normalize to exact ints immediately after the type check. A NumPy
        # integer argument is accepted (it always was), and the derived
        # shape and offset must be plain ints exactly as the shape
        # normalization used to make them — the private view constructor no
        # longer re-normalizes, so this is where it happens. The bounds
        # messages below are unaffected: these values format identically.
        dim, start, length = int(dim), int(start), int(length)
        if not 0 <= dim < self.ndim:
            raise ValueError(f"dim must be in [0, {self.ndim}), got {dim}")
        if start < 0 or length < 1 or start + length > self.shape[dim]:
            raise ValueError(
                f"narrow(dim={dim}, start={start}, length={length}) is out "
                f"of bounds for dimension size {self.shape[dim]}"
            )
        new_shape = tuple(
            length if axis == dim else size for axis, size in enumerate(self.shape)
        )
        new_offset = self.offset + start * self.strides[dim]
        return self._view_core(new_shape, self.strides, new_offset)

    def narrow_backward(self, dim, start, original_shape):
        """Scatter this upstream gradient into a fresh zero tensor of
        ``original_shape``, placing each element at its narrowed logical
        position — the adjoint of ``narrow(dim, start, length)``, where
        ``length`` is this gradient's own extent along ``dim``.

        A forward-shaped numerical method, not graph machinery — the core
        stays autograd-unaware; the NativeTensor layer calls this from its
        narrow backward closure (``self`` is the upstream gradient there,
        the data being scattered). This gradient may be a strided view (it
        is read through its own strides/offset); the result is a new
        **owning** row-major contiguous NativeTensorCore of
        ``original_shape``, zero everywhere outside the narrowed region and
        carrying this gradient's dtype/device. No NumPy touches the data.

        Validates the scatter arguments the way ``narrow`` validates its
        forward ones: ``dim`` a non-bool int in ``[0, ndim)``, ``start``
        non-negative, the gradient's rank equal to the original rank, its
        non-``dim`` extents equal to the original's, and
        ``start + length <= original_shape[dim]``."""
        self._require_open()
        original = _as_shape(original_shape)  # validates positive-int dims
        ndim = len(original)
        for name, value in (("dim", dim), ("start", start)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {value!r}")
        dim = int(dim)
        start = int(start)
        if not 0 <= dim < ndim:
            raise ValueError(f"dim must be in [0, {ndim}), got {dim}")
        if self.ndim != ndim:
            raise ValueError(
                f"narrow_backward gradient rank {self.ndim} does not match "
                f"the original rank {ndim} (shape {original})"
            )
        length = self.shape[dim]
        if start < 0 or start + length > original[dim]:
            raise ValueError(
                f"narrow_backward(dim={dim}, start={start}, length={length}) "
                f"is out of bounds for dimension size {original[dim]}"
            )
        for axis in range(ndim):
            if axis != dim and self.shape[axis] != original[axis]:
                raise ValueError(
                    f"narrow_backward gradient shape {self.shape} is not "
                    f"compatible with original shape {original} along axis "
                    f"{axis}"
                )
        # H1 REJECTED — this output must stay zero-initialized. It is the
        # clearest *partial-write* case in the runtime: tf_core_narrow_backward
        # assigns only the narrowed region, and every un-narrowed cell is
        # supposed to keep the zero the allocation gave it. That zero is
        # the gradient's value, not an initialization detail, so an
        # uninitialized buffer would leak heap contents straight into a
        # gradient. Its poison test pins exactly this.
        #
        # I4: the same zeroed constructor with the dtype trusted, for
        # ``sum``'s reason and with ``sum``'s argument — the dtype is this
        # gradient's own tag, and the scatter's output must carry it.
        out = NativeTensorCore.zeros(original, dtype=self.dtype,
                                     device=self.device, _trusted_dtype=True)
        # The gradient lives at the logical shape, so the output is always a
        # fresh row-major contiguous buffer (offset 0) regardless of the
        # narrowed parent's own layout. Each narrowed axis maps 1:1 to the
        # same output axis, so the write-strides are just the parent's full
        # row-major strides; the base offset skips the leading `start` slabs.
        out_full = _row_major_strides_checked(original)
        out_offset = start * out_full[dim]
        # This gradient's own layout comes from its view's cache; the
        # output write-strides belong to this scatter, not to a tensor, so
        # they stay an operation-local vector. ``out_full`` has one entry per
        # axis of ``original``, whose rank was proved equal to this
        # gradient's ``self.ndim`` above, so all three carriers describe the
        # rank passed below.
        shape_ptr, strides_ptr = self._layout_pointers()
        self._storage._lib.tf_core_narrow_backward(
            self._storage._require_open(),
            out._storage._require_open(),
            shape_ptr,
            strides_ptr,
            _layout_vector(out_full),
            self.offset, out_offset, self.ndim,
        )
        return out

    # -- lifetime -------------------------------------------------------

    def close(self):
        """Close this core. Owners release the native storage; views
        only close themselves and leave shared storage untouched.
        Safe to call repeatedly."""
        self._closed = True
        if self._owns_storage:
            self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        # Defensive cleanup only — never rely on GC timing; use close().
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        state = ", closed" if self._closed else ""
        return (
            f"NativeTensorCore(shape={self.shape}, strides={self.strides}, "
            f"contiguous={self.contiguous}{state})"
        )


# ---------------------------------------------------------------------------
# Phase-E classification Core plumbing (E5)
#
# The reduction mapping, the strict target contract (design §6), and the
# small result type NativeTensorCore.cross_entropy_forward returns. All
# private: none of this is public API, and none of it knows about the
# autograd graph — E6 consumes it, it does not consume E6.
# ---------------------------------------------------------------------------

# The reduction codes the C ABI carries (design §9.2). A small integer,
# never a string: the ABI stays free of allocation and encoding concerns,
# and both sides validate the code. These literals are mirrored by
# tf::kCrossEntropyReduction{Mean,Sum} in
# cpp/include/tf_classification_internal.h.
_REDUCTION_CODES = {"mean": 0, "sum": 1}


def _normalize_reduction(reduction, op_name):
    """Validate a cross-entropy reduction and map it to its ABI code.

    Exactly ``"mean"`` and ``"sum"``, by exact string match — no case or
    whitespace normalization, no coercion, and no ``"none"`` (both
    supported reductions produce a scalar, which is what a backward seed
    needs). The ``NativeMSELoss`` precedent, including its error types: a
    non-string (``None``, a bool, an int, a float, a list) raises
    TypeError; an unrecognized string raises ValueError. Returns the
    ``(name, code)`` pair. Pure Python — never touches the library, so it
    can run before anything is allocated."""
    if not isinstance(reduction, str):
        raise TypeError(
            f"{op_name} reduction must be a str, got "
            f"{type(reduction).__name__}"
        )
    if reduction not in _REDUCTION_CODES:
        # Exact match only: "Mean", "SUM", " mean ", "none", "" all land
        # here — nothing is normalized or coerced.
        raise ValueError(
            f"{op_name} reduction must be one of {list(_REDUCTION_CODES)}, "
            f"got {reduction!r}"
        )
    return reduction, _REDUCTION_CODES[reduction]


def _prepare_class_targets(targets, batch_size, num_classes, op_name):
    """Validate class labels and copy them into owned int64 metadata.

    The strict half of the Phase-E target contract (design §6). Targets
    are **not** native tensors — the runtime has no integer dtype — so
    they arrive as ordinary Python or NumPy integer data and leave as an
    independently owned, C-contiguous, read-only ``np.int64`` array of
    length ``batch_size``.

    Validation happens **before** the copy and before any native
    allocation, and it is deliberately stricter than
    ``np.asarray(targets, dtype=np.int64)``, which would silently
    truncate ``1.9`` and reinterpret ``True``:

    1. a ``str``/``bytes``/``bytearray`` is not a label sequence, even
       though iterating one yields ints (TypeError);
    2. a NumPy array must have an integer dtype — ``bool_``, floating,
       complex, object, and string dtypes are all rejected (TypeError) —
       and must be one-dimensional (ValueError naming the shape);
    3. any other object goes through the existing ``_as_int_tuple``
       contract: it must be a sequence whose every element is an actual
       integer scalar and not a ``bool`` (TypeError). Nested and ragged
       sequences fail here because their elements are not integers;
    4. the length must equal ``batch_size`` exactly (ValueError);
    5. every value must be representable as int64 (ValueError) and lie in
       ``[0, num_classes)`` (ValueError naming the index and the value).

    Only then is the copy taken. It is always a copy, even when the
    caller passed an already contiguous ``int64`` array, so no view into
    caller memory is ever retained and post-forward caller mutation
    cannot reach the kernel. The result is marked read-only so the
    forward's own saved copy cannot be edited in place either.

    Pure Python + one NumPy array construction: the values are compared
    as Python ints, never through NumPy arithmetic, so this stays clear
    of the numerical-NumPy tripwire that guards the native paths."""
    if isinstance(targets, (str, bytes, bytearray)):
        raise TypeError(
            f"{op_name} targets must be a one-dimensional sequence of "
            f"integer class labels, got {type(targets).__name__}"
        )
    if isinstance(targets, np.ndarray):
        if targets.dtype == np.bool_ or not np.issubdtype(
            targets.dtype, np.integer
        ):
            raise TypeError(
                f"{op_name} targets must be integer class labels, got a "
                f"NumPy array of dtype {targets.dtype} (bool and "
                f"floating-point targets are rejected outright — nothing is "
                f"truncated or reinterpreted)"
            )
        if targets.ndim != 1:
            raise ValueError(
                f"{op_name} targets must be one-dimensional, got shape "
                f"{targets.shape}"
            )
        # .tolist() yields exact Python ints for every integer dtype,
        # including uint64 values above the int64 maximum (caught below).
        values = targets.tolist()
    elif isinstance(targets, (int, np.integer, float, np.floating, complex,
                              bool, np.bool_)):
        raise TypeError(
            f"{op_name} targets must be a one-dimensional sequence of "
            f"{batch_size} integer class labels, got the scalar {targets!r}"
        )
    else:
        # Lists, tuples, and other simple sequences. _as_int_tuple is the
        # runtime's existing "sequence of real ints" contract: it rejects
        # non-sequences, bools, floats (1.0 included), complex values,
        # strings, and nested elements, all as TypeError.
        values = _as_int_tuple(targets, f"{op_name} targets")

    if len(values) != batch_size:
        raise ValueError(
            f"{op_name} needs exactly {batch_size} targets (one per logits "
            f"row), got {len(values)}"
        )
    for index, value in enumerate(values):
        value = int(value)
        if value < _INT64_MIN or value > _INT64_MAX:
            raise ValueError(
                f"{op_name} target at index {index} is {value}, outside the "
                f"int64 range the native runtime addresses"
            )
        if value < 0 or value >= num_classes:
            raise ValueError(
                f"{op_name} target at index {index} is {value}, outside the "
                f"valid class range [0, {num_classes})"
            )
    copy = np.array(values, dtype=np.int64)
    copy.flags.writeable = False
    return copy


def _require_target_copy(targets, batch_size, num_classes, op_name):
    """Re-validate an already prepared int64 target copy (E5 backward).

    The backward consumes the copy its own forward produced, so this is
    the *internal* trusted-copy contract rather than the permissive
    public one: the object must already be a 1-D, C-contiguous
    ``np.int64`` array. Its length and every class label are still
    re-checked before the ABI is entered (and C++ re-checks every index
    it dereferences), because a corrupted or hand-built copy must raise
    instead of reaching a kernel. Returns the array unchanged — no second
    copy is taken."""
    if not isinstance(targets, np.ndarray):
        raise TypeError(
            f"{op_name} requires the int64 target copy the forward produced, "
            f"got {type(targets).__name__}"
        )
    if targets.dtype != np.int64:
        raise TypeError(
            f"{op_name} requires an int64 target copy, got dtype "
            f"{targets.dtype}"
        )
    if targets.ndim != 1:
        raise ValueError(
            f"{op_name} requires a one-dimensional target copy, got shape "
            f"{targets.shape}"
        )
    if not targets.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{op_name} requires a contiguous target copy")
    if targets.size != batch_size:
        raise ValueError(
            f"{op_name} needs exactly {batch_size} targets (one per "
            f"probability row), got {targets.size}"
        )
    # Plain Python comparison over the labels — no NumPy arithmetic, so
    # this stays clear of the numerical tripwire guarding the native path.
    for index, value in enumerate(targets.tolist()):
        if value < 0 or value >= num_classes:
            raise ValueError(
                f"{op_name} target at index {index} is {value}, outside the "
                f"valid class range [0, {num_classes})"
            )
    return targets


class _CrossEntropyForwardResult:
    """What ``NativeTensorCore.cross_entropy_forward`` hands back (E5).

    A deliberately minimal, **graph-unaware** record of one fused forward
    — not a module, not a graph node, and not a public API object:

    * ``loss`` — a fresh owning scalar core (shape ``()``);
    * ``probabilities`` — a fresh owning contiguous
      ``(batch_size, num_classes)`` core, the softmax the backward
      consumes. It is **private state**: never a public NativeTensor,
      never a parameter or buffer, never in a ``state_dict()`` or a
      checkpoint, and never advertised in any capability inventory;
    * ``targets`` — the independently owned, read-only ``int64`` copy of
      the class labels;
    * ``reduction`` — the normalized ``"mean"``/``"sum"`` name.

    Ownership is explicit, as everywhere in the native runtime: the
    caller owns both cores and releases them with ``close()`` (which
    closes both and is idempotent, since ``NativeTensorCore.close()`` is)
    or by closing them individually. There is deliberately **no**
    ``__del__`` here — nothing is finalized behind the caller's back, and
    nothing claims graph ownership. E6 will adopt the probabilities as a
    graph resource; at E5 they are simply the caller's."""

    __slots__ = ("loss", "probabilities", "targets", "reduction")

    def __init__(self, loss, probabilities, targets, reduction):
        self.loss = loss
        self.probabilities = probabilities
        self.targets = targets
        self.reduction = reduction

    def close(self):
        """Release both native outputs. Idempotent; safe in any order."""
        self.probabilities.close()
        self.loss.close()

    def __repr__(self):
        return (
            f"_CrossEntropyForwardResult(reduction={self.reduction!r}, "
            f"probabilities={self.probabilities.shape}, "
            f"targets={self.targets.size})"
        )


# ---------------------------------------------------------------------------
# Phase-G Dropout argument validation (milestone G2)
#
# The random key and the probability are the two things the stateless
# Dropout Core accepts beyond a tensor, and both are validated here, in
# one place, so the accepted/rejected matrix is identical wherever they
# are taken (docs/native_rng_dropout_design.md §6.1 — the pattern Phase E
# used for cross-entropy targets and reductions). Pure Python: neither
# helper touches the compiled library, so a rejected call allocates
# nothing at all.
#
# These validate *values*, never generator state: nothing here reads,
# reserves, commits, or advances a NativeGenerator, and this module does
# not import the experimental package.
# ---------------------------------------------------------------------------

# The unsigned 64-bit range the random key lives in. Mirrors
# NativeGenerator's own bound (design §3.3); repeated rather than imported
# because backends/ stays decoupled from experimental/.
_UINT64_MAX = 2 ** 64 - 1


def _validate_random_key_field(value, what):
    """Validate one half of the random key as an exact ``uint64``.

    Exact-type discipline, matching ``NativeGenerator``'s seed validator:
    ``bool`` is not a seed (``True`` is not ``1`` here), a NumPy integer
    scalar is not a Python ``int``, and an ``int`` subclass is not an
    ``int``. Python ints are arbitrary precision, so an out-of-range value
    is a ValueError rather than a silent truncation at the ABI.

    The accepted range is the full ``[0, 2**64 - 1]`` the ctypes
    ``c_uint64`` argument carries. A ``NativeGenerator`` never *issues* a
    call index above ``2**64 - 2`` (design §4.6), but that is the
    generator's counter rule, not a property of this stateless key
    space."""
    if type(value) is not int:
        raise TypeError(
            f"{what} must be an int, got {type(value).__name__}"
        )
    if not 0 <= value <= _UINT64_MAX:
        raise ValueError(
            f"{what} must be in [0, {_UINT64_MAX}], got {value}"
        )
    return value


def _normalize_dropout_probability(p, op_name):
    """Validate a Dropout probability and normalize it to a plain float.

    The locked matrix (docs/native_rng_dropout_design.md §6.1):

    ============================ ==========================================
    ``bool``                     TypeError — a bool is not a probability
    non-real                     TypeError
    ``0`` / ``0.0``              accepted, normalized to ``0.0``
    real in ``[0.0, 1.0)``       accepted, normalized with ``float(p)``
    ``1`` / ``1.0``              ValueError — rejected (§6.3)
    ``p > 1`` / ``p < 0``        ValueError
    NaN                          ValueError naming NaN explicitly
    ``+inf`` / ``-inf``          ValueError
    ============================ ==========================================

    ``numbers.Real`` is the accepted abstract type, so a NumPy float or
    integer scalar is accepted and normalized — the same latitude the
    stable ``tensorforge.nn.Dropout`` gives — while ``bool`` and
    ``numpy.bool_`` are rejected before that test, since ``True`` would
    otherwise sail through as ``1.0``.

    ``p == 1`` is a genuine rejection rather than a special case: the
    inverted multiplier ``1 / (1 - p)`` would divide by zero, and every
    alternative (an ``inf`` multiplier, a silent all-zero output) changes
    the layer's contract instead of reporting the problem."""
    if isinstance(p, (bool, np.bool_)):
        raise TypeError(
            f"{op_name} probability must be a real number, not a bool "
            f"({p!r}); True is not a probability"
        )
    if not isinstance(p, numbers.Real):
        raise TypeError(
            f"{op_name} probability must be a real number, got "
            f"{type(p).__name__}"
        )
    value = float(p)
    if math.isnan(value):
        raise ValueError(
            f"{op_name} probability must not be NaN"
        )
    if math.isinf(value):
        raise ValueError(
            f"{op_name} probability must be finite, got {p!r}"
        )
    if not 0.0 <= value < 1.0:
        raise ValueError(
            f"{op_name} probability must satisfy 0 <= p < 1, got {p!r} "
            f"(p == 1 is rejected: the inverted multiplier 1/(1 - p) "
            f"would divide by zero)"
        )
    return value


# ---------------------------------------------------------------------------
# Shape/stride metadata
#
# The Python-facing metadata layer that prepares the path for a later
# native tensor storage object: shape, strides, element counts,
# contiguity checks, and flat-offset math. Implemented in Python on
# purpose — the deliverable is the *contract* (what shapes/strides
# mean), and integer arithmetic gains nothing from a ctypes round
# trip. These helpers never touch the compiled library, so they are
# safe whether or not the backend is built.
#
# Conventions:
# - strides count ELEMENTS, not bytes (unlike numpy.ndarray.strides);
#   row_major_strides((2, 3, 4)) == (12, 4, 1).
# - dimensions must be positive ints; zero-size dimensions are
#   rejected in v0.7 (their stride conventions deserve their own
#   tested milestone).
# - the scalar shape () has strides (), ndim 0, and numel 1.
# ---------------------------------------------------------------------------


def _as_int_tuple(values, name):
    try:
        items = tuple(values)
    except TypeError:
        raise TypeError(
            f"{name} must be a sequence of ints, got {values!r}"
        ) from None
    # Every element is type-checked exactly as before. The `plain` flag only
    # records whether the rebuild below is *needed*: a tuple that is already
    # all exact ``int`` is its own normalization, so re-materializing it
    # through a generator would allocate a second, equal tuple for nothing.
    # ``type(value) is int`` deliberately excludes ``bool`` and every ``int``
    # subclass, so those still take the checking branch and are still
    # converted (or rejected) exactly as they were.
    plain = True
    for value in items:
        if type(value) is int:
            continue
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(f"{name} must contain only ints, got {value!r}")
        plain = False
    return items if plain else tuple(int(value) for value in items)


def _as_shape(shape):
    dims = _as_int_tuple(shape, "shape")
    for dim in dims:
        if dim <= 0:
            raise ValueError(
                f"shape dimensions must be positive ints, got {dims} "
                f"(zero-size dimensions are not supported in v0.7)"
            )
    return dims


def _as_offset(offset):
    if not isinstance(offset, (int, np.integer)) or isinstance(offset, bool):
        raise TypeError(f"offset must be an int, got {offset!r}")
    return int(offset)


# ---------------------------------------------------------------------------
# Checked primitives (Phase H, milestone H3)
#
# Each ``_..._checked`` function below computes exactly what its public
# counterpart computes, but takes metadata **this module has already
# normalized** — a tuple of exact ``int`` dimensions from ``_as_shape``, or
# strides/axes derived from one. They perform no validation because there is
# nothing left to validate: re-running ``_as_int_tuple`` over a tuple this
# module just produced cannot reject it and cannot change it.
#
# They are private and unexported. The public ``row_major_strides`` /
# ``numel`` / ``reduce_shape`` / ``broadcast_shapes`` remain the validating
# entry points for anything a caller supplies, with unchanged signatures,
# unchanged behavior, and unchanged messages — each is now that validation
# followed by the matching primitive. The rule for using a ``_checked``
# variant is narrow and mechanical: the argument must be a shape tuple that
# came out of ``_as_shape`` (or out of a live view's ``shape``/``strides``,
# which is the same thing one construction earlier). An argument that came
# from a caller goes through the public function.
# ---------------------------------------------------------------------------


def _row_major_strides_checked(dims):
    """``row_major_strides`` over already-validated ``dims``."""
    strides = []
    running = 1
    for dim in reversed(dims):
        strides.append(running)
        running *= dim
    return tuple(reversed(strides))


def _numel_checked(dims):
    """``numel`` over already-validated ``dims``."""
    count = 1
    for dim in dims:
        count *= dim
    return count


def row_major_strides(shape):
    """Element strides for a row-major contiguous layout of ``shape``.

    The last dimension varies fastest: row_major_strides((2, 3, 4))
    is (12, 4, 1). The scalar shape () gives ().
    """
    return _row_major_strides_checked(_as_shape(shape))


def numel(shape):
    """Number of elements in ``shape``; 1 for the scalar shape ()."""
    return _numel_checked(_as_shape(shape))


def is_contiguous_shape(shape, strides):
    """True if ``strides`` is exactly the row-major contiguous layout
    for ``shape``. A scalar () with strides () is contiguous."""
    dims = _as_shape(shape)
    stride_tuple = _as_int_tuple(strides, "strides")
    if len(stride_tuple) != len(dims):
        raise ValueError(
            f"shape and strides must have the same length, "
            f"got {len(dims)} and {len(stride_tuple)}"
        )
    return stride_tuple == _row_major_strides_checked(dims)


def flat_offset(indices, strides, offset=0):
    """Flat storage position of a logical index: offset + sum(i * s).

    Pure stride math — no shape is involved, so no bounds checking is
    performed, and negative indices or strides are allowed (real
    strided views use negative strides).
    """
    index_tuple = _as_int_tuple(indices, "indices")
    stride_tuple = _as_int_tuple(strides, "strides")
    if len(index_tuple) != len(stride_tuple):
        raise ValueError(
            f"indices and strides must have the same length, "
            f"got {len(index_tuple)} and {len(stride_tuple)}"
        )
    return _as_offset(offset) + sum(
        index * stride for index, stride in zip(index_tuple, stride_tuple)
    )


def _normalized_layout(shape, strides=None, offset=0):
    """The one validating normalization boundary for a native layout
    (Phase H, milestone H3).

    Returns ``(dims, strides, offset, numel, contiguous)`` — the five
    values every internal consumer of a layout actually wants — having
    performed **exactly** the checks ``shape_info`` has always performed,
    in exactly the same order, with exactly the same messages:

    1. the shape (type, then positivity),
    2. the strides (element type, then length against the shape),
    3. the offset.

    The shape is normalized **once**. Everything downstream of that — the
    row-major strides, the element count, the contiguity comparison — is
    derived from the resulting tuple through the ``_checked`` primitives,
    because a tuple ``_as_shape`` just returned cannot fail ``_as_shape``
    again. Before H3 this function's work was spread across ``shape_info``
    in a form that re-validated the same tuple four times and computed the
    row-major strides twice; the values it produces are identical.
    """
    dims = _as_shape(shape)
    contiguous_strides = _row_major_strides_checked(dims)
    if strides is None:
        stride_tuple = contiguous_strides
        contiguous = True  # by construction, not by comparison
    else:
        stride_tuple = _as_int_tuple(strides, "strides")
        if len(stride_tuple) != len(dims):
            raise ValueError(
                f"shape and strides must have the same length, "
                f"got {len(dims)} and {len(stride_tuple)}"
            )
        contiguous = stride_tuple == contiguous_strides
    # The offset is validated last, as it always was: neither the stride
    # derivation nor the element count can raise on a validated shape, so
    # this is still the third and final rejection point.
    return dims, stride_tuple, _as_offset(offset), _numel_checked(dims), contiguous


def shape_info(shape, strides=None, offset=0):
    """A small metadata dictionary describing one array layout.

    With ``strides=None`` the row-major contiguous strides are used
    (and ``contiguous`` is True by construction). Explicit strides are
    validated against the shape's length and checked for contiguity.
    """
    dims, stride_tuple, offset_value, count, contiguous = _normalized_layout(
        shape, strides=strides, offset=offset
    )
    return {
        "shape": dims,
        "strides": stride_tuple,
        "ndim": len(dims),
        "numel": count,
        "offset": offset_value,
        "contiguous": contiguous,
    }


def broadcast_shapes(shape_a, shape_b):
    """The NumPy-style broadcast of two shapes, or a clear ValueError.

    Shapes are aligned from the trailing axis; the shorter one is
    conceptually left-padded with leading 1s. Two extents are compatible
    when they are equal or one of them is 1, and the result extent is
    their max. The scalar shape () broadcasts against anything.

        broadcast_shapes((), (3, 4))          # (3, 4)
        broadcast_shapes((3, 1), (1, 4))      # (3, 4)
        broadcast_shapes((4,), (3, 4))        # (3, 4)  (left-pad to (1, 4))
        broadcast_shapes((1, 3, 1), (2, 1, 5))# (2, 3, 5)
        broadcast_shapes((2, 3), (4, 3))      # ValueError (2 vs 4)

    Pure Python — it never calls NumPy and never touches the compiled
    library, so it is safe and testable whether or not the backend is
    built. Incompatible shapes raise a ValueError naming both original
    shapes and the conflicting extents.
    """
    return _broadcast_shapes_checked(
        _as_shape(shape_a),  # validates positive-int dims (v0.7 rules)
        _as_shape(shape_b),
    )


def _broadcast_shapes_checked(a, b):
    """``broadcast_shapes`` over two already-validated shape tuples."""
    rank = max(len(a), len(b))
    pa = (1,) * (rank - len(a)) + a  # left-pad with leading 1s
    pb = (1,) * (rank - len(b)) + b
    out = []
    for da, db in zip(pa, pb):
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            raise ValueError(
                f"cannot broadcast shapes {a} and {b}: incompatible "
                f"dimensions {da} and {db} (neither is 1)"
            )
    return tuple(out)


def _broadcast_strides(shape, strides, out_shape):
    """Read-strides that stretch a (real) ``shape``/``strides`` operand
    over ``out_shape`` without materializing it.

    For each output axis: a real axis that carries a genuine extent keeps
    its real stride; a size-1 axis (or a leading axis introduced by
    left-padding) gets stride 0, so the odometer re-reads the same
    element instead of advancing — that is exactly broadcasting. Assumes
    ``out_shape`` is the broadcast of ``shape`` with the other operand
    (compatibility already checked by ``broadcast_shapes``)."""
    rank = len(out_shape)
    pad = rank - len(shape)
    result = [0] * rank  # leading padded axes stay 0
    for i, dim in enumerate(shape):
        if dim != 1:  # a stretched size-1 axis keeps stride 0
            result[pad + i] = strides[i]
    return result


def _normalize_axis(axis, shape):
    """Normalize a single reduction ``axis`` against ``shape``.

    Accepts a plain int (negative allowed, NumPy-style: ``axis + ndim``),
    validates its type and bounds, and returns the non-negative axis.
    Raises ``TypeError`` for a non-int axis and ``ValueError`` naming both
    the axis and the shape when out of bounds (including any integer axis
    on a scalar). Pure Python — no NumPy, no compiled library."""
    return _normalize_axis_checked(axis, _as_shape(shape))


def _normalize_axis_checked(axis, dims):
    """``_normalize_axis`` over an already-validated shape tuple. The
    ``axis`` itself is still fully validated — it is the caller-supplied
    half of this pair and is never assumed."""
    ndim = len(dims)
    if not isinstance(axis, (int, np.integer)) or isinstance(axis, bool):
        raise TypeError(f"axis must be None or an int, got {axis!r}")
    value = int(axis)
    normalized = value + ndim if value < 0 else value
    if normalized < 0 or normalized >= ndim:
        raise ValueError(
            f"axis {value} is out of bounds for a tensor of shape {dims} "
            f"(ndim {ndim})"
        )
    return normalized


def reduce_shape(shape, axis=None, keepdims=False):
    """The output shape of reducing ``shape`` over ``axis``.

    ``axis=None`` reduces every element; a single integer ``axis``
    (negative allowed) reduces one dimension. ``keepdims=True`` leaves
    each reduced axis as size 1, ``keepdims=False`` (default) removes it.

        reduce_shape((2, 3))                       # ()
        reduce_shape((2, 3), keepdims=True)        # (1, 1)
        reduce_shape((2, 3), axis=0)               # (3,)
        reduce_shape((2, 3), axis=1, keepdims=True)# (2, 1)
        reduce_shape((), axis=None)                # ()
        reduce_shape((2, 3, 4), axis=-1)           # (2, 3)

    Pure Python — never calls NumPy, never touches the compiled library,
    so it is testable whether or not the backend is built. Tuple/multiple
    axes are not supported yet (a single int or None only). Raises
    ``TypeError`` for a non-bool ``keepdims`` or non-int ``axis``, and
    ``ValueError`` naming both axis and shape for an out-of-bounds axis.
    """
    return _reduce_shape_checked(_as_shape(shape), axis, keepdims)


def _reduce_shape_checked(dims, axis=None, keepdims=False):
    """``reduce_shape`` over an already-validated shape tuple. ``axis``
    and ``keepdims`` are caller-supplied and still fully validated."""
    if not isinstance(keepdims, bool):
        raise TypeError(f"keepdims must be a bool, got {keepdims!r}")
    ndim = len(dims)
    if axis is None:
        return (1,) * ndim if keepdims else ()
    normalized = _normalize_axis_checked(axis, dims)
    if keepdims:
        return tuple(1 if d == normalized else dims[d] for d in range(ndim))
    return tuple(dims[d] for d in range(ndim) if d != normalized)


def _reduce_out_strides(in_shape, reduced_axes, keepdims, out_shape):
    """Per-input-axis output write-strides for the sum kernel.

    For each input axis: 0 if it is reduced (so those elements
    accumulate into one output cell), otherwise the row-major stride of
    the axis it maps to in ``out_shape``. With ``keepdims`` the output
    keeps the reduced axes (as size 1), so input axis ``d`` maps to
    output axis ``d``; without it, the kept input axes map in order to the
    surviving output axes. Assumes ``out_shape == reduce_shape(in_shape,
    ...)`` for the same reduction, which makes ``out_shape`` already
    validated."""
    out_full = _row_major_strides_checked(out_shape)
    result = [0] * len(in_shape)
    if keepdims:
        for d in range(len(in_shape)):
            result[d] = 0 if d in reduced_axes else out_full[d]
    else:
        out_index = 0
        for d in range(len(in_shape)):
            if d in reduced_axes:
                continue  # reduced axis: stays 0, no output axis consumed
            result[d] = out_full[out_index]
            out_index += 1
    return result


def _spatial_pair(value, name, minimum):
    """Normalize a spatial argument to a ``(height, width)`` int pair.

    Mirrors the stable ``tensorforge.nn.conv._pair`` semantics in the
    native package's strict style (the two lines never cross-import): a
    plain int ``v`` becomes ``(v, v)``; a 2-element tuple/list of ints is
    taken as ``(height, width)``. Booleans are rejected (``bool`` is an
    ``int`` subclass, so ``True``/``False`` are never valid dimensions) and
    every member must be at least ``minimum`` (``1`` for kernel/stride,
    ``0`` for padding). Raises ``ValueError`` otherwise. Pure Python — never
    touches the compiled library, so it is safe whether or not the backend
    is built."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        pair = (int(value), int(value))
    elif (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool)
            for v in value
        )
    ):
        pair = (int(value[0]), int(value[1]))
    else:
        raise ValueError(
            f"{name} must be an int or a 2-element pair of ints, got {value!r}"
        )
    if any(v < minimum for v in pair):
        raise ValueError(f"{name} values must be >= {minimum}, got {pair}")
    return pair


def conv_output_shape(input_size, kernel_size, stride, padding):
    """The ``(out_h, out_w)`` of a 2-D convolution / pooling window.

    Applies the locked floor formula per spatial axis (identical to the
    stable Conv2d)::

        out = (size + 2*pad - kernel) // stride + 1

    Every argument is a validated ``(height, width)`` pair of ints. The
    arithmetic runs in Python ints (arbitrary precision), so the
    shape math itself can never overflow. Raises ``ValueError`` — naming
    the kernel, stride, padding, and input — when either extent would be
    ``< 1`` (the kernel does not fit the padded input), *before* any output
    is allocated. Pure Python; never touches the compiled library."""
    h, w = input_size
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    if out_h < 1 or out_w < 1:
        raise ValueError(
            f"kernel {(kh, kw)} with stride {(sh, sw)} and padding {(ph, pw)} "
            f"does not fit input {(h, w)}: computed output {(out_h, out_w)} has "
            f"a non-positive extent"
        )
    return out_h, out_w


def _binary_op(kernel_name, a, b, name):
    """Shared plumbing for the binary kernels: convert both inputs to
    contiguous float64, require identical shapes, call the kernel."""
    lib = _require_library()
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"the experimental C++ {name} requires identical "
            f"shapes (no broadcasting), got {a.shape} and {b.shape}"
        )
    out = np.empty_like(a)
    getattr(lib, kernel_name)(a, b, out, a.size)
    return out


def elementwise_add(a, b):
    """a + b elementwise, using the compiled C++ kernel."""
    return _binary_op("tf_elementwise_add", a, b, "elementwise_add")


def elementwise_subtract(a, b):
    """a - b elementwise, using the compiled C++ kernel."""
    return _binary_op("tf_elementwise_subtract", a, b, "elementwise_subtract")


def elementwise_multiply(a, b):
    """a * b elementwise, using the compiled C++ kernel."""
    return _binary_op("tf_elementwise_multiply", a, b, "elementwise_multiply")


def elementwise_divide(a, b):
    """a / b elementwise, using the compiled C++ kernel.

    IEEE float64 division: dividing by zero yields +-inf (or NaN for
    0/0), the same values NumPy produces — but without NumPy's runtime
    warning.
    """
    return _binary_op("tf_elementwise_divide", a, b, "elementwise_divide")


def relu(a):
    """max(a, 0) elementwise, using the compiled C++ kernel.

    Unary: accepts any shape, returns a new float64 array of the same
    shape.
    """
    lib = _require_library()
    a = np.ascontiguousarray(a, dtype=np.float64)
    out = np.empty_like(a)
    lib.tf_relu(a, out, a.size)
    return out


def _prepare_matmul_inputs(a, b, name):
    """Shared matmul validation: contiguous float64, strictly 2-D,
    compatible inner dimensions. Returns (a, b)."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(
            f"the experimental C++ {name} requires a 2-D left input, "
            f"got shape {a.shape}"
        )
    if b.ndim != 2:
        raise ValueError(
            f"the experimental C++ {name} requires a 2-D right input, "
            f"got shape {b.shape}"
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"inner dimensions do not match: "
            f"{a.shape} @ {b.shape} (need (m, n) @ (n, p))"
        )
    return a, b


def matmul(a, b):
    """(m, n) @ (n, p) matrix multiplication using the compiled C++
    kernel — the naive triple loop, kept as the reference that
    matmul_tiled is measured against. Correct but much slower than
    NumPy's BLAS-backed matmul.

    Strictly 2-D: vectors must be passed as (1, n) or (n, 1) matrices.
    Returns a new (m, p) float64 array.
    """
    lib = _require_library()
    a, b = _prepare_matmul_inputs(a, b, "matmul")
    m, n = a.shape
    p = b.shape[1]
    out = np.empty((m, p), dtype=np.float64)
    lib.tf_matmul(a, b, out, m, n, p)
    return out


def matmul_tiled(a, b, block_size=32):
    """(m, n) @ (n, p) matrix multiplication using the tiled C++
    kernel — an optimization experiment in cache blocking. Same
    contract as ``matmul``; ``block_size`` sets the tile edge length
    and any positive int works, including sizes that don't divide the
    matrix dimensions.

    Blocking improves memory locality over the naive loop, but this is
    still single-threaded scalar code: NumPy's BLAS may well remain
    faster. Returns a new (m, p) float64 array.
    """
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError(
            f"block_size must be a positive int, got {block_size!r}"
        )
    lib = _require_library()
    a, b = _prepare_matmul_inputs(a, b, "matmul_tiled")
    m, n = a.shape
    p = b.shape[1]
    out = np.empty((m, p), dtype=np.float64)
    lib.tf_matmul_tiled(a, b, out, m, n, p, block_size)
    return out
