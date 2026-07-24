"""Lightweight checks that the docs exist and stay in sync with the repo."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = (
    "architecture.md",
    "autograd.md",
    "training.md",
    "examples.md",
    "roadmap.md",
    "release_history.md",
    "project_summary.md",
    "backend_experiments.md",
    "dispatch_design.md",
    "native_tensor_wrapper_design.md",
    "native_contiguous_fast_path_design.md",
    "native_broadcasting_design.md",
    "native_reductions_design.md",
    "native_dtype_device_metadata_design.md",
    "native_autograd_design.md",
    "native_autograd_benchmarks.md",
    "native_support_matrix.md",
    "native_cnn_design.md",
    "native_classification_design.md",
    "native_normalization_design.md",
)

EXAMPLE_FILES = (
    "train_linear_regression.py",
    "train_xor.py",
    "train_multiclass.py",
    "train_binary_classification.py",
    "train_mlp_with_dropout.py",
)


def test_docs_files_exist():
    for name in DOCS:
        path = REPO_ROOT / "docs" / name
        assert path.is_file(), f"missing docs/{name}"
        assert path.read_text(encoding="utf-8").strip(), f"docs/{name} is empty"


def test_readme_links_to_all_docs():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for name in DOCS:
        assert f"docs/{name}" in readme, f"README does not link to docs/{name}"


def test_examples_doc_mentions_all_examples():
    text = (REPO_ROOT / "docs" / "examples.md").read_text(encoding="utf-8")
    for filename in EXAMPLE_FILES:
        assert filename in text, f"docs/examples.md does not mention {filename}"


def test_examples_doc_commands_reference_existing_files():
    text = (REPO_ROOT / "docs" / "examples.md").read_text(encoding="utf-8")
    referenced = re.findall(r"examples/(\w+\.py)", text)
    assert referenced, "docs/examples.md contains no example commands"
    for filename in referenced:
        assert (REPO_ROOT / "examples" / filename).is_file(), (
            f"docs/examples.md references examples/{filename}, which does not exist"
        )


def test_all_example_files_are_documented():
    """The reverse direction: every real example appears in the doc."""
    text = (REPO_ROOT / "docs" / "examples.md").read_text(encoding="utf-8")
    for path in (REPO_ROOT / "examples").glob("train_*.py"):
        assert path.name in text, f"examples/{path.name} is not documented"


def test_roadmap_does_not_list_shipped_stable_features_as_future():
    """A feature the **stable** framework already ships must never appear
    in the roadmap's future-work section.

    A *native* counterpart of a shipped stable feature is a different
    thing and legitimately can be future work — the native line has its
    own phase sequence — so explicitly native phrasings are stripped
    before the scan. (This generalizes the long-standing "NativeAdam"
    exemption instead of banning honest native-line statements.)"""
    import re

    text = (REPO_ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    future = text.split("## Practical next steps", 1)[1]
    future = future.split("## What this project is not", 1)[0]
    # Strip "Native<Thing>" identifiers and "native <thing>" prose: both
    # name the native counterpart, never the shipped stable feature.
    future = re.sub(r"Native[A-Z]\w*", "", future)
    future = re.sub(r"native [a-zA-Z/]+(\s+[a-z/]+){0,2}", "", future)
    shipped_features = (
        "Dropout",
        "BatchNorm",
        "LayerNorm",
        "Conv2d",
        "MaxPool2d",
        "Adam",
        "clip_grad",
        "RNG",
    )
    for shipped in shipped_features:
        assert shipped not in future, (
            f"docs/roadmap.md lists already-shipped stable {shipped!r} as "
            f"future work"
        )


def test_project_summary_covers_the_essentials():
    text = (REPO_ROOT / "docs" / "project_summary.md").read_text(encoding="utf-8")
    for topic in ("NumPy", "autograd", "checkpoint", "CNN", "C++", "CUDA"):
        assert topic in text, f"docs/project_summary.md does not mention {topic!r}"


def test_readme_presents_backends_accurately():
    """As of the Advanced C++ v3.10 checkpoint the native C++ CPU line
    exists and trains end to end, and CUDA still does not: the README
    must not regress to claiming the backend is absent or merely
    started, and must keep marking CUDA as future work."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"no C\+\+ backend yet", readme, re.IGNORECASE), (
        "README claims the C++ backend does not exist; the native line "
        "shipped in Phases A-C"
    )
    assert re.search(r"no CUDA backend\s+yet", readme, re.IGNORECASE), (
        "README must keep marking CUDA as future work"
    )
    # The native line's presence must be stated, not implied.
    for term in ("NativeTensor", "NativeSGD", "native_mlp_training.py",
                 "native_support_matrix.md", "tensorforge.experimental"):
        assert term in readme, f"README does not present {term!r}"


def test_native_support_matrix_is_canonical_and_honest():
    """The support matrix must cover the shipped native surface and
    keep unshipped work in its unsupported section."""
    text = (REPO_ROOT / "docs" / "native_support_matrix.md").read_text(
        encoding="utf-8"
    )
    shipped = (
        "NativeStorage", "NativeTensorView", "NativeTensorCore",
        "NativeTensor",
        "add", "subtract", "multiply", "relu", "matmul", "sum", "mean",
        "sqrt", "reciprocal",  # v3.11 optimizer math primitives
        "reshape", "transpose", "narrow", "contiguous_copy",
        "retain_graph", "Stale parameter-version detection",
        "NativeParameter", "NativeModule", "state_dict", "NativeLinear",
        "NativeReLU", "NativeSequential", "NativeMSELoss", "NativeSGD",
        "NativeAdam",  # v3.12 adaptive optimizer
        "load_state_dict",  # v3.13 optimizer state contract
        "save_native_checkpoint", "load_native_checkpoint",  # v3.14
        "native_checkpoint_resume.py",
        "native_mlp_training.py",
    )
    for term in shipped:
        assert term in text, f"support matrix does not cover {term!r}"
    # Phase D shipped Conv2d/MaxPool2d, so the matrix must now *present*
    # them (D12); what still has to stay in the unsupported section is the
    # work that genuinely does not exist.
    for term in ("NativeConv2d", "NativeMaxPool2d", "NativeFlatten",
                 "native_cnn_training.py"):
        assert term in text, f"support matrix does not cover shipped {term!r}"
    assert "## Unsupported or future" in text
    supported_part = text.split("## Unsupported or future", 1)[0]
    future_part = text.split("## Unsupported or future", 1)[1]
    for term in ("CUDA", "float32", "AMP",
                 "AdamW", "AMSGrad", "weight decay", "distributed"):
        assert term in future_part, (
            f"support matrix does not list {term!r} as unsupported/future"
        )
        assert term not in supported_part, (
            f"support matrix mentions unshipped {term!r} outside the "
            f"unsupported section"
        )


def _normalized_doc(relative_path):
    """A doc's text with runs of whitespace collapsed, so status
    guardrails survive line rewrapping instead of locking exact wraps."""
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text)


def test_phase_c_marked_complete_not_in_progress():
    """Phase C (the native training stack) is complete as of Advanced
    C++ v3.15. The project-facing docs must state that and must never
    silently revert to 'in progress'/'under way', nor describe shipped
    optimizer state or file resume as future work."""
    readme = _normalized_doc("README.md")
    summary = _normalized_doc("docs/project_summary.md")
    matrix = _normalized_doc("docs/native_support_matrix.md")
    # The exact regression phrases earlier milestones used (whitespace
    # normalized so rewrapping cannot defeat the guard).
    assert "(Phase C, in progress)" not in readme
    assert "Phase C (the native training stack) is in progress" not in summary
    assert "native training stack (under way)" not in matrix
    # Completion is positively stated in the summary and matrix.
    assert re.search(r"Phase C[^.]{0,90}complete", summary), (
        "docs/project_summary.md no longer states Phase C is complete"
    )
    assert "Phase C" in matrix and "complete" in matrix
    # NativeAdam and checkpointing are shipped native capabilities the
    # README must present, never omit.
    assert "NativeAdam" in readme
    assert re.search(r"checkpoint", readme, re.IGNORECASE), (
        "README no longer presents native checkpointing"
    )
    # The native CNN stack (Phase D) is complete, while CUDA remains not
    # started. The roadmap must keep naming the stack and keep marking
    # CUDA future work.
    roadmap = _normalized_doc("docs/roadmap.md")
    assert "native CNN stack" in roadmap
    assert "not started" in roadmap


def test_native_checkpoint_apis_stay_out_of_stable_serialization():
    """The native checkpoint APIs live only in tensorforge.experimental
    — the stable serialization module must not reference, import, or
    re-export them (the two lines never mix)."""
    stable = (
        REPO_ROOT / "src" / "tensorforge" / "serialization.py"
    ).read_text(encoding="utf-8")
    for banned in ("native", "experimental",
                   "save_native_checkpoint", "load_native_checkpoint"):
        assert banned not in stable, (
            f"stable serialization.py references {banned!r}; the native "
            f"checkpoint APIs must stay in tensorforge.experimental"
        )


def test_experimental_exports_stay_intentional():
    """The native public surface is explicit: exactly these names from
    tensorforge.experimental, and none of them leaking into the stable
    top-level namespace. Importing experimental is always safe (the
    compiled library loads lazily)."""
    import tensorforge
    import tensorforge.experimental as experimental

    assert set(experimental.__all__) == {
        "NativeTensor", "NativeParameter", "NativeParameterRegistry",
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeMSELoss", "NativeSGD", "NativeAdam",
        "save_native_checkpoint", "load_native_checkpoint",
        "NativeCrossEntropyLoss", "native_accuracy",   # Phase E, E7
        "NativeLayerNorm",                              # Phase F, F2
        "NativeBatchNorm1d",                            # Phase F, F3
    }
    for name in experimental.__all__:
        assert hasattr(experimental, name)
        assert not hasattr(tensorforge, name), (
            f"{name} leaked into the stable top-level tensorforge namespace"
        )


def test_roadmap_does_not_list_v3_as_upcoming():
    text = (REPO_ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    future = text.split("## Practical next steps", 1)[1]
    future = future.split("## What this project is not", 1)[0]
    assert "v3.0" not in future, (
        "docs/roadmap.md still lists v3.0 as upcoming work"
    )


def test_positioning_language():
    """TensorForge is positioned as a serious ML systems framework.
    Project-facing docs must not reintroduce weak positioning. The
    banned list is phrase-level, not word-level, so honest limitation
    language and historical notes stay possible."""
    banned = (
        "educational framework",
        "educational deep learning",
        "educational take",
        "educational project",
        "educational toy",
        "toy framework",
        "mini framework",
        "mini deep learning",
        "learning project",
        "teaching framework",
    )
    files = [REPO_ROOT / "README.md", REPO_ROOT / "CLAUDE.md",
             *(REPO_ROOT / "docs").glob("*.md")]
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        for term in banned:
            assert term not in text, (
                f"{path.name} uses banned positioning phrase {term!r}"
            )


def test_readme_mentions_all_example_files():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for path in (REPO_ROOT / "examples").glob("train_*.py"):
        assert path.name in readme, f"README does not mention examples/{path.name}"


def test_readme_commands_reference_existing_files():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    referenced = re.findall(r"examples/(\w+\.py)", readme)
    assert referenced, "README contains no example commands"
    for filename in referenced:
        assert (REPO_ROOT / "examples" / filename).is_file(), (
            f"README references examples/{filename}, which does not exist"
        )


def test_native_cnn_design_locks_the_phase_d_contract():
    """The Phase-D (native CNN) architecture contract must state its
    load-bearing decisions, so later milestones inherit an unambiguous
    design instead of re-deriving it."""
    text = _normalized_doc("docs/native_cnn_design.md")
    # Layout and math conventions.
    for token in ("NCHW", "OIHW", "cross-correlation"):
        assert token in text, f"native_cnn_design.md does not lock {token!r}"
    # The three planned public surfaces are named.
    for token in ("NativeFlatten", "NativeConv2d", "NativeMaxPool2d"):
        assert token in text, f"native_cnn_design.md does not mention {token!r}"
    # The non-contiguous-input policy and the max-pool winner contract.
    assert "winner" in text.lower(), "design omits the max-pool winner contract"
    assert "contiguous" in text.lower(), "design omits the contiguity policy"
    # The full milestone ladder must be present (D0 through D12), and D6
    # (autograd integration) must sit between D5 and D7.
    for i in range(13):
        assert f"D{i}" in text, f"native_cnn_design.md is missing milestone D{i}"
    # Anchor on the milestone-ladder headers ("D5 —", "D6 —", "D7 —"),
    # which use an em-dash and so do not collide with bare-"D5"/"D7"
    # mentions in the prose.
    d5 = text.index("D5 —")
    d6 = text.index("D6 —")
    d7 = text.index("D7 —")
    assert d5 < d6 < d7, "D6 (autograd integration) is not ordered between D5 and D7"


def test_native_cnn_design_locks_conditional_versioning_and_winner_safety():
    """The two subtle Phase-D contracts must stay pinned: Conv2d records a
    parameter version only for a value an *active* backward callback
    rereads, and the max-pool winner buffer stays an exact-integer float64
    buffer bounded by 2^53."""
    text = _normalized_doc("docs/native_cnn_design.md")
    # Conditional Conv2d versioning keys off which callbacks run.
    for token in ("input._requires_grad", "weight._requires_grad"):
        assert token in text, f"design no longer states the conditional {token!r} rule"
    # Bias-only backward must not version-guard input/weight.
    assert "bias-only" in text.lower(), "design omits the bias-only no-version case"
    # Winner-index float64 safety: the exactness bound and the -1 sentinel.
    assert "2^53" in text, "design no longer pins the float64 exactness bound"
    assert "-1" in text, "design no longer documents the -1 winner sentinel"
    # NaN handling is a documented, deliberate choice, not left implicit.
    assert "NaN" in text and "divergence" in text.lower(), (
        "design no longer documents the deliberate NaN divergence from stable"
    )


def test_phase_d_status_is_consistent_across_docs_and_registry():
    """Phase D is complete (D12). The durable guarantee is *agreement*: the
    design doc, the support matrix, the roadmap, the public exports, and the
    backend registry must all describe the same shipped surface, and the
    Phase-D artifacts they reference must exist. This replaces the earlier
    milestone-era guards that pinned transient wording or asserted a
    not-yet-written file was absent."""
    from tensorforge.backends import cpp
    import tensorforge.experimental as experimental

    design = _normalized_doc("docs/native_cnn_design.md")
    matrix = _normalized_doc("docs/native_support_matrix.md")
    roadmap = _normalized_doc("docs/roadmap.md")

    # Every milestone of the ladder is recorded as complete, and nothing
    # Phase-D is still described as planned/upcoming.
    for i in range(13):
        assert f"D{i}" in design, f"design is missing milestone D{i}"
    assert "Phase D" in matrix and "complete" in matrix.lower()

    # The shipped CNN surface agrees everywhere.
    for module in ("NativeFlatten", "NativeConv2d", "NativeMaxPool2d"):
        assert module in cpp.NATIVE_MODULES, module
        assert module in experimental.__all__, module
        assert module not in cpp.UNSUPPORTED, module
        assert module in design and module in matrix, module
        # A module is a module: never advertised as an op or a raw kernel.
        assert module not in cpp.AUTOGRAD_OPS and module not in cpp.RAW_KERNELS
    for op in ("conv2d", "maxpool2d"):
        assert op in cpp.AUTOGRAD_OPS, op
        assert op not in cpp.UNSUPPORTED, op
    for core_op in ("conv2d_forward", "conv2d_input_backward",
                    "conv2d_weight_backward", "maxpool2d_forward",
                    "maxpool2d_backward"):
        assert core_op in cpp.TENSOR_CORE_OPS, core_op

    # The Phase-D artifacts the docs point at are really there.
    assert (REPO_ROOT / "examples" / "native_cnn_training.py").is_file()
    assert (REPO_ROOT / "benchmarks" / "benchmark_native_cnn.py").is_file()
    assert (REPO_ROOT / "tests" / "test_native_phase_d.py").is_file()

    # Later phases must not be claimed. These are genuinely absent
    # capabilities, checked against the registry rather than against
    # prose. ("softmax" and "log_softmax" left this list when Phase E
    # milestones E3 and E4 implemented them, "cross_entropy" when E5 and
    # E6 implemented its Core and autograd layers, and
    # "NativeCrossEntropyLoss"/"native_accuracy" when E7 shipped the
    # public surface; the Phase-E boundary is tracked separately below.)
    # ("layernorm" left UNSUPPORTED at Phase F milestone F2, which shipped
    # NativeLayerNorm; "batchnorm" stays until F3/F4.)
    for absent in ("float32", "cuda", "amp", "batchnorm", "dropout"):
        assert absent in cpp.UNSUPPORTED, absent
        assert absent not in cpp.AUTOGRAD_OPS and absent not in cpp.NATIVE_MODULES
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)


def test_native_flatten_is_implemented_as_a_native_module():
    """D1: NativeFlatten is a shipped native module, present in the modern
    native-module inventory and public surface, and is NOT a raw C++
    kernel. Convolution (D7) and pooling (D10) are shipped modules too, so
    neither is a lingering `UNSUPPORTED` entry."""
    from tensorforge.backends import cpp
    import tensorforge.experimental as experimental

    assert "NativeFlatten" in cpp.NATIVE_MODULES
    assert "NativeFlatten" in experimental.__all__
    assert hasattr(experimental, "NativeFlatten")
    # Not a raw C++ kernel and not a lingering "unsupported" entry.
    assert "NativeFlatten" not in cpp.RAW_KERNELS
    assert "flatten" not in cpp.UNSUPPORTED
    # The Conv2d module is implemented as of D7 (over the D6 operation) and
    # the pooling module as of D10 (over the D8/D9 operation); neither is a
    # lingering unsupported entry.
    assert "NativeConv2d" not in cpp.UNSUPPORTED
    assert "NativeMaxPool2d" not in cpp.UNSUPPORTED
    assert "NativeMaxPool2d" in cpp.NATIVE_MODULES


def test_docs_do_not_reassert_a_stale_phase_d_status():
    """The project-facing docs must never regress to claiming the native
    CNN stack is unstarted, empty of layers, unavailable, or still awaiting
    its training proof. Phrase-level (not paragraph-verbatim), so accurate
    rewording survives."""
    docs = (
        "README.md",
        "docs/native_cnn_design.md",
        "docs/native_support_matrix.md",
        "docs/roadmap.md",
        "docs/backend_experiments.md",
        "docs/project_summary.md",
        "docs/architecture.md",
    )
    stale = (
        "no CNN layers",
        "Nothing in Phase D is implemented",
        "native CNN stack, which has not started",
        "native CNN phase has not started",
        "native CNN stack has not started",
        "Native Conv2d is unavailable",
        "native MaxPool2d and the end-to-end",
        "CNN training + checkpoint-resume proof are still upcoming",
        "CNN training + checkpoint-resume proof is still upcoming",
    )
    for name in docs:
        low = _normalized_doc(name).lower()
        for phrase in stale:
            assert phrase.lower() not in low, (
                f"{name} reasserts the stale Phase-D claim {phrase!r}"
            )


def test_docs_present_the_shipped_native_cnn_stack():
    """Every shipped native CNN layer and the D11 proof are positively
    presented in the README, roadmap, and support matrix."""
    for name in ("README.md", "docs/native_support_matrix.md"):
        text = _normalized_doc(name)
        for shipped in ("NativeConv2d", "NativeFlatten", "NativeMaxPool2d",
                        "native_cnn_training.py"):
            assert shipped in text, f"{name} no longer presents {shipped!r}"
    roadmap = _normalized_doc("docs/roadmap.md")
    assert "NativeFlatten" in roadmap
    assert "convolution" in roadmap.lower() and "pooling" in roadmap.lower()
    # The support matrix marks Phase D complete and still names what is not.
    matrix = _normalized_doc("docs/native_support_matrix.md")
    assert "Phase D" in matrix and "complete" in matrix.lower()
    assert "## Unsupported or future" in matrix


# --- Phase E (native classification) — E0 contract guardrails -------------
#
# E0 is a design-and-reconciliation milestone: it adds no numerical
# behavior. These guards therefore check two things — that the contract is
# written and internally coherent, and that nothing it describes has been
# accidentally advertised as implemented.

PHASE_E_DESIGN = "docs/native_classification_design.md"


def _design_section(token, relative_path=PHASE_E_DESIGN):
    """The whitespace-normalized body of the first ``##``-level section
    whose heading contains ``token``.

    Anchoring on headings (not paragraphs) keeps these checks semantic:
    the surrounding prose can be rewritten or rewrapped freely, but the
    load-bearing statement has to stay inside its own section."""
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    for chunk in re.split(r"\n#{2,4} ", text):
        heading = chunk.split("\n", 1)[0]
        if token in heading:
            return re.sub(r"\s+", " ", chunk)
    raise AssertionError(f"{relative_path} has no section naming {token!r}")


def test_phase_e_design_exists_and_is_linked():
    path = REPO_ROOT / PHASE_E_DESIGN
    assert path.is_file(), f"missing {PHASE_E_DESIGN}"
    assert path.read_text(encoding="utf-8").strip(), f"{PHASE_E_DESIGN} is empty"
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert PHASE_E_DESIGN in readme, f"README does not link to {PHASE_E_DESIGN}"


def test_phase_e_design_locks_the_contract():
    """The classification contract must name its phase, its whole public
    surface, and the decisions later milestones inherit."""
    text = _normalized_doc(PHASE_E_DESIGN)
    assert "Phase E" in text
    assert "Native Classification and Stable Math" in text
    for token in ("exp", "log", "softmax", "log_softmax", "cross_entropy",
                  "NativeCrossEntropyLoss", "native_accuracy"):
        assert token in text, f"{PHASE_E_DESIGN} does not lock {token!r}"
    # Targets are int64 host data (the native runtime has no integer dtype).
    assert "int64" in text, "the design does not state the int64 target contract"
    # The fused loss's private saved state and its graph lifetime.
    assert "saved probabilities" in text
    assert "retain_graph" in text
    # The checkpoint schema is explicitly preserved.
    assert re.search(r"checkpoint[^.]{0,80}format version 1", text), (
        "the design does not preserve native checkpoint format version 1"
    )


def test_phase_e_milestone_ladder_is_ordered():
    """E0 through E10 each have their own contract section, and those
    sections appear in increasing order. Anchored on the milestone
    headings, so status notes elsewhere in the document (which may name a
    milestone) cannot perturb the check."""
    text = _normalized_doc(PHASE_E_DESIGN)
    positions = []
    for i in range(11):
        marker = f"### E{i} —"
        assert marker in text, f"{PHASE_E_DESIGN} is missing milestone E{i}"
        positions.append(text.index(marker))
    assert positions == sorted(positions), (
        "the E0-E10 milestone ladder is out of order"
    )


def test_phase_e_design_distinguishes_the_backward_read_contracts():
    """The four distinct backward/versioning archetypes must each be
    pinned in their own operation section — this is the subtlety the
    whole phase turns on."""
    exp = _design_section("NativeTensor.exp()")
    assert "saved forward output" in exp
    assert re.search(r"no (parameter )?version snapshot", exp), (
        "exp must record no version snapshot (saved-output backward)"
    )

    log = _design_section("NativeTensor.log()")
    assert "rereads the live input" in log
    assert "version-checked" in log and "stale-graph" in log, (
        "log must be version-checked (live-input backward)"
    )

    for token in ("NativeTensor.softmax(", "NativeTensor.log_softmax("):
        section = _design_section(token)
        assert "saved output" in section, f"{token} backward must read the saved output"
        assert "no version snapshot" in section, f"{token} must record no version"

    cross_entropy = _design_section("NativeTensor.cross_entropy(")
    assert "saved probabilities" in cross_entropy
    assert "no logits version snapshot" in cross_entropy


def test_phase_e_implemented_surface_matches_the_milestones_reached():
    """Phase E ships one milestone at a time, and the registries are the
    honest record. E1-E4 implemented `exp`, `log`, `softmax`, and
    `log_softmax`; E5 implemented cross-entropy's **Core layer**; E6 the
    differentiable operation over it; E7 the public loss module and the
    reporting metric. E8 is an integration proof and deliberately adds
    **no** inventory entry, so the registries below must not grow."""
    from tensorforge.backends import cpp

    # E1/E2/E3/E4 — implemented, in the two inventories they belong to,
    # no others.
    for shipped in ("exp", "log", "softmax", "log_softmax"):
        assert shipped in cpp.TENSOR_CORE_OPS, shipped
        assert shipped in cpp.AUTOGRAD_OPS, shipped
        assert shipped not in cpp.UNSUPPORTED, shipped
        assert shipped not in cpp.NATIVE_MODULES, shipped
        assert shipped not in cpp.NATIVE_LOSSES, shipped
        # No raw NumPy-buffer stable-math kernel exists.
        assert shipped not in cpp.RAW_KERNELS, shipped

    # E5 — the layer-qualified Core wrappers, and nothing above them.
    for core_op in ("cross_entropy_forward", "cross_entropy_backward"):
        assert core_op in cpp.TENSOR_CORE_OPS, core_op
        assert core_op not in cpp.AUTOGRAD_OPS, core_op
        assert core_op not in cpp.UNSUPPORTED, core_op
        assert core_op not in cpp.RAW_KERNELS, core_op
        assert core_op not in cpp.NATIVE_MODULES, core_op
        assert core_op not in cpp.NATIVE_LOSSES, core_op
        assert hasattr(cpp.NativeTensorCore, core_op), core_op

    # E6 — the differentiable operation, under the bare name and at the
    # autograd layer only. It is deliberately NOT aliased into the Core
    # inventory, and no NativeTensorCore.cross_entropy exists.
    assert "cross_entropy" in cpp.AUTOGRAD_OPS
    assert "cross_entropy" not in cpp.TENSOR_CORE_OPS
    assert "cross_entropy" not in cpp.UNSUPPORTED
    assert "cross_entropy" not in cpp.RAW_KERNELS
    assert "cross_entropy" not in cpp.NATIVE_MODULES
    assert "cross_entropy" not in cpp.NATIVE_LOSSES
    assert not hasattr(cpp.NativeTensorCore, "cross_entropy")
    from tensorforge.experimental import NativeTensor
    assert hasattr(NativeTensor, "cross_entropy")
    assert callable(NativeTensor.cross_entropy)
    # E6 added no C++ or ABI surface: no new checked export appeared.
    for absent in ("tf_core_cross_entropy", "tf_core_nll_loss",
                   "tf_core_accuracy"):
        assert absent not in cpp._CHECKED_KERNELS, absent

    # E7 — the public surface, each name in exactly one layer-appropriate
    # inventory and none in an operation inventory.
    import tensorforge.experimental as experimental
    assert "NativeCrossEntropyLoss" in cpp.NATIVE_LOSSES
    assert "native_accuracy" in cpp.NATIVE_METRICS
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    for name in ("NativeCrossEntropyLoss", "native_accuracy"):
        assert name not in cpp.UNSUPPORTED, f"{name} is still unsupported"
        assert name not in cpp.TENSOR_CORE_OPS, name
        assert name not in cpp.AUTOGRAD_OPS, name
        assert name not in cpp.RAW_KERNELS, name
        assert hasattr(experimental, name), name
        assert name in experimental.__all__, name
    assert "NativeCrossEntropyLoss" not in cpp.NATIVE_METRICS
    assert "native_accuracy" not in cpp.NATIVE_LOSSES
    assert "native_accuracy" not in cpp.NATIVE_MODULES
    # Neither probability transform became a module (E0 §1 excludes both).
    for module in ("NativeSoftmax", "NativeLogSoftmax"):
        assert module not in cpp.NATIVE_MODULES, module
        assert not hasattr(experimental, module), module
    # E3 created the classification source unit locked by E0 §9.1, and E4
    # and E5 extended it; every kernel/export is defined there rather
    # than in the elementwise unit. Checked by symbol definition, not by
    # banning the word, so cross-referencing comments remain possible.
    classification = (REPO_ROOT / "cpp" / "src" / "classification.cpp")
    assert classification.is_file()
    classification_text = classification.read_text(encoding="utf-8")
    for export in ("tf_core_softmax_forward", "tf_core_log_softmax_forward",
                   "tf_core_cross_entropy_forward",
                   "tf_core_cross_entropy_backward"):
        assert export in classification_text, export
        assert export in cpp._CHECKED_KERNELS, export
    # Neither fused transform has a backward ABI symbol: those gradients
    # are composed from existing Core operations.
    for absent in ("tf_core_softmax_backward",
                   "tf_core_log_softmax_backward"):
        assert absent not in classification_text, absent
        assert absent not in cpp._CHECKED_KERNELS, absent
    elementwise = (REPO_ROOT / "cpp" / "src" / "elementwise.cpp").read_text(
        encoding="utf-8"
    )
    assert "tf_core_softmax_forward(" not in elementwise
    assert "tf_core_log_softmax_forward(" not in elementwise


def test_phase_e_milestone_status_is_reported_honestly():
    """Every Phase-E milestone E0-E10 is marked complete, and the phase
    itself is marked complete on every authoritative surface. Checked
    semantically against the design document's status table and the live
    registry — the milestone-era 'not yet shipped' rows are gone because
    the phase closed, not because the check was relaxed."""
    from tensorforge.backends import cpp

    # The ladder's status table is the one place per-milestone status is
    # declared, so the row checks run inside that section only.
    ladder = _design_section("Milestone ladder")
    for done in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9",
                 "E10"):
        row = re.search(rf"\|\s*{done}\s*\|[^|]*\|([^|]*)\|", ladder)
        assert row is not None, f"the ladder has no status row for {done}"
        assert "complete" in row.group(1).lower(), (
            f"the design does not mark {done} complete"
        )
    # Phase E as a whole is complete, stated positively, and no surface
    # may regress to the milestone-era "in progress" wording.
    design = _status_text(PHASE_E_DESIGN)
    assert "Phase-E status: complete" in design
    assert not re.search(r"Phase-E status: in progress", design)
    assert not re.search(r"Phase E is [^.]{0,30}not[^.]{0,30}complete", design)
    for surface in ("README.md", "docs/native_support_matrix.md",
                    "docs/roadmap.md", "docs/backend_experiments.md",
                    "docs/project_summary.md",
                    "src/tensorforge/experimental/__init__.py"):
        text = _status_text(surface)
        assert not re.search(r"Phase E[^.]{0,40}in progress", text, re.I), (
            f"{surface} still calls Phase E in progress"
        )
    matrix = _status_text("docs/native_support_matrix.md")
    assert re.search(r"Phase E[^.]{0,160}complete", matrix, re.I), (
        "the support matrix no longer marks Phase E complete"
    )
    # The registry agrees: E1-E4's capabilities and E6's cross-entropy are
    # live as differentiable operations, E5's Core wrappers at the Core
    # layer, and E7's module and metric in their own layer inventories —
    # while nothing E8-E10 delivered (a training proof, a benchmark, phase
    # closure) is a capability name at all: those milestones shipped
    # examples, benchmarks, and tests, never inventory entries.
    for shipped in ("exp", "log", "softmax", "log_softmax"):
        assert shipped in cpp.AUTOGRAD_OPS, shipped
        assert shipped not in cpp.UNSUPPORTED, shipped
    assert "cross_entropy_forward" in cpp.TENSOR_CORE_OPS
    assert "cross_entropy_backward" in cpp.TENSOR_CORE_OPS
    assert "cross_entropy" in cpp.AUTOGRAD_OPS
    assert "NativeCrossEntropyLoss" in cpp.NATIVE_LOSSES
    assert "native_accuracy" in cpp.NATIVE_METRICS
    for shipped in ("NativeCrossEntropyLoss", "native_accuracy"):
        assert shipped not in cpp.UNSUPPORTED, shipped


def test_docs_present_the_shipped_stable_math():
    """The status surfaces must present `exp` and `log` as implemented
    and keep documenting each one's load-bearing backward invariant —
    the contrast is the whole point of shipping them as a pair."""
    matrix = _normalized_doc("docs/native_support_matrix.md")
    # Both are presented as differentiable forward operations.
    for shipped in ("exp", "log"):
        assert re.search(rf"`{shipped}`\s*\|\s*Yes\s*\|\s*Yes", matrix), (
            f"the support matrix does not list {shipped} as a "
            f"differentiable operation"
        )
    # exp: saved output, no version. log: live input, version-checked.
    assert re.search(r"saved forward output", matrix)
    assert re.search(r"live input", matrix), (
        "the support matrix no longer documents log's live-input backward"
    )
    exp_section = _design_section("NativeTensor.exp()")
    assert "E1" in exp_section and "implemented" in exp_section.lower()
    assert "no version snapshot" in exp_section
    log_section = _design_section("NativeTensor.log()")
    assert "E2" in log_section and "implemented" in log_section.lower(), (
        "the design does not record log as implemented"
    )
    assert "rereads the live input" in log_section
    assert "version-checked" in log_section and "stale-graph" in log_section
    # And the reciprocal-based derivative, which is why no division exists.
    assert "reciprocal" in log_section


def test_docs_present_the_shipped_softmax():
    """E3's load-bearing decisions must stay documented: the fused
    maximum shift, the contiguous-only ABI with Core-level Policy B, and
    the saved-output backward composed at the Core layer."""
    matrix = _normalized_doc("docs/native_support_matrix.md")
    assert re.search(r"`softmax`\s*\|\s*Yes\s*\|\s*Yes", matrix), (
        "the support matrix does not list softmax as a differentiable "
        "operation"
    )
    section = _design_section("NativeTensor.softmax(")
    assert "E3" in section and "implemented" in section.lower(), (
        "the design does not record softmax as implemented"
    )
    lowered = section.lower()
    # Fused stable forward — not a composition of public ops.
    assert "maximum" in lowered and "shift" in lowered
    assert "fused" in lowered
    # Contiguous-only ABI + Policy-B copy at the Core layer.
    assert "contiguous-only" in lowered
    assert "policy-b" in lowered or "policy b" in lowered
    # Saved-output backward, composed at the Core layer, no version.
    assert "saved output" in lowered
    assert "no parameter version snapshot" in lowered or (
        "no version snapshot" in lowered
    )
    # And the two things E3 deliberately did NOT add.
    design = _status_text(PHASE_E_DESIGN)
    assert re.search(r"no dedicated .{0,30}backward kernel", design, re.I), (
        "the design no longer records that softmax has no backward kernel"
    )
    # E3.1: the Policy-B copy is native storage-to-storage, and the docs
    # must not describe a host NumPy round-trip as an accepted state.
    assert re.search(r"storage.to.storage", design, re.I), (
        "the design no longer records the native Policy-B copy"
    )
    assert re.search(r"no tensor.data NumPy round.trip", design, re.I), (
        "the design no longer states that no tensor-data round-trip remains"
    )
    matrix_normalized = _status_text("docs/native_support_matrix.md")
    assert re.search(r"tensor data never round.trips", matrix_normalized,
                     re.I), (
        "the support matrix no longer documents the native contiguous copy"
    )
    # Softmax stays contiguous-only at the ABI while the Core handles
    # strided inputs — both halves must remain stated.
    assert "contiguous-only" in design.lower()


def test_docs_present_the_shipped_log_softmax():
    """E4's load-bearing decisions must stay documented: the fused
    log-sum-exp forward that is explicitly never `softmax().log()`, the
    contiguous-only ABI with Core-level Policy B, the saved-output
    backward composed from Core ops with no backward kernel, and the
    absence of a module."""
    matrix = _normalized_doc("docs/native_support_matrix.md")
    assert re.search(r"`log_softmax`\s*\|\s*Yes\s*\|\s*Yes", matrix), (
        "the support matrix does not list log_softmax as a differentiable "
        "operation"
    )
    section = _design_section("NativeTensor.log_softmax(")
    assert "E4" in section and "implemented" in section.lower(), (
        "the design does not record log_softmax as implemented"
    )
    lowered = section.lower()
    # Fused stable forward — and explicitly not the composed form.
    assert "fused" in lowered
    assert "log-sum-exp" in lowered
    assert re.search(r"never[^.]{0,40}softmax\(\)\.log\(\)", lowered), (
        "the design no longer rejects softmax().log() as the implementation"
    )
    # Contiguous-only ABI + Policy-B copy at the Core layer.
    assert "contiguous-only" in lowered
    assert "policy-b" in lowered or "policy b" in lowered
    # Saved-output backward, no version snapshot, no backward kernel.
    assert "saved output" in lowered or "saved log probabilities" in lowered
    assert "no version snapshot" in lowered
    assert re.search(r"no dedicated log-softmax backward kernel", section,
                     re.I), (
        "the design no longer records that log_softmax has no backward kernel"
    )
    # The exact backward formula stays written down somewhere in the doc.
    design = _normalized_doc(PHASE_E_DESIGN)
    assert re.search(
        r"upstream\s*[−-]\s*exp\(y\)\s*\*\s*"
        r"sum\(upstream,\s*axis,\s*keepdims=True\)",
        design,
    ), "the design no longer states the log_softmax backward formula"
    # log's live-input/version-checked contrast is preserved alongside it.
    log_section = _design_section("NativeTensor.log()")
    assert "rereads the live input" in log_section
    assert "version-checked" in log_section
    # No module was added at any layer.
    from tensorforge.backends import cpp
    import tensorforge.experimental as experimental

    assert not hasattr(experimental, "NativeLogSoftmax")
    assert "NativeLogSoftmax" not in cpp.NATIVE_MODULES


def test_docs_present_the_shipped_cross_entropy_core():
    """E5's load-bearing decisions must stay documented: rank-2 logits
    with a fixed class axis, strict copied int64 targets (bool and
    floating-point rejected) that caller mutation cannot reach,
    mean/sum-only reduction, the fused stable forward with its private
    saved probabilities, a backward that never rereads the logits, and
    the contiguous-only ABI with Core-level Policy B."""
    from tensorforge.backends import cpp

    section = _design_section("NativeTensor.cross_entropy(")
    lowered = section.lower()
    assert "E5" in section and "implemented" in lowered, (
        "the design does not record the cross-entropy Core layer as shipped"
    )
    # Shape and axis contract.
    assert "rank" in lowered and "(batch_size, num_classes)" in section
    assert "no `axis` argument" in section or "no axis argument" in lowered
    # Fused stable forward — and explicitly not the naive form.
    assert "fused" in lowered
    assert "log-sum-exp" in lowered or "maximum-shift" in lowered
    assert re.search(r"not\*{0,2}\s*`?-log\(p", lowered) or (
        "−log(p[target])" in section) or ("-log(p[target])" in section), (
        "the design no longer rejects -log(probability[target])"
    )
    # Private saved probabilities.
    assert "saved probabilities" in lowered
    assert "private" in lowered
    # Backward reads the saved probabilities and never the logits.
    assert re.search(r"never rereads? the logits", lowered), (
        "the design no longer states that the backward never rereads logits"
    )
    # Strict int64 targets, copied, with bool/float rejected.
    assert "int64" in lowered
    assert "bool" in lowered and "floating-point" in lowered
    assert re.search(r"caller mutation|mutation.{0,40}cannot", lowered), (
        "the design no longer states target-copy mutation immunity"
    )
    # Reduction contract.
    assert '"mean"' in section and '"sum"' in section
    # Contiguous-only ABI + Policy-B copy at the Core layer.
    assert "contiguous-only" in lowered
    assert "policy-b" in lowered or "policy b" in lowered
    # Both exports exist and are guarded.
    for export in ("tf_core_cross_entropy_forward",
                   "tf_core_cross_entropy_backward"):
        assert export in section, export
        assert export in cpp._CHECKED_KERNELS, export
    # No tensor-data NumPy round-trip on this path.
    design = _status_text(PHASE_E_DESIGN)
    assert re.search(r"no tensor-data NumPy round-trip", design, re.I)
    # The support matrix agrees about the Core layer.
    matrix = _status_text("docs/native_support_matrix.md")
    assert re.search(r"E5[^.]{0,200}(Core|core)", matrix)
    assert "cross_entropy_forward" in matrix
    assert "cross_entropy_backward" in matrix
    # The registry is the final authority: the Core wrappers stay
    # layer-qualified and never acquire a bare Core alias.
    assert "cross_entropy_forward" in cpp.TENSOR_CORE_OPS
    assert "cross_entropy" not in cpp.TENSOR_CORE_OPS
    assert not hasattr(cpp.NativeTensorCore, "cross_entropy")


def test_docs_present_the_shipped_cross_entropy_autograd():
    """E6's load-bearing decisions must stay documented too: the public
    scalar-output operation, graph-owned saved probabilities with their
    retain/release rules, closure-owned immutable target metadata, no
    logits reread and therefore no expected version, no-grad immediate
    cleanup, and the fact that E6 added no C++ or ABI surface."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import NativeTensor

    section = _design_section("NativeTensor.cross_entropy(")
    lowered = section.lower()
    assert "E6" in section and "implemented" in lowered, (
        "the design does not record the differentiable operation as shipped"
    )
    # The public signature, exactly as locked.
    assert 'cross_entropy(targets, reduction="mean")' in section
    assert hasattr(NativeTensor, "cross_entropy")
    # Scalar output and mean/sum-only reduction.
    assert "scalar" in lowered
    assert '"mean"' in section and '"sum"' in section
    # Graph-owned saved probabilities, with the lifetime rules stated.
    assert re.search(r"graph-owned", lowered), (
        "the design no longer calls the saved probabilities graph-owned"
    )
    assert "retain_graph" in section
    assert re.search(r"no-grad forward closes it|closed immediately",
                     lowered), (
        "the design no longer states the no-grad immediate cleanup"
    )
    assert re.search(r"released exactly once", lowered), (
        "the design no longer states single release of the saved state"
    )
    assert re.search(r"failed retryable backward", lowered), (
        "the design no longer states retention across a failed backward"
    )
    # Closure-owned immutable target metadata, not a graph resource.
    assert re.search(r"closure", lowered)
    assert "int64" in lowered
    # No logits reread, therefore no expected version.
    assert re.search(r"never rereads? the logits", lowered)
    assert re.search(r"_expected_versions == \(\)|no expected parameter "
                     r"version|no logits version snapshot", lowered), (
        "the design no longer states that no version snapshot is recorded"
    )
    # E6 added no numerical capability.
    assert re.search(r"no numerical capability|added no numerical|no new "
                     r"kernel|changed no c\+\+ file", lowered), (
        "the design no longer states that E6 added no numerical capability"
    )
    # The saved-probability lifetime section is live now, not deferred.
    lifetime = _design_section("Saved-probability lifetime")
    assert re.search(r"live as of E6", lifetime), (
        "the design no longer marks the saved-probability lifetime live"
    )
    assert "graph_resources" in lifetime
    # The support matrix and the registry agree.
    matrix = _status_text("docs/native_support_matrix.md")
    assert re.search(r"\| E6 \| \*\*Implemented\*\* \|",
                     (REPO_ROOT / "docs/native_support_matrix.md").read_text(
                         encoding="utf-8")), (
        "the support matrix does not mark E6 implemented"
    )
    assert re.search(r"E6[^.]{0,200}differentiable", matrix, re.I)
    assert "cross_entropy" in cpp.AUTOGRAD_OPS
    assert "NativeCrossEntropyLoss" in cpp.NATIVE_LOSSES
    assert "native_accuracy" in cpp.NATIVE_METRICS


def test_docs_present_the_shipped_classification_loss_module():
    """E7's loss module must stay documented as what it is: a stateless
    NativeModule that *delegates* to the E6 operation, supports mean and
    sum only, and adds no kernel, ABI, or persistent state."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import NativeCrossEntropyLoss

    section = _design_section("NativeTensor.cross_entropy(")
    lowered = section.lower()
    assert "E7" in section and "NativeCrossEntropyLoss" in section, (
        "the design does not record the loss module as shipped"
    )
    assert 'NativeCrossEntropyLoss(reduction="mean")' in section
    # Delegation, not reimplementation.
    assert re.search(r"delegat", lowered), (
        "the design no longer states that the loss module delegates"
    )
    assert "logits.cross_entropy(targets" in section
    # Stateless, and the reduction is configuration rather than state.
    assert re.search(r"stateless|parameter-free|no parameters", lowered), (
        "the design no longer states that the loss module is stateless"
    )
    assert re.search(r"buffer-free|no buffers", lowered)
    assert re.search(r"state_dict\(\)`? is empty|no state-dictionary",
                     lowered), (
        "the design no longer states that state_dict() is empty"
    )
    assert re.search(r"constructor configuration", lowered), (
        "the design no longer states that the reduction is not model state"
    )
    assert re.search(r"checkpoint format stays version 1|version 1", lowered)
    assert '"mean"' in section and '"sum"' in section
    # The registry and the export agree.
    assert "NativeCrossEntropyLoss" in cpp.NATIVE_LOSSES
    assert "NativeCrossEntropyLoss" not in cpp.AUTOGRAD_OPS
    assert NativeCrossEntropyLoss("sum").state_dict() == {}
    # The support matrix presents it too.
    matrix = _status_text("docs/native_support_matrix.md")
    assert re.search(r"NativeCrossEntropyLoss[^|]{0,400}Supported", matrix) or (
        re.search(r"NativeCrossEntropyLoss", matrix)), (
        "the support matrix does not present the loss module"
    )


def test_docs_present_the_reporting_only_accuracy_metric():
    """E7's metric must stay documented **honestly**: reporting-only, a
    Python float, an explicit `to_numpy()` conversion and a NumPy argmax,
    no graph, and emphatically not native C++ compute."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import NativeTensor, native_accuracy

    section = _design_section("Metric contract")
    lowered = section.lower()
    assert "E7" in section and "implemented" in lowered, (
        "the design does not record the metric as shipped"
    )
    assert "native_accuracy(logits, targets) -> float" in section
    # Reporting-only, and not a kernel or an operation.
    assert re.search(r"reporting", lowered)
    assert re.search(r"not a native kernel|not native compute|no accuracy "
                     r"kernel", lowered), (
        "the design no longer denies that the metric is a native kernel"
    )
    assert re.search(r"not an autograd operation|no autograd node", lowered)
    # The two mechanics that must never be hidden.
    assert "to_numpy()" in section, (
        "the design no longer documents the explicit to_numpy() conversion"
    )
    assert re.search(r"argmax", lowered), (
        "the design no longer documents the NumPy argmax"
    )
    assert re.search(r"numpy", lowered)
    # No graph, no gradients, and a Python float out.
    assert re.search(r"build no graph|builds no graph", lowered)
    assert re.search(r"python `?float", lowered)
    # The shared strict target contract.
    assert re.search(r"same contract as §6|same private|same strict",
                     lowered), (
        "the design no longer ties the metric to the §6 target contract"
    )
    # The inventory placement is stated, and the registry agrees.
    assert "NATIVE_METRICS" in section
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert "native_accuracy" not in cpp.TENSOR_CORE_OPS
    assert "native_accuracy" not in cpp.AUTOGRAD_OPS
    assert "native_accuracy" not in cpp.NATIVE_MODULES
    assert "native_accuracy" not in cpp.NATIVE_LOSSES
    assert cpp.backend_info()["native_metrics"] == cpp.NATIVE_METRICS
    # No native surface was invented for it.
    assert not hasattr(NativeTensor, "native_accuracy")
    assert not hasattr(NativeTensor, "argmax")
    assert not hasattr(cpp.NativeTensorCore, "argmax")
    for absent in ("tf_core_accuracy", "tf_core_argmax"):
        assert absent not in cpp._CHECKED_KERNELS, absent
    assert callable(native_accuracy)
    # The support matrix says the same thing.
    matrix = _status_text("docs/native_support_matrix.md")
    assert re.search(r"native_accuracy[^|]{0,600}(reporting|to_numpy)",
                     matrix), (
        "the support matrix does not present the metric as reporting-only"
    )


def test_docs_present_the_shipped_classification_training_proof():
    """E8's proof must stay documented as what it is: a deterministic
    fixed-task training and **exact** checkpoint-resume integration
    result over the shipped stack, adding no capability — never a
    benchmark, a speed claim, or a generalization claim."""
    from tensorforge.backends import cpp

    # The example the docs point at is really there, and it is the one
    # the design document names.
    example = REPO_ROOT / "examples" / "native_classification_training.py"
    assert example.is_file()
    assert (REPO_ROOT / "tests"
            / "test_native_classification_training.py").is_file()

    section = _design_section("E8 —")
    lowered = section.lower()
    assert "complete" in lowered, "the design does not record E8 as shipped"
    assert "examples/native_classification_training.py" in section
    # The load-bearing facts: a deterministic fixed multi-class dataset,
    # the raw-logit path into the loss module, the reporting-only metric,
    # the stateful optimizer, and the exact resume at a fixed split.
    assert "three" in lowered and "deterministic" in lowered
    assert "raw logits" in lowered
    assert "NativeCrossEntropyLoss" in section
    assert "native_accuracy" in section
    assert re.search(r"reporting", lowered), (
        "the design no longer marks the metric reporting-only in E8"
    )
    assert "NativeAdam" in section
    assert re.search(r"no softmax or log-softmax", lowered), (
        "the design no longer states that no final softmax layer exists"
    )
    assert re.search(r"exactly", lowered) and "resume" in lowered
    assert re.search(r"format\s+\*{0,2}version 1", lowered), (
        "the design no longer records the unchanged checkpoint format"
    )
    # No new capability, and honest framing.
    assert re.search(r"no.{0,60}(kernel|capability)", lowered), (
        "the design no longer states that E8 added no capability"
    )
    assert re.search(r"integration proof", lowered)
    # The README documents the command and the proof.
    readme = _status_text("README.md")
    assert "examples/native_classification_training.py" in readme
    assert re.search(
        r"uv run python examples/native_classification_training.py", readme
    ), "the README does not document the example's command"
    # The support matrix presents it too, and the registries did not grow.
    matrix = _status_text("docs/native_support_matrix.md")
    assert "native_classification_training.py" in matrix
    assert re.search(r"E8 \| Implemented", matrix), (
        "the support matrix does not mark E8 implemented"
    )
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS):
        for banned in ("classifier", "training", "checkpoint_resume"):
            assert not [n for n in inventory if banned in n.lower()], banned


def test_docs_present_the_shipped_classification_benchmark():
    """E9's harness must stay documented as what it is: an honest local
    characterization with correctness gated before timing and **no**
    speed guarantee — never a performance contract or a CI gate."""
    benchmark = REPO_ROOT / "benchmarks" / "benchmark_native_classification.py"
    assert benchmark.is_file()
    assert (REPO_ROOT / "tests"
            / "test_native_classification_benchmark.py").is_file()

    section = _design_section("E9 —")
    lowered = section.lower()
    assert "complete" in lowered, "the design does not record E9 as shipped"
    assert "benchmarks/benchmark_native_classification.py" in section
    # The seven measured operations are named.
    for case in ("exp_forward", "log_forward", "softmax_forward",
                 "log_softmax_forward", "cross_entropy_forward",
                 "cross_entropy_backward", "classification_training_step"):
        assert case in section, case
    # The methodology commitments.
    assert re.search(r"correctness before timing|correctness .{0,30}before",
                     lowered), (
        "the design no longer states that correctness runs before timing"
    )
    assert "median" in lowered and "spread" in lowered
    assert "warm-up" in lowered or "warmup" in lowered
    for label in ("stable_tensorforge", "numpy", "native_only"):
        assert label in section, label
    assert "--smoke" in section and "--json" in section
    # And the honesty boundary, in the design and on the status surfaces.
    assert re.search(r"no.{0,40}speed", lowered), (
        "the design no longer states that no speed is asserted"
    )
    assert re.search(r"no ci timing threshold|no timing threshold", lowered)
    readme = _status_text("README.md")
    assert "benchmarks/benchmark_native_classification.py" in readme
    assert re.search(
        r"uv run python benchmarks/benchmark_native_classification.py --smoke",
        readme,
    ), "the README does not document the benchmark's smoke command"
    assert "--json" in readme
    for surface in ("README.md", "docs/native_support_matrix.md",
                    "docs/roadmap.md"):
        text = _status_text(surface)
        assert re.search(r"(no speed|not a (performance )?(contract|guarantee)"
                         r"|no timing threshold|no CI performance gate"
                         r"|characterization)", text, re.I), surface
    matrix = _status_text("docs/native_support_matrix.md")
    assert re.search(r"E9 \| Implemented", matrix), (
        "the support matrix does not mark E9 implemented"
    )


def test_phase_e_closure_artifacts_exist_and_are_documented():
    """E10 closed the phase. This replaces the milestone-era absence
    checks (which asserted that the integration test and the benchmark
    did *not* exist yet) with the durable positive form: the closure
    artifacts are present, referenced, and described honestly."""
    assert (REPO_ROOT / "tests" / "test_native_phase_e.py").is_file(), (
        "the Phase-E cross-cutting integration test is missing"
    )
    section = _design_section("E10 —")
    lowered = section.lower()
    assert "complete" in lowered, "the design does not record E10 as shipped"
    assert "tests/test_native_phase_e.py" in section
    # The closure milestone's own deliverables, stated in its section.
    for token in ("asan", "ubsan", "leaksanitizer", "release", "debug"):
        assert token in lowered, token
    assert re.search(r"no (new )?numerical (capability|behavior)", lowered), (
        "the design no longer states that E10 added no numerical capability"
    )
    # The phase-completion statement exists and names what shipped.
    completion = _status_text(PHASE_E_DESIGN)
    assert "Phase-E status: complete" in completion
    for token in ("exp", "log", "softmax", "log-softmax",
                  "NativeCrossEntropyLoss", "native_accuracy",
                  "checkpoint", "float64"):
        assert token in completion, token


def test_no_future_phase_is_claimed_by_phase_e_closure():
    """Closing Phase E must not imply anything about what comes next: no
    surface may present CUDA, AMP, other dtypes or devices, native
    integer targets, normalization, RNG/dropout, or data loaders as
    shipped, and the registry stays the authority."""
    from tensorforge.backends import cpp

    # ("layernorm" is no longer future work — F2 shipped NativeLayerNorm;
    # "batchnorm" remains.)
    for future in ("cuda", "amp", "float32", "batchnorm", "dropout"):
        assert future in cpp.UNSUPPORTED, future
        assert future not in cpp.AUTOGRAD_OPS and future not in cpp.TENSOR_CORE_OPS
        assert future not in cpp.NATIVE_MODULES
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.backend_info()["stable_framework_integration"] is False
    # ("LayerNorm" left this list at Phase F milestone F2, which shipped
    # NativeLayerNorm; "BatchNorm" stays until F3/F4.)
    claim = re.compile(
        r"(CUDA|AMP|float32|float16|bfloat16|GPU|BatchNorm|"
        r"dropout|data loader|dataloader|native RNG|integer tensor)"
        r"[^.]{0,60}(is|are|now)\s+(supported|implemented|shipped|available)",
        re.I,
    )
    for surface in AUTHORITATIVE_STATUS_SURFACES + (PHASE_E_DESIGN,):
        text = _status_text(surface)
        match = claim.search(text)
        assert match is None, (surface, match.group(0) if match else "")


def test_no_committed_benchmark_timing_promise():
    """Benchmark numbers are machine-specific characterizations, so no
    status surface may publish a classification timing as a project
    promise, and no benchmark result artifact may be tracked."""
    for surface in ("README.md", "docs/native_support_matrix.md",
                    "docs/roadmap.md", PHASE_E_DESIGN):
        text = _status_text(surface)
        window = re.findall(
            r"[^.]{0,120}benchmark_native_classification[^.]{0,200}", text
        )
        for chunk in window:
            assert not re.search(r"\d+\s*(us|ms|µs|microseconds|milliseconds)"
                                 r"\b", chunk, re.I), (surface, chunk[:120])
            assert not re.search(r"\bx faster|\bspeedup\b", chunk, re.I), surface
    assert not list((REPO_ROOT / "benchmarks").glob("*.json"))
    assert not list(REPO_ROOT.glob("benchmark*.json"))


def test_phase_e_keeps_the_checkpoint_format_and_the_shipped_surface():
    """Phase E adds no persistent state: the native checkpoint format
    version stays 1, and the Phase-D surface is untouched."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint
    import tensorforge.experimental as experimental

    assert native_checkpoint._FORMAT_VERSION == 1
    for module in ("NativeFlatten", "NativeConv2d", "NativeMaxPool2d"):
        assert module in cpp.NATIVE_MODULES and module in experimental.__all__
    for op in ("conv2d", "maxpool2d"):
        assert op in cpp.AUTOGRAD_OPS
    # Stable/native separation and the remaining future work stay explicit.
    assert cpp.backend_info()["stable_framework_integration"] is False
    # ("layernorm" left UNSUPPORTED at F2; "batchnorm" remains.)
    for future in ("cuda", "float32", "amp", "batchnorm", "dropout"):
        assert future in cpp.UNSUPPORTED, future


# Where the *current* capability status is stated. Per-milestone historical
# records (docs/native_cnn_design.md, the milestone log in
# docs/backend_experiments.md) deliberately preserve superseded wording and
# are not scanned.
AUTHORITATIVE_STATUS_SURFACES = (
    "README.md",
    "docs/native_support_matrix.md",
    "docs/architecture.md",
    "docs/project_summary.md",
    "docs/roadmap.md",
    "src/tensorforge/backends/cpp.py",
    "src/tensorforge/experimental/__init__.py",
)


# How a shipped Phase-D layer gets named in prose: by class name, or as
# "the convolution/pooling module". Emphasis markers are stripped before
# matching, so "the pooling *module*" and "`NativeMaxPool2d`" both count.
_MODULE_SUBJECTS = {
    "NativeConv2d": r"(NativeConv2d|convolution module|Conv2d module)",
    "NativeMaxPool2d": (
        r"(NativeMaxPool2d|pooling module|MaxPool2d module|max-pooling module)"
    ),
    "NativeLayerNorm": (
        r"(NativeLayerNorm|LayerNorm module|layer-normalization module)"
    ),
}
_ABSENT_CLAIM = (
    r"(not yet implemented|not implemented|still unsupported|is unsupported"
    r"|does not exist|has not shipped)"
)


def _status_text(relative_path):
    """Whitespace-normalized text with markdown emphasis stripped, so a
    status claim is matched by meaning rather than by formatting."""
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text))


def test_authoritative_surfaces_do_not_call_shipped_modules_unimplemented():
    """Semantic, not token-level: if a module is in the live registry and
    the public exports, no authoritative current-status text may claim it
    is missing. Both word orders are checked within a narrow same-sentence
    window, so unrelated prose cannot trip the guard."""
    from tensorforge.backends import cpp
    import tensorforge.experimental as experimental

    for module, subject in _MODULE_SUBJECTS.items():
        # The premise: these really are implemented.
        assert module in cpp.NATIVE_MODULES and module in experimental.__all__
        forward = re.compile(subject + r"[^.]{0,120}?" + _ABSENT_CLAIM, re.I)
        backward = re.compile(_ABSENT_CLAIM + r"[^.]{0,120}?" + subject, re.I)
        for name in AUTHORITATIVE_STATUS_SURFACES:
            text = _status_text(name)
            for pattern in (forward, backward):
                match = pattern.search(text)
                assert match is None, (
                    f"{name} claims {module} is unimplemented "
                    f"({match.group(0)!r}), but it is in NATIVE_MODULES"
                )


# --- Phase F (native normalization) — F0 contract guardrails -------------
#
# F0 is a design-and-reconciliation milestone: it adds no numerical
# behavior. These guards therefore establish two things — that Phase E is
# presented honestly as complete everywhere (the drift F0 was written to
# repair), and that Phase F is presented honestly as *designed only*,
# with nothing it describes accidentally advertised as implemented.
#
# Every check below derives from the live registry, the live exports, or
# a real file wherever that is practical, rather than from frozen prose.

PHASE_F_DESIGN = "docs/native_normalization_design.md"

# Where a *phase status narrative* is written. This is deliberately not
# AUTHORITATIVE_STATUS_SURFACES: that tuple includes the backend registry
# module, which states capabilities rather than phase status. The registry
# is still covered — by the negative and inventory checks below, which are
# the stronger form for it.
PHASE_STATUS_DOCS = (
    "README.md",
    "docs/native_support_matrix.md",
    "docs/roadmap.md",
    "docs/project_summary.md",
    "docs/architecture.md",
    "docs/backend_experiments.md",
    "src/tensorforge/experimental/__init__.py",
)

# The names Phase F contracts but has NOT implemented, in each of the
# forms a document or a registry might use them.
_NORMALIZATION_MODULES = (
    "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
)
_NORMALIZATION_OP_NAMES = (
    "layer_norm", "batch_norm", "layernorm", "batchnorm",
    "layer_norm_forward", "batch_norm_forward",
)


def _top_level_section(token, relative_path=PHASE_F_DESIGN):
    """The whitespace-normalized body of the first ``##``-level section
    whose heading contains ``token``, **including** its ``###``
    subsections.

    ``_design_section`` splits on every heading level, which truncates a
    section that has subsections (Phase F's §6, §7, and §14 all do). This
    variant slices between top-level ``##`` headings only, so a check can
    be scoped to one numbered section without pinning its internal
    structure."""
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    chunks = re.split(r"\n## ", text)
    for chunk in chunks:
        if token in chunk.split("\n", 1)[0]:
            return re.sub(r"\s+", " ", chunk)
    raise AssertionError(f"{relative_path} has no ## section naming {token!r}")


def test_phase_e_is_positively_marked_complete_on_every_status_doc():
    """Requirement 1 of the F0 audit. Before F0 several documents still
    described Phase D as the latest native phase; each must now state
    Phase E's completion positively, not merely avoid denying it."""
    for surface in PHASE_STATUS_DOCS:
        text = _status_text(surface)
        assert re.search(r"Phase E.{0,300}?\bcomplete\b", text, re.I), (
            f"{surface} does not positively state that Phase E is complete"
        )


def test_native_classification_is_presented_as_shipped_everywhere():
    """Requirement 2. Phase E's public surface really exists, so every
    status document must present it. The premise is checked against the
    live registry and exports first, so this can never assert a doc claim
    the code does not back."""
    from tensorforge.backends import cpp
    import tensorforge.experimental as experimental

    # Premise: these are genuinely shipped.
    assert "cross_entropy" in cpp.AUTOGRAD_OPS
    assert "NativeCrossEntropyLoss" in cpp.NATIVE_LOSSES
    assert "native_accuracy" in cpp.NATIVE_METRICS
    for name in ("NativeCrossEntropyLoss", "native_accuracy"):
        assert name in experimental.__all__

    for surface in PHASE_STATUS_DOCS:
        text = _status_text(surface)
        assert "NativeCrossEntropyLoss" in text, (
            f"{surface} does not present the shipped native classification "
            f"loss"
        )
        assert re.search(r"cross.entropy", text, re.I), surface


_CLASSIFICATION_SUBJECT = (
    r"(native classification|classification stack|NativeCrossEntropyLoss"
    r"|native_accuracy)"
)
_ABSENT_OR_PENDING = (
    r"(not yet implemented|not implemented|unimplemented|has not begun"
    r"|has not started|\bnot started\b|is unsupported|still unsupported"
    r"|does not exist|has not shipped|is upcoming|are upcoming"
    r"|still upcoming|in progress|under way|underway)"
)
# The third stale form the audit found: a bare negation ("no native
# classification stack, normalization, dropout, or RNG") in a
# limitations list, which neither word order above catches.
_BARE_ABSENCE = r"\bno\s+(?:\w+\s+){0,2}?" + _CLASSIFICATION_SUBJECT


def test_no_surface_calls_native_classification_absent_or_pending():
    """Requirement 3. Both word orders, inside a narrow same-sentence
    window so unrelated prose cannot trip the guard. This is the check
    that would have caught the pre-F0 backend_experiments claim that
    Phase E's 'implementation has not begun'."""
    forward = re.compile(
        _CLASSIFICATION_SUBJECT + r"[^.]{0,90}?" + _ABSENT_OR_PENDING, re.I
    )
    backward = re.compile(
        _ABSENT_OR_PENDING + r"[^.]{0,90}?" + _CLASSIFICATION_SUBJECT, re.I
    )
    # "Phase E ... has not begun" phrased through the phase name, too.
    phase_forms = re.compile(r"Phase E[^.]{0,90}?" + _ABSENT_OR_PENDING, re.I)
    bare = re.compile(_BARE_ABSENCE, re.I)
    for surface in AUTHORITATIVE_STATUS_SURFACES + PHASE_STATUS_DOCS:
        text = _status_text(surface)
        for pattern in (forward, backward, phase_forms, bare):
            match = pattern.search(text)
            assert match is None, (
                f"{surface} presents native classification as absent or "
                f"pending: {match.group(0)!r}" if match else ""
            )


def test_phase_f_design_exists_and_is_linked():
    """Requirement 4, first half."""
    path = REPO_ROOT / PHASE_F_DESIGN
    assert path.is_file(), f"missing {PHASE_F_DESIGN}"
    assert path.read_text(encoding="utf-8").strip(), f"{PHASE_F_DESIGN} is empty"
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert PHASE_F_DESIGN in readme, f"README does not link to {PHASE_F_DESIGN}"


def test_phase_f_milestone_ladder_is_complete_and_ordered():
    """Requirement 4, second half: F0 through F9 each have their own
    contract section, and those sections appear in increasing order."""
    text = _normalized_doc(PHASE_F_DESIGN)
    positions = []
    for i in range(10):
        marker = f"### F{i} —"
        assert marker in text, f"{PHASE_F_DESIGN} is missing milestone F{i}"
        positions.append(text.index(marker))
    assert positions == sorted(positions), (
        "the F0-F9 milestone ladder is out of order"
    )


def test_phase_f_design_locks_its_load_bearing_decisions():
    """The contract later milestones inherit must actually be written
    down: the phase name, the three planned modules, the composition rule
    that keeps Phase F out of C++, the normalization mathematics, the
    mutable-buffer graph-safety rule, the atomic two-buffer transaction,
    and the unchanged checkpoint format."""
    text = _normalized_doc(PHASE_F_DESIGN)
    assert "Phase F" in text
    assert "Native Normalization and Stateful Buffers" in text
    for module in _NORMALIZATION_MODULES:
        assert module in text, f"{PHASE_F_DESIGN} does not lock {module!r}"
    # The naming decisions (LayerNorm weight/bias vs BatchNorm gamma/beta).
    for token in ("weight", "bias", "gamma", "beta",
                  "running_mean", "running_var"):
        assert token in text, f"{PHASE_F_DESIGN} does not lock {token!r}"
    # Composition over existing ops, and therefore no new native surface.
    for op in ("mean", "subtract", "multiply", "sqrt", "reciprocal",
               "reshape", "contiguous_copy"):
        assert op in text, f"the design does not list the composed op {op!r}"
    assert re.search(r"no.{0,80}(kernel|C ABI export)", text, re.I), (
        "the design no longer states that Phase F adds no kernel or export"
    )
    # The normalization mathematics that a later milestone could get wrong.
    assert "population variance" in text.lower()
    assert re.search(r"sqrt\(var \+ eps\)", text), (
        "the design no longer pins epsilon inside the square root"
    )
    assert re.search(r"differentiat\w+ through the batch (mean|statistics)",
                     text, re.I) or "through the batch mean and variance" in text, (
        "the design no longer requires differentiating the batch statistics"
    )
    # Shapes.
    assert "(N, C)" in text and "(N, C, H, W)" in text
    assert "(1, C, 1, 1)" in text
    # The checkpoint format is explicitly preserved.
    assert re.search(r"version 1", text), (
        "the design does not preserve native checkpoint format version 1"
    )


def test_phase_f_design_locks_the_mutable_buffer_safety_rule():
    """The phase's central insight, pinned in its own section: a live
    mutable running buffer is never a rereadable graph operand, eval mode
    snapshots instead, and that is *why* buffers need no version."""
    section = _top_level_section("Mutable-buffer graph safety")
    lowered = section.lower()
    assert "snapshot" in lowered
    assert re.search(r"never be captured|never captured", lowered), (
        "the design no longer forbids capturing the live buffer"
    )
    assert "graph-free" in lowered
    # The two facts that make the hazard real, so the rule is justified
    # rather than asserted.
    assert re.search(r"no value version|unversioned", lowered)
    assert re.search(r"reread", lowered)
    assert "multiply" in lowered
    assert "nativeparameter" in lowered
    # And the explicit decision not to add buffer versions.
    assert re.search(r"not.{0,40}add(ing)?.{0,40}version|versions are \*?\*?not",
                     lowered) or "why buffer versions" in lowered, (
        "the design no longer explains why buffer versions are not required"
    )


def test_phase_f_design_locks_the_atomic_running_stat_transaction():
    """The second load-bearing rule: the two running buffers advance as
    one transaction, with staging, identity-preserving commit, complete
    rollback, and exactly-once closes — and no public in-place mutation
    API on ordinary NativeTensor."""
    section = _top_level_section("Running-statistics transaction")
    lowered = section.lower()
    for token in ("atomic", "stag", "commit", "rollback", "identity"):
        assert token in lowered, f"the transaction section omits {token!r}"
    assert re.search(r"exactly once", lowered), (
        "the design no longer states exactly-once closing"
    )
    assert re.search(r"parameter versions do not move|no parameter version",
                     lowered), (
        "the design no longer states that parameter versions stay put"
    )
    assert re.search(r"no general public in-place mutation|not add a general",
                     lowered), (
        "the design no longer refuses a public in-place mutation API"
    )
    # F1 is named as the milestone that extracts the existing behavior.
    assert "F1" in section and "load_state_dict" in section


def test_phase_f_is_described_as_in_progress_not_complete():
    """Requirement 5, in its F2 form. Phase F is now *in progress* — F2
    shipped a real module — but it is not finished. The design document
    states this about itself, every status document names Phase F, and
    none may claim the *phase* is complete (an honest per-milestone claim
    like "F2 is complete" must keep being possible)."""
    design = _status_text(PHASE_F_DESIGN)
    assert re.search(r"Phase-F status: in progress", design), (
        "the design document no longer states its in-progress status"
    )
    # F0/F1's no-numerical-behavior honesty is still recorded.
    assert re.search(r"no numerical behavior", design, re.I)
    for surface in PHASE_STATUS_DOCS:
        text = _status_text(surface)
        assert "Phase F" in text, f"{surface} does not name Phase F at all"
        # It must not claim the *phase* is finished. The `[^.F]` window is
        # deliberate: it forbids an intervening milestone label (F0 … F9),
        # so "Phase F — in progress: F2 is complete" reads as the milestone
        # claim it is, while "Phase F is now complete" is still caught.
        claim = re.search(
            r"Phase F\b[^.F]{0,40}?\b(is|was|are|now)\s+"
            r"(complete|completed|shipped|implemented)\b",
            text, re.I,
        )
        assert claim is None, (
            f"{surface} claims Phase F is complete: "
            f"{claim.group(0)!r}" if claim else ""
        )


def test_normalization_is_module_only_with_no_new_native_operation():
    """Checked against the live registry and exports rather than prose.
    Milestone F2 shipped ``NativeLayerNorm`` and milestone F3 shipped
    ``NativeBatchNorm1d``, each as a *module composed from existing
    operations*: the NCHW BatchNorm is still absent, and — for every
    normalization shape alike — there is no normalization operation,
    kernel, Core method, ``NativeTensor`` method, or C ABI symbol
    anywhere. F2 and F3 added modules, not numerical primitives."""
    from tensorforge.backends import cpp
    import tensorforge.experimental as experimental

    # LayerNorm shipped as a module (F2), the (N, C) BatchNorm as another
    # (F3); the NCHW shape has not (F4), which is exactly why the
    # unqualified "batchnorm" capability is still unsupported.
    assert "layernorm" not in cpp.UNSUPPORTED
    assert "batchnorm" in cpp.UNSUPPORTED
    for module in ("NativeLayerNorm", "NativeBatchNorm1d"):
        assert module in cpp.NATIVE_MODULES, module
        assert module in experimental.__all__, module
    for module in ("NativeBatchNorm2d",):
        assert module not in cpp.NATIVE_MODULES, module
        assert module not in experimental.__all__, module
        assert not hasattr(experimental, module), module

    # No normalization module is a loss or a metric.
    for module in _NORMALIZATION_MODULES:
        assert module not in cpp.NATIVE_LOSSES, module
        assert module not in cpp.NATIVE_METRICS, module

    # No normalization *operation* at any layer, and no raw kernel — this
    # is the load-bearing F2/F3 fact: both are compositions, not
    # primitives.
    for name in _NORMALIZATION_OP_NAMES:
        assert name not in cpp.TENSOR_CORE_OPS, name
        assert name not in cpp.AUTOGRAD_OPS, name
        assert name not in cpp.RAW_KERNELS, name
        assert not hasattr(cpp.NativeTensorCore, name), name

    from tensorforge.experimental import NativeTensor
    for name in ("layer_norm", "batch_norm", "layernorm", "batchnorm",
                 "normalize"):
        assert not hasattr(NativeTensor, name), name
    # No functional normalization helper was exported either.
    for name in ("layer_norm", "batch_norm"):
        assert not hasattr(experimental, name), name
        assert name not in experimental.__all__, name

    # No normalization C ABI symbol is declared or guarded, and none is
    # defined in any C++ source unit.
    for symbol in ("tf_core_layer_norm", "tf_core_batch_norm",
                   "tf_core_layer_norm_forward", "tf_core_batch_norm_forward",
                   "tf_core_layer_norm_backward",
                   "tf_core_batch_norm_backward"):
        assert symbol not in cpp._CHECKED_KERNELS, symbol
    for source in (REPO_ROOT / "cpp" / "src").glob("*.cpp"):
        text = source.read_text(encoding="utf-8")
        for symbol in ("tf_core_layer_norm", "tf_core_batch_norm"):
            assert symbol not in text, f"{source.name} defines {symbol!r}"
    # And no normalization source unit was created.
    assert not (REPO_ROOT / "cpp" / "src" / "normalization.cpp").exists()


def test_phase_f_export_surface_adds_only_the_shipped_normalization_modules():
    """The public experimental surface is exactly what Phase E left plus
    the Phase-F additions so far — ``NativeLayerNorm`` (F2) and
    ``NativeBatchNorm1d`` (F3) — in both directions: nothing else added,
    nothing lost. ``NativeBatchNorm2d`` (F4) has not shipped and must not
    appear here."""
    import tensorforge
    import tensorforge.experimental as experimental

    assert set(experimental.__all__) == {
        "NativeTensor", "NativeParameter", "NativeParameterRegistry",
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeMSELoss", "NativeSGD", "NativeAdam",
        "save_native_checkpoint", "load_native_checkpoint",
        "NativeCrossEntropyLoss", "native_accuracy",
        "NativeLayerNorm",       # Phase F, milestone F2
        "NativeBatchNorm1d",     # Phase F, milestone F3
    }
    for absent in ("NativeBatchNorm2d",):
        assert absent not in experimental.__all__, absent
    # No duplicates, and nothing leaks into the stable namespace.
    assert len(experimental.__all__) == len(set(experimental.__all__))
    for name in experimental.__all__:
        assert hasattr(experimental, name)
        assert not hasattr(tensorforge, name), name


def test_phase_f_changes_only_the_normalization_module_inventory():
    """Phase F's only *operation*-surface changes so far are two modules:
    ``NativeLayerNorm`` (F2) and ``NativeBatchNorm1d`` (F3) joined
    ``NATIVE_MODULES``, and ``"layernorm"`` left ``UNSUPPORTED`` while
    ``"batchnorm"`` stayed (F4 has not shipped the NCHW shape). Everything
    else is exactly the Phase-E surface — no new operation, kernel, loss,
    metric, or dtype — and ``STATE_SUPPORT``'s F1 reconciliation has its
    own guardrail below. Because both are *modules composed from existing
    operations*, no operation inventory grew."""
    from tensorforge.backends import cpp

    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d",
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.UNSUPPORTED == (
        "batchnorm", "dropout", "float32", "cuda", "amp",
    )
    # The Phase-E operation surface is intact and nothing normalization
    # shaped joined it — LayerNorm added no operation.
    for op in ("exp", "log", "softmax", "log_softmax", "cross_entropy"):
        assert op in cpp.AUTOGRAD_OPS, op
    for absent in ("layer_norm", "batch_norm", "layernorm", "batchnorm"):
        assert absent not in cpp.AUTOGRAD_OPS, absent
        assert absent not in cpp.TENSOR_CORE_OPS, absent
        assert absent not in cpp.RAW_KERNELS, absent
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)


def test_state_support_reports_persistent_buffers():
    """F1's capability reconciliation, pinned in both directions: the
    tuple is exact, the new name really does map to a live API, and the
    correction did not smuggle in a normalization claim."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import NativeModule

    assert cpp.STATE_SUPPORT == (
        "persistent_buffers",
        "state_dict", "load_state_dict",
        "save_native_checkpoint", "load_native_checkpoint",
    )
    assert cpp.backend_info()["state_support"] == cpp.STATE_SUPPORT
    # The advertised capability is real: this is the API behind the name.
    for attribute in ("register_buffer", "buffers", "named_buffers",
                      "state_dict", "load_state_dict"):
        assert callable(getattr(NativeModule, attribute)), attribute
    # A state capability is not an operation, a module, or a metric.
    for inventory in (cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS, cpp.RAW_KERNELS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS, cpp.UNSUPPORTED):
        assert "persistent_buffers" not in inventory


def test_f1_added_no_normalization_capability():
    """F1 is a state-management and reporting milestone. Its own design
    section must say so, and the registry must still show no
    normalization anything."""
    from tensorforge.backends import cpp

    f1 = _design_section("F1 —", relative_path=PHASE_F_DESIGN)
    lowered = f1.lower()
    assert "complete" in lowered, "the design does not record F1 as shipped"
    assert re.search(r"no normalization", lowered), (
        "the F1 section no longer states that it added no normalization"
    )
    assert "persistent_buffers" in f1 or "STATE_SUPPORT" in f1
    # The private helper exists, is used, and stays private.
    helper = REPO_ROOT / "src" / "tensorforge" / "experimental" / "_native_state.py"
    assert helper.is_file(), "the F1 transaction helper is missing"
    module_source = (
        REPO_ROOT / "src" / "tensorforge" / "experimental" / "native_module.py"
    ).read_text(encoding="utf-8")
    assert "_native_state" in module_source, (
        "load_state_dict no longer uses the shared state transaction"
    )
    import tensorforge.experimental as experimental
    assert "_native_state" not in experimental.__all__
    assert "replace_native_state" not in experimental.__all__
    assert not hasattr(experimental, "replace_native_state")
    # F1 itself added no normalization; "layernorm" only left UNSUPPORTED
    # at the later milestone F2, and "batchnorm" is still absent.
    assert "batchnorm" in cpp.UNSUPPORTED


def test_f3_shipped_the_first_stateful_native_module():
    """F3's own claims, checked against the live code and registry: the
    module exists and is exported, it is the first *stateful* native
    numerical module (parameters **and** persistent buffers), it added no
    operation/Core/kernel/ABI capability, the checkpoint format is still
    version 1, and the design section records it as complete while
    naming F4 as the milestone that may finally free ``"batchnorm"``."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint
    import tensorforge.experimental as experimental

    # It exists, is exported, and is in exactly one inventory.
    assert "NativeBatchNorm1d" in cpp.NATIVE_MODULES
    assert "NativeBatchNorm1d" in experimental.__all__
    assert hasattr(experimental, "NativeBatchNorm1d")
    for inventory in (cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS, cpp.RAW_KERNELS,
                      cpp.NATIVE_LOSSES, cpp.NATIVE_METRICS,
                      cpp.NATIVE_OPTIMIZERS, cpp.STATE_SUPPORT,
                      cpp.UNSUPPORTED):
        assert "NativeBatchNorm1d" not in inventory

    # It really is stateful: parameters first, persistent buffers second.
    module = experimental.NativeBatchNorm1d(3)
    try:
        assert [name for name, _ in module.named_parameters()] == [
            "gamma", "beta"
        ]
        assert [name for name, _ in module.named_buffers()] == [
            "running_mean", "running_var"
        ]
        state = module.state_dict()
        assert list(state) == [
            "gamma", "beta", "running_mean", "running_var"
        ]
        for snapshot in state.values():
            snapshot.close()
    finally:
        for tensor in module.parameters() + module.buffers():
            tensor.close()

    # The checkpoint format did not move for the new persistent keys.
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 1

    # No numerical primitive appeared at any layer, and no C++ unit.
    for name in ("batch_norm", "batchnorm", "batch_norm_forward",
                 "batch_norm_backward"):
        assert name not in cpp.TENSOR_CORE_OPS, name
        assert name not in cpp.AUTOGRAD_OPS, name
        assert name not in cpp.RAW_KERNELS, name
        assert not hasattr(cpp.NativeTensorCore, name), name
    for symbol in ("tf_core_batch_norm", "tf_core_batch_norm_forward",
                   "tf_core_batch_norm_backward"):
        assert symbol not in cpp._CHECKED_KERNELS, symbol
    for source in (REPO_ROOT / "cpp" / "src").glob("*.cpp"):
        assert "batch_norm" not in source.read_text(encoding="utf-8"), source.name
    # And no custom BatchNorm backward was written in Python either.
    implementation = (
        REPO_ROOT / "src" / "tensorforge" / "experimental" / "native_batchnorm.py"
    ).read_text(encoding="utf-8")
    assert "_from_op(" not in implementation
    assert "def _backward" not in implementation
    # It does use the F1 transaction rather than a second one.
    assert "_native_state" in implementation
    assert "replace_native_state" in implementation
    # ...and it never routes a running update through the public loader.
    assert not re.search(r"\.load_state_dict\(", implementation)

    # The design records F3 complete and keeps F4 as the batchnorm gate.
    f3 = _design_section("F3 —", relative_path=PHASE_F_DESIGN)
    assert "complete" in f3.lower(), "the design does not record F3 as shipped"
    assert "batchnorm" in cpp.UNSUPPORTED


def test_f3_documents_the_running_buffers_and_the_graph_capture_ban():
    """The two load-bearing F3 facts must be written down where a reader
    finds them: the running statistics are *persistent native state*, and
    capturing a live registered buffer in a graph is forbidden."""
    from tensorforge.backends import cpp

    assert "NativeBatchNorm1d" in cpp.NATIVE_MODULES     # premise

    design = _status_text(PHASE_F_DESIGN)
    matrix = _status_text("docs/native_support_matrix.md")
    implementation = (
        REPO_ROOT / "src" / "tensorforge" / "experimental" / "native_batchnorm.py"
    ).read_text(encoding="utf-8")

    for text, where in ((design, PHASE_F_DESIGN),
                        (matrix, "docs/native_support_matrix.md"),
                        (implementation, "native_batchnorm.py")):
        lowered = text.lower()
        assert "running_mean" in text and "running_var" in text, where
        assert "persistent" in lowered, where
        assert "snapshot" in lowered, where
        assert re.search(r"never.{0,60}(captur|rereadable|graph operand)"
                         r"|graph-free snapshot|immutable snapshot", lowered), where
        assert "atomic" in lowered, where
    # The registry-level premise for the ban: buffers still carry no
    # value version, which is only safe because graphs never read them.
    from tensorforge.experimental import NativeParameter, NativeTensor
    assert hasattr(NativeParameter, "version")
    assert not hasattr(NativeTensor, "version")


def test_f3_scopes_the_snapshot_rule_to_buffer_only_mutation():
    """The §7 snapshot rule protects an eval graph from *running-buffer*
    mutation. A full checkpoint load also replaces ``gamma``/``beta``, and
    the pre-existing parameter-version guard then correctly stales that
    graph. No status surface may flatten the two into "any checkpoint load
    leaves an earlier eval graph valid", and the design must say which is
    which — while still forbidding live-buffer capture."""
    from tensorforge.backends import cpp

    assert "NativeBatchNorm1d" in cpp.NATIVE_MODULES     # premise

    # The design states the distinction explicitly, in §7 and in F3.
    section = _top_level_section("Mutable-buffer graph safety")
    lowered = section.lower()
    assert "gamma" in lowered and "beta" in lowered, (
        "§7 no longer says which state a full load *does* stale"
    )
    assert re.search(r"buffer.only|only\s+running_mean", lowered), (
        "§7 no longer scopes the rule to buffer-only loads"
    )
    assert re.search(r"stale.?parameter|stale-value guard|parameter contract",
                     lowered), (
        "§7 no longer names the parameter-version guard as the other case"
    )
    # ...and it still forbids capturing the live buffer.
    assert re.search(r"never be captured|never captured", lowered)

    f3 = _design_section("F3 —", relative_path=PHASE_F_DESIGN)
    f3_lowered = f3.lower()
    assert "load_native_checkpoint" in f3_lowered, (
        "the F3 record does not name the real checkpoint path it proved"
    )
    assert re.search(r"buffer.only", f3_lowered), (
        "the F3 record does not scope its checkpoint proof to a "
        "buffer-only load"
    )

    # No surface may make the unqualified claim.
    overclaim = re.compile(
        r"(any|every|a)\s+checkpoint load[^.]{0,80}?"
        r"(leaves|keeps)[^.]{0,60}(valid|unchanged|correct)",
        re.I,
    )
    for surface in AUTHORITATIVE_STATUS_SURFACES + PHASE_STATUS_DOCS + (
        PHASE_F_DESIGN,
    ):
        text = _status_text(surface)
        match = overclaim.search(text)
        assert match is None, (surface, match.group(0) if match else "")

    # The test that proves the checkpoint half really drives the archive
    # path over the module's own buffer objects, not a state dictionary.
    suite = (REPO_ROOT / "tests" / "test_native_batchnorm1d.py").read_text(
        encoding="utf-8"
    )
    assert "test_buffer_only_checkpoint_load_cannot_change_an_earlier_eval_backward" in suite
    assert "test_full_checkpoint_load_stales_the_graph_through_parameters_not_buffers" in suite
    assert "load_native_checkpoint" in suite
    # ...and the buffer-only holder it uses stayed test-only.
    import tensorforge.experimental as experimental
    for name in ("_RunningStatHolder", "RunningStatHolder"):
        assert not hasattr(experimental, name), name
        assert name not in experimental.__all__, name


def test_no_document_claims_unshipped_normalization_is_done():
    """After F2 and F3, LayerNorm and the ``(N, C)`` BatchNorm really
    *are* shipped, so a document may say so — running statistics
    included. What no status surface may claim is the part that has
    **not** shipped: ``NativeBatchNorm2d`` / the NCHW shape, any milestone
    from F4 on, the unqualified ``batchnorm`` capability, or the whole
    phase. The registry premise for each is checked first."""
    from tensorforge.backends import cpp

    # Premise: the NCHW BatchNorm is not implemented, so the unqualified
    # capability stays unsupported.
    assert "batchnorm" in cpp.UNSUPPORTED
    assert "NativeBatchNorm1d" in cpp.NATIVE_MODULES
    assert "NativeBatchNorm2d" not in cpp.NATIVE_MODULES

    # The subject is only the *unshipped* surface — LayerNorm and the 1-D
    # BatchNorm (running statistics included) are excluded because they
    # have genuinely shipped.
    subject = r"(NativeBatchNorm2d|NCHW BatchNorm|BatchNorm2d)"
    shipped = (r"(is|are|now)\s+(supported|implemented|shipped|complete"
               r"|available)")
    claims = (
        # "NativeBatchNorm2d is implemented", either word order.
        re.compile(subject + r"[^.]{0,60}?" + shipped, re.I),
        # "F4 shipped NativeBatchNorm2d", ...
        re.compile(r"\bF[4-9]\b[^.]{0,60}?(ship|implement|add)\w*[^.]{0,30}?"
                   + subject, re.I),
        # Any milestone from F4 on described as done.
        re.compile(r"\bF[4-9]\b[^.]{0,40}?\b(is|was)\s+"
                   r"(complete|completed|shipped|implemented)\b", re.I),
        # The unqualified capability described as finished.
        re.compile(r"BatchNorm[^.]{0,40}?(support|capability)[^.]{0,30}?"
                   r"\b(is|are)\s+complete", re.I),
        # The phase itself described as finished.
        re.compile(r"Phase F\b[^.F]{0,40}?\b(is|was|are|now)\s+"
                   r"(complete|completed|shipped|implemented)\b", re.I),
    )
    # A matched span that carries its own negation ("normalization is
    # **not** implemented", "F2 is not complete") is the honest form, not
    # an over-claim. Filtering on the matched text keeps the guard
    # readable; the registry-derived tests above remain the authority on
    # what actually exists, so this prose guard never has to be exact.
    negations = re.compile(
        r"\b(not|never|neither|nor|nothing|none|planned|unsupported)\b"
        r"|\bno\b", re.I,
    )
    for surface in AUTHORITATIVE_STATUS_SURFACES + PHASE_STATUS_DOCS + (
        PHASE_F_DESIGN,
    ):
        text = _status_text(surface)
        for pattern in claims:
            offenders = []
            for match in pattern.finditer(text):
                # Include a little context before the match: the negation
                # often leads the sentence ("Nothing normalization-related
                # is shipped").
                window = text[max(0, match.start() - 45):match.end()]
                if not negations.search(window):
                    offenders.append(match.group(0))
            assert offenders == [], (
                f"{surface} over-claims Phase F progress: {offenders[:3]}"
            )


def test_no_document_claims_layernorm_is_still_unsupported_or_planned():
    """The inverse guard for F2: now that NativeLayerNorm has shipped, no
    status surface may still describe LayerNorm as unsupported, planned,
    unimplemented, or not started."""
    from tensorforge.backends import cpp

    assert "layernorm" not in cpp.UNSUPPORTED
    assert "NativeLayerNorm" in cpp.NATIVE_MODULES
    # A tight same-clause window: LayerNorm directly asserted unshipped.
    # "layernorm has left UNSUPPORTED" and "layernorm removed from
    # UNSUPPORTED; … F3–F9 are planned" are honest and must not trip it, so
    # the claim word must sit close to the subject and matches whose window
    # carries a shipped/removed marker are dropped.
    stale = re.compile(
        r"(NativeLayerNorm|LayerNorm)\s+"
        r"(is|are|remains|stays|still)\s+"
        r"(unsupported|planned|not implemented|unimplemented|not started"
        r"|not yet implemented|absent|missing)",
        re.I,
    )
    positive = re.compile(
        r"(shipped|complete|left|removed|joined|now in|supported|exists)",
        re.I,
    )
    for surface in AUTHORITATIVE_STATUS_SURFACES + PHASE_STATUS_DOCS + (
        PHASE_F_DESIGN,
    ):
        text = _status_text(surface)
        for match in stale.finditer(text):
            window = text[max(0, match.start() - 40):match.end() + 20]
            assert positive.search(window), (
                f"{surface} still calls LayerNorm unsupported/planned: "
                f"{match.group(0)!r}"
            )


def test_no_document_claims_persistent_buffers_are_unreported():
    """The inverse guard for F1's reconciliation: now that
    STATE_SUPPORT reports persistent buffers, no document may still
    describe that capability as missing from it."""
    from tensorforge.backends import cpp

    assert "persistent_buffers" in cpp.STATE_SUPPORT
    stale = re.compile(
        r"(STATE_SUPPORT|state_support)[^.]{0,120}?"
        r"(does not|never|no longer|fails to|under-?report)"
        r"[^.]{0,60}(buffer|persistent)"
        r"|persistent buffers?[^.]{0,80}?(absent from|missing from|not in)"
        r"[^.]{0,40}(STATE_SUPPORT|state_support)",
        re.I,
    )
    for surface in AUTHORITATIVE_STATUS_SURFACES + PHASE_STATUS_DOCS + (
        PHASE_F_DESIGN,
    ):
        text = _status_text(surface)
        match = stale.search(text)
        assert match is None, (
            f"{surface} still says persistent buffers are unreported: "
            f"{match.group(0)!r}" if match else ""
        )


def test_dropout_float32_cuda_and_amp_stay_unsupported_through_phase_f():
    """Requirement 10. Phase F excludes all four explicitly, and its
    design document must say so rather than leaving it to the registry."""
    from tensorforge.backends import cpp

    for future in ("dropout", "float32", "cuda", "amp"):
        assert future in cpp.UNSUPPORTED, future
        assert future not in cpp.AUTOGRAD_OPS
        assert future not in cpp.TENSOR_CORE_OPS
        assert future not in cpp.NATIVE_MODULES
    design = _status_text(PHASE_F_DESIGN)
    for excluded in ("dropout", "CUDA", "AMP", "float32"):
        assert excluded in design, (
            f"{PHASE_F_DESIGN} does not exclude {excluded!r}"
        )
    assert cpp.backend_info()["stable_framework_integration"] is False


def test_status_docs_agree_on_the_phase_sequence():
    """Requirement 11. Every status document that discusses the native
    line must place the phases in the same order and must not invent one
    beyond Phase F."""
    for surface in PHASE_STATUS_DOCS:
        text = _status_text(surface)
        assert "Phase E" in text and "Phase F" in text, surface
        # Phase F is the newest phase; nothing later may be named.
        for beyond in ("Phase G", "Phase H"):
            assert beyond not in text, f"{surface} names {beyond}"
    # The phase sequence is A..F with no gaps: the set of phases a
    # document names must be a contiguous prefix-suffix of that ladder,
    # never a set that skips one. (Ordering *within* a document is not
    # pinned — the support matrix legitimately leads with the newest
    # phase and then recaps the ladder.)
    ladder = "ABCDEF"
    for surface in ("docs/native_support_matrix.md", "docs/roadmap.md",
                    "docs/project_summary.md"):
        text = _status_text(surface)
        named = [letter for letter in ladder if f"Phase {letter}" in text]
        assert named, surface
        span = ladder[ladder.index(named[0]):ladder.index(named[-1]) + 1]
        assert "".join(named) == span, (
            f"{surface} skips a phase: names {named}, expected the "
            f"contiguous run {list(span)}"
        )
        # The newest phase named must be F — no document may stop at E
        # and thereby imply Phase E is still the current phase.
        assert named[-1] == "F", (
            f"{surface} stops at Phase {named[-1]}; Phase F is current"
        )


def test_every_doc_linked_from_the_readme_exists():
    """Requirement 12, in the general form: not just the curated DOCS
    tuple, but every docs/ path the README actually links to."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"docs/([\w./-]+\.md)", readme)))
    assert referenced, "README links to no documentation at all"
    for name in referenced:
        assert (REPO_ROOT / "docs" / name).is_file(), (
            f"README links to docs/{name}, which does not exist"
        )
    # And the reverse for the design documents: a contract that exists but
    # is unreachable from the README is an orphan.
    for design in ("native_cnn_design.md", "native_classification_design.md",
                   "native_normalization_design.md"):
        assert design in referenced, f"docs/{design} is not linked"


def test_phase_f_design_records_the_daedalus_comparison():
    """The phase must record what it took from the comparable Daedalus
    design and — more importantly — what it deliberately rejected, so the
    architectural choices are justified rather than inherited."""
    section = _top_level_section("Daedalus")
    lowered = section.lower()
    # Ideas taken.
    assert "composition" in lowered or "composed" in lowered
    assert "shared private" in lowered or "shared implementation" in lowered
    # Ideas rejected — each one names a real architectural boundary.
    for rejected in ("pybind11", "numpy", "detached", "host array"):
        assert rejected in lowered, (
            f"the Daedalus comparison does not record rejecting {rejected!r}"
        )
    assert re.search(r"c\+\+-managed autograd|autograd.{0,20}c\+\+", lowered), (
        "the comparison no longer rejects C++-managed autograd"
    )
    assert re.search(r"cop(y|ied|ying)", lowered), (
        "the comparison no longer states that no implementation is copied"
    )


def test_phase_f_ladder_marks_shipped_milestones_complete():
    """The ladder's honesty check, derived from the live registry rather
    than from a hard-coded milestone number: every module milestone whose
    module is actually in ``NATIVE_MODULES`` must read complete, and every
    milestone after the last shipped one must read planned. F0's section
    still denies adding numerical behavior, and the phase statement reads
    in-progress, not complete."""
    from tensorforge.backends import cpp

    text = _normalized_doc(PHASE_F_DESIGN)
    ladder = _design_section("Milestone ladder", relative_path=PHASE_F_DESIGN)
    # F0 (the contract) and F1 (the state transaction) always ship first;
    # F2/F3/F4 ship exactly when their module reaches the registry.
    shipped = [0, 1]
    for milestone, module in ((2, "NativeLayerNorm"),
                              (3, "NativeBatchNorm1d"),
                              (4, "NativeBatchNorm2d")):
        if module in cpp.NATIVE_MODULES:
            shipped.append(milestone)
    # The shipped set must be a contiguous prefix — no milestone may be
    # skipped.
    assert shipped == list(range(len(shipped))), shipped
    for done in shipped:
        row = re.search(rf"\|\s*F{done}\s*\|[^|]*\|([^|]*)\|", ladder)
        assert row is not None, f"the ladder has no status row for F{done}"
        assert "complete" in row.group(1).lower(), f"F{done} is not complete"
    # Everything after the last shipped milestone has not started.
    for planned in range(len(shipped), 10):
        row = re.search(rf"\|\s*F{planned}\s*\|[^|]*\|([^|]*)\|", ladder)
        assert row is not None, f"the ladder has no status row for F{planned}"
        status = row.group(1).lower()
        assert "planned" in status, f"F{planned} is not marked planned"
        assert "complete" not in status, f"F{planned} is marked complete"
        assert "in progress" not in status, f"F{planned} is marked in progress"
    # F0's own contract section denies adding numerical behavior.
    f0 = _design_section("F0 —", relative_path=PHASE_F_DESIGN)
    assert re.search(r"no.{0,60}numerical", f0, re.I), (
        "the F0 section no longer denies adding numerical behavior"
    )
    # And the phase statement reads in-progress, not complete.
    plain = re.sub(r"[*`]", "", text)
    assert "Phase F is in progress" in plain
    assert "Phase F is designed, not implemented" not in plain


def test_native_cnn_design_is_linked_and_referenced():
    """The design doc must be reachable from the roadmap and the support
    matrix so the contract is discoverable, not orphaned."""
    roadmap = (REPO_ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    matrix = (REPO_ROOT / "docs" / "native_support_matrix.md").read_text(
        encoding="utf-8"
    )
    for doc in (roadmap, matrix):
        assert "native_cnn_design.md" in doc
