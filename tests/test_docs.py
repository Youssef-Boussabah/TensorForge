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
        "NativeBatchNorm2d",                            # Phase F, F4
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
    # ("layernorm" left UNSUPPORTED at Phase F milestone F2 and
    # "batchnorm" at F4, once both BatchNorm shapes shipped as composed
    # modules; neither is a Phase-D boundary.)
    for absent in ("float32", "cuda", "amp", "dropout"):
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

    # ("layernorm" is no longer future work — F2 shipped NativeLayerNorm —
    # and neither is "batchnorm", which F3 and F4 completed.)
    for future in ("cuda", "amp", "float32", "dropout"):
        assert future in cpp.UNSUPPORTED, future
        assert future not in cpp.AUTOGRAD_OPS and future not in cpp.TENSOR_CORE_OPS
        assert future not in cpp.NATIVE_MODULES
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.backend_info()["stable_framework_integration"] is False
    # ("LayerNorm" left this list at Phase F milestone F2 and "BatchNorm"
    # at F4, because both are now genuinely shipped; a document saying so
    # is honest, not an over-claim. The remaining subjects are still
    # absent from the native line.)
    claim = re.compile(
        r"(CUDA|AMP|float32|float16|bfloat16|GPU|"
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
    # ("layernorm" left UNSUPPORTED at F2 and "batchnorm" at F4.)
    for future in ("cuda", "float32", "amp", "dropout"):
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


def test_phase_f_is_positively_marked_complete_everywhere():
    """The F9 closure form of the old in-progress guard. Phase F is now
    finished, so the requirement inverts: the design document must state
    its completed status, and **every** status surface must positively
    say Phase F is complete rather than merely avoid denying it. The
    premise — that all three normalization modules really shipped — is
    checked against the live registry first, so this can never assert a
    doc claim the code does not back."""
    from tensorforge.backends import cpp

    for module in _NORMALIZATION_MODULES:
        assert module in cpp.NATIVE_MODULES, module

    design = _status_text(PHASE_F_DESIGN)
    assert re.search(r"Phase-F status: complete", design), (
        "the design document no longer states its completed status"
    )
    # F0/F1's no-numerical-behavior honesty is still recorded.
    assert re.search(r"no numerical behavior", design, re.I)
    for surface in PHASE_STATUS_DOCS + (PHASE_F_DESIGN, "CLAUDE.md",
                                        "docs/release_history.md"):
        text = _status_text(surface)
        assert "Phase F" in text, f"{surface} does not name Phase F at all"
        assert re.search(r"Phase F\b[^.]{0,120}?\bcomplete\b", text, re.I), (
            f"{surface} does not positively state that Phase F is complete"
        )


def test_no_surface_still_calls_phase_f_or_f9_pending():
    """The inverse of the check above, and the one that would catch a
    surface left behind at F8. No authoritative surface may still present
    Phase F, or milestone F9, as in progress, planned, unfinished, or not
    started. Sentences that scope the claim to a past milestone ("at F5,
    F6-F9 had not started") are the honest historical form and are
    excluded by the past-tense filter."""
    pending = (r"(in progress|not finished|is planned|has not started"
               r"|have not started|not started|is next|is upcoming)")
    subjects = (r"Phase.F", r"\bF9\b")
    # A record explicitly scoped to an earlier moment is history, not a
    # stale status claim.
    historical = re.compile(
        r"\b(had not|was the next|were|since shipped|at F\d|then\b)", re.I
    )
    # "Beyond Phase F (not started)" is a statement about *later* work, and
    # is exactly the honest scoping the closure must keep. Checked in a
    # tight window immediately before the subject so it cannot excuse an
    # unrelated stale claim further along the sentence.
    scoped_later = re.compile(r"\b(beyond|outside|after)\s*$", re.I)
    for surface in PHASE_STATUS_DOCS + (PHASE_F_DESIGN, "CLAUDE.md",
                                        "docs/release_history.md"):
        text = _status_text(surface)
        for subject in subjects:
            pattern = re.compile(subject + r"[^.]{0,80}?" + pending, re.I)
            offenders = [
                match.group(0) for match in pattern.finditer(text)
                if not historical.search(
                    text[max(0, match.start() - 60):match.end() + 60]
                )
                and not scoped_later.search(
                    text[max(0, match.start() - 12):match.start()]
                )
            ]
            assert offenders == [], (
                f"{surface} still presents {subject} as pending: "
                f"{offenders[:3]}"
            )


def test_f9_is_documented_as_validation_and_documentation_only():
    """F9 closed the phase and must never be described as having added
    numerical capability. The design records what it actually did —
    Release/Debug builds, sanitizers, LeakSanitizer, and the
    reconciliation — and explicitly denies adding capability."""
    design = _status_text(PHASE_F_DESIGN)
    f9 = _design_section("F9 —", relative_path=PHASE_F_DESIGN)
    assert re.search(r"F9\b[^.]{0,80}\bcomplete\b", design, re.I)
    # It says what it was: validation and documentation.
    assert re.search(r"no (new )?numerical capability", f9, re.I), (
        "the F9 section no longer denies adding numerical capability"
    )
    # And it records the evidence the closure rests on.
    for token in ("Release", "Debug", "CTest", "ASan", "UBSan",
                  "LeakSanitizer", "baseline"):
        assert token.lower() in f9.lower(), f"F9 records no {token} evidence"


def test_phase_f_closure_records_the_build_and_sanitizer_evidence():
    """The closure's evidence must be present on the durable surfaces,
    not only in a chat transcript: Release **and** Debug builds, the
    sanitizer pass, and the honest LeakSanitizer attribution."""
    for surface in (PHASE_F_DESIGN, "docs/native_support_matrix.md",
                    "docs/release_history.md", "docs/backend_experiments.md"):
        text = _status_text(surface)
        assert re.search(r"Release", text) and re.search(r"Debug", text), surface
        assert re.search(r"CTest", text, re.I), surface
        assert re.search(r"ASan", text, re.I), surface
        assert re.search(r"UBSan", text, re.I), surface
    # The leak claim must stay the honest one: an exact return to
    # baseline plus no TensorForge-attributable frame — never a bare
    # "LeakSanitizer found zero leaks".
    for surface in (PHASE_F_DESIGN, "docs/release_history.md",
                    "docs/backend_experiments.md"):
        text = _status_text(surface)
        assert re.search(r"baseline", text, re.I), surface
        overclaim = re.search(
            r"(LeakSanitizer|LSan)[^.]{0,40}(found|reports?)[^.]{0,20}"
            r"(zero|no) leaks?\b", text, re.I,
        )
        assert overclaim is None, (
            f"{surface} over-claims a process-wide zero-leak result: "
            f"{overclaim.group(0)!r}" if overclaim else ""
        )


def test_normalization_is_module_only_with_no_new_native_operation():
    """Checked against the live registry and exports rather than prose.
    Milestones F2, F3, and F4 shipped ``NativeLayerNorm``,
    ``NativeBatchNorm1d``, and ``NativeBatchNorm2d``, each as a *module
    composed from existing operations*: for every normalization shape
    alike there is no normalization operation, kernel, Core method,
    ``NativeTensor`` method, or C ABI symbol anywhere. They added
    modules, not numerical primitives."""
    from tensorforge.backends import cpp
    import tensorforge.experimental as experimental

    # All three shipped as modules, so both capability names have left
    # UNSUPPORTED — and nothing normalization-shaped entered an
    # operation inventory on the way.
    assert "layernorm" not in cpp.UNSUPPORTED
    assert "batchnorm" not in cpp.UNSUPPORTED
    for module in _NORMALIZATION_MODULES:
        assert module in cpp.NATIVE_MODULES, module
        assert module in experimental.__all__, module
    for module in ("NativeBatchNorm3d", "NativeInstanceNorm2d",
                   "NativeGroupNorm", "NativeRMSNorm"):
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
    Phase F's three normalization modules — ``NativeLayerNorm`` (F2),
    ``NativeBatchNorm1d`` (F3), and ``NativeBatchNorm2d`` (F4) — in both
    directions: nothing else added, nothing lost. F5-F9 added no export at
    all, so the closed phase's surface is exactly F4's."""
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
        "NativeBatchNorm2d",     # Phase F, milestone F4
    }
    for absent in ("NativeBatchNorm3d", "NativeDropout"):
        assert absent not in experimental.__all__, absent
    # No duplicates, and nothing leaks into the stable namespace.
    assert len(experimental.__all__) == len(set(experimental.__all__))
    for name in experimental.__all__:
        assert hasattr(experimental, name)
        assert not hasattr(tensorforge, name), name


def test_phase_f_changes_only_the_normalization_module_inventory():
    """Phase F's only *operation*-surface changes so far are two modules:
    ``NativeLayerNorm`` (F2), ``NativeBatchNorm1d`` (F3), and
    ``NativeBatchNorm2d`` (F4) joined ``NATIVE_MODULES``, and
    ``"layernorm"`` (F2) then ``"batchnorm"`` (F4, once both BatchNorm
    shapes existed) left ``UNSUPPORTED``. Everything else is exactly the
    Phase-E surface — no new operation, kernel, loss, metric, or dtype —
    and ``STATE_SUPPORT``'s F1 reconciliation has its own guardrail below.
    Because all three are *modules composed from existing operations*, no
    operation inventory grew."""
    from tensorforge.backends import cpp

    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.UNSUPPORTED == (
        "dropout", "float32", "cuda", "amp",
    )
    # The Phase-E operation surface is intact and nothing normalization
    # shaped joined it — none of the three modules added an operation.
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
    # F1 itself added no normalization: "layernorm" only left UNSUPPORTED
    # at the later milestone F2, and "batchnorm" only at F4. What F1
    # contributed is the transaction and the STATE_SUPPORT reconciliation,
    # neither of which is an operation or a module.
    assert "persistent_buffers" in cpp.STATE_SUPPORT
    for inventory in (cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.RAW_KERNELS, cpp.NATIVE_MODULES):
        assert "persistent_buffers" not in inventory


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

    # The design records F3 complete. (Whether "batchnorm" has left
    # UNSUPPORTED is F4's question, guarded separately below — F3 itself
    # deliberately kept it.)
    f3 = _design_section("F3 —", relative_path=PHASE_F_DESIGN)
    assert "complete" in f3.lower(), "the design does not record F3 as shipped"


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
    """After the F9 closure the whole phase really is shipped, so a
    document may say so. What no status surface may claim is a
    normalization family Phase F **never scoped** — BatchNorm3d,
    InstanceNorm, GroupNorm, RMSNorm, synchronized/distributed BatchNorm,
    a fused normalization kernel, or a `NativeTensor.batch_norm`
    operation. Closing the phase must not quietly widen it. The
    registry/file premise is checked first."""
    from tensorforge.backends import cpp
    import tensorforge.experimental as experimental

    # Premise: the modules, the hardening, the F6 example, the F7
    # benchmark, and the F8 integration file all shipped.
    for module in _NORMALIZATION_MODULES:
        assert module in cpp.NATIVE_MODULES, module
    assert "batchnorm" not in cpp.UNSUPPORTED
    assert (REPO_ROOT / "examples"
            / "native_normalization_training.py").is_file()
    assert (REPO_ROOT / "benchmarks"
            / "benchmark_native_normalization.py").is_file()
    assert (REPO_ROOT / "tests" / "test_native_phase_f.py").is_file()

    # The subject is only the *unshipped* surface — the shipped modules are
    # excluded because they genuinely exist.
    subject = (r"(BatchNorm3d|InstanceNorm|GroupNorm|RMSNorm"
               r"|synchronized BatchNorm|distributed BatchNorm"
               r"|fused normalization|NativeTensor\.batch_norm)")
    shipped = (r"(is|are|now)\s+(supported|implemented|shipped|complete"
               r"|available)")
    claims = (
        # "GroupNorm is implemented", either word order.
        re.compile(subject + r"[^.]{0,60}?" + shipped, re.I),
        re.compile(shipped + r"[^.]{0,60}?" + subject, re.I),
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


def test_f4_completed_the_normalization_module_surface_not_the_phase():
    """F4's own claims, checked against the live code and registry: the
    NCHW shape exists and is exported, all three normalization modules
    are registered, both capability names have left ``UNSUPPORTED``, the
    remaining boundary is untouched, and no operation/Core/kernel/ABI
    surface grew. F4 completed the normalization *module* surface, not the
    phase — F5-F9 (hardening, the training proof, the benchmark, the
    integration suite, and the closure) followed, none of them adding a
    capability, so the boundary asserted here is still the current one."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint
    import tensorforge.experimental as experimental

    # The complete module surface, each in exactly one inventory.
    for module in _NORMALIZATION_MODULES:
        assert module in cpp.NATIVE_MODULES, module
        assert module in experimental.__all__, module
        assert hasattr(experimental, module), module
        for inventory in (cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                          cpp.RAW_KERNELS, cpp.NATIVE_LOSSES,
                          cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
                          cpp.STATE_SUPPORT, cpp.UNSUPPORTED):
            assert module not in inventory, (module, inventory)
    # Both normalization capability names are supported; the rest of the
    # boundary is exactly what it was, in its established order.
    assert "layernorm" not in cpp.UNSUPPORTED
    assert "batchnorm" not in cpp.UNSUPPORTED
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.backend_info()["unsupported"] == cpp.UNSUPPORTED
    # Nothing normalization-shaped appeared at any numerical layer.
    for name in _NORMALIZATION_OP_NAMES:
        assert name not in cpp.TENSOR_CORE_OPS, name
        assert name not in cpp.AUTOGRAD_OPS, name
        assert name not in cpp.RAW_KERNELS, name
        assert not hasattr(cpp.NativeTensorCore, name), name
    for symbol in ("tf_core_batch_norm", "tf_core_layer_norm",
                   "tf_core_batch_norm_forward", "tf_core_batch_norm_backward"):
        assert symbol not in cpp._CHECKED_KERNELS, symbol
    for source in (REPO_ROOT / "cpp" / "src").glob("*.cpp"):
        text = source.read_text(encoding="utf-8")
        assert "batch_norm" not in text and "layer_norm" not in text, source.name
    assert native_checkpoint._FORMAT_VERSION == 1
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    # Out-of-scope normalization families never appeared.
    for absent in ("NativeBatchNorm3d", "NativeInstanceNorm2d",
                   "NativeGroupNorm", "NativeRMSNorm"):
        assert absent not in cpp.NATIVE_MODULES, absent
        assert not hasattr(experimental, absent), absent


def test_both_batchnorm_shapes_share_one_private_implementation():
    """The structural F4 claim, derived from the live classes: both
    public shapes live in the same file, subclass the same private base,
    inherit every method by function identity, and the base stays
    private."""
    import tensorforge.experimental as experimental
    from tensorforge.experimental import native_batchnorm
    from tensorforge.experimental import NativeBatchNorm1d, NativeBatchNorm2d

    base = native_batchnorm._NativeBatchNorm
    assert issubclass(NativeBatchNorm1d, base)
    assert issubclass(NativeBatchNorm2d, base)
    assert NativeBatchNorm1d.__module__ == NativeBatchNorm2d.__module__
    assert not hasattr(experimental, "_NativeBatchNorm")
    assert "_NativeBatchNorm" not in experimental.__all__

    shared = ("forward", "_training_forward", "_eval_forward", "_mean_over",
              "_inverse_std", "_snapshot", "_blend", "_affine",
              "_commit_running_state", "_validate_forward",
              "_registered_running", "__init__", "__repr__")
    source = (REPO_ROOT / "src" / "tensorforge" / "experimental"
              / "native_batchnorm.py").read_text(encoding="utf-8")
    for method in shared:
        one = getattr(NativeBatchNorm1d, method)
        assert one is getattr(NativeBatchNorm2d, method), method
        assert one is getattr(base, method), method
        assert method not in vars(NativeBatchNorm1d), method
        assert method not in vars(NativeBatchNorm2d), method
        if not method.startswith("__"):
            assert source.count(f"    def {method}(") == 1, method
    # The NCHW subclass declares only shape/layout configuration.
    declared = {n for n in vars(NativeBatchNorm2d) if not n.startswith("__")}
    assert declared == {"_INPUT_NDIM", "_REDUCTION_AXES", "_TRAILING_DIMS",
                        "_LAYOUT", "_CHANNELS_LAST"}
    assert not any(callable(vars(NativeBatchNorm2d)[n]) for n in declared)
    assert NativeBatchNorm2d._INPUT_NDIM == 4
    assert NativeBatchNorm2d._REDUCTION_AXES == (0, 2, 3)
    assert NativeBatchNorm2d._TRAILING_DIMS == 2
    # No custom backward and no graph-node construction anywhere in it.
    assert "_from_op(" not in source
    assert "def _backward" not in source


def test_phase_f_closed_without_changing_the_capability_boundary():
    """The closure form of the old "still in progress after F8" guard.
    Every Phase-F deliverable file exists, the ladder marks F9 complete —
    and, crucially, the phase closed with the capability boundary exactly
    where F4 left it: F7 measured only, F8 tested only, and F9 validated
    and documented only, so none of the three moved an inventory."""
    from tensorforge.backends import cpp

    # Premise, from the live tree: F5's, F6's, F7's, and F8's own
    # deliverables all exist and survived the closure.
    for relative in ("examples/native_normalization_training.py",
                     "tests/test_native_normalization_training.py",
                     "benchmarks/benchmark_native_normalization.py",
                     "tests/test_native_normalization_benchmark.py",
                     "tests/test_native_normalization_state.py",
                     "tests/test_native_phase_f.py"):
        assert (REPO_ROOT / relative).is_file(), relative
    # The design says complete, and the ladder marks F9 complete.
    design = _status_text(PHASE_F_DESIGN)
    assert "Phase-F status: complete" in design
    ladder = _design_section("Milestone ladder", relative_path=PHASE_F_DESIGN)
    row = re.search(r"\|\s*F9\s*\|[^|]*\|([^|]*)\|", ladder)
    assert row is not None
    assert "complete" in row.group(1).lower()
    assert "planned" not in row.group(1).lower()
    # The registry is unchanged where F7-F9 would have touched it.
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)


def test_f6_shipped_the_normalized_training_and_resume_proof():
    """F6's own claims, checked against the live tree and registry: the
    example and its integration test exist and use the two normalization
    families, the design records F6 complete as an integration proof only,
    and the export set and every capability registry are exactly what F4
    left, with the checkpoint format still version 1. The design also keeps
    the ladder's own history readable: at F7 the next milestone was F8, and
    the record still says so in the past tense now that both have shipped."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint
    import tensorforge.experimental as experimental

    example = REPO_ROOT / "examples" / "native_normalization_training.py"
    assert example.is_file()
    assert (REPO_ROOT / "tests"
            / "test_native_normalization_training.py").is_file()
    text = example.read_text(encoding="utf-8")
    # The example runs both normalization families and neither the 2-D
    # BatchNorm nor a convolutional layer (F8's scope).
    for used in ("NativeBatchNorm1d(", "NativeLayerNorm(", "NativeMSELoss",
                 "NativeAdam", "save_native_checkpoint",
                 "load_native_checkpoint"):
        assert used in text, used
    for absent in ("NativeBatchNorm2d(", "NativeConv2d(", "NativeMaxPool2d("):
        assert absent not in text, absent
    # It never touches the stable framework and times nothing (the prose
    # may name "benchmark" to say measurement is F7's job; what it must not
    # do is import a timer or call one).
    assert "tensorforge.nn" not in text and "tensorforge.optim" not in text
    for banned in ("perf_counter", "import timeit", "import time",
                   "time.time("):
        assert banned not in text, banned

    # The design records F6 complete as an integration proof only.
    f6 = _design_section("F6 —", relative_path=PHASE_F_DESIGN)
    lowered = f6.lower()
    assert "complete" in lowered, "the design does not record F6 as shipped"
    assert "native_normalization_training.py" in f6
    assert re.search(r"no capability|adds no|integration proof|no new "
                     r"capability", lowered), (
        "the F6 section no longer scopes itself to an integration proof"
    )
    assert "version 1" in lowered

    # Exports and every capability registry are exactly what F4/F5 left.
    assert set(experimental.__all__) == {
        "NativeTensor", "NativeParameter", "NativeParameterRegistry",
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeMSELoss", "NativeSGD", "NativeAdam",
        "save_native_checkpoint", "load_native_checkpoint",
        "NativeCrossEntropyLoss", "native_accuracy",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
    }
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
    )
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "save_native_checkpoint", "load_native_checkpoint",
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    # No normalization operation, kernel, or checkpoint-schema change.
    for name in ("layer_norm", "batch_norm", "layernorm", "batchnorm"):
        assert name not in cpp.TENSOR_CORE_OPS
        assert name not in cpp.AUTOGRAD_OPS
        assert name not in cpp.RAW_KERNELS
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 1

    # F5's own design section still records the hardening milestone.
    f5 = _design_section("F5 —", relative_path=PHASE_F_DESIGN)
    assert "complete" in f5.lower()
    # F8 is named as the next milestone.
    design = _status_text(PHASE_F_DESIGN)
    assert re.search(r"F8[^.]{0,80}(next|planned|not started)", design, re.I)


def test_docs_present_the_shipped_normalization_benchmark():
    """F7's harness must stay documented as what it is: an honest local
    characterization with correctness gated before timing, honest
    reference labels, and **no** speed guarantee — never a performance
    contract or a CI gate. The premise is checked against the live tree
    first."""
    benchmark = REPO_ROOT / "benchmarks" / "benchmark_native_normalization.py"
    assert benchmark.is_file()
    assert (REPO_ROOT / "tests"
            / "test_native_normalization_benchmark.py").is_file()

    section = _design_section("F7 —", relative_path=PHASE_F_DESIGN)
    lowered = section.lower()
    assert "complete" in lowered, "the design does not record F7 as shipped"
    assert "benchmarks/benchmark_native_normalization.py" in section
    # The nine measured cases are named, and they are the ones the harness
    # actually declares.
    from benchmarks import benchmark_native_normalization as harness

    assert len(harness.CASES) == 9
    for case in harness.CASES:
        assert case in section, case
    # The methodology commitments.
    assert re.search(r"correctness.{0,40}before.{0,20}timing", lowered), (
        "the design no longer states that correctness runs before timing"
    )
    assert "median" in lowered and "spread" in lowered
    assert "warm-up" in lowered or "warmup" in lowered
    for label in ("stable_tensorforge", "native_only"):
        assert label in section, label
    assert "--smoke" in section and "--json" in section
    # The BatchNorm2d timing-label decision is justified, not asserted.
    assert re.search(r"no public .?BatchNorm2d", section, re.I), (
        "the design does not say why the BatchNorm2d cases are native_only"
    )
    assert "oracle" in lowered
    # And the honesty boundary, in the design and on the status surfaces.
    assert re.search(r"no.{0,40}speed", lowered), (
        "the design no longer states that no speed is asserted"
    )
    assert re.search(r"no ci timing threshold|no timing threshold", lowered)
    assert re.search(r"no result file", lowered)
    readme = _status_text("README.md")
    assert "benchmarks/benchmark_native_normalization.py" in readme
    for command in ("uv run python benchmarks/benchmark_native_normalization"
                    ".py --smoke",
                    "uv run python benchmarks/benchmark_native_normalization"
                    ".py --smoke --json"):
        assert command in readme, command
    for surface in ("README.md", "docs/native_support_matrix.md",
                    "docs/roadmap.md", "docs/project_summary.md",
                    "docs/architecture.md"):
        text = _status_text(surface)
        # A raw character window, not a sentence one: the file name itself
        # contains a period, which would truncate a "[^.]" span.
        window = [text[max(0, match.start() - 400):match.end() + 600]
                  for match in re.finditer("benchmark_native_normalization",
                                           text)]
        assert window, surface
        assert any(re.search(
            r"(no speed|not a (performance )?(contract|guarantee)"
            r"|no timing threshold|no committed timing|characteriz)",
            chunk, re.I) for chunk in window), surface
    matrix = _status_text("docs/native_support_matrix.md")
    assert re.search(r"\|\s*F7\s*\|[^|]*\|[^|]*Complete", matrix), (
        "the support matrix does not mark F7 complete"
    )


def test_docs_present_the_shipped_phase_f_integration_suite():
    """F8's deliverable must stay documented as what it is: one
    cross-cutting integration and guardrail suite that adds **no**
    capability — never a new feature, and never the phase closure. The
    premise is checked against the live tree and the live registry
    first."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint

    suite = REPO_ROOT / "tests" / "test_native_phase_f.py"
    assert suite.is_file()
    source = suite.read_text(encoding="utf-8")
    # The integrated model really combines every family the docs claim.
    for piece in ("NativeConv2d", "NativeBatchNorm2d", "NativeReLU",
                  "NativeMaxPool2d", "NativeFlatten", "NativeLinear",
                  "NativeBatchNorm1d", "NativeLayerNorm",
                  "NativeCrossEntropyLoss", "NativeAdam"):
        assert piece in source, piece
    # ...and it declares no runtime surface of its own: it imports no
    # ctypes machinery and defines no ABI entry point. Scanned as
    # top-of-line imports and definitions, so the guard cannot match the
    # suite's own assertions about the production modules.
    imported = re.findall(r"^(?:import|from)\s+([\w.]+)", source, re.M)
    assert not [name for name in imported if name.split(".")[0] == "ctypes"]
    assert re.search(r"^def tf_", source, re.M) is None
    assert re.search(r"^\s+_from_op\(", source, re.M) is None

    section = _design_section("F8 —", relative_path=PHASE_F_DESIGN)
    lowered = section.lower()
    assert "complete" in lowered, "the design does not record F8 as shipped"
    assert "tests/test_native_phase_f.py" in section
    # The interactions the milestone claims, named in its own section.
    for claim in ("batchnorm2d", "batchnorm1d", "layernorm", "maxpool2d",
                  "cross-entropy", "nativeadam", "checkpoint", "snapshot",
                  "shared", "frozen", "non-contiguous", "version"):
        assert claim in lowered, claim
    # It is scoped to tests and documentation, adding no capability.
    assert re.search(r"tests and documentation only|no capability|adds no",
                     lowered), (
        "the F8 section no longer scopes itself to tests and documentation"
    )
    assert "version 1" in lowered
    # The honest failure-boundary statement survives: transactions are per
    # module, so one whole training step is not globally transactional.
    assert re.search(r"per.module", lowered), (
        "the F8 section no longer states that transactions are per module"
    )
    assert re.search(r"not\W{0,10}globally transactional", lowered), (
        "the F8 section no longer denies whole-step global transactionality"
    )
    # F9 closed the phase, and the design records that.
    design = _status_text(PHASE_F_DESIGN)
    assert re.search(r"F9[^.]{0,80}\bcomplete\b", design, re.I)

    # Nothing changed in the registry, exports, or checkpoint format.
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
    )
    assert native_checkpoint._FORMAT_VERSION == 1

    # Every authoritative status surface agrees that F8 **and** F9
    # shipped — the closure form of the old "F9 has not" check.
    for surface in ("README.md", "docs/native_support_matrix.md",
                    "docs/roadmap.md", "docs/project_summary.md",
                    "docs/architecture.md", "docs/backend_experiments.md",
                    "CLAUDE.md"):
        text = _status_text(surface)
        assert re.search(r"F8[^.]{0,60}(complete|shipped)", text, re.I), surface
        assert re.search(r"(F9[^.]{0,120}(complete|shipped|clos)"
                         r"|F0.F9[^.]{0,60}(shipped|complete))",
                         text, re.I), surface


def test_the_phase_f_closure_claims_no_later_phase():
    """The closure form of the old guard. F9 owned the build, sanitizer,
    and completion work, and that work is now legitimately described as
    done — but closing Phase F must not become a claim about anything
    after it. No surface may present a *later* phase, or a capability
    Phase F never scoped, as started, in progress, or complete."""
    premise = _status_text(PHASE_F_DESIGN)
    assert "Phase-F status: complete" in premise

    started = (r"(is|are|was|were|now|has|have)\s+"
               r"(begun|started|under way|underway|in progress|complete"
               r"|completed|shipped|implemented|supported)")
    later = (r"(Phase G|Phase H|dropout phase|native RNG|CUDA (phase|runtime|"
             r"backend)|AMP (phase|path)|Tensor Core|CPU optimization phase"
             r"|distributed (phase|training)|float16|bfloat16)")
    # Negated or explicitly-future forms are the honest ones.
    excluded = re.compile(
        r"\b(not|never|no|future|beyond|planned|unplanned|outside|remains?"
        r"|remain|still|deliberately|excluded|unsupported)\b", re.I,
    )
    pattern = re.compile(later + r"[^.]{0,60}?" + started, re.I)
    for surface in PHASE_STATUS_DOCS + (PHASE_F_DESIGN, "CLAUDE.md",
                                        "docs/release_history.md"):
        text = _status_text(surface)
        offenders = [
            match.group(0) for match in pattern.finditer(text)
            if not excluded.search(
                text[max(0, match.start() - 70):match.end() + 30]
            )
        ]
        assert offenders == [], (
            f"{surface} claims a later phase has started: {offenders[:3]}"
        )


def test_the_normalization_benchmark_is_registered_nowhere():
    """A benchmark is a measurement tool, never a capability: it must not
    appear in any runtime inventory, and it must add no export."""
    from tensorforge.backends import cpp
    import tensorforge.experimental as experimental

    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                      cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
                      cpp.STATE_SUPPORT, cpp.UNSUPPORTED):
        for banned in ("benchmark", "characterization", "normalization"):
            assert not [n for n in inventory if banned in n.lower()], banned
    for banned in ("benchmark_native_normalization", "run_benchmark",
                   "format_report"):
        assert banned not in experimental.__all__, banned
        assert not hasattr(experimental, banned), banned


def test_phase_f_ladder_marks_every_milestone_complete():
    """The ladder's closure check, still derived from the live registry
    and the real tree rather than from hard-coded prose: every milestone
    whose deliverable actually exists must read complete, F0-F9 must form
    **one contiguous complete prefix** with no milestone left planned or
    in progress, F0's section still denies adding numerical behavior, and
    the phase statement reads complete."""
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
    # F5-F8 add no module or inventory entry, so each is detected from its
    # own deliverable file rather than from the registry.
    for milestone, relative in (
        (5, "tests/test_native_normalization_state.py"),
        (6, "examples/native_normalization_training.py"),
        (7, "benchmarks/benchmark_native_normalization.py"),
        (8, "tests/test_native_phase_f.py"),
    ):
        if (REPO_ROOT / relative).exists():
            shipped.append(milestone)
    # F9 is the closure itself: it ships no artifact of its own, so it is
    # detected from the design's own completed status statement.
    if re.search(r"Phase-F status: complete", _status_text(PHASE_F_DESIGN)):
        shipped.append(9)
    # The shipped set must be a contiguous prefix — no milestone skipped —
    # and at closure it must cover the whole ladder, F0 through F9.
    assert shipped == list(range(len(shipped))), shipped
    assert shipped == list(range(10)), (
        f"Phase F is closed, so F0-F9 must all be shipped; got {shipped}"
    )
    for done in shipped:
        row = re.search(rf"\|\s*F{done}\s*\|[^|]*\|([^|]*)\|", ladder)
        assert row is not None, f"the ladder has no status row for F{done}"
        status = row.group(1).lower()
        assert "complete" in status, f"F{done} is not complete"
        assert "planned" not in status, f"F{done} is still marked planned"
        assert "in progress" not in status, f"F{done} is marked in progress"
        assert "not started" not in status, f"F{done} is marked not started"
    # F0's own contract section denies adding numerical behavior.
    f0 = _design_section("F0 —", relative_path=PHASE_F_DESIGN)
    assert re.search(r"no.{0,60}numerical", f0, re.I), (
        "the F0 section no longer denies adding numerical behavior"
    )
    # And the phase statement reads complete.
    plain = re.sub(r"[*`]", "", text)
    assert "Phase F is complete" in plain
    assert "Phase F is in progress" not in plain
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


# --- Post-Phase-F repository-status reconciliation ------------------------
#
# Phases A-F are all complete, and the native line is merged into `main`,
# reached only through the explicit `tensorforge.backends` /
# `tensorforge.experimental` namespaces. The per-milestone guards above
# already pin what *shipped*; these pin the way the repository currently
# *describes itself*, which is where the post-merge drift collected: a
# high-level status still stopping at Phase E, an architecture sentence
# still placing the native line on "the advanced branch", a testing
# inventory written before Phase F, a native-example count contradicting
# the tree, and a capability comment still calling Phase F unfinished.
#
# Every check derives its premise from the live registry, the live
# exports, or the real tree, and matches meaning rather than wording, so
# an honest rewrite survives and a regression does not.

NATIVE_PHASES = "ABCDEF"

# "Phase F is unfinished", in the forms that survive a rewrite. This is
# deliberately a *different* vocabulary from the pending-phase guard
# above: that one catches "in progress"/"not started", this one catches
# the completion-denial forms ("itself is not", "is incomplete") that a
# closure pass can leave behind in explanatory prose.
_PHASE_F_UNFINISHED = (
    r"(is not complete|is not finished|itself is not|is incomplete"
    r"|not yet complete|remains incomplete|still incomplete|is unfinished"
    r"|awaiting (?:its )?closure|awaits closure|yet to close)"
)

# A present-tense claim that the native line lives on a separate advanced
# branch. Past-tense development history ("the advanced branch then built
# the native line", "leaving `advanced/cpp-backend` ready for its first
# pull request") is accurate and is deliberately not matched.
_PRESENT_TENSE_BRANCH = re.compile(
    r"\(\s*(?:this|the)\s+advanced\s+branch\s*\)"
    r"|(?:this|the)\s+advanced\s+branch\s+"
    r"(?:is|are|has|have|holds?|carries|adds?|contains?|provides?|lives?"
    r"|hosts?|owns?|keeps?|currently)\b"
    # "work continues on advanced branches" — the same claim without the
    # definite article, which the two forms above cannot see.
    r"|\b(?:continues?|continuing|lives?|happens?|proceeds?|sits?|resides?"
    r"|is|are)\s+(?:\w+\s+){0,3}?(?:on|in)\s+(?:the\s+|an?\s+)?"
    r"advanced\s+branch(?:es)?\b",
    re.I,
)

# A headline naming an *earlier* phase as the native line's newest
# completion — the exact drift the post-merge README carried ("the
# advanced branch has completed Phase E of its native line") — while
# leaving the range form ("has completed Phases A-F") alone.
_STALE_NEWEST_PHASE = re.compile(
    r"(?:has|have|had)\s+completed\s+Phase\s+([A-E])\b", re.I
)

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def test_high_level_status_states_phases_a_through_f_complete():
    """The README's own status must say the native line has completed
    Phases A-F — as a range ("Phases A-F ... complete") or phase by phase —
    and must never stop at an earlier phase again. The premise is the live
    registry: each phase's headline capability really is there, so this can
    only ever demand a claim the code already backs."""
    from tensorforge.backends import cpp

    assert "matmul" in cpp.TENSOR_CORE_OPS                        # Phase A
    assert cpp.backend_info()["native_autograd"] is True          # Phase B
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")   # Phase C
    assert "NativeConv2d" in cpp.NATIVE_MODULES                   # Phase D
    assert "cross_entropy" in cpp.AUTOGRAD_OPS                    # Phase E
    for module in _NORMALIZATION_MODULES:                         # Phase F
        assert module in cpp.NATIVE_MODULES, module

    readme = _status_text("README.md")
    span = r"Phases?\s+A\s*(?:-|–|—|through|to)\s*F\b"
    ranged = (
        re.search(span + r"[^.]{0,160}?\bcomplet", readme, re.I)
        or re.search(r"\bcomplet\w*[^.]{0,160}?" + span, readme, re.I)
    )
    per_phase = all(
        re.search(rf"Phase {letter}\b[^.]{{0,300}}?\bcomplete\b", readme, re.I)
        for letter in NATIVE_PHASES
    )
    assert ranged or per_phase, (
        "README's status no longer states that Phases A-F are complete"
    )
    # And it must not stop early: no headline may present an earlier phase
    # as the newest one the native line has completed.
    for surface in ("README.md", "docs/project_summary.md",
                    "docs/architecture.md", "CLAUDE.md"):
        text = _status_text(surface)
        assert "Phase F" in text, surface
        # An enumeration that reaches Phase F right there ("has completed
        # Phase A ... through Phase F") is the honest form and is excused;
        # a claim that stops earlier is the drift.
        offenders = [
            match.group(0) for match in _STALE_NEWEST_PHASE.finditer(text)
            if "Phase F" not in text[match.start():match.end() + 200]
        ]
        assert offenders == [], (
            f"{surface} presents an earlier phase as the newest completed "
            f"native phase: {offenders[:3]}"
        )


def test_no_current_status_surface_calls_phase_f_unfinished():
    """The completion-denial form of the pending-phase guard, extended to
    every authoritative surface — including the backend registry module,
    which the phase-narrative tuple deliberately excludes. This is the check
    that catches a capability comment left behind at F4 ("the module surface
    is complete; Phase F itself is not")."""
    from tensorforge.backends import cpp

    for module in _NORMALIZATION_MODULES:       # premise: the phase shipped
        assert module in cpp.NATIVE_MODULES, module

    forward = re.compile(r"Phase.F\b[^.]{0,70}?" + _PHASE_F_UNFINISHED, re.I)
    backward = re.compile(_PHASE_F_UNFINISHED + r"[^.]{0,70}?Phase.F\b", re.I)
    for surface in AUTHORITATIVE_STATUS_SURFACES + PHASE_STATUS_DOCS + (
        "CLAUDE.md", "docs/backend_experiments.md", PHASE_F_DESIGN,
    ):
        text = _status_text(surface)
        for pattern in (forward, backward):
            match = pattern.search(text)
            assert match is None, (
                f"{surface} still describes Phase F as unfinished: "
                f"{match.group(0)!r}" if match else ""
            )


def test_no_surface_places_the_native_line_on_an_advanced_branch_today():
    """The native line is merged into `main` and reached through the
    explicit `tensorforge.backends` / `tensorforge.experimental`
    namespaces, so no *present-tense* description may still confine it to a
    separate advanced branch. Historical branch references stay legal —
    only the present-tense forms are matched — and the surfaces that
    describe where the line lives must name the real namespaces."""
    import tensorforge.experimental as experimental
    from tensorforge.backends import cpp

    # Premise: those namespaces are the real, importable home.
    assert experimental.__name__ == "tensorforge.experimental"
    assert cpp.__name__ == "tensorforge.backends.cpp"

    for surface in AUTHORITATIVE_STATUS_SURFACES + PHASE_STATUS_DOCS + (
        "CLAUDE.md", "docs/backend_experiments.md",
    ):
        text = _status_text(surface)
        match = _PRESENT_TENSE_BRANCH.search(text)
        assert match is None, (
            f"{surface} still places the native line on an advanced branch: "
            f"{match.group(0)!r}" if match else ""
        )
    for surface in ("README.md", "docs/architecture.md",
                    "docs/project_summary.md"):
        text = _status_text(surface)
        for namespace in ("tensorforge.backends", "tensorforge.experimental"):
            assert namespace in text, (
                f"{surface} does not name {namespace} as where the native "
                f"line lives"
            )


def test_project_summary_testing_inventory_covers_phase_f():
    """The summary's testing section must cover every cross-cutting
    phase-integration suite that actually exists — Phase F included — and
    must not understate the suite's size. The size floor comes from the real
    test files, so a stale "Over 2000" cannot survive a far larger suite,
    while any honest larger number passes."""
    integration_suites = {
        "C": "test_native_phase_c.py",
        "D": "test_native_phase_d.py",
        "E": "test_native_phase_e.py",
        "F": "test_native_phase_f.py",
    }
    for name in integration_suites.values():
        assert (REPO_ROOT / "tests" / name).is_file(), name

    summary = _status_text("docs/project_summary.md")
    assert "Testing and reliability" in summary
    testing = summary.split("Testing and reliability", 1)[1]
    testing = testing.split("Current limitations", 1)[0]
    for phase in integration_suites:
        assert re.search(rf"Phase {phase}\b", testing), (
            f"the project summary's testing inventory omits Phase {phase}"
        )

    # Every `def test_...` in tests/ is at least one collected test, so a
    # claimed suite size below that count is stale by construction.
    floor = sum(
        len(re.findall(r"^\s*def test_\w+", path.read_text(encoding="utf-8"),
                       re.M))
        for path in (REPO_ROOT / "tests").glob("test_*.py")
    )
    assert floor > 0
    claimed = [int(number.replace(",", ""))
               for number in re.findall(r"([\d][\d,]*)\s+(?:pytest\s+)?tests\b",
                                        testing)]
    assert claimed, "the project summary states no test-suite size at all"
    for count in claimed:
        assert count >= floor, (
            f"the project summary claims {count} tests, but tests/ already "
            f"defines {floor} test functions"
        )


def test_readme_native_example_wording_matches_the_tree():
    """Every native example script must appear in the README, and any
    *counted* claim about them ("the four native examples") must match the
    real count. Uncounted, durable wording ("the native examples ... are
    listed in the native quickstart above") is deliberately allowed: the
    point is that a number can never silently contradict the tree."""
    native = sorted(path.name
                    for path in (REPO_ROOT / "examples").glob("native_*.py"))
    assert native, "no native example scripts exist"
    raw = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for name in native:
        assert name in raw, f"README does not mention examples/{name}"

    readme = _status_text("README.md")
    for match in re.finditer(
        r"\b(\w+)\s+native\s+(?:examples|demos|scripts)\b", readme, re.I
    ):
        token = match.group(1).lower()
        count = int(token) if token.isdigit() else _NUMBER_WORDS.get(token)
        if count is None:
            continue        # durable, uncounted wording — exactly what we want
        assert count == len(native), (
            f"README claims {count} native examples, but examples/ holds "
            f"{len(native)}: {native}"
        )


def test_capability_commentary_keeps_the_boundary_and_the_closed_phase():
    """`UNSUPPORTED` and the prose around it are the capability boundary's
    own record. The tuple is exactly what F4 left and F9 closed on, and its
    explanatory comment must describe a *closed* Phase F. Registry first,
    commentary second — and only comment lines are scanned, so the tuples
    themselves are never matched as prose."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint

    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.backend_info()["unsupported"] == cpp.UNSUPPORTED
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 1
    for module in _NORMALIZATION_MODULES:
        assert module in cpp.NATIVE_MODULES, module

    source = (REPO_ROOT / "src" / "tensorforge" / "backends"
              / "cpp.py").read_text(encoding="utf-8")
    commentary = " ".join(
        line.strip().lstrip("#").strip()
        for line in source.splitlines()
        if line.strip().startswith("#")
    )
    assert "UNSUPPORTED" in commentary, "the boundary lost its commentary"
    forward = re.compile(r"Phase.F\b[^.]{0,70}?" + _PHASE_F_UNFINISHED, re.I)
    backward = re.compile(_PHASE_F_UNFINISHED + r"[^.]{0,70}?Phase.F\b", re.I)
    for pattern in (forward, backward):
        match = pattern.search(commentary)
        assert match is None, (
            f"cpp.py's capability commentary still calls Phase F "
            f"unfinished: {match.group(0)!r}" if match else ""
        )


def test_the_status_reconciliation_moved_no_capability_surface():
    """A status reconciliation is documentation: it must move nothing real.
    The public exports, every capability registry, the checkpoint format,
    and normalization's module-only nature are exactly what the closed
    Phase F left."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import NativeTensor, native_checkpoint
    import tensorforge.experimental as experimental

    assert set(experimental.__all__) == {
        "NativeTensor", "NativeParameter", "NativeParameterRegistry",
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeMSELoss", "NativeSGD", "NativeAdam",
        "save_native_checkpoint", "load_native_checkpoint",
        "NativeCrossEntropyLoss", "native_accuracy",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
    }
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "save_native_checkpoint", "load_native_checkpoint",
    )
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.backend_info()["stable_framework_integration"] is False
    assert native_checkpoint._FORMAT_VERSION == 1
    # Normalization stayed module composition: no operation, Core method,
    # NativeTensor method, or guarded C ABI symbol anywhere.
    for name in _NORMALIZATION_OP_NAMES:
        assert name not in cpp.TENSOR_CORE_OPS, name
        assert name not in cpp.AUTOGRAD_OPS, name
        assert name not in cpp.RAW_KERNELS, name
        assert not hasattr(cpp.NativeTensorCore, name), name
        assert not hasattr(NativeTensor, name), name
    for symbol in ("tf_core_layer_norm", "tf_core_batch_norm",
                   "tf_core_layer_norm_forward", "tf_core_batch_norm_forward"):
        assert symbol not in cpp._CHECKED_KERNELS, symbol
