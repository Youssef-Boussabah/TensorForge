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
    # The native CNN stack (Phase D) is now under way — its Flatten and
    # Conv2d milestones have shipped — while CUDA remains not started. The
    # roadmap must keep naming the stack and keep marking CUDA future work.
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
    # milestones E3 and E4 implemented them; the Phase-E boundary is
    # tracked separately below.)
    for absent in ("float32", "cuda", "amp", "batchnorm", "layernorm",
                   "dropout", "cross_entropy"):
        assert absent in cpp.UNSUPPORTED, absent
        assert absent not in cpp.AUTOGRAD_OPS and absent not in cpp.NATIVE_MODULES
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)


def test_native_flatten_is_implemented_as_a_native_module():
    """D1: NativeFlatten is a shipped native module, present in the modern
    native-module inventory and public surface, and is NOT a raw C++
    kernel. Convolution and pooling stay unimplemented."""
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
    `log_softmax`; E5-E7 have not landed, so every later capability must
    still be absent from every implemented inventory."""
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

    # E5-E7 — still designed only.
    for name in ("cross_entropy",
                 "NativeCrossEntropyLoss", "native_accuracy"):
        assert name in cpp.UNSUPPORTED, f"{name} left the unsupported boundary"
        assert name not in cpp.TENSOR_CORE_OPS, name
        assert name not in cpp.AUTOGRAD_OPS, name
        assert name not in cpp.RAW_KERNELS, name
        assert name not in cpp.NATIVE_MODULES, name
        assert name not in cpp.NATIVE_LOSSES, name
    # No metrics inventory exists yet either — E7 introduces it.
    assert not hasattr(cpp, "NATIVE_METRICS")
    # Neither probability transform became a module (E0 §1 excludes both).
    import tensorforge.experimental as experimental
    for module in ("NativeSoftmax", "NativeLogSoftmax"):
        assert module not in cpp.NATIVE_MODULES, module
        assert not hasattr(experimental, module), module
    # E3 created the classification source unit locked by E0 §9.1, and E4
    # extended it; both forwards' kernels/exports are defined there rather
    # than in the elementwise unit. Checked by symbol definition, not by
    # banning the word, so cross-referencing comments remain possible.
    classification = (REPO_ROOT / "cpp" / "src" / "classification.cpp")
    assert classification.is_file()
    classification_text = classification.read_text(encoding="utf-8")
    for export in ("tf_core_softmax_forward", "tf_core_log_softmax_forward"):
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
    """E0-E4 are marked complete, E5-E10 are not, and Phase E itself is
    never declared complete. Checked semantically against the design
    document's status table and the live registry."""
    from tensorforge.backends import cpp

    # The ladder's status table is the one place per-milestone status is
    # declared, so the row checks run inside that section only.
    ladder = _design_section("Milestone ladder")
    for done in ("E0", "E1", "E2", "E3", "E4"):
        row = re.search(rf"\|\s*{done}\s*\|[^|]*\|([^|]*)\|", ladder)
        assert row is not None, f"the ladder has no status row for {done}"
        assert "complete" in row.group(1).lower(), (
            f"the design does not mark {done} complete"
        )
    for pending in ("E5", "E6", "E7", "E8", "E9", "E10"):
        row = re.search(rf"\|\s*{pending}\s*\|[^|]*\|([^|]*)\|", ladder)
        assert row is not None, f"the ladder has no status row for {pending}"
        assert "complete" not in row.group(1).lower(), (
            f"{pending} is marked complete but has not shipped"
        )
    # Phase E as a whole is in progress, and says so positively. (§17's
    # "Phase E is complete when ..." criteria list is a condition, not a
    # claim, so the check is on the status statement, not a banned word.)
    design = _status_text(PHASE_E_DESIGN)
    assert "Phase-E status: in progress" in design
    assert re.search(r"Phase E is [^.]{0,30}not[^.]{0,30}complete", design), (
        "the design no longer states that Phase E is not complete"
    )
    matrix = _normalized_doc("docs/native_support_matrix.md")
    assert re.search(r"Phase E[^.]{0,120}in progress", matrix, re.I), (
        "the support matrix no longer marks Phase E in progress"
    )
    # The registry agrees: exactly the E1-E4 capabilities are live and
    # the next milestone's capability is not.
    for shipped in ("exp", "log", "softmax", "log_softmax"):
        assert shipped in cpp.AUTOGRAD_OPS, shipped
        assert shipped not in cpp.UNSUPPORTED, shipped
    assert "cross_entropy" in cpp.UNSUPPORTED


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
    for future in ("cuda", "float32", "amp", "batchnorm", "layernorm",
                   "dropout"):
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


def test_native_cnn_design_is_linked_and_referenced():
    """The design doc must be reachable from the roadmap and the support
    matrix so the contract is discoverable, not orphaned."""
    roadmap = (REPO_ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    matrix = (REPO_ROOT / "docs" / "native_support_matrix.md").read_text(
        encoding="utf-8"
    )
    for doc in (roadmap, matrix):
        assert "native_cnn_design.md" in doc
