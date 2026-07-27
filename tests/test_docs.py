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
    "native_rng_dropout_design.md",
)

# The native line's phase ladder, oldest to newest. Phases A-F are
# complete; Phase G (native RNG and Dropout) is the current one and is in
# progress. Nothing after G exists.
NATIVE_PHASE_LADDER = "ABCDEFG"

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
    exemption instead of banning honest native-line statements.) The
    "native <thing>" strip is case-insensitive and keeps capitalized
    follow-words, so a phase title like "Native RNG and Dropout" counts
    as the native phrasing it is."""
    import re

    text = (REPO_ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    future = text.split("## Practical next steps", 1)[1]
    future = future.split("## What this project is not", 1)[0]
    # Strip "Native<Thing>" identifiers and "native <thing>" prose: both
    # name the native counterpart, never the shipped stable feature.
    future = re.sub(r"Native[A-Z]\w*", "", future)
    future = re.sub(r"\bnative [A-Za-z/]+(\s+[A-Za-z/]+){0,2}", "", future,
                    flags=re.IGNORECASE)
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
        "NativeGenerator",                              # Phase G, G1
        "NativeDropout",                                # Phase G, G4
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
    for absent in ("float32", "cuda", "amp"):
        assert absent in cpp.UNSUPPORTED, absent
        assert absent not in cpp.AUTOGRAD_OPS and absent not in cpp.NATIVE_MODULES
    # "dropout" is the one unsupported name that now also names a shipped
    # operation: Phase G milestone G3 added the differentiable
    # NativeTensor.dropout, while the *capability* stays unsupported until
    # the G10 closure (design §19). Neither half is a Phase-D claim, so
    # both are asserted rather than one being dropped.
    assert "dropout" in cpp.UNSUPPORTED
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert "dropout" not in cpp.NATIVE_MODULES
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
    for future in ("cuda", "amp", "float32"):
        assert future in cpp.UNSUPPORTED, future
        assert future not in cpp.AUTOGRAD_OPS and future not in cpp.TENSOR_CORE_OPS
        assert future not in cpp.NATIVE_MODULES
    # RNG/Dropout is Phase G's, not Phase E's. G2 shipped the Core wrapper
    # and G3 the differentiable operation, so the honest statement is that
    # the *capability* is still unsupported while those two layers exist.
    assert "dropout" in cpp.UNSUPPORTED
    assert "dropout" in cpp.AUTOGRAD_OPS          # G3
    assert "dropout_forward" in cpp.TENSOR_CORE_OPS   # G2
    assert "dropout" not in cpp.NATIVE_MODULES
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

    assert native_checkpoint._FORMAT_VERSION == 2
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

# The public experimental export surface exactly as Phase F closed it,
# and what has legitimately been added since. A Phase-F guard asserts
# ``PHASE_F_EXPORT_SURFACE | POST_PHASE_F_EXPORTS`` — still an exact
# equality, so nothing can slip in unnoticed, but it names *which* phase
# each addition belongs to instead of pretending the surface froze at F4.
# The full surface is also pinned once, on its own, by
# ``test_experimental_exports_stay_intentional``.
PHASE_F_EXPORT_SURFACE = frozenset({
    "NativeTensor", "NativeParameter", "NativeParameterRegistry",
    "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
    "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
    "NativeMSELoss", "NativeSGD", "NativeAdam",
    "save_native_checkpoint", "load_native_checkpoint",
    "NativeCrossEntropyLoss", "native_accuracy",
    "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
})
POST_PHASE_F_EXPORTS = frozenset({
    "NativeGenerator",   # Phase G, milestone G1 — random *state* only
    "NativeDropout",     # Phase G, milestone G4 — the Dropout module
})


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
    """Phase F added exactly three exports — ``NativeLayerNorm`` (F2),
    ``NativeBatchNorm1d`` (F3), and ``NativeBatchNorm2d`` (F4) — in both
    directions: nothing else added, nothing lost. F5-F9 added no export at
    all, so the closed phase's surface is exactly F4's.

    The claim is expressed as a set *difference* against the Phase-E
    baseline, so it keeps testing Phase F precisely while a later phase
    legitimately adds its own exports (Phase G milestone G1 adds
    ``NativeGenerator``); the full surface is pinned exactly by
    ``test_experimental_exports_stay_intentional``."""
    import tensorforge
    import tensorforge.experimental as experimental

    phase_e_surface = {
        "NativeTensor", "NativeParameter", "NativeParameterRegistry",
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeMSELoss", "NativeSGD", "NativeAdam",
        "save_native_checkpoint", "load_native_checkpoint",
        "NativeCrossEntropyLoss", "native_accuracy",
    }
    phase_f_surface = {
        "NativeLayerNorm",       # Phase F, milestone F2
        "NativeBatchNorm1d",     # Phase F, milestone F3
        "NativeBatchNorm2d",     # Phase F, milestone F4
    }
    exports = set(experimental.__all__)
    # Nothing Phase E shipped was lost...
    assert phase_e_surface <= exports
    # ...Phase F's three are all present...
    assert phase_f_surface <= exports
    # ...and Phase F added nothing else: everything beyond the Phase-E
    # baseline that is not a later phase's export is exactly those three.
    later_phase_exports = {
        "NativeGenerator",   # Phase G, milestone G1
        "NativeDropout",     # Phase G, milestone G4
    }
    assert exports - phase_e_surface - later_phase_exports == phase_f_surface
    for absent in ("NativeBatchNorm3d",):
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
        # Phase G milestone G4 appended the Dropout module. It is
        # unrelated to this milestone, which added no module of its own.
        "NativeDropout",
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


def test_state_support_reports_the_real_in_memory_state_surface():
    """The state inventory, pinned in both directions: the tuple is
    exact, every name maps to a live API, and neither capability name
    smuggled in a claim it does not own.

    Three entries are *capability* names rather than single callables —
    ``persistent_buffers`` (Phase F, F1), ``generator_state`` (Phase G,
    G1, the **in-memory** generator surface), and
    ``checkpoint_generator_state`` (Phase G, G5, the **file** half).
    ``STATE_SUPPORT`` is deliberately not an operation inventory, and the
    two generator names stay separate because G1's was explicitly scoped
    to memory: persistence arrived only at G5, with format version 2."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import NativeModule, native_checkpoint

    assert cpp.STATE_SUPPORT == (
        "persistent_buffers",
        "state_dict", "load_state_dict",
        "generator_state",   # Phase G, milestone G1 (in-memory only)
        "save_native_checkpoint", "load_native_checkpoint",
        "checkpoint_generator_state",   # Phase G, milestone G5 (the file half)
    )
    assert cpp.backend_info()["state_support"] == cpp.STATE_SUPPORT
    # Every advertised capability is real: these are the APIs behind them.
    for attribute in ("register_buffer", "buffers", "named_buffers",
                      "state_dict", "load_state_dict",
                      "register_generator", "generators", "named_generators",
                      "generator_state_dict", "load_generator_state_dict"):
        assert callable(getattr(NativeModule, attribute)), attribute
    # A state capability is not an operation, a module, or a metric.
    for inventory in (cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS, cpp.RAW_KERNELS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS, cpp.UNSUPPORTED):
        assert "persistent_buffers" not in inventory
        assert "generator_state" not in inventory
    # ...and "checkpoint_generator_state" is backed by the real format:
    # version 2, a "generators" manifest field, and both entry points.
    assert native_checkpoint._FORMAT_VERSION == 2
    assert "generators" in native_checkpoint._MANIFEST_KEYS
    assert "generators" not in native_checkpoint._MANIFEST_KEYS_V1
    assert callable(native_checkpoint.save_native_checkpoint)
    assert callable(native_checkpoint.load_native_checkpoint)
    # There is no third entry point: persistence rides the existing pair.
    public = [name for name in dir(native_checkpoint)
              if not name.startswith("_") and callable(
                  getattr(native_checkpoint, name))]
    assert "save_native_generator_state" not in public
    assert "load_native_generator_state" not in public
    # Neither generator name is a Dropout capability claim.
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")


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
    assert native_checkpoint._FORMAT_VERSION == 2

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
        assert future not in cpp.NATIVE_MODULES
    for future in ("float32", "cuda", "amp"):
        assert future not in cpp.AUTOGRAD_OPS
        assert future not in cpp.TENSOR_CORE_OPS
    # Phase F really did leave all four unsupported, and all four still
    # are. What has changed since is *below* the capability line and
    # belongs to Phase G: the G2 Core wrapper and the G3 differentiable
    # operation exist, and "dropout" is unsupported anyway until G10.
    assert "dropout_forward" in cpp.TENSOR_CORE_OPS
    assert "dropout" in cpp.AUTOGRAD_OPS
    design = _status_text(PHASE_F_DESIGN)
    for excluded in ("dropout", "CUDA", "AMP", "float32"):
        assert excluded in design, (
            f"{PHASE_F_DESIGN} does not exclude {excluded!r}"
        )
    assert cpp.backend_info()["stable_framework_integration"] is False


def test_status_docs_agree_on_the_phase_sequence():
    """Requirement 11. Every status document that discusses the native
    line must place the phases in the same order and must not invent one
    beyond the current phase.

    Phase G (native RNG and Dropout) opened with milestone G0, so naming
    it is now correct rather than an over-claim — what it may not do is
    claim a Phase-G *capability* exists, which the Phase-G guardrails
    below check against the live registry. Phase H is still invented."""
    for surface in PHASE_STATUS_DOCS:
        text = _status_text(surface)
        assert "Phase E" in text and "Phase F" in text, surface
        # Phase G is the newest phase; nothing later may be named.
        assert "Phase H" not in text, f"{surface} names Phase H"
    # The phase sequence is A..G with no gaps: the set of phases a
    # document names must be a contiguous prefix-suffix of that ladder,
    # never a set that skips one. (Ordering *within* a document is not
    # pinned — the support matrix legitimately leads with the newest
    # phase and then recaps the ladder.)
    ladder = NATIVE_PHASE_LADDER
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
        # The newest phase named must be G — no document may stop at F
        # and thereby imply Phase F is still the current phase.
        assert named[-1] == "G", (
            f"{surface} stops at Phase {named[-1]}; Phase G is current"
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
                   "native_normalization_design.md",
                   "native_rng_dropout_design.md"):
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
    assert native_checkpoint._FORMAT_VERSION == 2
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

    # Exports and every capability registry are exactly what F4/F5 left,
    # plus whatever a *later* phase has since shipped (named explicitly,
    # so this stays an exact equality rather than a loosened one).
    assert set(experimental.__all__) == (
        PHASE_F_EXPORT_SURFACE | POST_PHASE_F_EXPORTS
    )
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
        # Phase G milestone G4 appended the Dropout module. It is
        # unrelated to this milestone, which added no module of its own.
        "NativeDropout",
    )
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "generator_state",   # Phase G, milestone G1 (in-memory only)
        "save_native_checkpoint", "load_native_checkpoint",
        "checkpoint_generator_state",   # Phase G, milestone G5 (the file half)
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
    assert native_checkpoint._FORMAT_VERSION == 2

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
        # Phase G milestone G4 appended the Dropout module. It is
        # unrelated to this milestone, which added no module of its own.
        "NativeDropout",
    )
    assert native_checkpoint._FORMAT_VERSION == 2

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
    neither Phase F nor Phase G has delivered, as started, in progress,
    or complete.

    Phase G itself left this subject list when milestone G0 opened it:
    "Phase G is in progress" is now simply true. What Phase G may not
    claim is that a Phase-G *capability* exists, which
    ``test_no_surface_claims_a_phase_g_capability_exists`` checks against
    the live registry — a stronger check than a phase-name scan."""
    premise = _status_text(PHASE_F_DESIGN)
    assert "Phase-F status: complete" in premise

    started = (r"(is|are|was|were|now|has|have)\s+"
               r"(begun|started|under way|underway|in progress|complete"
               r"|completed|shipped|implemented|supported)")
    later = (r"(Phase H|CUDA (phase|runtime|"
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
    assert native_checkpoint._FORMAT_VERSION == 2
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

    assert set(experimental.__all__) == (
        PHASE_F_EXPORT_SURFACE | POST_PHASE_F_EXPORTS
    )
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
        # Phase G milestone G4 appended the Dropout module. It is
        # unrelated to this milestone, which added no module of its own.
        "NativeDropout",
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "generator_state",   # Phase G, milestone G1 (in-memory only)
        "save_native_checkpoint", "load_native_checkpoint",
        "checkpoint_generator_state",   # Phase G, milestone G5 (the file half)
    )
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.backend_info()["stable_framework_integration"] is False
    assert native_checkpoint._FORMAT_VERSION == 2
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


# --- Phase G (native RNG and Dropout) — G0 contract guardrails -----------
#
# G0 is a design-and-lock milestone: it adds no numerical behavior, no
# runtime surface, and no capability. These guards therefore establish
# two things — that the contract is written and internally coherent, and
# that **nothing it describes has been built**. Every premise is derived
# from the live registry, the live exports, the real file tree, or the
# C++ sources rather than from prose, so a guard can never assert a claim
# the code does not back.

PHASE_G_DESIGN = "docs/native_rng_dropout_design.md"

# What Phase G has actually shipped, and what it has not.
#
# Milestone G1 shipped ``NativeGenerator`` — random *state* and its module
# ownership, generating no random values. Milestone G2 shipped the
# stateless Dropout-forward **Core**: one internal derivation, one kernel,
# one guarded C ABI symbol, and the layer-qualified
# ``NativeTensorCore.dropout_forward``. Both are layer-qualified
# capabilities, and neither is the user-level Dropout, which does not
# exist.
_PHASE_G_SHIPPED_NAMES = ("NativeGenerator",)          # Phase G, G1
# The public *module* G4 shipped: exported, and in NATIVE_MODULES — and
# nowhere else, because a module is not an operation, a loss, a metric,
# an optimizer, or a kernel.
_PHASE_G_SHIPPED_MODULES = ("NativeDropout",)          # Phase G, G4
# Nothing public is unimplemented any more at this point in the ladder:
# G5 shipped checkpoint v2 and generator persistence, and G6 hardened all
# of it without adding a name. What remains absent is the G7-G10 work,
# each guarded by its own check rather than by a name.
_PHASE_G_PUBLIC_NAMES = ()
# The one Core operation G2 added, and the one C ABI symbol behind it.
_PHASE_G_SHIPPED_CORE_OPS = ("dropout_forward",)       # Phase G, G2
_PHASE_G_SHIPPED_ABI_SYMBOLS = ("tf_core_dropout_forward",)
# The one differentiable operation G3 added, on top of that Core. It is
# an AUTOGRAD_OPS entry and a NativeTensor method, and nothing else: no
# Core method, no kernel, no C ABI symbol, and no module answer to it.
_PHASE_G_SHIPPED_AUTOGRAD_OPS = ("dropout",)           # Phase G, G3
# Everything numerical that still does not exist at any layer: a backward
# kernel (there is none by design — §7.5, the gradient is the existing
# `multiply` over the saved mask) and the generic sampling API Phase G
# explicitly excludes. "dropout" left this tuple at G3.
_PHASE_G_OP_NAMES = (
    "dropout_backward",
    "random", "rand", "randn", "bernoulli", "uniform",
)
_PHASE_G_ABI_SYMBOLS = (
    "tf_core_dropout", "tf_core_dropout_backward",
    "tf_core_random", "tf_core_random_mask", "tf_core_bernoulli",
    "tf_core_uniform", "tf_core_philox", "tf_core_splitmix64",
)


def test_phase_g_design_exists_and_is_linked():
    """The contract is a real, non-empty file reachable from the README —
    an unreachable design document is an orphan, not a contract."""
    path = REPO_ROOT / PHASE_G_DESIGN
    assert path.is_file(), f"missing {PHASE_G_DESIGN}"
    assert path.read_text(encoding="utf-8").strip(), f"{PHASE_G_DESIGN} is empty"
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert PHASE_G_DESIGN in readme, f"README does not link to {PHASE_G_DESIGN}"


def test_phase_g_is_named_and_scoped():
    """Guardrail 1: Phase G exists, is named *Native RNG and Dropout*, and
    every status surface that tracks the phase sequence names it."""
    design = _status_text(PHASE_G_DESIGN)
    assert "Phase G" in design
    assert re.search(r"Native RNG and Dropout", design, re.I), (
        f"{PHASE_G_DESIGN} does not name the phase"
    )
    for surface in ("README.md", "docs/roadmap.md",
                    "docs/project_summary.md",
                    "docs/native_support_matrix.md",
                    "docs/backend_experiments.md", "CLAUDE.md"):
        text = _status_text(surface)
        assert "Phase G" in text, f"{surface} does not name Phase G"
        assert re.search(r"Native RNG and Dropout", text, re.I), surface


def test_phase_g_milestone_ladder_is_complete_and_ordered():
    """Guardrail 2: G0 through G10 each get exactly one milestone section,
    and those sections appear in increasing order. Anchored on the
    ``### G<n> —`` headings, so a milestone named in ordinary prose cannot
    perturb the check — and *exactly one* section each, so a ladder can
    never grow a duplicate milestone."""
    text = _normalized_doc(PHASE_G_DESIGN)
    positions = []
    for i in range(11):
        marker = f"### G{i} —"
        assert text.count(marker) == 1, (
            f"{PHASE_G_DESIGN} must define milestone G{i} exactly once, "
            f"found {text.count(marker)}"
        )
        positions.append(text.index(marker))
    assert positions == sorted(positions), (
        "the G0-G10 milestone ladder is out of order"
    )
    # And the ladder table records each milestone's status.
    for i in range(11):
        row = re.search(rf"\|\s*G{i}\s*\|[^|]*\|([^|]*)\|", text)
        assert row is not None, f"the ladder has no status row for G{i}"


def test_phase_g_design_locks_its_load_bearing_decisions():
    """The decisions later milestones inherit must actually be written
    down, not left to be re-derived: the generator contract, the stateless
    kernel rule, the call-consumption transaction, the probability
    boundary, the saved multiplier mask, and the checkpoint story."""
    text = _normalized_doc(PHASE_G_DESIGN)
    lowered = text.lower()
    for token in ("NativeGenerator", "NativeDropout", "NativeTensor.dropout",
                  "register_generator", "named_generators"):
        assert token in text, f"{PHASE_G_DESIGN} does not lock {token!r}"
    # The state representation: an explicit seed, counter, and algorithm id.
    for token in ("seed", "algorithm", "counter", "64-bit"):
        assert token in lowered, f"the design does not lock {token!r}"
    # The probability contract, including the rejected endpoint.
    assert re.search(r"0\s*<?=?\s*p\s*<\s*1", text), (
        "the design no longer pins the probability interval"
    )
    assert re.search(r"p\s*==\s*1[^.]{0,60}(reject|ValueError)", text, re.I), (
        "the design no longer rejects p == 1"
    )
    assert "1 / (1 - p)" in text or "1.0 / (1.0 - p)" in text or (
        "1/(1-p)" in text), "the design no longer states the inverted scale"
    # Inverted dropout, and the mask as the saved multiplier.
    assert "inverted" in lowered
    # Ownership and failure matrices exist as matrices, not as prose.
    assert re.search(r"ownership and lifecycle matrix", lowered)
    assert re.search(r"failure matrix", lowered)


def test_phase_g_design_forbids_global_rng_and_stateful_kernels():
    """Guardrails 7 and the phase's central architectural split: random
    state is Python-managed, native kernels are stateless and receive the
    whole key for one call, and no global or process-wide random state
    exists anywhere."""
    text = _normalized_doc(PHASE_G_DESIGN)
    lowered = text.lower()
    assert re.search(r"stateless", lowered), (
        "the design no longer requires stateless native kernels"
    )
    assert re.search(r"kernels? (are|remain|stay) stateless"
                     r"|stateless native (random )?kernel", lowered), (
        "the design no longer states the stateless-kernel rule directly"
    )
    assert re.search(r"python-managed", lowered), (
        "the design no longer states that random state is Python-managed"
    )
    # No global state, and no NumPy global RNG.
    assert re.search(r"no (process-)?global|global .{0,20}state[^.]{0,40}"
                     r"(forbidden|not|never)", lowered), (
        "the design no longer forbids global random state"
    )
    assert re.search(r"numpy global", lowered), (
        "the design no longer excludes the NumPy global RNG"
    )
    # The key handed across the boundary is complete and explicit.
    assert re.search(r"call[_ ]index", lowered)
    assert re.search(r"no .{0,40}(std::random_device|mt19937)", lowered) or (
        "std::random_device" in text and "mt19937" in lowered
    ), "the design no longer excludes the standard-library RNG sources"


def test_phase_g_design_locks_the_call_consumption_transaction():
    """Guardrail 8, the invariant the whole resume story rests on: one
    successful stochastic forward consumes exactly one generator call, and
    failures, evaluation mode, ``p == 0``, and backward consume none."""
    text = _normalized_doc(PHASE_G_DESIGN)
    lowered = text.lower()
    assert re.search(r"exactly one (generator )?call", lowered), (
        "the design no longer states the one-call-per-success rule"
    )
    # Each no-consumption case is named somewhere in the contract.
    for case in ("evaluation", "backward", "validation", "allocation",
                 "kernel", "graph"):
        assert case in lowered, f"the transaction omits the {case!r} case"
    assert re.search(r"p\s*==\s*0", text), (
        "the design no longer covers the p == 0 case"
    )
    # The transaction boundary is named, not implied.
    assert re.search(r"reserv", lowered) and re.search(r"commit", lowered)
    assert re.search(r"abandon", lowered)


def test_phase_g_design_locks_the_saved_mask_backward():
    """Guardrail 9: Dropout's backward uses a saved **private** multiplier
    mask — never a reread of the input, never the generator — and the mask
    is graph-owned and released exactly once."""
    text = _normalized_doc(PHASE_G_DESIGN)
    lowered = text.lower()
    assert "mask" in lowered
    assert re.search(r"multiplier mask", lowered), (
        "the design no longer calls the saved state a multiplier mask"
    )
    assert re.search(r"graph[- ]owned", lowered)
    assert "graph_resources" in text, (
        "the design no longer reuses the existing graph-resource contract"
    )
    assert re.search(r"released exactly once|exactly once", lowered)
    assert re.search(r"never rereads? the input|does not reread", lowered), (
        "the design no longer forbids rereading the input in backward"
    )
    assert re.search(r"private", lowered)
    # The mask must not become public state of any kind.
    assert re.search(r"never (a )?public|never a parameter or buffer"
                     r"|never in a state_dict", lowered), (
        "the design no longer keeps the mask out of the public surface"
    )


def test_phase_g_design_locks_the_checkpoint_versioning_contract():
    """Guardrail 10: the version-2 extension and, more importantly, the
    version-1 compatibility rule — a v1 archive stays loadable into a
    model with no generators, fails loudly into one that has them, and no
    seed or counter is ever fabricated."""
    from tensorforge.experimental import native_checkpoint

    text = _normalized_doc(PHASE_G_DESIGN)
    lowered = text.lower()
    # The format name never moves; only the version does, and only later.
    assert native_checkpoint._FORMAT in text, (
        "the design no longer pins the unchanged checkpoint format name"
    )
    assert re.search(r"version 2", lowered)
    assert re.search(r"version 1", lowered)
    assert re.search(r"(version 1|v1)[^.]{0,120}(loadable|remains loadable)",
                     lowered), (
        "the design no longer defines version-1 compatibility"
    )
    assert re.search(r"fabricat|invent", lowered), (
        "the design no longer forbids fabricating a seed or counter"
    )
    # The generator section and its exact fields.
    assert '"generators"' in text or "generators" in lowered
    for field in ("algorithm", "algorithm_version", "seed", "calls"):
        assert field in text, f"the design does not lock the {field!r} field"
    # And the version really has not moved yet.
    assert native_checkpoint._FORMAT_VERSION == 2


def test_phase_g_design_states_the_stable_native_separation_and_non_goals():
    """Guardrail 11: stable/native separation is explicit, and float32,
    CUDA, and AMP stay non-goals."""
    text = _normalized_doc(PHASE_G_DESIGN)
    lowered = text.lower()
    assert re.search(r"stable\s*/\s*native separation", lowered), (
        "the design has no stable/native separation section"
    )
    assert "tensorforge.nn.Dropout" in text, (
        "the design does not say what happens to the stable Dropout"
    )
    for phrase in ("no automatic backend dispatch", "implicit"):
        assert phrase.lower() in lowered, phrase
    # Non-goals, stated as non-goals.
    assert re.search(r"non-goal", lowered)
    for excluded in ("float32", "CUDA", "AMP", "Dropout2d",
                     "stochastic depth"):
        assert excluded.lower() in lowered, (
            f"{PHASE_G_DESIGN} does not exclude {excluded!r}"
        )


def test_phase_g_runtime_surface_is_generator_state_and_the_g2_core():
    """Guardrails 4, 5, 6, 12, and 13, all derived from reality.

    Phase G has shipped exactly three things: milestone G1's
    ``NativeGenerator`` and its ``NativeModule`` registration — random
    *state* — milestone G2's stateless Dropout-forward **Core**, which is
    one internal derivation, one kernel, one guarded C ABI symbol, and the
    layer-qualified ``NativeTensorCore.dropout_forward``, and milestone
    G3's differentiable ``NativeTensor.dropout``, which is one
    ``AUTOGRAD_OPS`` entry and one method over that Core. So the
    dtype/device boundary, the checkpoint format, and every
    module/loss/metric/optimizer inventory are still exactly what Phase F
    closed with; ``"dropout"`` is still in ``UNSUPPORTED``, deliberately,
    until the G10 closure; the generator export appears in **no**
    inventory at all; the Core op appears in ``TENSOR_CORE_OPS`` and
    nowhere else; and the autograd op appears in ``AUTOGRAD_OPS`` and
    nowhere else. (Before G1 this guard asserted ``NativeGenerator`` did
    not exist; before G2 it asserted the same of every random symbol;
    before G3, of the differentiable operation. Each absence check is
    replaced, as its milestone lands, by the stronger positive statement
    of what the thing is *allowed* to be.)"""
    from tensorforge.backends import cpp
    from tensorforge.experimental import NativeTensor, native_checkpoint
    import tensorforge
    import tensorforge.experimental as experimental

    # The capability boundary is untouched.
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.backend_info()["unsupported"] == cpp.UNSUPPORTED
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.backend_info()["stable_framework_integration"] is False

    # The checkpoint format has not moved.
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 2

    # G1 shipped: exported from the experimental namespace only, and in
    # no capability inventory — a generator is state, not an operation,
    # a module, a loss, a metric, an optimizer, or a kernel.
    for name in _PHASE_G_SHIPPED_NAMES:
        assert name in experimental.__all__, name
        assert hasattr(experimental, name), name
        assert not hasattr(tensorforge, name), (
            f"{name} leaked into the stable top-level namespace"
        )
        for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                          cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                          cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                          cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
                          cpp.UNSUPPORTED):
            assert name not in inventory, (name, inventory)

    # G2 shipped: exactly one Core operation, in exactly one inventory,
    # reachable as a NativeTensorCore method and nowhere above it.
    for name in _PHASE_G_SHIPPED_CORE_OPS:
        assert name in cpp.TENSOR_CORE_OPS, name
        assert hasattr(cpp.NativeTensorCore, name), name
        for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                          cpp.AUTOGRAD_OPS, cpp.NATIVE_MODULES,
                          cpp.NATIVE_LOSSES, cpp.NATIVE_METRICS,
                          cpp.NATIVE_OPTIMIZERS, cpp.STATE_SUPPORT,
                          cpp.UNSUPPORTED):
            assert name not in inventory, (name, inventory)
        assert not hasattr(NativeTensor, name), name
        assert not hasattr(experimental, name), name
    for symbol in _PHASE_G_SHIPPED_ABI_SYMBOLS:
        assert symbol in cpp._CHECKED_KERNELS, symbol

    # G3 shipped: exactly one differentiable operation, in exactly one
    # operation inventory, reachable as a NativeTensor method — and
    # nowhere else. It added no Core method, no kernel, and no C ABI
    # symbol of its own, because inverted Dropout's gradient is the
    # existing `multiply` over the saved mask (design §7.5).
    for name in _PHASE_G_SHIPPED_AUTOGRAD_OPS:
        assert name in cpp.AUTOGRAD_OPS, name
        assert hasattr(NativeTensor, name), name
        for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                          cpp.TENSOR_CORE_OPS, cpp.NATIVE_MODULES,
                          cpp.NATIVE_LOSSES, cpp.NATIVE_METRICS,
                          cpp.NATIVE_OPTIMIZERS, cpp.STATE_SUPPORT):
            assert name not in inventory, (name, inventory)
        assert not hasattr(cpp.NativeTensorCore, name), name
        assert not hasattr(experimental, name), name
        assert f"tf_core_{name}" not in cpp._CHECKED_KERNELS, name
        # ...and the *capability* by the same name is still unsupported:
        # G3 shipped an operation, not a closed capability (design §19).
        assert name in cpp.UNSUPPORTED, name

    # G4 shipped: exactly one module, exported from the experimental
    # namespace only, in NATIVE_MODULES and in no other inventory — a
    # module is not an operation, a loss, a metric, an optimizer, or a
    # kernel, and it did not become a capability either.
    for name in _PHASE_G_SHIPPED_MODULES:
        assert name in experimental.__all__, name
        assert hasattr(experimental, name), name
        assert not hasattr(tensorforge, name), (
            f"{name} leaked into the stable top-level namespace"
        )
        assert name in cpp.NATIVE_MODULES, name
        for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                          cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                          cpp.NATIVE_LOSSES, cpp.NATIVE_METRICS,
                          cpp.NATIVE_OPTIMIZERS, cpp.STATE_SUPPORT,
                          cpp.UNSUPPORTED):
            assert name not in inventory, (name, inventory)

    # Not shipped: nothing public beyond G4's remains unimplemented.
    for name in _PHASE_G_PUBLIC_NAMES:
        assert name not in experimental.__all__, name
        assert not hasattr(experimental, name), name
        assert not hasattr(tensorforge, name), name
        for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                          cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                          cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                          cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
                          cpp.STATE_SUPPORT):
            assert name not in inventory, (name, inventory)

    # No *later* Phase-G operation exists at any numerical layer, and no
    # Core or NativeTensor method answers to one.
    for name in _PHASE_G_OP_NAMES:
        assert name not in cpp.TENSOR_CORE_OPS, name
        assert name not in cpp.AUTOGRAD_OPS, name
        assert name not in cpp.RAW_KERNELS, name
        assert name not in cpp.TENSOR_CORE_KERNELS, name
        assert not hasattr(cpp.NativeTensorCore, name), name
        assert not hasattr(NativeTensor, name), name
        assert not hasattr(experimental, name), name
    # "dropout" is in exactly two places and no others: AUTOGRAD_OPS (the
    # G3 operation) and UNSUPPORTED (the capability, until G10).
    assert "dropout" in cpp.UNSUPPORTED
    assert "dropout" in cpp.AUTOGRAD_OPS
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                      cpp.TENSOR_CORE_OPS, cpp.NATIVE_MODULES,
                      cpp.NATIVE_LOSSES, cpp.NATIVE_METRICS,
                      cpp.NATIVE_OPTIMIZERS, cpp.STATE_SUPPORT):
        assert "dropout" not in inventory, inventory

    # No later Phase-G C ABI symbol is declared, guarded, or defined.
    for symbol in _PHASE_G_ABI_SYMBOLS:
        assert symbol not in cpp._CHECKED_KERNELS, symbol
    cpp_source = (REPO_ROOT / "src" / "tensorforge" / "backends"
                  / "cpp.py").read_text(encoding="utf-8")
    for symbol in _PHASE_G_ABI_SYMBOLS:
        assert f"{symbol}.argtypes" not in cpp_source, symbol
    # G2's random unit is the ONLY C++ source that may define a Phase-G
    # symbol, and it defines exactly the one.
    for source in (REPO_ROOT / "cpp" / "src").glob("*.cpp"):
        text = source.read_text(encoding="utf-8")
        if source.name == "random.cpp":
            assert text.count("TF_EXPORT") == 1, source.name
            continue
        for symbol in ("tf_core_dropout", "tf_core_random"):
            assert symbol not in text, f"{source.name} defines {symbol!r}"
    # G2's source unit and header exist; nothing above the Core does.
    for present in ("cpp/src/random.cpp", "cpp/include/tf_random_internal.h",
                    "cpp/tests/test_dropout_forward.cpp"):
        assert (REPO_ROOT / present).is_file(), (
            f"{present} is missing, but milestone G2 shipped it"
        )
    # G4's module unit exists; nothing above it does.
    assert (REPO_ROOT / "src" / "tensorforge" / "experimental"
            / "native_dropout.py").is_file(), (
        "milestone G4 shipped the NativeDropout module"
    )
    # G7's example and G8's benchmark exist, with their tests; nothing
    # above them does.
    for present in ("examples/native_dropout_training.py",
                    "tests/test_native_dropout_training.py",
                    "benchmarks/benchmark_native_dropout.py",
                    "tests/test_native_dropout_benchmark.py"):
        assert (REPO_ROOT / present).is_file(), (
            f"{present} is missing, but milestone G7 or G8 shipped it"
        )
    for absent in ("cpp/src/dropout.cpp",
                   "tests/test_native_phase_g.py"):
        assert not (REPO_ROOT / absent).exists(), (
            f"{absent} exists, but no Phase-G milestone has shipped it"
        )
    generator_module = (REPO_ROOT / "src" / "tensorforge" / "experimental"
                        / "native_generator.py")
    assert generator_module.is_file(), "milestone G1 shipped NativeGenerator"
    # ...and it really is pure Python state: no numerical derivation, no
    # NumPy, and no global or process-wide entropy source anywhere in it.
    generator_source = generator_module.read_text(encoding="utf-8")
    for forbidden in ("import numpy", "numpy.random", "np.random",
                      "import random", "random.getrandbits", "time.time",
                      "os.getpid",
                      # the SplitMix64 derivation itself (G2's work): the
                      # golden ratio and the two finalizer multipliers,
                      # and any call to a mixing function
                      "0x9E3779B97F4A7C15", "0xBF58476D1CE4E5B9",
                      "0x94D049BB133111EB", "mix64("):
        assert forbidden not in generator_source, (
            f"native_generator.py references {forbidden!r}; G1 holds state "
            f"and generates no random values"
        )

    # Generator registration is the G1 capability, and only that: the
    # module gained the four generator APIs and no Dropout surface.
    from tensorforge.experimental import NativeModule
    for attribute in ("register_generator", "generators", "named_generators",
                      "generator_state_dict", "load_generator_state_dict"):
        assert hasattr(NativeModule, attribute), attribute
    for attribute in ("dropout", "register_dropout"):
        assert not hasattr(NativeModule, attribute), attribute


def test_no_surface_claims_a_phase_g_capability_exists():
    """Guardrail 3, and the replacement for the phase-name scan the
    Phase-F closure guard used to run: naming Phase G is now correct, but
    no surface may claim a Phase-G *capability* is supported, implemented,
    shipped, or available. The premise comes from the live registry, so
    this guard tracks reality rather than a frozen expectation.

    Subjects have been deliberately **un**-banned as their milestones
    landed, each time leaving a smaller and sharper list.
    ``NativeGenerator`` left at G1; the bare "native RNG"/"native
    random"/"native dropout" phrasings left at G2, because the stateless
    Dropout-forward Core really does exist; ``NativeTensor.dropout``,
    "differentiable dropout", "dropout autograd", and "dropout backward"
    left at **G3**, which shipped exactly those; and ``NativeDropout``
    itself left at **G4**, which shipped and exported the module.

    Every *surface* Phase G contracts now exists, so what stays banned is
    no longer a name but the **capability claim**: nothing may say the
    Dropout capability is supported, or that stochastic training resumes
    exactly. Both are false at G4 — the checkpoint format is version 1
    and does not persist generator state — and both are exactly what a
    reader would take "native Dropout is supported now" to mean. The
    separate ``test_g4_module_claims_are_layer_qualified`` guard then
    requires each status surface to say so positively."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint
    import tensorforge.experimental as experimental

    # Premise, from the live registry: the operation and the module
    # exist, the capability is not closed, and nothing is persisted.
    assert "dropout" in cpp.UNSUPPORTED
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert "NativeDropout" in cpp.NATIVE_MODULES
    assert native_checkpoint._FORMAT_VERSION == 2
    for name in _PHASE_G_PUBLIC_NAMES:
        assert not hasattr(experimental, name), name

    subject = (r"(dropout[\w ]{0,20}capability|capability[\w ]{0,20}dropout"
               r"|exact stochastic resume|stochastic resume"
               r"|exact (?:random|RNG) resume)")
    claim = (r"(is|are|now|has been|have been)\s+"
             r"(supported|implemented|shipped|available|complete|completed"
             r"|live)")
    # A span that carries its own negation is the honest form.
    negations = re.compile(
        r"\b(not|never|neither|nor|nothing|none|no|planned|unsupported"
        r"|future|beyond|until|once|when|will|would)\b", re.I,
    )
    patterns = (
        re.compile(subject + r"[^.]{0,70}?" + claim, re.I),
        re.compile(claim + r"[^.]{0,70}?" + subject, re.I),
    )
    for surface in AUTHORITATIVE_STATUS_SURFACES + PHASE_STATUS_DOCS + (
        "CLAUDE.md", "docs/backend_experiments.md", PHASE_G_DESIGN,
    ):
        text = _status_text(surface)
        for pattern in patterns:
            offenders = [
                match.group(0) for match in pattern.finditer(text)
                if not negations.search(
                    text[max(0, match.start() - 50):match.end() + 20]
                )
            ]
            assert offenders == [], (
                f"{surface} claims a Phase-G capability exists: "
                f"{offenders[:3]}"
            )


def test_phase_g_is_presented_as_in_progress_not_complete():
    """The mirror of the Phase-F completion guard: no surface may present
    Phase G — or a milestone that has not shipped — as finished.

    The milestone boundary is derived from the ladder table rather than
    from a pinned sentence, so this guard survives each milestone landing
    while still catching a phase declared complete early or a milestone
    marked done before it exists. The *reality* check that the ladder
    matches the tree lives in
    ``test_phase_g_ladder_status_matches_the_shipped_tree``."""
    design = _status_text(PHASE_G_DESIGN)
    assert re.search(r"Phase-G status: in progress", design), (
        "the design document no longer states its in-progress status"
    )
    # Whatever has shipped, the tail of the ladder must still be open —
    # G10 is the closure milestone, so it can never be complete while the
    # phase says "in progress".
    assert re.search(r"not started", design, re.I), (
        "the design no longer marks any milestone as unstarted, but the "
        "phase still claims to be in progress"
    )
    finished = re.compile(
        r"Phase.G\b[^.]{0,70}?\b(is|are|was|has been)\s+"
        r"(complete|completed|finished|closed|shipped)\b", re.I,
    )
    # "Phase G — in progress: milestone G0 ... is complete" is the honest
    # form: the completion belongs to a milestone, not to the phase. A
    # span that names a milestone or says "in progress" is therefore not
    # a phase-completion claim.
    scoped = re.compile(r"in progress|\bG\d\b|milestone", re.I)
    for surface in ("README.md", "docs/roadmap.md",
                    "docs/project_summary.md",
                    "docs/native_support_matrix.md",
                    "docs/backend_experiments.md", "CLAUDE.md",
                    PHASE_G_DESIGN):
        text = _status_text(surface)
        offenders = [
            match.group(0) for match in finished.finditer(text)
            if not scoped.search(match.group(0))
        ]
        assert offenders == [], (
            f"{surface} presents Phase G as complete: {offenders[:3]}"
        )
        # ...and each one says what G0 actually was.
        assert re.search(r"G0", text), f"{surface} does not name milestone G0"


def test_phase_g_ladder_status_matches_the_shipped_tree():
    """The ladder's per-milestone status is checked against **reality**,
    not against prose: a milestone is marked Complete only if the thing
    it ships actually exists, and Not started only if it does not.

    This is what keeps the ladder from drifting in either direction — a
    milestone quietly marked done, or a shipped milestone still listed as
    pending. Each entry names the observable that decides it."""
    import tensorforge.experimental as experimental
    from tensorforge.backends import cpp
    from tensorforge.experimental import NativeModule, native_checkpoint

    ladder = _top_level_section("Milestone ladder", PHASE_G_DESIGN)

    def status(index):
        row = re.search(rf"\|\s*G{index}\s*\|[^|]*\|([^|]*)\|", ladder)
        assert row is not None, f"the ladder has no status row for G{index}"
        return re.sub(r"[*`]", "", row.group(1)).strip().lower()

    # G0 is documentation; it has shipped by definition once the design
    # exists, which test_phase_g_design_exists_and_is_linked proves.
    assert status(0) == "complete"

    # G1 ships NativeGenerator and module generator registration.
    g1_shipped = (
        hasattr(experimental, "NativeGenerator")
        and "NativeGenerator" in experimental.__all__
        and all(hasattr(NativeModule, name) for name in
                ("register_generator", "generators", "named_generators",
                 "generator_state_dict", "load_generator_state_dict"))
    )
    assert (status(1) == "complete") is g1_shipped, (
        f"the ladder says G1 is {status(1)!r} but the tree "
        f"{'has' if g1_shipped else 'does not have'} NativeGenerator"
    )

    # G2 ships the stateless Core: the derivation, the kernel, the guarded
    # C ABI symbol, and the layer-qualified Core method.
    g2_shipped = (
        "dropout_forward" in cpp.TENSOR_CORE_OPS
        and hasattr(cpp.NativeTensorCore, "dropout_forward")
        and "tf_core_dropout_forward" in cpp._CHECKED_KERNELS
        and (REPO_ROOT / "cpp" / "src" / "random.cpp").is_file()
    )
    assert (status(2) == "complete") is g2_shipped, (
        f"the ladder says G2 is {status(2)!r} but the tree "
        f"{'has' if g2_shipped else 'does not have'} the Dropout Core"
    )

    # G3 ships the differentiable operation on top of it.
    g3_shipped = (
        "dropout" in cpp.AUTOGRAD_OPS
        or hasattr(experimental.NativeTensor, "dropout")
    )
    if not g3_shipped:
        assert status(3) == "not started", (
            f"the ladder marks G3 as {status(3)!r}, but no differentiable "
            f"dropout operation exists"
        )

    # G4 ships the module and its export.
    g4_shipped = (
        hasattr(experimental, "NativeDropout")
        and "NativeDropout" in experimental.__all__
        and "NativeDropout" in cpp.NATIVE_MODULES
    )
    assert (status(4) == "complete") is g4_shipped, (
        f"the ladder says G4 is {status(4)!r} but the tree "
        f"{'has' if g4_shipped else 'does not have'} NativeDropout"
    )
    # G5 moves the checkpoint format.
    if native_checkpoint._FORMAT_VERSION == 1:
        assert status(5) == "not started", (
            "the ladder marks G5 complete, but the checkpoint format is "
            "still version 1"
        )
    # G6 ships tests only, so its observable is the hardening suite itself.
    g6_shipped = (
        REPO_ROOT / "tests" / "test_native_phase_g_hardening.py"
    ).is_file()
    assert status(6).startswith("complete") is g6_shipped, (
        f"the ladder says G6 is {status(6)!r} but the tree "
        f"{'has' if g6_shipped else 'does not have'} the hardening suite"
    )
    # G7 ships one example and its test module.
    g7_shipped = (
        (REPO_ROOT / "examples" / "native_dropout_training.py").is_file()
        and (REPO_ROOT / "tests"
             / "test_native_dropout_training.py").is_file()
    )
    assert status(7).startswith("complete") is g7_shipped, (
        f"the ladder says G7 is {status(7)!r} but the tree "
        f"{'has' if g7_shipped else 'does not have'} the resume example"
    )
    # G8 ships one benchmark harness and its test module.
    g8_shipped = (
        (REPO_ROOT / "benchmarks" / "benchmark_native_dropout.py").is_file()
        and (REPO_ROOT / "tests"
             / "test_native_dropout_benchmark.py").is_file()
    )
    assert status(8).startswith("complete") is g8_shipped, (
        f"the ladder says G8 is {status(8)!r} but the tree "
        f"{'has' if g8_shipped else 'does not have'} the benchmark"
    )
    # G9 ships the cross-cutting integration suite.
    if not (REPO_ROOT / "tests" / "test_native_phase_g.py").is_file():
        assert status(9) == "not started", (
            f"the ladder marks G9 as {status(9)!r}, but no Phase-G "
            f"integration suite exists"
        )
    # The closure milestone is open while "dropout" is still unsupported.
    if "dropout" in cpp.UNSUPPORTED:
        assert status(10) == "not started"


# The claims Phase G must never let a status surface make at its current
# milestone. G1 shipped random *state*, G2 the stateless Core, G3 the
# differentiable operation over it, and G4 the module over that; the
# **persistence** and every later milestone are still unbuilt, and prose
# is the easiest place for that distinction to erode.
#
# Three entries have been retired as their milestones landed, recorded
# here rather than silently dropped:
#   * "random values are generated" (retired at G2) — G2's kernel really
#     does generate them. What must still not be claimed is that a
#     *module-level* Dropout exists, which
#     ``test_no_surface_claims_a_phase_g_capability_exists`` owns.
#   * the unqualified "a mask exists" (retired at G2) — G2's Core really
#     does produce the private multiplier mask.
#   * "a graph-owned saved mask exists" (retired at **G3**) — G3 really
#     does adopt that mask into ``graph_resources`` and differentiate
#     through it. The claim that replaces it is narrower and is the one
#     G3 must not let erode: that the mask is *persisted*, which it is
#     not and will not be (design §8.3 — it is never a public tensor,
#     never a parameter or buffer, never in a state_dict or checkpoint).
_PHASE_G_OVERCLAIMS = (
    ("the saved mask is persisted",
     r"mask\w*[^.]{0,60}\b(is|are)\s+"
     r"(saved to|persisted|serialized|checkpointed|written to)"
     r"|(state_dict|checkpoint)[^.]{0,60}\b(includes?|contains?|holds?"
     r"|stores?)\b[^.]{0,30}mask"),
    # Retired at G5: checkpoint format version 2 and persisted generator
    # state are now real, so claiming them is accurate, not an overclaim.
    # What replaces them is the *next* boundary — the end-to-end resume.
    # Scoped to *stochastic* resume: Phase F's normalized training resume
    # (F6) is a different, shipped proof and must keep reading as one.
    # ...and this one was retired at **G7**: the end-to-end exact
    # stochastic *training* resume is now demonstrated by
    # examples/native_dropout_training.py, so claiming it is accurate.
    # What replaces it is narrower and is the claim G7 must not let
    # erode: that the checkpoint captures the *external* loop state — a
    # data loader, a shuffle order, an epoch counter, a scheduler, or a
    # global RNG — which it does not and will not (design §11.1). The
    # loop position is carried as explicit metadata instead.
    ("the checkpoint captures external loop or global RNG state",
     r"(checkpoint|archive|resume|save)\w*[^.]{0,70}"
     r"\b(data.?loader|shuffle\w*|epoch counter|scheduler state"
     r"|global rng|global random)\b[^.]{0,40}"
     r"\b(captured?|stored?|saved?|persisted?|restored?|included?)\b"),
    # Retired one milestone at a time: G6 (hardening), G7 (the
    # end-to-end resume), and now G8 (the honest benchmark
    # characterization) really have landed, so claiming them is accurate.
    # G9 is now the boundary a status surface must not cross.
    ("a later milestone has begun",
     r"\bG(?:9|10)\b[^.]{0,60}\b(is|are|has|have)\s+"
     r"(complete|completed|started|begun|shipped|landed|done)\b"),
)


def test_no_surface_overclaims_what_phase_g_has_shipped():
    """G1 shipped random **state**, G2 the stateless Dropout-forward
    **Core**, G3 the differentiable operation that adopts the Core's mask
    as graph-owned state, G4 the module over it, and G5 the version-2
    checkpoint that persists generator state. None of the five persists a
    *mask*, demonstrates end-to-end exact stochastic training resume, or
    starts a later milestone.

    (The "checkpoint version 2 exists" and "generator state is
    checkpointed" patterns were retired at G5 — both are now accurate.
    What replaces them is the next boundary: the §11 end-to-end resume,
    which is G7.)

    Every premise below comes from the live tree, so this guard tracks
    reality; the prose scan then holds documentation to it. Spans that
    carry their own negation ("does not generate", "not persisted",
    "G8-G10 have not started") are the honest form and pass."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import (
        NativeGenerator, NativeTensor, native_checkpoint,
    )
    import tensorforge.experimental as experimental

    # Premises, all from reality: the operation and the module exist, the
    # capability is not closed, and nothing is persisted.
    assert hasattr(NativeTensor, "dropout")
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert hasattr(experimental, "NativeDropout")
    assert "NativeDropout" in cpp.NATIVE_MODULES
    assert "dropout" in cpp.UNSUPPORTED
    assert native_checkpoint._FORMAT_VERSION == 2
    # The generator really does not produce a value of any kind: G2's
    # derivation lives in C++ behind the Core, never on this object.
    generator = NativeGenerator(1)
    for numerical in ("random", "rand", "randn", "bits", "next", "uniform",
                      "bernoulli", "mask", "dropout"):
        assert not hasattr(generator, numerical), numerical

    negations = re.compile(
        r"\b(not|never|neither|nor|nothing|none|no|without|planned|future"
        r"|beyond|until|once|when|will|would|before|yet)\b", re.I,
    )
    for surface in ("README.md", "docs/roadmap.md",
                    "docs/project_summary.md",
                    "docs/native_support_matrix.md",
                    "docs/backend_experiments.md", "CLAUDE.md",
                    "src/tensorforge/experimental/__init__.py",
                    PHASE_G_DESIGN):
        text = _status_text(surface)
        for label, pattern in _PHASE_G_OVERCLAIMS:
            offenders = [
                match.group(0)
                for match in re.finditer(pattern, text, re.I)
                if not negations.search(
                    text[max(0, match.start() - 60):match.end() + 25]
                )
            ]
            assert offenders == [], (
                f"{surface} claims {label}: {offenders[:3]}"
            )


def test_one_shared_state_transaction_guard_exists_and_is_outermost():
    """G5's serializability guarantee, taken from the live runtime rather
    than from prose: **one** private reentrant guard, shared by every
    participating state-replacement path, always acquired before any
    generator lock.

    Atomic is not serializable — two concurrent loads could each be
    all-or-nothing and still leave a state assembled from both — so the
    existence, the identity, and the *order* of this guard are all part of
    the contract (§10.8)."""
    import threading

    from tensorforge.experimental import (
        _native_checkpoint_transaction, _native_state, _native_state_lock,
        native_adam, native_checkpoint, native_generator, native_sgd,
    )
    import tensorforge
    import tensorforge.experimental as experimental

    guard = _native_state_lock.state_transaction()
    # One object, and it really is a reentrant lock.
    assert guard is _native_state_lock._STATE_TRANSACTION_LOCK
    assert isinstance(guard, type(threading.RLock()))
    with guard:
        assert _native_state_lock.held_by_current_thread()
        with guard:                      # reentrant, as the nesting needs
            assert _native_state_lock.held_by_current_thread()
    assert not _native_state_lock.held_by_current_thread()

    # Every participant shares that one object — not a lock each.
    for module in (_native_state, native_generator, native_adam, native_sgd,
                   native_checkpoint, _native_checkpoint_transaction):
        assert module.state_transaction() is guard, module.__name__

    # Private: not exported, not reachable at top level.
    assert "_native_state_lock" not in experimental.__all__
    for absent in ("state_transaction", "held_by_current_thread",
                   "STATE_TRANSACTION_LOCK"):
        assert absent not in experimental.__all__, absent
        assert not hasattr(tensorforge, absent), absent

    # The order is item 1 then item 2, enforced where the generator locks
    # are actually taken rather than in each caller.
    source = (REPO_ROOT / "src" / "tensorforge" / "experimental"
              / "native_generator.py").read_text(encoding="utf-8")
    body = source[source.index("def locked_generators("):]
    body = body[:body.index("\ndef ", 10)]
    assert body.index("with state_transaction():") < body.index(
        "_ordered_targets("
    ), "generator lock ordering is decided outside the shared guard"
    assert body.index("with state_transaction():") < body.index(
        "enter_context(generator._lock)"
    ), "a generator lock is entered before the shared guard"
    # Reservations deliberately stay out of it — that asymmetry is what
    # keeps the two systems from inverting.
    reserve = source[source.index("    def _reserve_call(self):"):]
    reserve = reserve[:reserve.index("\n    def ", 10)]
    assert "state_transaction" not in reserve, (
        "reservations must not take the shared guard"
    )


def test_status_surfaces_state_the_serializability_contract():
    """Every Phase-G status surface must say the same three things about
    concurrency: one shared guard, that concurrent loads **serialize**
    rather than merely avoiding deadlock, and — the honest limit — that
    ordinary training mutation does not participate, so thread-safe
    concurrent training snapshots are not claimed."""
    from tensorforge.experimental import _native_state_lock

    # Premise, from the live runtime.
    assert _native_state_lock.state_transaction() is (
        _native_state_lock._STATE_TRANSACTION_LOCK
    )

    for surface in PHASE_G_STATUS_SURFACES:
        text = _status_text(surface)
        # Anchored on the distinctive tokens of *this* claim, so an
        # unrelated sentence about the reservation RLock or about
        # deterministic serialization order cannot satisfy it.
        assert re.search(
            r"(shared|state[- ]transaction|one private|process-wide)"
            r".{0,120}?RLock"
            r"|RLock.{0,120}?(shared|state[- ]transaction|guard)",
            text, re.I,
        ), f"{surface} does not describe the shared state-transaction guard"
        assert re.search(
            r"concurrent (checkpoint )?(loads?|operations|state)"
            r".{0,260}?(serializ|followed by the other"
            r"|rather than a mixture|never a mixture)", text, re.I,
        ), f"{surface} does not say concurrent loads serialize"
        assert re.search(
            r"generator lock.{0,70}?id\(\)"
            r"|generator.{0,90}?id\(\).{0,40}?order", text, re.I,
        ), f"{surface} does not state the generator lock order"
        # The limit is stated wherever the guarantee is.
        assert re.search(
            r"(training|step\(\)|copy_value_).{0,160}?"
            r"(not |never |does not |deliberately )"
            r"|(not |never |no ).{0,160}?thread-safe"
            r"|thread-safe.{0,120}?(not |never )", text, re.I,
        ), (
            f"{surface} does not state that ordinary training mutation is "
            f"outside the guard"
        )


def test_no_surface_claims_thread_safe_concurrent_training():
    """The one overclaim this milestone makes newly tempting. A save that
    overlaps a concurrent ``step()`` can still capture a torn training
    state, because the step never takes the guard — so no surface may
    describe the native line as safe for concurrent training."""
    claim = re.compile(
        r"thread[- ]safe[^.]{0,60}(training|concurrent training)"
        r"|(concurrent|parallel)[^.]{0,40}training[^.]{0,60}"
        r"\b(is|are)\s+(safe|supported|thread-safe)"
        r"|safe[^.]{0,40}(to|for)\s+(train|checkpoint)[^.]{0,40}"
        r"(concurrently|in parallel|from (another|several) threads?)",
        re.I,
    )
    negations = re.compile(
        r"\b(not|never|no|without|deliberately|does not|is not|are not"
        r"|cannot)\b", re.I,
    )
    for surface in PHASE_G_STATUS_SURFACES + (
        "src/tensorforge/experimental/__init__.py",
    ):
        text = _status_text(surface)
        offenders = [
            match.group(0) for match in claim.finditer(text)
            if not negations.search(
                text[max(0, match.start() - 90):match.end() + 40]
            )
        ]
        assert offenders == [], (
            f"{surface} claims thread-safe concurrent training: "
            f"{offenders[:3]}"
        )


def test_every_participating_loader_takes_the_shared_guard():
    """Derived from the live sources, not from a list in prose: each
    documented participant really acquires the guard, and none of them
    invented a lock of its own."""
    experimental_dir = (
        REPO_ROOT / "src" / "tensorforge" / "experimental"
    )
    participants = {
        "_native_state.py": "replace_native_state",
        "native_generator.py": "replace_generator_states",
        "native_sgd.py": "load_state_dict",
        "native_adam.py": "load_state_dict",
        "native_checkpoint.py": "save_native_checkpoint",
        "_native_checkpoint_transaction.py": "commit_checkpoint",
    }
    for name, symbol in participants.items():
        text = (experimental_dir / name).read_text(encoding="utf-8")
        assert "from ._native_state_lock import state_transaction" in text, (
            f"{name} does not import the shared guard"
        )
        assert "with state_transaction():" in text, (
            f"{name} never enters the shared guard"
        )
        assert symbol in text, (name, symbol)
    # Nobody built a second lock: the guard module is the only place in
    # the experimental package that constructs one for state replacement.
    owners = []
    for module_path in experimental_dir.glob("*.py"):
        text = module_path.read_text(encoding="utf-8")
        if "threading.RLock()" in text or "threading.Lock()" in text:
            owners.append(module_path.name)
    assert sorted(owners) == ["_native_state_lock.py", "native_generator.py"], (
        f"an unexpected module constructs its own lock: {owners} — the "
        f"only two are the shared guard and each generator's own lock"
    )


def test_the_two_generator_capability_names_stay_distinct():
    """``STATE_SUPPORT`` carries **two** generator names, and conflating
    them is exactly the reporting error G1 was careful to avoid.

    ``"generator_state"`` (G1) reports the **in-memory** surface —
    registration, traversal, and the state-dict pair — and still means
    only that. ``"checkpoint_generator_state"`` (G5) reports the **file**
    surface: a native checkpoint really does persist and restore
    generator state, through the existing save/load pair and the
    version-2 manifest, with no third entry point. Keeping them separate
    is what lets a reader tell a G1 model from a G5 one.

    Every premise comes from the live tree."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import NativeModule, native_checkpoint

    assert "generator_state" in cpp.STATE_SUPPORT
    assert "checkpoint_generator_state" in cpp.STATE_SUPPORT
    assert cpp.STATE_SUPPORT.index("generator_state") == (
        cpp.STATE_SUPPORT.index("load_state_dict") + 1
    ), "generator_state should sit with the other in-memory state names"
    assert cpp.STATE_SUPPORT.index("checkpoint_generator_state") > (
        cpp.STATE_SUPPORT.index("load_native_checkpoint")
    ), "the file half should sit with the checkpoint names"
    for api in ("register_generator", "generators", "named_generators",
                "generator_state_dict", "load_generator_state_dict"):
        assert callable(getattr(NativeModule, api)), api
    # The file half is backed by the real format, not by a new API.
    assert native_checkpoint._FORMAT_VERSION == 2
    assert "generators" in native_checkpoint._MANIFEST_KEYS
    assert "generators" not in native_checkpoint._MANIFEST_KEYS_V1
    assert callable(native_checkpoint.save_native_checkpoint)
    assert callable(native_checkpoint.load_native_checkpoint)

    # The registry module itself must explain both names, and must still
    # say the G1 one is in-memory only.
    registry_source = (REPO_ROOT / "src" / "tensorforge" / "backends"
                       / "cpp.py").read_text(encoding="utf-8")
    marker = registry_source.index("STATE_SUPPORT = (")
    commentary = registry_source[max(0, marker - 4000):marker].lower()
    for required in ("generator_state", "checkpoint_generator_state",
                     "in-memory", "g5"):
        assert required in commentary, (
            f"the registry commentary no longer says {required!r} about "
            f"the generator state names"
        )

    # No status surface may claim persistence existed before G5, or that
    # a version-1 archive carries generator state.
    backdated = re.compile(
        r"version 1[^.]{0,80}(persist|serializ|carr\w+|save\w*)"
        r"[^.]{0,40}generator"
        r"|generator[^.]{0,60}(persist\w*|serializ\w*|checkpointed)"
        r"[^.]{0,40}(version 1|G[0-4]\b)",
        re.I,
    )
    negations = re.compile(
        r"\b(not|never|no|without|until|cannot|does not|is not|are not"
        r"|omit\w*|absent|fabricat\w*)\b", re.I
    )
    for surface in ("README.md", "docs/roadmap.md",
                    "docs/project_summary.md",
                    "docs/native_support_matrix.md",
                    "docs/backend_experiments.md", "CLAUDE.md",
                    PHASE_G_DESIGN):
        text = _status_text(surface)
        offenders = [
            match.group(0) for match in backdated.finditer(text)
            if not negations.search(
                text[max(0, match.start() - 80):match.end() + 40]
            )
        ]
        assert offenders == [], (
            f"{surface} backdates generator persistence before G5: "
            f"{offenders[:3]}"
        )


def test_g1_status_is_stated_positively_on_every_phase_surface():
    """The mirror of the guard above: G1 really did ship, so a surface
    that tracks Phase G must say so rather than still describing the
    phase as design-only. Paired with the overclaim scan, this pins the
    status from both sides."""
    for surface in ("README.md", "docs/roadmap.md",
                    "docs/project_summary.md",
                    "docs/native_support_matrix.md",
                    "docs/backend_experiments.md", "CLAUDE.md",
                    PHASE_G_DESIGN):
        text = _status_text(surface)
        assert re.search(r"\bG1\b", text), (
            f"{surface} does not name milestone G1"
        )
        assert "NativeGenerator" in text, (
            f"{surface} does not mention what G1 shipped"
        )
        # ...and none of them still says G1 is pending. The scan is
        # anchored so that an unstarted-range claim that *begins* after
        # G1 ("G2-G10 have not started") reads as the honest statement it
        # is: the second alternative refuses to cross another milestone
        # token, and the first catches a range that includes G1.
        still_pending = re.compile(
            r"\bG1\s*[-–—]\s*G\d+\b[^.]{0,30}?\bnot\s+(?:yet\s+)?started"
            r"|\bG1\b(?:(?!\bG\d)[^.]){0,40}?\bnot\s+(?:yet\s+)?started",
            re.I,
        )
        match = still_pending.search(text)
        assert match is None, (
            f"{surface} still describes G1 as unstarted: {match.group(0)!r}"
        )


# --- Phase G milestone G2 — the stateless Dropout-forward Core ----------
#
# G2 guardrails, all derived from the live registry, the real C++/Python
# sources, and the file tree. They pin what the milestone shipped **and**
# the boundaries it deliberately did not cross.

# The surfaces that track Phase-G status milestone by milestone.
PHASE_G_STATUS_SURFACES = (
    "README.md",
    "docs/roadmap.md",
    "docs/project_summary.md",
    "docs/native_support_matrix.md",
    "docs/backend_experiments.md",
    "CLAUDE.md",
    PHASE_G_DESIGN,
)


def test_g2_core_inventory_is_exactly_one_operation_and_one_abi_symbol():
    """The milestone's whole registry footprint: ``TENSOR_CORE_OPS`` gains
    ``"dropout_forward"`` and ``_CHECKED_KERNELS`` gains
    ``"tf_core_dropout_forward"``. Nothing else moved — not
    ``AUTOGRAD_OPS``, not ``NATIVE_MODULES``, not ``STATE_SUPPORT``, not
    ``UNSUPPORTED``, not the dtype/device tuples, not the checkpoint
    version."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint

    # Exactly one new Core op, appended last, and no sibling smuggled in.
    dropout_ops = [name for name in cpp.TENSOR_CORE_OPS if "dropout" in name]
    assert dropout_ops == ["dropout_forward"], dropout_ops
    assert cpp.TENSOR_CORE_OPS[-1] == "dropout_forward"

    # Exactly one new C ABI symbol, and it is checked (so native failures
    # become Python exceptions rather than silent wrong results).
    dropout_symbols = [name for name in cpp._CHECKED_KERNELS
                       if "dropout" in name or "random" in name]
    assert dropout_symbols == ["tf_core_dropout_forward"], dropout_symbols

    # Everything else Phase F closed with, unchanged.
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.RAW_KERNELS == (
        "elementwise_add", "elementwise_subtract", "elementwise_multiply",
        "elementwise_divide", "relu", "matmul", "matmul_tiled",
    )
    assert cpp.TENSOR_CORE_KERNELS == (
        "relu", "add", "subtract", "multiply", "matmul",
    )
    # G2 itself added nothing to AUTOGRAD_OPS: "cross_entropy" was still
    # the last entry when it landed. The one entry after it is G3's
    # differentiable "dropout", which is a *later* milestone's footprint
    # and is asserted by the G3 guard below.
    assert cpp.AUTOGRAD_OPS[-2] == "cross_entropy"
    assert cpp.AUTOGRAD_OPS[-1] == "dropout"
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
        # Phase G milestone G4 appended the Dropout module. It is
        # unrelated to this milestone, which added no module of its own.
        "NativeDropout",
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "generator_state",
        "save_native_checkpoint", "load_native_checkpoint",
        "checkpoint_generator_state",
    )
    assert native_checkpoint._FORMAT_VERSION == 2
    # backend_info() mirrors the live registry, so the report cannot drift.
    info = cpp.backend_info()
    assert tuple(info["tensor_core_ops"]) == cpp.TENSOR_CORE_OPS
    assert tuple(info["unsupported"]) == cpp.UNSUPPORTED


def test_g2_core_boundary_is_stateless_and_takes_the_whole_key():
    """The phase's central architectural split, checked in the code rather
    than in the prose: the Core takes an explicit ``(seed, call_index)``
    pair, the ctypes declaration carries both as ``c_uint64``, and neither
    the backend module nor the C++ unit can reach a generator."""
    import inspect

    from tensorforge.backends import cpp

    for name in ("dropout_forward", "_dropout_forward_with_mask"):
        signature = inspect.signature(getattr(cpp.NativeTensorCore, name))
        parameters = signature.parameters
        assert list(parameters) == ["self", "p", "seed", "call_index"], name
        for keyword_only in ("seed", "call_index"):
            assert (parameters[keyword_only].kind
                    is inspect.Parameter.KEYWORD_ONLY), (name, keyword_only)
            assert parameters[keyword_only].default is inspect.Parameter.empty
        # No generator parameter of any spelling.
        assert "generator" not in parameters, name

    backend_source = (REPO_ROOT / "src" / "tensorforge" / "backends"
                      / "cpp.py").read_text(encoding="utf-8")
    # The declaration carries the full unsigned 64-bit key.
    assert "tf_core_dropout_forward.argtypes" in backend_source
    assert backend_source.count("ctypes.c_uint64") == 2
    # backends/ never imports experimental/, so a generator is not even
    # reachable from the Core layer. (Checked structurally: the module's
    # imports and its live namespace, not its prose — the Core's docstring
    # legitimately explains *why* it holds no generator.)
    assert "import tensorforge.experimental" not in backend_source
    assert "from tensorforge.experimental" not in backend_source
    assert not hasattr(cpp, "NativeGenerator")
    leaked = [name for name in vars(cpp)
              if "generator" in name.lower() and name != "STATE_SUPPORT"]
    assert leaked == [], leaked

    # And the C++ side holds no random state at all.
    for relative in ("cpp/src/random.cpp", "cpp/include/tf_random_internal.h"):
        code = "\n".join(
            line.split("//")[0]
            for line in (REPO_ROOT / relative).read_text(
                encoding="utf-8"
            ).splitlines()
        )
        for forbidden in ("<random>", "random_device", "mt19937",
                          "thread_local", "static ", "srand", "std::time"):
            assert forbidden not in code, (relative, forbidden)


def test_g2_equality_threshold_vector_is_committed_on_both_sides():
    """The comparison direction is pinned by exactly one vector, and it
    must not quietly disappear from either suite.

    The premise is taken from the **live kernel**, not from prose: the
    Core is run at ``p == u`` and at ``nextafter(u, 1.0)`` for the
    committed word, and the keep/drop flip is observed. Then both test
    files are checked to still commit the same constants, so a future edit
    cannot delete the proof from one side and leave the other looking
    complete. (Inspecting the vectors directly is the point — no prose is
    matched.)"""
    import math

    import numpy as np
    import pytest

    from tensorforge.backends import cpp

    if not cpp.is_available():
        pytest.skip("experimental C++ backend not built")

    seed = 0x0123456789ABCDEF
    call_index = 0
    index = 2
    word = 0xA2A1796FEB7EF314
    uniform = float.fromhex("0x1.4542f2dfd6fdep-1")
    assert (word >> 11) * 2.0 ** -53 == uniform
    assert 0.0 < uniform < 1.0

    values = np.array([1.5, -2.25, 3.75, -4.5])
    masks = {}
    for label, probability in (("equal", uniform),
                               ("next", math.nextafter(uniform, 1.0))):
        source = cpp.NativeTensorCore.from_array(values)
        try:
            out, mask = source._dropout_forward_with_mask(
                probability, seed=seed, call_index=call_index
            )
            try:
                masks[label] = mask.to_numpy().copy()
                assert np.array_equal(out.to_numpy(),
                                      values * masks[label])
            finally:
                out.close()
                mask.close()
        finally:
            source.close()

    # The live behaviour: equality keeps, one ULP more drops.
    assert masks["equal"][index] == 1.0 / (1.0 - uniform)
    assert masks["next"][index] == 0.0
    assert not np.array_equal(masks["equal"], masks["next"])

    # ...and both suites still commit the vector that proves it.
    native = (REPO_ROOT / "cpp" / "tests"
              / "test_dropout_forward.cpp").read_text(encoding="utf-8")
    python = (REPO_ROOT / "tests"
              / "test_native_dropout_core.py").read_text(encoding="utf-8")
    for source_name, text in (("test_dropout_forward.cpp", native),
                              ("test_native_dropout_core.py", python)):
        for literal in ("0xA2A1796FEB7EF314", "0x1.4542f2dfd6fdep-1",
                        "nextafter", '"0010"', '"0000"'):
            assert literal in text, (source_name, literal)


def test_g7_did_not_begin_g8_or_any_later_milestone():
    """The milestone boundary from the tree: G3 shipped the operation, G4
    the module, G5 the version-2 checkpoint that persists generator state,
    G6 hardened all of it without adding a capability, and G7 proved the
    end-to-end exact stochastic resume with one example — and nothing
    above that. No benchmark and no Phase-G integration suite.

    (This guard was ``test_g2_did_not_begin_g3_...``, then
    ``test_g3_did_not_begin_g4_...``, then ``test_g5_did_not_begin_g6_...``,
    then ``test_g6_did_not_begin_g7_...``. The Core-layer half of it — that the
    Core method builds no graph and takes no generator — is kept verbatim
    at each step, because building a graph and then a module *above* the
    Core must not leak any of that vocabulary *into* it.)"""
    from pathlib import Path

    from tensorforge.backends import cpp
    from tensorforge.experimental import (
        NativeDropout, NativeTensor, native_checkpoint,
    )
    import tensorforge.experimental as experimental

    # G3 shipped the operation and G4 the module over it...
    assert hasattr(NativeTensor, "dropout")
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert hasattr(experimental, "NativeDropout")
    assert "NativeDropout" in cpp.NATIVE_MODULES
    # ...and G5 then persisted the stream: format version 2, with the
    # generator section that version-1 archives will never have.
    assert native_checkpoint._FORMAT_VERSION == 2
    assert "generators" in native_checkpoint._MANIFEST_KEYS
    assert "generators" not in native_checkpoint._MANIFEST_KEYS_V1
    # No Dropout variants, and the module is a *user* of the fourth
    # registration category, never an extension of it.
    for absent in ("NativeDropout2d", "NativeDropout3d", "NativeAlphaDropout"):
        assert not hasattr(experimental, absent), absent
        assert absent not in cpp.NATIVE_MODULES, absent
    from tensorforge.experimental import NativeModule
    for attribute in ("dropout", "register_dropout"):
        assert not hasattr(NativeModule, attribute), attribute
    # The module delegates rather than reimplementing: its forward
    # reaches no Core method, no reservation primitive, and no kernel.
    import inspect

    forward = inspect.getsource(NativeDropout.forward)
    for forbidden in ("_dropout_forward_with_mask", "dropout_forward",
                      "_reserve_call", "_commit_call", "_abandon_call",
                      "graph_resources", "np.random"):
        assert forbidden not in forward, forbidden
    # The Core method still builds no graph: no autograd vocabulary
    # reaches it, even though a graph is now built one layer above.
    import inspect

    source = inspect.getsource(
        cpp.NativeTensorCore._dropout_forward_with_mask
    )
    for graph_token in ("_from_op", "graph_resources", "requires_grad",
                        "_backward", "NativeTensor("):
        assert graph_token not in source, graph_token
    # ...and the Core still takes no generator, so G3's transaction stayed
    # entirely in the Python operation layer.
    core_signature = inspect.signature(
        cpp.NativeTensorCore._dropout_forward_with_mask
    )
    assert "generator" not in core_signature.parameters
    # G4's module unit exists and has its own focused suite, and G6's
    # hardening suite exists...
    for present in ("src/tensorforge/experimental/native_dropout.py",
                    "tests/test_native_dropout_module.py",
                    "tests/test_native_phase_g_hardening.py"):
        assert (Path(REPO_ROOT) / present).is_file(), (
            f"{present} is missing, but an earlier milestone shipped it"
        )
    # ...and none of the later milestones' artifacts exists. The G9
    # integration suite in particular is a *different* file from G6's
    # hardening suite. (G8's benchmark now exists; what must not is a
    # committed *result* artifact from it.)
    for absent in ("tests/test_native_phase_g.py", "benchmark_results"):
        assert not (Path(REPO_ROOT) / absent).exists(), absent
    # G6 is hardening only: it added no export, no inventory entry, and no
    # schema field, so the whole public surface is still exactly G5's.
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2)
    assert native_checkpoint._GENERATOR_SECTION_KEYS == {
        "keys", "entries", "aliases"
    }


def test_g2_status_is_stated_positively_on_every_phase_surface():
    """The mirror of the overclaim scan: G2 really did ship, so a surface
    that tracks Phase G must name the milestone and say what it delivered
    — using the layer-qualified name, so a reader cannot mistake a Core
    wrapper for a user-level Dropout."""
    for surface in PHASE_G_STATUS_SURFACES:
        text = _status_text(surface)
        assert re.search(r"\bG2\b", text), (
            f"{surface} does not name milestone G2"
        )
        assert "dropout_forward" in text, (
            f"{surface} does not name the layer-qualified Core operation "
            f"milestone G2 shipped"
        )
        # ...and none of them still describes G2 as pending. Anchored so
        # that "G3-G10 have not started" reads as the honest statement it
        # is (the second alternative refuses to cross another milestone
        # token, and the first catches a range that includes G2).
        still_pending = re.compile(
            r"\bG[0-2]\s*[-–—]\s*G\d+\b[^.]{0,30}?\bnot\s+(?:yet\s+)?started"
            r"|\bG2\b(?:(?!\bG\d)[^.]){0,40}?\bnot\s+(?:yet\s+)?started",
            re.I,
        )
        match = still_pending.search(text)
        assert match is None, (
            f"{surface} still describes G2 as unstarted: {match.group(0)!r}"
        )


def test_g6_is_stated_as_hardening_only_on_every_phase_surface():
    """G6 really did land, so every Phase-G surface must name it and say
    what it was: **hardening**, adding no capability.

    The positive requirement matters as much as the negative one. A
    milestone that ships only tests is the easiest kind to either forget
    (leaving the ladder stale) or overstate (reading as though Dropout were
    now closed), so each surface has to name G6, describe it as hardening,
    and still carry the unchanged capability boundary."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint

    # Premises from the live tree: the suite exists, and nothing moved.
    assert (REPO_ROOT / "tests"
            / "test_native_phase_g_hardening.py").is_file()
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert native_checkpoint._FORMAT_VERSION == 2

    for surface in PHASE_G_STATUS_SURFACES:
        text = _status_text(surface)
        assert re.search(r"\bG6\b", text), (
            f"{surface} does not name milestone G6"
        )
        assert re.search(r"harden", text, re.I), (
            f"{surface} names G6 without saying it was a hardening milestone"
        )
        # ...and none of them still describes G6 as pending. Anchored the
        # same way the G2 guard is, so "G7-G10 have not started" reads as
        # the honest statement it is.
        still_pending = re.compile(
            r"\bG[0-6]\s*[-–—]\s*G\d+\b[^.]{0,30}?"
            r"\bnot\s+(?:yet\s+)?started"
            r"|\bG6\b(?:(?!\bG\d)[^.]){0,40}?\bnot\s+(?:yet\s+)?started",
            re.I,
        )
        match = still_pending.search(text)
        assert match is None, (
            f"{surface} still describes G6 as unstarted: {match.group(0)!r}"
        )


def test_g7_resume_claims_are_paired_with_what_is_not_captured():
    """G7 really did demonstrate exact stochastic training resume, so
    every Phase-G surface must name it — **and**, in the same document,
    say what a checkpoint does not capture.

    That pairing is the whole point. "TensorForge resumes stochastic
    training exactly" is true and useful; read without "...for the state
    actually captured, which is not a data loader, a shuffle order, a
    scheduler, or a global RNG", it is a promise the format does not
    make."""
    assert (REPO_ROOT / "examples" / "native_dropout_training.py").is_file()
    assert (REPO_ROOT / "tests"
            / "test_native_dropout_training.py").is_file()

    for surface in PHASE_G_STATUS_SURFACES:
        text = _status_text(surface)
        assert re.search(r"\bG7\b", text), (
            f"{surface} does not name milestone G7"
        )
        assert "native_dropout_training" in text, (
            f"{surface} does not name the G7 example"
        )
        lowered = text.lower()
        # The honest limit, stated somewhere in the same document.
        assert any(term in lowered for term in
                   ("data-loader", "data loader", "dataloader")), (
            f"{surface} claims a stochastic resume without saying a data "
            f"loader is not captured"
        )
        assert "shuffle" in lowered, surface
        assert any(term in lowered for term in
                   ("global rng", "global random", "numpy's global",
                    "numpy global")), surface
        # ...and none of them still describes G7 as pending.
        still_pending = re.compile(
            r"\bG[0-7]\s*[-–—]\s*G\d+\b[^.]{0,30}?"
            r"\bnot\s+(?:yet\s+)?started"
            r"|\bG7\b(?:(?!\bG\d)[^.]){0,40}?\bnot\s+(?:yet\s+)?started",
            re.I,
        )
        match = still_pending.search(text)
        assert match is None, (
            f"{surface} still describes G7 as unstarted: {match.group(0)!r}"
        )


def test_g7_added_no_capability_and_no_public_training_api():
    """One example and its tests. Nothing about the runtime moved, and the
    example's helpers are not a new framework surface."""
    import tensorforge
    import tensorforge.experimental as experimental
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint

    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert native_checkpoint._FORMAT_VERSION == 2
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2)
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    # The example defines helpers; none of them is exported anywhere.
    for absent in ("NativeDropoutClassifier", "run_training",
                   "run_resume_proof", "run_next_mask_proof", "train_step",
                   "build_model", "batch_index_for_step",
                   "progress_metadata", "validated_progress"):
        assert not hasattr(experimental, absent), absent
        assert not hasattr(tensorforge, absent), absent
        assert absent not in cpp.NATIVE_MODULES, absent
    # No integration suite and no result artifact. (G7 shipped no
    # benchmark; G8's exists and writes nothing unless asked.)
    for absent in ("tests/test_native_phase_g.py", "benchmark_results"):
        assert not (REPO_ROOT / absent).exists(), absent


def test_the_g7_example_carries_all_four_state_families_and_no_timing():
    """Derived from the example file itself: the model the resume proof
    rests on really does contain Dropout, both normalization families,
    cross-entropy, and NativeAdam — and the example asserts nothing about
    speed."""
    example = (REPO_ROOT / "examples"
               / "native_dropout_training.py").read_text(encoding="utf-8")
    for required in ("NativeDropout", "NativeBatchNorm1d", "NativeLayerNorm",
                     "NativeCrossEntropyLoss", "NativeAdam", "NativeLinear",
                     "NativeReLU", "save_native_checkpoint",
                     "load_native_checkpoint"):
        assert required in example, required
    # Explicit loop progress, validated rather than defaulted.
    for required in ("training_step", "next_batch_index",
                     "validated_progress", "batch_index_for_step"):
        assert required in example, required
    # No timing, no benchmark vocabulary, no stable-framework import.
    for banned in ("perf_counter", "time.time(", "import time",
                   "BENCHMARK_NAME", "speedup", "tensorforge.nn",
                   "tensorforge.optim", "np.random", "import random"):
        assert banned not in example, banned


def test_g6_claims_no_new_capability_anywhere():
    """The one thing G6 must never be described as: a capability. Every
    surface that names it has to keep the boundary where it is, and the live
    registries have to agree."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint

    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert native_checkpoint._FORMAT_VERSION == 2
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2)
    # G6 shipped no example and no integration suite. (The benchmark
    # arrived at G8, two milestones later, and is guarded there.)
    for absent in ("tests/test_native_phase_g.py",):
        assert not (REPO_ROOT / absent).exists(), absent

    overclaim = re.compile(
        r"\bG6\b[^.]{0,80}\b(ships?|shipped|adds?|added|introduc\w+)\b"
        r"[^.]{0,40}\b(capabilit\w+|operation|module|export|kernel"
        r"|benchmark|example)\b",
        re.I,
    )
    for surface in PHASE_G_STATUS_SURFACES:
        text = _status_text(surface)
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            match = overclaim.search(sentence)
            if match is None:
                continue
            # A sentence carrying its own negation is the honest form.
            assert re.search(r"\b(no|not|never|without|nothing)\b",
                             sentence, re.I), (
                f"{surface} claims G6 shipped a capability: "
                f"{match.group(0)!r}"
            )


def test_g5_persistence_claims_are_layer_qualified():
    """A surface that describes the shipped ``NativeDropout`` and its
    now-persisted stream must also say, in the same document, what is
    *still* missing: end-to-end exact stochastic **training** resume is a
    G7 deliverable, and the capability is still unsupported. That pairing
    is what keeps "native Dropout resumes exactly now" from being a fair
    reading of any status page.

    (Until G3 this guard required the differentiable operation to be named
    as absent, until G4 the module, and until G5 the persistence gap. Each
    requirement was retired by the milestone that shipped the thing; what
    replaces it each time is a stronger *positive* requirement — here,
    that the version-2 format and its alias topology are described
    wherever the module is, together with the remaining gap.)"""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint

    # Premise, from the live registry.
    assert "dropout_forward" in cpp.TENSOR_CORE_OPS
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert "NativeDropout" in cpp.NATIVE_MODULES
    assert "dropout" in cpp.UNSUPPORTED
    assert native_checkpoint._FORMAT_VERSION == 2
    assert "checkpoint_generator_state" in cpp.STATE_SUPPORT

    for surface in PHASE_G_STATUS_SURFACES:
        text = _status_text(surface)
        assert re.search(r"\bCore\b", text), surface
        # The module is named, and so is the operation beneath it with its
        # explicit generator, so a reader cannot mistake either for an
        # implicit global stream.
        assert "NativeDropout" in text, (
            f"{surface} does not name the G4 module"
        )
        assert re.search(
            r"NativeTensor\.dropout|dropout\(p", text,
        ), f"{surface} does not name the G3 operation"
        assert re.search(
            r"explicit[^.]{0,80}NativeGenerator"
            r"|NativeGenerator[^.]{0,80}(explicit|required|keyword)"
            r"|generator=", text, re.I,
        ), f"{surface} does not say the generator is explicit"
        # The shipped format is named by version, not merely implied.
        assert re.search(
            r"(format )?version 2|version 2 |v2\b", text, re.I,
        ), f"{surface} does not name checkpoint format version 2"
        # ...and it says what version 2 actually carries, topology
        # included — restoring states without the topology would resume a
        # different model.
        assert re.search(
            r"(alias|topolog|shar\w+)[^.]{0,140}generator"
            r"|generator[^.]{0,140}(alias|topolog)", text, re.I,
        ), f"{surface} does not describe the generator alias topology"
        # The remaining gap is stated in the same document: G5 restored
        # the state, G7 owns the end-to-end training resume.
        assert re.search(
            r"\bG7\b[^.]{0,140}resume"
            r"|resume[^.]{0,140}\bG7\b"
            r"|resume[^.]{0,80}(not yet|is not|does not|still)"
            r"|(not yet|still)[^.]{0,80}resume", text, re.I,
        ), f"{surface} does not say end-to-end exact resume is still G7"
        # ...and "dropout" is still advertised as unsupported.
        assert re.search(r"unsupported|UNSUPPORTED", text), surface


def test_g4_inventory_is_exactly_one_module():
    """The milestone's whole registry footprint: ``NATIVE_MODULES`` gains
    ``"NativeDropout"``, appended last, and the experimental exports gain
    the same name. Nothing else moved — not ``AUTOGRAD_OPS``, not
    ``TENSOR_CORE_OPS``, not ``_CHECKED_KERNELS``, not ``STATE_SUPPORT``,
    not ``UNSUPPORTED``, not the dtype/device tuples, and not the
    checkpoint version."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint
    import tensorforge
    import tensorforge.experimental as experimental

    assert cpp.NATIVE_MODULES[-1] == "NativeDropout"
    assert cpp.NATIVE_MODULES.count("NativeDropout") == 1
    assert "NativeDropout" in experimental.__all__
    assert experimental.__all__.count("NativeDropout") == 1
    assert not hasattr(tensorforge, "NativeDropout")

    # G4 added no operation, kernel, or C ABI symbol of its own.
    assert "NativeDropout" not in cpp.AUTOGRAD_OPS
    assert "NativeDropout" not in cpp.TENSOR_CORE_OPS
    dropout_core_ops = [name for name in cpp.TENSOR_CORE_OPS
                        if "dropout" in name or "random" in name]
    assert dropout_core_ops == ["dropout_forward"], dropout_core_ops
    dropout_symbols = [name for name in cpp._CHECKED_KERNELS
                       if "dropout" in name or "random" in name]
    assert dropout_symbols == ["tf_core_dropout_forward"], dropout_symbols
    dropout_autograd_ops = [name for name in cpp.AUTOGRAD_OPS
                            if "dropout" in name]
    assert dropout_autograd_ops == ["dropout"], dropout_autograd_ops

    # Everything else exactly as G3 left it.
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "generator_state",
        "save_native_checkpoint", "load_native_checkpoint",
        "checkpoint_generator_state",
    )
    assert native_checkpoint._FORMAT_VERSION == 2
    assert tuple(cpp.backend_info()["native_modules"]) == cpp.NATIVE_MODULES


def test_g4_module_semantics_are_what_the_contract_locks():
    """The behavioral guardrails, taken from the live runtime rather than
    from prose: the locked signature, mutually-exclusive seed/generator,
    generator ownership versus sharing, registration as the fourth state
    category (and absence from ``state_dict()``), training delegation,
    evaluation identity that consumes nothing, ``p == 0`` identity, and a
    stream with no gap across a mode switch."""
    import inspect

    import numpy as np
    import pytest

    from tensorforge.backends import cpp
    from tensorforge.experimental import (
        NativeDropout, NativeGenerator, NativeTensor,
    )

    if not cpp.is_available():
        pytest.skip("experimental C++ backend not built")

    # The exact locked signature.
    parameters = inspect.signature(NativeDropout.__init__).parameters
    assert list(parameters) == ["self", "p", "seed", "generator"]
    assert parameters["p"].default == 0.5
    assert parameters["seed"].default is None
    assert parameters["generator"].default is None

    # seed and generator are mutually exclusive.
    supplied = NativeGenerator(11)
    with pytest.raises(TypeError):
        NativeDropout(0.5, seed=1, generator=supplied)
    assert supplied.calls == 0 and supplied.seed == 11

    # An explicit generator is the exact object; the default is owned.
    shared = NativeDropout(0.5, generator=supplied)
    assert shared.generator is supplied
    owned = NativeDropout(0.5, seed=13)
    assert owned.generator is not supplied
    assert owned.generator.seed == 13

    # Ownership is expressed by identity and registration, never by a
    # stored flag (§19/G4): neither construction path leaves a public
    # `owns_generator` attribute, which would go stale the moment a
    # created generator is shared with a second module.
    for module in (shared, owned):
        assert not hasattr(module, "owns_generator")
        assert not hasattr(type(module), "owns_generator")

    # Registered as the fourth state category, and nowhere else.
    assert [name for name, _ in owned.named_generators()] == ["generator"]
    assert set(owned.generator_state_dict()) == {"generator"}
    assert owned.state_dict() == {}
    assert list(owned.parameters()) == []
    assert list(owned.named_buffers()) == []
    assert [name for name, _ in owned.named_modules()] == [""]

    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)

    # Training consumes exactly one call and produces a fresh tensor.
    assert owned.training is True
    first = owned(x)
    assert first is not x
    assert owned.generator.calls == 1

    # Evaluation is the input object itself and consumes nothing.
    owned.eval()
    for _ in range(3):
        assert owned(x) is x
    assert owned.generator.calls == 1

    # Returning to training leaves no gap: the next index, checked
    # against the Core rather than against "these two look different".
    owned.train()
    second = owned(x)
    source = cpp.NativeTensorCore.from_array(values)
    reference = source.dropout_forward(0.5, seed=13, call_index=1)
    assert np.array_equal(second.to_numpy(), reference.to_numpy())
    reference.close()
    source.close()
    assert owned.generator.calls == 2

    # p == 0 is identity in both modes and consumes nothing.
    identity = NativeDropout(0.0, seed=17)
    assert identity(x) is x
    identity.eval()
    assert identity(x) is x
    assert identity.generator.calls == 0

    for tensor in (first, second, x):
        tensor.close()

    # The module delegates rather than reimplementing, and reaches for no
    # global stream. (The docstring is stripped first — it legitimately
    # says "no process-global stream" — by splitting on the docstring
    # delimiters rather than on __doc__, which inspect.getsource does not
    # reproduce byte-for-byte under a non-UTF-8 locale.)
    source_text = inspect.getsource(NativeDropout.forward)
    opening, _docstring, body = source_text.split('"""', 2)
    assert "def forward" in opening
    assert "dropout(" in body, "forward must delegate to the operation"
    for forbidden in ("np.random", "numpy.random", "import random",
                      "secrets.", "default_generator", "global ",
                      "_reserve_call", "_commit_call", "_abandon_call",
                      "_dropout_forward_with_mask", "graph_resources"):
        assert forbidden not in body, forbidden


def test_g3_inventory_is_exactly_one_autograd_operation():
    """The milestone's whole registry footprint: ``AUTOGRAD_OPS`` gains
    ``"dropout"``, appended last. Nothing else moved — not
    ``TENSOR_CORE_OPS``, not ``_CHECKED_KERNELS``, not ``NATIVE_MODULES``,
    not ``STATE_SUPPORT``, not ``UNSUPPORTED``, not the dtype/device
    tuples, and not the checkpoint version."""
    from tensorforge.backends import cpp
    from tensorforge.experimental import native_checkpoint

    assert cpp.AUTOGRAD_OPS[-1] == "dropout"
    assert cpp.AUTOGRAD_OPS.count("dropout") == 1
    # G3 added no Core op, no kernel, and no C ABI symbol: the Dropout
    # gradient is the existing `multiply` over the saved mask (§7.5).
    dropout_core_ops = [name for name in cpp.TENSOR_CORE_OPS
                        if "dropout" in name or "random" in name]
    assert dropout_core_ops == ["dropout_forward"], dropout_core_ops
    dropout_symbols = [name for name in cpp._CHECKED_KERNELS
                       if "dropout" in name or "random" in name]
    assert dropout_symbols == ["tf_core_dropout_forward"], dropout_symbols
    assert "multiply" in cpp.TENSOR_CORE_OPS

    # Everything else exactly as G2 left it.
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.RAW_KERNELS == (
        "elementwise_add", "elementwise_subtract", "elementwise_multiply",
        "elementwise_divide", "relu", "matmul", "matmul_tiled",
    )
    assert cpp.TENSOR_CORE_KERNELS == (
        "relu", "add", "subtract", "multiply", "matmul",
    )
    assert cpp.NATIVE_MODULES == (
        "NativeModule", "NativeLinear", "NativeReLU", "NativeFlatten",
        "NativeConv2d", "NativeMaxPool2d", "NativeSequential",
        "NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d",
        # Phase G milestone G4 appended the Dropout module. It is
        # unrelated to this milestone, which added no module of its own.
        "NativeDropout",
    )
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "generator_state",
        "save_native_checkpoint", "load_native_checkpoint",
        "checkpoint_generator_state",
    )
    assert native_checkpoint._FORMAT_VERSION == 2
    assert tuple(cpp.backend_info()["autograd_ops"]) == cpp.AUTOGRAD_OPS


def test_g3_operation_semantics_are_what_the_contract_locks():
    """The behavioral guardrails, taken from the live runtime rather than
    from prose: the explicit keyword-only generator, ``p == 0`` identity
    that reserves nothing, one call per successful stochastic forward,
    none per failure or backward, a graph-owned mask, and a backward that
    reads only the mask."""
    import inspect

    import numpy as np
    import pytest

    from tensorforge.backends import cpp
    from tensorforge.experimental import NativeGenerator, NativeTensor
    from tensorforge.experimental import native_tensor as native_tensor_module

    if not cpp.is_available():
        pytest.skip("experimental C++ backend not built")

    # The exact signature: `p` positional, `generator` keyword-only with
    # no default, so there is no implicit stream to fall back to.
    parameters = inspect.signature(NativeTensor.dropout).parameters
    assert list(parameters) == ["self", "p", "generator"]
    assert (parameters["generator"].kind
            is inspect.Parameter.KEYWORD_ONLY)
    assert parameters["generator"].default is inspect.Parameter.empty

    values = np.arange(1.0, 13.0)
    generator = NativeGenerator(2024)
    x = NativeTensor.from_array(values, requires_grad=True)

    # p == 0 is the input object, and consumes nothing.
    assert x.dropout(0.0, generator=generator) is x
    assert generator.calls == 0

    # A failure consumes nothing either.
    with pytest.raises(ValueError):
        x.dropout(1.0, generator=generator)
    assert generator.calls == 0

    # One successful stochastic forward consumes exactly one call, and the
    # mask is graph-owned private state — a Core object, never a tensor.
    y = x.dropout(0.5, generator=generator)
    assert generator.calls == 1
    assert len(y._graph_resources) == 1
    mask = y._graph_resources[0]
    assert isinstance(mask, cpp.NativeTensorCore)
    assert not isinstance(mask, NativeTensor)
    # No parameter version is recorded: backward does not reread the input.
    assert y._expected_versions == ()

    # Backward uses the mask and consumes no call.
    saved = mask.to_numpy().copy()
    g = NativeTensor.from_array(np.ones(12))
    y.backward(gradient=g)
    assert np.array_equal(x.grad.to_numpy(), saved)
    assert generator.calls == 1
    # ...and the mask was released exactly once with the graph.
    assert mask._closed is True
    assert y._graph_resources == ()
    for tensor in (y, g, x):
        tensor.close()

    # The operation's *code* never reaches for a global stream, a NumPy
    # RNG, or a second probability rule. The docstring is stripped first
    # — it legitimately says "no process-global stream", and a scan that
    # cannot tell a prohibition from an implementation is worthless. The
    # split is on the docstring delimiters rather than on ``__doc__``,
    # which ``inspect.getsource`` does not reproduce byte-for-byte when
    # the file's non-ASCII characters meet a non-UTF-8 locale.
    source = inspect.getsource(NativeTensor.dropout)
    opening, _docstring, body = source.split('"""', 2)
    assert "def dropout" in opening
    for forbidden in ("np.random", "numpy.random", "import random",
                      "secrets.", "default_generator", "global ",
                      "NativeGenerator("):
        assert forbidden not in body, forbidden
    assert "_normalize_dropout_probability" in source, (
        "the operation must reuse the shared probability validator"
    )
    assert "_reserve_call" in source and "_commit_call" in source
    assert "graph_resources" in source
    # Cancellation lives in the outcome-aware cleanup helper, which is
    # where the pre-commit / post-commit distinction is made (design §5).
    settle = inspect.getsource(native_tensor_module._settle_failed_dropout)
    assert "_abandon_call" in settle
    assert "_call_committed" in settle


def test_g3_commit_boundary_cleanup_is_outcome_aware():
    """The commit boundary has three outcomes (design §5), and the two
    failing ones are **not** interchangeable: before the commit takes
    effect no call is consumed and the reservation is abandoned; after it
    the index is irreversibly spent and the committed token must not be
    abandoned.

    Both premises are taken from the **live runtime** by injecting a
    `_commit_call` that raises on each side of the real commit, so this
    guard tracks behavior rather than prose; the document is then held to
    what it observes. The decisive property — that the outcome is read
    from the token rather than from a flag set after `_commit_call` — is
    checked structurally, because a flag-based implementation passes the
    behavioral half only by luck of statement ordering."""
    import inspect

    import numpy as np
    import pytest

    from tensorforge.backends import cpp
    from tensorforge.experimental import NativeGenerator, NativeTensor
    from tensorforge.experimental import native_tensor as native_tensor_module

    if not cpp.is_available():
        pytest.skip("experimental C++ backend not built")

    values = np.arange(1.0, 13.0)
    real_commit = NativeGenerator._commit_call
    real_abandon = NativeGenerator._abandon_call

    # -- outcome 1: the commit raises *instead of* committing.
    generator = NativeGenerator(3001)
    x = NativeTensor.from_array(values, requires_grad=True)
    try:
        NativeGenerator._commit_call = lambda self, token: (
            (_ for _ in ()).throw(KeyboardInterrupt("pre-commit"))
        )
        with pytest.raises(KeyboardInterrupt):
            x.dropout(0.5, generator=generator)
    finally:
        NativeGenerator._commit_call = real_commit
    assert generator.calls == 0, "a failed commit consumed a call"
    assert generator._has_active_reservation() is False

    # -- outcome 3: the commit succeeds, then an exception arrives.
    abandoned = []
    try:
        def commit_then_raise(self, token):
            real_commit(self, token)
            raise KeyboardInterrupt("post-commit")

        def recording_abandon(self, token):
            abandoned.append(token)
            return real_abandon(self, token)

        NativeGenerator._commit_call = commit_then_raise
        NativeGenerator._abandon_call = recording_abandon
        with pytest.raises(KeyboardInterrupt) as info:
            x.dropout(0.5, generator=generator)
    finally:
        NativeGenerator._commit_call = real_commit
        NativeGenerator._abandon_call = real_abandon

    # The call is spent, honestly, and the original error propagates.
    assert generator.calls == 1
    assert "post-commit" in str(info.value)
    assert generator._has_active_reservation() is False
    assert abandoned == [], "a committed reservation was abandoned"
    # ...and the spent index is never reissued.
    y = x.dropout(0.5, generator=generator)
    assert generator.calls == 2
    y.close()
    x.close()

    # The outcome really is read from the token, not from a local flag.
    settle = inspect.getsource(
        native_tensor_module._settle_failed_dropout
    )
    assert "_call_committed" in settle, (
        "the cleanup must ask the token whether it committed"
    )
    query = inspect.getsource(NativeGenerator._call_committed)
    # Assignment, not comparison: `token._outcome == "committed"` is the
    # read this method exists for, so the pattern must exclude `==`.
    for slot in ("_calls", "_active_serial", "_active_index",
                 "_claim_serial", "_claim_index", "_next_serial",
                 "_seed", "_outcome"):
        assert re.search(rf"{slot}\s*(=(?!=)|\+=|-=)", query) is None, (
            f"the committed-outcome query must be read-only, but it "
            f"assigns {slot!r}"
        )
    # It is private, and no public spelling of it exists.
    assert not hasattr(NativeGenerator, "call_committed")

    # The design says all three outcomes, and does not claim the
    # commit-to-return window consumes nothing.
    design = _status_text(PHASE_G_DESIGN)
    assert re.search(
        r"commit[- ]to[- ]return|after a successful commit", design, re.I,
    ), "the design does not describe the commit-to-return window"
    assert re.search(
        r"irreversibl", design, re.I,
    ), "the design does not say the committed index is irreversible"


def _phase_g_milestone(index):
    """The body of one ``### G<n> —`` block of the Phase-G ladder,
    whitespace-collapsed and with markdown emphasis stripped.

    Scoping a check to a single milestone is what makes "this rule holds
    at G4 but not at G10" testable at all: a document-wide search would
    happily accept a rule stated once and contradicted per milestone."""
    text = (REPO_ROOT / PHASE_G_DESIGN).read_text(encoding="utf-8")
    # Scope to the ladder section first: without that bound, the last
    # milestone's block would run on into whatever follows it.
    for chunk in re.split(r"\n## ", text):
        if "Milestone ladder" in chunk.split("\n", 1)[0]:
            break
    else:
        raise AssertionError(f"{PHASE_G_DESIGN} has no milestone ladder")
    for block in re.split(r"\n#{3,4} (?=G\d+ )", chunk):
        if re.match(rf"G{index}\s*[-—]", block):
            return re.sub(r"\s+", " ", re.sub(r"[*`]", "", block))
    raise AssertionError(
        f"{PHASE_G_DESIGN} has no milestone block for G{index}"
    )


def test_phase_g_keeps_dropout_unsupported_until_the_g10_closure():
    """Revised guardrail 1. ``"dropout"`` is a *closure* capability, not an
    implementation one: G4 implements and exports ``NativeDropout``, but
    the registry does not advertise Dropout until the whole Phase-G
    closure matrix has passed at G10.

    This is checked per milestone rather than document-wide, because the
    failure this guards against is exactly a later milestone quietly
    reinstating the old "move it at G4" rule."""
    from tensorforge.backends import cpp

    # Premise, straight from the live registry: it is unsupported today,
    # and G0 did not move it.
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")

    text = _status_text(PHASE_G_DESIGN)
    # The opening contract summary — what a reader sees before §1 — must
    # carry both tuples and the timing, not bury them in a milestone.
    header = text.split("## 1.")[0]
    assert '("dropout", "float32", "cuda", "amp")' in header, (
        "the design header no longer pins the current UNSUPPORTED tuple"
    )
    assert '("float32", "cuda", "amp")' in header, (
        "the design header no longer pins the post-closure UNSUPPORTED tuple"
    )
    assert re.search(r"G0.{0,3}G9|through G9", header), (
        "the design header no longer says the name is held through G9"
    )
    # The removal is bound to G10 and to the closure matrix, never to G4.
    assert re.search(r"(leaves?|left|removed?|remove)\b[^.]{0,120}\bG10\b"
                     r"|\bG10\b[^.]{0,120}(leaves?|removes?|removed)", text), (
        "the design no longer binds the UNSUPPORTED removal to G10"
    )

    # Every milestone from G1 through G9 says the name stays put...
    keeps = re.compile(r"(stays|still|remains)\s+(?:\w+\s+){0,3}?in"
                       r"\s+UNSUPPORTED"
                       r"|any change to UNSUPPORTED"
                       r"|UNSUPPORTED still reads", re.I)
    # ...and none of them may claim it leaves.
    leaves = re.compile(r"(leaves?|left|removed?|drops? out of|gone from)"
                        r"\s+(?:\w+\s+){0,4}?UNSUPPORTED", re.I)
    for index in range(0, 10):
        block = _phase_g_milestone(index)
        if index:
            assert keeps.search(block), (
                f"milestone G{index} no longer states that \"dropout\" "
                f"stays in UNSUPPORTED"
            )
        assert not leaves.search(block), (
            f"milestone G{index} claims \"dropout\" leaves UNSUPPORTED, "
            f"but only G10 may do that"
        )

    # G10 is where it actually happens, and it happens after the matrix.
    closure = _phase_g_milestone(10)
    assert re.search(r"(removed?|remove)\b[^.]{0,120}UNSUPPORTED"
                     r"|UNSUPPORTED[^.]{0,120}(removed?|remove)",
                     closure, re.I), (
        "milestone G10 no longer removes \"dropout\" from UNSUPPORTED"
    )
    assert '("float32", "cuda", "amp")' in closure, (
        "milestone G10 does not state the resulting UNSUPPORTED tuple"
    )
    assert re.search(r"\blast\b|\bafter\b|\bonly after\b", closure, re.I), (
        "milestone G10 does not order the removal after the closure matrix"
    )
    # The closure requirements own the gate.
    requirements = _top_level_section("Phase-closure requirements",
                                      PHASE_G_DESIGN)
    assert re.search(r"UNSUPPORTED", requirements), (
        "the closure requirements no longer mention the capability boundary"
    )

    # And every status surface tells the same story: the boundary moves
    # at G10, and nothing binds it to G4.
    at_g4 = (
        re.compile(r"(leaves?|left|removed?|moves?|gone from)[^.]{0,60}"
                   r"unsupported[^.]{0,60}\bG4\b", re.I),
        re.compile(r"unsupported[^.]{0,60}(until|at)\s+(milestone\s+)?"
                   r"G4\b", re.I),
        re.compile(r"capability name moves[^.]{0,20}G4", re.I),
        re.compile(r"\bG4\b[^.]{0,80}(leaves?|removes?|moves?)[^.]{0,40}"
                   r"unsupported", re.I),
    )
    at_g10 = re.compile(
        r"(leaves?|removed?|remove|gone from)[^.]{0,80}\bG10\b"
        r"|\bG10\b[^.]{0,80}(leaves?|removes?|removed)"
        r"|(until|only at|at)\s+(milestone\s+)?G10\b", re.I,
    )
    for surface in ("README.md", "docs/roadmap.md",
                    "docs/native_support_matrix.md",
                    "docs/backend_experiments.md", "CLAUDE.md"):
        surface_text = _status_text(surface)
        assert re.search(r"G10", surface_text), (
            f"{surface} does not name the G10 closure milestone"
        )
        for pattern in at_g4:
            assert not pattern.search(surface_text), (
                f"{surface} still moves the capability boundary at G4: "
                f"{pattern.search(surface_text).group(0)!r}"
            )
        assert at_g10.search(surface_text), (
            f"{surface} does not bind the capability-boundary move to G10"
        )


def _phase_g_generator_contract():
    """The Phase-G ``NativeGenerator`` section with markdown emphasis
    stripped, so a rule is matched by meaning rather than by whether it
    happened to be bolded."""
    section = _top_level_section("NativeGenerator", PHASE_G_DESIGN)
    return re.sub(r"[*`]", "", section)


def test_phase_g_locks_a_lock_protected_token_validated_reservation():
    """Guardrail 2. The call counter is the one piece of Phase-G state a
    race could corrupt invisibly, so the design must lock a real
    synchronization contract: one lock over every state transition, the
    native work outside it, and opaque single-use tokens so a stale,
    foreign, or duplicated commit cannot be mistaken for a live one."""
    generator = _phase_g_generator_contract()
    lowered = generator.lower()

    # The governing invariant: nothing that can run a callback executes
    # while a generator lock is held. Token construction is the one
    # allocation in the reservation path — allocation can run
    # finalization, and a finalizer could start a multi-generator
    # transaction — so it must happen with no lock held, and the design
    # must not go back to calling the opposite an invariant.
    assert re.search(r"outside\s+(\*\*)?the\s+lock|no\s+generator\s+lock\s+"
                     r"is\s+held|holding\s+no\s+generator\s+lock", lowered), (
        "the design no longer builds the reservation token outside the lock"
    )
    # Narrow to *construction*: matching or committing a token under the
    # lock is correct and must stay sayable.
    build = r"(construct|mint|allocat|built|building)\w*"
    inside = r"(under|inside|while holding)\s+(the|it|this)?\s*" \
             r"(critical\s+section|lock)"
    assert not re.search(
        rf"{build}[^.]{{0,60}}token[^.]{{0,120}}{inside}"
        rf"|token[^.]{{0,40}}{build}[^.]{{0,120}}{inside}"
        rf"|{inside}[^.]{{0,80}}{build}[^.]{{0,40}}token"
        rf"|allocation[^.]{{0,60}}under\s+it[^.]{{0,40}}token",
        lowered), (
        "the design again holds a generator lock across token construction, "
        "which lets a finalizer invert the multi-generator lock order"
    )
    assert re.search(r"no\s+(user\s+code|callback)[^.]{0,160}"
                     r"(lock\s+is\s+held|holding\s+a\s+lock)"
                     r"|no[^.]{0,80}callback[^.]{0,120}lock\s+is\s+held",
                     lowered), (
        "the design no longer states that no callback-capable operation "
        "runs while a generator lock is held"
    )

    # The two-phase claim / construct / publish transaction itself.
    assert re.search(r"claim", lowered), (
        "the design no longer has a reservation-construction claim"
    )
    for phase in ("claim", "construct", "publish", "deliver"):
        assert re.search(rf"phase\s+\d\s+—\s+{phase}", lowered), (
            f"the design no longer names the {phase!r} phase of reservation "
            f"construction"
        )
    # ...and what it blocks, named. A design that only refused a second
    # *reservation* would satisfy a looser check while leaving state
    # replacement free to move the seed under a claimed index.
    windows = [lowered[match.end():match.end() + 900]
               for match in re.finditer(r"construction claim", lowered)]
    assert windows, (
        "the design no longer names a reservation-construction claim"
    )
    required = {
        "says the blocked operations fail": r"(fail|raise|refus|reject)",
        "blocks another reservation": r"another reservation",
        "blocks load_state": r"load_state",
        "blocks reseed": r"reseed",
        "blocks reset": r"reset",
        "blocks replace_generator_states": r"replace_generator_states",
        "says a blocked operation mutates nothing":
            r"mutates? nothing|changes? nothing|no mutation",
        "still allows state inspection": r"inspection",
    }
    best, missing = 0, list(required)
    for window in windows:
        absent = [name for name, pattern in required.items()
                  if not re.search(pattern, window)]
        if len(required) - len(absent) > best:
            best, missing = len(required) - len(absent), absent
    assert not missing, (
        "the construction claim contract no longer "
        + "; no longer ".join(missing)
    )
    # The four failure positions, scoped to the subsection that owns
    # them. Matching anywhere in §3.6 is not enough: "foreign",
    # "committed", and "abandoned" all appear in the *token* contract
    # below, so an unscoped check passes even with the whole
    # failed-delivery rule deleted.
    # The helper collapses newlines, so the subsection is bounded by the
    # next "####" heading marker rather than by a line break.
    positions = re.search(r"####\s+the four failure positions.*?(?=####|\Z)",
                          lowered, re.S)
    assert positions, (
        "the design no longer distinguishes the four reservation failure "
        "positions"
    )
    positions = positions.group(0)
    for position in ("construction", "publication", "deliver"):
        assert position in positions, (
            f"the failure-position contract no longer covers {position!r}"
        )
    assert re.search(
        r"(after|before)[^.|]{0,60}(publication|published)[^.|]{0,140}deliver"
        r"|deliver\w*[^.|]{0,140}(after|before)[^.|]{0,60}publi", positions), (
        "the design no longer names the publication-to-delivery failure "
        "position separately from construction failure"
    )
    assert re.search(r"strand", positions), (
        "the design no longer says an undelivered reservation would "
        "permanently strand the generator"
    )
    # The specific wrong claim — that clearing the claim covers the
    # publication-to-return window — must be stated as false, and must
    # not come back in any form.
    assert re.search(r"clearing the claim does nothing"
                     r"|the claim is gone", positions), (
        "the design no longer states that clearing the construction claim "
        "does nothing for the publication-to-return window"
    )
    assert not re.search(
        r"claim[^.]{0,120}covers?\s+(every|all|the whole|both)"
        r"|(finally|cleanup)[^.]{0,80}covers?[^.]{0,60}"
        r"(publication|between the publish|publish-to-return)", lowered), (
        "the design again claims that clearing the construction claim "
        "covers the publication-to-return window"
    )
    # The cleanup is exact-match, and names what it must never cancel.
    for protected in ("newer", "foreign", "committed", "abandoned"):
        assert protected in positions, (
            f"the failed-delivery cleanup no longer protects a {protected!r} "
            f"reservation"
        )
    assert re.search(r"serial[^.]{0,60}(and|,)[^.]{0,30}index", positions), (
        "the failed-delivery cleanup no longer matches serial and index"
    )
    assert re.search(r"never advanc\w*\s+calls|without advancing\s+calls"
                     r"|calls\s+untouched|never consumes a call index",
                     positions), (
        "the design no longer says a failed delivery leaves calls alone"
    )
    # And it must not be able to break the multi-generator lock order.
    assert re.search(r"only its own generator.?s lock|only[^.]{0,40}own lock",
                     positions), (
        "the failed-delivery cleanup no longer takes only its own lock"
    )

    # Reentry through a finalizer cannot invert the global lock order.
    assert re.search(r"finaliz", lowered), (
        "the design no longer explains what runs during construction"
    )
    assert re.search(r"invert[^.]{0,80}order|order[^.]{0,80}invert", lowered), (
        "the design no longer states that finalizer or callback reentry "
        "cannot invert the multi-generator lock order"
    )
    assert re.search(r"(deadlock|hang)", lowered), (
        "the design no longer states that reentry must not deadlock"
    )
    assert re.search(r"finally", lowered), (
        "the design no longer releases the claim on the failure path"
    )
    # An RLock is still required — the transaction re-enters through the
    # shared write seam — but it must be justified, not merely asserted.
    assert "threading.RLock" in generator, (
        "the design no longer gives NativeGenerator a reentrant lock"
    )
    assert not re.search(r"plain\s+.?Lock.?,?\s+not\s+an\s+.?RLock",
                         generator, re.I), (
        "the design has reverted to a plain Lock, which self-deadlocks in "
        "the multi-generator transaction"
    )
    assert re.search(r"why it stays an rlock|structural", lowered), (
        "the design no longer justifies keeping a reentrant lock"
    )
    # Every operation the prompt requires the lock to cover is named as
    # covered, not merely mentioned somewhere in the section.
    for operation in ("_reserve_call", "_abandon_call", "_commit_call",
                      "state()", "load_state", "reseed", "reset"):
        assert operation in generator, (
            f"the reservation contract does not cover {operation!r}"
        )
    assert re.search(r"(counter|calls)[^.]{0,60}read", lowered) or (
        re.search(r"read[^.]{0,60}(counter|calls)", lowered)
    ), "the lock contract does not cover counter reads"
    assert re.search(r"under the lock|holding the lock", lowered), (
        "the design no longer says which operations hold the lock"
    )
    assert re.search(r"outside the lock", lowered), (
        "the design no longer keeps native computation outside the lock"
    )

    # The token, and what makes it unforgeable in practice.
    assert re.search(r"opaque token", lowered), (
        "the reservation no longer returns an opaque token"
    )
    assert re.search(r"serial", lowered), (
        "the token no longer carries a non-reused reservation serial"
    )
    for rejected in ("stale", "foreign", "already committed",
                     "already abandoned"):
        assert rejected in lowered, (
            f"the token contract does not reject a {rejected!r} token"
        )
    assert re.search(r"advances? (the counter )?exactly once"
                     r"|advances exactly once", lowered), (
        "the design no longer states that commit advances exactly once"
    )
    assert re.search(r"cancel[^.]{0,80}never advanc"
                     r"|never advanc[^.]{0,80}cancel"
                     r"|abandon[^.]{0,80}never advanc", lowered), (
        "the design no longer states that cancellation never advances"
    )
    assert re.search(r"at most one", lowered), (
        "the design no longer limits the generator to one live reservation"
    )
    # Rejected tokens are inert: nothing moves.
    assert re.search(r"(without|no)[^.]{0,40}(chang|state)"
                     r"|leaves? .{0,60}exactly as they were"
                     r"|nothing changes", lowered), (
        "the design no longer says a rejected token changes nothing"
    )

    # State replacement cannot slip underneath a live reservation, and a
    # checkpoint load is held to the same rule.
    assert re.search(r"refus\w+[^.]{0,120}reservation is active"
                     r"|reservation is active[^.]{0,120}(refus|rais)",
                     lowered), (
        "the design no longer refuses state replacement during a reservation"
    )
    checkpoint = _top_level_section("Checkpoint format version 2",
                                    PHASE_G_DESIGN)
    assert re.search(r"active reservation", checkpoint, re.I), (
        "the checkpoint contract does not handle a live reservation"
    )


def test_phase_g_locks_the_multi_generator_state_transaction():
    """The load transaction is where a per-generator check-then-write
    would silently break the reservation rule: another caller can reserve
    in the gap, and the write then moves the seed out from under a live
    token. The contract must therefore lock **every** target across the
    recheck and the commit, in an order that cannot deadlock."""
    registration = _top_level_section("NativeModule", PHASE_G_DESIGN)
    section = re.sub(r"[*`]", "", registration)
    lowered = section.lower()

    # Every unique target is locked, together, not one at a time.
    assert re.search(r"every unique target", lowered), (
        "the transaction no longer locks every target"
    )
    assert re.search(r"held together|all of them, held|holding them all"
                     r"|while every lock is held", lowered), (
        "the transaction no longer holds the target locks simultaneously"
    )
    # The order is global and caller-independent, which is what makes
    # two overlapping loads deadlock-free.
    assert re.search(r"independent of the caller|global order", lowered), (
        "the acquisition order is no longer caller-independent"
    )
    assert re.search(r"identity|id\(\)", lowered), (
        "the design no longer says what the global order is keyed on"
    )
    assert re.search(r"deadlock", lowered), (
        "the design no longer explains the deadlock the order prevents"
    )
    assert re.search(r"reverse", lowered), (
        "the design no longer releases the locks in reverse order"
    )
    # The recheck happens under those locks, and covers construction too.
    assert re.search(r"recheck", lowered), (
        "the design no longer rechecks for reservations under the locks"
    )
    assert re.search(r"under construction|being constructed"
                     r"|construction claim", lowered), (
        "the recheck no longer covers a reservation under construction"
    )
    assert re.search(
        r"no reservation may begin[^.]{0,120}(recheck|check)"
        r"|between the recheck and the end of the commit", lowered
    ), "the design no longer forbids a reservation starting mid-commit"
    # The order survives reentry, which is only true because token
    # construction holds no lock (§3.6).
    assert re.search(r"finaliz", lowered), (
        "the transaction no longer says what happens when it is reached "
        "from a finalizer"
    )
    assert re.search(r"invert", lowered), (
        "the design no longer states that finalizer or callback reentry "
        "cannot invert the global lock order"
    )
    assert re.search(r"no generator lock is held|owning nothing"
                     r"|constructed with no", lowered), (
        "the design no longer connects the order guarantee to token "
        "construction holding no lock"
    )
    # The commit cannot fail, and the rollback finishes before unlocking.
    assert re.search(r"cannot fail", lowered), (
        "the commit is no longer non-failing by construction"
    )
    assert re.search(r"before any lock is released", lowered), (
        "the rollback may now complete after the locks are released"
    )
    # The two honest outcomes for a racing reservation are stated.
    assert re.search(r"never overlap the commit|can never overlap", lowered), (
        "the design no longer rules out a reservation overlapping the commit"
    )
    # Aliases fold by identity rather than half-applying.
    assert re.search(r"conflict", lowered), (
        "the design no longer rejects conflicting aliased states"
    )


def test_phase_g_forbids_duplicate_call_indices_under_concurrency():
    """Guardrail 3. Two callers must never receive the same call index —
    that would produce two identical masks that a resume could not
    distinguish — and the fix must be honest about what it is: a
    correctness serialization, not a parallelism feature."""
    # Scoped to the generator's own contract: a promise made only in a
    # summary elsewhere would leave the contract free to contradict it.
    lowered = _phase_g_generator_contract().lower()

    assert re.search(r"no two threads[^.]{0,80}same[^.]{0,40}call index"
                     r"|same call index[^.]{0,80}(never|cannot|no two)",
                     lowered), (
        "the design no longer forbids two callers sharing a call index"
    )
    # The second caller fails *before* an index exists, so it cannot
    # consume, duplicate, or skip one.
    assert re.search(r"before an index is minted"
                     r"|no index is minted"
                     r"|without (receiving|being given)[^.]{0,40}index",
                     lowered), (
        "the design no longer refuses the second reservation before an "
        "index is handed out"
    )
    for case in ("concurrent", "reentrant"):
        assert case in lowered, f"the concurrency contract omits {case!r}"
    # Exhaustion is a counter read too, so it is inside the lock.
    assert re.search(r"exhaust\w*[^.]{0,120}lock"
                     r"|lock[^.]{0,120}exhaust", lowered), (
        "counter exhaustion is no longer checked under the lock"
    )
    # And the honest limit: this buys correctness, not throughput.
    assert re.search(r"(not|never)\s+claim\w*[^.]{0,80}parallel"
                     r"|parallel[^.]{0,80}(not|never)\s+claim", lowered), (
        "the generator contract no longer disclaims parallel stochastic "
        "execution"
    )
    assert re.search(r"serializ\w+ for correctness", lowered), (
        "the design no longer names the lock as correctness serialization"
    )
    # No surface may claim the opposite, either — a lock that is sold as
    # a parallelism feature is a lock people will lean on.
    whole = _status_text(PHASE_G_DESIGN).lower()
    for surface_text in (whole,) + tuple(
        _status_text(name).lower() for name in
        ("README.md", "docs/roadmap.md", "docs/project_summary.md",
         "docs/backend_experiments.md", "CLAUDE.md")
    ):
        assert not re.search(
            r"(supports?|enables?|allows?|makes?)\s+(?:\w+\s+){0,3}?"
            r"parallel stochastic"
            r"|parallel stochastic execution (is|are)\s+"
            r"(safe|supported|allowed|possible)", surface_text,
        ), "a surface claims parallel stochastic execution is supported"
    # The concurrency rows are testable work, not prose: the milestones
    # that build and harden the generator own them.
    owns = re.compile(r"concurren\w*|reentran\w*|reservation|token", re.I)
    for index in (1, 3, 6):
        block = _phase_g_milestone(index)
        assert owns.search(block), (
            f"milestone G{index} does not own the reservation tests"
        )


def _checkpoint_subsection(token):
    """The body of one subsection of the Phase-G checkpoint section,
    matched by heading text rather than by section number.

    Section-wide searches are too forgiving here: the checkpoint contract
    repeats "alias" and "prevalidation" often enough that deleting the
    load-bearing statement still leaves the words present. Scoping to the
    subsection that *owns* the rule makes the guard bite."""
    text = (REPO_ROOT / PHASE_G_DESIGN).read_text(encoding="utf-8")
    for chunk in re.split(r"\n## ", text):
        if "Checkpoint format version 2" in chunk.split("\n", 1)[0]:
            break
    else:
        raise AssertionError(f"{PHASE_G_DESIGN} has no checkpoint section")
    for block in re.split(r"\n#{3,4} ", chunk):
        if token in block.split("\n", 1)[0]:
            return re.sub(r"\s+", " ", re.sub(r"[*`]", "", block))
    raise AssertionError(
        f"the checkpoint section has no subsection naming {token!r}"
    )


def test_phase_g_checkpoint_v2_records_the_generator_alias_topology():
    """Guardrail 4. Generator *sharing* is semantic state: two layers on
    one stream behave differently from two layers on two. An archive that
    saved only canonical states could be restored into a model with a
    different topology and silently diverge, so version 2 records every
    registered path and its canonical target."""
    checkpoint = _top_level_section("Checkpoint format version 2",
                                    PHASE_G_DESIGN)
    lowered = checkpoint.lower()

    # The alias map is part of the *manifest shape*, not merely discussed.
    manifest = _checkpoint_subsection("Manifest change")
    assert "aliases" in manifest, (
        "the version-2 manifest no longer carries an alias map"
    )
    assert re.search(r"aliases[^.]{0,120}canonical", manifest, re.I), (
        "the manifest no longer defines aliases as path-to-canonical"
    )
    for field in ("keys", "entries", "aliases"):
        assert field in manifest, (
            f"the manifest shape no longer names the {field!r} field"
        )
    assert re.search(r"canonical", lowered), (
        "the design no longer names canonical generator entries"
    )
    # The four things the representation must preserve.
    assert re.search(r"every registered[^.]{0,60}path"
                     r"|every[^.]{0,30}registered generator path", lowered), (
        "the alias map no longer covers every registered generator path"
    )
    assert re.search(r"alias[^.]{0,60}canonical", lowered), (
        "the design no longer defines the alias-to-canonical relationship"
    )
    assert re.search(r"shared[^.]{0,80}independent"
                     r"|independent[^.]{0,80}shared", lowered), (
        "the design no longer preserves shared-versus-independent identity"
    )
    # Determinism, both of the canonical choice and of the byte order.
    assert re.search(r"canonical.name selection is deterministic"
                     r"|deterministic[^.]{0,60}canonical", lowered), (
        "canonical-name selection is no longer deterministic"
    )
    assert re.search(r"serializ\w+ order is deterministic"
                     r"|deterministic[^.]{0,60}order", lowered), (
        "the serialization order is no longer deterministic"
    )
    # Whether a canonical name also appears as an alias is answered, not
    # left implicit.
    assert re.search(r"canonical name also appears in"
                     r"|mapped to itself|self.mapped", lowered), (
        "the design no longer says whether canonical names appear in the "
        "alias map"
    )
    # Cycles are addressed even though the shape is chosen to exclude them.
    assert "cycle" in lowered, (
        "the design no longer addresses alias cycles"
    )
    # Loading matches a real traversal and never swaps the objects out.
    assert re.search(r"named_generators\(\)", checkpoint), (
        "the load no longer compares against a real module traversal"
    )
    assert re.search(r"identity is preserved|never (constructs|replaces)"
                     r"[^.]{0,60}generator|preserv\w+[^.]{0,40}identit",
                     lowered), (
        "the load no longer preserves live generator identity"
    )


def test_phase_g_alias_and_topology_mismatches_fail_before_live_mutation():
    """Guardrail 5. Every topology check belongs in prevalidation. A
    mismatch discovered mid-commit would already have changed live state,
    which is the failure mode the whole transaction contract exists to
    prevent."""
    checkpoint = _top_level_section("Checkpoint format version 2",
                                    PHASE_G_DESIGN)
    lowered = checkpoint.lower()

    # The validation subsection itself has to say when it runs — a rule
    # stated only in the transaction section would leave the table free
    # to drift into the commit phase.
    validation = _checkpoint_subsection("Validation").lower()
    assert "prevalidation" in validation, (
        "the validation table no longer says it runs in prevalidation"
    )
    assert re.search(r"before any (staging|live)", validation), (
        "the validation table no longer runs before any live change"
    )
    assert re.search(r"prevalidation", lowered), (
        "the checkpoint contract no longer names a prevalidation phase"
    )
    assert re.search(r"before any (live|staging)", lowered), (
        "the design no longer places the checks before any live change"
    )
    # The specific mismatch classes the contract must enumerate.
    for mismatch in ("missing", "unexpected", "duplicate", "malformed"):
        assert mismatch in lowered, (
            f"the validation table omits the {mismatch!r} case"
        )
    assert re.search(r"alias[^|.]{0,80}(absent|missing)"
                     r"|absent canonical entry"
                     r"|targeting an absent", lowered), (
        "an alias targeting an absent entry is no longer rejected"
    )
    assert re.search(r"saved.{0,10}shared[^|]{0,60}live.{0,10}independent"
                     r"|shared generator, live independent"
                     r"|archive shares two paths", lowered), (
        "a saved-shared / live-independent mismatch is no longer rejected"
    )
    assert re.search(r"independent generators, live[^|]{0,40}shared"
                     r"|archive has independent generators", lowered), (
        "a saved-independent / live-shared mismatch is no longer rejected"
    )
    assert re.search(r"canonical name[^|.]{0,60}(differs|changed)", lowered), (
        "a changed canonical name is no longer rejected"
    )
    assert re.search(r"strict in both directions", lowered), (
        "the generator section is no longer strict in both directions"
    )


def test_phase_g_locks_whole_checkpoint_synchronous_commit_atomicity():
    """Guardrail 6. A checkpoint load is one transaction over the whole
    archive. Any ordinary synchronous failure during the commit must
    restore *all four* state families together — a rollback that fixed
    the model but left the optimizer or the generators advanced would
    produce a resume that looks valid and is not."""
    checkpoint = _top_level_section("Checkpoint format version 2",
                                    PHASE_G_DESIGN)
    lowered = checkpoint.lower()

    # The four phases are distinguished, because they carry different
    # guarantees.
    for phase in ("prevalidation", "staging", "commit"):
        assert phase in lowered, f"the transaction omits the {phase!r} phase"
    assert re.search(r"stage[sd]? every|stag\w+[^.]{0,60}before", lowered), (
        "the design no longer stages everything before committing"
    )
    assert re.search(r"(allocate|allocation)[^.]{0,80}(stag|before)"
                     r"|everything that can (allocate|fail)", lowered), (
        "the design no longer confines allocation and failure to staging"
    )
    assert re.search(r"rollback", lowered), (
        "the design no longer keeps rollback state"
    )
    assert re.search(r"non-failing by construction"
                     r"|cannot itself fail|unfailable", lowered), (
        "the commit steps are no longer non-failing or rollback-covered"
    )
    # The commit phase itself carries the guarantee, so scope there: all
    # four state families restored together, identities intact, nothing
    # partial visible, and pre-existing graphs untouched.
    commit = _checkpoint_subsection("Phase 3").lower()
    for family in ("parameter", "buffer", "optimizer", "generator"):
        assert family in commit, (
            f"the commit rollback does not cover {family} state"
        )
    assert re.search(r"no partial\w*[^.]{0,80}observable"
                     r"|nothing partial\w*[^.]{0,40}observable", commit), (
        "the commit phase no longer forbids observing a partial load"
    )
    assert re.search(r"identit\w+[^.]{0,60}(unchanged|preserved)"
                     r"|(unchanged|preserved)[^.]{0,60}identit", commit), (
        "the commit phase no longer preserves object identity across a "
        "failed load"
    )
    assert re.search(r"mask\w*[^.]{0,80}unchanged"
                     r"|unchanged[^.]{0,60}mask", commit), (
        "the commit phase no longer protects pre-existing graph-owned masks"
    )
    assert re.search(r"rollback", commit), (
        "the commit phase no longer rolls anything back"
    )
    assert re.search(r"synchronous", commit), (
        "the commit phase no longer distinguishes synchronous failures"
    )
    # The same four families still have to be named somewhere in the
    # section as a whole, so the guarantee cannot be scoped away.
    for family in ("parameter", "buffer", "optimizer", "generator"):
        assert family in lowered, (
            f"the checkpoint contract does not cover {family} state"
        )
    # And the milestone that must implement it owns the requirement.
    g5 = _phase_g_milestone(5).lower()
    assert "rollback" in g5 and "commit" in g5, (
        "milestone G5 no longer owns the whole-checkpoint transaction"
    )
    g6 = _phase_g_milestone(6).lower()
    assert "matrix" in g6 or "10.7" in g6, (
        "milestone G6 no longer hardens the transaction contract"
    )


def test_phase_g_names_process_death_as_the_only_atomicity_exception():
    """Guardrail 7. The honest limit is stated once and precisely: only
    external process termination or interpreter death escapes the
    whole-checkpoint guarantee. A deliverable ``KeyboardInterrupt`` is
    explicitly *not* an exception — the earlier per-component wording let
    one leave the model restored and the optimizer stale, and this guard
    exists so that wording cannot come back."""
    checkpoint = _top_level_section("Checkpoint format version 2",
                                    PHASE_G_DESIGN)
    lowered = checkpoint.lower()

    assert re.search(r"only[^.]{0,80}(exception|uncovered)", lowered), (
        "the design no longer names a single documented exception"
    )
    assert re.search(r"(process|interpreter)[^.]{0,60}"
                     r"(termination|death|kill|crash)"
                     r"|(termination|death) of the (process|interpreter)",
                     lowered), (
        "the documented exception is no longer external process or "
        "interpreter death"
    )
    assert "keyboardinterrupt" in lowered, (
        "the design no longer says where KeyboardInterrupt falls"
    )
    assert re.search(r"keyboardinterrupt[^.]{0,120}not[^.]{0,60}exception"
                     r"|not[^.]{0,60}exception[^.]{0,160}keyboardinterrupt"
                     r"|deliverable[^.]{0,120}not\W{0,4}\*{0,2}an? exception",
                     lowered), (
        "a deliverable KeyboardInterrupt is no longer covered by the "
        "rollback guarantee"
    )
    # The stale "window between component commits" framing must be gone.
    assert not re.search(r"honest window|window between", lowered), (
        "the design still describes a multi-component commit window "
        "instead of one transaction"
    )
    assert not re.search(r"components may disagree", lowered), (
        "the design still permits components to disagree after a failure"
    )


def test_docs_present_the_shipped_dropout_benchmark():
    """G8's harness must stay documented as what it is: an honest local
    characterization with correctness gated before timing, honest
    reference labels, and **no** speed guarantee — never a performance
    contract, a committed number, or a CI gate. Every premise is checked
    against the live tree and the live harness first, so this guard
    tracks reality rather than prose."""
    benchmark = REPO_ROOT / "benchmarks" / "benchmark_native_dropout.py"
    assert benchmark.is_file()
    assert (REPO_ROOT / "tests"
            / "test_native_dropout_benchmark.py").is_file()

    from benchmarks import benchmark_native_dropout as harness

    # The contract section names every case and family the harness
    # actually declares — the design cannot drift from the code.
    section = _top_level_section("16. Benchmark contract",
                                 relative_path=PHASE_G_DESIGN)
    assert len(harness.CASES) == 35
    for case in harness.CASES:
        assert case in section, case
    for family in harness.FAMILIES:
        assert family in section, family
    # The six cases the contract originally named are still named.
    for case in ("core_dropout_forward", "tensor_dropout_forward",
                 "tensor_dropout_backward", "module_training_forward",
                 "module_eval_forward", "dropout_training_step"):
        assert case in harness.CASES and case in section, case

    lowered = section.lower()
    assert re.search(r"correctness.{0,40}before.{0,20}timing", lowered), (
        "the design no longer states that correctness runs before timing"
    )
    assert "median" in lowered and "spread" in lowered
    assert "warm-up" in lowered or "warmup" in lowered
    for label in (harness.NUMPY, harness.NATIVE_ONLY, harness.HARNESS):
        assert label in section, label
    for flag in ("--case", "--family", "--smoke", "--quick", "--json",
                 "--json-out"):
        assert flag in section, flag
    # The reference is exact, and the native_only decision is justified
    # rather than asserted.
    assert re.search(r"bit for bit|bit-for-bit", lowered)
    assert re.search(r"native_only[^|]{0,600}no\W{0,4}\*{0,2}(timing )?ratio",
                     section, re.I | re.S), (
        "the design no longer says why the operation and module cases "
        "publish no ratio"
    )
    # The honesty boundary.
    assert re.search(r"no.{0,40}speed", lowered), (
        "the design no longer states that no speed is asserted"
    )
    assert re.search(r"no ci timing threshold|no timing threshold", lowered)
    assert re.search(r"no result file", lowered)
    assert re.search(r"json-out[^.]{0,120}(destination|names)", lowered), (
        "the design no longer scopes the result file to an explicit "
        "destination"
    )
    assert "measurement only" in lowered

    # The milestone block records it as shipped, and as measurement only.
    milestone = _design_section("G8 —", relative_path=PHASE_G_DESIGN)
    milestone_lowered = milestone.lower()
    assert "complete" in milestone_lowered
    assert "measurement" in milestone_lowered
    assert re.search(r"no runtime file", milestone_lowered), (
        "the G8 block no longer says it changed no runtime file"
    )
    assert re.search(r"(stays|still|remains)\s+(?:\w+\s+){0,3}?in\s+"
                     r"UNSUPPORTED", re.sub(r"[*`]", "", milestone), re.I)

    # The README documents how to run it.
    readme = _status_text("README.md")
    for command in ("uv run python benchmarks/benchmark_native_dropout"
                    ".py --smoke",
                    "uv run python benchmarks/benchmark_native_dropout"
                    ".py --smoke --json"):
        assert command in readme, command

    # Every status surface that names it carries the honesty boundary...
    surfaces = ("README.md", "docs/native_support_matrix.md",
                "docs/roadmap.md", "docs/project_summary.md",
                "docs/backend_experiments.md")
    for surface in surfaces:
        text = _status_text(surface)
        # A raw character window, not a sentence one: the file name itself
        # contains a period, which would truncate a "[^.]" span.
        windows = [text[max(0, match.start() - 400):match.end() + 800]
                   for match in re.finditer("benchmark_native_dropout", text)]
        assert windows, surface
        assert any(re.search(
            r"(no speed|not a (performance )?(contract|guarantee)"
            r"|no timing threshold|no committed timing|characteriz)",
            chunk, re.I) for chunk in windows), surface

    # ...and none of them — nor CLAUDE.md, nor the design — commits a
    # timing number, a ratio, or a speed verdict as a project promise.
    for surface in surfaces + ("CLAUDE.md", PHASE_G_DESIGN):
        text = (REPO_ROOT / surface).read_text(encoding="utf-8")
        chunks = [text[max(0, match.start() - 400):match.end() + 800]
                  for match in re.finditer("benchmark_native_dropout", text)]
        for chunk in chunks:
            assert not re.search(
                r"\d+(\.\d+)?\s*(us|ms|µs|ns|microseconds|milliseconds|"
                r"seconds)\b", chunk, re.I
            ), (surface, chunk[:160])
            assert not re.search(r"\bx faster|\bspeedup\b|\d+(\.\d+)?x\b",
                                 chunk, re.I), (surface, chunk[:160])

    matrix = _status_text("docs/native_support_matrix.md")
    assert re.search(r"\|\s*G8\s*\|[^|]*\|[^|]*Complete", matrix), (
        "the support matrix does not mark G8 complete"
    )
    # G8 is a benchmark, not a capability: the boundary is where G7 left it.
    from tensorforge.backends import cpp

    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert not (REPO_ROOT / "benchmark_results").exists()
    assert not list((REPO_ROOT / "benchmarks").glob("*.json"))
