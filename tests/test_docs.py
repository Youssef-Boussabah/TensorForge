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


def test_roadmap_does_not_list_shipped_features_as_future():
    text = (REPO_ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    # Split at the future-work heading; shipped features must not appear
    # in the future section. "NativeAdam" is exempted by name: the
    # *native* adaptive optimizer genuinely is future work (v3.12),
    # while the stable framework's Adam stays banned as future work.
    future = text.split("## Practical next steps", 1)[1]
    future = future.split("## What this project is not", 1)[0]
    future = future.replace("NativeAdam", "")
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
            f"docs/roadmap.md lists already-shipped {shipped!r} as future work"
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
    # Unshipped native work must appear only after the unsupported
    # heading — never marked supported above it.
    assert "## Unsupported or future" in text
    supported_part = text.split("## Unsupported or future", 1)[0]
    future_part = text.split("## Unsupported or future", 1)[1]
    for term in ("CUDA", "float32",
                 "Conv2d", "MaxPool2d", "AMP",
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


def test_native_cnn_design_is_honest_about_its_remaining_work():
    """Phase D is not finished: the design doc must keep saying so, and the
    backend registry must keep the shipped/unshipped boundary accurate."""
    design = _normalized_doc("docs/native_cnn_design.md")
    # The doc states plainly that Phase D still has unimplemented work.
    assert "not implemented" in design.lower() or "not yet" in design.lower(), (
        "native_cnn_design.md must state that Phase D is incomplete"
    )
    # D11 (the deterministic CNN training + checkpoint-resume proof) and D12
    # are still planned, and Phase D is not marked complete.
    assert "D11" in design and "D12" in design

    # The backend capability registry keeps the boundary honest. As of D6/D7
    # the "conv2d" operation and the NativeConv2d module are implemented; as
    # of D8/D9/D10 the "maxpool2d" operation and the NativeMaxPool2d module
    # are too. Operation support and module support stay distinguishable.
    from tensorforge.backends import cpp

    assert "maxpool2d" not in cpp.UNSUPPORTED
    assert "maxpool2d" in cpp.AUTOGRAD_OPS
    assert "maxpool2d_forward" in cpp.TENSOR_CORE_OPS
    assert "maxpool2d_backward" in cpp.TENSOR_CORE_OPS
    assert "NativeMaxPool2d" not in cpp.UNSUPPORTED
    assert "NativeMaxPool2d" in cpp.NATIVE_MODULES
    # The module is a module only — never advertised as an op or kernel.
    assert "NativeMaxPool2d" not in cpp.AUTOGRAD_OPS
    assert "NativeMaxPool2d" not in cpp.TENSOR_CORE_OPS
    assert "NativeMaxPool2d" not in cpp.RAW_KERNELS
    assert "NativeConv2d" in cpp.NATIVE_MODULES
    assert "NativeConv2d" not in cpp.UNSUPPORTED

    # Both shipped CNN modules are exported from the experimental surface.
    import tensorforge.experimental as experimental

    assert "NativeConv2d" in experimental.__all__
    assert "NativeMaxPool2d" in experimental.__all__
    # The D11 training/checkpoint-resume proof has shipped as a runnable
    # example; the D12 completion pass (cross-cutting guardrails,
    # benchmarks, sanitizer validation) has not.
    assert (REPO_ROOT / "examples" / "native_cnn_training.py").is_file()
    assert not (REPO_ROOT / "tests" / "test_native_phase_d.py").exists()
    assert not (REPO_ROOT / "benchmarks" / "benchmark_native_cnn.py").exists()


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


def test_docs_do_not_reassert_stale_phase_d_status():
    """Phase D has begun and its Flatten and Conv2d milestones (D1–D7)
    have shipped. The project-facing docs must never regress to claiming
    the native CNN stack is unstarted, empty of layers, or that native
    convolution is unavailable. Phrase-level (not paragraph-verbatim) so
    accurate rewording and honest "pooling is upcoming" wording survive."""
    docs = (
        "README.md",
        "docs/native_cnn_design.md",
        "docs/native_support_matrix.md",
        "docs/roadmap.md",
        "docs/backend_experiments.md",
    )
    stale = (
        "no CNN layers",
        "Nothing in Phase D is implemented",
        "native CNN stack, which has not started",
        "native CNN phase has not started",
        "native CNN stack has not started",
        "Native Conv2d is unavailable",
    )
    for name in docs:
        low = _normalized_doc(name).lower()
        for phrase in stale:
            assert phrase.lower() not in low, (
                f"{name} reasserts the stale Phase-D claim {phrase!r}"
            )


def test_docs_present_shipped_native_cnn_layers():
    """The shipped native CNN layers (D1 NativeFlatten, D7 NativeConv2d)
    are positively presented in the README, roadmap, and support matrix,
    while pooling and the CNN training proof stay marked upcoming."""
    # README and support matrix name the trainable module directly.
    for name in ("README.md", "docs/native_support_matrix.md"):
        text = _normalized_doc(name)
        assert "NativeConv2d" in text, f"{name} no longer presents shipped NativeConv2d"
        assert "NativeFlatten" in text, f"{name} no longer presents shipped NativeFlatten"
    # The roadmap presents the shipped convolution line (it deliberately
    # says "convolution", not the banned "Conv2d" token, in its forward-
    # looking section) plus the shipped NativeFlatten module.
    roadmap = _normalized_doc("docs/roadmap.md")
    assert "NativeFlatten" in roadmap
    assert "convolution" in roadmap.lower()
    # Pooling and the end-to-end proof remain honestly upcoming.
    matrix = _normalized_doc("docs/native_support_matrix.md")
    assert "NativeMaxPool2d" in matrix
    assert "Planned" in matrix or "upcoming" in matrix.lower()


def test_native_cnn_design_is_linked_and_referenced():
    """The design doc must be reachable from the roadmap and the support
    matrix so the contract is discoverable, not orphaned."""
    roadmap = (REPO_ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    matrix = (REPO_ROOT / "docs" / "native_support_matrix.md").read_text(
        encoding="utf-8"
    )
    for doc in (roadmap, matrix):
        assert "native_cnn_design.md" in doc
