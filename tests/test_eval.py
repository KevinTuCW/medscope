"""Tests for medscope.eval -- the delivery gate (Task 3.2).

Structure:
  1. Each suite computes its metric correctly on a small, controlled fixture.
  2. Four "prove it's not a tautology" tests -- one per hard gate (G1-G4) --
     each breaks the implementation the gate is supposed to be grading and
     asserts the gate is observed going red, with exit code 2. A gate that
     cannot be made to fail is not a gate; these are what distinguish this
     suite from one of the tautological gates this project has shipped
     before (see medscope/eval.py's module docstring).
  3. --min-cases refuses a suspiciously small corpus.
  4. Exit codes: 0 on pass, 2 on hard failure.
  5. The per-label critical breakdown appears in the report.
"""

from __future__ import annotations

import json
import re
from fnmatch import fnmatch
from pathlib import Path

import pytest

from medscope import critical as critical_mod
from medscope import deid as deid_mod
from medscope import evidence as evidence_mod
from medscope import language as language_mod
from medscope.config import Settings
from medscope.eval import (
    CRITICAL_FPR_SOFT_CEILING,
    CRITICAL_GOLDSET_PATH,
    INJECTION_FIXTURE,
    REDLINE_FIXTURE,
    main,
    run_critical_suite,
    run_eval,
    run_evidence_suite,
    run_golden_suite,
    run_language_suite,
    run_phi_suite,
    run_robustness_suite,
)
from medscope.state import Finding, ReadResult, ReportDraft, ReportSentence, StudyState

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Small, hand-built fixtures independent of the real data/evals corpora
# ---------------------------------------------------------------------------


def _write_critical_goldset(tmp_path: Path, cases: list[dict]) -> Path:
    """Write a fixture gold set AND the image files its patterns resolve to.

    The images matter: `run_critical_suite` resolves each case's
    `image_path` pattern against `Settings.openi_root` and skips a case
    whose image is missing. Without a self-contained image directory these
    fixture patterns (`CXR1_*.png` ... `CXR5_*.png`) glob against the real
    Open-i archive, where `CXR1_IM-0001-3001.png` really exists -- so a
    "small, hand-built fixture independent of the real corpora" silently
    started reading real pixels with the real CNN. Pair with
    `_settings_for(tmp_path)` so the glob can only see these files.

    The files are empty on purpose: the tests using them inject
    `_GoldLabelReader`, which never opens them.
    """
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    for case in cases:
        (images / case["image_path"].replace("*", "IM-0001")).write_bytes(b"")
    path = tmp_path / "critical.json"
    path.write_text(json.dumps({"metadata": {"note": "test fixture"}, "cases": cases}))
    return path


def _settings_for(tmp_path: Path) -> Settings:
    return Settings(openi_root=tmp_path)


class _GoldLabelReader:
    """A reader_a stand-in returning exactly the gold labels for a case.

    These tests are about the *harness*: does `triage()` alert on every
    critical label handed to it without dropping, mis-thresholding or
    de-duplicating one away, and does the suite account for the result
    honestly? That is answered by feeding known-correct findings in, and it
    is a different question from "can the CNN see a pneumothorax in this
    image" -- which is what `make eval` measures, by running the real
    `CNNReader` over the real gold set. Keeping the model out of here is
    what keeps the fast suite deterministic and weight-free.

    Negative cases get a confident NON-critical finding, so `critical_fpr`
    exercises triage()'s ontology filter instead of being undefined.
    """

    def __init__(self, cases: list[dict]):
        # Matched by the case's own glob pattern, not by a reconstructed
        # filename: the real archive resolves `CXR1021_*.png` to something
        # like `CXR1021_IM-0013-1001.png`, which no naming convention here
        # could guess.
        self._patterns = [(case["image_path"], case["expected_critical"]) for case in cases]

    def read(self, image_path) -> ReadResult:
        return self.read_study([image_path])

    def read_study(self, image_paths) -> ReadResult:
        # Any film of the study identifies the study -- the suite hands
        # over every film it resolved, not one of them.
        names = [Path(p).name for p in image_paths]
        labels: list[str] = next(
            (
                expected
                for pattern, expected in self._patterns
                if any(fnmatch(name, pattern) for name in names)
            ),
            [],
        )
        if labels:
            findings = [
                Finding(
                    label=label,
                    prob=0.9,
                    source="vlm" if label == "Pneumomediastinum" else "cnn",
                    raw_label=label,
                )
                for label in labels
            ]
        else:
            findings = [Finding(label="Cardiomegaly", prob=0.95, source="cnn", raw_label="Cardiomegaly")]
        return ReadResult(reader="a", findings=findings, latency_ms=0)


def _critical_case(study_id: str, expected: list[str]) -> dict:
    return {
        "study_id": study_id,
        "image_path": f"CXR{study_id}_*.png",
        "image_available_locally": False,
        "expected_critical": expected,
        "source": "openi",
        "note": "test fixture case",
    }


SMALL_CRITICAL_CASES = [
    _critical_case("1", ["Pneumothorax"]),
    _critical_case("2", ["PleuralEffusion"]),
    _critical_case("3", ["Pneumothorax", "PleuralEffusion"]),
    _critical_case("4", []),  # negative
    _critical_case("5", []),  # negative
]


def _finding(label: str, prob: float = 0.8, source: str = "cnn") -> Finding:
    return Finding(label=label, prob=prob, source=source, raw_label=label)


def _clean_state(study_id: str = "s1") -> StudyState:
    findings = [_finding("Cardiomegaly", 0.9)]
    draft = ReportDraft(
        sentences=[
            ReportSentence(text="本检查为胸部正位X线摄片。", section="technique", evidence_ids=[]),
            ReportSentence(
                text="心影增大。", section="findings", evidence_ids=[findings[0].evidence_id]
            ),
            ReportSentence(
                text="考虑心影增大可能。", section="impression", evidence_ids=[findings[0].evidence_id]
            ),
            ReportSentence(
                text="建议结合临床随诊。",
                section="recommendation",
                evidence_ids=[findings[0].evidence_id],
            ),
        ],
        disclaimer="本报告由 AI 生成，仅为草稿，需由执业医师复核后方可使用。",
    )
    return StudyState(
        study_id=study_id,
        image_path="x.png",
        findings=findings,
        report=draft,
        status="DRAFT_READY",
    )


def _bare_assertion_state(study_id: str = "bad1") -> StudyState:
    findings = [_finding("Cardiomegaly", 0.9)]
    draft = ReportDraft(
        sentences=[
            ReportSentence(text="本检查为胸部正位X线摄片。", section="technique", evidence_ids=[]),
            ReportSentence(
                text="心影增大。", section="findings", evidence_ids=[findings[0].evidence_id]
            ),
            # bare assertion: a findings-section claim with NO citation
            ReportSentence(text="另见可疑征象。", section="findings", evidence_ids=[]),
            ReportSentence(
                text="考虑心影增大可能。", section="impression", evidence_ids=[findings[0].evidence_id]
            ),
            ReportSentence(
                text="建议结合临床随诊。",
                section="recommendation",
                evidence_ids=[findings[0].evidence_id],
            ),
        ],
        disclaimer="本报告由 AI 生成，仅为草稿，需由执业医师复核后方可使用。",
    )
    return StudyState(
        study_id=study_id,
        image_path="x.png",
        findings=findings,
        report=draft,
        status="DRAFT_READY",
    )


# ---------------------------------------------------------------------------
# 1. Each suite computes its metric correctly on a small fixture
# ---------------------------------------------------------------------------


def test_critical_suite_computes_recall_and_per_label_breakdown(tmp_path):
    goldset = _write_critical_goldset(tmp_path, SMALL_CRITICAL_CASES)
    result = run_critical_suite(
        _settings_for(tmp_path), goldset_path=goldset, min_cases=1, reader=_GoldLabelReader(SMALL_CRITICAL_CASES)
    )

    assert result.metrics["critical_recall"] == 1.0
    assert result.passed is True
    by_label = result.metrics["critical_recall_by_label"]
    assert by_label["Pneumothorax"] == 1.0
    assert by_label["PleuralEffusion"] == 1.0
    assert by_label["Pneumomediastinum"] is None  # zero cases in this small fixture
    assert result.metrics["critical_fpr"] == 0.0
    assert result.n_cases == 5
    assert result.details["n_positive_cases"] == 4  # case1(1) + case2(1) + case3(2 labels)
    assert result.details["n_negative_cases"] == 2


def test_min_cases_counts_what_ran_not_what_the_goldset_lists(tmp_path):
    # How CI stayed red without anyone being told why. `critical.json` is
    # committed, so n_cases reads 53 on any checkout; whether a case can run
    # depends on the Open-i archive, which is not committed. On a fresh
    # checkout every case is unrunnable and `--min-cases 3` -- added to catch
    # exactly a corpus that shrank -- compared 53 against 3 and said nothing.
    #
    # The recall check reddens here regardless (None != 1.0), so asserting
    # `passed is False` would pass with or without the fix. The failure
    # *message* is what distinguishes them: only a guard that counts runs can
    # say the corpus was empty rather than that recall came out wrong.
    goldset = _write_critical_goldset(tmp_path, SMALL_CRITICAL_CASES)
    empty_archive = tmp_path / "no-archive"
    empty_archive.mkdir()

    result = run_critical_suite(
        Settings(openi_root=empty_archive),
        goldset_path=goldset,
        min_cases=3,
        reader=_GoldLabelReader(SMALL_CRITICAL_CASES),
    )

    assert result.n_cases == 5
    assert result.n_runnable == 0
    assert result.passed is False
    corpus_failures = [f for f in result.failures if "corpus too small" in f]
    assert corpus_failures, "an unreadable corpus must be named as such, not left to the recall check"
    assert "0 runnable case(s) of 5 listed" in corpus_failures[0]


def test_critical_suite_reports_coverage_caveat():
    # Uses the REAL gold set -- confirms the honest-accounting requirement
    # (per-label breakdown + coverage caveat) survives against the actual
    # data/evals/critical.json, not just a synthetic fixture. The reader is
    # still stubbed: what is under test is the suite's accounting of the
    # real corpus, not the CNN's eyesight.
    cases = json.loads(CRITICAL_GOLDSET_PATH.read_text())["cases"]
    result = run_critical_suite(Settings(), reader=_GoldLabelReader(cases))
    assert result.n_cases >= 50
    by_label = result.metrics["critical_recall_by_label"]
    assert set(by_label) == {"Pneumothorax", "PleuralEffusion", "Pneumomediastinum"}
    # Pneumomediastinum's single-case coverage must be visible, not averaged away.
    joined_warnings = " ".join(result.warnings)
    assert "Pneumomediastinum" in joined_warnings
    assert "reader_b" in joined_warnings
    # Image coverage must be *measured and shown*, never a fixed sentence:
    # the previous hardcoded "no confirmed-positive case has a locally
    # available image" became false the moment the full archive was fetched
    # and nothing in the gate could notice. Requiring the ratio in the text
    # is what makes the caveat track reality instead of restating history.
    assert re.search(r"\d+/\d+ confirmed-positive cases", joined_warnings)


def test_evidence_suite_clean_draft_passes():
    result = run_evidence_suite([_clean_state()], min_cases=1)
    assert result.metrics["evidence_coverage"] == 1.0
    assert result.metrics["bare_assertions"] == 0
    assert result.passed is True


def test_evidence_suite_bare_assertion_fails():
    result = run_evidence_suite([_bare_assertion_state()], min_cases=1)
    assert result.metrics["evidence_coverage"] < 1.0
    assert result.metrics["bare_assertions"] == 1
    assert result.passed is False
    assert result.failures  # non-empty


def test_evidence_suite_ignores_studies_with_no_report():
    incomplete = StudyState(study_id="qc-failed", image_path="x.png", status="NEEDS_REPEAT", report=None)
    result = run_evidence_suite([incomplete, _clean_state()], min_cases=1)
    assert result.n_cases == 1  # only the completed study counts


def test_language_suite_block_rate_and_disclaimer_rate():
    result = run_language_suite([_clean_state()], min_cases=1)
    assert result.metrics["redline_block_rate"] == 1.0  # REDLINE_FIXTURE is all real redlines
    assert result.metrics["disclaimer_rate"] == 1.0
    assert result.passed is True


def test_language_suite_missing_disclaimer_fails():
    state = _clean_state()
    state.report.disclaimer = "仅供参考"  # vague, not the required two-fact disclaimer
    result = run_language_suite([state], min_cases=1)
    assert result.metrics["disclaimer_rate"] == 0.0
    assert result.passed is False


def test_phi_suite_computes_zero_leaks_on_real_fixture():
    result = run_phi_suite()
    assert result.n_cases == 20
    assert result.metrics["phi_leaks"] == 0
    assert result.passed is True


def test_golden_suite_is_soft_and_never_gates():
    incomplete = StudyState(study_id="qc-failed", image_path="x.png", status="NEEDS_REPEAT", report=None)
    result = run_golden_suite([incomplete], min_cases=100)  # would hard-fail if gated
    assert result.hard is False
    assert result.passed is True  # soft suites never fail the gate
    assert result.metrics["pass_rate"] == 0.0


def test_robustness_suite_computes_injection_block_rate():
    result = run_robustness_suite(Settings(), samples_dir=Path("data/samples/studies"), min_cases=1)
    assert result.metrics["injection_block_rate"] == 1.0
    assert set(INJECTION_FIXTURE)  # fixture is non-empty and used


# ---------------------------------------------------------------------------
# 2. Prove-not-tautological: break each hard gate, watch it go red
# ---------------------------------------------------------------------------


def test_G1_critical_gate_reddens_when_triage_drops_pneumothorax(tmp_path, monkeypatch):
    goldset = _write_critical_goldset(tmp_path, SMALL_CRITICAL_CASES)

    # Sanity check first: unpatched, this exact fixture is green.
    reader = _GoldLabelReader(SMALL_CRITICAL_CASES)
    settings = _settings_for(tmp_path)
    baseline = run_critical_suite(settings, goldset_path=goldset, min_cases=1, reader=reader)
    assert baseline.passed is True
    assert baseline.metrics["critical_recall"] == 1.0

    real_triage = critical_mod.triage

    def _triage_that_drops_pneumothorax(findings, settings, image_ref=""):
        filtered = [f for f in findings if f.label != "Pneumothorax"]
        return real_triage(filtered, settings, image_ref=image_ref)

    monkeypatch.setattr(critical_mod, "triage", _triage_that_drops_pneumothorax)

    broken = run_critical_suite(settings, goldset_path=goldset, min_cases=1, reader=reader)

    assert broken.passed is False
    assert broken.metrics["critical_recall"] < 1.0
    assert broken.metrics["critical_recall_by_label"]["Pneumothorax"] == 0.0
    assert broken.metrics["critical_recall_by_label"]["PleuralEffusion"] == 1.0  # untouched label unaffected
    assert any("critical_recall" in f for f in broken.failures)

    report = run_eval(
        suites=("critical",),
        settings=settings,
        min_cases=1,
        critical_goldset_path=goldset,
        critical_reader=reader,
    )
    assert report["gate_pass"] is False
    assert "critical" in report["hard_failures"]


def test_G2_evidence_gate_reddens_when_check_evidence_is_blinded(monkeypatch):
    bad_state = _bare_assertion_state()

    # Sanity check first: unpatched, this fixture is correctly caught.
    baseline = run_evidence_suite([bad_state], min_cases=1)
    assert baseline.passed is False

    # Break "the evidence checker": make check_evidence blind to the bare
    # assertion (always reports zero violations), same as neutering a
    # detector in the other three gates.
    monkeypatch.setattr(evidence_mod, "check_evidence", lambda draft, findings: [])

    broken = run_evidence_suite([bad_state], min_cases=1)

    # bare_assertions itself goes blind -- disclosed limitation of gating on
    # that counter alone -- but the gate as a whole still reddens because
    # coverage() is a second, independently-implemented signal gated
    # alongside it (see run_evidence_suite's docstring).
    assert broken.metrics["bare_assertions"] == 0
    assert broken.passed is False
    assert broken.metrics["evidence_coverage"] < 1.0

    report = run_eval(suites=("evidence",), min_cases=1)
    # This report run uses the REAL sample-study pipeline (already-clean
    # drafts per the runtime gate), so it stays green even with
    # check_evidence blinded -- the monkeypatch above only proves the
    # *function*, called directly against a known-bad fixture, is what
    # catches a real defect. Assert the direct-fixture result instead of
    # relying on run_eval's happy-path pipeline output for this assertion.
    assert report["gate_pass"] is True  # sanity: this run's real drafts are genuinely clean


def test_G3_language_gate_reddens_when_a_redline_rule_is_disabled(monkeypatch):
    baseline = run_language_suite([_clean_state()], min_cases=1)
    assert baseline.passed is True
    assert baseline.metrics["redline_block_rate"] == 1.0

    real_detect = language_mod.detect_redlines

    def _detect_without_diagnostic_rule(text):
        return [hit for hit in real_detect(text) if hit.kind != "diagnostic"]

    monkeypatch.setattr(language_mod, "detect_redlines", _detect_without_diagnostic_rule)

    broken = run_language_suite([_clean_state()], min_cases=1)

    assert broken.metrics["redline_block_rate"] < 1.0
    assert broken.passed is False
    assert any("redline_block_rate" in f for f in broken.failures)

    report = run_eval(suites=("language",), min_cases=1)
    assert report["gate_pass"] is False
    assert "language" in report["hard_failures"]


def test_G4_phi_gate_reddens_when_deid_text_is_neutered(monkeypatch):
    baseline = run_phi_suite()
    assert baseline.passed is True
    assert baseline.metrics["phi_leaks"] == 0

    monkeypatch.setattr(deid_mod, "deid_text", lambda text: (text, {}))  # no-op scrubber

    broken = run_phi_suite()

    assert broken.metrics["phi_leaks"] > 0
    assert broken.passed is False
    assert any("phi_leaks" in f for f in broken.failures)

    report = run_eval(suites=("phi",), min_cases=1)
    assert report["gate_pass"] is False
    assert "phi" in report["hard_failures"]
    assert report["results"]["phi"]["metrics"]["phi_leaks"] > 0


def test_G5_robustness_injection_gate_reddens_when_detect_injection_is_neutered(monkeypatch):
    # medscope.guardrails.input does `from medscope.security.sanitize import
    # detect_injection` (a name import, not a module reference), so the
    # patch must target the name as bound in guardrails.input -- patching
    # medscope.security.sanitize.detect_injection itself would silently
    # no-op here, since screen_intake already holds its own reference to
    # the pre-patch function object (same pitfall as G3/G4's break-tests
    # would hit if they patched the wrong module).
    import medscope.guardrails.input as guardrails_input_mod

    baseline = run_robustness_suite(Settings(), samples_dir=Path("data/samples/studies"), min_cases=1)
    assert baseline.passed is True
    assert baseline.metrics["injection_block_rate"] == 1.0

    monkeypatch.setattr(guardrails_input_mod, "detect_injection", lambda text: (False, ""))

    broken = run_robustness_suite(Settings(), samples_dir=Path("data/samples/studies"), min_cases=1)

    assert broken.metrics["injection_block_rate"] == 0.0  # every fixture sample now sails through unflagged
    assert broken.passed is False
    assert any("injection_block_rate" in f for f in broken.failures)

    # Patch is still active here -- confirm the full CLI-level report/exit-code
    # path reddens too, not just the suite function in isolation.
    broken_report = run_eval(suites=("robustness",), min_cases=1)
    assert broken_report["gate_pass"] is False
    assert "robustness" in broken_report["hard_failures"]


def test_G5_robustness_invariance_gate_reddens_when_qc_becomes_perturbation_sensitive(monkeypatch):
    import numpy as np
    from PIL import Image

    import medscope.qc as qc_mod
    from medscope.data.openi import load_studies
    from medscope.qc import QCResult

    baseline = run_robustness_suite(Settings(), samples_dir=Path("data/samples/studies"), min_cases=1)
    assert baseline.passed is True
    assert baseline.metrics["invariance_pass_rate"] == 1.0

    studies = [s for s in load_studies(Path("data/samples/studies")) if s.image_paths]
    reference_image = Image.open(studies[0].image_paths[0]).convert("L")
    reference_std = float(np.asarray(reference_image, dtype=np.float32).std())
    real_check_quality = qc_mod.check_quality

    def _hypersensitive_check_quality(image, settings=None):
        # A QC stage that should be blind to mild noise, but isn't: any
        # image whose pixel std strays even slightly from the exact
        # reference value now hard-fails. The un-perturbed sample image
        # matches the reference exactly (diff 0); the perturbed copy this
        # suite generates never will, by construction of the perturbation.
        result = real_check_quality(image, settings)
        std = float(np.asarray(image.convert("L"), dtype=np.float32).std())
        if abs(std - reference_std) > 1e-6:
            result = QCResult(ok=False, issues=[*result.issues, "perturbation_sensitive"], metrics=result.metrics)
        return result

    monkeypatch.setattr(qc_mod, "check_quality", _hypersensitive_check_quality)

    broken = run_robustness_suite(Settings(), samples_dir=Path("data/samples/studies"), min_cases=1)

    assert broken.metrics["invariance_pass_rate"] < 1.0
    assert broken.passed is False
    assert any("invariance_pass_rate" in f for f in broken.failures)

    report = run_eval(suites=("robustness",), min_cases=1)
    assert report["gate_pass"] is False
    assert "robustness" in report["hard_failures"]


# ---------------------------------------------------------------------------
# 3. --min-cases refuses a suspiciously small corpus
# ---------------------------------------------------------------------------


def test_min_cases_fails_a_small_critical_corpus(tmp_path):
    goldset = _write_critical_goldset(tmp_path, SMALL_CRITICAL_CASES)  # 5 cases
    result = run_critical_suite(
        _settings_for(tmp_path), goldset_path=goldset, min_cases=1000, reader=_GoldLabelReader(SMALL_CRITICAL_CASES)
    )
    assert result.passed is False
    assert any("corpus too small" in f for f in result.failures)


def test_min_cases_does_not_apply_to_soft_golden_suite():
    result = run_golden_suite([_clean_state()], min_cases=1000)
    assert result.passed is True  # soft suites are never failed by --min-cases


# ---------------------------------------------------------------------------
# 4. Exit codes
# ---------------------------------------------------------------------------


def test_main_exits_0_on_pass(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    monkeypatch.chdir(Path(__file__).parent.parent)  # repo root, for relative data/ paths
    code = main(["--suite", "phi", "--report-path", str(report_path)])
    assert code == 0
    assert report_path.exists()
    written = json.loads(report_path.read_text())
    assert written["gate_pass"] is True


def test_main_exits_2_on_hard_failure(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    monkeypatch.chdir(Path(__file__).parent.parent)
    monkeypatch.setattr(deid_mod, "deid_text", lambda text: (text, {}))
    code = main(["--suite", "phi", "--report-path", str(report_path)])
    assert code == 2
    written = json.loads(report_path.read_text())
    assert written["gate_pass"] is False
    assert "phi" in written["hard_failures"]


# ---------------------------------------------------------------------------
# 5. The per-label critical breakdown appears in the report
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_full_report_carries_per_label_critical_breakdown(tmp_path, monkeypatch):
    # `main` takes no reader override on purpose -- it is the CLI the gate
    # actually runs, so this exercises the real CNN over the real gold set
    # and costs ~10 minutes. That is what the `slow` marker is for; the
    # harness-level assertions live in the stubbed tests above.
    monkeypatch.chdir(Path(__file__).parent.parent)
    report_path = tmp_path / "report.json"
    main(["--suite", "critical", "--report-path", str(report_path)])
    written = json.loads(report_path.read_text())
    by_label = written["results"]["critical"]["metrics"]["critical_recall_by_label"]
    assert set(by_label) == {"Pneumothorax", "PleuralEffusion", "Pneumomediastinum"}


def test_critical_suite_requires_1_0_recall_not_none(tmp_path):
    # A goldset with zero positive cases must NOT silently pass G1 --
    # `None == 1.0` is False in Python, so this should already fail; pin it
    # explicitly so a future refactor can't accidentally coerce None to 1.0.
    cases = [_critical_case(str(i), []) for i in range(3)]
    goldset = _write_critical_goldset(tmp_path, cases)
    result = run_critical_suite(Settings(), goldset_path=goldset, min_cases=1)
    assert result.metrics["critical_recall"] is None
    assert result.passed is False


assert CRITICAL_FPR_SOFT_CEILING == 0.35
assert REDLINE_FIXTURE
