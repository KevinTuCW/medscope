"""Tests for medscope.arbiter -- the LLM judge that resolves reader_a/
reader_b disagreements.

The central economic claim under test: the arbiter spends an LLM call
only on a disagreement, never on an already-agreed finding
(`test_only_disagreements_cost_a_call`). Around that: verdict handling
(CONFIRM enters the final set with source="arbiter", REJECT stays out with
a recorded reason, UNCERTAIN is never auto-resolved), the budget cap
(`Settings.max_llm_judgments`) never silently truncating, retrieved
guideline text always reaching the prompt wrapped never raw, and that the
arbiter actually looks at the image -- it gets the same `image_path` every
disagreement call, not just the two readers' reported claims.

All doubles here are hermetic (no network, no key) -- `FakeArbiterClient`
counts and records calls instead of hitting a real model, `FakeRetriever`
returns fixed Docs instead of doing real retrieval.
"""
import json

import pytest

from medscope.arbiter import ArbiterDeps, ArbitrationOutcome, arbitrate
from medscope.config import Settings
from medscope.llm import ModelResponse
from medscope.rag.store import Doc
from medscope.state import Disagreement, Finding

_IMAGE_PATH = "study.png"


class FakeArbiterClient:
    """Test double for arbiter.ArbiterClient (== medscope.llm.ModelClient):
    returns canned verdicts in order, records every prompt and image_path
    it was called with."""

    name = "fake-arbiter"

    def __init__(self, verdicts: list[str], reasoning: str = "canned"):
        self._verdicts = list(verdicts)
        self.calls = 0
        self.prompts: list[str] = []
        self.image_paths: list = []

    def chat_with_image(self, prompt: str, image_path, *, system: str | None = None) -> ModelResponse:
        self.prompts.append(prompt)
        self.image_paths.append(image_path)
        verdict = self._verdicts[self.calls] if self.calls < len(self._verdicts) else "UNCERTAIN"
        self.calls += 1
        return ModelResponse(text=json.dumps({"verdict": verdict, "reasoning": f"canned {verdict}"}))


class FakeRetriever:
    def __init__(self, docs: list[Doc]):
        self._docs = docs

    def search(self, query: str, k: int = 3) -> list[Doc]:
        return self._docs[:k]


def _agreed_findings(n: int) -> list[Finding]:
    return [
        Finding(label=f"Nodule", prob=0.7, source="cnn", raw_label=f"nodule-{i}", evidence_id=f"cnn:nodule-{i}")
        for i in range(n)
    ]


def _disagreement(label: str = "Pneumothorax", a_prob=0.6, b_prob=0.2, kind="presence") -> Disagreement:
    return Disagreement(label=label, a_prob=a_prob, b_prob=b_prob, kind=kind)


def test_only_disagreements_cost_a_call():
    findings = _agreed_findings(5)
    disagreements = [_disagreement("Pneumothorax"), _disagreement("PleuralEffusion", 0.4, 0.7, "presence")]
    client = FakeArbiterClient(["CONFIRM", "REJECT"])
    deps = ArbiterDeps(client=client, retriever=FakeRetriever([]))

    outcome = arbitrate(findings, disagreements, deps, Settings(), _IMAGE_PATH)

    assert client.calls == 2
    assert isinstance(outcome, ArbitrationOutcome)


def test_confirm_adds_finding_with_source_arbiter():
    disagreements = [_disagreement("Pneumothorax")]
    client = FakeArbiterClient(["CONFIRM"])
    deps = ArbiterDeps(client=client, retriever=FakeRetriever([]))

    outcome = arbitrate([], disagreements, deps, Settings(), _IMAGE_PATH)

    assert len(outcome.findings) == 1
    assert outcome.findings[0].label == "Pneumothorax"
    assert outcome.findings[0].source == "arbiter"
    assert outcome.findings[0].needs_human is False


def test_reject_keeps_out_and_records_reason():
    disagreements = [_disagreement("Pneumothorax")]
    client = FakeArbiterClient(["REJECT"])
    deps = ArbiterDeps(client=client, retriever=FakeRetriever([]))

    outcome = arbitrate([], disagreements, deps, Settings(), _IMAGE_PATH)

    assert outcome.findings == []
    assert any("Pneumothorax" in note for note in outcome.notes)
    assert outcome.status is None
    assert outcome.needs_human is False


def test_uncertain_keeps_finding_marks_needs_human_and_holds_study():
    disagreements = [_disagreement("Pneumothorax")]
    client = FakeArbiterClient(["UNCERTAIN"])
    deps = ArbiterDeps(client=client, retriever=FakeRetriever([]))

    outcome = arbitrate([], disagreements, deps, Settings(), _IMAGE_PATH)

    assert len(outcome.findings) == 1
    assert outcome.findings[0].label == "Pneumothorax"
    assert outcome.findings[0].needs_human is True
    assert outcome.needs_human is True
    assert outcome.status == "HELD"


def test_budget_exceeded_marks_remaining_needs_human_caps_calls():
    disagreements = [_disagreement(f"Label{i}") for i in range(5)]
    client = FakeArbiterClient(["CONFIRM", "CONFIRM"])
    deps = ArbiterDeps(client=client, retriever=FakeRetriever([]))
    settings = Settings(max_llm_judgments=2)

    outcome = arbitrate([], disagreements, deps, settings, _IMAGE_PATH)

    assert client.calls == 2
    assert outcome.status == "HELD"
    assert outcome.needs_human is True
    # the 2 arbitrated (both CONFIRM) + 3 unarbitrated all end up in the
    # final set; the unarbitrated ones must be flagged, not silently
    # dropped or silently presented as resolved
    unarbitrated = [f for f in outcome.findings if f.label in {"Label2", "Label3", "Label4"}]
    assert len(unarbitrated) == 3
    for f in unarbitrated:
        assert f.needs_human is True


def test_retrieved_guideline_text_wrapped_not_raw():
    marker = "XYZ_DISTINCT_GUIDELINE_TEXT_MARKER"
    disagreements = [_disagreement("Pneumothorax")]
    client = FakeArbiterClient(["CONFIRM"])
    deps = ArbiterDeps(client=client, retriever=FakeRetriever([Doc(id="g1", text=marker)]))

    arbitrate([], disagreements, deps, Settings(), _IMAGE_PATH)

    prompt = client.prompts[0]
    assert marker in prompt
    assert "<UNTRUSTED_GUIDELINE>" in prompt
    # the marker text must appear inside the wrapper, not as bare untagged text
    wrapped_start = prompt.index("<UNTRUSTED_GUIDELINE>")
    wrapped_end = prompt.index("</UNTRUSTED_GUIDELINE>")
    assert wrapped_start < prompt.index(marker) < wrapped_end


def test_arbiter_receives_the_image_for_every_disagreement():
    disagreements = [_disagreement("Pneumothorax"), _disagreement("PleuralEffusion", 0.4, 0.7, "presence")]
    client = FakeArbiterClient(["CONFIRM", "REJECT"])
    deps = ArbiterDeps(client=client, retriever=FakeRetriever([]))

    arbitrate([], disagreements, deps, Settings(), _IMAGE_PATH)

    # the arbiter is a third reader, not a re-weighing of the other two's
    # claims -- every disagreement call must carry the actual image path,
    # not just the two readers' reported probabilities
    assert client.image_paths == [_IMAGE_PATH, _IMAGE_PATH]


def test_zero_disagreements_zero_calls_findings_unchanged():
    findings = _agreed_findings(3)
    client = FakeArbiterClient([])
    deps = ArbiterDeps(client=client, retriever=FakeRetriever([]))

    outcome = arbitrate(findings, [], deps, Settings(), _IMAGE_PATH)

    assert client.calls == 0
    assert outcome.findings == findings
    assert outcome.needs_human is False
    assert outcome.status is None


def test_budget_goes_to_labels_both_readers_addressed_first():
    """When the budget runs out -- and measured over 40 real studies it
    runs out on essentially every study (12.35 disagreements vs a cap of
    12) -- the *order* is the decision.

    Here the out-of-vocabulary items come first in the input list, so a
    first-come implementation spends the single available call on one of
    them and leaves the genuine two-reader conflict unarbitrated.
    """
    out_of_vocab = [
        Disagreement(label=f"Fibrosis{i}", a_prob=0.9, b_prob=None, kind="unique", in_vocabulary=False)
        for i in range(3)
    ]
    real_conflict = Disagreement(
        label="Pneumothorax", a_prob=0.7, b_prob=0.2, kind="presence", in_vocabulary=True
    )
    client = FakeArbiterClient(["CONFIRM"])
    deps = ArbiterDeps(client=client, retriever=FakeRetriever([]))
    settings = Settings(max_llm_judgments=1)

    outcome = arbitrate([], [*out_of_vocab, real_conflict], deps, settings, _IMAGE_PATH)

    arbitrated = [r for r in outcome.records if r.verdict != "UNARBITRATED"]
    assert [r.label for r in arbitrated] == ["Pneumothorax"]
    # Nothing is dropped by the reordering -- the rest are still recorded.
    assert len(outcome.records) == 4


def test_reordering_does_not_drop_or_duplicate_anything():
    mixed = [
        Disagreement(label="Emphysema", a_prob=0.9, b_prob=None, kind="unique", in_vocabulary=False),
        Disagreement(label="Cardiomegaly", a_prob=0.8, b_prob=0.3, kind="presence"),
        Disagreement(label="LungOpacity", a_prob=0.9, b_prob=None, kind="unique"),
    ]
    client = FakeArbiterClient(["CONFIRM", "CONFIRM", "CONFIRM"])
    deps = ArbiterDeps(client=client, retriever=FakeRetriever([]))

    outcome = arbitrate([], mixed, deps, Settings(), _IMAGE_PATH)

    assert sorted(r.label for r in outcome.records) == ["Cardiomegaly", "Emphysema", "LungOpacity"]
