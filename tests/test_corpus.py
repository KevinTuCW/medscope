"""Which studies the workbench can run, and which readers read them.

Two changes are pinned here. The workbench used to be able to run exactly
the three studies committed to the repo, no matter how much of the OpenI
corpus had been fetched; and it read every one of them with
`OfflineVLMClient` no matter how the machine was configured. Both were
defensible for a three-study demo and stop being defensible at ~3.8k.

The corpus is built under `tmp_path` rather than pointed at the real
`data/openi/`: that directory exists on a machine that has run
`scripts/fetch_openi.py` and nowhere else, so a test that reads it passes
or fails depending on whose laptop it is running on. See `conftest.py`,
which forces `OPENI_ROOT` to an absent path for exactly this reason.
"""

from __future__ import annotations

import pytest

from medscope.config import Settings
from medscope import workbench


SAMPLE_IMAGE = "data/samples/studies/images/CXR38_IM-1911-1001.png"

_REPORT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<eCitation>
  <MedlineCitation>
    <Article>
      <Abstract>
        <AbstractText Label="INDICATION">{indication}</AbstractText>
        <AbstractText Label="FINDINGS">{findings}</AbstractText>
        <AbstractText Label="IMPRESSION">{impression}</AbstractText>
      </Abstract>
    </Article>
  </MedlineCitation>
  <MeSH>{mesh}</MeSH>
</eCitation>
"""


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A fake fetched corpus: 6 studies, one image each.

    Ids deliberately include the three pinned ones so the "pinned studies
    float to the top" behaviour can be checked without special-casing.
    """
    import shutil

    reports = tmp_path / "ecgen-radiology"
    images = tmp_path / "images"
    reports.mkdir()
    images.mkdir()

    specs = [
        ("38", "chest pain", "Cardiomegaly"),
        ("797", "shortness of breath", "Pneumothorax"),
        ("1187", "cough", "Effusion"),
        ("2001", "routine screening", "Normal"),
        ("2002", "follow up pneumothorax", "Pneumothorax"),
        ("2003", "", "Atelectasis"),
    ]
    for study_id, indication, mesh in specs:
        mesh_xml = f"<major>{mesh}</major>"
        (reports / f"{study_id}.xml").write_text(
            _REPORT_XML.format(
                indication=indication,
                findings=f"{mesh} noted.",
                impression=f"{mesh}.",
                mesh=mesh_xml,
            ),
            encoding="utf-8",
        )
        shutil.copyfile(SAMPLE_IMAGE, images / f"CXR{study_id}_IM-0001-1001.png")

    monkeypatch.setenv("OPENI_ROOT", str(tmp_path))
    workbench.reset_corpus_cache()
    yield tmp_path
    workbench.reset_corpus_cache()


# --- corpus selection ------------------------------------------------------


def test_corpus_falls_back_to_samples_when_nothing_is_fetched():
    """Offline-first: a fresh clone still gets a usable workbench.

    `conftest` points OPENI_ROOT at an absent path, which is what a machine
    that has never run `fetch_openi.py` looks like.
    """
    assert workbench.corpus_root() == workbench.SAMPLES_DIR
    assert {s.study_id for s in workbench.load_corpus()} == {"38", "797", "1187"}


def test_corpus_uses_the_fetched_dataset_when_present(corpus):
    assert workbench.corpus_root() == corpus
    assert len(workbench.load_corpus()) == 6


def test_corpus_is_cached_per_root(corpus, monkeypatch):
    """A full load walks thousands of files; doing it per keystroke would
    make the search box unusable."""
    first = workbench.load_corpus()
    second = workbench.load_corpus()
    assert first is second


def test_cache_does_not_serve_one_root_from_another(corpus, monkeypatch, tmp_path):
    fetched = workbench.load_corpus()
    assert len(fetched) == 6

    monkeypatch.setenv("OPENI_ROOT", str(tmp_path / "definitely-absent"))
    samples = workbench.load_corpus()
    assert {s.study_id for s in samples} == {"38", "797", "1187"}


# --- search ----------------------------------------------------------------


def test_empty_search_floats_the_pinned_studies_first(corpus):
    page, total = workbench.search_studies()
    assert total == 6
    assert [s.study_id for s in page[:3]] == list(workbench.PINNED_STUDY_IDS)


def test_search_matches_on_study_id(corpus):
    page, total = workbench.search_studies(q="2002")
    assert [s.study_id for s in page] == ["2002"]
    assert total == 1


def test_search_matches_on_indication(corpus):
    page, _total = workbench.search_studies(q="shortness")
    assert [s.study_id for s in page] == ["797"]


def test_search_matches_on_mesh_terms(corpus):
    """The indication is what the referrer wrote; MeSH is what the study
    turned out to show. Hunting for a case that demonstrates a critical
    finding means searching the latter."""
    page, total = workbench.search_studies(q="pneumothorax")
    assert total == 2
    assert {s.study_id for s in page} == {"797", "2002"}


def test_search_is_case_insensitive(corpus):
    upper, _ = workbench.search_studies(q="PNEUMOTHORAX")
    lower, _ = workbench.search_studies(q="pneumothorax")
    assert {s.study_id for s in upper} == {s.study_id for s in lower}


def test_limit_truncates_the_page_but_not_the_total(corpus):
    """"50 studies" and "50 of 812" are different things to show someone,
    and a truncated page cannot tell them apart on its own."""
    page, total = workbench.search_studies(limit=2)
    assert len(page) == 2
    assert total == 6


def test_search_with_no_match_is_empty_not_everything(corpus):
    page, total = workbench.search_studies(q="zzz-no-such-study")
    assert page == []
    assert total == 0


def test_find_study_reaches_beyond_the_committed_samples(corpus):
    study = workbench.find_study("2002")
    assert study.study_id == "2002"

    with pytest.raises(KeyError):
        workbench.find_study("999999")


# --- which readers read them -----------------------------------------------


def test_deps_use_offline_standins_when_real_vlm_is_off(corpus):
    from medscope.readers.vlm import OfflineVLMClient

    study = workbench.find_study("38")
    deps = workbench.deps_for_study(study, Settings(use_real_vlm=False))
    assert isinstance(deps.vlm_client, OfflineVLMClient)


def test_deps_honour_use_real_vlm(corpus):
    """The workbench used to hardwire the offline stand-in, so a machine
    configured for a real VLM still got reader_b's findings derived from
    the study's own ground-truth report -- perfect agreement, displayed in
    the panel built to show whether two readers independently agree."""
    from medscope.llm import OpenAICompatibleModelClient

    study = workbench.find_study("38")
    settings = Settings(
        use_real_vlm=True,
        vlm_model="some-vlm",
        vlm_base_url="https://example.invalid/v1",
        vlm_api_key="k",
    )
    deps = workbench.deps_for_study(study, settings)
    assert isinstance(deps.vlm_client, OpenAICompatibleModelClient)


def test_deps_refuse_to_run_half_configured(corpus):
    """Raises rather than silently falling back -- an operator told a study
    was read by a real VLM, when a stand-in read it, has a false account of
    the reading."""
    study = workbench.find_study("38")
    with pytest.raises(RuntimeError, match="VLM"):
        workbench.deps_for_study(study, Settings(use_real_vlm=True))
