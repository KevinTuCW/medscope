"""Tests for medscope.rag -- offline embedding + retrieval used by the
arbiter (Task 2.4) to pull guideline passages for a disagreement.

Structurally mirrors wealthwise's test_rag.py (the sibling project this
module's shape was copied from): the hashing embedder is deterministic and
offline, cosine search over an InMemoryVectorStore returns the closest
passage first, and the guideline corpus loads with every entry carrying
non-empty text.
"""
from medscope.rag.embed import Embedder, LocalHashingEmbedder
from medscope.rag.store import Doc, InMemoryVectorStore, Retriever
from medscope.rag.corpus import load_guideline_retriever, load_guidelines


# --------------------------------------------------------------------------- #
# Embedder                                                                     #
# --------------------------------------------------------------------------- #

def test_embedder_satisfies_protocol():
    e = LocalHashingEmbedder(dim=64)
    assert isinstance(e, Embedder)


def test_embedder_is_deterministic():
    e = LocalHashingEmbedder(dim=128)
    text = "气胸，深沟征，脏层胸膜线"
    assert e.embed(text) == e.embed(text)


def test_embedder_is_l2_normalized():
    e = LocalHashingEmbedder(dim=256)
    v = e.embed("胸腔积液，肋膈角，meniscus")
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6


def test_embedder_differs_for_different_text():
    e = LocalHashingEmbedder(dim=64)
    assert e.embed("气胸") != e.embed("胸腔积液")


# --------------------------------------------------------------------------- #
# InMemoryVectorStore                                                          #
# --------------------------------------------------------------------------- #

def test_store_satisfies_retriever_protocol():
    store = InMemoryVectorStore(LocalHashingEmbedder(dim=256))
    assert isinstance(store, Retriever)


def test_store_retrieves_most_similar_first():
    store = InMemoryVectorStore(LocalHashingEmbedder(dim=256))
    store.add([
        Doc(id="pneumothorax", text="气胸，深沟征，脏层胸膜线可见"),
        Doc(id="effusion", text="胸腔积液，肋膈角变钝，meniscus征"),
        Doc(id="cardiomegaly", text="心影增大，心胸比，床旁AP片"),
    ])
    top = store.search("深沟征 气胸表现", k=1)
    assert top[0].id == "pneumothorax"


def test_search_k_limits_results():
    store = InMemoryVectorStore(LocalHashingEmbedder(dim=128))
    store.add([Doc(id=str(i), text=f"结节 {i} 随访") for i in range(5)])
    assert len(store.search("结节", k=3)) == 3


# --------------------------------------------------------------------------- #
# Corpus loader                                                               #
# --------------------------------------------------------------------------- #

def test_load_guidelines_entries_have_nonempty_text():
    entries = load_guidelines("data/samples")
    assert entries
    for entry in entries:
        assert entry["id"]
        assert entry["text"].strip()


def test_load_guideline_retriever_retrieves_relevant_passage():
    retriever = load_guideline_retriever("data/samples", LocalHashingEmbedder(dim=256))
    top = retriever.search("气胸 深沟征 脏层胸膜线", k=3)
    ids = {d.id for d in top}
    assert ids & {"pneumothorax-deep-sulcus", "pneumothorax-pleural-line"}
