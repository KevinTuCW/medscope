"""Text embedding for the arbiter's guideline retrieval.

Structurally mirrors `wealthwise.rag.embed` (the sibling project's
equivalent module): a small `Embedder` Protocol plus a dependency-free
offline implementation, so the arbiter's retrieval step never needs a
model, an API key, or a network call to run in tests or in the offline
default configuration.
"""

import hashlib
import math
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class LocalHashingEmbedder:
    """Dependency-free bag-of-words hashing embedder, L2-normalized (offline).

    Not semantic -- tokens hash into fixed buckets -- but needs no model or
    API key, so guideline retrieval runs and tests deterministically. A
    real embedder can be substituted later behind the same `Embedder`
    Protocol without touching `store.py` or `arbiter.py`.

    Blunt edge: tokenizing on `\\w+` treats a punctuation-free run of CJK
    characters as a single token (there's no whitespace between Chinese
    words for the regex to split on), so a whole unpunctuated clause hashes
    into one bucket instead of several. Corpus entries (e.g.
    `data/samples/guidelines.json`) should stay short and
    punctuation-delimited -- commas/periods between clauses -- so related
    passages still share enough small tokens to rank sensibly.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
