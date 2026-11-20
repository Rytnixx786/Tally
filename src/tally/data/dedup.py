"""Near-duplicate clustering so paraphrases of one ticket never straddle splits (ADR-005)."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from datasketch import MinHash, MinHashLSH

_PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, strip accents and punctuation, and collapse Bitext {{placeholders}} to one token."""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    t = _PLACEHOLDER.sub(" slot ", t)
    t = _NON_ALNUM.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


def shingles(text: str, k: int) -> set[str]:
    if len(text) <= k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster(
    texts: Sequence[str],
    *,
    shingle_chars: int = 5,
    num_perm: int = 128,
    threshold: float = 0.8,
    seed: int = 1,
) -> list[int]:
    """Return a cluster id per text. Candidate pairs come from MinHash LSH and are confirmed with
    exact Jaccard, so LSH false positives never merge clusters."""
    norm = [normalise(t) for t in texts]
    sh = [shingles(t, shingle_chars) for t in norm]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    uf = _UnionFind(len(texts))
    exact_seen: dict[str, int] = {}
    for i, s in enumerate(sh):
        if norm[i] in exact_seen:
            uf.union(i, exact_seen[norm[i]])
            continue
        exact_seen[norm[i]] = i
        m = MinHash(num_perm=num_perm, seed=seed)
        for g in s:
            m.update(g.encode("utf-8"))
        for j in lsh.query(m):
            jj = int(j)
            if jaccard(s, sh[jj]) >= threshold:
                uf.union(i, jj)
        lsh.insert(str(i), m)
    roots = [uf.find(i) for i in range(len(texts))]
    remap: dict[int, int] = {}
    return [remap.setdefault(r, len(remap)) for r in roots]
