"""
The example questions of the engines, for in-context learning.

Each engine can have a bank of questions with their SQUALL and Datalog
queries, ``examples/<engine>.yaml`` next to this module, and inherits
the banks of the engines it extends. The examples
put in the prompt are those whose question is the most similar to the
user's (:func:`select_examples`), ranked by BM25 over their words and,
given an embeddings provider, by the cosine similarity of the questions'
embeddings too, the two rankings fused. Engines without a bank fall back
on the example queries of their configuration.
"""

import hashlib
import json
import math
import os
import re
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import yaml

from neurolang_apps.utils import engine_registry

EXAMPLES_DIR = Path(__file__).with_name("examples")
_WORD = re.compile(r"[a-z0-9]+")
STOP_WORDS = frozenset(
    "a an and are as at be by do does each for from give have in is it "
    "list of on or show that the their them there these this to what "
    "which who whose with".split()
)


class Example(NamedTuple):
    """A question with its queries, by language (``squall``, ``datalog``)."""

    question: str
    queries: dict

    def query(self, language: str) -> Optional[str]:
        """Return the query of the example in *language*, if any."""
        return self.queries.get(language)


def words(text: str) -> list:
    """Return the words of *text* that tell what it is about."""
    return [
        w for w in _WORD.findall(text.lower())
        if w not in STOP_WORDS
    ]


def bank_path(engine_name: str) -> Path:
    """Return the path of the example bank of *engine_name*."""
    return EXAMPLES_DIR / f"{engine_name}.yaml"


@lru_cache(maxsize=None)
def _load_bank(path: Path) -> tuple:
    """Return the examples of the bank at *path*."""
    with open(path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    return tuple(
        Example(
            entry["question"],
            {
                language: entry[language].strip()
                for language in ("squall", "datalog")
                if entry.get(language)
            },
        )
        for entry in data.get("examples", ())
    )


def _configuration_examples(engine_name: str) -> list:
    """
    Return the example queries of the engine's configuration.

    They have no question: the model sees them as examples of the
    syntax on the engine.
    """
    from neurolang_apps.utils.tui.commands import engine_examples
    examples = []
    for language, queries in engine_examples(engine_name).items():
        examples.extend(Example("", {language: q}) for q in queries)
    return examples


def load_examples(engine_name: str) -> list:
    """
    Return the examples of *engine_name*, from its banks or its config.

    The banks of the engines it ``extends`` come first, then its own:
    the ``neurosynth`` and ``neurostore`` engines share the bank of
    ``coordinate_meta_analysis``.
    """
    if not engine_name:
        return []
    banks = [
        bank_path(name)
        for name in engine_registry.engine_bases(engine_name) + [engine_name]
    ]
    banks = [path for path in banks if path.exists()]
    if banks:
        return [example for path in banks for example in _load_bank(path)]
    if engine_name in engine_registry.list_engine_names():
        try:
            return _configuration_examples(engine_name)
        except ImportError:
            return []
    return []


def bm25_scores(query: str, documents: list, k1=1.5, b=0.75) -> list:
    """Return the BM25 score of each of the *documents* for *query*."""
    tokenized = [words(document) for document in documents]
    if not tokenized:
        return []
    average = sum(map(len, tokenized)) / len(tokenized) or 1.0
    frequency = Counter(w for document in tokenized for w in set(document))
    n = len(tokenized)
    scores = []
    for document in tokenized:
        counts = Counter(document)
        score = 0.0
        for word in set(words(query)):
            if word not in counts:
                continue
            idf = math.log(1 + (n - frequency[word] + 0.5) /
                           (frequency[word] + 0.5))
            tf = counts[word]
            score += idf * tf * (k1 + 1) / (
                tf + k1 * (1 - b + b * len(document) / average)
            )
        scores.append(score)
    return scores


RRF_K = 60
_VECTORS = {}
_FAILED = set()


def _cache_dir() -> Path:
    """Return the directory of the embeddings of the banks."""
    root = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(root).expanduser() / "neurolang" / "llm"


def embed(provider, texts: tuple, disk: bool = False) -> np.ndarray:
    """
    Return the unit embeddings of *texts* by *provider*, kept in memory
    and, with *disk*, in the cache directory.
    """
    key = hashlib.sha256(
        json.dumps([provider.base_url, provider.model, texts]).encode()
    ).hexdigest()[:24]
    path = _cache_dir() / f"embeddings-{key}.npy"
    if key not in _VECTORS and disk and path.exists():
        _VECTORS[key] = np.load(path)
    if key not in _VECTORS:
        from .providers import make_client
        client = make_client(provider)
        try:
            vectors = np.array(client.embed(list(texts)), dtype=float)
        finally:
            client.close()
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        _VECTORS[key] = vectors
        if disk:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, vectors)
            except OSError:
                pass
    return _VECTORS[key]


def similarities(provider, question: str, bank: list) -> Optional[dict]:
    """
    Return the cosine similarity of *question* to each question of
    *bank*, or None, with a warning the first time, if *provider* fails.
    """
    bank = tuple(dict.fromkeys(q for q in bank if q))
    if not bank or (provider.base_url, provider.model) in _FAILED:
        return None
    try:
        vectors = embed(provider, bank, disk=True)
        cosines = vectors @ embed(provider, (question,))[0]
    except Exception as exc:
        _FAILED.add((provider.base_url, provider.model))
        warnings.warn(
            f"The examples are chosen by BM25 only: the embeddings of "
            f"{provider.model!r} are not available ({exc})."
        )
        return None
    return dict(zip(bank, cosines))


def _ranks(scores: list) -> list:
    """Return the rank of each score, tied scores sharing theirs."""
    return [sum(other > score for other in scores) for score in scores]


def select_examples(
    examples: list, question: str, language: str, k: int,
    exclude: str = "", embeddings=None,
) -> list:
    """
    Return the *k* examples in *language* most similar to *question*.

    They are ranked by BM25 or, given an *embeddings* provider, by the
    reciprocal rank fusion of BM25 and of the cosine similarity of the
    questions' embeddings (BM25 alone if those are not available).
    Ties keep the order of the bank, so that its first, simplest
    examples come first. The example whose question is *exclude* is
    left out (to evaluate the assistant on its own bank).
    """
    candidates = [
        example for example in examples
        if example.query(language) and example.question != exclude
    ]
    scores = bm25_scores(question, [e.question for e in candidates])
    cosines = embeddings and similarities(
        embeddings, question, [e.question for e in examples]
    )
    if cosines:
        lexical = _ranks(scores)
        semantic = _ranks([cosines.get(e.question, -1.0) for e in candidates])
        scores = [
            (1 / (RRF_K + lexical[i] + 1) if scores[i] > 0 else 0.0)
            + 1 / (RRF_K + semantic[i] + 1)
            for i in range(len(candidates))
        ]
    ranked = sorted(
        range(len(candidates)), key=lambda i: (-scores[i], i)
    )
    return [candidates[i] for i in ranked[:k]]
