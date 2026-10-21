"""
The example questions of the engines, for in-context learning.

Each engine can have a bank of questions with their SQUALL and Datalog
queries, ``examples/<engine>.yaml`` next to this module. The examples
put in the prompt are those whose question is the most similar to the
user's (:func:`select_examples`), ranked by BM25 over their words: no
embedding model is needed. Engines without a bank fall back on the
example queries of their configuration.
"""

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

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
    """Return the examples of *engine_name*, from its bank or its config."""
    if not engine_name:
        return []
    path = bank_path(engine_name)
    if path.exists():
        return list(_load_bank(path))
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


def select_examples(
    examples: list, question: str, language: str, k: int,
    exclude: str = "",
) -> list:
    """
    Return the *k* examples in *language* most similar to *question*.

    Ties keep the order of the bank, so that its first, simplest
    examples come first. The example whose question is *exclude* is
    left out (to evaluate the assistant on its own bank).
    """
    candidates = [
        example for example in examples
        if example.query(language) and example.question != exclude
    ]
    scores = bm25_scores(question, [e.question for e in candidates])
    ranked = sorted(
        range(len(candidates)), key=lambda i: (-scores[i], i)
    )
    return [candidates[i] for i in ranked[:k]]
