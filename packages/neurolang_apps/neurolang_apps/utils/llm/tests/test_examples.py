import os
from pathlib import Path

import pytest

from ..examples import (
    Example,
    bm25_scores,
    load_examples,
    select_examples,
    words,
)
from ..validation import check_query

BANK_ENGINES = ("neurosynth", "neurostore")


def test_words_drop_stop_words():
    assert words("Which studies mention Pain?") == [
        "studies", "mention", "pain"
    ]


def test_bm25_ranks_the_matching_document_first():
    scores = bm25_scores(
        "voxels reported for pain",
        ["studies on memory", "voxels reported by pain studies", ""],
    )
    assert scores[1] > scores[0] == scores[2] == 0


def test_select_examples():
    examples = [
        Example(
            "Which studies mention pain?", {"squall": "a", "datalog": "b"}
        ),
        Example("How many studies are there?", {"datalog": "c"}),
        Example("Which voxels do pain studies report?", {"squall": "d"}),
    ]
    chosen = select_examples(examples, "voxels of pain", "squall", 1)
    assert chosen == [examples[2]]
    assert select_examples(examples, "count", "squall", 5) == [
        examples[0], examples[2]
    ]
    assert select_examples(
        examples, "pain", "squall", 5, exclude=examples[0].question
    ) == [examples[2]]


@pytest.mark.parametrize("engine", BANK_ENGINES)
def test_banks_load(engine):
    examples = load_examples(engine)
    assert len(examples) >= 5
    for example in examples:
        assert example.question and example.queries
        assert set(example.queries) <= {"squall", "datalog"}


def test_banks_are_inherited():
    # Both engines extend coordinate_meta_analysis and share its bank;
    # neurostore adds examples of its own after it.
    shared = load_examples("coordinate_meta_analysis")
    assert load_examples("neurosynth") == shared
    neurostore = load_examples("neurostore")
    assert neurostore[:len(shared)] == shared
    assert len(neurostore) > len(shared)


def test_engines_without_bank():
    assert load_examples("") == []
    assert load_examples("no-such-engine") == []
    configured = load_examples("destrieux")
    assert configured and not configured[0].question


DATA_DIR = os.environ.get("NEUROLANG_TEST_DATA_DIR")


@pytest.mark.skipif(
    not DATA_DIR,
    reason="set NEUROLANG_TEST_DATA_DIR to the engines' data directory "
    "to check the example banks (builds the engines)",
)
@pytest.mark.parametrize("engine", BANK_ENGINES)
def test_bank_queries_are_valid_on_the_engine(engine):
    from neurolang_apps.utils import engine_registry

    nl = engine_registry.build_engine(engine, Path(DATA_DIR), None)
    failures = []
    for example in load_examples(engine):
        for language, query in example.queries.items():
            result = check_query(nl, query, language)
            if not result.ok:
                failures.append(
                    f"{example.question} [{language}]: {result.message}"
                )
    assert not failures, "\n".join(failures)
