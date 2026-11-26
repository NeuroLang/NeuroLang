import os
import re
from pathlib import Path

import pandas as pd
import pytest

from neurolang_apps.utils import engine_registry

from ..examples import (
    Example,
    bm25_scores,
    load_examples,
    select_examples,
    words,
)
from ..validation import check_query

# Before the session fixture of ../../conftest.py replaces it by a small file.
REAL_ONTOLOGY_FILE = engine_registry._ontology_file

BANK_ENGINES = ("structural_connectivity", "functional_connectivity",
                "neurosynth", "neurostore")


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
    # ...which begins with those of the connectivities it extends
    # (those of the atlases they both extend, connectivity_atlases, once)
    connectivity = []
    for example in (load_examples("structural_connectivity")
                    + load_examples("functional_connectivity")):
        if example not in connectivity:
            connectivity.append(example)
    assert shared[:len(connectivity)] == connectivity
    assert len(shared) > len(connectivity)
    neurostore = load_examples("neurostore")
    assert neurostore[:len(shared)] == shared
    assert len(neurostore) > len(shared)


ANSWER_HEAD = re.compile(r"^\s*ans\s*(?:\[\s*\w+\s*\])?\s*\(([^)]*)\)", re.M)
VOXEL_INDICES = {"i", "j", "k"}


@pytest.mark.parametrize("engine", ("neurostore", "neurosynth_iit"))
def test_bank_answers_name_their_columns(engine):
    """
    The columns of the answers of the examples say what they hold,
    ans(region, probability) and not ans(l, p): the model writes its
    answers as the examples do. The voxel indices keep their names.
    """
    short = []
    for example in load_examples(engine):
        heads = ANSWER_HEAD.findall(example.query("datalog") or "")
        assert heads, example.question
        for head in heads:
            names = {n.strip() for n in head.split(",") if n.strip()}
            if any(len(n) == 1 for n in names - VOXEL_INDICES):
                short.append(f"{example.question}: ans({head})")
    assert not short, "\n".join(short)


def test_the_expectation_examples_parse():
    """The examples of EXPECT are written in the syntax the parser reads."""
    from neurolang.frontend.datalog.standard_syntax import parser

    expectations = [
        example for example in load_examples("neurosynth")
        if "EXPECT" in (example.query("datalog") or "")
    ]
    assert len(expectations) >= 2
    for example in expectations:
        program = parser(example.query("datalog").replace(
            "{kernel_radius_mm}", "6"
        ))
        assert program is not None, example.question


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
@pytest.mark.parametrize("engine", BANK_ENGINES + ("neurosynth_iit",))
def test_bank_queries_are_valid_on_the_engine(engine):
    nl = engine_registry.build_engine(engine, Path(DATA_DIR), None)
    # What .image loads: the questions on an image need one, on the
    # engines of a brain mask (those of peaks of studies).
    if "brain_mask" in nl.symbols:
        voxels = pd.DataFrame(list(nl.symbols["brain_mask"].value)[:10],
                              columns=["i", "j", "k"])
        nl.add_tuple_set(voxels.assign(value=4.0), name="image")
        nl.add_tuple_set(voxels.assign(p=0.9), name="image_probability")
    failures = []
    for example in load_examples(engine):
        for language, query in example.queries.items():
            result = check_query(nl, query, language)
            if not result.ok:
                failures.append(
                    f"{example.question} [{language}]: {result.message}"
                )
    assert not failures, "\n".join(failures)


@pytest.mark.skipif(
    not DATA_DIR,
    reason="set NEUROLANG_TEST_DATA_DIR to the engines' data directory "
    "to run the start-up examples (builds the engine)",
)
def test_startup_examples_return_rows(monkeypatch):
    """
    The examples shown to every user of the engine (its ``examples``,
    not the assistant's bank) answer with rows: one that returns nothing
    teaches nothing, as the Cognitive Atlas subclasses of a concept none
    of whose subclasses is a term of the studies.
    """
    # The Cognitive Atlas of the data directory, not the small file of the
    # tests: what its subclasses are is what the example is about.
    monkeypatch.setattr(
        engine_registry, "_ontology_file", REAL_ONTOLOGY_FILE
    )
    nl = engine_registry.build_engine("neurosynth", Path(DATA_DIR), None)
    config = engine_registry.get_engine_config("neurosynth")
    empty = []
    for query in config["examples"]["datalog"]:
        if "uniformity_llm" in query:  # embeds the question: needs a model
            continue
        with nl.scope:
            result = nl.execute_datalog_program(query)
        if not len(result.as_pandas_dataframe()):
            empty.append(query)
    assert not empty, "\n".join(empty)
