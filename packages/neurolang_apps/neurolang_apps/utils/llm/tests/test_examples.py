import json
import os
from pathlib import Path

import httpx
import pytest

from .. import examples as examples_module
from .. import providers
from ..config import ProviderConfig
from ..examples import (
    Example,
    bm25_scores,
    embed,
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


EMBEDDINGS = ProviderConfig(
    "embeddings", "openai", "embed-x", "https://llm.test/v1"
)
# The fake embeddings: "nociception" is close to "pain", whatever words
# and language say it.
VECTORS = {
    "Which brain regions do memory studies report?": [0, 0, 1],
    "Which voxels do pain studies report?": [0.9, 0.1, 0],
    "How many studies are there?": [0.3, 1, 0],
    "Which studies mention fear?": [0.2, 0, 1],
    "Which studies mention pain?": [1, 0.2, 0],
    "brain regions processing pain nociception": [1, 0, 0],
    "Quelles études parlent de la douleur ?": [1, 0.25, 0],
}
BANK = [Example(q, {"datalog": q}) for q in list(VECTORS)[:5]]


@pytest.fixture
def embedder(monkeypatch, tmp_path):
    """
    Answer the embedding requests with :data:`VECTORS`, recording the
    texts of each request; a fresh cache directory and memory.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(examples_module, "_VECTORS", {})
    monkeypatch.setattr(examples_module, "_FAILED", set())
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body["input"])
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": VECTORS.get(text, [0, 1, 1])}
            for i, text in reversed(list(enumerate(body["input"])))
        ]})

    make_client = providers.make_client
    monkeypatch.setattr(
        providers, "make_client",
        lambda provider, *args, **kwargs: make_client(
            provider, transport=httpx.MockTransport(handler)
        ),
    )
    return requests


def test_hybrid_puts_the_semantic_match_first(embedder):
    question = "brain regions processing pain nociception"
    lexical = select_examples(BANK, question, "datalog", 2)
    assert lexical == [BANK[0], BANK[4]] and embedder == []
    hybrid = select_examples(
        BANK, question, "datalog", 2, embeddings=EMBEDDINGS
    )
    assert hybrid[0] == BANK[1]


def test_hybrid_finds_english_examples_of_a_french_question(embedder):
    question = "Quelles études parlent de la douleur ?"
    assert select_examples(BANK, question, "datalog", 1) == [BANK[0]]
    assert select_examples(
        BANK, question, "datalog", 1, embeddings=EMBEDDINGS
    ) == [BANK[4]]


def test_embeddings_are_cached(embedder):
    select_examples(BANK, "pain", "datalog", 2, embeddings=EMBEDDINGS)
    assert embedder == [[e.question for e in BANK], ["pain"]]
    select_examples(BANK, "pain", "datalog", 2, embeddings=EMBEDDINGS)
    assert len(embedder) == 2
    # A new process reads the bank's embeddings from the disk.
    examples_module._VECTORS.clear()
    select_examples(BANK, "fear", "datalog", 2, embeddings=EMBEDDINGS)
    assert embedder[2:] == [["fear"]]


def test_hybrid_excludes_the_question(embedder):
    question = "Which studies mention pain?"
    chosen = select_examples(
        BANK, question, "datalog", 5, exclude=question,
        embeddings=EMBEDDINGS,
    )
    assert BANK[4] not in chosen and chosen[0] == BANK[1]
    # the bank is embedded whole, whatever is left out
    assert embedder[0] == [e.question for e in BANK]


def test_embeddings_by_batches(embedder):
    texts = tuple(f"question {i}" for i in range(130))
    assert embed(EMBEDDINGS, texts).shape == (130, 3)
    assert [len(batch) for batch in embedder] == [64, 64, 2]


def test_bm25_when_the_embeddings_fail(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(examples_module, "_VECTORS", {})
    monkeypatch.setattr(examples_module, "_FAILED", set())
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("offline", request=request)

    make_client = providers.make_client
    monkeypatch.setattr(
        providers, "make_client",
        lambda provider, *args, **kwargs: make_client(
            provider, transport=httpx.MockTransport(handler)
        ),
    )
    question = "brain regions processing pain nociception"
    lexical = select_examples(BANK, question, "datalog", 3)
    with pytest.warns(UserWarning, match="BM25 only"):
        assert select_examples(
            BANK, question, "datalog", 3, embeddings=EMBEDDINGS
        ) == lexical
    # Warned once; the provider is not asked again.
    assert select_examples(
        BANK, question, "datalog", 3, embeddings=EMBEDDINGS
    ) == lexical
    assert len(calls) == 1


def test_bm25_without_the_key_of_the_embeddings(embedder, monkeypatch):
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    provider = ProviderConfig(
        "embeddings", "openai", "e", "https://llm.test/v1",
        api_key_env="NO_SUCH_KEY",
    )
    with pytest.warns(UserWarning, match="NO_SUCH_KEY"):
        assert select_examples(
            BANK, "pain", "datalog", 1, embeddings=provider
        ) == [BANK[4]]
    assert embedder == []


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
