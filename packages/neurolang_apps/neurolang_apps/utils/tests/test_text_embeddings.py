import json
import os
import sys
import urllib.parse

import numpy as np
import pytest

from nl_expr.exceptions import UnboundVariablesError
from neurolang.frontend import NeurolangDL, NeurolangPDL
from neurolang_apps.utils import engine_registry
from neurolang_apps.utils import text_embeddings as te

VOCABULARY = ["reading", "word", "visual", "motor", "hand", "memory"]

DOCUMENTS = {
    "1": ("Reading words", "Visual word form area and reading."),
    "2": ("Hand movements", "Motor cortex and hand motor control."),
    "3": ("Memory", "Hippocampus and memory, with some reading."),
    "4": ("Motor learning", ""),
}


class FakeEncoder:

    """Bag-of-words encoder over a fixed vocabulary."""

    cache_key = "fake-encoder"

    def __init__(self, *args, **kwargs):
        """Accept the Specter2Encoder arguments and count calls."""
        self.model_dir = kwargs.get("model_dir")
        self.document_calls = 0
        self.query_calls = 0

    @staticmethod
    def _vector(text):
        words = text.lower().replace(".", " ").replace(",", " ").split()
        return [sum(w.startswith(v) for w in words) for v in VOCABULARY]

    def encode_documents(self, titles, texts):
        self.document_calls += 1
        return np.array(
            [self._vector(t + " " + a) for t, a in zip(titles, texts)],
            dtype=float,
        )

    def encode_query(self, text):
        self.query_calls += 1
        return np.array(self._vector(text), dtype=float)


def load_documents(keys):
    return {k: DOCUMENTS[k] for k in keys if k in DOCUMENTS}


@pytest.fixture
def index(tmp_path):
    return te.EmbeddingIndex(
        [1, 2, 3, 4], load_documents, FakeEncoder(), tmp_path / "idx.npz"
    )


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

PUBMED_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID Version="1">11</PMID><Article>
<ArticleTitle>Reading <i>words</i></ArticleTitle>
<Abstract><AbstractText Label="AIM">First.</AbstractText>
<AbstractText Label="RESULTS">Second.</AbstractText></Abstract>
</Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID Version="1">12</PMID><Article>
<ArticleTitle>No abstract</ArticleTitle>
</Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_pubmed_xml():
    records = te.parse_pubmed_xml(PUBMED_XML.encode())
    assert records == {
        "11": ("Reading words", "First. Second."),
        "12": ("No abstract", ""),
    }


@pytest.fixture
def fake_post(monkeypatch):
    requests = []

    def post(url, data):
        requests.append(urllib.parse.parse_qs(data.decode()))
        return PUBMED_XML.encode()

    monkeypatch.setattr(te, "_post", post)
    monkeypatch.setattr(te.time, "sleep", lambda s: None)
    monkeypatch.delenv(te.EMAIL_ENV, raising=False)
    monkeypatch.delenv(te.API_KEY_ENV, raising=False)
    return requests


def test_fetch_pubmed_abstracts_incremental_cache(tmp_path, fake_post):
    cache = tmp_path / "pubmed.json"
    records = te.fetch_pubmed_abstracts([11, 13], cache_path=cache)
    assert records == {"11": ("Reading words", "First. Second."),
                       "13": ("", "")}
    assert fake_post[0]["id"] == ["11,13"]
    assert fake_post[0]["tool"] == ["neurolang"]
    assert "email" not in fake_post[0]

    records = te.fetch_pubmed_abstracts([11, 12, 13], cache_path=cache)
    assert len(fake_post) == 2
    assert fake_post[1]["id"] == ["12"]
    assert records["12"] == ("No abstract", "")

    te.fetch_pubmed_abstracts(["12", "11"], cache_path=cache)
    assert len(fake_post) == 2


def test_fetch_pubmed_abstracts_batches(fake_post):
    te.fetch_pubmed_abstracts(range(5), batch_size=2)
    assert [r["id"] for r in fake_post] == [["0,1"], ["2,3"], ["4"]]


def test_fetch_pubmed_abstracts_email_and_key(fake_post, monkeypatch):
    monkeypatch.setenv(te.EMAIL_ENV, "someone@example.org")
    monkeypatch.setenv(te.API_KEY_ENV, "key")
    te.fetch_pubmed_abstracts([11])
    assert fake_post[0]["email"] == ["someone@example.org"]
    assert fake_post[0]["api_key"] == ["key"]
    te.fetch_pubmed_abstracts([11], email="other@example.org")
    assert fake_post[1]["email"] == ["other@example.org"]


def test_load_text_table(tmp_path):
    path = tmp_path / "docs.tsv"
    path.write_text(
        "id\ttitle\tabstract\tother\n"
        "a\tTitle A\tText A\tx\n"
        "b\tTitle B\t\tx\n"
    )
    assert te.load_text_table(path) == {
        "a": ("Title A", "Text A"),
        "b": ("Title B", ""),
    }
    path = tmp_path / "docs.csv"
    path.write_text("doc,name,s1,s2\n7,T,x,y\n")
    assert te.load_text_table(path, "doc", ["name", "s1", "s2"]) == {
        "7": ("T", "x y")
    }


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def test_index_scores_order(index):
    scores = index.scores("visual word reading")
    assert scores.shape == (4,)
    assert np.argmax(scores) == 0
    assert scores[0] > scores[2] > scores[1] >= 0
    np.testing.assert_allclose(np.linalg.norm(index.matrix, axis=1), 1)


def test_index_documents_without_text(tmp_path):
    index = te.EmbeddingIndex(
        [1, 5], load_documents, FakeEncoder(), tmp_path / "i.npz"
    )
    similarities = index.similarities("reading")
    assert similarities[5] == 0
    assert similarities[1] > 0


def test_index_cache_reload_and_invalidation(index, tmp_path):
    index.build()
    assert index.encoder.document_calls == 1
    assert index.cache_path.exists()

    reloaded = te.EmbeddingIndex(
        [1, 2, 3, 4], load_documents, FakeEncoder(), index.cache_path
    )
    np.testing.assert_array_equal(reloaded.matrix, index.matrix)
    assert reloaded.encoder.document_calls == 0

    changed = te.EmbeddingIndex(
        [1, 2, 3], load_documents, FakeEncoder(), index.cache_path
    )
    assert changed.matrix.shape[0] == 3
    assert changed.encoder.document_calls == 1


def test_index_is_lazy(tmp_path):
    def fail(keys):
        raise AssertionError("texts loaded too early")

    index = te.EmbeddingIndex([1], fail, FakeEncoder(), tmp_path / "i.npz")
    te.make_similarity_function(index)
    assert not index.cache_path.exists()


def test_encoder_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "adapters", None)
    with pytest.raises(ImportError, match="neurolang\\[embeddings\\]"):
        te.Specter2Encoder().encode_query("reading")


# ---------------------------------------------------------------------------
# NeuroLang integration
# ---------------------------------------------------------------------------


STUDIES = [(1,), (2,), (3,), (4,)]

# The rule and query of the issue: the query text reaches the rule only
# through magic sets, which bind ``w`` to 'reading'.
BOUND_BY_QUERY = """
qqqq(w, s) :- study(s) & (sim == query_similarity(w, s)) & (sim > .5)
ans(s) :- qqqq('reading', s)
"""


def _engine(frontend, index):
    nl = frontend()
    nl.add_tuple_set(STUDIES, name="study")
    nl.add_symbol(te.make_similarity_function(index), name="query_similarity")
    return nl


def test_similarity_function():
    index = te.EmbeddingIndex([1, 2, 3, 4], load_documents, FakeEncoder())
    query_similarity = te.make_similarity_function(index)
    assert query_similarity.__annotations__ == {
        "query": str, "doc": int, "return": float
    }
    scores = index.scores("visual word reading")
    assert [query_similarity("visual word reading", d) for d in index.ids] \
        == pytest.approx(scores.tolist())
    assert np.isnan(query_similarity("visual word reading", 5))
    assert index.encoder.query_calls == 2


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_similarity_function_with_constant_query(frontend, index):
    nl = _engine(frontend, index)
    res = nl.execute_datalog_program(
        "ans(s, sim) :- study(s) & "
        "(sim == query_similarity('visual word reading', s))"
    )
    scores = index.scores("visual word reading")
    assert dict(res) == pytest.approx(dict(zip(index.ids, scores)))


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_similarity_function_query_bound_by_magic_sets(frontend, index):
    nl = _engine(frontend, index)
    assert set(nl.execute_datalog_program(BOUND_BY_QUERY)) == {(1,)}
    # The rule and the query submitted separately, as in the TUI.
    nl = _engine(frontend, index)
    nl.execute_datalog_program(BOUND_BY_QUERY.splitlines()[1])
    res = nl.execute_datalog_program("ans(s) :- qqqq('motor hand', s)")
    assert set(res) == {(2,), (4,)}


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_similarity_function_unbound_query(frontend, index):
    nl = _engine(frontend, index)
    nl.execute_datalog_program(BOUND_BY_QUERY.splitlines()[1])
    with pytest.raises(UnboundVariablesError, match=r"In qqqq: .*\bw\b"):
        nl.execute_datalog_program("ans(w, s) :- qqqq(w, s)")


@pytest.mark.parametrize("frontend", [NeurolangDL, NeurolangPDL])
def test_similarity_function_several_queries(frontend, index):
    nl = _engine(frontend, index)
    res = nl.execute_datalog_program("""
    close(w, s) :- study(s) & (sim == query_similarity(w, s)) & (sim > .5)
    both(s) :- close('reading', s) & close('visual word', s)
    either(s) :- close('memory', s)
    either(s) :- close('motor hand', s)
    ans(s, t) :- both(s) & either(t)
    """)
    assert set(res) == {(1, 2), (1, 3), (1, 4)}
    assert index.encoder.query_calls == 4



# Query texts from a relation, e.g. the descriptions of the Cognitive Atlas
# tasks: ``task_query`` binds ``q`` in the body of ``cutoff`` and ``close``.
TASKS = [
    ("t1", "motor task", "Motor hand movements"),
    ("t2", "reading task", "Reading visual words"),
]
REPORTS = [("a", 1), ("b", 2), ("c", 4)]
QUERY_TEXT_RULES = """
cutoff(q, percentile(sim, 50)) :- {range}study(s) & (sim == query_similarity(q, s))
close(q, s) :- {range}study(s) & (sim == query_similarity(q, s)) & cutoff(q, c) & (sim >= c)
voxel_given_sim(q, v, PROB) :- (reports(v, s) & selected_study(s)) // (close(q, s) & selected_study(s))
"""
TASK_QUERY = 'ans(v, p) :- task(_, "motor task", d) & voxel_given_sim(d, v, p)'


def _task_engine(index):
    nl = _engine(NeurolangPDL, index)
    nl.add_tuple_set(TASKS, name="task")
    nl.add_tuple_set(REPORTS, name="reports")
    nl.add_uniform_probabilistic_choice_over_set(
        STUDIES, name="selected_study"
    )
    return nl


def test_query_texts_joined_in_the_query(index):
    """The description of a task, joined with ``voxel_given_sim`` in the
    query, is passed by magic sets to ``cutoff`` and ``close``."""
    nl = _task_engine(index)
    nl.execute_datalog_program(QUERY_TEXT_RULES.format(range=""))
    res = nl.execute_datalog_program(TASK_QUERY)
    assert dict(res) == pytest.approx({"b": .5, "c": .5})
    assert index.encoder.query_calls == 1


def test_query_texts_from_a_relation(index):
    nl = _task_engine(index)
    nl.execute_datalog_program(
        'task_query(d) :- task(_, "motor task", d)\n'
        + QUERY_TEXT_RULES.format(range="task_query(q) & ")
    )
    res = nl.execute_datalog_program(TASK_QUERY)
    assert dict(res) == pytest.approx({"b": .5, "c": .5})
    assert index.encoder.query_calls == 1

def test_similarity_function_is_cached(index):
    query_similarity = te.make_similarity_function(index)
    similarities = index.similarities("visual word reading")
    assert query_similarity("visual word reading", 1) == similarities[1]
    assert index.similarities("visual word reading") is similarities
    with pytest.raises(TypeError):
        similarities[1] = 0.
    for doc in index.ids:
        query_similarity("visual word reading", doc)
        query_similarity("motor hand", doc)
    assert index.encoder.query_calls == 2
    for i in range(16):
        query_similarity(f"query {i}", 1)
    query_similarity("visual word reading", 1)
    assert index.encoder.query_calls == 19
    assert index.similarities.cache_info().currsize == 16


def test_similarity_function_is_lazy(tmp_path):
    def fail(keys):
        raise AssertionError("texts loaded too early")

    index = te.EmbeddingIndex([1], fail, FakeEncoder(), tmp_path / "i.npz")
    nl = _engine(NeurolangPDL, index)
    nl.execute_datalog_program(BOUND_BY_QUERY.splitlines()[1])
    assert index.encoder.query_calls == 0
    assert not index.cache_path.exists()


# ---------------------------------------------------------------------------
# Engine YAML
# ---------------------------------------------------------------------------


@pytest.fixture
def engine_yaml(tmp_path, monkeypatch):
    (tmp_path / "study.tsv").write_text("study_id\n1\n2\n3\n4\n")
    lines = ["id\ttitle\tabstract"]
    lines += [f"{k}\t{t}\t{a}" for k, (t, a) in DOCUMENTS.items()]
    (tmp_path / "docs.tsv").write_text("\n".join(lines) + "\n")
    path = tmp_path / "engines.yaml"
    path.write_text(
        """
engines:
  mini:
    description: "Mini"
    relations:
      study: study.tsv
    text_embeddings:
      docs:
        ids_from: study
        source: {file: docs.tsv, id_column: id,
                 text_columns: [title, abstract]}
        function: query_similarity
        description: "Doc similarity"
"""
    )
    monkeypatch.setattr(te, "Specter2Encoder", FakeEncoder)
    return path


def test_engine_yaml_text_embeddings(engine_yaml, tmp_path):
    nl = engine_registry.build_engine(
        "mini", tmp_path / "data", yaml_path=engine_yaml
    )
    index = nl.text_embedding_indexes["docs"]
    assert index.encoder.document_calls == 0
    assert index.cache_path == tmp_path / "data/mini/embeddings/docs.npz"
    assert index.encoder.model_dir == tmp_path / "data/mini/embeddings/models"
    assert not index.cache_path.exists()

    res = nl.execute_datalog_program(
        "ans(s) :- study(s) & (x == query_similarity('motor hand', s)) "
        "& (x > 0.5)"
    )
    assert set(res) == {(2,), (4,)}
    assert index.cache_path.exists()

    # A function, like the builtins, is not a declared predicate.
    preds = engine_registry.get_predicates("mini", yaml_path=engine_yaml)
    assert "query_similarity" not in preds


def test_engine_yaml_unknown_source(engine_yaml, tmp_path):
    engine_yaml.write_text(
        engine_yaml.read_text().replace(
            "source: {file: docs.tsv, id_column: id,\n"
            "                 text_columns: [title, abstract]}",
            "source: arxiv",
        )
    )
    with pytest.raises(ValueError, match="text_embeddings source"):
        engine_registry.build_engine(
            "mini", tmp_path / "data", yaml_path=engine_yaml
        )


def test_neurosynth_yaml_declares_similarity_function():
    cfg = engine_registry.get_engine_config("neurosynth")
    entry = cfg["text_embeddings"]["study_abstracts"]
    assert entry["function"] == "query_similarity"
    assert "query_similarity" not in engine_registry.get_predicates(
        "neurosynth"
    )


def test_engine_yaml_albert_encoder(engine_yaml, tmp_path, monkeypatch):
    monkeypatch.setattr(te, "AlbertQwenEncoder", FakeEncoder)
    monkeypatch.setenv(te.ENCODER_ENV, "albert-qwen")
    nl = engine_registry.build_engine(
        "mini", tmp_path / "data", yaml_path=engine_yaml
    )
    index = nl.text_embedding_indexes["docs"]
    # Its own cache, next to the SPECTER2 one.
    assert index.cache_path == (
        tmp_path / "data/mini/embeddings/docs.albert-qwen.npz"
    )
    res = nl.execute_datalog_program(
        "ans(s) :- study(s) & (x == query_similarity('motor hand', s)) "
        "& (x > 0.5)"
    )
    assert set(res) == {(2,), (4,)}


# ---------------------------------------------------------------------------
# Albert encoder (HTTP calls replaced by a fake server)
# ---------------------------------------------------------------------------


class FakeResponse:

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def albert_server(monkeypatch):
    """Record the request bodies; answer with one vector per input."""
    requests = []

    def urlopen(request, timeout=None):
        body = json.loads(request.data)
        requests.append((request, body))
        n = len(body["input"]) if "input" in body else 1
        # Out of order, as the API may answer.
        data = [{"index": i, "embedding": [float(i), 1.0]} for i in range(n)]
        return FakeResponse({"data": data[::-1]})

    monkeypatch.setattr(te.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv(te.ALBERT_KEY_ENV, "sk-test")
    return requests


def test_albert_documents_are_batched_raw_inputs(albert_server):
    encoder = te.AlbertQwenEncoder(batch_size=2)
    out = encoder.encode_documents(["A", "B", "C"], ["a", "", "c"])
    assert out.tolist() == [[0, 1], [1, 1], [0, 1]]
    bodies = [body for _, body in albert_server]
    assert [b["input"] for b in bodies] == [["A\n\na", "B"], ["C\n\nc"]]
    assert all(b["model"] == "qwen3-vl-embedding-8b" for b in bodies)
    request = albert_server[0][0]
    assert request.full_url == te.ALBERT_URL + "/embeddings"
    assert request.get_header("Authorization") == "Bearer sk-test"


def test_albert_query_has_the_instruction(albert_server):
    encoder = te.AlbertQwenEncoder(query_instruction="Find studies.")
    assert encoder.encode_query("reading").tolist() == [0, 1]
    body = albert_server[0][1]
    assert "input" not in body and body["continue_final_message"]
    system, user, assistant = body["messages"]
    assert system == {"role": "system", "content": "Find studies."}
    assert user["content"] == [{"type": "text", "text": "reading"}]
    assert assistant == {"role": "assistant", "content": ""}


def test_albert_retries_rate_limits(albert_server, monkeypatch):
    calls = []
    answer = te.urllib.request.urlopen

    def urlopen(request, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise te.urllib.error.HTTPError(
                request.full_url, 429, "Too Many Requests",
                {"Retry-After": "3"}, None,
            )
        return answer(request, timeout)

    sleeps = []
    monkeypatch.setattr(te.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(te.time, "sleep", sleeps.append)
    te.AlbertQwenEncoder().encode_query("reading")
    assert len(calls) == 2 and sleeps == [3.0]


def test_albert_other_errors_are_raised(albert_server, monkeypatch):
    def urlopen(request, timeout=None):
        raise te.urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", {}, None
        )

    monkeypatch.setattr(te.urllib.request, "urlopen", urlopen)
    with pytest.raises(te.urllib.error.HTTPError):
        te.AlbertQwenEncoder().encode_query("reading")


def test_albert_needs_a_key(monkeypatch):
    monkeypatch.delenv(te.ALBERT_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=te.ALBERT_KEY_ENV):
        te.AlbertQwenEncoder().download()


def test_make_encoder(monkeypatch, tmp_path):
    monkeypatch.delenv(te.ENCODER_ENV, raising=False)
    config = {"model": "allenai/specter2_base", "albert": {"batch_size": 8}}
    specter = te.make_encoder(config, tmp_path)
    assert isinstance(specter, te.Specter2Encoder)
    assert specter.model_dir == tmp_path
    albert = te.make_encoder({**config, "encoder": "albert-qwen"})
    # The SPECTER2 model name does not reach the Albert encoder.
    assert albert.model_name == "qwen3-vl-embedding-8b"
    assert albert.batch_size == 8
    assert specter.cache_key != albert.cache_key
    monkeypatch.setenv(te.ENCODER_ENV, "albert-qwen")
    assert isinstance(te.make_encoder(config), te.AlbertQwenEncoder)
    monkeypatch.setenv(te.ENCODER_ENV, "word2vec")
    with pytest.raises(ValueError, match="Unknown text encoder"):
        te.make_encoder(config)


# ---------------------------------------------------------------------------
# Real model (downloads SPECTER2; run with --runslow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_specter2_ranks_reading_paper_first():
    pytest.importorskip("adapters")
    encoder = te.Specter2Encoder(device="cpu")
    index = te.EmbeddingIndex(
        ["reading", "motor"],
        lambda keys: {
            "reading": (
                "Neural correlates of reading words",
                "fMRI of the visual word form area during reading.",
            ),
            "motor": (
                "Finger tapping",
                "Primary motor cortex activation during finger movements.",
            ),
        },
        encoder,
    )
    similarities = index.similarities("reading")
    assert similarities["reading"] > similarities["motor"]


@pytest.mark.slow
def test_albert_ranks_reading_paper_first():
    if not os.environ.get(te.ALBERT_KEY_ENV):
        pytest.skip(f"needs {te.ALBERT_KEY_ENV}")
    index = te.EmbeddingIndex(
        ["reading", "motor"],
        lambda keys: {
            "reading": (
                "Neural correlates of reading words",
                "fMRI of the visual word form area during reading.",
            ),
            "motor": (
                "Finger tapping",
                "Primary motor cortex activation during finger movements.",
            ),
        },
        te.AlbertQwenEncoder(),
    )
    # A single word: the instruction is what makes it a good query.
    similarities = index.similarities("reading")
    assert similarities["reading"] > similarities["motor"]


# ---------------------------------------------------------------------------
# Local copies of the model, for offline use
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_hub(monkeypatch):
    """Replace huggingface_hub by a stub recording the downloads."""
    calls = []

    def snapshot_download(repo_id, local_dir, allow_patterns):
        calls.append(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

    module = type(sys)("huggingface_hub")
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def test_local_model_is_downloaded_once(fake_hub, tmp_path):
    path = te.local_model("allenai/specter2", tmp_path)
    assert path == tmp_path / "allenai--specter2"
    assert (path / "config.json").exists()
    assert te.local_model("allenai/specter2", tmp_path) == path
    assert fake_hub == ["allenai/specter2"]


def test_interrupted_download_is_resumed(fake_hub, tmp_path):
    (tmp_path / "allenai--specter2").mkdir()
    te.local_model("allenai/specter2", tmp_path)
    assert fake_hub == ["allenai/specter2"]
    assert (tmp_path / "allenai--specter2" / te.COMPLETE_MARKER).exists()


def test_encoder_downloads_its_model_and_adapters(fake_hub, tmp_path):
    encoder = te.Specter2Encoder(model_dir=tmp_path)
    paths = encoder.download()
    assert paths == tuple(
        tmp_path / name for name in (
            "allenai--specter2_base",
            "allenai--specter2",
            "allenai--specter2_adhoc_query",
        )
    )
    encoder.download()
    assert len(fake_hub) == 3


def test_encoder_without_model_dir_uses_the_hub(fake_hub):
    encoder = te.Specter2Encoder()
    assert encoder.download() == (
        "allenai/specter2_base",
        "allenai/specter2",
        "allenai/specter2_adhoc_query",
    )
    assert fake_hub == []
