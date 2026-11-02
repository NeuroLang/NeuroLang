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


def test_encoders_are_prepared_for_fork(monkeypatch, tmp_path):
    """On macOS a forked child, as the assistant checks queries in, cannot
    load a model nor use MPS: before the assistant forks, the encoders with
    local copies are loaded and moved to the CPU; the others are left
    alone, and nothing is done on other systems."""
    loads = []

    class Model:
        def to(self, device):
            loads.append(device)
            return self

    def fake_load(self):
        if self._model is None:
            self._model = Model()
            loads.append(("load", self.device))

    monkeypatch.setattr(te.Specter2Encoder, "_load", fake_load)
    local = te.Specter2Encoder(model_dir=tmp_path, device="mps")
    for repo in (local.model_name, local.document_adapter, local.query_adapter):
        copy = tmp_path / repo.replace("/", "--")
        copy.mkdir()
        (copy / te.COMPLETE_MARKER).touch()
    remote = te.Specter2Encoder(device="mps")       # would download
    monkeypatch.setattr(te.sys, "platform", "darwin")
    te.prepare_for_fork()
    assert loads == [("load", "mps"), "cpu"]
    assert local.device == "cpu" and isinstance(local._model, Model)
    assert remote._model is None and remote.device == "mps"
    monkeypatch.setattr(te.sys, "platform", "linux")
    other = te.Specter2Encoder(model_dir=tmp_path, device="mps")
    te.prepare_for_fork()
    assert other._model is None


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


def test_torch_runs_on_one_thread_in_forked_children(monkeypatch):
    """The OpenMP pool of torch does not survive a fork: a child using it
    hangs, as the assistant's query checks did."""
    calls = []
    torch = type(sys)("torch")
    torch.set_num_threads = calls.append
    monkeypatch.setitem(sys.modules, "torch", torch)
    te._after_fork_in_child()
    assert calls == [1]
    monkeypatch.delitem(sys.modules, "torch")
    te._after_fork_in_child()                   # without torch: nothing
    assert calls == [1]


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


# ---------------------------------------------------------------------------
# Without torch, as in a browser
# ---------------------------------------------------------------------------


def _numpy_query_model(encoder, rng_seed=0):
    from neurolang_apps.utils import specter2_numpy as sn
    from neurolang_apps.utils.tests.test_specter2_numpy import (
        VOCAB, tiny_weights,
    )

    encoder.numpy_path.parent.mkdir(parents=True, exist_ok=True)
    sn.save(
        encoder.numpy_path, tiny_weights(np.random.default_rng(rng_seed)),
        VOCAB, heads=2,
    )


def test_without_torch_queries_are_encoded_with_numpy(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", None)
    encoder = te.Specter2Encoder(model_dir=tmp_path)
    _numpy_query_model(encoder)
    vector = encoder.encode_query("visual word forms")
    assert vector.shape == (8,) and np.isfinite(vector).all()


def test_without_torch_nor_numpy_model_the_error_says_what_to_install(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="neurolang\\[embeddings\\]"):
        te.Specter2Encoder(model_dir=tmp_path).encode_query("reading")


def test_missing_files_are_fetched_on_demand(monkeypatch, tmp_path):
    """In a browser the model and the document embeddings are fetched
    from the page's site when a query first needs them."""
    monkeypatch.setitem(sys.modules, "torch", None)
    encoder = te.Specter2Encoder(model_dir=tmp_path / "models")
    source = te.Specter2Encoder(model_dir=tmp_path / "site")
    _numpy_query_model(source)
    fetched = []

    def fetch(path):
        fetched.append(path.name)
        if path == encoder.numpy_path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.numpy_path.read_bytes())

    monkeypatch.setattr(te, "fetch_missing", fetch)
    encoder.encode_query("reading")
    assert fetched == [encoder.numpy_path.name]
    index = te.EmbeddingIndex([1], load_documents, encoder,
                              tmp_path / "idx.npz")
    with pytest.raises(ImportError):
        index.build()       # no cache fetched: documents need torch
    assert fetched[-1] == "idx.npz"


def test_in_a_browser_a_missing_model_says_the_page_was_built_without(
    monkeypatch, tmp_path
):
    """pip cannot install torch in a browser: the model is missing because
    the page's bundle was built without the text embeddings."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(te.sys, "platform", "emscripten")
    with pytest.raises(ImportError, match="built without") as error:
        te.Specter2Encoder(model_dir=tmp_path).encode_query("reading")
    assert "pip install" not in str(error.value)
