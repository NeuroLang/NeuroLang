"""Tests for the script that builds the bundle of the REPL in the browser."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")

SCRIPT = Path(__file__).resolve().parents[6] / "doc" / "make_repl_bundle.py"


@pytest.fixture
def bundle():
    spec = importlib.util.spec_from_file_location("make_repl_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_wheels_leave_no_gitignore_that_would_hide_the_bundle(
    bundle, monkeypatch, tmp_path
):
    """
    uv writes a .gitignore of ``*`` in the directory it builds in, and the
    documentation is deployed by committing that directory: the page and its
    bundle would not be committed. The wheels are those of all the packages,
    the engine first, as the REPL installs them in that order.
    """
    def uv_build(command):
        assert "--all-packages" in command
        out = Path(command[command.index("--out-dir") + 1])
        for name in ("neurolang_apps", "neurolang", "nl_core", "nl_expr"):
            (out / f"{name}-1-py3-none-any.whl").write_bytes(b"wheel")
        (out / ".gitignore").write_text("*")

    monkeypatch.setattr(bundle.subprocess, "check_call", uv_build)
    (tmp_path / "page.html").write_text("<html>")
    assert bundle.build_wheels(tmp_path) == [
        "nl_expr-1-py3-none-any.whl",
        "nl_core-1-py3-none-any.whl",
        "neurolang-1-py3-none-any.whl",
        "neurolang_apps-1-py3-none-any.whl",
    ]
    assert not (tmp_path / ".gitignore").exists()
    assert (tmp_path / "page.html").exists()


def test_a_certificate_error_is_retried_with_the_system_certificates(
    bundle, monkeypatch, tmp_path
):
    from neurolang_apps.utils import engine_registry

    system = tmp_path / "ca.pem"
    system.write_text("certificates")
    monkeypatch.setattr(bundle, "SYSTEM_CERTIFICATES", (system,))
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    calls = []

    def build(name, data_dir, resolution, use_cache=True):
        calls.append(os.environ.get("REQUESTS_CA_BUNDLE"))
        if len(calls) == 1:
            raise requests.exceptions.SSLError("certificate has expired")

    monkeypatch.setattr(engine_registry, "build_engine", build)
    bundle.build_engine(tmp_path)
    assert calls == [None, str(system)]
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)


def test_without_system_certificates_the_error_is_raised(
    bundle, monkeypatch, tmp_path
):
    from neurolang_apps.utils import engine_registry

    monkeypatch.setattr(bundle, "SYSTEM_CERTIFICATES", (tmp_path / "none",))

    def build(name, data_dir, resolution, use_cache=True):
        raise requests.exceptions.SSLError("certificate has expired")

    monkeypatch.setattr(engine_registry, "build_engine", build)
    with pytest.raises(requests.exceptions.SSLError):
        bundle.build_engine(tmp_path)


def test_the_words_the_page_highlights_are_those_of_the_terminal(
    bundle, tmp_path
):
    import json

    from neurolang_apps.utils import query_highlight

    bundle.write_highlight_words(tmp_path)
    words = json.loads((tmp_path / "highlight.json").read_text())
    assert set(words["datalog_keywords"]) == query_highlight.DATALOG_KEYWORDS
    assert set(words["datalog_operators"]) == query_highlight.DATALOG_OPERATORS
    assert "PROB" in words["datalog_keywords"]
    assert "obtain" in words["squall_keywords"]


def test_a_bundle_without_the_kept_tables_is_refused(
    bundle, monkeypatch, tmp_path
):
    """The page would find them again, a minute and a half at its start."""
    from neurolang_apps.utils import engine_cache

    monkeypatch.setattr(engine_cache, "cached_files", lambda *args: [])
    with pytest.raises(SystemExit, match="kept no tables"):
        bundle.check_engine_cache(tmp_path)


def test_the_kept_tables_are_told(bundle, monkeypatch, tmp_path, capsys):
    from neurolang_apps.utils import engine_cache

    table = tmp_path / "table.npz"
    table.write_bytes(b"x" * 2048)
    monkeypatch.setattr(engine_cache, "cached_files", lambda *args: [table])
    bundle.check_engine_cache(tmp_path)
    assert "1 files" in capsys.readouterr().out


@pytest.mark.parametrize("size, parts", [(250, 3), (100, 1), (40, 1)])
def test_files_larger_than_github_takes_are_split_in_parts(
    bundle, monkeypatch, tmp_path, size, parts
):
    """GitHub refuses files of more than 100 MB, and the documentation is
    deployed by committing it: the SPECTER2 model is written in parts,
    which joined are the file."""
    monkeypatch.setattr(bundle, "PART_SIZE", 100)
    source = tmp_path / "model.npz"
    source.write_bytes(os.urandom(size))
    out = tmp_path / "out"
    out.mkdir()
    names = bundle.write_parts(source, out, "neurosynth-model.npz")
    assert len(names) == parts
    assert all((out / n).stat().st_size <= 100 for n in names)
    assert b"".join((out / n).read_bytes() for n in names) == \
        source.read_bytes()
    assert sorted(p.name for p in out.iterdir()) == sorted(names)


def test_without_torch_nor_saved_embeddings_the_page_goes_without(
    bundle, monkeypatch, tmp_path
):
    from neurolang.utils import engine_registry, text_embeddings

    class Index:
        encoder = type("E", (), {"numpy_path": tmp_path / "m.npz"})()
        cache_path = tmp_path / "e.npz"

    nl = type("NL", (), {"text_embedding_indexes": {"abstracts": Index()}})()
    monkeypatch.setattr(engine_registry, "build_engine", lambda *a: nl)
    monkeypatch.setattr(text_embeddings, "_torch_available", lambda: False)
    assert bundle.export_embeddings(tmp_path, tmp_path) == {}


def test_the_files_of_the_embeddings_have_plain_names(bundle):
    """A ``+`` in a URL is a space for some servers."""
    name = bundle.bundle_name(
        Path("neurosynth/embeddings/models/a--base+a--query.npz")
    )
    assert name == "neurosynth-a--base_a--query.npz"


def _fake_bundle(bundle, monkeypatch, failing=()):
    """Stub the slow steps of the script: the wheels and the engines."""
    monkeypatch.setattr(bundle, "build_wheels", lambda out: ["w.whl"])
    monkeypatch.setattr(bundle, "write_highlight_words", lambda out: None)

    def bundle_engine(engine, output, embedded=None):
        if engine in failing:
            raise RuntimeError("no network")
        return {
            "label": bundle.LABELS[engine], "data": bundle.data_zip(engine),
            "files": {f"{engine}/model.npz": [f"{engine}-model.npz"]},
        }

    monkeypatch.setattr(bundle, "bundle_engine", bundle_engine)


def test_the_bundle_names_the_data_and_files_of_each_engine(
    bundle, monkeypatch, tmp_path
):
    _fake_bundle(bundle, monkeypatch)
    bundle.main(tmp_path)
    described = json.loads((tmp_path / "bundle.json").read_text())
    assert described["default"] == "neurosynth"
    assert list(described["engines"]) == ["neurosynth", "neurostore"]
    assert described["engines"]["neurostore"] == {
        "label": "NeuroStore", "data": "neurolang-neurostore-data.zip",
        "files": {"neurostore/model.npz": ["neurostore-model.npz"]},
    }
    # What a page of before the choice of the engine reads: the default's.
    assert described["data"] == "neurolang-neurosynth-data.zip"
    assert described["files"] == {"neurosynth/model.npz": ["neurosynth-model.npz"]}


def test_an_engine_that_cannot_be_built_is_left_out_unless_it_is_the_default(
    bundle, monkeypatch, tmp_path, capsys
):
    _fake_bundle(bundle, monkeypatch, failing={"neurostore"})
    bundle.main(tmp_path)
    described = json.loads((tmp_path / "bundle.json").read_text())
    assert list(described["engines"]) == ["neurosynth"]
    assert "neurostore is left out" in capsys.readouterr().out
    _fake_bundle(bundle, monkeypatch, failing={"neurosynth"})
    with pytest.raises(RuntimeError):
        bundle.main(tmp_path / "again")


def test_only_the_engines_asked_for_are_built(bundle, monkeypatch, tmp_path):
    _fake_bundle(bundle, monkeypatch)
    bundle.main(tmp_path, engines=("neurostore",))
    described = json.loads((tmp_path / "bundle.json").read_text())
    assert list(described["engines"]) == ["neurostore"]
    assert described["default"] == "neurostore"


def test_the_files_of_the_embeddings_are_named_for_their_engine(bundle):
    name = bundle.bundle_name(Path("a+b.npz"), "neurostore")
    assert name == "neurostore-a_b.npz"


def test_a_neurostore_release_keeps_the_tables_and_columns_the_engine_reads(
    bundle, tmp_path
):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    release = tmp_path / "neurostore" / "2026-10"
    release.mkdir(parents=True)
    pd.DataFrame(
        {"study_id": ["s"], "x": [1.0], "y": [2.0], "z": [3.0],
         "space": ["MNI"], "t_stat": [1.0], "p": [0.1]}
    ).to_parquet(release / "coordinates.parquet")
    pd.DataFrame(
        {"study_id": ["s"], "name": ["n"], "description": ["d"],
         "authors": ["a"]}
    ).to_parquet(release / "studies.parquet")
    pd.DataFrame(
        {"study_id": ["s"], "TaskExtractor.fMRITasks[0].Concepts": ["[]"],
         "TaskExtractor.fMRITasks[0].Domain": ["[]"],
         "TaskExtractor.fMRITasks[0].TaskName": ["n-back"]}
    ).to_parquet(release / "annotations.parquet")
    for name in ("analyses", "images", "metadata", "texts"):
        (release / f"{name}.parquet").write_bytes(b"x")
    (release / "studyset.json").write_text("{}")
    bundle.trim(tmp_path, "neurostore")
    assert sorted(p.name for p in release.iterdir()) == [
        "annotations.parquet", "coordinates.parquet", "studies.parquet"
    ]
    assert list(pd.read_parquet(release / "coordinates.parquet")) == [
        "study_id", "x", "y", "z", "space"
    ]
    assert list(pd.read_parquet(release / "annotations.parquet")) == [
        "study_id", "TaskExtractor.fMRITasks[0].Concepts",
        "TaskExtractor.fMRITasks[0].Domain",
    ]
    assert list(pd.read_parquet(release / "studies.parquet")) == [
        "study_id", "name", "description"
    ]


def test_the_data_of_an_engine_is_written_in_parts_when_it_is_large(
    bundle, monkeypatch, tmp_path
):
    monkeypatch.setattr(bundle, "PART_SIZE", 10)
    monkeypatch.setattr(bundle, "build_engine", lambda *a, **k: None)
    monkeypatch.setattr(bundle, "check_engine_cache", lambda *a: None)
    monkeypatch.setattr(bundle, "trim", lambda *a: None)
    monkeypatch.setattr(bundle, "export_embeddings", lambda *a: {})
    monkeypatch.setattr(
        bundle, "zip_directory",
        lambda directory, archive: archive.write_bytes(b"x" * 25),
    )
    entry = bundle.bundle_engine("neurostore", tmp_path)
    assert entry["data"] == [f"neurolang-neurostore-data.zip.{n}" for n in range(3)]
    assert b"".join((tmp_path / name).read_bytes() for name in entry["data"]) == b"x" * 25
    monkeypatch.setattr(bundle, "PART_SIZE", 100)
    entry = bundle.bundle_engine("neurostore", tmp_path)
    assert entry["data"] == "neurolang-neurostore-data.zip"
