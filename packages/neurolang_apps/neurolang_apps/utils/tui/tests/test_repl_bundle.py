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


def test_the_wheels_of_an_earlier_build_are_not_taken(
    bundle, monkeypatch, tmp_path
):
    """Their version is that of the commit they were built from."""
    def uv_build(command):
        out = Path(command[command.index("--out-dir") + 1])
        for name in bundle.PACKAGES:
            (out / f"{name}-2-py3-none-any.whl").write_bytes(b"new")

    monkeypatch.setattr(bundle.subprocess, "check_call", uv_build)
    for name in bundle.PACKAGES:
        (tmp_path / f"{name}-1-py3-none-any.whl").write_bytes(b"old")
    assert bundle.build_wheels(tmp_path) == [
        f"{name}-2-py3-none-any.whl" for name in bundle.PACKAGES
    ]
    assert sorted(path.name for path in tmp_path.glob("*.whl")) == sorted(
        f"{name}-2-py3-none-any.whl" for name in bundle.PACKAGES
    )


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


def test_the_downloads_are_kept_even_when_the_build_fails(
    bundle, monkeypatch, tmp_path
):
    """A data host that is down (a 504) does not stop the next build, which
    finds what this one downloaded; the engine's kept tables and its
    embeddings are made by each build, not kept with the downloads."""
    downloads = tmp_path / "downloads"
    (downloads / "fsl").mkdir(parents=True)
    (downloads / "fsl" / "atlas.nii.gz").write_bytes(b"atlas")
    found = []

    def build(data_dir, use_cache=True):
        found.append(sorted(
            str(p.relative_to(data_dir)) for p in data_dir.rglob("*")
            if p.is_file()
        ))
        (data_dir / "peaks.tsv.gz").write_bytes(b"peaks")
        for kept in ("engine_cache", "embeddings"):
            (data_dir / "neurosynth" / kept).mkdir(parents=True)
            (data_dir / "neurosynth" / kept / "table.npz").write_bytes(b"t")
        raise requests.HTTPError("504 Gateway Time-out")

    monkeypatch.setattr(bundle, "build_wheels", lambda output: [])
    monkeypatch.setattr(bundle, "build_engine", build)
    with pytest.raises(requests.HTTPError):
        bundle.main(tmp_path / "out", downloads=downloads)
    assert found == [["fsl/atlas.nii.gz"]]
    assert sorted(
        str(p.relative_to(downloads)) for p in downloads.rglob("*")
        if p.is_file()
    ) == ["fsl/atlas.nii.gz", "peaks.tsv.gz"]


def test_the_downloads_directory_is_an_option(bundle, tmp_path):
    args = bundle.parse_args([str(tmp_path), "emb", "--downloads", "dl"])
    assert (args.output, args.embedded, args.downloads) == (
        tmp_path, "emb", Path("dl")
    )
    assert bundle.parse_args([str(tmp_path)]).downloads is None


def test_the_data_can_be_linked_to_another_site(bundle, tmp_path):
    """
    A pull request's preview loads the data and the embeddings of the main
    site: its bundle names them under the main site's directory, and they
    are not deployed with it. The wheels stay, as the page's NeuroLang.
    """
    files = {"neurosynth/embeddings/model.npz": ["model.npz.0", "model.npz.1"]}
    (tmp_path / "bundle.json").write_text(json.dumps(
        {"wheels": ["nl-1.whl"], "data": bundle.DATA_ZIP, "files": files}
    ))
    for name in ("nl-1.whl", bundle.DATA_ZIP, "model.npz.0", "model.npz.1"):
        (tmp_path / name).write_bytes(b"x")
    args = bundle.parse_args([str(tmp_path), "--data-url", "/_static/repl/"])
    assert args.data_url == "/_static/repl/"
    assert bundle.parse_args([str(tmp_path)]).data_url is None

    for _ in range(2):  # linking again changes nothing
        bundle.link_data(tmp_path, args.data_url)
        assert json.loads((tmp_path / "bundle.json").read_text()) == {
            "wheels": ["nl-1.whl"],
            "data": f"/_static/repl/{bundle.DATA_ZIP}",
            "files": {"neurosynth/embeddings/model.npz": [
                "/_static/repl/model.npz.0", "/_static/repl/model.npz.1",
            ]},
        }
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "bundle.json", "nl-1.whl"
    ]
