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
    embeddings are made by each build, not kept with the downloads. Each
    engine's downloads are kept apart from the others'."""
    downloads = tmp_path / "downloads"
    (downloads / "neurosynth" / "fsl").mkdir(parents=True)
    (downloads / "neurosynth" / "fsl" / "atlas.nii.gz").write_bytes(b"atlas")
    (downloads / "neurostore").mkdir()
    (downloads / "neurostore" / "release.parquet").write_bytes(b"other")
    found = []

    def build(data_dir, use_cache=True, engine="neurosynth"):
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
    # neurosynth is left out, and the default, neurosynth_iit, stops the
    # build: each found the downloads kept for it
    assert found == [["fsl/atlas.nii.gz"], []]
    kept = downloads / "neurosynth"
    assert sorted(
        str(p.relative_to(kept)) for p in kept.rglob("*") if p.is_file()
    ) == ["fsl/atlas.nii.gz", "peaks.tsv.gz"]


def test_the_downloads_directory_is_an_option(bundle, tmp_path):
    args = bundle.parse_args([str(tmp_path), "emb", "--downloads", "dl"])
    assert (args.output, args.embedded, args.downloads) == (
        tmp_path, "emb", Path("dl")
    )
    assert bundle.parse_args([str(tmp_path)]).downloads is None


def _fake_bundle(bundle, monkeypatch, failing=()):
    """Stub the slow steps of the script: the wheels and the engines."""
    monkeypatch.setattr(bundle, "build_wheels", lambda out: ["w.whl"])
    monkeypatch.setattr(bundle, "write_highlight_words", lambda out: None)

    def bundle_engine(engine, output, embedded=None, downloads=None,
                      shared=False):
        if engine in failing:
            raise RuntimeError("no network")
        return {
            "label": bundle.LABELS[engine], "data": bundle.data_zip(engine),
            "files": {} if shared else {
                f"{engine}/model.npz": [f"{engine}-model.npz"]},
            "cache_keys": {},
        }

    monkeypatch.setattr(bundle, "bundle_engine", bundle_engine)


def test_the_bundle_names_the_data_and_files_of_each_engine(
    bundle, monkeypatch, tmp_path
):
    _fake_bundle(bundle, monkeypatch)
    bundle.main(tmp_path)
    described = json.loads((tmp_path / "bundle.json").read_text())
    # the page starts with Neurosynth and the tracts, built after
    # Neurosynth, whose text embeddings it shares, as NeuroStore and the
    # tracts share those of NeuroStore
    assert described["default"] == "neurosynth_iit"
    assert list(described["engines"]) == [
        "neurosynth", "neurosynth_iit", "neurostore", "neurostore_iit"]
    assert described["engines"]["neurosynth_iit"]["files"] == {
        "neurosynth/model.npz": ["neurosynth-model.npz"]}
    assert described["engines"]["neurostore_iit"] == {
        "label": "NeuroStore + tracts",
        "data": "neurolang-neurostore_iit-data.zip",
        "files": {"neurostore/model.npz": ["neurostore-model.npz"]},
    }
    assert described["engines"]["neurostore"] == {
        "label": "NeuroStore", "data": "neurolang-neurostore-data.zip",
        "files": {"neurostore/model.npz": ["neurostore-model.npz"]},
    }
    # What a page of before the choice of the engine reads: the default's.
    assert described["data"] == "neurolang-neurosynth_iit-data.zip"
    assert described["files"] == {
        "neurosynth/model.npz": ["neurosynth-model.npz"]}


def test_an_engine_of_shared_embeddings_is_built_alone(
    bundle, monkeypatch, tmp_path
):
    """Without the engine whose studies it has in the bundle, the engine has
    its own embeddings files (taken from those kept for the other)."""
    _fake_bundle(bundle, monkeypatch)
    bundle.main(tmp_path, engines=("neurosynth_iit",))
    described = json.loads((tmp_path / "bundle.json").read_text())
    assert described["default"] == "neurosynth_iit"
    assert described["engines"]["neurosynth_iit"]["files"] == {
        "neurosynth_iit/model.npz": ["neurosynth_iit-model.npz"]}


def test_the_embeddings_of_an_engine_are_kept_as_those_of_another(
    bundle, tmp_path, monkeypatch
):
    """``kept_as`` reads and saves the embeddings under the other engine's
    directory of the kept data."""
    seen = []
    monkeypatch.setattr(bundle.shutil, "copytree",
                        lambda src, dst, **kw: seen.append(Path(src)))
    kept = tmp_path / "kept" / "neurosynth" / "embeddings"
    kept.mkdir(parents=True)
    (tmp_path / "data").mkdir()
    bundle.export_embeddings(
        tmp_path / "data", tmp_path, tmp_path / "kept", "neurosynth_iit",
        "neurosynth")
    assert kept in seen


def test_an_engine_that_cannot_be_built_is_left_out_unless_it_is_the_default(
    bundle, monkeypatch, tmp_path, capsys
):
    _fake_bundle(bundle, monkeypatch,
                 failing={"neurostore", "neurostore_iit"})
    bundle.main(tmp_path)
    described = json.loads((tmp_path / "bundle.json").read_text())
    assert list(described["engines"]) == ["neurosynth", "neurosynth_iit"]
    assert "neurostore is left out" in capsys.readouterr().out
    _fake_bundle(bundle, monkeypatch, failing={"neurosynth_iit"})
    with pytest.raises(RuntimeError):
        bundle.main(tmp_path / "again")
    # without the tracts, the page starts with the first engine
    _fake_bundle(bundle, monkeypatch)
    bundle.main(tmp_path / "plain", engines=("neurosynth", "neurostore"))
    described = json.loads((tmp_path / "plain" / "bundle.json").read_text())
    assert described["default"] == "neurosynth"


def test_only_the_engines_asked_for_are_built(bundle, monkeypatch, tmp_path):
    _fake_bundle(bundle, monkeypatch)
    bundle.main(tmp_path, engines=("neurostore",))
    described = json.loads((tmp_path / "bundle.json").read_text())
    assert list(described["engines"]) == ["neurostore"]
    assert described["default"] == "neurostore"


def test_the_files_of_the_embeddings_are_named_for_their_engine(bundle):
    name = bundle.bundle_name(Path("a+b.npz"), "neurostore")
    assert name == "neurostore-a_b.npz"


@pytest.mark.parametrize("engine", ["neurostore", "neurostore_iit"])
def test_a_neurostore_release_keeps_the_tables_and_columns_the_engine_reads(
    bundle, tmp_path, engine
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
    bundle.trim(tmp_path, engine)
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


def test_the_tractogram_of_the_page_is_a_seeded_thinned_draw(bundle, tmp_path):
    """15,000 of the IIT atlas's in the page; here 5 of 8 streamlines (one
    of a point, never drawn) of a ``.tck``, in the order of the file, every
    second point and the last: the same draw for the same seed, in the same
    format."""
    import nibabel as nib
    import numpy as np

    lines = [
        np.cumsum(np.ones((n, 3)), axis=0).astype(np.float32) + k
        for k, n in enumerate([5, 1, 6, 2, 9, 3, 4, 7])
    ]
    source = tmp_path / "whole_brain.tck"
    nib.streamlines.save(
        nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4)),
        str(source),
    )
    drawn = []
    for name in ("a.tck", "b.tck"):
        bundle.subsample_tractogram(source, tmp_path / name, 5, 2, seed=1)
        drawn.append(nib.streamlines.load(str(tmp_path / name)).streamlines)
    assert [list(map(len, d)) for d in drawn] == 2 * [list(map(len, drawn[0]))]
    assert len(drawn[0]) == 5
    for one, other in zip(*drawn):
        np.testing.assert_array_equal(one, other)
    # each is a thinned streamline of the source, ends included
    for line in drawn[0]:
        origin = next(
            full for full in lines
            if len(full) > 1 and (full[0] == line[0]).all()
        )
        assert (line[-1] == origin[-1]).all()
        assert len(line) == len(range(0, len(origin), 2)) + (
            0 if (len(origin) - 1) % 2 == 0 else 1)


def test_the_tractogram_entry_of_the_engine_tells_how_to_subsample(
    bundle, monkeypatch, tmp_path
):
    """engines.yaml: ``tractogram: {path, subsample, step, seed}``. The file
    of the data directory at ``path`` is replaced by its subsample."""
    entry = bundle.tractogram_entry("neurosynth_iit")
    assert dict(entry) == {
        "path": "tracts/iit/IIT3mean.tck", "subsample": 15000, "step": 3,
        "seed": 0}
    assert bundle.tractogram_entry("neurosynth") is None
    drawn = {}

    def subsample(source, target, count, step, seed):
        drawn.update(source=source, target=target, count=count, step=step,
                     seed=seed)
        target.write_bytes(b"x")

    monkeypatch.setattr(bundle, "subsample_tractogram", subsample)
    data = tmp_path / "data"
    (data / "tracts" / "iit").mkdir(parents=True)
    path = data / "tracts" / "iit" / "IIT3mean.tck"
    path.write_bytes(b"full")
    bundle.reduce_tractogram(
        data, {"path": "tracts/iit/IIT3mean.tck", "subsample": 7, "step": 3,
               "seed": 5})
    assert drawn == {
        "source": path, "target": path, "count": 7, "step": 3, "seed": 5}
    assert path.read_bytes() == b"x"


def test_an_engine_with_the_studies_of_another_shares_its_embeddings(bundle):
    donor = {
        "cache_keys": {"study_abstracts": "k"},
        "files": {
            "neurosynth/embeddings/study_abstracts.npz": ["a.0", "a.1"],
            "neurosynth/embeddings/models/m.npz": ["m"],
        },
    }
    entry = {"cache_keys": {"study_abstracts": "k"}, "files": {}}
    bundle.share_embeddings(entry, donor)
    assert entry["files"] == donor["files"]
    entry = {"cache_keys": {"study_abstracts": "other"}, "files": {}}
    with pytest.raises(ValueError, match="cache_key"):
        bundle.share_embeddings(entry, donor)


def test_the_embeddings_of_neurosynth_are_in_both_engines_of_the_bundle(
    bundle, monkeypatch, tmp_path
):
    _fake_bundle(bundle, monkeypatch)
    original = bundle.bundle_engine

    def with_keys(engine, *args):
        entry = original(engine, *args)
        return {**entry, "cache_keys": {"study_abstracts": "k"}}

    monkeypatch.setattr(bundle, "bundle_engine", with_keys)
    bundle.main(tmp_path, engines=("neurosynth", "neurosynth_iit"))
    described = json.loads((tmp_path / "bundle.json").read_text())
    assert described["default"] == "neurosynth_iit"
    assert described["engines"]["neurosynth_iit"] == {
        "label": "Neurosynth + tracts",
        "data": "neurolang-neurosynth_iit-data.zip",
        "files": {"neurosynth/model.npz": ["neurosynth-model.npz"]},
    }
    assert all("cache_keys" not in e for e in described["engines"].values())
