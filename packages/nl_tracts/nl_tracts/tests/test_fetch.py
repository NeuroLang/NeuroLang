"""The atlases of Yeh, downloaded (``nl_tracts.fetch``) and loaded by
``.load_tractogram(..., kind="hcp842")``: offline, from small archives
built here and served as ``file://`` URLs."""

import gzip
import hashlib
import urllib.request
import zipfile

import nibabel as nib
import numpy as np
import pytest

from nl_expr.exceptions import InvalidCommandExpression
from nl_tracts.fetch import ATLASES, fetch_tractogram
from nl_tracts.frontend import init_engine
from nl_tracts.tests.engine import tracts_engine
from nl_tracts.tractogram import Tractogram

from .test_commands import atlas_command, files, lines, rows  # noqa: F401


def trk_bytes(streamlines, tmp_path):
    path = tmp_path / "tmp.trk"
    nib.streamlines.save(
        nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4)),
        str(path),
    )
    data = path.read_bytes()
    path.unlink()
    return data


@pytest.fixture
def whole_brain():
    # the third streamline has one point, as 37 of HCP842's
    return lines(0, 6)[:2] + [np.ones((1, 3), np.float32)] + lines(1, 3)


@pytest.fixture
def archives(tmp_path, monkeypatch, whole_brain):
    """Small archives of the layout of HCP842 and HCP1065, which
    ``ATLASES`` downloads instead: ``{atlas: (path, its md5)}``."""
    source = tmp_path / "source"
    source.mkdir()
    built = {}
    for atlas, members in (
        ("hcp842", {
            "Atlas_80_Bundles/bundles/UF_L.trk": lines(2, 4),
            "Atlas_80_Bundles/bundles/AF_R.trk": lines(3, 5),
            "Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk": whole_brain,
        }),
        ("hcp1065", {"association/UF_L.trk.gz": lines(4, 7)}),
    ):
        path = source / ATLASES[atlas]["archive"]
        with zipfile.ZipFile(path, "w") as z:
            for name, streamlines in members.items():
                data = trk_bytes(streamlines, tmp_path)
                if name.endswith(".gz"):
                    data = gzip.compress(data)
                z.writestr(name, data)
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        monkeypatch.setitem(ATLASES, atlas, {
            **ATLASES[atlas], "urls": (path.as_uri(),), "md5": md5})
        built[atlas] = path, md5
    return built


def test_a_bundle_and_the_whole_brain_are_downloaded_once(
        archives, tmp_path, whole_brain):
    data = tmp_path / "data"
    uf = fetch_tractogram("hcp842", "uf_l", data_dir=data)
    assert uf == data / "hcp842" / "UF_L.trk"
    assert Tractogram.from_file("u", uf, "MNI152").n_streamlines == 4
    brain = fetch_tractogram("hcp842", data_dir=data)
    assert brain == data / "hcp842" / "whole_brain_MNI.trk"
    with pytest.warns(UserWarning, match="1 streamlines of one point"):
        tg = Tractogram.from_file("w", brain, "MNI152", drop_short=True)
    assert tg.n_streamlines == len(whole_brain)
    # only what was asked is extracted, next to the archive
    assert sorted(p.name for p in (data / "hcp842").iterdir()) == [
        "Atlas_80_Bundles.zip", "UF_L.trk", "whole_brain_MNI.trk"]
    # without the source: the files and the archive kept are reused
    archives["hcp842"][0].unlink()
    assert fetch_tractogram("hcp842", "UF_L", data_dir=data) == uf
    af = fetch_tractogram("hcp842", "AF_R", data_dir=data)
    assert Tractogram.from_file("a", af, "MNI152").n_streamlines == 5


def test_a_trk_gz_is_gunzipped(archives, tmp_path):
    path = fetch_tractogram("hcp1065", "UF_L", data_dir=tmp_path)
    assert path == tmp_path / "hcp1065" / "UF_L.trk"
    assert Tractogram.from_file("u", path, "MNI152").n_streamlines == 7


def test_the_checksum_is_checked_and_the_next_url_tried(
        archives, tmp_path, monkeypatch):
    path = archives["hcp842"][0]
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not the archive")
    monkeypatch.setitem(ATLASES, "hcp842", {
        **ATLASES["hcp842"], "urls": (bad.as_uri(),)})
    with pytest.raises(OSError, match="md5"):
        fetch_tractogram("hcp842", "UF_L", data_dir=tmp_path / "d")
    assert not any((tmp_path / "d" / "hcp842").iterdir())
    monkeypatch.setitem(ATLASES, "hcp842", {
        **ATLASES["hcp842"],
        "urls": ((tmp_path / "missing.zip").as_uri(), bad.as_uri(),
                 path.as_uri())})
    assert fetch_tractogram("hcp842", "UF_L", data_dir=tmp_path / "d")


def test_a_url_and_its_checksum_override_the_atlas(archives, tmp_path):
    path = archives["hcp842"][0]
    with pytest.raises(OSError, match="md5"):
        fetch_tractogram("hcp842", "UF_L", data_dir=tmp_path / "d",
                         url=path.as_uri(), md5="0" * 32)
    assert fetch_tractogram("hcp842", "UF_L", data_dir=tmp_path / "d",
                            url=path.as_uri()).is_file()


@pytest.mark.parametrize("atlas,bundle,match", [
    ("hcp9", None, "the atlases are hcp842, hcp1065"),
    ("hcp842", "UF", "no bundle 'UF'; its bundles are AC, AF_L"),
    ("hcp1065", None, "no whole-brain tractogram"),
])
def test_unknown_names_are_errors(atlas, bundle, match):
    with pytest.raises(ValueError, match=match):
        fetch_tractogram(atlas, bundle, data_dir="/nonexistent")


def test_the_lists_of_bundles():
    assert len(ATLASES["hcp842"]["bundles"]) == 78
    assert len(ATLASES["hcp1065"]["bundles"]) == 87
    assert ATLASES["hcp842"]["bundles"]["UF_L"] == (
        "uncinate fasciculus, left")
    assert ATLASES["hcp1065"]["bundles"]["SLF2_R"] == (
        "superior longitudinal fasciculus II, right")


def test_the_command_loads_the_whole_brain_and_a_bundle(
        archives, files, tmp_path):  # noqa: F811
    data = tmp_path / "data"
    nl = tracts_engine()
    nl.execute_datalog_program(
        atlas_command(files)
        + f'.load_tractogram("w", kind="hcp842", data_dir="{data}", '
        'space="s")\n'
        f'.load_tractogram("u", kind="hcp842", bundle="UF_L", '
        f'data_dir="{data}", space="s")\n'
        '.index_tracts("w", "fs")\n.index_tracts("u", "fs")\n'
    )
    got = rows(nl.execute_datalog_program(
        'ans(t, s) :- streamline(x, t, s)\n'))
    # the streamline of one point, s = 2, is in no relation
    assert got == {("w", s) for s in (0, 1, 3, 4, 5)} | {
        ("u", s) for s in range(4)}


def test_the_engine_data_dir_is_the_default(archives, tmp_path):
    nl = tracts_engine()
    init_engine(nl, data_dir=tmp_path)
    nl.execute_datalog_program(
        '.load_tractogram("u", kind="hcp842", bundle="UF_L")\n')
    assert (tmp_path / "tracts" / "hcp842" / "UF_L.trk").is_file()


@pytest.mark.parametrize("command,match", [
    ('.load_tractogram("w", kind="hcp9")\n', "one of hcp842, hcp1065"),
    ('.load_tractogram("w", "a.trk", kind="hcp842")\n', "takes 1"),
    ('.load_tractogram("w", kind="hcp842", bundle="XX", data_dir="/n")\n',
     "no bundle 'XX'"),
    ('.load_tractogram("w", kind="hcp842", fetcher=1)\n', "no option"),
])
def test_the_command_fails_with_a_message(command, match):
    with pytest.raises(InvalidCommandExpression, match=match):
        tracts_engine().execute_datalog_program(command)


def test_a_file_with_drop_short(files, tmp_path, whole_brain):  # noqa: F811
    (tmp_path / "w.trk").write_bytes(trk_bytes(whole_brain, tmp_path))
    program = f'.load_tractogram("w", "{tmp_path}/w.trk", space="s"'
    with pytest.raises(InvalidCommandExpression, match="drop_short"):
        tracts_engine().execute_datalog_program(program + ")\n")
    nl = tracts_engine()
    nl.execute_datalog_program(
        atlas_command(files) + program + ", drop_short=1)\n"
        '.index_tracts("w", "fs")\n')
    assert rows(nl.execute_datalog_program(
        'ans(s) :- streamline(x, "w", s)\n')) == {
            (s,) for s in (0, 1, 3, 4, 5)}


# --- the real atlases ------------------------------------------------------

@pytest.mark.network
@pytest.mark.parametrize("atlas", sorted(ATLASES))
def test_the_urls_serve_the_archives(atlas):
    """The first kilobyte of each URL, of an archive of the size known."""
    sizes = {"hcp842": 319_525_089, "hcp1065": 587_869_457}
    for url in ATLASES[atlas]["urls"]:
        request = urllib.request.Request(url, headers={"Range": "bytes=0-1"})
        try:
            with urllib.request.urlopen(request, timeout=30) as r:
                assert r.read(2) == b"PK"
                total = r.headers["Content-Range"].rsplit("/", 1)[-1]
        except OSError as e:
            pytest.skip(f"offline or {url} unreachable: {e!r}")
        assert int(total) == sizes[atlas]


@pytest.mark.slow
@pytest.mark.network
def test_the_real_hcp842(tmp_path, fetch_or_skip):
    """Downloads the 320 MB of HCP842 (``--runslow``)."""
    with pytest.warns(UserWarning, match="37 streamlines of one point"):
        brain = Tractogram.from_file(
            "w", fetch_or_skip(fetch_tractogram, "hcp842", data_dir=tmp_path),
            "MNI152", drop_short=True)
    assert brain.n_streamlines == 144_678
    uf = Tractogram.from_file(
        "u", fetch_tractogram("hcp842", "UF_L", data_dir=tmp_path), "MNI152")
    assert uf.n_streamlines == 370
