"""Tests of the tables an engine keeps to start without finding them again."""

import nibabel as nib
import numpy as np
import pytest

from neurolang_apps.utils import engine_cache, engine_registry

PEAKS = [
    (6.0, 12.0, 3.0, 7),   # the centre of the voxel (2, 4, 1)
    (7.0, 12.0, 3.0, 8),   # 1 mm from it
    (90.0, 90.0, 90.0, 9),  # far from every region
]


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """An engine ``e`` with an atlas of two regions and the peaks above,
    that finds the studies activating its regions."""
    data = np.zeros((3, 5, 2), dtype=np.int16)
    data[2, 4, 1], data[0, 0, 0] = 1, 2
    nib.save(
        nib.Nifti1Image(data, np.diag([3.0, 3.0, 3.0, 1.0])),
        tmp_path / "atlas.nii",
    )
    monkeypatch.setitem(
        engine_registry._ATLAS_REGISTRY, "schaefer", {
            "fetch": lambda **kw: {
                "maps": str(tmp_path / "atlas.nii"),
                "labels": ["Background", "A", "B"],
            },
            "probabilistic": False,
        },
    )
    path = tmp_path / "engines.yaml"
    path.write_text(
        "engines:\n  e:\n    atlases:\n      schaefer:\n"
        "        predicate_name: parcels\n"
        "        activated: {predicate_name: activated, radius_mm: 3}\n"
    )
    monkeypatch.setattr(engine_registry, "_ENGINES_YAML", path)
    python_init = engine_registry._phase_python_init

    def with_peaks(nl, cfg, mask, data_dir):
        python_init(nl, cfg, mask, data_dir)
        nl.add_tuple_set(PEAKS, name="peak_mni")

    monkeypatch.setattr(engine_registry, "_phase_python_init", with_peaks)
    return tmp_path / "data"


@pytest.fixture
def finds(monkeypatch):
    """The searches for the studies activating the regions."""
    calls = []
    find = engine_registry.atlas_activation

    def spy(*args, **kwargs):
        calls.append(args)
        return find(*args, **kwargs)

    monkeypatch.setattr(engine_registry, "atlas_activation", spy)
    return calls


def activated(nl):
    return sorted(nl.symbol_table["activated"].value.unwrap())


def test_the_second_build_reads_what_the_first_kept(engine, finds):
    first = engine_registry.build_engine("e", engine)
    assert len(finds) == 1
    assert len(engine_cache.cached_files(engine, "e")) == 1
    second = engine_registry.build_engine("e", engine)
    assert len(finds) == 1
    assert activated(second) == activated(first) == [("A", 7), ("A", 8)]


def test_an_engine_built_without_the_cache_neither_reads_nor_writes_it(
    engine, finds
):
    nl = engine_registry.build_engine("e", engine, use_cache=False)
    nl = engine_registry.build_engine("e", engine, use_cache=False)
    assert len(finds) == 2
    assert engine_cache.cached_files(engine, "e") == []
    assert activated(nl) == [("A", 7), ("A", 8)]


def test_prebuild_keeps_the_tables_and_clear_removes_them(engine, finds):
    files = engine_cache.prebuild("e", engine)
    assert files == engine_cache.cached_files(engine, "e") and len(files) == 1
    assert engine_cache.cache_dir(engine, "e") == (
        engine / "e" / "atlas_activation"
    )
    assert engine_cache.clear(engine, "e") == 1
    assert engine_cache.cached_files(engine, "e") == []
    assert engine_cache.clear(engine, "e") == 0


def test_the_command_lists_what_is_kept(engine, capsys):
    assert engine_cache.main(["e", "--data-dir", str(engine)]) == 0
    built = capsys.readouterr().out
    assert "1 files" in built and "atlas_activation" in built
    assert engine_cache.main(["e", "--data-dir", str(engine), "--list"]) == 0
    assert "1 files" in capsys.readouterr().out
    assert engine_cache.main(
        ["e", "--data-dir", str(engine), "--clear"]
    ) == 0
    assert "Removed 1 files" in capsys.readouterr().out
    assert engine_cache.main(["e", "--data-dir", str(engine), "--list"]) == 0
    assert "0 files" in capsys.readouterr().out
