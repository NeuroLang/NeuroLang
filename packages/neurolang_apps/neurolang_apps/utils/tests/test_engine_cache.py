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


def kinds(data_dir):
    """The kinds of the tables kept for the engine ``e``, by file name."""
    return sorted({
        path.name.split("-")[0]
        for path in engine_cache.cached_files(data_dir, "e")
    })


def activated(nl):
    return sorted(nl.symbol_table["activated"].value.unwrap())


def test_the_second_build_reads_what_the_first_kept(engine, finds):
    first = engine_registry.build_engine("e", engine)
    assert len(finds) == 1
    assert kinds(engine) == ["atlas_activation", "fetched"]
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
    assert files == engine_cache.cached_files(engine, "e") and len(files) == 2
    assert engine_cache.cache_dir(engine, "e") == (
        engine / "e" / "engine_cache"
    )
    assert engine_cache.clear(engine, "e") == 2
    assert engine_cache.cached_files(engine, "e") == []
    assert engine_cache.clear(engine, "e") == 0


def test_the_command_lists_what_is_kept(engine, capsys):
    assert engine_cache.main(["e", "--data-dir", str(engine)]) == 0
    built = capsys.readouterr().out
    assert "2 files" in built and "atlas_activation" in built
    assert engine_cache.main(["e", "--data-dir", str(engine), "--list"]) == 0
    assert "2 files" in capsys.readouterr().out
    assert engine_cache.main(
        ["e", "--data-dir", str(engine), "--clear"]
    ) == 0
    assert "Removed 2 files" in capsys.readouterr().out
    assert engine_cache.main(["e", "--data-dir", str(engine), "--list"]) == 0
    assert "0 files" in capsys.readouterr().out


# The primitives the engines keep their tables with.
def test_arrays_kept_are_read_as_the_build_that_kept_them_saw_them(tmp_path):
    computed = []

    def compute():
        computed.append(1)
        return {"i": np.arange(4, dtype=np.int32), "x": np.linspace(0, 1, 3)}

    first = engine_cache.keep_arrays(tmp_path, "t", "k", compute)
    again = engine_cache.keep_arrays(tmp_path, "t", "k", compute)
    assert computed == [1]
    for name in ("i", "x"):
        assert np.array_equal(first[name], again[name])
        assert first[name].dtype == again[name].dtype


def test_an_image_kept_comes_back_with_the_same_dtype_the_first_time(tmp_path):
    image = nib.Nifti1Image(
        np.arange(24, dtype=np.float64).reshape(2, 3, 4),
        np.diag([2.0, 2, 2, 1]),
    )
    first = engine_cache.keep_image(tmp_path, "mask", "k", lambda: image)
    again = engine_cache.keep_image(tmp_path, "mask", "k", lambda: 1 / 0)
    assert np.array_equal(
        np.asanyarray(first.dataobj), np.asanyarray(again.dataobj)
    )
    assert first.get_data_dtype() == again.get_data_dtype()
    assert np.array_equal(again.affine, image.affine)


def test_json_kept_is_read_not_computed_again(tmp_path):
    value = {"maps": "a.nii.gz", "labels": [[1, "A"], [2, "B"]]}
    assert engine_cache.keep_json(tmp_path, "f", "k", lambda: value) == value
    assert engine_cache.keep_json(tmp_path, "f", "k", lambda: 1 / 0) == value


def test_no_directory_means_no_cache():
    calls = []
    for _ in range(2):
        engine_cache.keep_json(None, "f", "k", lambda: calls.append(1))
    assert calls == [1, 1]


def test_a_damaged_file_is_computed_again_and_replaced(tmp_path):
    engine_cache.keep_json(tmp_path, "f", "k", lambda: {"a": 1})
    [path] = tmp_path.glob("*.json")
    path.write_text("{not json")
    again = engine_cache.keep_json(tmp_path, "f", "k", lambda: {"a": 2})
    assert again == {"a": 2}
    assert engine_cache.keep_json(tmp_path, "f", "k", lambda: 1 / 0) == again


def test_a_directory_that_cannot_be_written_only_costs_the_computation(
    tmp_path
):
    blocked = tmp_path / "file"
    blocked.write_text("a file, not a directory")
    calls = []
    for _ in range(2):
        value = engine_cache.keep_json(
            blocked / "cache", "f", "k", lambda: calls.append(1) or {"a": 1}
        )
        assert value == {"a": 1}
    assert calls == [1, 1]


def test_digest_names_the_same_values_the_same_way_on_any_platform():
    peaks = np.array([[1.0, 2.0, 3.0]])
    assert engine_cache.digest(peaks) == engine_cache.digest(peaks + 1e-9)
    assert engine_cache.digest(peaks) != engine_cache.digest(peaks + 0.01)
    assert engine_cache.digest(np.array([1, 2], dtype=np.int32)) == (
        engine_cache.digest(np.array([1, 2], dtype=np.int64))
    )
    assert engine_cache.digest(np.array(["a", "b"])) != (
        engine_cache.digest(np.array(["a", "c"]))
    )
    assert engine_cache.digest(2.0, "a") != engine_cache.digest(2.0, "b")


def test_the_grid_of_an_image_names_it_not_its_voxels():
    affine = np.diag([2.0, 2, 2, 1])
    a = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), affine)
    b = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.int16), affine)
    c = nib.Nifti1Image(np.zeros((2, 3, 5), dtype=np.float32), affine)
    assert engine_cache.grid_digest(a) == engine_cache.grid_digest(b)
    assert engine_cache.grid_digest(a) != engine_cache.grid_digest(c)
    assert engine_cache.image_digest(a) != engine_cache.image_digest(b)


# What the engine does with them.
def test_the_second_build_neither_fetches_the_atlas_nor_imports_nilearn(
    engine, monkeypatch
):
    fetched = []
    fetch = engine_registry._ATLAS_REGISTRY["schaefer"]["fetch"]

    def counting(**kwargs):
        fetched.append(1)
        return fetch(**kwargs)

    monkeypatch.setitem(
        engine_registry._ATLAS_REGISTRY, "schaefer",
        {"fetch": counting, "probabilistic": False},
    )
    first = engine_registry.build_engine("e", engine)
    second = engine_registry.build_engine("e", engine)
    assert fetched == [1]
    assert activated(second) == activated(first)
    assert sorted(second.symbol_table["parcels"].value.unwrap()) == sorted(
        first.symbol_table["parcels"].value.unwrap()
    )


def test_an_atlas_that_moved_is_fetched_again(engine, monkeypatch, tmp_path):
    fetched = []
    fetch = engine_registry._ATLAS_REGISTRY["schaefer"]["fetch"]

    def counting(**kwargs):
        fetched.append(1)
        return fetch(**kwargs)

    monkeypatch.setitem(
        engine_registry._ATLAS_REGISTRY, "schaefer",
        {"fetch": counting, "probabilistic": False},
    )
    engine_registry.build_engine("e", engine)
    [memo] = [
        path for path in engine_cache.cached_files(engine, "e")
        if path.name.startswith("fetched-")
    ]
    memo.write_text(memo.read_text().replace("atlas.nii", "gone.nii"))
    engine_registry.build_engine("e", engine)
    assert fetched == [1, 1]


def test_the_mask_is_kept_and_nilearn_not_asked_for_it_again(
    tmp_path, monkeypatch
):
    template = nib.Nifti1Image(
        np.arange(64, dtype=np.float32).reshape(4, 4, 4), np.eye(4)
    )
    asked = []

    def get_mask(data_dir):
        asked.append(1)
        return template

    monkeypatch.setattr(engine_registry, "_get_mni_mask", get_mask)
    resampled = nib.Nifti1Image(np.ones((2, 2, 2)), np.diag([2.0, 2, 2, 1]))
    monkeypatch.setattr(
        engine_registry, "_nilearn_image", lambda: type(
            "image", (), {"resample_img": staticmethod(lambda *a: resampled)}
        ),
    )
    cfg = {"requires_mni_mask": True}
    cache = tmp_path / "cache"
    first = engine_registry._resolve_mask(cfg, tmp_path, 2.0, cache)
    second = engine_registry._resolve_mask(cfg, tmp_path, 2.0, cache)
    assert asked == [1]
    assert first.shape == second.shape == (2, 2, 2)
    assert np.array_equal(first.affine, second.affine)
    # another resolution is another mask
    engine_registry._resolve_mask(cfg, tmp_path, 3.0, cache)
    assert asked == [1, 1]
    # without a cache, the template is found every time
    engine_registry._resolve_mask(cfg, tmp_path, 2.0, None)
    engine_registry._resolve_mask(cfg, tmp_path, 2.0, None)
    assert asked == [1, 1, 1, 1]
