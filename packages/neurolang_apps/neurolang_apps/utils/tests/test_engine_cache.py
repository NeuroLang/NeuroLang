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


def test_an_atlas_on_the_template_is_kept_for_the_grid_of_the_template(
    tmp_path, monkeypatch
):
    """GH-1075: the atlas resampled to the template is kept, by the grid
    of the template: another grid resamples it again."""
    from types import SimpleNamespace

    from neurolang.atlases import Atlas

    data = np.zeros((3, 5, 2), dtype=np.int16)
    data[2, 4, 1], data[0, 0, 0] = 1, 2
    atlas = Atlas.from_labels(
        "a", nib.Nifti1Image(data, np.diag([3.0, 3, 3, 1])),
        ["Background", "A", "B"],
    )
    resampled = []
    nilearn_image = engine_registry._nilearn_image

    def counting():
        resampled.append(1)
        return nilearn_image()

    monkeypatch.setattr(engine_registry, "_nilearn_image", counting)
    cache = tmp_path / "cache"

    def on(shape, voxel_size):
        template = nib.Nifti1Image(
            np.ones(shape, np.float32), np.diag([voxel_size] * 3 + [1.0])
        )
        return engine_registry._atlas_on_template(
            SimpleNamespace(template=template), atlas, cache
        )

    first, again = on((9, 15, 6), 1.0), on((9, 15, 6), 1.0)
    assert resampled == [1]
    assert first.shape == again.shape == (9, 15, 6)
    assert np.array_equal(first.labels, again.labels)
    assert first.region_names() == {1: "A", 2: "B"}
    assert on((3, 5, 2), 3.0).shape == (3, 5, 2)
    assert resampled == [1, 1]


def test_the_build_tells_what_it_is_about_to_load(engine):
    told = []
    engine_registry.build_engine("e", engine, progress=told.append)
    assert told == ["Loading the atlases: schaefer"]


def test_an_engine_without_such_steps_tells_nothing(tmp_path, monkeypatch):
    path = tmp_path / "engines.yaml"
    path.write_text("engines:\n  bare:\n    description: nothing to load\n")
    monkeypatch.setattr(engine_registry, "_ENGINES_YAML", path)
    told = []
    engine_registry.build_engine("bare", tmp_path, progress=told.append)
    assert told == []


def test_the_registry_imports_the_atlases_when_it_loads_them():
    """neurolang.atlases imports pandas, which listing the engines does
    not need."""
    import subprocess
    import sys

    code = (
        "import sys, neurolang_apps.utils.engine_registry;"
        "print('pandas' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


# -- The tables an engine derives from its source tables (activated) ------

VOXELS = [(i, j, k) for i in range(4) for j in range(4) for k in range(4)]
SOURCE_PEAKS = [(1, 1, 1, "s1"), (3, 2, 1, "s2"), (0, 0, 3, "s3")]
CLOSE = (
    "close(i, j, k, s) :- voxel(i, j, k) & peak_reported(i1, j1, k1, s) "
    "& (EUCLIDEAN(i, j, k, i1, j1, k1) <= {radius})"
)


@pytest.fixture
def deriving(tmp_path, monkeypatch):
    """An engine ``d`` whose rule derives the voxels within 1.5 of each
    peak, and the calls of the computation of such pairs."""
    path = tmp_path / "engines.yaml"
    path.write_text(
        "engines:\n  d:\n    datalog_init: |\n      "
        + CLOSE.format(radius=1.5) + "\n"
    )
    monkeypatch.setattr(engine_registry, "_ENGINES_YAML", path)
    peaks = {"rows": list(SOURCE_PEAKS)}

    def sources(nl, cfg, mask, data_dir):
        nl.add_tuple_set(VOXELS, name="voxel")
        nl.add_tuple_set(peaks["rows"], name="peak_reported")

    monkeypatch.setattr(engine_registry, "_phase_python_init", sources)
    from neurolang.frontend.datalog.sugar import spatial

    calls = []
    join = spatial.euclidean_join

    def spy(*args, **kwargs):
        calls.append(args)
        return join(*args, **kwargs)

    monkeypatch.setattr(spatial, "euclidean_join", spy)
    return tmp_path / "data", calls, peaks


def derived_files(data_dir):
    return sorted(
        (data_dir / engine_cache.DERIVED_DIRECTORY).glob("*.arrow")
    )


def close(nl):
    return sorted(nl.execute_datalog_program(
        "ans(i, j, k, s) :- close(i, j, k, s)"
    ).to_unnamed())


def test_a_derived_table_is_kept_and_read_mapped_by_the_next_build(
        deriving):
    data_dir, calls, _ = deriving
    first = engine_registry.build_engine("d", data_dir)
    assert len(calls) == 1 and len(derived_files(data_dir)) == 1
    second = engine_registry.build_engine("d", data_dir)
    assert len(calls) == 1  # read, not computed again
    assert close(second) == close(first) and close(first)
    # nothing is kept from what a session's own rules derive
    assert not hasattr(second.program_ir, "derived_tables")
    second.execute_datalog_program(
        CLOSE.format(radius=2.5).replace("close(", "near(", 1)
    )
    assert len(calls) == 2 and len(derived_files(data_dir)) == 1


def test_a_change_of_a_source_table_derives_the_table_again(deriving):
    data_dir, calls, peaks = deriving
    engine_registry.build_engine("d", data_dir)
    peaks["rows"] = SOURCE_PEAKS[:2]
    changed = engine_registry.build_engine("d", data_dir)
    assert len(calls) == 2 and len(derived_files(data_dir)) == 2
    assert {row[3] for row in close(changed)} == {"s1", "s2"}


def test_an_engine_built_without_the_cache_keeps_no_derived_table(deriving):
    data_dir, calls, _ = deriving
    engine_registry.build_engine("d", data_dir, use_cache=False)
    engine_registry.build_engine("d", data_dir, use_cache=False)
    assert len(calls) == 2 and not derived_files(data_dir)


def test_a_derived_table_comes_back_as_it_was_computed(tmp_path):
    import pandas as pd

    frame = pd.DataFrame({
        "i": np.arange(5, dtype="int64"),
        "s": pd.array(["a", "b", "c", "a", "é"], dtype="str"),
        "d": np.linspace(0, 1, 5),
    })
    derived = engine_cache.DerivedTables(tmp_path)
    kept = derived("pairs", [frame], (1.5,), lambda: frame)
    again = derived("pairs", [frame], (1.5,), lambda: pytest.fail("kept"))
    for table in (kept, again):
        assert list(table.columns) == ["i", "s", "d"]
        assert list(table.dtypes) == list(frame.dtypes)
        assert table.equals(frame)
    # read from the file, mapped: its numbers are not a copy
    assert not np.asarray(again["i"].array).flags.writeable
    # other parameters, or other source rows, name another table
    derived("pairs", [frame], (2.5,), lambda: frame)
    derived("pairs", [frame.iloc[:4]], (1.5,), lambda: frame)
    assert len(list(tmp_path.glob("pairs-*.arrow"))) == 3


def test_a_derived_table_that_cannot_be_kept_is_computed(tmp_path):
    import pandas as pd

    frame = pd.DataFrame({"i": [1, 2]})
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    derived = engine_cache.DerivedTables(blocked / "derived")
    assert derived("pairs", [frame], (), lambda: frame).equals(frame)
    damaged = engine_cache.DerivedTables(tmp_path / "d")
    derived = damaged("pairs", [frame], (), lambda: frame)
    for path in (tmp_path / "d").glob("*.arrow"):
        path.write_bytes(b"damaged")
    assert damaged("pairs", [frame], (), lambda: frame).equals(frame)


# -- The source tables of a build, kept as the derived ones ---------------

def test_the_tables_of_an_atlas_are_read_mapped_by_the_next_build(
    engine, finds, monkeypatch
):
    first = engine_registry.build_engine("e", engine)
    assert {path.name.split("-")[0] for path in derived_files(engine)} == {
        "atlas_activation"
    }
    # read from the Arrow file: the pairs kept for the browser are not
    # read, nor their checksum computed
    monkeypatch.setattr(
        engine_registry, "cached_atlas_activation",
        lambda *args: pytest.fail("found again"),
    )
    second = engine_registry.build_engine("e", engine)
    assert len(finds) == 1
    assert activated(second) == activated(first) == [("A", 7), ("A", 8)]
    container = second.symbol_table["activated"].value.unwrap()._container
    assert not np.asarray(container.iloc[:, 1].array).flags.writeable


def test_prebuild_writes_the_tables_of_the_browser_the_arrow_files_hold(
    engine, finds
):
    """The tables of the browser's bundle are all written, even those of
    which an Arrow file is kept, which the browser does not read."""
    engine_registry.build_engine("e", engine)
    engine_cache.clear(engine, "e")
    assert len(engine_cache.prebuild("e", engine)) == 2
    assert len(finds) == 2


def test_a_relation_file_is_read_mapped_until_it_changes(
    tmp_path, monkeypatch
):
    rows = tmp_path / "rows.csv"
    rows.write_text("a,b\n1,x\n2,y\n")
    path = tmp_path / "engines.yaml"
    path.write_text(
        f"engines:\n  r:\n    relations:\n      rows: {rows}\n"
    )
    monkeypatch.setattr(engine_registry, "_ENGINES_YAML", path)
    reads = []
    read = engine_registry.read_relation_file
    monkeypatch.setattr(
        engine_registry, "read_relation_file",
        lambda *args: reads.append(args) or read(*args),
    )

    def table(nl):
        return sorted(nl.symbol_table["rows"].value.unwrap())

    data_dir = tmp_path / "data"
    first = engine_registry.build_engine("r", data_dir)
    second = engine_registry.build_engine("r", data_dir)
    assert len(reads) == 1 and table(second) == table(first)
    assert table(first) == [(1, "x"), (2, "y")]
    rows.write_text("a,b\n3,z\n")
    assert table(engine_registry.build_engine("r", data_dir)) == [(3, "z")]
    assert len(reads) == 2


def test_a_table_is_computed_without_a_program_keeping_tables(tmp_path):
    import pandas as pd

    frame = pd.DataFrame({"i": [1, 2]})

    class Program:
        pass

    program = Program()
    assert engine_cache.keep_table(
        program, "t", [], (), lambda: frame
    ) is frame
    program.derived_tables = engine_cache.DerivedTables(tmp_path)
    engine_cache.keep_table(program, "t", [], (1,), lambda: frame)
    kept = engine_cache.keep_table(
        program, "t", [], (1,), lambda: pytest.fail("kept")
    )
    assert kept.equals(frame)


def test_a_file_is_named_by_its_bytes(tmp_path):
    one, other = tmp_path / "one", tmp_path / "other"
    one.write_bytes(b"rows")
    other.write_bytes(b"rows")
    assert engine_cache.file_digest(one) == engine_cache.file_digest(other)
    other.write_bytes(b"rows!")
    assert engine_cache.file_digest(one) != engine_cache.file_digest(other)
