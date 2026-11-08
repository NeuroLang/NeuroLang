import dataclasses

import numpy
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ..euclidean_join import euclidean_join, euclidean_pairs
from ..euclidean_join_settings import (
    CONFIG_ENV,
    CONFIG_PATH,
    SETTINGS,
    JoinSettings,
    override,
    profile,
    read_profile,
    use_profile,
)


@pytest.fixture(autouse=True)
def restore_settings():
    before = dataclasses.replace(SETTINGS)
    yield
    for field in dataclasses.fields(before):
        setattr(SETTINGS, field.name, getattr(before, field.name))


def test_the_file_defines_each_setting_once():
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH))
    names = {field.name for field in dataclasses.fields(JoinSettings)}
    assert set(config["defaults"]) == names
    for changes in config["profiles"].values():
        assert set(changes) <= names


def test_the_default_profile_is_the_defaults():
    defaults = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH))["defaults"]
    assert dataclasses.asdict(read_profile("default")) == defaults


def test_the_memory_profile_changes_only_what_it_lists():
    default, memory = read_profile("default"), read_profile("memory")
    changed = {
        name for name, value in dataclasses.asdict(memory).items()
        if value != getattr(default, name)
    }
    assert {"stencil_count_first", "kdtree_chunk_pairs"} <= changed
    assert memory.kdtree_chunk_pairs < default.kdtree_chunk_pairs
    assert memory.stencil_min_work > default.stencil_min_work


def test_use_profile_changes_the_settings_in_use():
    use_profile("memory")
    assert SETTINGS.stencil_count_first
    use_profile("default")
    assert not SETTINGS.stencil_count_first


def test_profile_restores_the_settings_it_found():
    use_profile("default")
    with profile("memory") as settings:
        assert settings is SETTINGS and SETTINGS.stencil_count_first
    assert not SETTINGS.stencil_count_first


def test_override_changes_some_of_the_settings_in_use():
    use_profile("default")
    override({"kdtree_chunk_pairs": 123, "grid_steps": [1, 0.5]})
    assert SETTINGS.kdtree_chunk_pairs == 123
    assert list(SETTINGS.grid_steps) == [1, 0.5]
    assert SETTINGS.kdtree_sample == read_profile("default").kdtree_sample


def test_use_profile_applies_overrides_over_the_profile():
    use_profile("memory", overrides={"block_pairs": 5})
    assert SETTINGS.block_pairs == 5 and SETTINGS.stencil_count_first


@pytest.mark.parametrize(
    "overrides",
    [{"kdtree_chunk": 1}, {"kdtree_chunk_pairs": "many"}],
)
def test_overrides_must_be_known_settings_of_the_right_type(overrides):
    use_profile("default")
    before = dataclasses.replace(SETTINGS)
    with pytest.raises(OmegaConfBaseException):
        override(overrides)
    assert SETTINGS == before


def test_an_unknown_profile_is_an_error():
    with pytest.raises(ValueError, match="no profile 'fast'"):
        use_profile("fast")


def test_another_file_can_be_read(tmp_path, monkeypatch):
    path = tmp_path / "join.yaml"
    path.write_text(
        "defaults: {stencil_max_cells: 1, stencil_min_work: 2,"
        " stencil_work_per_point: 3, stencil_grid_cells_per_point: 4,"
        " stencil_count_first: false, grid_cells_per_point: 5,"
        " grid_min_cells: 6, grid_steps: [1], grid_max_coordinate: 7,"
        " grid_tolerance: 0.1, kdtree_chunk_pairs: 8, kdtree_sample: 9,"
        " block_pairs: 10}\n"
        "profiles: {default: {}, tiny: {block_pairs: 1}}\n"
    )
    monkeypatch.setenv(CONFIG_ENV, str(path))
    use_profile("tiny")
    assert SETTINGS.block_pairs == 1 and SETTINGS.kdtree_sample == 9


def _brute_force(first, second, radius):
    return {
        (i, j) for i, p in enumerate(first) for j, q in enumerate(second)
        if numpy.linalg.norm(numpy.subtract(p, q)) < radius
    }


@pytest.mark.parametrize("step", [1, 0.5])
@pytest.mark.parametrize("name", ["default", "memory"])
def test_each_profile_finds_the_same_pairs(name, step):
    rng = numpy.random.default_rng(5)
    first = numpy.unique(rng.integers(0, 14, size=(80, 3)), axis=0) * step
    second = numpy.unique(rng.integers(0, 14, size=(300, 3)), axis=0) * step
    with profile(name):
        # Blocks of one pair exercise each block loop.
        SETTINGS.block_pairs = 1
        SETTINGS.kdtree_chunk_pairs = 1
        i, j, d = euclidean_pairs(first, second, 3 * step, True)
    assert set(zip(i.tolist(), j.tolist())) == _brute_force(
        first, second, 3 * step
    )
    assert len(i) == len(d)


@pytest.mark.parametrize("name", ["default", "memory"])
def test_each_profile_gives_the_same_join(name):
    import pandas

    rng = numpy.random.default_rng(6)
    grid = pandas.DataFrame(
        numpy.stack(numpy.meshgrid(*[numpy.arange(10)] * 3), -1).reshape(-1, 3),
        columns=["i", "j", "k"],
    )
    peaks = pandas.DataFrame(
        rng.integers(0, 10, size=(400, 3)), columns=["i1", "j1", "k1"]
    ).assign(s=rng.integers(0, 20, 400)).drop_duplicates()
    args = (
        (grid, ["i", "j", "k"]), (peaks, ["i1", "j1", "k1"]), 2.5, True,
        ["i", "j", "k", "s"],
    )
    expected = euclidean_join(*args)
    with profile(name):
        SETTINGS.block_pairs = 7
        found = euclidean_join(*args)
    key = ["i", "j", "k", "s"]
    pandas.testing.assert_frame_equal(
        found.sort_values(key).reset_index(drop=True),
        expected.sort_values(key).reset_index(drop=True),
    )
