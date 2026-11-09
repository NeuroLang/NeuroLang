"""The shipped bundle modules (``nl_tracts/bundles``): they are the
translation of tract_querier's ``.qry`` files, and the uncinate of
``freesurfer_queries`` is that of ``examples/uncinate.dl``."""

import importlib.util
from pathlib import Path

import pytest

from nl_tracts.frontend import tracts_engine
from nl_tracts.wmql_vocab import BUNDLES, bundle

from . import uncinate_files
from .reference_tq import data_file
from .test_uncinate import example

TOOL = Path(__file__).parents[2] / "tools" / "make_bundle_modules.py"


@pytest.mark.parametrize("name", BUNDLES)
def test_the_shipped_module_is_the_translation(name):
    if not TOOL.exists():
        pytest.skip(f"{TOOL} not found (an installed package)")
    spec = importlib.util.spec_from_file_location("make_bundles", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    data_dir = Path(data_file(f"{name}.qry")).parent
    assert tool.module_text(name, data_dir) == bundle(name), (
        "run tools/make_bundle_modules.py")


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    return uncinate_files.write_files(
        tmp_path_factory.mktemp("uncinate"), n_random=300, seed=3)


def rows(solution, name):
    return set(solution[name].as_pandas_dataframe().itertuples(
        index=False, name=None))


def test_the_uncinate_of_freesurfer_queries_is_the_example(files):
    """``uf(x, h)``, translated from ``UF.side``, and ``uncinate(x, h)``,
    written by hand, select the same streamlines, designed and random; every
    rule of the module runs."""
    nl = tracts_engine()
    nl.execute_datalog_program(example(files).replace(
        'ans(x) :- uncinate(x, "left")', ""))
    nl.execute_datalog_program(bundle("freesurfer_queries"))
    solution = nl.solve_all()
    uf, uncinate = rows(solution, "uf"), rows(solution, "uncinate")
    assert uf == uncinate
    for side in ("left", "right"):
        designed = {(x, side) for x in uncinate_files.expected(side)}
        assert designed <= uf
        assert len(designed) + 10 < len({x for x, h in uf if h == side})
