"""Switching the backend of the relational algebra sets."""
import pytest
from nl_expr.config import config
from .. import pandas

try:
    from .. import polars
except ImportError:
    polars = None

@pytest.fixture
def clean_config_fixture():
    backend = config["RAS"]["backend"]
    sort_join_predicates = config["CHASE"]["sort_join_predicates"]
    try:
        yield
    finally:
        config.set_query_backend(backend)
        config["CHASE"]["sort_join_predicates"] = sort_join_predicates

def test_config_set_query_backend(clean_config_fixture):
    with pytest.raises(ValueError, match="Unknown query backend 'hello'"):
        config.set_query_backend("hello")

    config.set_query_backend("pandas")
    assert config["RAS"]["backend"] == "pandas"
    from .. import RelationalAlgebraFrozenSet
    assert RelationalAlgebraFrozenSet is pandas.RelationalAlgebraFrozenSet


@pytest.mark.skipif(polars is None, reason="polars is not installed")
def test_config_set_query_backend_polars(clean_config_fixture):
    config.set_query_backend("polars")
    assert config["RAS"]["backend"] == "polars"
    from .. import RelationalAlgebraFrozenSet
    assert RelationalAlgebraFrozenSet is polars.RelationalAlgebraFrozenSet


def test_config_set_query_backend_dask_removed(clean_config_fixture):
    backend = config["RAS"]["backend"]
    with pytest.raises(ValueError, match="no longer available") as excinfo:
        config.set_query_backend("dask")
    assert "'pandas'" in str(excinfo.value)
    assert "'polars'" in str(excinfo.value)
    assert config["RAS"]["backend"] == backend


def test_backend_dependent_modules_registered_for_reload():
    from ....probabilistic import weighted_model_counting  # noqa: F401
    modules = config._backend_dependent_modules
    assert modules.index("nl_core.utils.relational_algebra_set") < (
        modules.index("nl_core.probabilistic.weighted_model_counting")
    )
