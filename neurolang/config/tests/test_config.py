import pytest
from .. import config

@pytest.fixture
def clean_config_fixture():
    backend = config["RAS"]["backend"]
    yield
    config["RAS"]["backend"] = backend

def test_config_set_query_backend(clean_config_fixture):
    with pytest.raises(ValueError):
        config.set_query_backend("hello")

    config.set_query_backend("dask")
    assert config["RAS"]["backend"] == "dask"

    config.set_query_backend("pandas")
    assert config["RAS"]["backend"] == "pandas"
