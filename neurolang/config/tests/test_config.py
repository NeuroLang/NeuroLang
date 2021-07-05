import pytest
from .. import config

@pytest.fixture
def clean_config_fixture():
    backend = config["RAS"]["backend"]
    yield
    config["RAS"]["backend"] = backend

def test_config_set_backend(clean_config_fixture):
    with pytest.raises(AssertionError):
        config.set_backend("hello")

    config.set_backend("dask")
    assert config["RAS"]["backend"] == "dask"

    config.set_backend("pandas")
    assert config["RAS"]["backend"] == "pandas"
