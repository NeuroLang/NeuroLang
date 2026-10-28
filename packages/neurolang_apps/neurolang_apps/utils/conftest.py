"""Keep the engine tests off the network: the Cognitive Atlas ontology of
the ``coordinate_meta_analysis`` engines is replaced by a small file."""
from pathlib import Path

import pytest

from neurolang_apps.utils import engine_registry

MINI_COGAT = Path(__file__).parent / "tests" / "mini_cogat.owl"


# Session-scoped, so that it also covers module-scoped engine fixtures.
@pytest.fixture(autouse=True, scope="session")
def offline_cognitive_atlas():
    original = engine_registry._ontology_file

    def ontology_file(source, *args, **kwargs):
        if "CognitiveAtlas/cogat-ontology" in source:
            return str(MINI_COGAT)
        return original(source, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(engine_registry, "_ontology_file", ontology_file)
        yield
