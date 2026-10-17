"""Keep the engine tests off the network: the Cognitive Atlas ontology of
the ``coordinate_meta_analysis`` engines is replaced by a small file."""
from pathlib import Path

import pytest

from neurolang.utils import engine_registry

MINI_COGAT = Path(__file__).parent / "data" / "mini_cogat.owl"


@pytest.fixture(autouse=True)
def offline_cognitive_atlas(monkeypatch):
    original = engine_registry._ontology_file

    def ontology_file(source, *args, **kwargs):
        if "CognitiveAtlas/cogat-ontology" in source:
            return str(MINI_COGAT)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(engine_registry, "_ontology_file", ontology_file)
