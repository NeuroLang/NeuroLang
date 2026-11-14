"""Keep the engine tests off the network: the Cognitive Atlas ontology of
the ``coordinate_meta_analysis`` engines is replaced by a small file, and
PubMed and NeuroStore's listing of its studies are not asked for the
abstracts and years of the studies (they are those the data directory
kept)."""
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


@pytest.fixture(autouse=True, scope="session")
def offline_publications():
    from neurolang_apps.utils import text_embeddings
    from neurolang_apps.utils.engines.neurostore import init as neurostore

    def offline(*args, **kwargs):
        # Raised, not answered empty: an empty answer would be kept, in a
        # real data directory, as the studies without an abstract.
        raise OSError("the tests are offline")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(text_embeddings, "_post", offline)
        monkeypatch.setattr(neurostore, "_fetch_json", offline)
        yield
