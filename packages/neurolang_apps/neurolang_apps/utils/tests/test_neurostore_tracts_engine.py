"""The engines ``neurostore_tracts`` and ``neurostore_iit`` of engines.yaml:
neurostore with the tracts, as ``neurosynth_tracts`` and ``neurosynth_iit``
are neurosynth with them."""

from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.llm.examples import load_examples

NS_INIT = "neurolang_apps.utils.engines.neurostore.init"
TRACTS = "neurolang_apps.utils.engines.tracts.frontend"


def test_neurostore_tracts_is_neurostore_and_tracts():
    cfg = engine_registry.get_engine_config("neurostore_tracts")
    assert cfg["python_init"] == [NS_INIT, TRACTS]
    assert cfg["solver_mixins"] == [f"{TRACTS}:TractsCommandsMixin"]
    schaefer = cfg["atlases"]["schaefer"]
    assert schaefer["activated"]["predicate_name"] == "schaefer_activated"
    assert schaefer["points"]["predicate_name"] == "schaefer_points"
    # the studies of neurostore, its 200 replicates and its embeddings
    neurostore = engine_registry.get_engine_config("neurostore")
    assert cfg["resamplings"] == neurostore["resamplings"]
    assert cfg["text_embeddings"] == neurostore["text_embeddings"]
    predicates = engine_registry.get_predicates("neurostore_tracts")
    assert {"study_concept", "schaefer_activated", "endpoint_point",
            "streamline_label"} <= set(predicates)


def test_neurostore_tracts_has_the_uncinate_question():
    bank = load_examples("neurostore_tracts")
    assert bank[:-1] == load_examples("neurostore")
    assert "uncinate" in bank[-1].question
    assert bank[-1] == load_examples("neurosynth_tracts")[-1]


def test_neurostore_iit_loads_the_tractogram_of_neurosynth_iit():
    cfg = engine_registry.get_engine_config("neurostore_iit")
    iit = engine_registry.get_engine_config("neurosynth_iit")
    assert cfg["python_init"] == [NS_INIT, TRACTS]
    tail = iit["datalog_init"].rstrip().splitlines()[-4:]
    assert cfg["datalog_init"].rstrip().splitlines()[-4:] == tail
    assert tail[0].startswith('.load_tractogram("iit"')
    for key in ("downloads", "tractogram", "questions"):
        assert cfg[key] == iit[key]
    assert cfg["text_embeddings"] == engine_registry.get_engine_config(
        "neurostore")["text_embeddings"]


def test_neurostore_iit_has_the_questions_of_the_bundles():
    bank = load_examples("neurostore_iit")
    assert bank[:-3] == load_examples("neurostore_tracts")
    assert bank[-3:] == load_examples("neurosynth_iit")[-3:]
    assert bank[-3].queries["datalog"] == \
        'ans(streamline) :- uf(streamline, "left")'
