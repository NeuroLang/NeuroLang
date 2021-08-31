import pytest
from pathlib import Path
from neurolang import NeurolangDL
from neurolang.frontend.probabilistic_frontend import NeurolangPDL
from neurolang.utils.server.engines import (
    NeurolangEngineSet,
    NeurosynthEngineConf,
)


@pytest.fixture
def engine():
    return NeurolangDL()


def test_neurolang_engine_set(engine):
    """
    NeurolangEngineSet should manage engines with a locking semaphore
    to control that threads cannot acquire more engines than are
    available
    """
    es = NeurolangEngineSet(engine)
    assert es.counter == 1
    assert es.sema.acquire()

    # try to get engine, wait at most 1s
    with es.engine(timeout=1) as engine_:
        # should not return engine since we haven't released semaphore
        assert engine_ is None

    es.sema.release()
    with es.engine() as engine_:
        assert engine_ is engine

    # test that resources are released even when an exception is raised
    try:
        with es.engine() as engine_:
            raise RuntimeError("Oops an error occured")
    except RuntimeError:
        pass
    assert len(es.engines) == 1
    assert es.sema.acquire()


def test_neurosynth_engine_configuration():
    """
    NeurosynthEngineConf should have a unique key == "neurosynth"
    and `create` method should return an instance of `NeurolangPDL` with
    `PeakReported`, `Study`, `TermInStudyTFIDF`, `SelectedStudy` & `Voxel`
    symbols.
    """
    data_dir = Path("neurolang_data")
    conf = NeurosynthEngineConf(data_dir)
    assert conf.key == "neurosynth"

    engine = conf.create()
    assert isinstance(engine, NeurolangPDL)
    assert "PeakReported" in engine.symbols
    assert "Study" in engine.symbols
    assert "TermInStudyTFIDF" in engine.symbols
    assert "SelectedStudy" in engine.symbols
    assert "Voxel" in engine.symbols
