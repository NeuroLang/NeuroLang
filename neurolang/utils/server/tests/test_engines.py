import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import nibabel as nib
import numpy as np
from neurolang import NeurolangDL
from neurolang.frontend.probabilistic_frontend import NeurolangPDL
from ..engines import (
    NeurolangEngineSet,
    NeurosynthEngineConf,
    DestrieuxEngineConf,
    load_destrieux_atlas,
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
            raise RuntimeError("Oops an error occurred")
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


def test_destrieux_engine_conf_key():
    """DestrieuxEngineConf should have key == 'destrieux'."""
    data_dir = Path("neurolang_data")
    conf = DestrieuxEngineConf(data_dir)
    assert conf.key == "destrieux"


def test_load_destrieux_atlas_new_format():
    """
    load_destrieux_atlas should correctly handle the new nilearn format
    where labels is a plain list of strings (nilearn >= 0.10).
    Background is excluded; hyphens and underscores are replaced with spaces.
    """
    # Build a tiny fake atlas image (4x4x4 with labels 0, 1, 2)
    data = np.zeros((4, 4, 4), dtype=np.int32)
    data[0, 0, 0] = 1
    data[1, 1, 1] = 2
    fake_image = nib.Nifti1Image(data, np.eye(4))

    # New-format labels: list of plain strings
    new_format_labels = [
        "Background",
        "L_G-and-S_frontomargin",
        "R_G-and-S_occipital",
    ]

    fake_atlas = {"maps": "/fake/path.nii.gz", "labels": new_format_labels}

    nl_mock = MagicMock()

    with (
        patch(
            "neurolang.utils.server.engines.datasets"
            ".fetch_atlas_destrieux_2009",
            return_value=fake_atlas,
        ),
        patch(
            "neurolang.utils.server.engines.nib.load",
            return_value=fake_image,
        ),
    ):
        load_destrieux_atlas(Path("neurolang_data"), nl_mock)

    nl_mock.add_atlas_set.assert_called_once()
    _, call_kwargs = nl_mock.add_atlas_set.call_args[0], {}
    labels_dict = nl_mock.add_atlas_set.call_args[0][1]

    # Background (index 0) should be excluded
    assert 0 not in labels_dict
    # Label 1 should be present with hyphens/underscores replaced
    assert 1 in labels_dict
    assert labels_dict[1] == "L G and S frontomargin"
    # Label 2 should be present
    assert 2 in labels_dict
    assert labels_dict[2] == "R G and S occipital"


def test_load_destrieux_atlas_old_format():
    """
    load_destrieux_atlas should correctly handle the old nilearn format
    where labels is a list of (int_label, bytes_name) tuples.
    """
    data = np.zeros((4, 4, 4), dtype=np.int32)
    data[0, 0, 0] = 1
    fake_image = nib.Nifti1Image(data, np.eye(4))

    # Old-format labels: list of (int, bytes) tuples
    old_format_labels = [
        (0, b"Background"),
        (1, b"L_G-and-S_frontomargin"),
    ]

    fake_atlas = {"maps": "/fake/path.nii.gz", "labels": old_format_labels}
    nl_mock = MagicMock()

    with (
        patch(
            "neurolang.utils.server.engines.datasets"
            ".fetch_atlas_destrieux_2009",
            return_value=fake_atlas,
        ),
        patch(
            "neurolang.utils.server.engines.nib.load",
            return_value=fake_image,
        ),
    ):
        load_destrieux_atlas(Path("neurolang_data"), nl_mock)

    labels_dict = nl_mock.add_atlas_set.call_args[0][1]
    # Background should be excluded
    assert 0 not in labels_dict
    assert 1 in labels_dict
    assert labels_dict[1] == "L G and S frontomargin"


def test_destrieux_engine_configuration():
    """
    DestrieuxEngineConf.create() should return a NeurolangPDL instance
    with a 'destrieux' atlas set symbol.
    """
    data_dir = Path("neurolang_data")
    conf = DestrieuxEngineConf(data_dir)
    assert conf.key == "destrieux"

    engine = conf.create()
    assert isinstance(engine, NeurolangPDL)
    assert "destrieux" in engine.symbols
