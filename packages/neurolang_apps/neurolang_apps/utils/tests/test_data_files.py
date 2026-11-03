"""Tests of the files of a data directory fetched when first needed."""

import nibabel as nib
import numpy as np

from neurolang_apps.utils import data_files, engine_registry


def test_without_a_fetcher_nothing_is_fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(data_files, "fetch_missing", None)
    path = tmp_path / "missing.npz"
    assert data_files.ensure(path) == path and not path.exists()


def test_only_missing_files_are_fetched(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(data_files, "fetch_missing", fetched.append)
    present = tmp_path / "present.npz"
    present.write_bytes(b"x")
    data_files.ensure(present)
    data_files.ensure(tmp_path / "missing.npz")
    assert fetched == [tmp_path / "missing.npz"]


def test_the_mni_template_is_fetched_before_nilearn_looks_for_it(
    tmp_path, monkeypatch
):
    """In a browser nilearn cannot download the template: the page's site
    has it, fetched before nilearn's fetcher, which then finds it."""
    template = tmp_path / "icbm" / engine_registry.ICBM_T1
    image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))

    def fetch(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(image, path)

    class Datasets:
        @staticmethod
        def fetch_icbm152_2009(data_dir):
            assert template.exists(), "nilearn would download it"
            return {"t1": str(template)}

    monkeypatch.setattr(data_files, "fetch_missing", fetch)
    monkeypatch.setattr(engine_registry, "_nilearn_datasets", Datasets)
    mask = engine_registry._get_mni_mask(tmp_path)
    assert mask.shape == (2, 2, 2)
