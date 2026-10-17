import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pytest

from neurolang.utils import engine_registry as er


def test_nilearn_mask_is_bundled(tmp_path):
    mask = er._get_mni_mask(tmp_path, {"source": "nilearn"})
    assert mask.shape == (99, 117, 95)
    assert mask.affine[0, 0] == 2
    assert list(tmp_path.iterdir()) == []  # nothing downloaded
    mask1 = er._get_mni_mask(tmp_path, {"source": "nilearn", "resolution": 1})
    assert mask1.shape == (197, 233, 189)


def test_default_source_is_the_icbm_download(tmp_path):
    img = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    fetch = MagicMock(return_value={"t1": "unused"})
    with patch.object(er, "_nilearn_datasets") as datasets, \
            patch.object(er.nib, "load", return_value=img):
        datasets.return_value.fetch_icbm152_2009 = fetch
        assert er._get_mni_mask(tmp_path) is img
    fetch.assert_called_once_with(data_dir=str(tmp_path / "icbm"))


def test_templateflow_mask(tmp_path):
    path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), "uint8"), np.eye(4)), path)
    get = MagicMock(return_value=[path])
    api = types.SimpleNamespace(get=get)
    fake = types.ModuleType("templateflow")
    fake.api = api
    with patch.dict(sys.modules, {"templateflow": fake}):
        mask = er._get_mni_mask(
            tmp_path,
            {
                "source": "templateflow", "template": "MNI152NLin6Asym",
                "suffix": "mask", "desc": "brain", "resolution": 2,
            },
        )
    assert mask.shape == (3, 3, 3)
    get.assert_called_once_with(
        "MNI152NLin6Asym", suffix="mask", extension=".nii.gz",
        resolution=2, desc="brain",
    )


def test_unknown_source(tmp_path):
    with pytest.raises(ValueError, match="mni_template source"):
        er._get_mni_mask(tmp_path, {"source": "nowhere"})


def test_engine_config_selects_the_template(tmp_path):
    cfg = {"requires_mni_mask": True, "mni_template": {"source": "nilearn"}}
    assert er._resolve_mask(cfg, Path(tmp_path)).shape == (99, 117, 95)
    resampled = er._resolve_mask(cfg, Path(tmp_path), resolution=4)
    assert resampled.affine[0, 0] == 4
    assert er._resolve_mask({}, Path(tmp_path)) is None


def test_nilearn_template_follows_the_requested_resolution(tmp_path):
    cfg = {"requires_mni_mask": True, "mni_template": {"source": "nilearn"}}
    assert er._resolve_mask(cfg, tmp_path, 1).shape == (197, 233, 189)
    assert er._resolve_mask(cfg, tmp_path, 2.0).shape == (99, 117, 95)
    # Not a native resolution: the 2 mm template is resampled.
    assert er._resolve_mask(cfg, tmp_path, 4).affine[0, 0] == 4


def test_placeholders():
    expand = er._expand_placeholders
    cfg = {
        "a": "${resolution}",
        "b": {"c": "${resolution:2}", "d": "${resolution}", "e": "x"},
        "f": ["${resolution}", "${other:1.5}", "keep"],
    }
    assert expand(cfg, {"resolution": 3}) == {
        "a": 3,
        "b": {"c": 3, "d": 3, "e": "x"},
        "f": [3, 1.5, "keep"],
    }
    # Without a value: the default, or the entry is dropped.
    assert expand(cfg, {}) == {
        "b": {"c": 2, "e": "x"},
        "f": [1.5, "keep"],
    }
    assert expand("a ${resolution} b", {"resolution": 1}) == "a ${resolution} b"


def test_the_yaml_reads_the_requested_resolution(tmp_path):
    yaml_path = tmp_path / "engines.yaml"
    yaml_path.write_text(
        "engines:\n"
        "  e:\n"
        "    requires_mni_mask: true\n"
        "    mni_template:\n"
        "      source: nilearn\n"
        "      resolution: ${resolution:2}\n"
    )
    for requested, shape in ((None, (99, 117, 95)), (1.0, (197, 233, 189))):
        cfg = er._expand_placeholders(
            er.get_engine_config("e", yaml_path),
            {"resolution": er._plain_number(requested)},
        )
        assert er._resolve_mask(cfg, tmp_path, requested).shape == shape


def test_engine_resolution_is_the_default(tmp_path):
    yaml_path = tmp_path / "engines.yaml"
    yaml_path.write_text(
        "engines:\n"
        "  base:\n"
        "    abstract: true\n"
        "    requires_mni_mask: true\n"
        "    resolution: 1\n"
        "    mni_template: {source: nilearn, resolution: '${resolution}'}\n"
        "  e:\n"
        "    extends: base\n"
        "  f:\n"
        "    extends: base\n"
        "    resolution: 2\n"
    )

    def shape(engine, requested):
        captured = {}

        def fake_mask(cfg, data_dir, res=None):
            captured["cfg"] = cfg
            return None

        with patch.object(er, "_resolve_mask", fake_mask), patch(
            "neurolang.frontend.NeurolangPDL"
        ):
            try:
                er.build_engine(
                    engine, tmp_path, resolution=requested,
                    yaml_path=yaml_path,
                )
            except Exception:  # only the resolved config matters here
                pass
        return captured["cfg"]["mni_template"]["resolution"]

    assert shape("e", None) == 1      # inherited default
    assert shape("f", None) == 2      # overridden default
    assert shape("e", 2.0) == 2       # command line wins
