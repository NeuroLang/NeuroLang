"""Declarative engine registry for the neurolang-query CLI.

Engines are declared in :file:`engines/engines.yaml`.  Each engine may
define a **Python initialisation script** (``python_init``) and an
optional **Datalog initialisation file** (``datalog_init``) that is
evaluated after the Python script completes.

Usage
-----
::

    from neurolang.utils.engine_registry import (
        build_engine,
        list_engine_names,
        get_engine_config,
    )

    nl = build_engine("neurosynth", data_dir=Path("data"))
    for name in list_engine_names():
        print(name)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import yaml
from nilearn import datasets, image

if TYPE_CHECKING:
    from neurolang.frontend import NeurolangPDL

_ENGINES_DIR = Path(__file__).parent / "engines"
_ENGINES_YAML = _ENGINES_DIR / "engines.yaml"


def _load_config() -> dict:
    with open(_ENGINES_YAML) as f:
        return yaml.safe_load(f)


def list_engine_names() -> List[str]:
    """Return the names of all registered engines."""
    config = _load_config()
    return sorted(config.get("engines", {}).keys())


def get_engine_config(name: str) -> Dict[str, Any]:
    """Return the YAML configuration dict for a single engine.

    Raises
    ------
    ValueError
        If *name* is not a registered engine.
    """
    config = _load_config()
    engines = config.get("engines", {})
    if name not in engines:
        available = ", ".join(engines.keys())
        raise ValueError(
            f"Unknown engine: {name!r}. Available engines: {available}"
        )
    return dict(engines[name])


def get_predicates(name: str) -> Dict[str, Dict[str, Any]]:
    """Return the predicate metadata dict for an engine from its YAML config.

    Returns an empty dict if the engine has no predicates declared.
    """
    cfg = get_engine_config(name)
    return dict(cfg.get("predicates", {}))


def _get_mni_mask(data_dir: Path) -> nib.Nifti1Image:
    return nib.load(
        datasets.fetch_icbm152_2009(
            data_dir=str(data_dir / "icbm")
        )["t1"]
    )


def build_engine(
    name: str,
    data_dir: Path,
    resolution: Optional[float] = None,
) -> "NeurolangPDL":
    """Create and populate a :class:`NeurolangPDL` engine from its
    declarative YAML config.

    Parameters
    ----------
    name :
        Registered engine name (e.g. ``"neurosynth"``, ``"destrieux"``).
    data_dir :
        Directory under which downloaded data is cached.
    resolution :
        If set, resample the MNI mask to the given isotropic resolution (mm).

    Returns
    -------
    NeurolangPDL
        A fully initialised engine ready for query execution.

    Raises
    ------
    ValueError
        If *name* is not a registered engine.
    ImportError
        If the engine's Python init module cannot be loaded.
    """
    from neurolang.frontend import NeurolangPDL

    cfg = get_engine_config(name)
    data_dir = Path(data_dir)

    mask = None
    if cfg.get("requires_mni_mask", False):
        mask = _get_mni_mask(data_dir)
        if resolution is not None:
            mask = image.resample_img(
                mask, np.eye(4) * resolution
            )

    nl = NeurolangPDL()

    py_init = cfg.get("python_init")
    if py_init:
        import importlib

        mod = importlib.import_module(py_init)
        mod.init_engine(nl, mask, data_dir)

    datalog_init = cfg.get("datalog_init")
    if datalog_init:
        nl.execute_datalog_program(datalog_init)

    return nl
