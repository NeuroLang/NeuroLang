"""Declarative engine registry for the neurolang-query CLI.

Engines are declared in :file:`engines/engines.yaml`.  Each engine may
define:

* a **Python initialisation script** (``python_init``);
* optional **Datalog initialisation code** (``datalog_init``) evaluated
  after the Python script;
* optional **CSV/TSV relations** (``relations``) loaded as extensional
  predicates after the Datalog init.  Each relation entry may carry
  optional ``name`` and ``description`` fields for predicate metadata
  used by the ``--list-predicates`` command.  The ``file`` value may be
  a URL (``http://`` / ``https://``) for automatic download;
* ``downloads`` — files to fetch (with optional archive extraction);
* ``ontologies`` — OWL/RDF ontologies to load from URLs or local paths;
* ``builtins`` — list of known function names (``exp``, ``log``,
  ``startswith``, etc.) to register as callable symbols;
* ``probabilistic_choice`` — uniform probabilistic choices declared as
  ``name: {source: other_predicate}``.

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

import shutil
import tarfile
import urllib.request
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

def _builtin_startswith(prefix: str, s: str) -> bool:
    """Return True if *s* starts with *prefix*."""
    return s.startswith(prefix)


_BUILTIN_SYMBOLS: Dict[str, object] = {
    "exp": np.exp,
    "log": np.log,
    "startswith": _builtin_startswith,
}


def _download_url(url: str, dest: Path) -> Path:
    """Download *url* to *dest* (a file path) and return *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def _extract_archive(path: Path, dest_dir: Path) -> None:
    """Extract a tar or tar.gz archive into *dest_dir*."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if path.name.endswith((".tar.gz", ".tgz")):
        mode = "r:gz"
    elif path.name.endswith(".tar"):
        mode = "r:"
    else:
        shutil.unpack_archive(str(path), str(dest_dir))
        return
    with tarfile.open(path, mode) as tf:
        tf.extractall(path=dest_dir)


def _resolve_relation_file(
    rel_cfg: dict, engine_name: str, data_dir: Path
) -> Path:
    """Return a local path for a relation entry's ``file`` field.

    If the ``file`` value is a URL it is downloaded to
    ``{data_dir}/{engine_name}/`` first.  If ``extract: true`` and the
    downloaded file is an archive, it is extracted and *extract* (or the
    first member) is used.
    """
    raw = rel_cfg["file"]
    extract_member = rel_cfg.get("extract")

    if raw.startswith(("http://", "https://")):
        local = data_dir / engine_name / Path(raw).name
        _download_url(raw, local)
        resolved = local
    else:
        resolved = _ENGINES_DIR / raw

    if extract_member:
        extracted_dir = resolved.parent / resolved.stem
        _extract_archive(resolved, extracted_dir)
        if isinstance(extract_member, list):
            return extracted_dir / extract_member[0]
        members = sorted(extracted_dir.iterdir())
        if members:
            return members[0]
        return extracted_dir

    return resolved


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

    Combines the ``predicates`` section with metadata harvested from the
    ``relations`` section (each relation entry may carry ``name`` and
    ``description`` fields).
    """
    cfg = get_engine_config(name)
    predicates = dict(cfg.get("predicates", {}))

    for rel_key, rel_cfg in cfg.get("relations", {}).items():
        if isinstance(rel_cfg, str):
            continue
        name = rel_cfg.get("name", rel_key)
        desc = rel_cfg.get("description", "")
        if name not in predicates:
            predicates[name] = {"description": desc, "columns": []}
        elif desc and not predicates[name].get("description"):
            predicates[name]["description"] = desc

    for ch_name, ch_cfg in cfg.get("probabilistic_choice", {}).items():
        desc = ch_cfg.get("description", "")
        if ch_name not in predicates:
            predicates[ch_name] = {"description": desc, "columns": []}
        elif desc and not predicates[ch_name].get("description"):
            predicates[ch_name]["description"] = desc

    return predicates


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

    # ── Phase 0: builtins ──────────────────────────────────────────
    for sym_name in cfg.get("builtins", []):
        func = _BUILTIN_SYMBOLS.get(sym_name)
        if func is not None:
            nl.add_symbol(func, name=sym_name)

    # ── Phase 1: Python init ───────────────────────────────────────
    py_init = cfg.get("python_init")
    if py_init:
        import importlib

        mod = importlib.import_module(py_init)
        mod.init_engine(nl, mask, data_dir)

    # ── Phase 2: downloads ─────────────────────────────────────────
    for dl_entry in cfg.get("downloads", []):
        url = dl_entry["url"]
        dl_dest = data_dir / dl_entry.get("dest", name)
        dl_dest.mkdir(parents=True, exist_ok=True)
        local = dl_dest / Path(url).name
        _download_url(url, local)
        if dl_entry.get("extract", False):
            _extract_archive(local, dl_dest)

    # ── Phase 3: Datalog init ──────────────────────────────────────
    datalog_init = cfg.get("datalog_init")
    if datalog_init:
        nl.execute_datalog_program(datalog_init)

    # ── Phase 4: relations (CSV/TSV, local or URL) ─────────────────
    relations = cfg.get("relations", {})
    if relations:
        import pandas as pd

        for rel_name, rel_cfg in relations.items():
            if isinstance(rel_cfg, str):
                rel_cfg = {"file": rel_cfg}
            rel_path = _resolve_relation_file(
                rel_cfg, name, data_dir
            )
            name_lower = rel_path.name.lower()
            if name_lower.endswith(".csv") or name_lower.endswith(".csv.gz"):
                sep = ","
            elif name_lower.endswith(".tsv") or name_lower.endswith(".tsv.gz"):
                sep = "\t"
            else:
                raise ValueError(
                    f"Unsupported relation file format: {rel_path.suffix} "
                    f"for relation {rel_name!r}. "
                    "Supported: .csv, .tsv, .csv.gz, .tsv.gz"
                )
            df = pd.read_csv(rel_path, sep=sep)
            nl.add_tuple_set(
                [tuple(row) for row in df.to_numpy()], name=rel_name
            )

    # ── Phase 5: probabilistic choices ─────────────────────────────
    for ch_name, ch_cfg in cfg.get("probabilistic_choice", {}).items():
        source_val = nl.symbols[ch_cfg["source"]]
        # Unwrap Symbol → its value (DataFrame or RelationalAlgebraFrozenSet)
        if hasattr(source_val, "value"):
            source_val = source_val.value
        nl.add_uniform_probabilistic_choice_over_set(
            source_val, name=ch_name
        )

    # ── Phase 6: ontologies ────────────────────────────────────────
    for onto_cfg in cfg.get("ontologies", []):
        url = onto_cfg["url"]
        onto_path = _download_url(
            url,
            data_dir / name / Path(url).name,
        )
        nl.load_ontology(str(onto_path))

    return nl
