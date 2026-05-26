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
* ``atlases`` — brain atlases downloaded via nilearn (e.g. ``destrieux``,
  ``schaefer``, ``difumo``) and registered as predicates with
  :class:`~neurolang.regions.ExplicitVBR` regions.  Deterministic atlases
  use ``ExplicitVBR.from_spatial_image_label``; probabilistic atlases
  are thresholded (default 0.5) to create binary region masks.

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

try:
    from nilearn.datasets.utils import _fetch_files
except ImportError:
    from nilearn.datasets._utils import fetch_files as _fetch_files

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


def _resolve_relation_file(
    rel_cfg: dict, engine_name: str, data_dir: Path
) -> Path:
    """Return a local path for a relation entry's ``file`` field.

    If the ``file`` value is a URL it is downloaded via nilearn's
    ``_fetch_files`` (with caching and optional archive extraction)
    to ``{data_dir}/{engine_name}/``.
    """
    raw = rel_cfg["file"]
    extract_member = rel_cfg.get("extract")

    if raw.startswith(("http://", "https://")):
        dl_dir = str(data_dir / engine_name)
        name = Path(raw).name
        opts = {}
        if extract_member:
            opts["uncompress"] = True
            # If extract specifies a member name, use that as the target file
            if isinstance(extract_member, list):
                name = extract_member[0]
            elif extract_member is not True:
                name = extract_member
        result = _fetch_files(dl_dir, [(name, raw, opts)], verbose=0)
        return Path(result[0])
    else:
        resolved = _ENGINES_DIR / raw
        if extract_member:
            dl_dir = str(resolved.parent)
            name = resolved.name
            opts = {"uncompress": True}
            result = _fetch_files(dl_dir, [(name, str(resolved), opts)], verbose=0)
            return Path(result[0])
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

    for atl_name, atl_params in cfg.get("atlases", {}).items():
        pred_name = atl_params.get("predicate_name", atl_name)
        desc = atl_params.get("description", f"{atl_name} atlas regions")
        if pred_name not in predicates:
            predicates[pred_name] = {"description": desc, "columns": []}
        elif not predicates[pred_name].get("description"):
            predicates[pred_name]["description"] = desc

    return predicates


# ---------------------------------------------------------------------------
# Atlas helpers
# ---------------------------------------------------------------------------

def _fetch_atlas_destrieux(**kwargs):
    return datasets.fetch_atlas_destrieux_2009(**kwargs)


def _fetch_atlas_schaefer(**kwargs):
    return datasets.fetch_atlas_schaefer_2018(**kwargs)


def _fetch_atlas_difumo(**kwargs):
    return datasets.fetch_atlas_difumo(**kwargs)


_ATLAS_REGISTRY = {
    "destrieux": {
        "fetch": _fetch_atlas_destrieux,
        "probabilistic": False,
    },
    "schaefer": {
        "fetch": _fetch_atlas_schaefer,
        "probabilistic": False,
    },
    "difumo": {
        "fetch": _fetch_atlas_difumo,
        "probabilistic": True,
    },
}


def _parse_labels(raw_labels):
    """Yield ``(index, name)`` pairs from a nilearn labels value.

    Handles the three formats returned by different nilearn versions:
    ``dict``, ``list``, and numpy structured array.
    """
    if isinstance(raw_labels, dict):
        yield from raw_labels.items()
    elif isinstance(raw_labels, list):
        yield from enumerate(raw_labels)
    else:
        # numpy structured array with two columns
        yield from raw_labels


def _label_name(val: Any) -> str:
    """Normalise a label value to a clean string."""
    if isinstance(val, bytes):
        val = val.decode("utf8")
    name = str(val).replace("-", " ").replace("_", " ")
    return name


def _load_deterministic_atlas(
    nl, atlas_name: str, params: dict, data_dir: Path
) -> None:
    """Load a deterministic atlas (e.g. Destrieux, Schaefer) as engine predicates."""
    info = _ATLAS_REGISTRY[atlas_name]
    fetch_kw = {k: v for k, v in params.items() if k != "predicate_name"}
    fetch_kw["data_dir"] = str(data_dir)
    atl_data = info["fetch"](**fetch_kw)

    from neurolang.regions import ExplicitVBR

    img = nib.load(atl_data["maps"])
    pred_name = params.get("predicate_name", atlas_name)
    rows = []
    for k, v in _parse_labels(atl_data["labels"]):
        if k == 0:
            continue
        name = _label_name(v)
        rows.append((name, ExplicitVBR.from_spatial_image_label(img, k)))

    nl.add_tuple_set(rows, name=pred_name)


def _load_probabilistic_atlas(
    nl, atlas_name: str, params: dict, data_dir: Path
) -> None:
    """Load a probabilistic atlas (e.g. DiFuMo) as engine predicates.

    Each probability map is thresholded (default 0.5) to create a binary
    :class:`~neurolang.regions.ExplicitVBR`.
    """
    info = _ATLAS_REGISTRY[atlas_name]
    fetch_kw = {k: v for k, v in params.items() if k != "predicate_name"}
    fetch_kw["data_dir"] = str(data_dir)
    atl_data = info["fetch"](**fetch_kw)

    from neurolang.regions import ExplicitVBR

    img = nib.load(atl_data["maps"])
    data = img.get_fdata()
    labels = list(atl_data["labels"])
    threshold = float(params.get("threshold", 0.5))
    pred_name = params.get("predicate_name", atlas_name)

    rows = []
    for i in range(data.shape[-1]):
        mask = data[..., i] > threshold
        if not mask.any():
            continue
        voxels = np.argwhere(mask)
        region = ExplicitVBR(voxels, img.affine, image_dim=img.shape[:3])
        label = _label_name(labels[i] if i < len(labels) else f"component_{i}")
        rows.append((label, region))

    nl.add_tuple_set(rows, name=pred_name)


def _load_atlas(
    nl, atlas_name: str, params: dict, data_dir: Path
) -> None:
    """Download a brain atlas via nilearn and register it as a predicate."""
    info = _ATLAS_REGISTRY.get(atlas_name)
    if info is None:
        raise ValueError(
            f"Unknown atlas {atlas_name!r}. "
            f"Available: {', '.join(sorted(_ATLAS_REGISTRY))}"
        )
    if info["probabilistic"]:
        _load_probabilistic_atlas(nl, atlas_name, params, data_dir)
    else:
        _load_deterministic_atlas(nl, atlas_name, params, data_dir)


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
        dl_dest = str(data_dir / dl_entry.get("dest", name))
        extract_members = dl_entry.get("extract", [])
        if isinstance(extract_members, bool):
            # extract: true → download the archive and extract all
            archive_name = Path(url).name
            _fetch_files(dl_dest, [(archive_name, url, {"uncompress": True})], verbose=1)
        elif extract_members:
            # extract: [member1, member2] → download archive, extract specific files
            files = [(m, url, {"uncompress": True}) for m in extract_members]
            _fetch_files(dl_dest, files, verbose=1)
        else:
            # plain file download
            file_name = Path(url).name
            _fetch_files(dl_dest, [(file_name, url, {})], verbose=1)

    # ── Phase 3: atlases ───────────────────────────────────────────
    for atl_name, atl_params in cfg.get("atlases", {}).items():
        _load_atlas(nl, atl_name, atl_params, data_dir / name)

    # ── Phase 4: Datalog init ──────────────────────────────────────
    datalog_init = cfg.get("datalog_init")
    if datalog_init:
        nl.execute_datalog_program(datalog_init)

    # ── Phase 5: relations (CSV/TSV, local or URL) ─────────────────
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
        onto_dir = str(data_dir / name)
        onto_file = Path(url).name
        result = _fetch_files(onto_dir, [(onto_file, url, {})], verbose=1)
        nl.load_ontology(result[0])

    return nl
