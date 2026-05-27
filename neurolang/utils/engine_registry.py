"""
Declarative engine registry for the neurolang-query CLI.

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
* ``templates`` — neuroimaging templates (brain masks, anatomical
  templates) downloaded via **nilearn** or **TemplateFlow** (if the
  ``templateflow`` package is installed).  Each template may optionally
  register a ``voxel(i, j, k)`` predicate from its non-zero mask::

      templates:
        mni_brain:
          source: nilearn
          variant: brain_mask
          predicate: voxel
        t1_ref:
          source: templateflow
          template: MNI152NLin2009cAsym
          resolution: 1

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

    from pathlib import Path
    from neurolang.utils.engine_registry import (
        build_engine,
        show_engines,
        list_engine_names,
        get_engine_config,
    )

    # Load engine from the built-in YAML
    nl = build_engine("neurosynth", data_dir=Path("data"))

    # Load engine from a custom YAML file
    nl = build_engine("my_engine", yaml_path=Path("my_engines.yaml"),
                      data_dir=Path("data"))

    # Print available engines with descriptions
    show_engines()

    # List engine names from a custom YAML
    for name in list_engine_names(yaml_path=Path("my_engines.yaml")):
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


def _merge_predicate_metadata(
    predicates: Dict[str, Dict[str, Any]],
    name: str,
    description: str = "",
    default_desc: str = "",
) -> None:
    """
    Insert or update a predicate entry's description in *predicates*.

    If *name* is not yet in *predicates*, a new ``{"description": ...,
    "columns": []}`` entry is added.  If it already exists and has no
    description, *description* (falling back to *default_desc*) is set.
    """
    if name not in predicates:
        predicates[name] = {
            "description": description or default_desc,
            "columns": [],
        }
    elif description and not predicates[name].get("description"):
        predicates[name]["description"] = description


def _load_yaml_config(yaml_path: Optional[Path] = None) -> dict:
    """Load engine YAML from *yaml_path* or the built-in engines file."""
    if yaml_path is not None:
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    with open(_ENGINES_YAML) as f:
        return yaml.safe_load(f)


def _resolve_relation_file(
    rel_cfg: dict,
    engine_name: str,
    data_dir: Path,
    base_dir: Optional[Path] = None,
) -> Path:
    """
    Return a local path for a relation entry's ``file`` field.

    If the ``file`` value is a URL it is downloaded via nilearn's
    ``_fetch_files`` (with caching and optional archive extraction)
    to ``{data_dir}/{engine_name}/``.

    Local paths are resolved relative to *base_dir* (defaults to the
    built-in ``engines/`` directory).
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
        resolved = (base_dir or _ENGINES_DIR) / raw
        if extract_member:
            dl_dir = str(resolved.parent)
            name = resolved.name
            opts = {"uncompress": True}
            result = _fetch_files(dl_dir, [(name, str(resolved), opts)], verbose=0)
            return Path(result[0])
        return resolved


def list_engine_names(yaml_path: Optional[Path] = None) -> List[str]:
    """
    Return the names of all registered engines.

    Parameters
    ----------
    yaml_path :
        Path to an engine YAML file.  Defaults to the built-in
        ``engines/engines.yaml``.

    """
    config = _load_yaml_config(yaml_path)
    return sorted(config.get("engines", {}).keys())


def get_engine_config(
    name: str, yaml_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Return the YAML configuration dict for a single engine.

    Parameters
    ----------
    name :
        Engine name (key in the YAML ``engines`` dict).
    yaml_path :
        Path to an engine YAML file.  Defaults to the built-in
        ``engines/engines.yaml``.

    Raises
    ------
    ValueError
        If *name* is not a registered engine.

    """
    config = _load_yaml_config(yaml_path)
    engines = config.get("engines", {})
    if name not in engines:
        available = ", ".join(engines.keys())
        raise ValueError(
            f"Unknown engine: {name!r}. Available engines: {available}"
        )
    return dict(engines[name])


def show_engines(yaml_path: Optional[Path] = None) -> None:
    """
    Print a formatted description of all available engines.

    Parameters
    ----------
    yaml_path :
        Path to an engine YAML file.  Defaults to the built-in
        ``engines/engines.yaml``.

    """
    config = _load_yaml_config(yaml_path)
    engines = config.get("engines", {})
    print("Available engines:\n")
    for name in sorted(engines):
        desc = engines[name].get("description", "").replace("\n", " ").strip()
        print(f"  {name}")
        if desc:
            print(f"      {desc}")
        print()


def get_predicates(
    name: str, yaml_path: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Return the predicate metadata dict for an engine from its YAML config.

    Combines the ``predicates`` section with metadata harvested from the
    ``relations`` section (each relation entry may carry ``name`` and
    ``description`` fields).

    Parameters
    ----------
    name :
        Engine name.
    yaml_path :
        Path to an engine YAML file.  Defaults to the built-in
        ``engines/engines.yaml``.

    """
    cfg = get_engine_config(name, yaml_path)
    predicates = dict(cfg.get("predicates", {}))

    for rel_key, rel_cfg in cfg.get("relations", {}).items():
        if isinstance(rel_cfg, str):
            continue
        _merge_predicate_metadata(
            predicates,
            rel_cfg.get("name", rel_key),
            rel_cfg.get("description", ""),
        )

    for ch_name, ch_cfg in cfg.get("probabilistic_choice", {}).items():
        _merge_predicate_metadata(
            predicates,
            ch_name,
            ch_cfg.get("description", ""),
        )

    for atl_name, atl_params in cfg.get("atlases", {}).items():
        pred_name = atl_params.get("predicate_name", atl_name)
        _merge_predicate_metadata(
            predicates,
            pred_name,
            atl_params.get("description", ""),
            default_desc=f"{atl_name} atlas regions",
        )
        # Probabilistic atlases also register a probability predicate
        if _ATLAS_REGISTRY.get(atl_name, {}).get("probabilistic"):
            prob_name = atl_params.get(
                "prob_predicate_name", f"{pred_name}_prob"
            )
            prob_desc = atl_params.get(
                "prob_description",
                f"Raw probabilities for {atl_name} atlas components",
            )
            _merge_predicate_metadata(predicates, prob_name, prob_desc)

    for tpl_name, tpl_params in cfg.get("templates", {}).items():
        pred_name = tpl_params.get("predicate")
        if pred_name:
            _merge_predicate_metadata(
                predicates,
                pred_name,
                tpl_params.get("description", ""),
                default_desc=f"Voxels inside the {tpl_name} template mask",
            )

    return predicates


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_NILEARN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "brain_mask": {
        "func": datasets.load_mni152_brain_mask,
        "kind": "mask",
    },
    "gm_mask": {
        "func": datasets.load_mni152_gm_mask,
        "kind": "mask",
    },
    "wm_mask": {
        "func": datasets.load_mni152_wm_mask,
        "kind": "mask",
    },
    "template": {
        "func": datasets.load_mni152_template,
        "kind": "anatomical",
    },
    "gm_template": {
        "func": datasets.load_mni152_gm_template,
        "kind": "anatomical",
    },
    "wm_template": {
        "func": datasets.load_mni152_wm_template,
        "kind": "anatomical",
    },
}


def _register_template_predicate(nl, img, pred_name):
    """Register ``pred_name(i, j, k)`` from all non-zero voxels of *img*."""
    data = np.asanyarray(img.dataobj)
    mask = data > 0
    if not mask.any():
        mask = data > data.min()  # fallback for masks with negative values
    coords = np.argwhere(mask)
    nl.add_tuple_set(coords, name=pred_name)


def _fetch_nilearn_template(
    nl, tpl_name: str, params: dict, data_dir: Path
) -> None:
    """
    Download a template via nilearn and optionally register a voxel predicate.

    Parameters
    ----------
    nl :
        Engine instance.
    tpl_name :
        YAML key for this template entry.
    params :
        Template config with keys:

        * ``variant`` — one of the keys in ``_NILEARN_TEMPLATES``
        * ``resolution`` — mm resolution (1 or 2, default 2; only for
          anatomical variants)
        * ``predicate`` — optional name for a ``(i, j, k)`` voxel set
          from non-zero mask voxels
    data_dir :
        Data cache directory.  Note that nilearn's ``load_mni152_*``
        functions use their own global cache (``~/nilearn_data``) rather
        than this path.
    """
    variant = params.get("variant", "brain_mask")
    info = _NILEARN_TEMPLATES.get(variant)
    if info is None:
        raise ValueError(
            f"Unknown nilearn template variant {variant!r}. "
            f"Available: {', '.join(sorted(_NILEARN_TEMPLATES))}"
        )

    kwargs: Dict[str, Any] = {}
    if info["kind"] == "anatomical":
        kwargs["resolution"] = params.get("resolution", 2)
    elif variant in ("brain_mask", "gm_mask", "wm_mask"):
        kwargs["resolution"] = params.get("resolution")

    img = info["func"](**kwargs)

    pred_name = params.get("predicate")
    if pred_name:
        _register_template_predicate(nl, img, pred_name)


def _fetch_templateflow_template(
    nl, tpl_name: str, params: dict
) -> None:
    """
    Download a template via TemplateFlow.

    Optionally registers a voxel predicate.
    Requires the ``templateflow`` Python package.

    """
    try:
        from templateflow import api as tflow
    except ImportError:
        raise ImportError(
            "The 'templateflow' source requires the `templateflow` package. "
            "Install it with: pip install templateflow"
        )

    tpl_id = params.get("template")
    if not tpl_id:
        raise ValueError(
            f"TemplateFlow entry {tpl_name!r} is missing the required "
            f"'template' field (e.g. 'MNI152NLin2009cAsym')."
        )

    resolution = params.get("resolution")
    suffix = params.get("suffix", "T1w")

    kwargs: Dict[str, Any] = dict(suffix=suffix)
    if resolution is not None:
        kwargs["resolution"] = resolution

    # templateflow.api.get() returns a Path to the downloaded file
    tpl_path = tflow.get(tpl_id, **kwargs)
    if tpl_path is None:
        raise FileNotFoundError(
            f"TemplateFlow template {tpl_id!r} (suffix={suffix}, "
            f"resolution={resolution}) could not be resolved."
        )

    pred_name = params.get("predicate")
    if pred_name:
        img = nib.load(str(tpl_path))
        _register_template_predicate(nl, img, pred_name)


def _load_template(
    nl, tpl_name: str, params: dict, data_dir: Path
) -> None:
    """
    Download a neuroimaging template and optionally register a voxel predicate.

    Dispatches to the backend named by *params['source']* (``nilearn`` or
    ``templateflow``).

    """
    source = params.get("source", "nilearn")
    if source == "nilearn":
        _fetch_nilearn_template(nl, tpl_name, params, data_dir)
    elif source == "templateflow":
        _fetch_templateflow_template(nl, tpl_name, params)
    else:
        raise ValueError(
            f"Unknown template source {source!r}. "
            f"Supported: 'nilearn', 'templateflow'."
        )


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
    """
    Yield ``(index, name)`` pairs from a nilearn labels value.

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
    """
    Load a probabilistic atlas (e.g. DiFuMo) as engine predicates.

    Registers two predicates:

    * ``{predicate_name}(component, region)`` — binary regions obtained by
      thresholding each probability map (default threshold 0.5).
    * ``{prob_predicate_name}(component, i, j, k, prob)`` — the raw
      probability value at each voxel for each component (filtered by
      ``prob_threshold``, default 0.01, to keep the set size manageable).

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
    prob_threshold = float(params.get("prob_threshold", 0.01))
    pred_name = params.get("predicate_name", atlas_name)
    prob_pred_name = params.get(
        "prob_predicate_name", f"{pred_name}_prob"
    )

    # ── Binary region predicate (thresholded) ──
    region_rows = []
    for i in range(data.shape[-1]):
        mask = data[..., i] > threshold
        if not mask.any():
            continue
        voxels = np.argwhere(mask)
        region = ExplicitVBR(voxels, img.affine, image_dim=img.shape[:3])
        label = _label_name(labels[i] if i < len(labels) else f"component_{i}")
        region_rows.append((label, region))

    nl.add_tuple_set(region_rows, name=pred_name)

    # ── Probability predicate (component, i, j, k, prob) ──
    prob_rows = []
    for i in range(data.shape[-1]):
        prob_map = data[..., i]
        mask = prob_map > prob_threshold
        if not mask.any():
            continue
        indices = np.argwhere(mask)
        values = prob_map[mask]
        label = _label_name(labels[i] if i < len(labels) else f"component_{i}")
        for (j_idx, k_idx, l_idx), prob in zip(indices, values):
            prob_rows.append((label, int(j_idx), int(k_idx), int(l_idx), float(prob)))

    nl.add_tuple_set(prob_rows, name=prob_pred_name)


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


def _resolve_mask(
    cfg: dict, data_dir: Path, resolution: Optional[float] = None
) -> Optional[nib.Nifti1Image]:
    """Download the MNI mask if required by the engine config."""
    if not cfg.get("requires_mni_mask", False):
        return None
    mask = _get_mni_mask(data_dir)
    if resolution is not None:
        mask = image.resample_img(mask, np.eye(4) * resolution)
    return mask


def _phase_builtins(nl, cfg):
    for sym_name in cfg.get("builtins", []):
        func = _BUILTIN_SYMBOLS.get(sym_name)
        if func is not None:
            nl.add_symbol(func, name=sym_name)


def _phase_base_symbols(nl, cfg, mask):
    """Register common neuroimaging symbols when ``use_base_symbols`` is set."""
    if cfg.get("use_base_symbols", False) and mask is not None:
        from neurolang.utils.engines.base import init_base_engine

        init_base_engine(nl, mask)


def _phase_python_init(nl, cfg, mask, data_dir):
    py_init = cfg.get("python_init")
    if not py_init:
        return
    import importlib

    mod = importlib.import_module(py_init)
    mod.init_engine(nl, mask, data_dir)


def _phase_downloads(cfg, data_dir, engine_name):
    for dl_entry in cfg.get("downloads", []):
        url = dl_entry["url"]
        dl_dest = str(data_dir / dl_entry.get("dest", engine_name))
        extract_members = dl_entry.get("extract", [])
        if isinstance(extract_members, bool):
            archive_name = Path(url).name
            _fetch_files(dl_dest, [(archive_name, url, {"uncompress": True})], verbose=1)
        elif extract_members:
            files = [(m, url, {"uncompress": True}) for m in extract_members]
            _fetch_files(dl_dest, files, verbose=1)
        else:
            file_name = Path(url).name
            _fetch_files(dl_dest, [(file_name, url, {})], verbose=1)


def _phase_templates(nl, cfg, data_dir, engine_name):
    for tpl_name, tpl_params in cfg.get("templates", {}).items():
        _load_template(nl, tpl_name, tpl_params, data_dir / engine_name)


def _phase_atlases(nl, cfg, data_dir, engine_name):
    for atl_name, atl_params in cfg.get("atlases", {}).items():
        _load_atlas(nl, atl_name, atl_params, data_dir / engine_name)


def _phase_datalog_init(nl, cfg):
    datalog_init = cfg.get("datalog_init")
    if datalog_init:
        nl.execute_datalog_program(datalog_init)


def _phase_relations(nl, cfg, engine_name, data_dir, rel_base_dir):
    relations = cfg.get("relations", {})
    if not relations:
        return
    import pandas as pd

    for rel_name, rel_cfg in relations.items():
        if isinstance(rel_cfg, str):
            rel_cfg = {"file": rel_cfg}
        rel_path = _resolve_relation_file(
            rel_cfg, engine_name, data_dir, base_dir=rel_base_dir
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


def _phase_probabilistic_choices(nl, cfg):
    for ch_name, ch_cfg in cfg.get("probabilistic_choice", {}).items():
        source_val = nl.symbols[ch_cfg["source"]]
        if hasattr(source_val, "value"):
            source_val = source_val.value
        nl.add_uniform_probabilistic_choice_over_set(
            source_val, name=ch_name
        )


def _phase_ontologies(nl, cfg, data_dir, engine_name):
    for onto_cfg in cfg.get("ontologies", []):
        url = onto_cfg["url"]
        onto_dir = str(data_dir / engine_name)
        onto_file = Path(url).name
        result = _fetch_files(onto_dir, [(onto_file, url, {})], verbose=1)
        nl.load_ontology(result[0])


def build_engine(
    name: str,
    data_dir: Path,
    resolution: Optional[float] = None,
    yaml_path: Optional[Path] = None,
) -> "NeurolangPDL":
    """
    Create and populate a :class:`NeurolangPDL` engine from its
    declarative YAML config.

    Parameters
    ----------
    name :
        Registered engine name (e.g. ``"neurosynth"``, ``"destrieux"``).
    data_dir :
        Directory under which downloaded data is cached.
    resolution :
        If set, resample the MNI mask to the given isotropic resolution (mm).
    yaml_path :
        Path to an engine YAML file.  Defaults to the built-in
        ``engines/engines.yaml``.

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

    cfg = get_engine_config(name, yaml_path)
    data_dir = Path(data_dir)
    rel_base_dir = (Path(yaml_path).parent if yaml_path else None)
    mask = _resolve_mask(cfg, data_dir, resolution)

    nl = NeurolangPDL()

    _phase_builtins(nl, cfg)
    _phase_base_symbols(nl, cfg, mask)
    _phase_python_init(nl, cfg, mask, data_dir)
    _phase_downloads(cfg, data_dir, name)
    _phase_templates(nl, cfg, data_dir, name)
    _phase_atlases(nl, cfg, data_dir, name)
    _phase_datalog_init(nl, cfg)
    _phase_relations(nl, cfg, name, data_dir, rel_base_dir)
    _phase_probabilistic_choices(nl, cfg)
    _phase_ontologies(nl, cfg, data_dir, name)

    return nl
