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
* ``extends`` — one engine name, or a list of them, whose configuration
  is inherited (see :func:`resolve_engine_config`); engines marked
  ``abstract: true`` only serve as bases and cannot be built or listed;
* ``text_embeddings`` — semantic search over document texts (e.g. PubMed
  abstracts) with SPECTER2 or, through the Albert API, Qwen3-VL-Embedding,
  registered as a function
  ``f(query, doc) -> similarity``; see
  :mod:`neurolang.utils.text_embeddings`::

      text_embeddings:
        study_abstracts:
          ids_from: study
          source: pubmed          # or {file: ...} or {predicate: ...}
          function: query_similarity
          encoder: specter2       # or albert-qwen (Albert API)

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

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

import nibabel as nib

import numpy as np
import yaml

from neurolang.utils.nilearn_compat import fetch_files as _fetch_files
from neurolang.utils.nilearn_compat import (
    nilearn_datasets as _nilearn_datasets,
)
from neurolang.utils.nilearn_compat import nilearn_image as _nilearn_image

if TYPE_CHECKING:
    import pandas as pd

    from neurolang.frontend import NeurolangPDL

_ENGINES_DIR = Path(__file__).parent / "engines"
_ENGINES_YAML = _ENGINES_DIR / "engines.yaml"


def _builtin_startswith(prefix: str, s: str) -> bool:
    """Return True if *s* starts with *prefix*."""
    return s.startswith(prefix)


def _builtin_lcase(s) -> str:
    """Return *s* in lower case, as SPARQL's ``LCASE``."""
    return str(s).lower()


def _builtin_ucase(s) -> str:
    """Return *s* in upper case, as SPARQL's ``UCASE``."""
    return str(s).upper()


def _builtin_strlen(s) -> int:
    """Return the number of characters of *s*, as SPARQL's ``STRLEN``."""
    return len(str(s))


def _builtin_strstarts(s, prefix) -> bool:
    """Return True if *s* starts with *prefix*, as SPARQL's ``STRSTARTS``."""
    return str(s).startswith(str(prefix))


def _builtin_strends(s, suffix) -> bool:
    """Return True if *s* ends with *suffix*, as SPARQL's ``STRENDS``."""
    return str(s).endswith(str(suffix))


def _builtin_str_contains(s, part) -> bool:
    """Return True if *part* is in *s*, as SPARQL's ``CONTAINS``."""
    return str(part) in str(s)


def _builtin_regex(s, pattern) -> bool:
    """
    Return True if *pattern* matches in *s*, as SPARQL's ``REGEX``.

    Flags are written in the pattern, e.g. ``(?i)`` for SPARQL's ``"i"``.
    """
    return re.search(str(pattern), str(s)) is not None


def _builtin_round(x) -> float:
    """Return *x* rounded to an integer, halves away from 0, as ``ROUND``."""
    x = float(x)
    return float(np.sign(x) * np.floor(abs(x) + 0.5))


def _builtin_int(x) -> int:
    """Return *x* as an integer, as SPARQL's ``xsd:integer`` cast."""
    return int(float(x))


def _builtin_float(x) -> float:
    """Return *x* as a float, as SPARQL's ``xsd:double`` cast."""
    return float(x)


def _builtin_str(x) -> str:
    """Return *x* as a string, as SPARQL's ``STR``."""
    return str(x)


_BUILTIN_SYMBOLS: Dict[str, object] = {
    "exp": np.exp,
    "log": np.log,
    "startswith": _builtin_startswith,
    # String and numeric functions of SPARQL, e.g. for RDF engines.
    "lcase": _builtin_lcase,
    "ucase": _builtin_ucase,
    "strlen": _builtin_strlen,
    "strstarts": _builtin_strstarts,
    "strends": _builtin_strends,
    "str_contains": _builtin_str_contains,
    "regex": _builtin_regex,
    "round": _builtin_round,
    "abs": abs,
    "floor": lambda x: float(np.floor(float(x))),
    "ceil": lambda x: float(np.ceil(float(x))),
    "int": _builtin_int,
    "float": _builtin_float,
    "str": _builtin_str,
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


# Top-level keys holding lists that are concatenated (parent first,
# duplicates dropped) when inherited; every other list is replaced.
_MERGED_LIST_KEYS = frozenset(
    {"builtins", "downloads", "ontologies", "dimension_type_predicates"}
)
# Keys describing the engine itself, never inherited.
_NON_INHERITED_KEYS = frozenset({"extends", "abstract"})


def _merge_mappings(parent: Dict[str, Any], child: Dict[str, Any]):
    """Deep-merge two mappings; ``None`` in *child* deletes the entry."""
    merged = dict(parent)
    for key, value in child.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mappings(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_engine_configs(
    parent: Dict[str, Any], child: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge *child* over *parent* engine configurations.

    * ``datalog_init`` texts are concatenated, parent rules first;
    * the lists in :data:`_MERGED_LIST_KEYS` are concatenated without
      duplicates;
    * mappings (``atlases``, ``predicates``, ...) are merged recursively,
      and an entry set to ``null`` in the child removes the inherited one;
    * any other value in the child replaces the inherited one.
    """
    merged = {
        k: v for k, v in parent.items() if k not in _NON_INHERITED_KEYS
    }
    for key, value in child.items():
        inherited = merged.get(key)
        if key == "datalog_init" and inherited and value:
            merged[key] = inherited.rstrip("\n") + "\n" + value
        elif key in _MERGED_LIST_KEYS and inherited and value:
            merged[key] = list(inherited) + [
                v for v in value if v not in inherited
            ]
        elif isinstance(value, dict) and isinstance(inherited, dict):
            merged[key] = _merge_mappings(inherited, value)
        elif isinstance(value, dict):
            merged[key] = _merge_mappings({}, value)
        else:
            merged[key] = value
    return merged


def resolve_engine_config(
    engines: Dict[str, Dict[str, Any]],
    name: str,
    _stack: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """
    Return the configuration of *name* with all ``extends`` resolved.

    Parameters
    ----------
    engines :
        The raw ``engines`` mapping of an engine YAML file.
    name :
        Engine (or abstract base) to resolve.

    Raises
    ------
    ValueError
        If *name* or one of its bases is unknown, or the ``extends``
        chain is cyclic.

    """
    if name in _stack:
        chain = " -> ".join(_stack + (name,))
        raise ValueError(f"Cyclic engine inheritance: {chain}")
    if name not in engines:
        available = ", ".join(engines.keys())
        raise ValueError(
            f"Unknown engine: {name!r}. Available engines: {available}"
        )
    raw = engines[name] or {}
    bases = raw.get("extends", [])
    if isinstance(bases, str):
        bases = [bases]
    resolved: Dict[str, Any] = {}
    for base in bases:
        resolved = _merge_engine_configs(
            resolved, resolve_engine_config(engines, base, _stack + (name,))
        )
    merged = _merge_engine_configs(resolved, raw)
    return {k: v for k, v in merged.items() if k not in _NON_INHERITED_KEYS}


def _is_abstract(engines: Dict[str, Dict[str, Any]], name: str) -> bool:
    return bool((engines[name] or {}).get("abstract", False))


def list_engine_names(yaml_path: Optional[Path] = None) -> List[str]:
    """
    Return the names of all registered, non-abstract engines.

    Parameters
    ----------
    yaml_path :
        Path to an engine YAML file.  Defaults to the built-in
        ``engines/engines.yaml``.

    """
    engines = _load_yaml_config(yaml_path).get("engines", {})
    return sorted(n for n in engines if not _is_abstract(engines, n))


def get_engine_config(
    name: str, yaml_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Return the YAML configuration dict for a single engine.

    Inherited (``extends``) configuration is already merged in.

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
        If *name* is not a registered, non-abstract engine.

    """
    engines = _load_yaml_config(yaml_path).get("engines", {})
    if name in engines and _is_abstract(engines, name):
        raise ValueError(
            f"Engine {name!r} is abstract and can only be extended"
        )
    return resolve_engine_config(engines, name)


def show_engines(yaml_path: Optional[Path] = None) -> None:
    """
    Print a formatted description of all available engines.

    Parameters
    ----------
    yaml_path :
        Path to an engine YAML file.  Defaults to the built-in
        ``engines/engines.yaml``.

    """
    engines = _load_yaml_config(yaml_path).get("engines", {})
    print("Available engines:\n")
    for name in list_engine_names(yaml_path):
        cfg = resolve_engine_config(engines, name)
        desc = cfg.get("description", "").replace("\n", " ").strip()
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
    predicates = _declared_predicates(cfg)

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


class Argument(NamedTuple):
    """
    An argument of a predicate: an entity of a *sort* over *columns*.

    A voxel is one argument of sort ``voxel`` over the columns ``i, j,
    k``; *type* names the type of the values of its columns.
    """

    sort: str
    columns: Tuple[str, ...]
    type: str = ""


class Sort(NamedTuple):
    """
    A sort of entities an engine declares once, in its ``sorts`` section.

    Its entities span *columns*, whose values are of the *type*; a sort of
    N > 1 columns reads ``in ND`` in SQUALL (``every Voxel in 3D``).

    The *space* of a sort of points tells what its columns are: ``voxel``
    for the indices ``(i, j, k)`` of a grid, ``mm`` for coordinates
    ``(x, y, z)`` in millimetres in the space of the engine's template,
    ``""`` otherwise. The *grid* of a sort of voxels is the grid it
    indexes: that of the engine's template (see :func:`build_engine`)
    when ``""`` or ``template``, else that of the atlas of this name.
    The *atlas* of a sort of region labels is the atlas they name the
    regions of, one the engine loads: ``.view`` paints the regions of a
    result on its grid (see :func:`atlas_label_values`).
    """

    name: str
    columns: Tuple[str, ...]
    type: str = ""
    description: str = ""
    space: str = ""
    grid: str = ""
    atlas: str = ""


SORT_SPACES = ("voxel", "mm")

# The grid of the engine's template, which the voxels of a sort index
# unless it names another grid.
TEMPLATE_GRID = "template"


def engine_grid_names(cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """
    Return the names of the grids of the engine *cfg*.

    They are the template's and those of the atlases it loads.
    """
    return (TEMPLATE_GRID, *(cfg.get("atlases") or {}))


def _sort_grid(name: str, info: Dict[str, Any], grids) -> str:
    """
    Return the grid the sort *name* declared by *info* indexes, or "".

    Only a sort of ``space: voxel`` names a grid, one of *grids*.
    """
    grid = str(info.get("grid", ""))
    if grid and info.get("space") != "voxel":
        raise ValueError(
            f"sort {name!r} names the grid {grid!r}; only a sort of "
            "space: voxel indexes a grid"
        )
    if grid and grid not in grids:
        raise ValueError(
            f"sort {name!r} names the grid {grid!r}; the grids of the "
            f"engine are {', '.join(grids)}"
        )
    return grid


def _sort_atlas(name: str, info: Dict[str, Any], atlases) -> str:
    """
    Return the atlas whose regions the sort *name* declared by *info*
    labels, or "".

    Only a sort of one column and no space labels the regions of an
    atlas, one of *atlases*.
    """
    atlas = str(info.get("atlas", ""))
    if atlas and (info.get("space") or len(info.get("columns", ())) != 1):
        raise ValueError(
            f"sort {name!r} names the atlas {atlas!r}; only a sort of one "
            "column and no space labels the regions of an atlas"
        )
    if atlas and atlas not in atlases:
        raise ValueError(
            f"sort {name!r} names the atlas {atlas!r}; the atlases of the "
            f"engine are {', '.join(atlases) or 'none'}"
        )
    return atlas


def _declared_sort(
    name: str, info: Dict[str, Any], grids=(TEMPLATE_GRID,)
) -> Sort:
    """
    Return the sort *name* declared by *info*, checking its space.

    Its grid, if any, is one of *grids*, and its atlas one of them but
    the template's.
    """
    space = str(info.get("space", ""))
    if space and space not in SORT_SPACES:
        raise ValueError(
            f"sort {name!r} declares the space {space!r}; a space is "
            f"one of {', '.join(SORT_SPACES)}"
        )
    return Sort(
        name, tuple(info["columns"]), str(info.get("type", "")),
        str(info.get("description", "")).strip(), space,
        _sort_grid(name, info, grids),
        _sort_atlas(
            name, info, tuple(g for g in grids if g != TEMPLATE_GRID)
        ),
    )


def _declared_sorts(cfg: Dict[str, Any]) -> Dict[str, Sort]:
    """Return the sorts the engine *cfg* declares, by name."""
    grids = engine_grid_names(cfg)
    return {
        name: _declared_sort(name, info, grids)
        for name, info in (cfg.get("sorts") or {}).items()
    }


def get_sorts(name: str, yaml_path: Optional[Path] = None) -> Dict[str, Sort]:
    """Return the sorts an engine declares, by name."""
    return _declared_sorts(get_engine_config(name, yaml_path))


def _resolved_argument(
    name: str, argument: Any, sorts: Dict[str, Sort]
) -> Dict[str, Any]:
    """
    Return the *argument* of the predicate *name* with its columns and type.

    A sort's name alone, or ``{sort: …}`` without columns, references a
    declared sort, whose columns and type it takes; declared columns or
    type override the sort's.
    """
    if isinstance(argument, str):
        argument = {"sort": argument}
    sort = sorts.get(argument["sort"])
    if sort is None and "columns" not in argument:
        raise ValueError(
            f"predicate {name!r} references the sort {argument['sort']!r}, "
            f"which the engine does not declare"
        )
    if sort is None:
        return dict(argument)
    return {"columns": list(sort.columns), "type": sort.type, **argument}


def _declared_predicate(
    name: str, info: Dict[str, Any], sorts: Dict[str, Sort]
) -> Dict[str, Any]:
    """
    Return the metadata *info* of *name* completed from its arguments.

    When *info* declares ``arguments``, each a sort of *sorts* or a
    ``{sort, columns, type}``, its ``arguments`` are completed with
    their columns and type, its ``columns`` are theirs in order, its
    ``arity`` their number of columns and its ``entities`` their number;
    declared ``columns`` or ``arity`` must agree with them.
    """
    info = dict(info or {})
    if not info.get("arguments"):
        return info
    arguments = [
        _resolved_argument(name, argument, sorts)
        for argument in info["arguments"]
    ]
    columns = [
        column for argument in arguments for column in argument["columns"]
    ]
    if list(info.get("columns", columns)) != columns:
        raise ValueError(
            f"predicate {name!r} declares the columns {info['columns']} "
            f"but its arguments span {columns}"
        )
    if info.get("arity", len(columns)) != len(columns):
        raise ValueError(
            f"predicate {name!r} declares the arity {info['arity']} "
            f"but its arguments span {len(columns)} columns"
        )
    info.update(
        arguments=arguments, columns=columns, arity=len(columns),
        entities=len(arguments),
    )
    return info


def _declared_predicates(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return the ``predicates`` of the engine *cfg*, completed."""
    sorts = _declared_sorts(cfg)
    return {
        name: _declared_predicate(name, info, sorts)
        for name, info in (cfg.get("predicates") or {}).items()
    }


def _check_widths(pname: str, arguments, widths: Dict[str, int]) -> None:
    """
    Check each of the *arguments* of *pname* spans its sort's width.

    *widths* holds the number of columns of each sort met so far.
    """
    for argument in arguments:
        width = widths.setdefault(argument.sort, len(argument.columns))
        if width != len(argument.columns):
            raise ValueError(
                f"sort {argument.sort!r} spans {width} columns elsewhere "
                f"but {len(argument.columns)} in {pname!r}"
            )


def get_signatures(
    name: str, yaml_path: Optional[Path] = None
) -> Dict[str, Tuple[Argument, ...]]:
    """
    Return the arguments of the predicates of an engine, by predicate.

    Only the predicates declaring their ``arguments`` are returned. A
    sort spans the same number of columns wherever it appears, and as
    many as the engine declares for it, or a :class:`ValueError` is
    raised.
    """
    signatures = {}
    widths = {
        sort.name: len(sort.columns)
        for sort in get_sorts(name, yaml_path).values()
    }
    for pname, info in get_predicates(name, yaml_path).items():
        arguments = tuple(
            Argument(
                arg["sort"], tuple(arg["columns"]), str(arg.get("type", ""))
            )
            for arg in info.get("arguments") or ()
        )
        _check_widths(pname, arguments, widths)
        if arguments:
            signatures[pname] = arguments
    return signatures


def get_hidden_predicates(
    name: str, yaml_path: Optional[Path] = None
) -> frozenset:
    """Return the predicates of an engine its SQUALL completion withholds."""
    cfg = get_engine_config(name, yaml_path)
    return frozenset(cfg.get("hidden_predicates") or ())


def get_probabilistic_predicates(
    name: str, yaml_path: Optional[Path] = None
) -> frozenset:
    """Return the predicates of an engine declared with a probability."""
    return frozenset(
        pname for pname, info in get_predicates(name, yaml_path).items()
        if info.get("probability")
    )


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_NILEARN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "brain_mask": {
        "func": "load_mni152_brain_mask",
        "kind": "mask",
    },
    "gm_mask": {
        "func": "load_mni152_gm_mask",
        "kind": "mask",
    },
    "wm_mask": {
        "func": "load_mni152_wm_mask",
        "kind": "mask",
    },
    "template": {
        "func": "load_mni152_template",
        "kind": "anatomical",
    },
    "gm_template": {
        "func": "load_mni152_gm_template",
        "kind": "anatomical",
    },
    "wm_template": {
        "func": "load_mni152_wm_template",
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

    img = getattr(_nilearn_datasets(), info["func"])(**kwargs)

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
    return _nilearn_datasets().fetch_atlas_destrieux_2009(**kwargs)


def _fetch_atlas_schaefer(**kwargs):
    return _nilearn_datasets().fetch_atlas_schaefer_2018(**kwargs)


def _fetch_atlas_difumo(**kwargs):
    return _nilearn_datasets().fetch_atlas_difumo(**kwargs)


TIAN_URL = (
    "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
    "Group-Parcellation/3T/Subcortex-Only/"
)


def _fetch_atlas_tian(scale: int = 2, data_dir: str = "", **kwargs):
    """
    Fetch the Melbourne subcortex atlas (Tian et al., 2020, Nat Neurosci,
    doi:10.1038/s41593-020-00711-6), 3T, at *scale* 1 to 4 (16, 32, 50 or
    54 regions), on the 2 mm grid of the MNI152 2009c asymmetric template,
    from the authors' repository. Its label file names the region of value
    *n* on its line *n*; region names end with the hemisphere (``-lh``,
    ``-rh``).
    """
    base = f"Tian_Subcortex_S{int(scale)}_3T"
    image, labels = _fetch_files(
        str(Path(data_dir) / "tian"),
        [
            (f"{base}_2009cAsym.nii.gz",
             f"{TIAN_URL}{base}_2009cAsym.nii.gz", {}),
            (f"{base}_label.txt", f"{TIAN_URL}{base}_label.txt", {}),
        ],
        verbose=0,
    )
    names = Path(labels).read_text().split()
    return {"maps": image, "labels": ["Background", *names]}


_ATLAS_REGISTRY = {
    "destrieux": {
        "fetch": _fetch_atlas_destrieux,
        "probabilistic": False,
    },
    "schaefer": {
        "fetch": _fetch_atlas_schaefer,
        "probabilistic": False,
    },
    "tian": {
        "fetch": _fetch_atlas_tian,
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


def _register_grid(nl, name: str, img) -> None:
    """
    Register the grid of the image *img* as the grid *name* of *nl*.

    The voxel indices of a sort naming the grid (``grid: name``) index
    it: ``.save`` writes them with its shape and affine.
    """
    if getattr(nl, "grids", None) is None:
        nl.grids = {}
    nl.grids[name] = img


def _register_atlas_labels(nl, name: str, labels: dict) -> None:
    """
    Register the values of the image of the atlas *name* labelling each
    of its regions, *labels* mapping each value to the region's name.
    """
    if getattr(nl, "atlas_labels", None) is None:
        nl.atlas_labels = {}
    values = {}
    for value, label in labels.items():
        values.setdefault(label, []).append(int(value))
    nl.atlas_labels[name] = values


def atlas_label_values(nl, name: str) -> Dict[str, List[int]]:
    """
    Return the values labelling each region of the atlas *name* of *nl*
    in the image of its grid (``nl.grids[name]``), by region name.

    The names are those of the atlas's relation, ``{}`` for an atlas the
    engine did not load.
    """
    return (getattr(nl, "atlas_labels", None) or {}).get(name, {})


def atlas_activation(
    img, label_names: dict, peaks_mm: np.ndarray, studies: np.ndarray,
    radius_mm: float = 3.0, chunk: int = 4096,
) -> List[Tuple[str, Any]]:
    """
    Return the ``(label, study)`` pairs of the studies activating the
    regions of the atlas image *img*: one of their peaks is within
    *radius_mm* of the centre of a voxel of the region.

    *peaks_mm* are the peaks' MNI coordinates in millimetres, of the
    *studies*, and *label_names* names the values of the image. Distances
    are measured in millimetres through the affine of the image, so the
    atlas's grid need not be that of the peaks.
    """
    data = np.asanyarray(img.dataobj).astype(np.int64)
    shape = np.array(data.shape[:3])
    affine = img.affine
    inverse = np.linalg.inv(affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    reach = np.ceil(radius_mm / spacing).astype(int) + 1
    offsets = np.stack(np.meshgrid(
        *(np.arange(-r, r + 1) for r in reach), indexing="ij"
    ), -1).reshape(-1, 3)
    pairs = set()
    for start in range(0, len(peaks_mm), chunk):
        peaks = np.asarray(peaks_mm[start:start + chunk], dtype=float)
        owners = np.asarray(studies[start:start + chunk])
        centres = np.round(nib.affines.apply_affine(inverse, peaks))
        candidates = centres.astype(int)[:, None, :] + offsets[None]
        inside = np.all((candidates >= 0) & (candidates < shape), axis=-1)
        distances = np.linalg.norm(
            nib.affines.apply_affine(affine, candidates) - peaks[:, None],
            axis=-1,
        )
        near = inside & (distances <= radius_mm)
        values = data[tuple(candidates[near].T)]
        owner = np.broadcast_to(owners[:, None], near.shape)[near]
        labelled = values > 0
        pairs.update(zip(
            values[labelled].tolist(), owner[labelled].tolist()
        ))
    return sorted(
        (label_names[value], study) for value, study in pairs
        if value in label_names
    )


def _relation_frame(nl, name: str):
    """Return the rows of the stored relation *name* of *nl*, as a frame."""
    import pandas as pd

    relation = nl.symbol_table[name].value
    relation = relation.unwrap() if hasattr(relation, "unwrap") else relation
    if hasattr(relation, "as_pandas_dataframe"):
        return relation.as_pandas_dataframe()
    return pd.DataFrame(list(relation))


def _register_atlas_activation(
    nl, labels: dict, img, settings: dict, pred_name: str
) -> None:
    """
    Register ``{predicate_name}(label, study)``, the studies activating
    each region of the atlas (see :func:`atlas_activation`), from the
    peaks in millimetres of the relation ``peaks`` (``x, y, z, study``),
    ``peak_mni`` by default, within ``radius_mm`` (3 by default).
    """
    peaks = _relation_frame(nl, settings.get("peaks", "peak_mni"))
    pairs = atlas_activation(
        img, labels, peaks.iloc[:, :3].to_numpy(dtype=float),
        peaks.iloc[:, 3].to_numpy(), float(settings.get("radius_mm", 3)),
    )
    name = settings.get("predicate_name", f"{pred_name}_activated")
    nl.add_tuple_set(pairs, name=name)
    nl.symbol_table[name].value.might_have_duplicates = False


def _load_deterministic_atlas(
    nl, atlas_name: str, params: dict, data_dir: Path
) -> None:
    """
    Load a deterministic atlas (e.g. Destrieux, Schaefer) as engine
    predicates: ``{predicate_name}(label, region)``, and with ``activated``
    the studies activating each region (see
    :func:`_register_atlas_activation`).
    """
    info = _ATLAS_REGISTRY[atlas_name]
    fetch_kw = {
        k: v for k, v in params.items()
        if k not in ("predicate_name", "activated")
    }
    fetch_kw["data_dir"] = str(data_dir)
    atl_data = info["fetch"](**fetch_kw)

    from neurolang.regions import ExplicitVBR

    img = nib.load(atl_data["maps"])
    _register_grid(nl, atlas_name, img)
    pred_name = params.get("predicate_name", atlas_name)
    labels = [(k, v) for k, v in _parse_labels(atl_data["labels"]) if k != 0]
    regions = ExplicitVBR.from_spatial_image_labels(
        img, [k for k, _ in labels]
    )
    # A label of the look-up table may have no voxel in the image (the
    # medial walls of Destrieux): it is no region.
    labels = [(k, v) for k, v in labels if hasattr(regions[k], "voxels")]
    rows = [(_label_name(v), regions[k]) for k, v in labels]
    names = {k: _label_name(v) for k, v in labels}

    nl.add_tuple_set(rows, name=pred_name)
    _register_atlas_labels(nl, atlas_name, names)
    if params.get("activated"):
        _register_atlas_activation(
            nl, names, img, params["activated"], pred_name,
        )


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

    import pandas as pd

    from neurolang.regions import ExplicitVBR

    img = nib.load(atl_data["maps"])
    _register_grid(nl, atlas_name, img)
    n_components = img.shape[-1]
    labels = list(atl_data["labels"])
    threshold = float(params.get("threshold", 0.5))
    prob_threshold = float(params.get("prob_threshold", 0.01))
    pred_name = params.get("predicate_name", atlas_name)
    prob_pred_name = params.get(
        "prob_predicate_name", f"{pred_name}_prob"
    )

    # Read one component at a time: loading the whole 4D image with
    # ``get_fdata`` takes several GB for large atlases such as DiFuMo.
    region_rows = []
    prob_frames = []
    for i in range(n_components):
        prob_map = np.asarray(img.dataobj[..., i])
        label = _label_name(labels[i] if i < len(labels) else f"component_{i}")

        # ── Binary region predicate (thresholded) ──
        mask = prob_map > threshold
        if mask.any():
            voxels = np.argwhere(mask)
            region = ExplicitVBR(voxels, img.affine, image_dim=img.shape[:3])
            region_rows.append((label, region))

        # ── Probability predicate (component, i, j, k, prob) ──
        mask = prob_map > prob_threshold
        if mask.any():
            indices = np.argwhere(mask)
            prob_frames.append(
                pd.DataFrame(
                    {
                        "component": label,
                        "i": indices[:, 0],
                        "j": indices[:, 1],
                        "k": indices[:, 2],
                        "prob": prob_map[mask].astype(float),
                    }
                )
            )

    nl.add_tuple_set(region_rows, name=pred_name)

    if prob_frames:
        prob_rows = pd.concat(prob_frames, ignore_index=True)
    else:
        prob_rows = []
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
        _nilearn_datasets().fetch_icbm152_2009(
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
        # A 3x3 target affine keeps the original origin; scaling a 4x4
        # identity would also scale its homogeneous coordinate.
        mask = _nilearn_image().resample_img(mask, np.eye(3) * resolution)
    return mask


def _phase_builtins(nl, cfg):
    for sym_name in cfg.get("builtins", []):
        func = _BUILTIN_SYMBOLS.get(sym_name)
        if func is not None:
            nl.add_symbol(func, name=sym_name)


def _phase_base_symbols(nl, cfg, mask):
    """Register common neuroimaging symbols when ``use_base_symbols`` is set."""
    if cfg.get("use_base_symbols", False) and mask is not None:
        from neurolang_apps.utils.engines.base import init_base_engine

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


RELATION_FILE_SEPARATORS = {".csv": ",", ".tsv": "\t"}


class UnsupportedRelationFile(ValueError):
    """Raised for a relation file whose extension is not supported."""


def relation_file_separator(path) -> str:
    """
    Return the column separator of the relation file *path*.

    It follows from the extension: ``.csv`` or ``.tsv``, optionally
    followed by ``.gz``. Other extensions raise
    :class:`UnsupportedRelationFile`.
    """
    suffixes = [s.lower() for s in Path(path).suffixes]
    if suffixes[-1:] == [".gz"]:
        suffixes = suffixes[:-1]
    sep = RELATION_FILE_SEPARATORS.get(suffixes[-1] if suffixes else "")
    if sep is None:
        raise UnsupportedRelationFile(
            f"Unsupported relation file format: {Path(path).name}. "
            "Supported: .csv, .tsv, .csv.gz, .tsv.gz"
        )
    return sep


def read_relation_file(path) -> pd.DataFrame:
    """
    Return the rows of the CSV or TSV file *path* as a DataFrame.

    The first line of the file is the header. ``~`` in *path* is
    expanded and the separator follows from the extension (see
    :func:`relation_file_separator`); gzipped files are decompressed.
    """
    import pandas as pd

    path = Path(path).expanduser()
    return pd.read_csv(path, sep=relation_file_separator(path))


def _phase_relations(nl, cfg, engine_name, data_dir, rel_base_dir):
    relations = cfg.get("relations", {})
    for rel_name, rel_cfg in relations.items():
        if isinstance(rel_cfg, str):
            rel_cfg = {"file": rel_cfg}
        rel_path = _resolve_relation_file(
            rel_cfg, engine_name, data_dir, base_dir=rel_base_dir
        )
        try:
            df = read_relation_file(rel_path)
        except UnsupportedRelationFile as exc:
            raise UnsupportedRelationFile(
                f"Relation {rel_name!r}: {exc}"
            ) from exc
        nl.add_tuple_set(df, name=rel_name)


def _phase_probabilistic_choices(nl, cfg):
    for ch_name, ch_cfg in cfg.get("probabilistic_choice", {}).items():
        source_val = nl.symbols[ch_cfg["source"]]
        if hasattr(source_val, "value"):
            source_val = source_val.value
        nl.add_uniform_probabilistic_choice_over_set(
            source_val, name=ch_name
        )


_ONTOLOGY_OPTIONS = ("format", "individuals", "names")


def _ontology_file(source: str, engine_name, data_dir, base_dir) -> str:
    """
    Return the local path of the ontology file *source*.

    A URL is downloaded (and cached) to ``{data_dir}/{engine_name}/``; a
    local path is relative to *base_dir* (default: the built-in
    ``engines/`` directory).
    """
    if source.startswith(("http://", "https://")):
        onto_dir = str(data_dir / engine_name)
        result = _fetch_files(
            onto_dir, [(Path(source).name, source, {})], verbose=1
        )
        return result[0]
    return str((base_dir or _ENGINES_DIR) / Path(source).expanduser())


def _phase_ontologies(nl, cfg, data_dir, engine_name, base_dir=None):
    """
    Load the engine's ``ontologies``.

    Each entry gives a file with ``url`` (a URL or a local path), or
    several, loaded together, with ``urls``. ``format``, ``individuals``
    and ``names`` are passed to
    :meth:`~neurolang.frontend.NeurolangPDL.load_ontology`.
    """
    for onto_cfg in cfg.get("ontologies", []):
        sources = onto_cfg.get("urls") or [onto_cfg["url"]]
        paths = [
            _ontology_file(source, engine_name, data_dir, base_dir)
            for source in sources
        ]
        options = {
            key: onto_cfg[key] for key in _ONTOLOGY_OPTIONS if key in onto_cfg
        }
        nl.load_ontology(paths if len(paths) > 1 else paths[0], **options)


def _phase_text_embeddings(nl, cfg, data_dir, engine_name, rel_base_dir):
    """
    Register the document indexes declared in ``text_embeddings``.

    Only the ids are read here; texts are fetched and embedded on first use.
    """
    entries = cfg.get("text_embeddings", {})
    if not entries:
        return
    from neurolang_apps.utils import text_embeddings

    emb_dir = data_dir / engine_name / "embeddings"
    nl.text_embedding_indexes = getattr(nl, "text_embedding_indexes", {})
    for emb_name, emb_cfg in entries.items():
        encoder = text_embeddings.make_encoder(emb_cfg, emb_dir / "models")
        # One cache per encoder, so that switching encoders keeps both.
        kind = text_embeddings.encoder_kind(emb_cfg)
        suffix = "" if kind == "specter2" else f".{kind}"
        index = text_embeddings.EmbeddingIndex(
            ids=[row[0] for row in nl.symbols[emb_cfg["ids_from"]].value],
            load_texts=_text_loader(
                nl, emb_cfg, emb_dir / f"{emb_name}_pubmed.json",
                engine_name, data_dir, rel_base_dir,
            ),
            encoder=encoder,
            cache_path=emb_dir / f"{emb_name}{suffix}.npz",
        )
        nl.text_embedding_indexes[emb_name] = index
        nl.add_symbol(
            text_embeddings.make_similarity_function(index),
            name=emb_cfg["function"],
        )


def _predicate_texts(nl, source):
    """Read ``{id: (title, text)}`` from engine predicates of the form
    ``p(id, text)``: ``predicate`` gives the texts, the optional
    ``title_predicate`` the titles."""
    def column(name):
        return {str(row[0]): row[1] for row in nl.symbols[name].value}

    texts = column(source["predicate"])
    titles = column(source["title_predicate"]) if (
        "title_predicate" in source
    ) else {}
    return {
        key: (titles.get(key, ""), text) for key, text in texts.items()
    }


def _text_loader(
    nl, emb_cfg, pubmed_cache, engine_name, data_dir, base_dir
):
    """Return ``load_texts(keys)`` for a ``text_embeddings`` source."""
    from neurolang_apps.utils import text_embeddings

    source = emb_cfg.get("source", "pubmed")
    if source == "pubmed":
        def load_texts(keys):
            return text_embeddings.fetch_pubmed_abstracts(
                keys, cache_path=pubmed_cache, email=emb_cfg.get("email")
            )
    elif isinstance(source, dict) and "predicate" in source:
        def load_texts(keys):
            return _predicate_texts(nl, source)
    elif isinstance(source, dict) and "file" in source:
        def load_texts(keys):
            path = _resolve_relation_file(
                source, engine_name, data_dir, base_dir=base_dir
            )
            return text_embeddings.load_text_table(
                path,
                source.get("id_column", "id"),
                source.get("text_columns", ["title", "abstract"]),
            )
    else:
        raise ValueError(
            f"Unknown text_embeddings source {source!r}: use 'pubmed' "
            "or a mapping with a 'file' or a 'predicate' key."
        )
    return load_texts


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
        A fully initialised engine ready for query execution. Its
        ``template`` attribute is the MNI image whose grid the voxel
        indices of the engine refer to (the sorts of ``space: voxel``),
        resampled to *resolution* if given, or None if the engine does
        not require the MNI mask. Its ``grids`` attribute holds the
        images of the grids voxel indices may index, by name: the
        template, as ``template``, and each atlas, under its name (the
        sorts naming a ``grid``).

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

    # Wire dimension_type_predicates from YAML
    dimension_predicates = cfg.get("dimension_type_predicates")
    if dimension_predicates is not None:
        from neurolang.frontend.datalog.squall import (
            make_dimension_type_stripper,
        )
        Mixin = make_dimension_type_stripper(dimension_predicates)

        from neurolang.frontend.probabilistic_frontend import (
            RegionFrontendCPLogicSolver as _BaseSolver,
        )

        class _CustomSolver(Mixin, _BaseSolver):
            pass

        _CustomSolver.__qualname__ = _BaseSolver.__qualname__
        _CustomSolver.__module__ = _BaseSolver.__module__
        nl = NeurolangPDL(solver_class=_CustomSolver)
    else:
        nl = NeurolangPDL()

    nl.template = mask
    nl.grids = {} if mask is None else {TEMPLATE_GRID: mask}
    nl.atlas_labels = {}
    _phase_builtins(nl, cfg)
    _phase_base_symbols(nl, cfg, mask)
    _phase_python_init(nl, cfg, mask, data_dir)
    _phase_downloads(cfg, data_dir, name)
    _phase_templates(nl, cfg, data_dir, name)
    _phase_atlases(nl, cfg, data_dir, name)
    # Before the rules, which may use the predicates of the ontologies.
    _phase_ontologies(nl, cfg, data_dir, name, rel_base_dir)
    _phase_datalog_init(nl, cfg)
    _phase_relations(nl, cfg, name, data_dir, rel_base_dir)
    _phase_text_embeddings(nl, cfg, data_dir, name, rel_base_dir)
    _phase_probabilistic_choices(nl, cfg)

    return nl
