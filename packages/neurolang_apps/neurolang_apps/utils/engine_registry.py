"""
Declarative engine registry for the neurolang-query CLI.

Engines are declared in :file:`engines/engines.yaml`, into whose entries
the files :file:`engines/<engine>/*.yaml` (their example queries and the
query assistant's guidance) are merged with OmegaConf.  Each engine may
define:

* **Python initialisation scripts** (``python_init``): a module name, or
  a list of them, whose ``init_engine(nl, mask, data_dir, **options)``
  run in order, each given all the keyword arguments of
  ``python_init_options`` (an ``init_engine`` takes ``**options`` for
  those of the others);
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
* ``resamplings`` — replicates of a probabilistic choice for confidence
  intervals, ``name: {choice: selected_study, n: 1000, scheme: bayesian,
  seed: 0}``, used as ``RESAMPLE[name](pred(x)) = b`` (see
  :meth:`~neurolang.frontend.NeurolangPDL.add_resampling`).
* ``atlases`` — brain atlases downloaded via nilearn (e.g. ``destrieux``,
  ``schaefer``, ``difumo``) and registered as predicates with
  :class:`~neurolang.regions.ExplicitVBR` regions.  Deterministic atlases
  are :class:`~neurolang.atlases.Atlas` objects, fetched as
  ``.load_atlas(name, kind=...)`` fetches them, added by ``add_atlas_set``
  and recorded in the engine's ``atlases``; probabilistic atlases
  are thresholded (default 0.5) to create binary region masks.
* ``extends`` — one engine name, or a list of them, whose configuration
  is inherited (see :func:`resolve_engine_config`); engines marked
  ``abstract: true`` only serve as bases and cannot be built or listed;
* ``solver_mixins`` — classes, as ``"module:Class"`` strings, put
  before the solver of the engine's :class:`NeurolangPDL` (the classes of
  ``dimension_type_predicates`` after them), e.g. pattern-matching mixins
  adding Datalog commands ``.name(...)``; inherited parent first;
* ``text_embeddings`` — semantic search over document texts (e.g. PubMed
  abstracts) with SPECTER2, registered as a function
  ``f(query, doc) -> similarity``, with ``f_cutoff(query)`` the 95th
  percentile of the similarities to the query, or ``inf`` if the mean of
  its 20 largest is below ``similarity_gate`` (0.75 by default); see
  :mod:`neurolang_apps.utils.text_embeddings`::

      text_embeddings:
        study_abstracts:
          ids_from: study
          source: pubmed          # or {file: ...} or {predicate: ...}
          function: query_similarity
          similarity_gate: 0.75

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

import contextlib
import copy
import functools
import re
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Tuple,
)

import nibabel as nib

import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

from neurolang.utils import euclidean_join_settings
from neurolang_apps.utils import engine_cache
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


@functools.lru_cache(maxsize=None)
def _composed_config(path: Path) -> DictConfig:
    """
    Return the engines of the file *path* merged with the files
    ``<engine>/*.yaml`` next to it, each into the entry of its engine.
    """
    config = OmegaConf.load(path)
    return OmegaConf.merge(config, *(
        {"engines": {name: OmegaConf.load(part)}}
        for name in config.get("engines", {})
        for part in sorted(path.parent.glob(f"{name}/*.yaml"))
    ))


def _load_yaml_config(yaml_path: Optional[Path] = None) -> dict:
    """Load engine YAML from *yaml_path* or the built-in engines files."""
    if yaml_path is not None:
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    return OmegaConf.to_container(
        _composed_config(_ENGINES_YAML), resolve=True
    )


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


# Top-level keys whose texts or lists are concatenated (parent first,
# duplicates dropped) when inherited; every other list is replaced.
_CONCATENATED_KEYS = frozenset({
    "datalog_init", "builtins", "downloads", "ontologies",
    "dimension_type_predicates", "assistant_examples", "solver_mixins",
    "python_init", "hidden_predicates",
})
# Keys whose value is a list, or a single item standing for a list of one.
_LIST_KEYS = frozenset({"python_init"})
# Keys describing the engine itself, never inherited.
_NON_INHERITED_KEYS = frozenset({"extends", "abstract"})


def _extends(raw: Optional[Dict[str, Any]]) -> List[str]:
    """The engines that the engine configuration *raw* extends."""
    bases = (raw or {}).get("extends", [])
    return [bases] if isinstance(bases, str) else bases


def _concatenate(inherited, value):
    """Return *value* after *inherited*, if both are texts or lists."""
    if not (inherited and value):
        return value
    if isinstance(value, str):
        return inherited.rstrip("\n") + "\n" + value
    return list(inherited) + [v for v in value if v not in inherited]


def _without_nulls(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Return *mapping* without its entries set to ``None``, recursively."""
    return {
        k: _without_nulls(v) if isinstance(v, dict) else v
        for k, v in mapping.items() if v is not None
    }


def _merge_engine_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge engine *configs*, each over the previous ones.

    * mappings (``atlases``, ``predicates``, ...) are merged recursively
      by :meth:`OmegaConf.merge`, and an entry set to ``null`` removes the
      inherited one;
    * the values of :data:`_CONCATENATED_KEYS` are concatenated (see
      :func:`_concatenate`);
    * any other value replaces the inherited one.
    """
    configs = [
        {
            k: [v] if k in _LIST_KEYS and isinstance(v, str) else v
            for k, v in c.items() if k not in _NON_INHERITED_KEYS
        }
        for c in configs
    ]
    merged = OmegaConf.to_container(OmegaConf.merge(*configs))
    for key in _CONCATENATED_KEYS:
        values = [c[key] for c in configs if key in c]
        if values:
            merged[key] = functools.reduce(_concatenate, values)
    return _without_nulls(merged)


def _linearized(
    engines: Dict[str, Dict[str, Any]],
    name: str,
    _stack: Tuple[str, ...] = (),
) -> List[str]:
    """
    Return *name* after its bases, depth first in the order of their
    ``extends``, each engine once: a base two of them extend is merged
    once, where it is first reached.
    """
    if name in _stack:
        chain = " -> ".join(_stack + (name,))
        raise ValueError(f"Cyclic engine inheritance: {chain}")
    if name not in engines:
        available = ", ".join(engines.keys())
        raise ValueError(
            f"Unknown engine: {name!r}. Available engines: {available}"
        )
    order = []
    for base in _extends(engines[name]):
        for engine in _linearized(engines, base, _stack + (name,)):
            if engine not in order:
                order.append(engine)
    return order + [name]


def resolve_engine_config(
    engines: Dict[str, Dict[str, Any]],
    name: str,
) -> Dict[str, Any]:
    """
    Return the configuration of *name* with all ``extends`` resolved: the
    configurations of its bases and its own, merged in the order of
    :func:`_linearized`.

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
    return _with_kernel_radius(_merge_engine_configs(*(
        engines[engine] or {} for engine in _linearized(engines, name)
    )))


def _with_kernel_radius(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return *config* with its ``kernel_radius_mm``, if it has one, written
    where its texts say ``{kernel_radius_mm}`` (the descriptions, the
    guidance and the examples of the assistant), and as the ``radius_mm``
    of the activations of its atlases that do not set theirs.
    """
    radius = config.get("kernel_radius_mm")
    if radius is None:
        return config
    for atlas in (config.get("atlases") or {}).values():
        if isinstance(atlas.get("activated"), dict):
            atlas["activated"].setdefault("radius_mm", radius)

    def fill(value):
        if isinstance(value, str):
            return value.replace("{kernel_radius_mm}", f"{radius:g}")
        if isinstance(value, dict):
            return {k: fill(v) for k, v in value.items()}
        if isinstance(value, list):
            return [fill(v) for v in value]
        return value

    return fill(config)


def kernel_radius_voxels(cfg: Dict[str, Any], mask) -> float:
    """
    Return the ``kernel_radius_mm`` of the engine configuration *cfg* in
    voxels of the grid of *mask*, whose voxels are cubes.
    """
    return cfg["kernel_radius_mm"] / abs(float(mask.affine[0, 0]))


@functools.lru_cache(maxsize=None)
def _resolved_config(path: Path, name: str) -> Dict[str, Any]:
    """
    Return the configuration of the engine *name* of the engines file
    *path*, composed with its files and its ``extends`` resolved.

    It is kept for the next calls: copy it before modifying it.
    """
    config = OmegaConf.to_container(_composed_config(path), resolve=True)
    return resolve_engine_config(config["engines"], name)


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
    if yaml_path is None:
        return copy.deepcopy(_resolved_config(_ENGINES_YAML, name))
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

    for ch_name, ch_cfg in (
        *cfg.get("probabilistic_choice", {}).items(),
        *cfg.get("resamplings", {}).items(),
    ):
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


def get_euclidean_join_settings(
    name: str, yaml_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Return the settings of the Euclidean join an engine changes.

    They are the ``euclidean_join`` mapping of the engine, merged with that
    of the engines it extends: settings of
    :class:`~neurolang.utils.euclidean_join_settings.JoinSettings`, as
    ``kdtree_chunk_pairs``.
    """
    cfg = get_engine_config(name, yaml_path)
    return dict(cfg.get("euclidean_join") or {})


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

def _fetch_atlas_difumo(**kwargs):
    return _nilearn_datasets().fetch_atlas_difumo(**kwargs)


def _atlas_fetcher(kind: str) -> Callable[..., Any]:
    """The fetcher of the deterministic atlas *kind*, as ``.load_atlas(name,
    kind=...)`` fetches it (:data:`neurolang.atlases.FETCHERS`), imported
    when called: ``neurolang.atlases`` imports pandas."""

    def fetch(**kwargs):
        from neurolang.atlases import FETCHERS

        return FETCHERS[kind](**kwargs)

    return fetch


_ATLAS_REGISTRY = {
    **{
        kind: {"fetch": _atlas_fetcher(kind), "probabilistic": False}
        for kind in ("destrieux", "schaefer", "tian")
    },
    "difumo": {
        "fetch": _fetch_atlas_difumo,
        "probabilistic": True,
    },
}


def _register_grid(nl, name: str, img) -> None:
    """
    Register the grid of the image *img* as the grid *name* of *nl*.

    The voxel indices of a sort naming the grid (``grid: name``) index
    it: ``.save`` writes them with its shape and affine.
    """
    if getattr(nl, "grids", None) is None:
        nl.grids = {}
    nl.grids[name] = img


def _register_colours(nl, name: str, colours) -> None:
    """Register the colours of the atlas *name* of *nl*, ``(value,
    "#rrggbb")`` pairs, those its authors gave its regions."""
    if getattr(nl, "atlas_colours", None) is None:
        nl.atlas_colours = {}
    nl.atlas_colours[name] = {int(value): colour for value, colour in colours}


def atlas_colours(nl, name: str) -> Dict[int, str]:
    """
    Return the colour, ``#rrggbb``, of each value labelling a region of
    the atlas *name* of *nl* in the image of its grid, as its authors
    coloured it; ``{}`` for an atlas without colours.
    """
    return (getattr(nl, "atlas_colours", None) or {}).get(name, {})


def atlas_label_values(nl, name: str) -> Dict[str, List[int]]:
    """
    Return the values labelling each region of the atlas *name* of *nl*
    in the image of its grid (``nl.grids[name]``), by region name.

    The names are those of the atlas's relation, ``{}`` for an atlas the
    engine did not load: the atlas is the one *nl* records, by name
    (``nl.atlases``, :class:`~neurolang.atlases.Atlas`).
    """
    atlas = (getattr(nl, "atlases", None) or {}).get(name)
    return {} if atlas is None else atlas.label_values()


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


def cached_atlas_activation(
    cache_dir, img, label_names: dict, peaks_mm: np.ndarray,
    studies: np.ndarray, radius_mm: float = 3.0,
) -> List[Tuple[str, Any]]:
    """
    Return the pairs of :func:`atlas_activation`, kept in *cache_dir*.

    They take about 15 s per atlas to find and depend only on the atlas
    image, its labels, the peaks, their studies and the radius, which
    name the file they are kept in: a change of any of them finds another
    file, or none and finds the pairs again (see
    :mod:`neurolang_apps.utils.engine_cache`).
    """
    key = engine_cache.digest(
        float(radius_mm), engine_cache.image_digest(img),
        sorted((int(value), str(name)) for value, name in label_names.items()),
        np.asarray(peaks_mm, dtype=float), np.asarray(studies),
    )

    def find():
        pairs = atlas_activation(
            img, label_names, peaks_mm, studies, radius_mm
        )
        names, codes = np.unique(
            [name for name, _ in pairs], return_inverse=True
        )
        return {
            "names": names.astype(str), "codes": codes.astype(np.int32),
            "studies": np.array([study for _, study in pairs]),
        }

    arrays = engine_cache.keep_arrays(cache_dir, "atlas_activation", key, find)
    names = arrays["names"][arrays["codes"]].tolist()
    return list(zip(names, arrays["studies"].tolist()))


def _relation_frame(nl, name: str):
    """Return the rows of the stored relation *name* of *nl*, as a frame."""
    import pandas as pd

    relation = nl.symbol_table[name].value
    relation = relation.unwrap() if hasattr(relation, "unwrap") else relation
    if hasattr(relation, "as_pandas_dataframe"):
        return relation.as_pandas_dataframe()
    return pd.DataFrame(list(relation))


def _register_atlas_activation(
    nl, labels: dict, img, settings: dict, pred_name: str, cache_dir=None
) -> None:
    """
    Register ``{predicate_name}(label, study)``, the studies activating
    each region of the atlas (see :func:`atlas_activation`), from the
    peaks in millimetres of the relation ``peaks`` (``x, y, z, study``),
    ``peak_mni`` by default, within ``radius_mm`` (3 by default). With a
    *cache_dir* the pairs are kept there (:func:`cached_atlas_activation`).
    """
    peaks = _relation_frame(nl, settings.get("peaks", "peak_mni"))
    find = atlas_activation if cache_dir is None else functools.partial(
        cached_atlas_activation, cache_dir
    )
    pairs = find(
        img, labels, peaks.iloc[:, :3].to_numpy(dtype=float),
        peaks.iloc[:, 3].to_numpy(), float(settings.get("radius_mm", 3)),
    )
    name = settings.get("predicate_name", f"{pred_name}_activated")
    nl.add_tuple_set(pairs, name=name)
    nl.symbol_table[name].value.might_have_duplicates = False


def _register_atlas_points(nl, atlas, settings: dict) -> None:
    """
    Register ``{predicate_name}(x, y, z, label)``: the centre of every
    voxel of a region of *atlas*, in millimetres through its affine, with
    the name of its region (:meth:`~neurolang.atlases.Atlas.points`);
    stored, so that ``EUCLIDEAN`` joins it with a tree.
    """
    nl.add_tuple_set(atlas.points(), name=settings["predicate_name"])


def _atlas_on_template(nl, atlas, cache_dir=None):
    """
    Return the *atlas* on the grid of the template of *nl*, resampled by
    nearest neighbour, so that its voxels (i, j, k) are those of ``voxel``
    and of the peaks (GH-1075); the *atlas* itself if *nl* has no template.
    The image is kept in *cache_dir*, by the atlas's voxels and the grid.
    """
    from neurolang.atlases import Atlas

    template = getattr(nl, "template", None)
    if template is None:
        return atlas
    image = engine_cache.keep_image(
        cache_dir, "atlas", engine_cache.digest(
            engine_cache.image_digest(atlas.image),
            engine_cache.grid_digest(template),
        ),
        lambda: _nilearn_image().resample_to_img(
            atlas.image, template, interpolation="nearest"
        ),
    )
    return Atlas(
        atlas.name, np.asanyarray(image.dataobj), image.affine, atlas.table,
        atlas.space, atlas.background, kind=atlas.kind, image=image,
    )


def _load_deterministic_atlas(
    nl, atlas_name: str, params: dict, data_dir: Path, cache_dir=None
) -> None:
    """
    Load a deterministic atlas (e.g. Destrieux, Schaefer) as engine
    predicates: ``{predicate_name}(label, region)``, with ``points`` its
    voxels in millimetres (see :func:`_register_atlas_points`), and with
    ``activated`` the studies activating each region (see
    :func:`_register_atlas_activation`), kept in *cache_dir* if there is one,
    which also remembers where the atlas is and its labels, so that nilearn
    is not imported to find them.
    """
    from neurolang.atlases import Atlas, label_colours, label_table

    info = _ATLAS_REGISTRY[atlas_name]
    fetch_kw = {
        k: v for k, v in params.items()
        if k not in ("predicate_name", "activated", "points")
    }
    fetch_kw["data_dir"] = str(data_dir)

    def fetch():
        atl_data = info["fetch"](**fetch_kw)
        maps = Path(atl_data["maps"])
        table = label_table(atl_data["labels"])
        return {
            "maps": str(
                maps.relative_to(data_dir)
                if maps.is_relative_to(data_dir) else maps
            ),
            "labels": [
                [int(k), v] for k, v in zip(table["id"], table["name"])
            ],
            "colors": sorted(label_colours(
                atl_data.get("lut") if isinstance(atl_data, dict)
                else getattr(atl_data, "lut", None)
            ).items()),
        }

    found = engine_cache.keep_json(
        cache_dir, "fetched", engine_cache.digest(
            # what is kept: the place of the atlas (Destrieux's on the
            # MNI152 template since it was registered), its labels and colours
            atlas_name, "colours", "mni152", sorted(
                (k, repr(v)) for k, v in fetch_kw.items() if k != "data_dir"
            ),
        ), fetch,
    )
    maps = Path(data_dir) / found["maps"]
    if not maps.exists():  # the atlas was moved: ask for it again
        found = fetch()
        maps = Path(data_dir) / found["maps"]

    # A label of the look-up table may have no voxel in the image (the
    # medial walls of Destrieux): it is no region, as the background.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atlas = Atlas.from_labels(
            atlas_name, nib.load(str(maps)), found["labels"], "MNI152",
            normalise=True, kind=atlas_name,
        )
        on_template = _atlas_on_template(nl, atlas, cache_dir)
    _register_grid(nl, atlas_name, on_template.image)
    _register_colours(nl, atlas_name, found.get("colors", ()))
    pred_name = params.get("predicate_name", atlas_name)
    nl.add_atlas_set(pred_name, on_template)
    # The points and the activations are in millimetres: those of the
    # atlas on its own grid, which may be finer than the template.
    if params.get("points"):
        _register_atlas_points(nl, atlas, params["points"])
    if params.get("activated"):
        _register_atlas_activation(
            nl, atlas.region_names(), atlas.image, params["activated"],
            pred_name, cache_dir,
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
    from neurolang.atlases import label_name

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
        label = label_name(labels[i] if i < len(labels) else f"component_{i}")

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
    nl, atlas_name: str, params: dict, data_dir: Path, cache_dir=None
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
        _load_deterministic_atlas(nl, atlas_name, params, data_dir, cache_dir)


def _get_mni_mask(data_dir: Path) -> nib.Nifti1Image:
    return nib.load(
        _nilearn_datasets().fetch_icbm152_2009(
            data_dir=str(data_dir / "icbm")
        )["t1"]
    )


def _resolve_mask(
    cfg: dict, data_dir: Path, resolution: Optional[float] = None,
    cache_dir=None,
) -> Optional[nib.Nifti1Image]:
    """
    Download the MNI mask if required by the engine config.

    The mask at the *resolution* is kept in *cache_dir* if there is one:
    nilearn, which finds the template and resamples it, is then not imported.
    """
    if not cfg.get("requires_mni_mask", False):
        return None

    def find():
        mask = _get_mni_mask(data_dir)
        if resolution is not None:
            # A 3x3 target affine keeps the original origin; scaling a 4x4
            # identity would also scale its homogeneous coordinate.
            mask = _nilearn_image().resample_img(mask, np.eye(3) * resolution)
        return mask

    return engine_cache.keep_image(
        cache_dir, "mask", engine_cache.digest("icbm152_2009", resolution),
        find,
    )


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
    """
    Run the ``init_engine`` of each ``python_init`` module, in order.

    An init that takes a ``cache_dir`` is given the directory where the
    tables of the engine are kept (see
    :mod:`~neurolang_apps.utils.engine_cache`), or None without cache.
    """
    import importlib
    import inspect

    py_init = cfg.get("python_init") or []
    for module in [py_init] if isinstance(py_init, str) else py_init:
        mod = importlib.import_module(module)
        kwargs = dict(cfg.get("python_init_options", {}))
        if "cache_dir" in inspect.signature(mod.init_engine).parameters:
            kwargs["cache_dir"] = nl.engine_cache_dir
        mod.init_engine(nl, mask, data_dir, **kwargs)


def _solver_mixins(cfg, engine_name) -> list:
    """Return the classes named ``module:Class`` in ``solver_mixins``."""
    import importlib

    mixins = []
    for entry in cfg.get("solver_mixins", []):
        module_name, colon, class_name = entry.partition(":")
        try:
            mixin = getattr(importlib.import_module(module_name), class_name)
            problem = None if isinstance(mixin, type) else "is not a class"
        except (ImportError, AttributeError) as error:
            problem = f"cannot be loaded: {error}"
            if isinstance(error, ModuleNotFoundError):
                problem += (
                    "; install the package that provides it (the "
                    "engine's description, --list-engines, says which)"
                )
        if not colon:
            problem = "is not of the form module:Class"
        if problem:
            raise ValueError(
                f"Entry {entry!r} of solver_mixins of engine "
                f"{engine_name!r} {problem}"
            )
        mixins.append(mixin)
    return mixins


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
            # a name for a URL that has none, and the checksum of the file
            file_name = dl_entry.get("name", Path(url).name)
            options = {}
            if "md5" in dl_entry:
                options["md5sum"] = dl_entry["md5"]
            if "name" in dl_entry:
                options["move"] = file_name
            _fetch_files(dl_dest, [(file_name, url, options)], verbose=1)


def _phase_templates(nl, cfg, data_dir, engine_name):
    for tpl_name, tpl_params in cfg.get("templates", {}).items():
        _load_template(nl, tpl_name, tpl_params, data_dir / engine_name)


def _phase_atlases(nl, cfg, data_dir, engine_name, use_cache=True):
    cache_dir = (
        engine_cache.cache_dir(data_dir, engine_name) if use_cache else None
    )
    for atl_name, atl_params in cfg.get("atlases", {}).items():
        _load_atlas(
            nl, atl_name, atl_params, data_dir / engine_name, cache_dir
        )


def _phase_datalog_init(nl, cfg, data_dir, mask=None):
    """
    Run the ``datalog_init`` program of the engine, from its data directory:
    the paths of its commands (``.load_tractogram``, ``.load_atlas``...) are
    relative to it, where ``downloads`` puts the files. Its
    ``{kernel_radius_voxels}`` is the ``kernel_radius_mm`` of the engine in
    voxels of the grid of *mask*, the distances between voxel indices.
    """
    datalog_init = cfg.get("datalog_init")
    if datalog_init and "{kernel_radius_voxels}" in datalog_init:
        datalog_init = datalog_init.replace(
            "{kernel_radius_voxels}", f"{kernel_radius_voxels(cfg, mask):g}"
        )
    if datalog_init:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        with contextlib.chdir(data_dir):
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


def _phase_resamplings(nl, cfg):
    for name, options in cfg.get("resamplings", {}).items():
        nl.add_resampling(name, **{
            k: v for k, v in options.items() if k != "description"
        })


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
        encoder = text_embeddings.Specter2Encoder(
            model_dir=emb_dir / "models",
            **{
                key: emb_cfg[key]
                for key in ("model", "document_adapter", "query_adapter")
                if key in emb_cfg
            },
        )
        index = text_embeddings.EmbeddingIndex(
            ids=[row[0] for row in nl.symbols[emb_cfg["ids_from"]].value],
            load_texts=_text_loader(
                nl, emb_cfg, emb_dir / f"{emb_name}_pubmed.json",
                engine_name, data_dir, rel_base_dir,
            ),
            encoder=encoder,
            cache_path=emb_dir / f"{emb_name}.npz",
        )
        nl.text_embedding_indexes[emb_name] = index
        nl.add_symbol(
            text_embeddings.make_similarity_function(index),
            name=emb_cfg["function"],
        )
        nl.add_symbol(
            text_embeddings.make_cutoff_function(
                index, emb_cfg.get("similarity_gate", 0.75)
            ),
            name=emb_cfg["function"] + "_cutoff",
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
    use_cache: bool = True,
    progress: Optional[Callable[[str], None]] = None,
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
    use_cache :
        Whether the tables the build takes long to find, the studies
        activating the regions of the atlases, are read from the data
        directory when they are there, and written there when they are
        not (see :mod:`neurolang_apps.utils.engine_cache`).
    progress :
        A function called with a sentence telling what the build is about to
        load, before each of the steps that take a while (``Loading the
        ontologies``…), e.g. for a page to show; only the steps the engine
        has are told.

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
    euclidean_join_settings.override(cfg.get("euclidean_join"))
    data_dir = Path(data_dir)
    rel_base_dir = (Path(yaml_path).parent if yaml_path else None)
    cache_dir = (
        engine_cache.cache_dir(data_dir, name) if use_cache else None
    )
    tell = progress or (lambda message: None)
    if cfg.get("requires_mni_mask"):
        tell("Reading the MNI template")
    mask = _resolve_mask(cfg, data_dir, resolution, cache_dir)

    # The solver mixins of the YAML, then the dimension-type stripper, come
    # before the base solver, whose catch-all refuses unknown commands.
    mixins = _solver_mixins(cfg, name)
    dimension_predicates = cfg.get("dimension_type_predicates")
    if dimension_predicates is not None:
        from neurolang.frontend.datalog.squall import (
            make_dimension_type_stripper,
        )
        mixins.append(make_dimension_type_stripper(dimension_predicates))
    if mixins:
        from neurolang.frontend.probabilistic_frontend import (
            RegionFrontendCPLogicSolver as _BaseSolver,
        )

        class _CustomSolver(*mixins, _BaseSolver):
            pass

        _CustomSolver.__qualname__ = _BaseSolver.__qualname__
        _CustomSolver.__module__ = _BaseSolver.__module__
        nl = NeurolangPDL(solver_class=_CustomSolver)
    else:
        nl = NeurolangPDL()

    nl.template = mask
    nl.grids = {} if mask is None else {TEMPLATE_GRID: mask}
    # Where the Python init of the engine may keep its tables (None: nowhere).
    nl.engine_cache_dir = cache_dir
    _phase_builtins(nl, cfg)
    _phase_base_symbols(nl, cfg, mask)
    if cfg.get("python_init"):
        tell(f"Loading the data of {name}")
    _phase_python_init(nl, cfg, mask, data_dir)
    _phase_downloads(cfg, data_dir, name)
    _phase_templates(nl, cfg, data_dir, name)
    if cfg.get("atlases"):
        tell("Loading the atlases: " + ", ".join(cfg["atlases"]))
    _phase_atlases(nl, cfg, data_dir, name, use_cache)
    # Before the rules, which may use the predicates of the ontologies.
    if cfg.get("ontologies"):
        tell("Loading the ontologies (parsing them takes a few seconds)")
    _phase_ontologies(nl, cfg, data_dir, name, rel_base_dir)
    if cfg.get("datalog_init"):
        tell(f"Defining the rules of {name}")
    _phase_datalog_init(nl, cfg, data_dir, mask)
    _phase_relations(nl, cfg, name, data_dir, rel_base_dir)
    _phase_text_embeddings(nl, cfg, data_dir, name, rel_base_dir)
    _phase_probabilistic_choices(nl, cfg)
    if cfg.get("resamplings"):
        tell("Drawing the bootstrap replicates")
    _phase_resamplings(nl, cfg)

    return nl
