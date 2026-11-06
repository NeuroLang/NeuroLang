"""Hand the relations of nl_tracts to a NeuroLang engine (DESIGN.md, 8.2).

The only module of nl_tracts that is written against NeuroLang. Two ways:

* :func:`register` gives the relations of a :class:`RelationStore` to an engine
  with ``add_tuple_set(DataFrame, name=...)``, such as ``NeurolangDL`` and
  ``NeurolangPDL``;
* the commands ``.load_tract_atlas``, ``.load_tractogram`` and
  ``.index_tracts`` of a Datalog program build the relations from files, by
  :class:`TractsCommandsMixin`, which :func:`tracts_engine` has in its solver,
  and :func:`init_engine` gives such an engine the vocabulary of rules over
  them (:mod:`nl_tracts.vocabulary`) and the hooks that ``.view`` of the
  apps draws the streamlines with.
"""

import warnings
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import nibabel as nib
import pandas as pd

from nl_expr.exceptions import InvalidCommandExpression
from nl_expr.expression_pattern_matching import add_match
from nl_expr.expression_walker import PatternWalker
from nl_expr.expressions import Command, Constant, Symbol
from neurolang.frontend.probabilistic_frontend import (
    NeurolangPDL,
    RegionFrontendCPLogicSolver,
)

from .atlas import Atlas
from .ids import decode, encode
from .indexing import IndexingParams, RelationStore
from .spaces import SpaceMismatchError
from .terms import load_term_mapping
from .tractogram import Tractogram
from .vocabulary import predicates, prelude

__all__ = [
    "ATLAS_MAPPINGS",
    "RELATION_NAMES",
    "TractsCommandsMixin",
    "TractsSolver",
    "engine_atlas",
    "engine_atlases",
    "engine_label_name",
    "mni_atlas",
    "nilearn_atlas",
    "init_engine",
    "register",
    "relation_frames",
    "tracts_engine",
]

#: The relations, with the names of their columns (DESIGN.md, 3).
RELATION_NAMES = {
    "tractogram": ("t",),
    "streamline": ("x", "t", "s"),
    "atlas": ("a",),
    "label": ("a", "l", "name", "hemi"),
    "label_term": ("a", "l", "term"),
    "opposite_hemisphere": ("h1", "h2"),
    "streamline_label": ("a", "x", "l"),
    "endpoint_label": ("a", "x", "e", "l"),
    "streamline_extent": ("x", "axis", "lo", "hi"),
    "endpoint_coord": ("x", "e", "axis", "v"),
    "endpoint_point": ("x", "e", "px", "py", "pz"),
    "label_extent": ("a", "l", "axis", "lo", "hi"),
    "streamline_length": ("x", "mm"),
}


def register(
    engine: Any,
    store: RelationStore,
    only: Optional[Iterable[str]] = None,
) -> None:
    """Add the relations of ``store`` to ``engine``.

    Each relation is passed as a DataFrame to ``engine.add_tuple_set``,
    under its own name (``streamline_label``, ...), so that no Python tuple is
    built. Relations the store does not have are skipped;
    ``opposite_hemisphere`` is always added.

    Parameters
    ----------
    engine
        A NeuroLang query builder with ``add_tuple_set``.
    store : RelationStore
        The relations to add.
    only : iterable of str, optional
        Add only these relations.
    """
    frames = relation_frames(store)
    for name in (frames if only is None else only):
        if name in frames:
            engine.add_tuple_set(frames[name], name=name)


def relation_frames(store: RelationStore) -> Dict[str, pd.DataFrame]:
    """The relations of a store and ``opposite_hemisphere``, which is
    always there (``left`` and ``right`` are opposite, ``none`` its own)."""
    frames = dict(store.relations)
    frames["opposite_hemisphere"] = pd.DataFrame(
        {
            "h1": ["left", "right", "none"],
            "h2": ["right", "left", "none"],
        }
    )
    return frames


# --- the commands of a Datalog program ------------------------------------

ATLAS_KINDS = {
    "freesurfer": Atlas.from_freesurfer,
    "bids": Atlas.from_bids_dseg,
    "tsv": Atlas.from_tsv,
}


#: The shipped term mapping of the atlases of MNI152 that an engine of the
#: apps may load (``atlases:`` of its ``engines.yaml``) or nilearn fetch
#: (``.load_tract_atlas(..., kind="nilearn", fetcher=...)``), by the
#: atlas's name there or by the fetcher's.
ATLAS_MAPPINGS = {
    "destrieux": "freesurfer_destrieux",
    "destrieux_2009": "freesurfer_destrieux",
    "tian": "tian_subcortex_s2",
}


def engine_label_name(name: Any) -> str:
    """A raw label name as an engine of the apps names its region: ``-``
    and ``_`` are spaces. It is ``_label_name`` of
    ``neurolang_apps.utils.engine_registry``, which nl_tracts does not
    import; a test of the apps checks that they agree."""
    if isinstance(name, bytes):
        name = name.decode("utf8")
    return str(name).replace("-", " ").replace("_", " ")


def mni_atlas(
    name: str, bunch: Any, key: str, space: str = "MNI152",
    term_mapping: Any = None, **kwargs: Any,
) -> Atlas:
    """An atlas of the object a nilearn fetcher returns
    (:meth:`Atlas.from_nilearn`), its label names normalised as an engine
    of the apps names its regions (:func:`engine_label_name`), and so are
    the raw names of its term mapping: ``term_mapping``, else that of
    ``key`` in :data:`ATLAS_MAPPINGS`, if any. Keyword arguments are
    those of :class:`Atlas`."""
    lut = getattr(bunch, "lut", None)
    named = SimpleNamespace(
        maps=bunch.maps,
        lut=None if lut is None else pd.DataFrame({
            "index": lut["index"],
            "name": [engine_label_name(n) for n in lut["name"]],
        }),
        labels=[engine_label_name(n) for n in getattr(bunch, "labels", ())],
    )
    if term_mapping is None:
        term_mapping = ATLAS_MAPPINGS.get(key)
    if term_mapping is not None:
        term_mapping = load_term_mapping(term_mapping)
        term_mapping["atlas_label_name"] = term_mapping[
            "atlas_label_name"].map(engine_label_name)
    return Atlas.from_nilearn(
        name, named, space, term_mapping=term_mapping, **kwargs)


def engine_atlas(engine: Any, name: str) -> Optional[Atlas]:
    """The atlas ``name`` an engine of the apps loaded, or None.

    Its label image is ``engine.grids[name]`` and its regions
    ``engine.atlas_labels[name]`` (a name -> the values of its voxels), so
    that its label names are those of the engine's relation of the atlas,
    ``destrieux(name, region)``. Its space is ``MNI152``, its term mapping
    that of :data:`ATLAS_MAPPINGS` (:func:`mni_atlas`).
    """
    values = engine_atlases(engine).get(name)
    image = (getattr(engine, "grids", None) or {}).get(name)
    if not values or image is None:
        return None
    lut = pd.DataFrame(
        [(v, n) for n, vs in values.items() for v in vs],
        columns=["index", "name"],
    )
    return mni_atlas(name, SimpleNamespace(maps=image, lut=lut), name)


def engine_atlases(engine: Any) -> Dict[str, Any]:
    """The atlases an engine of the apps loaded: name -> its regions."""
    return getattr(engine, "atlas_labels", None) or {}


def nilearn_atlas(
    name: str, fetcher: str, space: str = "MNI152", term_mapping: Any = None,
    background: int = 0, **options: Any,
) -> Atlas:
    """The atlas ``nilearn.datasets.fetch_atlas_<fetcher>(**options)``
    fetches (``destrieux_2009``, ``harvard_oxford`` with ``atlas_name=...``,
    ``schaefer_2018`` with ``n_rois=...``), by :func:`mni_atlas`: its label
    names are those of an engine of the apps, and its term mapping, unless
    given, that of the fetcher in :data:`ATLAS_MAPPINGS`. nilearn is
    imported here only."""
    try:
        from nilearn import datasets
    except ImportError as e:
        raise ImportError(
            'an atlas of kind="nilearn" needs nilearn (pip install nilearn)'
        ) from e
    fetch = getattr(datasets, f"fetch_atlas_{fetcher}", None)
    if fetch is None:
        raise ValueError(f"nilearn has no fetch_atlas_{fetcher}")
    return mni_atlas(
        name, fetch(**options), fetcher, space, term_mapping,
        background=int(background),
    )


class _State:
    """What the commands have built so far, on one solver."""

    def __init__(self):
        self.engine = None  # whose atlases .index_tracts may index by
        self.store = RelationStore()
        self.atlases: Dict[str, Atlas] = {}
        self.tractograms: Dict[str, Tractogram] = {}
        self.in_store = set()  # the atlases and tractograms in the store
        self.length_threshold: Dict[str, float] = {}


def _value(expression):
    return (
        expression.value if isinstance(expression, Constant)
        else expression.name
    )


def _arguments(command, positional, options, more=False):
    """The positional arguments, by name, and the keyword arguments,
    with their defaults; any other is an error, unless ``more``."""
    name = command.name.name
    args = [_value(a) for a in command.args]
    if len(args) != len(positional):
        raise InvalidCommandExpression(
            f".{name} takes {len(positional)} arguments "
            f"({', '.join(positional)}), not {len(args)}"
        )
    given = {_value(k): _value(v) for k, v in command.kwargs}
    unknown = set(given) - set(options)
    if unknown and not more:
        raise InvalidCommandExpression(
            f".{name} has no option {sorted(unknown)}; "
            f"the options are {sorted(options)}"
        )
    return dict(zip(positional, args)), {**options, **given}


class TractsCommandsMixin(PatternWalker):
    """The commands that build the relations of nl_tracts from files.

    ``.load_tract_atlas(name, image, table, kind="freesurfer", space=...,
    term_mapping=..., background=0)``
        An atlas from a label image and a table: ``kind`` is
        ``freesurfer`` (a colour LUT), ``bids`` (a ``dseg.tsv``) or ``tsv``.
        ``term_mapping`` is the key of a shipped mapping or the path of one.
    ``.load_tract_atlas(name, kind="nilearn", fetcher=..., space="MNI152",
    term_mapping=..., background=0, ...)``
        An atlas of nilearn, by the name of its fetcher:
        ``fetcher="destrieux_2009"`` calls
        ``nilearn.datasets.fetch_atlas_destrieux_2009`` with the other
        options (``atlas_name="cort-maxprob-thr25-1mm"`` of
        ``harvard_oxford``, ``n_rois=100`` of ``schaefer_2018``,
        ``data_dir``...). Its label names are those of the atlases of the
        engine (:func:`nilearn_atlas`).
    ``.load_tractogram(id, path, space=...)``
        A ``.trk``, ``.tck`` or ``.vtk``, taken to RAS+ mm.
    ``.index_tracts(tractogram, atlas, crossing_threshold=2.0,
    length_threshold=0.0, emit_background=0, cache=None,
    allow_space_mismatch=0)``
        Index the tractogram by the atlas, loaded by ``.load_tract_atlas``
        or by the engine (:func:`engine_atlas`): ``streamline_label`` and
        ``endpoint_label``, with the relations of the tractogram and of the
        atlas. It is what
        publishes the relations to the program; to be repeated for another
        atlas or tractogram, whose rows are added to the same relations.

    The matches are exact: a command that is not one of these is left to the
    ones after it in the solver, which refuse it.

    The solver has none of the functions of the engine that are named as a
    predicate of the vocabulary (:mod:`nl_tracts.vocabulary`): the
    relations of ``Region`` values ``anterior_of(r1, r2)``, ``left_of``...,
    of the solver of ``NeurolangPDL``, are its rules over streamlines here.
    """

    @property
    def included_functions(self):
        vocabulary = set(predicates())
        return {
            name: function
            for name, function in super().included_functions.items()
            if name not in vocabulary
        }

    @property
    def _tracts(self) -> _State:
        return self.__dict__.setdefault("_tracts_state", _State())

    @add_match(Command(Symbol("load_tract_atlas"), ..., ...))
    def load_tract_atlas(self, command):
        nilearn = any(
            _value(k) == "kind" and _value(v) == "nilearn"
            for k, v in command.kwargs
        )
        options = {"kind": "freesurfer", "space": "unspecified",
                   "term_mapping": None, "background": 0}
        if nilearn:
            names, options = _arguments(
                command, ("name",),
                {**options, "space": "MNI152", "fetcher": None}, more=True,
            )
        else:
            names, options = _arguments(
                command, ("name", "image", "table"), options)
        name = names["name"]
        if not nilearn and options["kind"] not in ATLAS_KINDS:
            raise InvalidCommandExpression(
                f"atlas {name!r}: the kind is one of "
                f"{', '.join(ATLAS_KINDS)} or nilearn, not "
                f"{options['kind']!r}"
            )
        if nilearn and not options["fetcher"]:
            raise InvalidCommandExpression(
                f'atlas {name!r}: kind="nilearn" needs fetcher=..., the '
                "name of a nilearn fetcher, such as destrieux_2009"
            )
        if name in self._tracts.atlases:
            raise InvalidCommandExpression(
                f"atlas {name!r} was already loaded")
        try:
            if nilearn:
                options.pop("kind")
                atlas = nilearn_atlas(name, **options)
            else:
                atlas = ATLAS_KINDS[options["kind"]](
                    name, names["image"], names["table"], options["space"],
                    term_mapping=options["term_mapping"],
                    background=int(options["background"]),
                )
        except (OSError, ValueError, NotImplementedError, ImportError) as e:
            raise InvalidCommandExpression(
                f"cannot load the atlas {name!r}: {e}"
            ) from e
        self._tracts.atlases[name] = atlas

    @add_match(Command(Symbol("load_tractogram"), ..., ...))
    def load_tractogram(self, command):
        names, options = _arguments(
            command, ("id", "path"), {"space": "unspecified"}
        )
        ident = names["id"]
        if ident in self._tracts.tractograms:
            raise InvalidCommandExpression(
                f"tractogram {ident!r} was already loaded")
        try:
            tractogram = Tractogram.from_file(
                ident, names["path"], options["space"]
            )
        except (OSError, ValueError, NotImplementedError) as e:
            raise InvalidCommandExpression(
                f"cannot load the tractogram {ident!r}: {e}"
            ) from e
        self._tracts.tractograms[ident] = tractogram

    @add_match(Command(Symbol("index_tracts"), ..., ...))
    def index_tracts(self, command):
        names, options = _arguments(
            command, ("tractogram", "atlas"),
            {"crossing_threshold": 2.0, "length_threshold": 0.0,
             "emit_background": 0, "cache": None,
             "allow_space_mismatch": 0},
        )
        state = self._tracts
        ident, atlas_name = names["tractogram"], names["atlas"]
        if ident not in state.tractograms:
            raise InvalidCommandExpression(
                f"tractogram {ident!r} was not loaded: .load_tractogram "
                "comes before .index_tracts"
            )
        if atlas_name not in state.atlases:
            atlas = engine_atlas(state.engine, atlas_name)
            if atlas is None:
                known = sorted(
                    set(state.atlases) | set(engine_atlases(state.engine)))
                raise InvalidCommandExpression(
                    f"atlas {atlas_name!r} was not loaded: .load_tract_atlas "
                    "comes before .index_tracts; the atlases loaded and "
                    f"those of the engine are {known}"
                )
            state.atlases[atlas_name] = atlas
        tractogram, atlas = state.tractograms[ident], state.atlases[atlas_name]
        try:
            params = IndexingParams(
                crossing_threshold=float(options["crossing_threshold"]),
                length_threshold=float(options["length_threshold"]),
                emit_background=bool(options["emit_background"]),
            )
        except ValueError as e:
            raise InvalidCommandExpression(str(e)) from e
        previous = state.length_threshold.get(ident)
        if previous is not None and previous != params.length_threshold:
            raise InvalidCommandExpression(
                f"tractogram {ident!r} was indexed with length_threshold="
                f"{previous}, its relations have those streamlines: "
                f"{params.length_threshold} cannot differ"
            )
        allow = bool(options["allow_space_mismatch"])
        try:
            atlas.check_same_space(tractogram, allow)
            if (atlas_name, ident) in state.store._indexed:
                raise ValueError(
                    f"tractogram {ident!r} was already indexed by atlas "
                    f"{atlas_name!r}"
                )
        except SpaceMismatchError as e:
            raise InvalidCommandExpression(f"{e} (space)") from e
        except ValueError as e:
            raise InvalidCommandExpression(str(e)) from e
        if atlas_name not in state.in_store:
            state.store.add_atlas(atlas)
            state.in_store.add(atlas_name)
        if ident not in state.in_store:
            state.store.add_tractogram(tractogram, params)
            state.in_store.add(ident)
            state.length_threshold[ident] = params.length_threshold
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            state.store.add_index(
                tractogram, atlas, params, cache_dir=options["cache"],
                allow_space_mismatch=allow,
            )
        self._publish_tracts()

    def _publish_tracts(self):
        """Give the program the relations built so far (what ``load_csv``
        does for a table), with positions as the names of the columns."""
        for name, frame in relation_frames(self._tracts.store).items():
            self.add_extensional_predicate_from_tuples(
                Symbol(name),
                frame.rename(
                    columns={c: i for i, c in enumerate(frame.columns)}
                ),
            )


class TractsSolver(TractsCommandsMixin, RegionFrontendCPLogicSolver):
    """The solver of ``NeurolangPDL`` with the commands of nl_tracts, which
    come before the ``CommandsMixin`` that refuses the commands it does not
    know."""


def tracts_engine(**kwargs) -> NeurolangPDL:
    """A ``NeurolangPDL`` that understands ``.load_tract_atlas``,
    ``.load_tractogram`` and ``.index_tracts``, with the vocabulary and the
    functions of :func:`init_engine`.

    Parameters
    ----------
    **kwargs
        Those of ``NeurolangPDL``.
    """
    engine = NeurolangPDL(solver_class=TractsSolver, **kwargs)
    init_engine(engine)
    return engine


def init_engine(engine, mask=None, data_dir=None) -> None:
    """Give an engine whose solver has :class:`TractsCommandsMixin` the
    vocabulary of tractography, three functions and two hooks.

    The vocabulary is the rules of :func:`nl_tracts.vocabulary.prelude`,
    over the relations the commands publish. The functions encode and
    decode the id ``x`` of a streamline (``nl_tracts.ids``), over the
    tractograms the commands have indexed:

    ``streamline_id(t, s)``
        ``x`` of the streamline ``s`` of tractogram ``t``;
    ``subject_of(x)``
        the id ``t`` of its tractogram;
    ``index_of(x)``
        its position ``s`` in the tractogram's file.

    They are for a comparison in a rule, ``(t == subject_of(x))``; the
    relation ``streamline(x, t, s)`` does the same by a join. The hooks,
    which the ``.view`` of the apps calls, are the methods
    ``streamline_points(ids)``, the points of each streamline ``x`` of
    ``ids`` in mm (RAS+), and ``streamline_background()``, the label image
    of the first atlas indexed (the tractograms are in its space), or None.

    Its signature is that of the ``python_init`` of an engine of the apps
    (``engines.yaml``): ``mask`` and ``data_dir`` are not used.
    """
    state = engine.program_ir._tracts
    state.engine = engine
    tractograms = state.store.tractograms

    def streamline_id(t: str, s: int) -> int:
        if t not in tractograms:
            raise ValueError(f"tractogram {t!r} was not indexed")
        return int(encode(tractograms.index(t), s))

    def subject_of(x: int) -> str:
        return tractograms[int(decode(x)[0])]

    def index_of(x: int) -> int:
        return int(decode(x)[1])

    for function in (streamline_id, subject_of, index_of):
        engine.add_symbol(function, name=function.__name__)

    def streamline_points(ids):
        index, s = decode(ids)
        lines = []
        for i, k in zip(index.tolist(), s.tolist()):
            tractogram = state.tractograms[tractograms[i]]
            lines.append(tractogram.points[
                tractogram.offsets[k]:tractogram.offsets[k + 1]
            ])
        return lines

    def streamline_background():
        atlases = state.store.relations.get("atlas")
        if atlases is None:
            return None
        atlas = state.atlases[atlases["a"].iloc[0]]
        return nib.Nifti1Image(atlas.labels, atlas.affine)

    engine.streamline_points = streamline_points
    engine.streamline_background = streamline_background
    with warnings.catch_warnings():
        # its relations are published by .index_tracts, later
        warnings.filterwarnings("ignore", "Symbol .* has no entry")
        engine.execute_datalog_program(prelude())
