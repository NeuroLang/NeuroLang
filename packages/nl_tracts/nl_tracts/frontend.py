"""Hand the relations of nl_tracts to a NeuroLang engine (DESIGN.md, 8.2).

The only module of nl_tracts that is written against NeuroLang. Two ways:

* :func:`register` gives the relations of a :class:`RelationStore` to an engine
  with ``add_tuple_set(DataFrame, name=...)``, such as ``NeurolangDL`` and
  ``NeurolangPDL``;
* the commands ``.load_tractogram`` and ``.index_tracts`` of a Datalog
  program build the relations from files and from an atlas of the engine
  (``.load_atlas``, ``add_atlas_set``, its ``atlases:``), by
  :class:`TractsCommandsMixin`, which :func:`tracts_engine` has in its solver,
  and :func:`init_engine` gives such an engine the vocabulary of rules over
  them (:mod:`nl_tracts.vocabulary`) and the hooks that ``.view`` of the
  apps draws the streamlines with. ``neurolang-query --engine tracts`` is
  this engine.
"""

import warnings
from typing import Any, Dict, Iterable, Optional

import nibabel as nib
import pandas as pd

from nl_expr.exceptions import InvalidCommandExpression
from nl_expr.expression_pattern_matching import add_match
from nl_expr.expression_walker import PatternWalker
from nl_expr.expressions import Command, Constant, Symbol
from neurolang.atlases import Atlas, SpaceMismatchError
from neurolang.frontend.probabilistic_frontend import (
    NeurolangPDL,
    RegionFrontendCPLogicSolver,
)

from .atlas import atlas_term_mapping
from .ids import decode, encode
from .indexing import IndexingParams, RelationStore
from .tractogram import Tractogram
from .vocabulary import predicates, prelude

__all__ = [
    "RELATION_NAMES",
    "TractsCommandsMixin",
    "TractsSolver",
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

class _State:
    """What the commands have built so far, on one solver."""

    def __init__(self):
        self.store = RelationStore()
        self.atlases: Dict[str, Atlas] = {}  # those in the store
        self.term_mappings: Dict[str, Any] = {}  # theirs, as given
        self.tractograms: Dict[str, Tractogram] = {}
        self.in_store = set()  # the tractograms in the store
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

    ``.load_tractogram(id, path, space=...)``
        A ``.trk``, ``.tck`` or ``.vtk``, taken to RAS+ mm.
    ``.index_tracts(tractogram, atlas, crossing_threshold=2.0,
    length_threshold=0.0, emit_background=0, cache=None,
    allow_space_mismatch=0, term_mapping=None)``
        Index the tractogram by the atlas, one of the engine
        (``.load_atlas``, ``add_atlas_set``, the ``atlases:`` of an engine
        of the apps: its ``atlases``): ``streamline_label`` and
        ``endpoint_label``, with the relations of the tractogram and of the
        atlas. ``term_mapping`` (the key of a shipped mapping or the path
        of one) gives the terms of its labels, ``label_term``; without it,
        the default of the atlas's kind
        (:data:`nl_tracts.atlas.DEFAULT_TERM_MAPPINGS`: ``freesurfer``,
        ``destrieux``, ``tian``), else none. It is what publishes the
        relations to the program; to be repeated for another atlas or
        tractogram, whose rows are added to the same relations.

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
             "allow_space_mismatch": 0, "term_mapping": None},
        )
        state = self._tracts
        ident, atlas_name = names["tractogram"], names["atlas"]
        if ident not in state.tractograms:
            raise InvalidCommandExpression(
                f"tractogram {ident!r} was not loaded: .load_tractogram "
                "comes before .index_tracts"
            )
        atlas = state.atlases.get(atlas_name, self.atlases.get(atlas_name))
        if atlas is None:
            raise InvalidCommandExpression(
                f"atlas {atlas_name!r} was not loaded: .load_atlas comes "
                "before .index_tracts; the atlases of the engine are "
                f"{sorted(self.atlases)}"
            )
        mapping = options["term_mapping"]
        if atlas_name in state.atlases and mapping not in (
            None, state.term_mappings[atlas_name]
        ):
            raise InvalidCommandExpression(
                f"atlas {atlas_name!r} was indexed with term_mapping="
                f"{state.term_mappings[atlas_name]!r}, its relations have "
                f"those terms: {mapping!r} cannot differ"
            )
        tractogram = state.tractograms[ident]
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
        if atlas_name not in state.atlases:
            try:
                state.store.add_atlas(
                    atlas, atlas_term_mapping(atlas, mapping)
                )
            except (OSError, ValueError) as e:
                raise InvalidCommandExpression(
                    f"the term mapping of atlas {atlas_name!r}: {e}"
                ) from e
            state.atlases[atlas_name] = atlas
            state.term_mappings[atlas_name] = mapping
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
    """A ``NeurolangPDL`` that understands ``.load_tractogram`` and
    ``.index_tracts`` (and ``.load_atlas``), with the vocabulary and the
    functions of :func:`init_engine`.

    Parameters
    ----------
    **kwargs
        Those of ``NeurolangPDL``.
    """
    engine = NeurolangPDL(solver_class=TractsSolver, **kwargs)
    init_engine(engine)
    return engine


def init_engine(engine, mask=None, data_dir=None, **options) -> None:
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

    It is the ``python_init`` of the engine ``tracts`` of the apps
    (``engines.yaml``), hence its signature: ``mask``, ``data_dir`` and the
    ``options`` (the ``python_init_options`` of an engine extending
    ``tracts`` and another, given to the inits of both) are not used.
    """
    state = engine.program_ir._tracts
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
