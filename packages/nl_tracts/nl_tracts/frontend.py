"""Hand the relations of nl_tracts to a NeuroLang engine (DESIGN.md, 8.2).

The only module of nl_tracts that is written against NeuroLang. Two ways:

* :func:`register` gives the relations of a :class:`RelationStore` to an engine
  with ``add_tuple_set(DataFrame, name=...)``, such as ``NeurolangDL`` and
  ``NeurolangPDL``;
* the commands ``.load_tract_atlas``, ``.load_tractogram`` and
  ``.index_tracts`` of a Datalog program build the relations from files, by
  :class:`TractsCommandsMixin`, which :func:`tracts_engine` has in its solver.
  That is what the ``.dl`` files written by ``nl_tracts.dl`` use.
"""

import warnings
from typing import Any, Dict, Iterable, Optional

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
from .tractogram import Tractogram

__all__ = [
    "RELATION_NAMES",
    "TractsCommandsMixin",
    "TractsSolver",
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


class _State:
    """What the commands have built so far, on one solver."""

    def __init__(self):
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


def _arguments(command, positional, options):
    """The positional arguments, by name, and the keyword arguments,
    with their defaults; anything else is an error."""
    name = command.name.name
    args = [_value(a) for a in command.args]
    if len(args) != len(positional):
        raise InvalidCommandExpression(
            f".{name} takes {len(positional)} arguments "
            f"({', '.join(positional)}), not {len(args)}"
        )
    given = {_value(k): _value(v) for k, v in command.kwargs}
    unknown = set(given) - set(options)
    if unknown:
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
    ``.load_tractogram(id, path, space=...)``
        A ``.trk``, ``.tck`` or ``.vtk``, taken to RAS+ mm.
    ``.index_tracts(tractogram, atlas, crossing_threshold=2.0,
    length_threshold=0.0, emit_background=0, cache=None,
    allow_space_mismatch=0)``
        Index the tractogram by the atlas: ``streamline_label`` and
        ``endpoint_label``, with the relations of the tractogram and of the
        atlas. It is what
        publishes the relations to the program; to be repeated for another
        atlas or tractogram, whose rows are added to the same relations.

    The matches are exact: a command that is not one of these is left to the
    ones after it in the solver, which refuse it.
    """

    @property
    def _tracts(self) -> _State:
        return self.__dict__.setdefault("_tracts_state", _State())

    @add_match(Command(Symbol("load_tract_atlas"), ..., ...))
    def load_tract_atlas(self, command):
        names, options = _arguments(
            command, ("name", "image", "table"),
            {"kind": "freesurfer", "space": "unspecified",
             "term_mapping": None, "background": 0},
        )
        name = names["name"]
        if options["kind"] not in ATLAS_KINDS:
            raise InvalidCommandExpression(
                f"atlas {name!r}: the kind is one of "
                f"{', '.join(ATLAS_KINDS)}, not {options['kind']!r}"
            )
        if name in self._tracts.atlases:
            raise InvalidCommandExpression(
                f"atlas {name!r} was already loaded")
        try:
            atlas = ATLAS_KINDS[options["kind"]](
                name, names["image"], names["table"], options["space"],
                term_mapping=options["term_mapping"],
                background=int(options["background"]),
            )
        except (OSError, ValueError, NotImplementedError) as e:
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
            raise InvalidCommandExpression(
                f"atlas {atlas_name!r} was not loaded: .load_tract_atlas "
                "comes before .index_tracts"
            )
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
    ``.load_tractogram`` and ``.index_tracts``, and has the functions that
    encode and decode the id ``x`` of a streamline (``nl_tracts.ids``),
    over the tractograms its commands have indexed:

    ``streamline_id(t, s)``
        ``x`` of the streamline ``s`` of tractogram ``t``;
    ``subject_of(x)``
        the id ``t`` of its tractogram;
    ``index_of(x)``
        its position ``s`` in the tractogram's file.

    They are for a comparison in a rule, ``(t == subject_of(x))``; the
    relation ``streamline(x, t, s)`` does the same by a join.

    Parameters
    ----------
    **kwargs
        Those of ``NeurolangPDL``.
    """
    engine = NeurolangPDL(solver_class=TractsSolver, **kwargs)
    tractograms = engine.program_ir._tracts.store.tractograms

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
    return engine
