"""The engine of nl_tracts: what an engine of the apps needs to have the
relations of a tractogram (DESIGN.md, 8.2), its only module written against
NeuroLang.

* :class:`TractsCommandsMixin`, a mixin of the solver: the commands
  ``.load_tractogram`` and ``.index_tracts`` of a Datalog program build the
  relations from files and from an atlas of the engine (``.load_atlas``,
  ``add_atlas_set``, its ``atlases:``);
* :func:`init_engine`, its ``python_init``: the functions of the ids of the
  streamlines and the hooks that ``.view`` draws them with.

The vocabulary of rules over the relations is ``vocabulary.dl``, which
:func:`init_engine` imports. The engine ``tracts`` of ``engines.yaml``
puts them together; an engine of the apps has them by extending it
(``extends: [neurosynth, tracts]``).
"""

import warnings
from pathlib import Path
from typing import Any, Dict

import nibabel as nib
import pandas as pd

from nl_expr.exceptions import InvalidCommandExpression
from nl_expr.expression_pattern_matching import add_match
from nl_expr.expression_walker import PatternWalker
from nl_expr.expressions import Command, Constant, Symbol
from neurolang.atlases import Atlas, SpaceMismatchError

from . import vocabulary_predicates
from .atlas import atlas_term_mapping
from .fetch import ATLASES, fetch_tractogram
from .ids import decode, encode
from .indexing import IndexingParams, RelationStore
from .tractogram import Tractogram

__all__ = ["TractsCommandsMixin", "init_engine"]

#: The predicates of the vocabulary (``vocabulary.dl``).
VOCABULARY_PREDICATES = vocabulary_predicates()


def _relation_frames(store: RelationStore) -> Dict[str, pd.DataFrame]:
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
        self.data_dir = None  # where the atlases of Yeh are downloaded
        self.store = RelationStore()
        self.atlases: Dict[str, Atlas] = {}  # those in the store
        self.term_mappings: Dict[str, Any] = {}  # theirs, as given
        self.extents: Dict[str, str] = {}  # theirs, of label_extent
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

    ``.load_tractogram(id, path, space=..., drop_short=0)``
        A ``.trk``, ``.tck`` or ``.vtk``, taken to RAS+ mm; with
        ``drop_short=1``, its streamlines of one point are in no relation.
    ``.load_tractogram(id, kind="hcp842", bundle=None, space="MNI152",
    data_dir=None, url=None, md5=None, drop_short=1)``
        The whole brain or a bundle (``bundle="UF_L"``) of an atlas of
        Yeh, ``hcp842`` or ``hcp1065`` (bundles only), downloaded the
        first time into ``data_dir``, by default that of the engine
        (:func:`init_engine`) or ``~/.cache/nl_tracts``
        (:func:`nl_tracts.fetch.fetch_tractogram`).
    ``.index_tracts(tractogram, atlas, crossing_threshold=2.0,
    length_threshold=0.0, emit_background=0, cache=None,
    allow_space_mismatch=0, term_mapping=None, extent=None)``
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
        ``extent`` is that of ``label_extent``, fixed by the first
        indexing by the atlas, ``"edge"`` if not given: ``"edge"`` (the
        voxel edges, DESIGN.md 6.1.1) or ``"centre"`` (the voxel centres,
        as tract_querier's boxes, for the directional terms to reproduce
        its own).

    The matches are exact: a command that is not one of these is left to the
    ones after it in the solver, which refuse it.

    The solver has none of the functions of the engine that are named as a
    predicate of the vocabulary (``vocabulary.dl``): the
    relations of ``Region`` values ``anterior_of(r1, r2)``, ``left_of``...,
    of the solver of ``NeurolangPDL``, are its rules over streamlines here.
    """

    @property
    def included_functions(self):
        return {
            name: function
            for name, function in super().included_functions.items()
            if name not in VOCABULARY_PREDICATES
        }

    @property
    def _tracts(self) -> _State:
        return self.__dict__.setdefault("_tracts_state", _State())

    @add_match(Command(Symbol("load_tractogram"), ..., ...))
    def load_tractogram(self, command):
        kind = next(
            (_value(v) for k, v in command.kwargs if _value(k) == "kind"),
            None,
        )
        if kind is None:
            names, options = _arguments(
                command, ("id", "path"),
                {"space": "unspecified", "drop_short": 0},
            )
        elif kind in ATLASES:
            names, options = _arguments(
                command, ("id",),
                {"kind": kind, "bundle": None, "space": "MNI152",
                 "data_dir": self._tracts.data_dir, "url": None,
                 "md5": None, "drop_short": 1},
            )
        else:
            raise InvalidCommandExpression(
                f"the kind of a tractogram is one of {', '.join(ATLASES)}, "
                f"or none for a file, not {kind!r}"
            )
        ident = names["id"]
        if ident in self._tracts.tractograms:
            raise InvalidCommandExpression(
                f"tractogram {ident!r} was already loaded")
        try:
            path = names.get("path") or fetch_tractogram(
                kind, options["bundle"], options["data_dir"],
                options["url"], options["md5"],
            )
            tractogram = Tractogram.from_file(
                ident, path, options["space"], bool(options["drop_short"])
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
             "allow_space_mismatch": 0, "term_mapping": None,
             "extent": None},
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
        extent = options["extent"]
        if atlas_name in state.atlases and extent not in (
            None, state.extents[atlas_name]
        ):
            raise InvalidCommandExpression(
                f"atlas {atlas_name!r} was indexed with extent="
                f"{state.extents[atlas_name]!r}, its relations have that "
                f"extent: {extent!r} cannot differ"
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
                    atlas, atlas_term_mapping(atlas, mapping),
                    extent or "edge",
                )
            except (OSError, ValueError) as e:
                raise InvalidCommandExpression(
                    f"the term mapping or extent of atlas {atlas_name!r}: {e}"
                ) from e
            state.atlases[atlas_name] = atlas
            state.term_mappings[atlas_name] = mapping
            state.extents[atlas_name] = extent or "edge"
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
        for name, frame in _relation_frames(self._tracts.store).items():
            self.add_extensional_predicate_from_tuples(
                Symbol(name),
                frame.rename(
                    columns={c: i for i, c in enumerate(frame.columns)}
                ),
            )


def init_engine(engine, mask=None, data_dir=None, **options) -> None:
    """Give an engine whose solver has :class:`TractsCommandsMixin` the
    vocabulary of tractography, ``.import('nl_tracts:vocabulary.dl')``,
    three functions and two hooks.

    The functions encode and decode the id ``x`` of a streamline
    (``nl_tracts.ids``), over the tractograms the commands have indexed:

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
    (``engines.yaml``), hence its signature: ``mask`` and the ``options``
    (the ``python_init_options`` of an engine extending ``tracts`` and
    another, given to the inits of both) are not used, and the atlases of
    ``.load_tractogram(..., kind="hcp842")`` are downloaded into
    ``<data_dir>/tracts`` (an engine's directory of the apps), if given.
    """
    state = engine.program_ir._tracts
    if data_dir is not None:
        state.data_dir = str(Path(data_dir) / "tracts")
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
        engine.execute_datalog_program(".import('nl_tracts:vocabulary.dl')")
