"""A Datalog file from a WMQL program: EDB commands, rules, answer.

The file is for the NeuroLang frontend (``nl-tracts run``, or
``tracts_engine().execute_datalog_program``) and is meant to be read and
changed by hand:

* the **commands** ``.load_tract_atlas``, ``.load_tractogram`` and
  ``.index_tracts`` build the relations (``traverses``, ``endpoint``,
  ``label``, ``streamline_extent``, ...) from the files of the atlas and of
  the tractograms when it runs (``nl_tracts.frontend``);
* the **rules** of the WMQL program, each statement a ``%`` comment over the
  rules it gives, cut to what the answer needs (``nl_tracts.wmql``);
* the **answer**, ``ans(t, s) :- q_<name>(t, s)``: the streamlines of the
  query, for each tractogram.

Text in, text out: this module imports nothing of NeuroLang.
"""

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from .wmql import WMQLError, transpile

__all__ = ["AtlasSource", "TractogramSource", "datalog_file"]


@dataclass(frozen=True)
class AtlasSource:
    """The files of the atlas whose label ids a WMQL program uses.

    Attributes
    ----------
    name : str
        The atlas's name, the first column of ``traverses`` and ``label``.
    image, table : path
        The label image and its table: a FreeSurfer colour LUT
        (``kind="freesurfer"``), a ``dseg.tsv`` (``"bids"``) or a TSV with
        ``id``, ``name``, ``hemi`` (``"tsv"``).
    kind, space : str
        The kind of table, and the space tag (an atlas and a tractogram in
        different spaces do not combine).
    term_mapping : str, optional
        A shipped term mapping (``freesurfer_aseg_aparc``,
        ``freesurfer_destrieux``) or the path of one, for ``label_term``.
    background : int
        The label of "no region".
    """

    name: str
    image: str
    table: str
    kind: str = "freesurfer"
    space: str = "unspecified"
    term_mapping: Optional[str] = None
    background: int = 0


@dataclass(frozen=True)
class TractogramSource:
    """A tractogram file (``.trk``, ``.tck``, ``.vtk``): its id, the value
    of ``t`` in the relations, its path and its space tag."""

    id: str
    path: str
    space: str = "unspecified"


def _string(value: Union[str, os.PathLike]) -> str:
    """A Datalog string literal; a quote, a backslash or a line break would
    end it or change it."""
    text = os.fspath(value)
    if any(c in text for c in '"\\\n\r'):
        raise ValueError(
            f"{text!r} cannot be written in a string of a Datalog file "
            "(a double quote, a backslash or a line break)"
        )
    return f'"{text}"'


def datalog_file(
    source: str,
    atlas: AtlasSource,
    tractograms: Sequence[TractogramSource],
    answer: Optional[str] = None,
    include_folders: Sequence[str] = (),
    crossing_threshold: float = 2.0,
    length_threshold: float = 0.0,
    emit_background: bool = True,
    cache: Optional[str] = None,
    anatomical_vertical: bool = False,
    title: Optional[str] = None,
) -> str:
    """The text of the Datalog file for a WMQL program.

    Parameters
    ----------
    source : str
        The WMQL program.
    atlas : AtlasSource
        The atlas of the label ids of the program.
    tractograms : sequence of TractogramSource
        The tractograms to run it on.
    answer : str, optional
        The WMQL name whose streamlines are the answer; default, the last
        name the program saves (``=``, or named alone). The frontend allows
        one ``ans`` rule: another query is another file, or this line
        changed.
    include_folders : sequence of str
        Where ``import`` in the program looks for files.
    crossing_threshold, length_threshold : float
        Percent of its points that a streamline needs in a label to
        traverse it, and the minimum length in mm (6.1 of the design).
    emit_background : bool
        Whether ``traverses`` has rows for the background. ``True``, as
        tract_querier's ``not`` and ``only`` read them (11.2).
    cache : str, optional
        A directory where the indexing is kept, and read from the next time.
    anatomical_vertical : bool
        ``superior_of`` and ``inferior_of`` as their names say, not as
        tract_querier has them (11.5).
    title : str, optional
        What to call the program in the header (a file name).

    Raises
    ------
    WMQLError
        On a program that tract_querier would reject, or an unknown answer.
    ValueError
        If there is no answer to default to, or a value cannot be written
        in a Datalog string.
    """
    program = transpile(
        source, atlas=atlas.name, include_folders=include_folders,
        anatomical_vertical=anatomical_vertical,
    )
    if answer is None:
        if not program.saved:
            raise ValueError(
                "the program saves no query (`name = ...`, or a name alone): "
                "say which is the answer"
            )
        answer = program.saved[-1]
    elif answer not in program.defined:
        raise WMQLError(
            f"the answer {answer!r} is not a query of the program; its "
            f"queries: {', '.join(sorted(program.defined))}"
        )
    predicate = program.defined[answer].tracts
    if predicate is None:
        raise WMQLError(
            f"the query {answer!r} has no streamlines whatever the data"
        )

    out = [
        f"% Datalog for the NeuroLang frontend, from the WMQL program "
        f"{title or '<stdin>'}.",
        f"% Run it: nl-tracts run <this file>   (the answer is ans(t, s): "
        f"the streamlines s of tractogram t, for {answer}).",
        f"% Atlas {atlas.name!r} ({atlas.kind}), space {atlas.space!r}; "
        f"{len(tractograms)} tractogram(s): "
        + ", ".join(t.id for t in tractograms) + ".",
        "% Other queries of the program: "
        + (", ".join(n for n in program.saved if n != answer) or "none")
        + ".",
        "",
        "% The relations (traverses, endpoint, label, streamline_extent, ...)"
        " are built from the files by these commands.",
    ]
    atlas_options = [
        f"kind={_string(atlas.kind)}", f"space={_string(atlas.space)}",
    ]
    if atlas.term_mapping:
        atlas_options.append(f"term_mapping={_string(atlas.term_mapping)}")
    if atlas.background:
        atlas_options.append(f"background={int(atlas.background)}")
    out.append(
        f".load_tract_atlas({_string(atlas.name)}, {_string(atlas.image)}, "
        f"{_string(atlas.table)}, {', '.join(atlas_options)})"
    )
    for tractogram in tractograms:
        out.append(
            f".load_tractogram({_string(tractogram.id)}, "
            f"{_string(tractogram.path)}, space={_string(tractogram.space)})"
        )
    index_options = (
        f"crossing_threshold={float(crossing_threshold)}, "
        f"length_threshold={float(length_threshold)}, "
        f"emit_background={int(bool(emit_background))}"
    )
    if cache:
        index_options += f", cache={_string(cache)}"
    for tractogram in tractograms:
        out.append(
            f".index_tracts({_string(tractogram.id)}, "
            f"{_string(atlas.name)}, {index_options})"
        )
    out += [
        "",
        "% The query: the rules of each statement, "
        "cut to what the answer needs.",
        program.program(
            answer, components=("tracts",), comments=True
        ).rstrip(),
        "",
        f"% The answer: the streamlines of {answer}.",
        f"ans(t, s) :- {predicate}(t, s)",
    ]
    return "\n".join(out) + "\n"
