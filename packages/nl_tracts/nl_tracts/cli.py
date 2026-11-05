"""``nl-tracts``: a Datalog file from a WMQL program, and a way to run it.

``nl-tracts dl PROGRAM.qry --atlas NAME KIND IMAGE TABLE --tractogram ID PATH``
writes the file (:mod:`nl_tracts.dl`): the commands that build the relations
from the files, the rules of the query with a comment over each statement it
comes from, and ``ans(t, s)``. ``nl-tracts run FILE.dl`` runs it with the
NeuroLang frontend that has those commands (:func:`nl_tracts.frontend.
tracts_engine`) and prints the answer.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .dl import AtlasSource, TractogramSource, datalog_file
from .wmql import WMQLError


def _read(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text()


def _dl(args) -> int:
    source = _read(args.program)
    name, kind, image, table = args.atlas
    text = datalog_file(
        source,
        AtlasSource(
            name, image, table, kind=kind, space=args.space,
            term_mapping=args.term_mapping, background=args.background,
        ),
        [TractogramSource(i, p, args.space) for i, p in args.tractogram],
        answer=args.ans,
        include_folders=args.include,
        crossing_threshold=args.crossing_threshold,
        length_threshold=args.length_threshold,
        emit_background=not args.no_background,
        cache=args.cache,
        anatomical_vertical=args.anatomical_vertical,
        title=None if args.program == "-" else Path(args.program).name,
    )
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text)
    return 0


def _run(args) -> int:
    from nl_expr.exceptions import NeuroLangException

    from .frontend import tracts_engine

    text = _read(args.file)
    try:
        result = tracts_engine().execute_datalog_program(text)
    except (NeuroLangException, OSError) as e:
        print(f"nl-tracts: {e}", file=sys.stderr)
        return 1
    if result is None:
        print("nl-tracts: the program has no ans(...) query", file=sys.stderr)
        return 1
    frame = result.as_pandas_dataframe()
    frame.columns = (
        ["t", "s"] if len(frame.columns) == 2
        else [f"c{i}" for i in range(len(frame.columns))]
    )
    frame = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    shown = frame if args.limit is None else frame.head(args.limit)
    if args.format == "csv":
        sys.stdout.write(shown.to_csv(index=False))
    else:
        print(shown.to_string(index=False))
    print(f"{len(frame)} rows", file=sys.stderr)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nl-tracts", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser(
        "dl", help="write the Datalog file of a WMQL program",
        description="Write the Datalog file of a WMQL program: the commands "
        "that build the relations from the files, the rules, the answer.",
    )
    dl.add_argument("program", help="the WMQL file, or - for stdin")
    dl.add_argument(
        "--atlas", nargs=4, required=True,
        metavar=("NAME", "KIND", "IMAGE", "TABLE"),
        help="the atlas whose label ids the program uses: KIND is "
        "freesurfer (a colour LUT), bids (a dseg.tsv) or tsv",
    )
    dl.add_argument(
        "--tractogram", nargs=2, action="append", required=True,
        metavar=("ID", "PATH"), help="a tractogram (.trk, .tck, .vtk); "
        "repeat for more",
    )
    dl.add_argument("--space", default="unspecified",
                    help="the space tag of the atlas and the tractograms")
    dl.add_argument("--term-mapping", metavar="KEY_OR_PATH",
                    help="a term mapping for label_term, e.g. "
                    "freesurfer_aseg_aparc")
    dl.add_argument("--background", type=int, default=0,
                    help="the label of no region (default 0)")
    dl.add_argument("--ans", metavar="NAME",
                    help="the query that is the answer (default: the last "
                    "the program saves)")
    dl.add_argument("--include", action="append", default=[], metavar="DIR",
                    help="where `import` looks for files; repeat")
    dl.add_argument("--crossing-threshold", type=float, default=2.0,
                    metavar="PERCENT", help="percent of its points a "
                    "streamline has in a label to traverse it (default 2)")
    dl.add_argument("--length-threshold", type=float, default=0.0,
                    metavar="MM", help="shorter streamlines are left out")
    dl.add_argument("--no-background", action="store_true",
                    help="no streamline_label rows for the background "
                    "(WMQL's not and only read them: the default has them)")
    dl.add_argument("--cache", metavar="DIR",
                    help="keep the indexing here, and read it next time")
    dl.add_argument("--anatomical-vertical", action="store_true",
                    help="superior_of and inferior_of as their names say, "
                    "not as tract_querier has them")
    dl.add_argument("-o", "--output", default="-",
                    help="the file to write (default: stdout)")
    dl.set_defaults(function=_dl)

    run = sub.add_parser(
        "run", help="run a Datalog file with the NeuroLang frontend")
    run.add_argument("file", help="the Datalog file, or - for stdin")
    run.add_argument("--format", choices=("table", "csv"), default="table")
    run.add_argument("--limit", type=int, metavar="N",
                     help="show the first N rows")
    run.set_defaults(function=_run)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.function(args)
    except (WMQLError, ValueError, OSError) as e:
        print(f"nl-tracts: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
