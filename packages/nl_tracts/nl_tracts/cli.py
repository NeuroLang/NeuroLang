"""``nl-tracts wmql FILE.qry``: a WMQL program of tract_querier in the
vocabulary of tractography (:mod:`nl_tracts.wmql_vocab`), on the standard
output; what is not translated is listed in the header of the rules, and
on the standard error."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .terms import SHIPPED_MAPPINGS, load_term_mapping
from .wmql_vocab import WMQLError, translate


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="nl-tracts")
    commands = parser.add_subparsers(dest="command", required=True)
    wmql = commands.add_parser(
        "wmql", help="translate a .qry program to the vocabulary")
    wmql.add_argument("program", help="the .qry file, - for standard input")
    wmql.add_argument(
        "-I", "--include", action="append", default=[],
        help="a folder where import looks for files (the program's first)")
    wmql.add_argument(
        "--terms", metavar="MAPPING",
        help="the terms of a mapping, a shipped one "
        f"({', '.join(SHIPPED_MAPPINGS)}) or a TSV file: a name of labels "
        "that is not one of them stands for the terms it is made of")
    wmql.add_argument(
        "--strict", action="store_true",
        help="fail on what cannot be translated instead of leaving it out")
    args = parser.parse_args(argv)

    if args.program == "-":
        source, folders, title = sys.stdin.read(), args.include, "<stdin>"
        qry_file = None
    else:
        path = Path(args.program)
        source, title = path.read_text(), path.name
        folders = [str(path.parent)] + args.include
        qry_file = path.name
    terms = None if args.terms is None else set(
        load_term_mapping(args.terms)["term"])
    try:
        translation = translate(
            source, folders, terms=terms,
            skip_unsupported=not args.strict, title=title,
            qry_file=qry_file)
    except WMQLError as e:
        print(f"nl-tracts: {e}", file=sys.stderr)
        return 1
    sys.stdout.write(translation.text)
    if translation.skipped:
        print(f"nl-tracts: {len(translation.skipped)} queries not "
              "translated (see the header)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
