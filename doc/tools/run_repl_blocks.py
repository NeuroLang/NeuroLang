"""
Run the queries of a page of the documentation in the terminal REPL.

The ``.. nl-repl::`` blocks of a page are typed, in order, into
``neurolang-query -i`` on their engine (SQUALL ones in SQUALL mode), as a
reader of the page would; the REPL's output is printed. The exit status is
1 if a statement failed, so that a page whose queries no longer run is seen.

    python doc/tools/run_repl_blocks.py doc/tutorial.rst [-d DATA_DIR]
"""

import argparse
import re
import subprocess
import sys
from itertools import groupby
from pathlib import Path

DIRECTIVE = re.compile(r"^(\s*)\.\. nl-repl::\s*$")
OPTION = re.compile(r"^\s*:(\w+):\s*(.*)$")


def repl_blocks(text: str) -> list:
    """Return the ``(code, mode, engine)`` of the nl-repl blocks of *text*."""
    lines = text.splitlines()
    blocks = []
    for number, line in enumerate(lines):
        match = DIRECTIVE.match(line)
        if not match:
            continue
        indent = len(match.group(1))
        rest = iter(lines[number + 1:])
        options, body = {}, []
        for inner in rest:
            option = OPTION.match(inner)
            if not option:
                body.append(inner)
                break
            options[option.group(1)] = option.group(2).strip()
        for inner in rest:
            if inner.strip() and len(inner) - len(inner.lstrip()) <= indent:
                break
            body.append(inner)
        code = "\n".join(
            row[indent + 3:] if row.strip() else "" for row in body
        ).strip("\n")
        blocks.append((
            code, options.get("mode", "datalog"),
            options.get("engine", "neurosynth"),
        ))
    return blocks


def session(blocks: list) -> str:
    """Return the input of a REPL session typing *blocks*, in order."""
    typed = []
    for code, mode, _ in blocks:
        typed += [f".mode {mode}", code, ""]
    return "\n".join(typed) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("page", type=Path)
    parser.add_argument("-d", "--data-dir", default="neurolang_data")
    args = parser.parse_args(argv)
    blocks = repl_blocks(args.page.read_text())
    failed = False
    for engine, of_engine in groupby(blocks, key=lambda block: block[2]):
        command = [
            sys.executable, "-m", "neurolang_apps.utils.cli", "-i",
            "--engine", engine, "--data-dir", args.data_dir,
        ]
        result = subprocess.run(command, input=session(list(of_engine)), text=True)
        failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
