"""
The files and the addresses the statements of a session may reach.

A program run on a server, for its users (``neurolang-repl-serve``), must
not read the server's files, write them, or fetch addresses on its
network: its commands (``.load_csv``, ``.load_atlas``, ``.import``,
``.load_tractogram``... and the REPL's ``.save``, ``.load``...) would
otherwise do so with the rights of the server. The variable
:data:`CONFINE_ENV` confines them: the directories it lists (separated as
in ``PATH``) are those they may read, they write no file and fetch no
address given by a statement. Without it they reach what the process can.
What the server itself runs, building an engine (its ``datalog_init``
included), runs :func:`lifted`: in its own thread, or context, only.
"""

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import List, Optional

CONFINE_ENV = "NEUROLANG_CONFINE"

# An address of a statement: a scheme of two characters or more, then a
# colon (http:, s3:, and file:/etc/passwd, which pandas opens without
# ``//``); a drive letter (C:) is a path.
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]+:")


_LIFTED = ContextVar("neurolang_confine_lifted", default=False)


class ConfinedError(PermissionError):
    """A statement reaches what its session may not."""


@contextmanager
def lifted():
    """Run what the server runs, not a statement of its users (building an
    engine), unconfined: in this thread, or context, only."""
    token = _LIFTED.set(True)
    try:
        yield
    finally:
        _LIFTED.reset(token)


def confined_roots() -> Optional[List[Path]]:
    """The directories the statements may read, or None if they are not
    confined."""
    value = os.environ.get(CONFINE_ENV)
    if value is None or _LIFTED.get():
        return None
    return [Path(os.path.abspath(root))
            for root in value.split(os.pathsep) if root]


def _inside(path: Path, roots: List[Path]) -> bool:
    """Whether *path*, as written (its ``..`` taken away) or once its links
    are followed, is in one of *roots*: the links of the data (to a shared
    cache) are the server's, as the statements write no file."""
    paths = (Path(os.path.abspath(path)), path.resolve())
    return any(p.is_relative_to(root) or p.is_relative_to(root.resolve())
               for p in paths for root in roots)


def is_url(text) -> bool:
    """Whether *text* is an address (``https://...``), not a path."""
    return isinstance(text, str) and bool(_URL.match(text))


def check_read(path, what: str = "file") -> None:
    """Raise ConfinedError if the statements are confined and *path* (a
    path, or an address) is not in one of the directories they may read;
    *what* names it in the message."""
    roots = confined_roots()
    if roots is None:
        return
    if is_url(path):
        raise ConfinedError(
            f"This server fetches no address given by a statement: {path}")
    if not _inside(Path(path).expanduser(), roots):
        raise ConfinedError(
            f"This server reads no {what} outside its data: {path}")


def check_write(path, what: str = "file") -> None:
    """Raise ConfinedError if the statements are confined: they write no
    file of the server (the page downloads them)."""
    if confined_roots() is not None:
        raise ConfinedError(
            f"This server writes no {what} for a statement ({path}): "
            "download it from the page instead")
