"""
The results of the queries every session would find the same, kept for all
the sessions of a REPL server.

A query that reads nothing of its session (no rule, relation, fact or upload
of it) has the same answer in every session on the same engine, data and
NeuroLang: 29 representative queries of ``neurosynth_iit`` gave the same
rows in two builds and in a new scope. The examples of the page and the
queries of its example questions, shown to every user, take 3.4 to 10.5 s
each on Margaret (0.4 to 0.9 s read back); :class:`ResultCache` runs them
once.

The rows of such a result are kept in an Arrow file of its directory, named
by a key (:meth:`ResultCache.key`) of

- the engine (:func:`engine_digest`): its name, resolution and
  configuration, the files of the data directory and of NeuroLang's packages
  (path, size and time of change) and the versions of the libraries that
  compute the answers;
- the program as parsed, without its comments and spaces, its fresh
  variables numbered in order;
- the settings of the session the solving depends on (the query backend, the
  profile of the Euclidean join).

A program is shared only if it is a Datalog query with no command and no
symbol it reaches, through its rules and the engine's rules and constraints,
is defined in the session, nor a constraint: otherwise its result is the
session's own, neither read nor kept.
A command (``.load_tractogram``, ``.import``...) changes the engine in ways
its symbols may not tell: after one, the engine shares nothing more.

A file is written whole or not at all, and kept only if Arrow holds its
columns as they are (none of Python objects, each of the same type back in
pandas) and the file reads back equal to them; reading it makes it the most recently used, and the least
recently used go when the files take more than the limit
(``result_cache_mb`` of the server, :doc:`/repl_server`). Another process
removing a file only makes the query run.
"""

import json
import os
import re
from importlib import import_module, metadata
from pathlib import Path
from typing import NamedTuple, Optional

from nl_expr.expression_walker import expression_iterator
from nl_expr.expressions import Command, Constant, Symbol

from . import engine_cache

#: Raise it when what a key is made of, or the files, change.
VERSION = 1

#: The directories of the data directory that hold what builds and queries
#: keep, not data.
_NOT_DATA = {engine_cache.DIRECTORY, "derived_tables", "__pycache__", ".git"}
#: Whose code makes the answers: their files and versions are in the key.
_PACKAGES = ("nl_expr", "nl_core", "neurolang", "neurolang_apps")
_LIBRARIES = ("numpy", "scipy", "pandas", "pyarrow", "polars")
#: Where a program keeps its constraints (``body \N{RIGHTWARDS ARROW} head``).
_CONSTRAINTS = Symbol("__constraints__")
#: The fresh symbols of a parsed program, numbered as they were made.
_FRESH = re.compile(r"(?<=S\{)fresh_[0-9a-f]+(?=:)")


class Kept(NamedTuple):
    """A result kept: its rows and the notes its run printed."""

    frame: object
    notes: list


def _files(root, skip=()) -> list:
    """Return the files under *root*, ``(path, size, time of change)``, but
    for those of the directories *skip* (paths) and :data:`_NOT_DATA`."""
    root, found = Path(root), []
    skip = {str(Path(path).resolve()) for path in skip}
    for directory, names, files in os.walk(root):
        names[:] = sorted(
            name for name in names if name not in _NOT_DATA
            and str(Path(directory, name).resolve()) not in skip
        )
        for name in sorted(files):
            if name.endswith((".pyc", ".tmp")):
                continue
            try:
                stat = os.stat(os.path.join(directory, name))
            except OSError:  # gone meanwhile
                continue
            found.append((os.path.relpath(os.path.join(directory, name), root),
                          stat.st_size, stat.st_mtime_ns))
    return found


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def engine_digest(engine_name: str, resolution, data_dir, light=False,
                  skip=()) -> str:
    """
    Return a name for what the answers of the engine *engine_name* depend
    on: its name, *resolution*, configuration (*light* or not), the files
    of *data_dir* but those of *skip* (path, size, time of change), the
    files of NeuroLang's packages and the versions of the libraries.
    """
    from . import engine_registry

    try:
        config = engine_registry.get_engine_config(engine_name)
    except ValueError:  # not one of the registry's
        config = None
    return engine_cache.digest(
        VERSION, engine_name, resolution, bool(light),
        json.dumps(config, sort_keys=True, default=str),
        [(name, _version(name)) for name in _PACKAGES + _LIBRARIES],
        [_files(Path(import_module(name).__file__).parent)
         for name in _PACKAGES],
        _files(data_dir, skip) if data_dir else [],
    )


def _symbols(expression):
    return (e for _, e in expression_iterator(expression)
            if isinstance(e, Symbol))


class ResultCache:
    """
    The results of the queries of the engine *nl*, :func:`engine_digest`
    *engine*, shared by its sessions, kept in *directory* within *most_mb*
    megabytes. Made when the engine is built, before any session: its scope
    then is the engine's, those above it the sessions'.
    """

    def __init__(self, nl, engine: str, directory, most_mb: float):
        self.engine = engine
        self.directory = Path(directory).expanduser()
        self.most = int(most_mb * (1 << 20))
        self._engine_scope = nl.program_ir.symbol_table
        #: The engine's constraints by the symbols of their head: a query
        #: reading one of them may read their body too.
        self._constraints = {}
        for constraint in nl.program_ir.constraints().formulas:
            for symbol in _symbols(constraint.consequent):
                self._constraints.setdefault(symbol, []).append(constraint)
        #: Whether a command changed the engine: nothing is shared then.
        self.changed = False

    # -- What a program is ------------------------------------------------

    def key(self, nl, program: str, *settings) -> Optional[str]:
        """
        Return the key of the result of the Datalog *program* on *nl*, with
        the *settings* of the session, if it is shared, else None: when it
        does not parse, reads what the session defined, or the engine was
        changed by a command, this one's included (see :attr:`changed`).
        """
        if self.changed:
            return None
        try:
            parsed = nl.datalog_parser(program)
        except Exception:
            return None
        if any(isinstance(e, Command) for _, e in expression_iterator(parsed)):
            self.changed = True
            return None
        if self._reads_session(nl, parsed):
            return None
        from nl_expr.config import config

        names = {}
        text = _FRESH.sub(
            lambda m: names.setdefault(m.group(), f"fresh_{len(names)}"),
            repr(parsed),
        )
        return engine_cache.digest(
            self.engine, text, settings,
            {section: dict(config[section]) for section in config.sections()},
        )

    def session_symbols(self, nl) -> Optional[set]:
        """Return the symbols defined in the scopes of *nl* above the
        engine's, the session's and its submission's, or None if its scope
        is not on the engine's."""
        session, scope = set(), nl.program_ir.symbol_table
        while scope is not self._engine_scope:
            if scope is None:
                return None
            session.update(scope._symbols)
            scope = scope.enclosing_scope
        return session

    def _reads_session(self, nl, parsed) -> bool:
        """Whether a symbol *parsed* reaches, through the rules and the
        constraints of the engine, is defined in a scope above the
        engine's: the session's, or that of its submission. A session with
        constraints of its own reads them in every query."""
        session = self.session_symbols(nl)
        if session is None or _CONSTRAINTS in session:
            return True
        if not session:
            return False
        symbol_table, seen, to_read = nl.program_ir.symbol_table, set(), [
            parsed]
        while to_read:
            for symbol in _symbols(to_read.pop()):
                if symbol in session:
                    return True
                if symbol not in seen:
                    seen.add(symbol)
                    value = symbol_table.get(symbol)
                    if value is not None and not isinstance(value, Constant):
                        to_read.append(value)  # its rules
                    to_read.extend(self._constraints.get(symbol, ()))
        return False

    # -- The files --------------------------------------------------------

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.arrow"

    def get(self, key: str) -> Optional[Kept]:
        """Return the result kept as *key*, now the most recently used, or
        None."""
        path = self._path(key)
        try:
            kept = self._read(path)
            os.utime(path)
        except Exception:  # none, removed meanwhile, or damaged
            return None
        return kept

    def put(self, key: str, frame, notes=()) -> bool:
        """
        Keep the rows *frame* of a result, with the *notes* its run printed,
        as *key*; return whether it is kept: not if Arrow does not hold its
        values as they are (a column of Python objects, or one that comes
        back of another type), or if it alone is above the limit. The file
        must read back equal to what was written.

        Arrow does it all, without a copy of the rows in pandas: copies
        made and freed here left the heap of a session hundreds of
        megabytes larger after the next queries.
        """
        import pyarrow as pa

        path = self._path(key)
        try:
            if len(frame) and any(
                    dtype == object for dtype in frame.dtypes):
                return False
            table = pa.Table.from_pandas(frame, preserve_index=False)
            empty = table.slice(0, 0).to_pandas()
            if list(empty.columns) != list(frame.columns) or \
                    list(empty.dtypes) != list(frame.dtypes):
                return False
            table = table.replace_schema_metadata({
                **(table.schema.metadata or {}),
                b"neurolang_notes": json.dumps(list(notes)).encode(),
            })
            if table.nbytes > self.most:
                return False
            engine_cache._write(path, lambda file: self._write(file, table))
            same = self._table(path).equals(table, check_metadata=True)
        except Exception:
            same = False
        if not same:
            path.unlink(missing_ok=True)
            return False
        self._evict()
        return path.exists()

    @staticmethod
    def _write(file, table) -> None:
        import pyarrow.ipc as ipc

        with ipc.new_file(file, table.schema) as writer:
            writer.write_table(table)

    @staticmethod
    def _table(path: Path):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        with pa.OSFile(str(path)) as file:  # its rows writable, not mapped
            return ipc.open_file(file).read_all()

    @classmethod
    def _read(cls, path: Path) -> Kept:
        table = cls._table(path)
        notes = json.loads(
            (table.schema.metadata or {}).get(b"neurolang_notes", b"[]")
        )
        return Kept(table.to_pandas(), notes)

    def _evict(self) -> None:
        """Remove the least recently used files until they take no more
        than the limit."""
        files = []
        for path in self.directory.glob("*.arrow"):
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime_ns, stat.st_size, path))
        total = sum(size for _, size, _ in files)
        for _, size, path in sorted(files):
            if total <= self.most:
                break
            path.unlink(missing_ok=True)
            total -= size


def attach(nl, engine_name: str, resolution, data_dir, directory,
           most_mb: float, light: bool = False) -> Optional[ResultCache]:
    """
    Give the engine *nl* its :class:`ResultCache` (``nl.result_cache``), in
    *directory* within *most_mb* megabytes (none if 0), unless it has one:
    an engine forked into a session keeps that of its build.
    """
    cache = getattr(nl, "result_cache", None)
    if cache is None and most_mb and directory:
        engine = engine_digest(engine_name, resolution, data_dir, light,
                               skip=(Path(directory).expanduser(),))
        cache = nl.result_cache = ResultCache(nl, engine, directory, most_mb)
    return cache
