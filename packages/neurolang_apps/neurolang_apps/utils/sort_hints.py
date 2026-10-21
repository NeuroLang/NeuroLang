"""
Notes on the entities of several columns, for the errors on them.

An engine declares the sorts of its predicates' arguments (see
:func:`~neurolang.utils.engine_registry.get_signatures`). A voxel spans
three columns, ``i, j, k``, so the noun ``Voxel`` reads ``every Voxel in
3D`` in SQUALL, and ``reports``, relating a study and a voxel, reads
``a Study reports a Voxel in 3D``. When a query fails on the number of
columns of such a predicate, :func:`sort_notes` tells how to write it.
"""

import difflib
import re
from typing import Optional

from nl_expr.exceptions import SymbolNotFoundError

from . import engine_registry

_ARITY_MESSAGES = (
    "same arity", "wrong number of arguments",
)
_NOT_FOUND = re.compile(r"Symbol not found\s+(\w+)")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def sort_notes(exc, program: str, engine_name: str, squall: bool) -> list:
    """
    Return the notes on the predicates of several columns *exc* concerns.

    *exc* concerns the predicate it names (as the frontend's arity
    error, through its ``predicate`` attribute), the declared predicate
    closest to an unknown symbol, or, for an error on the number of
    columns, the predicates of *program*. Only the predicates of the
    engine *engine_name* with an argument of several columns get a note,
    written for SQUALL if *squall*, or else for Datalog.
    """
    try:
        signatures = engine_registry.get_signatures(engine_name)
    except ValueError:
        return []
    names = [
        name for name in dict.fromkeys(_concerned(exc, program, signatures))
        if any(len(arg.columns) > 1 for arg in signatures.get(name, ()))
    ]
    return [_note(name, signatures, squall) for name in names]


def _concerned(exc, program: str, signatures: dict) -> list:
    """Return the names of the predicates *exc* is about, if known."""
    predicate = getattr(exc, "predicate", None)
    if predicate is not None:
        return [getattr(predicate, "name", predicate)]
    if isinstance(exc, SymbolNotFoundError):
        return _closest_declared(str(exc), signatures)
    if any(message in str(exc) for message in _ARITY_MESSAGES):
        words = (word.lower() for word in _WORD.findall(program))
        return [word for word in words if word in signatures]
    return []


def _closest_declared(message: str, signatures: dict) -> list:
    """Return the declared predicate closest to the unknown one, if any."""
    match = _NOT_FOUND.search(message)
    if match is None:
        return []
    name = match.group(1).lower()
    return difflib.get_close_matches(name, signatures, n=1, cutoff=0.8)


def _entity(argument) -> str:
    """Return ``sort(col, ...)``, or the sort of a column named after it."""
    if argument.columns == (argument.sort,):
        return argument.sort
    return f"{argument.sort}({', '.join(argument.columns)})"


def _note(name: str, signatures: dict, squall: bool) -> str:
    """Return the note on the predicate *name* of several columns."""
    arguments = signatures[name]
    if len(arguments) == 1:
        argument = arguments[0]
        text = (
            f"{name} is {_entity(argument)}, a {len(argument.columns)}"
            f"-column entity"
        )
    else:
        entities = [_entity(argument) for argument in arguments]
        text = f"{name} relates {' to '.join(entities)}"
    example = (
        _squall_example(name, arguments, signatures) if squall
        else f"{name}({', '.join(c for a in arguments for c in a.columns)})"
    )
    return f"note: {text}" + (f": write {example}" if example else "")


def _noun(sort: str, signatures: dict) -> Optional[str]:
    """Return the plainest predicate holding the entities of *sort*."""
    nouns = [
        name for name, arguments in signatures.items()
        if len(arguments) == 1 and arguments[0].sort == sort
    ]
    return min(nouns, key=lambda name: (len(name), name)) if nouns else None


def _phrase(noun: str, argument, determiner: str = "a") -> str:
    """Return the noun phrase of *noun*: ``a Voxel in 3D``."""
    word = noun[:1].upper() + noun[1:]
    if determiner == "a" and word[:1] in "AEIOU":
        determiner = "an"
    width = len(argument.columns)
    return f"{determiner} {word}" + (f" in {width}D" if width > 1 else "")


def _squall_example(name: str, arguments, signatures: dict) -> str:
    """
    Return how to write the predicate *name* in SQUALL, or "".

    A noun reads ``every Voxel in 3D`` or with the tuple of its columns,
    and a verb ``a Study reports a Voxel in 3D``, when its entities have
    nouns.
    """
    if len(arguments) == 1:
        variables = "; ".join(f"?{c}" for c in arguments[0].columns)
        noun = name[:1].upper() + name[1:]
        return (
            f'"{_phrase(name, arguments[0], "every")}" or '
            f'"every {noun} ({variables})"'
        )
    nouns = [_noun(argument.sort, signatures) for argument in arguments]
    if len(arguments) != 2 or None in nouns:
        return ""
    subject, object_ = (
        _phrase(noun, argument) for noun, argument in zip(nouns, arguments)
    )
    return f'"{subject} {name} {object_}"'
