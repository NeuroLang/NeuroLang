"""Functions annotated under postponed evaluation of annotations."""

from __future__ import annotations


def starts(prefix: str, s: str) -> bool:
    return s.startswith(prefix)


def unresolvable(x: NotAType) -> AlsoNotAType:  # noqa: F821
    return x
