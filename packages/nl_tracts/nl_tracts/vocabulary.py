"""The vocabulary of tractography: Datalog rules over the relations
(DESIGN.md, 16).

The rules are in ``vocabulary.dl``, a Datalog file whose ``%`` comments
document each predicate (``traverses(x, term, h)``, ``endpoint_in(x, e,
term, h)``, ``anterior_of(x, term, h)``...). The engines of
:mod:`nl_tracts.frontend` have them; :func:`prelude` gives the text to
another engine.
"""

import re
from importlib.resources import files
from typing import Tuple

__all__ = ["predicates", "prelude"]


def prelude() -> str:
    """The rules of the vocabulary, as a Datalog program."""
    return files("nl_tracts").joinpath("vocabulary.dl").read_text()


def predicates() -> Tuple[str, ...]:
    """The predicates the vocabulary defines, in the order of the file."""
    heads = re.findall(r"^(\w+)\(", prelude(), flags=re.MULTILINE)
    return tuple(dict.fromkeys(heads))
