"""
Completion of SQUALL from the sorts of the engine's predicates.

An engine declares the arguments of its predicates (see
:func:`~neurolang.utils.engine_registry.get_signatures`): the entities
they relate, each of a sort spanning one or more columns. A voxel
``(i, j, k)`` is one entity, so ``reports(study_id, i, j, k)`` relates a
study and a voxel. The sorts tell which words fit where a SQUALL
sentence takes a name (:func:`name_slot`):

* a noun, after a determiner: the predicates, capitalised;
* a verb, after ``that``: the predicates relating two entities, the
  first of the sort of the noun the clause is about (``every Study that
  mentions``), or, for the predicates whose *second* entity is of that
  sort, the object clause (``every Term that a Study mentions``);
* after a noun whose entity spans N > 1 columns: ``in ND`` (``every
  Voxel in 3D``), with ``with Probability`` for a probabilistic one.

The inverse verb form ``~verb`` is never offered: the parser reads
``every Term that ~mentions a Study`` as ``mentions(term, study)``,
which fails on the engine (and on a multi-column entity, such as a
voxel, it does not translate). The object clause reads the same.
"""

import re
from typing import NamedTuple, Optional

DETERMINERS = frozenset({"every", "all", "a", "an", "some", "no", "the"})
RELATIVES = frozenset({"that", "which", "who", "whom"})

_WORD = re.compile(r"'[^']*'?|[?@]?[A-Za-z_][A-Za-z0-9_]*|\S")


class Slot(NamedTuple):
    """
    What a SQUALL sentence takes next, where a name is allowed.

    *kind* is ``noun``, ``verb``, ``after_noun`` (a noun was just
    typed), ``determiner`` (after ``obtain``) or ``""`` (unknown). *subject* is the noun a verb is about,
    *object* the noun of the object of an object clause (``every Voxel
    that a Study``) and *noun* the noun just typed.
    """

    kind: str
    subject: Optional[str] = None
    object: Optional[str] = None
    noun: Optional[str] = None


def sentence_words(text: str) -> list:
    """Return the words, labels, strings and symbols of *text*."""
    return _WORD.findall(text)


def _last_noun(words, nouns) -> Optional[str]:
    """Return the last of *words* naming a noun after a determiner."""
    pairs = zip(words[-2::-1], words[::-1])
    for before, word in pairs:
        if word.lower() in nouns and before.lower() in DETERMINERS:
            return word.lower()
    return None


_WORD_SLOTS = {"obtain": Slot("determiner")}
_WORD_SLOTS.update({word: Slot("noun") for word in DETERMINERS})


def name_slot(text: str, nouns) -> Slot:
    """
    Return the :class:`Slot` at the end of the SQUALL *text*.

    *nouns* are the lowercase names of the predicates. A noun is a
    name after a determiner; after ``that a Study`` comes the verb of
    the object clause, whose subject is the study.
    """
    words = sentence_words(text)
    last = words[-1].lower() if words else ""
    if last in _WORD_SLOTS:
        return _WORD_SLOTS[last]
    if last in RELATIVES:
        return Slot("verb", subject=_last_noun(words[:-1], nouns))
    return _slot_after_noun(words, nouns)


def _slot_after_noun(words, nouns) -> Slot:
    """Return the :class:`Slot` after *words* ending with a noun, if so."""
    lowered = [word.lower() for word in words[-3:]]
    if len(lowered) < 2 or lowered[-2] not in DETERMINERS:
        return Slot("")
    noun = lowered[-1]
    if noun not in nouns:
        return Slot("")
    if len(lowered) == 3 and lowered[0] in RELATIVES:
        return Slot(
            "verb", subject=noun, object=_last_noun(words[:-3], nouns),
            noun=noun,
        )
    return Slot("after_noun", noun=noun)


def argument_text(argument, columns: bool = False) -> str:
    """
    Return ``sort`` or, for several columns, ``sort(col, ...)``.

    With *columns*, a column not named after its sort is shown too:
    ``study(study_id)``.
    """
    if argument.columns == (argument.sort,) or (
        len(argument.columns) == 1 and not columns
    ):
        return argument.sort
    return f"{argument.sort}({', '.join(argument.columns)})"


def entities_text(arguments, separator: str = ", ") -> str:
    """Return the entities of *arguments*, joined by *separator*."""
    return separator.join(argument_text(argument) for argument in arguments)


def sort_of(info) -> Optional[str]:
    """Return the sort of the entity of the noun *info*, if it has one."""
    arguments = info.arguments or ()
    return arguments[0].sort if len(arguments) == 1 else None


def width(info) -> int:
    """Return the number of columns of the noun *info* (1 if unknown)."""
    arguments = info.arguments or ()
    if arguments:
        return sum(len(argument.columns) for argument in arguments)
    return info.arity or 1


def is_verb(info, subject: Optional[str], object_: Optional[str]) -> bool:
    """
    Return True if *info* is a verb from the sort *subject* to *object_*.

    It relates two entities, the first of the sort *subject* and the
    second of the sort *object_*, unless they are None. A predicate whose
    entities are unknown is a verb if it has two columns or more.
    """
    arguments = info.arguments
    if arguments is None:
        return (info.arity or 0) >= 2
    if len(arguments) != 2:
        return False
    first, second = (argument.sort for argument in arguments)
    return subject in (None, first) and object_ in (None, second)


def plainest_noun(sort: str, infos) -> Optional[str]:
    """
    Return the plainest noun of the entities of *sort*: the shortest.

    ``study`` rather than ``study_with_peaks``; probabilistic nouns are
    left out.
    """
    nouns = [
        info.name for info in infos
        if sort_of(info) == sort and not info.probabilistic
    ]
    return min(nouns, key=lambda name: (len(name), name)) if nouns else None


def noun_phrases(info) -> list:
    """
    Return the phrases following the noun *info* that name its columns.

    ``in ND`` when it spans N > 1 columns, ``with Probability`` when its
    probability is inferred by a rule, both (``in 3D with Probability``)
    when both. The probability of a probabilistic fact or choice cannot
    be obtained.
    """
    columns = width(info)
    dimension = f"in {columns}D" if columns > 1 else ""
    phrases = [dimension] if dimension else []
    if info.probabilistic and info.kind != "probabilistic":
        phrases.append(f"{dimension} with Probability".strip())
    return phrases


def capitalised(name: str, prefix: str = "") -> str:
    """Return the noun *name* capitalised, unless *prefix* is lowercase."""
    if prefix[:1].islower():
        return name
    return name[:1].upper() + name[1:]


def article(noun: str) -> str:
    """Return the indefinite article of *noun*."""
    return "an" if noun[:1].lower() in "aeiou" else "a"
