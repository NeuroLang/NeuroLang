"""
Completion of SQUALL from its grammar.

The SQUALL parser is an Earley parser, which cannot be fed tokens
interactively, but parsing the current sentence followed by a character
that no terminal accepts fails *at the end* of the sentence with the set
of terminals allowed there (:meth:`SquallGrammarOptions.options`). Their
literal words are offered as keywords, and the engine's predicates where
a name is allowed. When the sentence fails before its end, there are no
options and the completer falls back on :func:`squall_keywords`.
"""

import functools
import re
from typing import NamedTuple, Optional

from lark.exceptions import UnexpectedInput

# The terminals of the names of nouns and verbs, i.e. predicates.
NAME_TERMINALS = frozenset({
    "LOWER_NAME", "LOWER_NAME_QUOTED", "NAME", "CNAME",
})

# A character that no terminal accepts, appended to make the parser
# report the terminals allowed at the end of the text.
_SENTINEL = "\x01"

_KEYWORD = re.compile(r"[A-Za-z]{2,}|a")
_ALTERNATIVES = re.compile(r"\(\?:([A-Za-z|]+)\)")
_SENTENCE_END = re.compile(r"\.(?=\s)")


def terminal_words(pattern) -> list:
    """
    Return the keywords a terminal *pattern* of the grammar matches.

    A string pattern gives its word, a regular expression made of
    alternative words, as ``(?:every|all)``, gives each word; other
    patterns, and punctuation, give none.
    """
    if pattern.type == "str":
        candidates = [pattern.value]
    else:
        match = _ALTERNATIVES.fullmatch(pattern.value)
        candidates = match.group(1).split("|") if match else []
    return [word for word in candidates if _KEYWORD.fullmatch(word)]


@functools.lru_cache(maxsize=None)
def _grammar():
    """Return the compiled SQUALL grammar (compiled on first use)."""
    from neurolang.frontend.datalog.squall_syntax_lark import (
        compiled_grammar,
    )

    return compiled_grammar()


@functools.lru_cache(maxsize=None)
def _terminal_patterns() -> dict:
    """Return the patterns of the SQUALL grammar's terminals, by name."""
    return {
        terminal.name: terminal.pattern for terminal in _grammar().terminals
    }


@functools.lru_cache(maxsize=None)
def squall_keywords() -> tuple:
    """Return the keywords of the SQUALL grammar, sorted."""
    words = {
        word
        for name, pattern in _terminal_patterns().items()
        if name not in NAME_TERMINALS
        for word in terminal_words(pattern)
    }
    return tuple(sorted(words))


class SquallOptions(NamedTuple):
    """
    What SQUALL accepts after a text.

    *names* tells whether a noun or a verb, i.e. a predicate, may
    come next, *keywords* are the words that may come next and *end*
    whether the sentence may end there, with a ``.``.
    """

    names: bool
    keywords: tuple
    end: bool


def current_sentence(text: str) -> str:
    """Return the sentence of *text* being typed: after the last ``.``."""
    parts = _SENTENCE_END.split(text)
    return parts[-1].lstrip() if text.rstrip()[-1:] != "." else ""


def allowed_terminals(sentence: str) -> Optional[set]:
    """
    Return the names of the terminals allowed after *sentence*.

    Return None if *sentence* is not the beginning of a SQUALL sentence.
    """
    try:
        _grammar().parse(sentence + _SENTINEL)
    except UnexpectedInput as error:
        position = getattr(error, "pos_in_stream", None)
        if position is None or position < len(sentence.rstrip()):
            return None
        return set(getattr(error, "allowed", None) or ())
    return None


def options_of(terminals: set) -> SquallOptions:
    """Return the :class:`SquallOptions` of the allowed *terminals*."""
    patterns = _terminal_patterns()
    keywords = {
        word
        for name in terminals - NAME_TERMINALS
        if name in patterns
        for word in terminal_words(patterns[name])
    }
    end = any(
        getattr(patterns.get(name), "value", None) == "."
        for name in terminals
    )
    return SquallOptions(
        bool(terminals & NAME_TERMINALS), tuple(sorted(keywords)), end
    )


class SquallGrammarOptions:
    """
    Compute what SQUALL accepts after a text, remembering the last one.

    Pressing Tab again, or typing more letters of the same word, does
    not parse the text again. The time to parse a sentence grows faster
    than its length: about 5 to 20 ms up to 80 characters, 45 to 120 ms
    around 200, so sentences longer than *parse_limit* characters are
    not parsed and the completer falls back on the keywords.
    """

    def __init__(self, parse_limit: int = 200):
        """Start with an empty cache."""
        self._last = (None, None)
        self.parse_limit = parse_limit

    def options(self, text: str) -> Optional[SquallOptions]:
        """
        Return what SQUALL accepts after *text*.

        Only the sentence being typed is parsed. Return None if it is
        not the beginning of a SQUALL sentence or is too long to parse.
        """
        sentence = current_sentence(text)
        if len(sentence) > self.parse_limit:
            return None
        if self._last[0] != sentence:
            terminals = allowed_terminals(sentence)
            options = None if terminals is None else options_of(terminals)
            self._last = (sentence, options)
        return self._last[1]
