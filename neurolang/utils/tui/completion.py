"""
Completion of the interactive REPL input.

The predicates offered are read from the engine each time a completion
is requested (:func:`~.catalog.predicate_catalog`), so that the
relations and the rules added during the session, e.g. with ``.load``,
are offered as soon as they exist. Inside the arguments of a predicate,
the names of its columns and the variables of the rule are offered. The
Datalog grammar adds its keywords and operators, after the predicates;
its placeholders (``<integer>``…) and punctuation are never offered.
"""

import re
from typing import Callable, Optional

from prompt_toolkit.completion import Completer, Completion

from neurolang.frontend.datalog.standard_syntax import (
    AGGREGATION_FUNCS,
    strip_comment,
)
from neurolang.utils.interactive_parsing import LarkCompleter

from .commands import COMMANDS
from . import squall_sorts
from .squall_completion import (
    SquallGrammarOptions,
    allowed_terminals,
    current_sentence,
)
from .value_completion import (
    open_quote,
    squall_quoted_column,
    tuple_completions,
    value_completions,
    value_source,
)


# Kinds of predicates offered after the relations and rules.
_LATE_KINDS = frozenset({"function"})


_IDENTIFIER_END = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# The grammar's operators worth offering; other punctuation is not.
GRAMMAR_OPERATORS = frozenset({":-", "~", "//", "::", ":=", "-:"})
# The grammar's words not worth offering.
GRAMMAR_NOISE = frozenset({"lambda", "st"})
# The words of the Datalog grammar, which cannot name variables.
KEYWORDS = frozenset({
    "AGGREGATE", "EXISTS", "MARG", "PROB", "SUCC", "ans", "exists",
    "False", "True", "lambda", "st",
})


def identifier_prefix(text: str) -> str:
    """Return the identifier being typed at the end of *text*."""
    match = _IDENTIFIER_END.search(text)
    return match.group(0) if match else ""


def argument_context(text: str):
    """
    Return the predicate whose arguments end *text*, and the position.

    The result is ``(name, position)``, *position* counting the
    arguments from 0, or None when *text* does not end inside the
    arguments of a predicate.
    """
    stack = []
    for match in re.finditer(r"[(),]|[A-Za-z_][A-Za-z0-9_]*\s*\(", text):
        token = match.group(0)
        if token == ")":
            if stack:
                stack.pop()
        elif token == ",":
            if stack:
                stack[-1][1] += 1
        else:
            stack.append([token.rstrip("( \t\n"), 0])
    if not stack or not stack[-1][0]:
        return None
    return tuple(stack[-1])


def rule_variables(text: str) -> list:
    """Return the names of the variables in *text*, in order."""
    calls = {match.start(1) for match in _CALL.finditer(text)}
    names = [
        match.group(0) for match in _WORD.finditer(text)
        if match.start() not in calls and match.group(0) not in KEYWORDS
    ]
    return list(dict.fromkeys(names))


def is_offered_grammar_token(value: str) -> bool:
    """Return True if the grammar token *value* is worth offering."""
    if value in GRAMMAR_OPERATORS:
        return True
    return bool(_WORD.fullmatch(value)) and value not in GRAMMAR_NOISE


def argument_candidates(text, prefix, catalog, context) -> list:
    """
    Return the words to offer in the arguments of a predicate.

    They are the column of the predicate of *context* (as
    :func:`argument_context` returns it) at the cursor's position, then
    the other variables of the rule, typed in *text* before *prefix*, and
    columns of the predicate; in the head of a rule, the aggregation
    functions come last. The result is a list of ``(word, meta)``.
    """
    name, position = context
    info = catalog.get(name)
    columns = list(getattr(info, "columns", None) or ())
    candidates = _entity_candidates(info, position)
    candidates += [
        (word, f"column {position + 1} of {name}")
        for word in columns[position:position + 1]
    ]
    candidates += [
        (word, "variable")
        for word in rule_variables(text[:len(text) - len(prefix)])
    ]
    candidates += [(word, f"column of {name}") for word in columns]
    if ":-" not in text and "←" not in text:
        candidates += [
            (word, "aggregation") for word in sorted(AGGREGATION_FUNCS)
        ]
    return candidates


def _entity_candidates(info, position: int) -> list:
    """
    Return the columns of the entity starting at column *position*.

    When the predicate *info* declares that an entity of several
    columns, as a voxel ``i, j, k``, starts there, they are offered
    together, as ``(word, meta)``.
    """
    start = 0
    for argument in getattr(info, "arguments", None) or ():
        if start == position and len(argument.columns) > 1:
            meta = f"{squall_sorts.argument_text(argument)} of {info.name}"
            return [(", ".join(argument.columns), meta)]
        start += len(argument.columns)
    return []


def wants_completion(text: str, prefix: str, complete_event) -> bool:
    """
    Return True if the completion of *text* is worth computing.

    While typing, it is only when a word, *prefix*, is being typed;
    Tab (or no *complete_event*) also completes after a space or an
    opening bracket, but not after a number.
    """
    if prefix:
        return True
    requested = getattr(complete_event, "completion_requested", True)
    return requested and not text[-1:].isdigit()


def _keyword_completions(words, prefix: str):
    """Yield the keywords of *words* starting with *prefix*."""
    for word in words:
        if word.startswith(prefix):
            yield Completion(
                word, start_position=-len(prefix), display_meta="keyword"
            )


class NeuroLangReplCompleter(Completer):
    """
    Complete dot-commands, predicates, their columns and the grammar.

    *predicates* returns the :class:`~.catalog.PredicateInfo` of the
    engine's predicates, by name; it is called at each completion.
    Datalog input is completed with the LALR grammar's *lark_completer*,
    SQUALL input with the Earley grammar's *squall_grammar* (see
    :mod:`~.squall_completion`). Without them, or where the input is
    invalid, the predicates and *squall_keywords* starting with the word
    being typed are offered. *is_squall* tells whether the input is SQUALL.
    The predicates named in *hidden* are not offered in SQUALL, and
    their declared arguments choose the nouns and verbs offered (see
    :mod:`~.squall_sorts`).

    Inside a quoted string, the values of the column it fills are
    offered, as *column_values* gives them (see
    :class:`~.value_completion.ColumnValues`); inside a SQUALL tuple, the
    variables of the noun's next columns.

    While typing, completions are offered only for a word being typed;
    Tab also offers them after a space or an opening bracket.
    """

    def __init__(
        self,
        lark_completer: Optional[LarkCompleter] = None,
        predicates: Optional[Callable[[], dict]] = None,
        squall_keywords: Optional[list] = None,
        is_squall: Optional[Callable[[], bool]] = None,
        squall_grammar: Optional[SquallGrammarOptions] = None,
        hidden: frozenset = frozenset(),
        column_values: Optional[Callable[[str, int], tuple]] = None,
    ):
        """Set up the completion sources."""
        self._column_values = column_values
        self._hidden = hidden
        self._lark = lark_completer
        self._predicates = predicates or dict
        self._squall_keywords = squall_keywords or []
        self._is_squall = is_squall or (lambda: False)
        self._squall_grammar = squall_grammar

    def get_completions(self, document, complete_event):
        """Yield the completions of the text before the cursor."""
        text = document.text_before_cursor
        if text.startswith("."):
            yield from self._command_completions(text)
            return
        quote = open_quote(text)
        if quote is not None:
            yield from self._quoted_completions(text, quote, complete_event)
            return
        prefix = identifier_prefix(text)
        if wants_completion(text, prefix, complete_event):
            yield from self._completions(text, prefix, self._predicates())

    def _completions(self, text: str, prefix: str, catalog: dict):
        """Yield the completions of *text*, outside of strings."""
        variables = (
            tuple_completions(text, self._lowered(catalog))
            if self._is_squall() else None
        )
        context = argument_context(text)
        if variables is not None:
            yield from variables
        elif self._is_squall() or self._lark is None:
            yield from self._squall_completions(text, prefix, catalog)
        elif context is not None:
            yield from self._argument_completions(
                text, prefix, catalog, context
            )
        else:
            yield from self._datalog_completions(text, prefix, catalog)

    def _lowered(self, catalog: dict) -> dict:
        """Return the predicates of *catalog* SQUALL offers, lowercased."""
        return {
            name.lower(): info for name, info in catalog.items()
            if name not in self._hidden
        }

    def _quoted_column(self, before: str, catalog: dict):
        """Return the predicate and column a string after *before* fills."""
        if self._is_squall():
            return squall_quoted_column(before, self._lowered(catalog))
        context = argument_context(before)
        info = catalog.get(context[0]) if context else None
        return (info, context[1]) if info else None

    def _quoted_completions(self, text: str, quote, complete_event):
        """
        Yield the values of the column the string *quote* opens fills.

        While typing, they are offered once a character is typed.
        """
        typed = text[quote.start + 1:]
        requested = getattr(complete_event, "completion_requested", True)
        if self._column_values is None or not (typed or requested):
            return
        catalog = self._predicates()
        target = self._quoted_column(text[:quote.start], catalog)
        source = value_source(*target, catalog) if target else None
        if source is None:
            return
        name, position, meta = source
        yield from value_completions(
            self._column_values(name, position), typed, quote.quote, meta
        )

    @staticmethod
    def _command_completions(text: str):
        """Yield the dot-commands starting with *text*."""
        for name in COMMANDS:
            if name.startswith(text):
                yield Completion(name, start_position=-len(text))

    @staticmethod
    def _predicate_completions(catalog: dict, prefix: str):
        """Yield the predicates starting with *prefix*, functions last."""
        infos = sorted(
            (info for name, info in catalog.items()
             if name.startswith(prefix)),
            key=lambda info: (info.kind in _LATE_KINDS, info.name),
        )
        for info in infos:
            yield Completion(
                info.name, start_position=-len(prefix),
                display_meta=info.meta(),
            )

    def _squall_options(self, text: str, prefix: str):
        """Return what SQUALL accepts before *prefix*, or None."""
        if self._squall_grammar is None:
            return None
        try:
            return self._squall_grammar.options(text[:len(text) - len(prefix)])
        except Exception:
            return None

    def _squall_completions(self, text: str, prefix: str, catalog: dict):
        """
        Yield the predicates and keywords SQUALL accepts before *prefix*.

        Where a name is allowed, the nouns or the verbs that fit (see
        :func:`~.squall_sorts.name_slot`) are offered; after a noun, the
        phrases naming its columns. If the sentence is invalid, yield
        the predicates and the *squall_keywords* starting with *prefix*.
        """
        catalog = self._lowered(catalog)
        options = self._squall_options(text, prefix)
        if options is None:
            if prefix:
                yield from _squall_names(catalog.values(), prefix, "")
                yield from _keyword_completions(
                    self._squall_keywords, prefix
                )
            return
        before = text[:len(text) - len(prefix)]
        slot = squall_sorts.name_slot(before, catalog)
        yield from _phrase_completions(before, prefix, slot, catalog)
        if options.names:
            yield from _slot_completions(slot, catalog, prefix)
        yield from _keyword_completions(options.keywords, prefix)
        if options.end and not prefix:
            yield Completion(".", display_meta="end of the sentence")

    @staticmethod
    def _argument_completions(text, prefix, catalog, context):
        """Yield the :func:`argument_candidates` starting with *prefix*."""
        seen = set()
        for word, meta in argument_candidates(text, prefix, catalog, context):
            if word.startswith(prefix) and word not in seen:
                seen.add(word)
                yield Completion(
                    word, start_position=-len(prefix), display_meta=meta
                )

    def _grammar_options(self, text: str):
        """
        Return the tokens the Datalog grammar accepts after *text*.

        The result is ``(identifier, values)``: whether an identifier
        is accepted and the other tokens, or None if *text* is invalid.
        """
        try:
            result = self._lark.complete(text)
        except Exception:
            return None
        if not isinstance(result.token_options, dict):
            # The lexer rejected a character: only the terminal names
            # are given, so *text* is treated as invalid.
            return None
        values = {
            value
            for container in result.token_options.values()
            for value in container.get("values", ())
        }
        return "<cmd_identifier>" in values, values

    def _datalog_completions(self, text: str, prefix: str, catalog: dict):
        """
        Yield the predicates and grammar tokens accepted after *text*.

        Predicates are offered where the grammar accepts an identifier,
        or, if the text before *prefix* is invalid, after any word.
        Nothing is offered inside a ``%`` comment, and the comments
        of earlier lines are left out.
        """
        lines = text[:len(text) - len(prefix)].split("\n")
        if strip_comment(lines[-1]) != lines[-1]:
            return
        options = self._grammar_options(
            "\n".join(strip_comment(line) for line in lines)
        )
        identifier, values = options or (bool(prefix), set())
        if identifier:
            yield from self._predicate_completions(catalog, prefix)
        tokens = sorted(
            (
                value for value in values
                if value.startswith(prefix) and is_offered_grammar_token(value)
            ),
            key=lambda value: (not value[:1].isalpha(), value),
        )
        for value in tokens:
            meta = "keyword" if value[:1].isalpha() else "operator"
            yield Completion(
                value, start_position=-len(prefix), display_meta=meta
            )


def _matches(word: str, prefix: str) -> bool:
    """Return True if *word* starts with *prefix*, ignoring the case."""
    return word.lower().startswith(prefix.lower())


def _ranked(infos) -> list:
    """Return the predicates *infos* sorted by name, functions left out."""
    return sorted(
        (info for info in infos if info.kind not in _LATE_KINDS),
        key=lambda info: info.name,
    )


def _noun_meta(info) -> str:
    """Return the meta of the noun *info*: its entities and its kind."""
    if not info.arguments:
        return info.meta()
    meta = squall_sorts.entities_text(info.arguments)
    phrases = squall_sorts.noun_phrases(info)
    if len(info.arguments) == 1 and phrases:
        meta += f" · {phrases[0]}"
    return f"{meta} · {info.kind}"


def _verb_meta(info) -> str:
    """Return the meta of the verb *info*: ``study → voxel(i, j, k)``."""
    if not info.arguments:
        return info.meta()
    entities = squall_sorts.entities_text(info.arguments, " → ")
    return f"{entities} · {info.kind}"


def _squall_names(infos, prefix: str, kind: str):
    """
    Yield the predicates *infos* starting with *prefix*, as a *kind*.

    Nouns are capitalised; verbs, and the names of an unknown *kind*,
    are offered as they are, the latter with the functions, last.
    """
    infos = list(infos)
    late = [info for info in infos if info.kind in _LATE_KINDS]
    for info in _ranked(infos) + (late if not kind else []):
        if not _matches(info.name, prefix):
            continue
        if kind == "noun":
            text = squall_sorts.capitalised(info.name, prefix)
            meta = _noun_meta(info)
        else:
            text = info.name
            meta = _verb_meta(info) if kind == "verb" else info.meta()
        yield Completion(
            text, start_position=-len(prefix), display_meta=meta
        )


def _sort(catalog: dict, noun: Optional[str]) -> Optional[str]:
    """Return the sort of the entities of *noun*, if it is known."""
    info = catalog.get(noun) if noun else None
    return squall_sorts.sort_of(info) if info else None


def _slot_completions(slot, catalog: dict, prefix: str):
    """Yield the names fitting *slot* (see :func:`name_slot`)."""
    if slot.kind == "noun":
        yield from _squall_names(catalog.values(), prefix, "noun")
    elif slot.kind == "verb":
        subject = _sort(catalog, slot.subject)
        object_ = _sort(catalog, slot.object)
        verbs = [
            info for info in catalog.values()
            if squall_sorts.is_verb(info, subject, object_)
        ]
        yield from _squall_names(verbs, prefix, "verb")
        if slot.noun is None:
            yield from _object_clauses(catalog, subject, prefix)
    elif slot.kind not in ("after_noun", "determiner"):
        yield from _squall_names(catalog.values(), prefix, "")


def _object_clause(info, sort: str, catalog: dict) -> Optional[str]:
    """
    Return the object clause of the verb *info*, if its object is *sort*.

    ``a Study mentions``: the subject is the plainest noun of the verb's
    first entity.
    """
    arguments = info.arguments or ()
    if len(arguments) != 2 or arguments[1].sort != sort:
        return None
    noun = squall_sorts.plainest_noun(arguments[0].sort, catalog.values())
    if noun is None:
        return None
    noun = squall_sorts.capitalised(noun)
    return f"{squall_sorts.article(noun)} {noun} {info.name}"


def _object_clauses(catalog: dict, sort: Optional[str], prefix: str):
    """Yield the object clauses (``a Study mentions``) on a *sort*."""
    if sort is None:
        return
    for info in _ranked(catalog.values()):
        text = _object_clause(info, sort, catalog)
        if text and _matches(text, prefix):
            yield Completion(
                text, start_position=-len(prefix),
                display_meta=f"object clause · {_verb_meta(info)}",
            )


def _noun_phrase_texts(info) -> list:
    """Return the phrases naming the columns of the noun *info*."""
    phrases = squall_sorts.noun_phrases(info)
    columns = info.columns or ()
    if len(columns) > 1:
        phrases.append(
            "(" + "; ".join(f"?{column}" for column in columns) + ")"
        )
    return phrases


def _phrase_completions(text: str, prefix: str, slot, catalog: dict):
    """
    Yield the phrases naming the columns of the noun ending *text*.

    ``in 3D`` after ``every Voxel``, ``with Probability`` after a
    probabilistic noun, and the tuple of its columns; only those the
    grammar accepts there are offered.
    """
    if slot.kind != "after_noun" or slot.noun not in catalog:
        return
    sentence = current_sentence(text)
    for phrase in _noun_phrase_texts(catalog[slot.noun]):
        if not _matches(phrase, prefix):
            continue
        if allowed_terminals(f"{sentence}{phrase} ") is None:
            continue
        yield Completion(
            phrase, start_position=-len(prefix),
            display_meta=f"columns of {slot.noun}",
        )
