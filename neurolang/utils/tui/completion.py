"""
Completion of the interactive REPL input.

The predicates offered are read from the engine each time a completion
is requested (:func:`predicate_catalog`), so that the relations and the
rules added during the session, e.g. with ``.load``, are offered as soon
as they exist. Inside the arguments of a predicate, the names of its
columns and the variables of the rule are offered. The Datalog grammar
adds its keywords and operators, after the predicates; its placeholders
(``<integer>``…) and punctuation are never offered.
"""

import re
from typing import Callable, NamedTuple, Optional

from prompt_toolkit.completion import Completer, Completion

from neurolang.datalog.basic_representation import UnionOfConjunctiveQueries
from neurolang.datalog.wrapped_collections import WrappedRelationalAlgebraSet
from neurolang.frontend.datalog.standard_syntax import AGGREGATION_FUNCS
from neurolang.utils.interactive_parsing import LarkCompleter

from .commands import COMMANDS


class PredicateInfo(NamedTuple):
    """
    A predicate offered by the completion.

    *kind* tells where it comes from: ``stored`` (EDB relation),
    ``derived`` (IDB predicate of the engine), ``session`` (rule defined
    in the session), ``loaded`` (relation loaded with ``.load``),
    ``probabilistic`` or ``function``. *columns* are the names of its
    columns, or None when they are unknown, and *arity* its number of
    columns, or None when it is unknown.
    """

    name: str
    kind: str
    columns: Optional[tuple] = None
    arity: Optional[int] = None

    def signature(self) -> str:
        """Return ``name(col, ...)``, ``name/arity`` or ``name``."""
        if self.columns:
            return f"{self.name}({', '.join(self.columns)})"
        if self.arity is not None:
            return f"{self.name}/{self.arity}"
        return self.name

    def meta(self) -> str:
        """Return the description shown next to the completion."""
        return f"{self.signature()} · {self.kind}"


# Kinds of predicates offered after the relations and rules.
_LATE_KINDS = frozenset({"function"})


def _scopes(symbol_table):
    """Yield the scopes of *symbol_table*, innermost first."""
    while symbol_table is not None:
        yield symbol_table
        symbol_table = getattr(symbol_table, "enclosing_scope", None)


def _probabilistic_symbols(program_ir) -> set:
    """Return the symbols of the probabilistic facts and choices."""
    symbols = set()
    for attribute in ("pfact_pred_symbs", "pchoice_pred_symbs"):
        symbols |= set(getattr(program_ir, attribute, ()))
    return symbols


def _rule_columns(union) -> Optional[tuple]:
    """Return the parameter names of the first rule of *union*, if any."""
    args = union.formulas[0].consequent.args
    names = tuple(getattr(arg, "name", None) for arg in args)
    if all(isinstance(name, str) for name in names):
        return names
    return None


def _relation_info(name, value, kind, columns) -> PredicateInfo:
    """Return the info of the relation *name* whose set is *value*."""
    arity = value.arity
    if kind == "probabilistic":
        arity -= 1
    columns = columns.get(name)
    if columns is not None and len(columns) != arity:
        columns = None
    return PredicateInfo(name, kind, columns and tuple(columns), arity)


def _function_arity(symbol) -> Optional[int]:
    """Return the number of parameters of the typed function *symbol*."""
    args = getattr(symbol.type, "__args__", None)
    if not args or args[0] is Ellipsis:
        return None
    return len(args) - 1


class _CatalogBuilder:
    """Collect the :class:`PredicateInfo` of the symbols of an engine."""

    def __init__(self, nl, columns: dict, loaded: dict):
        """Set up the collection on *nl* with the known *columns*."""
        self.program_ir = nl.program_ir
        self.columns = {**columns, **loaded}
        self.loaded = loaded
        self.probabilistic = _probabilistic_symbols(self.program_ir)
        self.protected = getattr(self.program_ir, "protected_keywords", ())

    def info(self, symbol, value, session: bool):
        """Return the info of *symbol* defined as *value*, or None."""
        name = getattr(symbol, "name", None)
        if not name or name.startswith("_") or symbol in self.protected:
            return None
        inner = getattr(value, "value", None)
        if isinstance(inner, WrappedRelationalAlgebraSet):
            return _relation_info(
                name, inner, self._relation_kind(symbol), self.columns
            )
        if symbol.type is UnionOfConjunctiveQueries:
            columns = _rule_columns(value)
            arity = len(value.formulas[0].consequent.args)
            kind = "session" if session else "derived"
            return PredicateInfo(name, kind, columns, arity)
        if callable(inner):
            return PredicateInfo(
                name, "function", None, _function_arity(symbol)
            )
        return None

    def _relation_kind(self, symbol) -> str:
        """Return the kind of the relation *symbol*."""
        if symbol in self.probabilistic:
            return "probabilistic"
        if symbol.name in self.loaded:
            return "loaded"
        return "stored"


def predicate_catalog(nl, columns=None, loaded=None) -> dict:
    """
    Return the predicates defined in the engine *nl*, by name.

    They are its relations, its rules (those of the innermost scope,
    when the session opened one, being the session's), its probabilistic
    facts and choices and its functions. *columns* maps relation names to
    their declared columns (e.g. from the engine's configuration) and
    *loaded* those of the relations loaded with ``.load``.
    """
    builder = _CatalogBuilder(nl, columns or {}, loaded or {})
    scopes = list(_scopes(nl.program_ir.symbol_table))
    catalog = {}
    for depth, scope in enumerate(scopes):
        session = depth == 0 and len(scopes) > 1
        for symbol, value in scope._symbols.items():
            if getattr(symbol, "name", None) in catalog:
                continue
            info = builder.info(symbol, value, session)
            if info is not None:
                catalog[info.name] = info
    return catalog


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
    columns = list(getattr(catalog.get(name), "columns", None) or ())
    candidates = [
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


class NeuroLangReplCompleter(Completer):
    """
    Complete dot-commands, predicates, their columns and the grammar.

    *predicates* returns the :class:`PredicateInfo` of the engine's
    predicates, by name; it is called at each completion. Datalog input
    is completed with the LALR grammar's *lark_completer*; SQUALL input,
    or any input without *lark_completer*, with the *squall_keywords*.
    *is_squall* tells whether the input is SQUALL.

    While typing, completions are offered only for a word being typed;
    Tab also offers them after a space or an opening bracket.
    """

    def __init__(
        self,
        lark_completer: Optional[LarkCompleter] = None,
        predicates: Optional[Callable[[], dict]] = None,
        squall_keywords: Optional[list] = None,
        is_squall: Optional[Callable[[], bool]] = None,
    ):
        """Set up the completion sources."""
        self._lark = lark_completer
        self._predicates = predicates or dict
        self._squall_keywords = squall_keywords or []
        self._is_squall = is_squall or (lambda: False)

    def get_completions(self, document, complete_event):
        """Yield the completions of the text before the cursor."""
        text = document.text_before_cursor
        if text.startswith("."):
            yield from self._command_completions(text)
            return
        prefix = identifier_prefix(text)
        if not wants_completion(text, prefix, complete_event):
            return
        catalog = self._predicates()
        context = argument_context(text)
        if self._is_squall() or self._lark is None:
            yield from self._squall_completions(catalog, prefix)
        elif context is not None:
            yield from self._argument_completions(
                text, prefix, catalog, context
            )
        else:
            yield from self._datalog_completions(text, prefix, catalog)

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

    def _squall_completions(self, catalog: dict, prefix: str):
        """Yield the predicates, then the SQUALL keywords."""
        if not prefix:
            return
        yield from self._predicate_completions(catalog, prefix)
        for word in self._squall_keywords:
            if word.startswith(prefix):
                yield Completion(
                    word, start_position=-len(prefix), display_meta="keyword"
                )

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
        """
        options = self._grammar_options(text[:len(text) - len(prefix)])
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
