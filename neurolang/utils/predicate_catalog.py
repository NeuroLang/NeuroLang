"""
The catalog of the predicates defined in an engine.

It lists the relations, the rules, the probabilistic facts and choices
and the functions of an engine, with their columns and the entities they
relate when they are known. The interactive REPL offers them in its
completion and the LLM query assistant describes them to the model.
"""

from typing import NamedTuple, Optional

from neurolang.datalog.basic_representation import UnionOfConjunctiveQueries
from neurolang.datalog.wrapped_collections import WrappedRelationalAlgebraSet
from neurolang.probabilistic.expressions import ProbabilisticQuery
from neurolang.utils import squall_sorts


class PredicateInfo(NamedTuple):
    """
    A predicate defined in an engine.

    *kind* tells where it comes from: ``stored`` (EDB relation),
    ``derived`` (IDB predicate of the engine), ``session`` (rule defined
    in the session), ``loaded`` (relation loaded with ``.load``),
    ``probabilistic`` or ``function``. *columns* are the names of its
    columns, or None when they are unknown, and *arity* its number of
    columns, or None when it is unknown. *arguments* are the
    :class:`~neurolang.utils.engine_registry.Argument` the engine
    declares, the entities it relates, or None. A *probabilistic*
    predicate has a probability column besides its *arity* columns.
    """

    name: str
    kind: str
    columns: Optional[tuple] = None
    arity: Optional[int] = None
    arguments: Optional[tuple] = None
    probabilistic: bool = False

    def signature(self) -> str:
        """
        Return ``name(entity, ...)``, ``name(col, ...)``, ``name/arity``.

        The entities of several columns, or of a column not named after
        their sort, show their columns: ``reports(study(study_id),
        voxel(i, j, k))``.
        """
        arguments = self.arguments or ()
        if len(arguments) > 1:
            entities = ", ".join(
                squall_sorts.argument_text(argument, columns=True)
                for argument in arguments
            )
            return f"{self.name}({entities})"
        if self.columns:
            return f"{self.name}({', '.join(self.columns)})"
        if self.arity is not None:
            return f"{self.name}/{self.arity}"
        return self.name

    def meta(self) -> str:
        """Return the description shown next to the completion."""
        return f"{self.signature()} · {self.kind}"


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


def _declared_arguments(name, arity, signatures) -> Optional[tuple]:
    """Return the arguments *signatures* gives *name*, if they fit."""
    arguments = signatures.get(name)
    if arguments is None:
        return None
    if sum(len(argument.columns) for argument in arguments) != arity:
        return None
    return tuple(arguments)


def _relation_info(name, value, kind, columns, signatures) -> PredicateInfo:
    """Return the info of the relation *name* whose set is *value*."""
    arity = value.arity
    probabilistic = kind == "probabilistic"
    if probabilistic:
        arity -= 1
    columns = columns.get(name)
    if columns is not None and len(columns) != arity:
        columns = None
    return PredicateInfo(
        name, kind, columns and tuple(columns), arity,
        _declared_arguments(name, arity, signatures), probabilistic,
    )


def _rule_info(name, union, kind, columns, signatures) -> PredicateInfo:
    """
    Return the info of the rules *union* of *name*.

    Its columns are those *columns* declares for it, or else the
    parameters of its first rule. A rule with a probability in its
    head, as ``with inferred probability`` defines, is probabilistic.
    """
    args = union.formulas[0].consequent.args
    probabilistic = any(isinstance(arg, ProbabilisticQuery) for arg in args)
    arity = len(args) - probabilistic
    declared = columns.get(name)
    if declared is None or len(declared) != arity:
        declared = _rule_columns(union)
    return PredicateInfo(
        name, kind, declared and tuple(declared), arity,
        _declared_arguments(name, arity, signatures), probabilistic,
    )


def _function_arity(symbol) -> Optional[int]:
    """Return the number of parameters of the typed function *symbol*."""
    args = getattr(symbol.type, "__args__", None)
    if not args or args[0] is Ellipsis:
        return None
    return len(args) - 1


class _CatalogBuilder:
    """Collect the :class:`PredicateInfo` of the symbols of an engine."""

    def __init__(self, nl, columns, loaded, signatures):
        """Set up the collection on *nl* with the known *columns*."""
        loaded = loaded or {}
        self.program_ir = nl.program_ir
        self.columns = {**(columns or {}), **loaded}
        self.signatures = signatures or {}
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
                name, inner, self._relation_kind(symbol), self.columns,
                self.signatures,
            )
        if symbol.type is UnionOfConjunctiveQueries:
            kind = "session" if session else "derived"
            return _rule_info(
                name, value, kind, self.columns, self.signatures
            )
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


def predicate_catalog(nl, columns=None, loaded=None, signatures=None) -> dict:
    """
    Return the predicates defined in the engine *nl*, by name.

    They are its relations, its rules (those of the innermost scope,
    when the session opened one, being the session's), its probabilistic
    facts and choices and its functions. *columns* maps relation names to
    their declared columns (e.g. from the engine's configuration) and
    *loaded* those of the relations loaded with ``.load``. *signatures*
    maps predicate names to the arguments the engine declares (see
    :func:`~neurolang.utils.engine_registry.get_signatures`).
    """
    builder = _CatalogBuilder(nl, columns, loaded, signatures)
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
