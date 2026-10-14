"""
Catalog of the predicates of an engine.

:func:`predicate_catalog` reads the predicates the engine has, each
time it is called, so that the relations and the rules added during the
session are in it as soon as they exist: the relations stored in the
engine, its rules, those of the session, the relations loaded with
``.load``, its probabilistic facts and choices and its functions, each
described by a :class:`PredicateInfo`.
"""

from typing import NamedTuple, Optional

from neurolang.datalog.basic_representation import UnionOfConjunctiveQueries
from neurolang.datalog.expressions import AdornedSymbol
from neurolang.datalog.wrapped_collections import WrappedRelationalAlgebraSet
from neurolang.probabilistic.expressions import ProbabilisticQuery
from neurolang.utils.cli import _is_generated_column


class PredicateInfo(NamedTuple):
    """
    A predicate of an engine.

    *kind* tells where it comes from: ``stored`` (EDB relation),
    ``derived`` (IDB predicate of the engine), ``session`` (rule defined
    in the session), ``loaded`` (relation loaded with ``.load``),
    ``probabilistic`` (probabilistic fact or choice) or ``function``.
    *columns* are the names of its columns, or None when they are
    unknown, and *arity* its number of columns, or None when it is
    unknown. A *probabilistic* predicate has a probability column
    besides its *arity* columns.
    """

    name: str
    kind: str
    columns: Optional[tuple] = None
    arity: Optional[int] = None
    probabilistic: bool = False

    def signature(self) -> str:
        """Return ``name(col, ...)``, or ``name/arity``, or ``name``."""
        if self.columns:
            return f"{self.name}({', '.join(self.columns)})"
        if self.arity is not None:
            return f"{self.name}/{self.arity}"
        return self.name


# The kinds of predicates, in the order they are listed.
KINDS = (
    "session", "loaded", "stored", "derived", "probabilistic", "function"
)


def is_user_symbol(symbol, protected=()) -> bool:
    """
    Return whether *symbol* names a predicate of the user or the engine.

    The symbols NeuroLang generates are left out: fresh symbols, the
    adorned symbols of magic sets and of the helpers of a conditional
    probability (``P^cond_num`` of ``P``, named ``P`` too), private
    names such as ``__constraints__`` and the *protected* keywords.
    """
    name = getattr(symbol, "name", None)
    return bool(
        isinstance(name, str) and name
        and not name.startswith(("_", "fresh_"))
        and not isinstance(symbol, AdornedSymbol)
        and symbol not in protected
    )


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
    """
    Return the parameter names of the first rule of *union*, if any.

    A rule defined in SQUALL has generated names, such as
    ``fresh_00000002``, which tell nothing: it has no column names then.
    """
    args = union.formulas[0].consequent.args
    names = tuple(getattr(arg, "name", None) for arg in args)
    if all(
        isinstance(name, str) and not _is_generated_column(name)
        for name in names
    ):
        return names
    return None


def _fitting(columns, arity) -> Optional[tuple]:
    """Return *columns* as a tuple if there are *arity* of them."""
    if columns is None or len(columns) != arity:
        return None
    return tuple(columns)


def _relation_info(name, rows, kind, columns) -> PredicateInfo:
    """Return the info of the relation *name* whose set is *rows*."""
    probabilistic = kind == "probabilistic"
    arity = rows.arity - probabilistic
    return PredicateInfo(
        name, kind, _fitting(columns.get(name), arity), arity, probabilistic
    )


def _rule_info(name, union, kind, columns) -> PredicateInfo:
    """
    Return the info of the rules *union* of *name*.

    Its columns are those *columns* declares for it, or else the
    parameters of its first rule. A rule with a probability in its
    head, as ``with inferred probability`` defines, is probabilistic.
    """
    args = union.formulas[0].consequent.args
    probabilistic = any(isinstance(arg, ProbabilisticQuery) for arg in args)
    arity = len(args) - probabilistic
    declared = _fitting(columns.get(name), arity)
    if declared is None:
        declared = _rule_columns(union)
    return PredicateInfo(name, kind, declared, arity, probabilistic)


def _function_arity(symbol) -> Optional[int]:
    """Return the number of parameters of the typed function *symbol*."""
    args = getattr(symbol.type, "__args__", None)
    if not args or args[0] is Ellipsis:
        return None
    return len(args) - 1


class CatalogBuilder:
    """Collect the :class:`PredicateInfo` of the symbols of an engine."""

    def __init__(self, nl, columns=None, loaded=None):
        """
        Set up the collection on the engine *nl*.

        See :func:`predicate_catalog` for the *columns* and the *loaded*
        relations.
        """
        loaded = loaded or {}
        self.program_ir = nl.program_ir
        self.columns = {**(columns or {}), **loaded}
        self.loaded = loaded
        self.probabilistic = _probabilistic_symbols(self.program_ir)
        self.protected = getattr(self.program_ir, "protected_keywords", ())

    def info(self, symbol, value, session: bool):
        """
        Return the info of *symbol* defined as *value*, or None.

        *session* tells whether it is defined in the session's scope.
        None is returned for a symbol that is not a predicate or that
        NeuroLang generates.
        """
        if not is_user_symbol(symbol, self.protected):
            return None
        inner = getattr(value, "value", None)
        if isinstance(inner, WrappedRelationalAlgebraSet):
            return _relation_info(
                symbol.name, inner, self._relation_kind(symbol),
                self.columns,
            )
        if symbol.type is UnionOfConjunctiveQueries:
            kind = "session" if session else "derived"
            return _rule_info(symbol.name, value, kind, self.columns)
        if callable(inner):
            return PredicateInfo(
                symbol.name, "function", None, _function_arity(symbol)
            )
        return None

    def _relation_kind(self, symbol) -> str:
        """Return the kind of the relation *symbol*."""
        if symbol in self.probabilistic:
            return "probabilistic"
        if symbol.name in self.loaded:
            return "loaded"
        return "stored"


def collect_catalog(builder, symbol_table) -> dict:
    """
    Return the :class:`PredicateInfo` that *builder* gives the symbols.

    The symbols are those of *symbol_table* and its enclosing scopes;
    a symbol defined in several scopes is described as the innermost
    defines it. When there are several scopes, the innermost one is the
    session's.
    """
    scopes = list(_scopes(symbol_table))
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


def predicate_catalog(nl, columns=None, loaded=None) -> dict:
    """
    Return the predicates defined in the engine *nl*, by name.

    They are its relations, its rules (those of the innermost scope,
    when the session opened one, being the session's), its probabilistic
    facts and choices and its functions. *columns* maps relation names to
    their declared columns (e.g. from the engine's configuration) and
    *loaded* those of the relations loaded with ``.load``.
    """
    builder = CatalogBuilder(nl, columns, loaded)
    return collect_catalog(builder, nl.program_ir.symbol_table)


def listed(catalog: dict) -> list:
    """Return the infos of *catalog* by kind, as :data:`KINDS`, and name."""
    def order(info):
        kind = KINDS.index(info.kind) if info.kind in KINDS else len(KINDS)
        return kind, info.name

    return sorted(catalog.values(), key=order)
