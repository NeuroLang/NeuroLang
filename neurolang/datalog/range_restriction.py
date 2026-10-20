"""
Range restriction of Datalog rules.

A rule is *range restricted* (safe) when every variable of its body is
bound: it occurs in a positive atom, or it is defined by an equality
``x == expression`` whose expression only uses bound variables, or it is
an element of a bound set, ``contains(s, x)``. Functions, comparisons and
negated atoms only filter the values of bound variables. The named
relational algebra translation (:mod:`.translate_to_named_ra`) can only
evaluate range-restricted rules; this module finds the variables that
break the condition and builds an
:class:`~neurolang.exceptions.UnboundVariablesError` naming them.
"""

from operator import contains, eq

from ..exceptions import UnboundVariablesError
from ..expression_walker import ExpressionWalker, add_match
from ..expressions import Constant, FunctionApplication, Lambda, Symbol
from ..logic import Conjunction, ExistentialPredicate, Negation
from ..logic.expression_processing import extract_logic_free_variables

_ATOM, _DEFINITION, _FILTER = "atom", "definition", "filter"


def check_range_restriction(rule, symbol_table=None, query=False):
    """
    Raise an error if a variable of the body of *rule* can never be
    bound.

    The variables of the head of a rule can be bound by the queries using
    it: magic sets give ``x`` the value ``'a'`` in ``A(x, y) :- P(w, y),
    w != x`` for the query ``ans(y) :- A('a', y)``, and probabilistic
    solvers unfold rules into queries. They are taken as bound, as are
    the variables they define, e.g. ``sim`` in ``A(q, s) :- P(s),
    sim == f(q, s), sim > .5``, unless *rule* is a query. Rules with a fresh head
    functor are generated, by syntactic sugar or for queries, and are
    left to the translation to relational algebra, see
    :func:`unbound_variables_error`.

    Parameters
    ----------
    rule : Implication
        The rule.
    symbol_table : Mapping, optional
        Symbols of the program: functors that are callable constants in
        it are functions, other functors are predicates, and the symbols
        it defines are constants, not variables.
    query : bool
        Whether *rule* is a query.

    Raises
    ------
    UnboundVariablesError
        If a variable of the body is not bound.
    """
    functor = rule.consequent.functor
    if functor.is_fresh and not query:
        return
    error = unbound_variables_error(
        rule.antecedent, symbol_table,
        rule=None if query else functor.name, query=query,
        exempt=() if query else extract_logic_free_variables(rule.consequent),
    )
    if error is not None:
        raise error


def unbound_variables_error(
    body, symbol_table=None, rule=None, query=False, exempt=()
):
    """
    Return the error describing the unbound variables of *body*.

    Variables that only occur in negated formulas are left to the
    checks of negation, see
    :class:`~neurolang.exceptions.NegativeFormulaNotSafeRangeException`.

    Parameters
    ----------
    body : Expression
        The body of a rule.
    symbol_table : Mapping, optional
        See :func:`check_range_restriction`.
    rule : str, optional
        The name of the rule, for the message.
    query : bool
        Whether *body* is the body of a query, for the message.
    exempt : iterable of Symbol
        Variables that may be bound from outside *body*.

    Returns
    -------
    UnboundVariablesError or None
        None if all variables of *body* are bound.
    """
    symbol_table = {} if symbol_table is None else symbol_table
    literals = tuple(
        (literal, _classify(literal, symbol_table))
        for literal in _literals(body)
    )
    bound = _bound_variables(literals, symbol_table, exempt)
    bound |= _negated_only_variables(literals)
    unbound = [
        v for v in extract_logic_free_variables(body) if v not in bound
    ]
    if not unbound:
        return None
    return _error(literals, unbound, rule, query)


def _negated_only_variables(literals):
    """Return the variables only occurring in negated *literals*."""
    negated, other = set(), set()
    for literal, _ in literals:
        variables = extract_logic_free_variables(literal)
        (negated if isinstance(literal, Negation) else other).update(
            variables
        )
    return negated - other


def _error(literals, unbound, rule, query):
    """
    Return the error for the *unbound* variables of the classified
    *literals* of a body.
    """
    occurrences = [
        literal for literal, kind in literals
        if kind != _ATOM
        and not extract_logic_free_variables(literal).isdisjoint(unbound)
    ]
    atoms = [literal for literal, kind in literals if kind == _ATOM]
    example = _example(occurrences[:1], unbound, atoms)
    render = _renderer()
    return UnboundVariablesError(
        rule, tuple(map(render, unbound)), tuple(map(render, occurrences)),
        None if example is None else render(example), query,
    )


def _literals(formula):
    """Yield the literals of the conjunctions and existentials of *formula*."""
    if isinstance(formula, Conjunction):
        for subformula in formula.formulas:
            yield from _literals(subformula)
    elif isinstance(formula, ExistentialPredicate):
        yield from _literals(formula.body)
    else:
        yield formula


def _resolve(functor, symbol_table):
    """Return the value of *functor* in *symbol_table*, if any."""
    if isinstance(functor, Symbol):
        try:
            return symbol_table[functor]
        except KeyError:
            return None
    return functor


def _function(functor, symbol_table):
    """
    Return the callable *functor* stands for, None for a predicate.
    """
    value = _resolve(functor, symbol_table)
    if isinstance(value, Lambda):
        return value
    if isinstance(value, Constant) and callable(value.value):
        return value.value
    return None


def _classify(literal, symbol_table):
    """
    Return whether *literal* is an atom, a definition or a filter.

    Literals that are not function applications are taken as atoms, so
    that unknown constructs never make a rule unsafe.
    """
    if isinstance(literal, Negation):
        return _FILTER
    if not isinstance(literal, FunctionApplication):
        return _ATOM
    function = _function(literal.functor, symbol_table)
    if function is None:
        return _ATOM
    if function in (eq, contains) and len(literal.args) == 2:
        return _DEFINITION
    return _FILTER


def _bound_variables(literals, symbol_table, exempt):
    """
    Return the variables bound by the classified *literals* and by
    *exempt*, and the symbols they use that are constants of
    *symbol_table*.
    """
    bound = set(exempt)
    for literal, kind in literals:
        variables = extract_logic_free_variables(literal)
        bound |= {v for v in variables if v in symbol_table}
        if kind == _ATOM:
            bound |= set(variables)
    definitions = [literal for literal, kind in literals if kind == _DEFINITION]
    return _close_under_definitions(bound, definitions, symbol_table)


def _close_under_definitions(bound, definitions, symbol_table):
    """
    Return *bound* with the variables the *definitions* bind, in any
    order: ``y == x + 1, x == 2 * z`` binds ``x`` then ``y`` from ``z``.
    """
    size = None
    while size != len(bound):
        size = len(bound)
        for definition in definitions:
            bound |= _defined_variables(definition, bound, symbol_table)
    return bound


def _defined_variables(definition, bound, symbol_table):
    """
    Return the variables the equality or ``contains`` *definition*
    binds once the variables in *bound* are bound.
    """
    left, right = definition.args
    if _function(definition.functor, symbol_table) is contains:
        sides = ((left, right),)
    else:
        sides = ((right, left), (left, right))
    defined = set()
    for source, target in sides:
        if (
            isinstance(target, Symbol)
            and extract_logic_free_variables(source) <= bound
        ):
            defined |= set(extract_logic_free_variables(target))
    return defined


def _example(occurrences, unbound, atoms):
    """
    Return an atom that would bind the unbound variables of the first
    of *occurrences*, if any.

    The bound and unbound variables of the occurrence are paired in
    order, as in ``euclidean(i, j, k, i1, j1, k1)``, and the first atom
    over all the bound ones is copied with them replaced by their pair:
    ``voxel(i, j, k)`` gives ``voxel(i1, j1, k1)``.
    """
    replacement = _variable_pairs(occurrences, unbound)
    if not replacement:
        return None
    return next(
        (
            atom.functor(*(replacement.get(arg, arg) for arg in atom.args))
            for atom in atoms if _has_arguments(atom, replacement)
        ),
        None,
    )


def _variable_pairs(occurrences, unbound):
    """
    Return the bound variables of *occurrences* mapped in order to their
    *unbound* ones, empty if they are not as many.
    """
    variables = extract_logic_free_variables(Conjunction(occurrences))
    targets = [v for v in variables if v in unbound]
    sources = [v for v in variables if v not in unbound]
    if len(sources) != len(targets):
        return {}
    return dict(zip(sources, targets))


def _has_arguments(atom, variables):
    """Whether the atom *atom* has all *variables* as arguments."""
    return isinstance(atom, FunctionApplication) and all(
        v in atom.args for v in variables
    )


def _renderer():
    """
    Return a function printing expressions as Datalog text, with the
    same names for the fresh variables of all of them.
    """
    # Imported here: the printer depends on the whole Datalog layer.
    from .pretty_printer import DatalogPrettyPrinter

    printer = DatalogPrettyPrinter()

    def render(expression):
        return printer.walk(_NameFunctions().walk(expression))

    return render


def _is_named_function(application):
    """Whether *application* applies a named Python function."""
    function = application.functor.value
    return (
        callable(function)
        and hasattr(function, "__name__")
        and getattr(function, "__module__", None) not in (
            "operator", "_operator"
        )
    )


class _NameFunctions(ExpressionWalker):
    """
    Replace the Python functions applied in an expression by their name,
    to print ``euclidean(i, j)`` as written rather than the constant.
    """

    @add_match(FunctionApplication(Constant, ...), _is_named_function)
    def named_function(self, application):
        return Symbol(application.functor.value.__name__)(
            *self.walk(application.args)
        )
