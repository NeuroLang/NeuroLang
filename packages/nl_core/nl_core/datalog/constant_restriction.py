"""
Restricting the rules of a predicate to the constants it is read with.

When every literal of a predicate ``p`` in a program has a constant at an
argument, as ``endpoint_anterior_of(x, e, 'amygdala', h)``, the tuples of
``p`` with another value there are read by no rule: ``p`` can be computed
for those constants only. Each rule of ``p`` is replaced by a copy for
each tuple of constants it is read with, whose body equates the
variables of its head there with the constants: a selection of the
rule's plan, pushed to the relations it reads
(``term_box_hi(term, h, 1, b)`` with ``term == 'amygdala'``).

This is the restriction magic sets make of a predicate bound by
constants alone, without magic predicates: it needs no sideways passing
of bindings, and adds no rule a cycle of the program could go through,
so it keeps a stratified program stratified (magic sets leave the tract
queries' predicates unbound for that reason, 5.7 million tuples of
``endpoint_anterior_of`` computed for 2 terms).
"""

from itertools import chain

from nl_expr import expression_walker as ew
from nl_expr.expressions import Constant, FunctionApplication, Symbol

from .expression_processing import EQ
from .expressions import Conjunction, Implication


def restrict_to_constants_used(rules, readers=(), kept=(), max_copies=64):
    """
    Return *rules*, those of a predicate always read with constants at
    some of its arguments restricted to them.

    Parameters
    ----------
    rules : iterable of Implication
        The rules that may be restricted.
    readers : iterable of Implication
        Other rules of the program, whose literals are read as those of
        *rules*, but which are not restricted.
    kept : iterable of Symbol
        Predicates not restricted: those answering a query, read outside
        the program.
    max_copies : int
        A predicate whose rules would have more copies than this is not
        restricted.

    Returns
    -------
    tuple of Implication
    """
    rules, readers, kept = tuple(rules), tuple(readers), set(kept)
    restriction = _restriction(rules, readers, kept, max_copies)
    if not restriction:
        return rules
    return tuple(chain.from_iterable(
        _restricted(rule, restriction) for rule in rules
    ))


def _restriction(rules, readers, kept, max_copies):
    """The positions and the tuples of constants each predicate of
    *rules* is restricted to, by predicate."""
    heads = {}
    for rule in rules:
        if not isinstance(rule, Implication) or not isinstance(
            rule.consequent, FunctionApplication
        ):
            return {}
        heads.setdefault(rule.consequent.functor, []).append(
            rule.consequent.args
        )
    literals, references = {}, {}
    for rule in rules + readers:
        for _, expression in ew.expression_iterator(rule.antecedent):
            if isinstance(expression, FunctionApplication) and (
                expression.functor in heads
            ):
                literals.setdefault(expression.functor, []).append(
                    expression.args
                )
            elif isinstance(expression, Symbol) and expression in heads:
                references[expression] = references.get(expression, 0) + 1
    restriction = {}
    for functor, read in literals.items():
        # read as the functor of its literals only, not passed as an
        # argument, nor answering a query
        if functor in kept or references.get(functor, 0) != len(read):
            continue
        positions = tuple(
            i for i in range(len(read[0]))
            if all(isinstance(args[i], Constant) for args in read)
            and all(
                isinstance(args[i], (Symbol, Constant))
                for args in heads[functor]
            )
            and any(isinstance(args[i], Symbol) for args in heads[functor])
        )
        if not positions:
            continue
        constants = {
            tuple(args[i].value for i in positions): tuple(
                args[i] for i in positions
            )
            for args in read
        }
        if len(constants) * len(heads[functor]) <= max_copies:
            restriction[functor] = positions, tuple(constants.values())
    return restriction


def _restricted(rule, restriction):
    """The copies of *rule* for the constants its predicate is restricted
    to, or *rule* if it is not."""
    head = rule.consequent
    if head.functor not in restriction:
        return (rule,)
    positions, constants = restriction[head.functor]
    body = (
        rule.antecedent.formulas if isinstance(rule.antecedent, Conjunction)
        else (rule.antecedent,)
    )
    copies = []
    for values in constants:
        equalities = {}
        for i, value in zip(positions, values):
            arg = head.args[i]
            if isinstance(arg, Constant):
                if arg.value != value.value:
                    break
            elif equalities.setdefault(arg, value).value != value.value:
                break  # one variable at two positions, two constants
        else:
            copies.append(Implication(head, Conjunction(body + tuple(
                EQ(variable, value) for variable, value in equalities.items()
            ))))
    return tuple(copies)
