"""
Restricting the rules of a predicate to the constants it is read with.

When every literal of a predicate ``p`` in a program has a constant at an
argument, as ``endpoint_anterior_of(x, e, 'amygdala', h)``, the tuples of
``p`` with another value there are read by no rule: ``p`` can be computed
for those constants only. Each rule of ``p`` gets, in its body, the
variables of its head there equated with the constants, if they are one
tuple, or a test that they are one of the tuples of constants it is read
with (:class:`OneOf`, computed on whole columns): a selection of the
rule's plan, pushed to the relations it reads
(``term_box_hi(term, h, 1, b)`` with ``term == 'amygdala'``).

This is the restriction magic sets make of a predicate bound by
constants alone, without magic predicates: it needs no sideways passing
of bindings, and adds no rule a cycle of the program could go through,
so it keeps a stratified program stratified (magic sets leave the tract
queries' predicates unbound for that reason, 5.7 million tuples of
``endpoint_anterior_of`` computed for 2 terms).
"""

from typing import Callable

import numpy as np
import pandas as pd

from nl_expr import expression_walker as ew
from nl_expr.expressions import Constant, FunctionApplication, Symbol

from .expression_processing import EQ
from .expressions import Conjunction, Implication


def restrict_to_constants_used(rules, readers=(), kept=()):
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

    Returns
    -------
    tuple of Implication
    """
    rules, readers, kept = tuple(rules), tuple(readers), set(kept)
    restriction = _restriction(rules, readers, kept)
    if not restriction:
        return rules
    return tuple(_restricted(rule, restriction) for rule in rules)


def _restriction(rules, readers, kept):
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
            and all(isinstance(args[i], Symbol) for args in heads[functor])
        )
        if not positions:
            continue
        constants = {
            tuple(args[i].value for i in positions): tuple(
                args[i] for i in positions
            )
            for args in read
        }
        restriction[functor] = positions, tuple(constants.values())
    return restriction


def _restricted(rule, restriction):
    """*rule*, restricted to the constants its predicate is read with."""
    head = rule.consequent
    if head.functor not in restriction:
        return rule
    positions, constants = restriction[head.functor]
    variables = tuple(head.args[i] for i in positions)
    if len(constants) == 1:
        restricting = tuple(
            EQ(variable, value)
            for variable, value in zip(variables, constants[0])
        )
    else:
        types = [value.type for value in constants[0]]
        restricting = (Constant[Callable[types, bool]](OneOf(
            tuple(value.value for value in values) for values in constants
        ))(*variables),)
    body = (
        rule.antecedent.formulas if isinstance(rule.antecedent, Conjunction)
        else (rule.antecedent,)
    )
    return Implication(head, Conjunction(body + restricting))


class OneOf:
    """
    Whether its arguments are one of the tuples *rows*: a built-in of a
    rule's body, as an equality, rather than a relation of the tuples,
    which the ontology rewriter does not keep in a rule. Its arguments
    may be columns (``columnwise``): one ``isin`` for all the rows.
    """

    columnwise = True

    def __init__(self, rows):
        self.rows = frozenset(rows)

    def __call__(self, *values):
        if not any(np.ndim(value) for value in values):
            return values in self.rows
        if len(values) == 1:
            return pd.Series(values[0]).isin(
                [row[0] for row in self.rows]
            ).to_numpy()
        return pd.MultiIndex.from_arrays(
            [np.asarray(value) for value in values]
        ).isin(list(self.rows))

    def __eq__(self, other):
        return isinstance(other, OneOf) and self.rows == other.rows

    def __hash__(self):
        return hash(self.rows)

    def __repr__(self):
        return f"one of {sorted(self.rows)}"
