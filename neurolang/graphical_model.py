from uuid import uuid1
import operator
import itertools
import copy
from collections import defaultdict
import logging

import numpy as np
import sympy

from .expressions import (
    Expression, NeuroLangException, FunctionApplication, Constant, Symbol,
    Definition, ExpressionBlock
)
from .solver_datalog_naive import Implication, Fact
from .expression_walker import ExpressionWalker
from .expression_pattern_matching import add_match
from . import unification
from .generative_datalog import DeltaAtom, DeltaTerm


def get_antecedent_literals(rule):
    if not isinstance(rule, Implication):
        raise NeuroLangException('Implication expected')

    def aux_get_antecedent_literals(expression):
        if not (
            isinstance(expression, FunctionApplication) and
            isinstance(expression.functor, Constant) and
            expression.functor.value == operator.and_
        ):
            return [expression]
        else:
            return (
                aux_get_antecedent_literals(expression.args[0]) +
                aux_get_antecedent_literals(expression.args[1])
            )

    return aux_get_antecedent_literals(rule.antecedent)


def get_antecedent_predicate_names(rule):
    antecedent_literals = get_antecedent_literals(rule)
    return [literal.functor.name for literal in antecedent_literals]


def produce(rule, facts):
    if (
        not isinstance(facts, (list, tuple)) or
        any(not isinstance(f, FunctionApplication) for f in facts)
    ):
        raise Exception(
            'Expected a list/tuple of function applications but got {}'
            .format(type(facts))
        )
    consequent = rule.consequent
    antecedent_literals = get_antecedent_literals(rule)
    if len(antecedent_literals) != len(facts):
        raise Exception(
            'Expected same number of facts as number of antecedent literals'
        )
    for i in range(len(antecedent_literals)):
        res = unification.most_general_unifier(
            antecedent_literals[i], facts[i]
        )
        if res is None:
            return None
        else:
            unifier, _ = res
            for j in range(len(antecedent_literals)):
                consequent = unification.apply_substitution(
                    consequent, unifier
                )
                antecedent_literals[j] = unification.apply_substitution(
                    antecedent_literals[j], unifier
                )
    return consequent


def infer(rule, facts):
    '''
    Return the set of facts that can be inferred in one step from a set of
    facts by applying the Elementary Production (EP) inference rule.

    Arguments
    ---------
    rule : Implication
        Rule used for inferring new facts from the original set of facts.
    facts : set of function applications on constants
        Available facts on which the rule will be applied.

    Note
    ----
    See Logic Programming and Databases, section 7.1.1

    '''

    n = len(get_antecedent_literals(rule))
    result = set()
    for facts in itertools.permutations(facts, n):
        new = produce(rule, facts)
        if new is not None:
            result.add(new)
    return result


def group_facts_by_predicate(facts, predicates):
    result = defaultdict(set)
    for fact in facts:
        pred = fact.functor.name
        if pred in predicates:
            result[pred].add(fact)
    return result


def substitute_dterm(datom, value):
    return FunctionApplication[datom.type](
        datom.functor,
        tuple(
            value if isinstance(term, DeltaTerm) else term
            for term in datom.terms
        )
    )


# def delta_infer1(rule, facts):
    # antecedent_predicate_names = get_antecedent_predicate_names(rule)
    # facts_by_predicate = group_facts_by_predicate(
        # facts, set(antecedent_predicate_names)
    # )
    # antecedent_facts = tuple(
        # facts_by_predicate[pred] for pred in antecedent_predicate_names
    # )
    # inferred_facts = set()
    # for fact_list in itertools.product(*antecedent_facts):
        # new = produce(rule, fact_list)
        # if new is not None:
            # inferred_facts.add(new)
    # if isinstance(rule.consequent, DeltaAtom):
        # new_result = set()
        # for cpd_entries in itertools.product(
            # *[
                # get_dterm_dist(dfact.delta_term).cpd_map.items()
                # for dfact in inferred_facts
            # ]
        # ):
            # new_facts = set(
                # substitute_dterm(dfact, entry[0])
                # for dfact, entry in zip(inferred_facts, cpd_entries)
            # )
            # prob = multiply_n_expressions(*[entry[1] for entry in cpd_entries])
            # new_result.add((frozenset(new_facts), prob))
        # return new_result
    # else:
        # return {(frozenset(inferred_facts), Constant[int](1))}


class FactSetRV(Definition):
    def __init__(self, name):
        if not isinstance(name, Constant) or name.type is not str:
            raise NeuroLangException(
                'Name of random variable expected to be a Constant[str]'
            )
        self.name = name

    def cpd_support(self, parent_values):
        raise NeuroLangException('Not implemented in general class')

    def __hash__(self):
        return hash(self.name)


class ExtensionalFactSetRV(FactSetRV):
    def __init__(self, predicate):
        super().__init__(Constant[str](predicate))
        self.predicate = predicate
        self.ground_facts = set()

    def cpd_support(self, parent_values):
        if len(parent_values) != 0:
            raise NeuroLangException(
                'Unexpected parents for extensional fact set random variable'
            )
        return frozenset({self.ground_facts})


class IntensionalFactSetRV(FactSetRV):
    def __init__(self, predicate, rule_id):
        super().__init__(Constant[str](f'{predicate}_{rule_id}'))
        self.predicate = predicate
        self.rule_id = rule_id

    def cpd_support(self, parent_values):
        pass


class GraphicalModel(Expression):
    def __init__(self):
        self.random_variables = dict()
        self.parents = defaultdict(set)


class GDatalogToGraphicalModelTranslator(ExpressionWalker):
    '''Expression walker generating the graphical model
    representation of a GDatalog[Δ] program.
    '''

    def __init__(self):
        self.gm = GraphicalModel()
        self.intensional_predicate_rule_count = defaultdict(int)

    @add_match(ExpressionBlock)
    def expression_block(self, block):
        for exp in block.expressions:
            self.walk(exp)
        return self.gm

    @add_match(Fact)
    def fact(self, expression):
        predicate = expression.consequent.functor.name
        if predicate not in self.gm.random_variables:
            rv = ExtensionalFactSetRV(predicate)
            self.gm.random_variables[predicate] = rv
        else:
            rv = self.gm.random_variables[predicate]
        rv.ground_facts.add(expression.fact)
        return expression

    @add_match(Implication(Definition, ...))
    def rule(self, rule):
        predicate = rule.consequent.functor.name
        self.intensional_predicate_rule_count[predicate] += 1
        rule_id = self.intensional_predicate_rule_count[predicate]
        rv_name = f'{predicate}_{rule_id}'
        rv = IntensionalFactSetRV(predicate, rule_id)
        if rv_name in self.gm.random_variables:
            raise NeuroLangException(
                f'Random variable {rv_name} already efined'
            )
        self.gm.random_variables[rv_name] = rv
        self.gm.parents[rv_name] = set()
        for pred in get_antecedent_predicate_names(rule):
            self.gm.parents[rv_name].add(pred)
        return rule


class GraphicalModelSolver(ExpressionWalker):
    pass
