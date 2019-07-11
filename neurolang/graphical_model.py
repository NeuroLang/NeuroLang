from uuid import uuid1
import operator
import itertools
import copy
from collections import defaultdict
import logging
from typing import Set, FrozenSet, Tuple, Iterable, Callable

import numpy as np

from .expressions import (
    Expression, NeuroLangException, FunctionApplication, Constant, Symbol,
    Definition, ExpressionBlock
)
from .solver_datalog_naive import Implication, Fact
from .expression_walker import PatternWalker
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


def is_dterm_constant(dterm):
    return all(isinstance(param, Constant) for param in dterm.dist_params)


def get_constant_dterm_dist(dterm):
    if not is_dterm_constant(dterm):
        raise NeuroLangException('Expected a constant Δ-term')
    if dterm.dist_name == Constant[str]('bernoulli'):
        p = dterm.dist_params[0].value
        return frozenset({
            (Constant[int](1), Constant[float](p)),
            (Constant[int](0), Constant[float](1.0 - p)),
        })


FactSet = FrozenSet[Fact]
FactSetSymbol = Symbol[FactSet]
FactSetTableCPD = FrozenSet[Tuple[FactSet, float]]
FactSetTableCPDFunctor = Callable[[Iterable[FactSet]], FactSetTableCPD]


class ExtensionalPredicateCPDFunctor(Definition[FactSetTableCPDFunctor]):
    def __init__(self, predicate):
        self.predicate = predicate
        self.facts = set()


class InferredFactSetCPDFunctor(Definition[FactSetTableCPDFunctor]):
    def __init__(self, rule):
        self.rule = rule


class UnionFactSetCPDFunctor(Definition[FactSetTableCPDFunctor]):
    def __init__(self, predicate):
        self.predicate = predicate


class GraphicalModel(Expression):
    def __init__(self):
        self.rv_to_cpd_functor = dict()
        self.parents = defaultdict(frozenset)

    def add_parenting(self, child, parent):
        self.parents[child] = self.parents[child].union({parent})


class GDatalogToGraphicalModelTranslator(PatternWalker):
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

    @add_match(Fact)
    def fact(self, expression):
        predicate = expression.consequent.functor.name
        rv_symbol = FactSetSymbol(predicate)
        if rv_symbol not in self.gm.rv_to_cpd_functor:
            self.gm.rv_to_cpd_functor[rv_symbol] = (
                ExtensionalPredicateCPDFunctor(predicate)
            )
            self.gm.rv_to_cpd_functor[rv_symbol].facts.add(expression)

    @add_match(Implication(Definition, ...))
    def rule(self, rule):
        predicate = rule.consequent.functor.name
        self.intensional_predicate_rule_count[predicate] += 1
        rule_id = self.intensional_predicate_rule_count[predicate]
        rule_rv_symbol = FactSetSymbol(f'{predicate}_{rule_id}')
        pred_rv_symbol = FactSetSymbol(f'{predicate}')
        if rule_rv_symbol in self.gm.rv_to_cpd_functor:
            raise NeuroLangException(
                f'Random variable {rule_rv_symbol} already defined'
            )
        self.gm.rv_to_cpd_functor[rule_rv_symbol
                                  ] = (InferredFactSetCPDFunctor(rule))
        for antecedent_pred in get_antecedent_predicate_names(rule):
            antecedent_rv_symbol = FactSetSymbol(f'{antecedent_pred}')
            self.gm.add_parenting(rule_rv_symbol, antecedent_rv_symbol)
        if pred_rv_symbol not in self.gm.rv_to_cpd_functor:
            self.gm.rv_to_cpd_functor[pred_rv_symbol] = \
                UnionFactSetCPDFunctor(predicate)
            self.gm.add_parenting(pred_rv_symbol, rule_rv_symbol)


def gdatalog2gm(program):
    translator = GDatalogToGraphicalModelTranslator()
    translator.walk(program)
    return translator.gm


def sort_rvs(gm):
    result = list()
    sort_rvs_aux(gm, '__dummy__', set(gm.rv_to_cpd_functor.keys()), result)
    return result[:-1]


def sort_rvs_aux(gm, rv, parents, result):
    for parent_rv in parents:
        sort_rvs_aux(gm, parent_rv, gm.parents[parent_rv], result)
    if rv not in result:
        result.append(rv)


def delta_infer1(rule, facts):
    antecedent_predicate_names = get_antecedent_predicate_names(rule)
    facts_by_predicate = group_facts_by_predicate(
        facts, set(antecedent_predicate_names)
    )
    antecedent_facts = tuple(
        facts_by_predicate[pred] for pred in antecedent_predicate_names
    )
    inferred_facts = set()
    for fact_list in itertools.product(*antecedent_facts):
        new = produce(rule, fact_list)
        if new is not None:
            inferred_facts.add(new)
    if isinstance(rule.consequent, DeltaAtom):
        new_result = set()
        for cpd_entries in itertools.product(
            *[
                get_constant_dterm_dist(dfact.delta_term)
                for dfact in inferred_facts
            ]
        ):
            new_facts = set(
                substitute_dterm(dfact, entry[0])
                for dfact, entry in zip(inferred_facts, cpd_entries)
            )
            prob = np.prod([entry[1].value for entry in cpd_entries])
            new_result.add((frozenset(new_facts), prob))
        return frozenset(new_result)
    else:
        return frozenset({(frozenset(inferred_facts), Constant[int](1))})


class GraphicalModelSolver(PatternWalker):
    @add_match(GraphicalModel)
    def graphical_model(self, graphical_model):
        pass

    @add_match(FunctionApplication(ExtensionalPredicateCPDFunctor, ...))
    def extensional_predicate_cpd(self, expression):
        return frozenset({
            (frozenset(expression.functor.facts), Constant[float](1.0))
        })

    @add_match(FunctionApplication(UnionFactSetCPDFunctor, ...))
    def union_cpd(self, expression):
        parent_values = expression.args
        return frozenset({
            (frozenset().union(*parent_values), Constant[float](1.0))
        })

    @add_match(FunctionApplication(InferredFactSetCPDFunctor, ...))
    def rule_cpd(self, expression):
        parent_values = expression.args
        rule = expression.functor.rule
        return delta_infer1(rule, frozenset().union(*parent_values))
