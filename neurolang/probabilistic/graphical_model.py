import itertools
from collections import defaultdict
from typing import Iterable, Callable, AbstractSet, Mapping
from copy import deepcopy
import operator
import logging

import numpy as np

from ..expressions import (
    Expression, NeuroLangException, FunctionApplication, Constant, Symbol,
    Definition, ExpressionBlock
)
from ..solver_datalog_naive import (
    Implication, Fact, extract_datalog_free_variables, DatalogBasic
)
from ..expression_walker import ExpressionWalker
from ..expression_pattern_matching import add_match
from .. import unification
from .ppdl import (
    DeltaTerm, get_antecedent_predicates, get_antecedent_literals,
    is_gdatalog_rule, get_dterm
)
from .distributions import TableDistribution
from ..datalog.instance import Instance, SetInstance
from ..datalog_chase import DatalogChase
from ..unification import apply_substitution


def produce(rule, facts):
    if (
        not isinstance(facts, (list, tuple)) or
        any(not isinstance(f, Fact) for f in facts)
    ):
        raise Exception(
            'Expected a list/tuple of facts but got {}'.format(type(facts))
        )
    consequent = rule.consequent
    antecedent_literals = get_antecedent_literals(rule)
    if len(antecedent_literals) != len(facts):
        raise Exception(
            'Expected same number of facts as number of antecedent literals'
        )
    n = len(facts)
    for i in range(n):
        res = unification.most_general_unifier(
            antecedent_literals[i], facts[i].fact
        )
        if res is None:
            return None
        else:
            unifier, _ = res
            for j in range(n):
                consequent = unification.apply_substitution(
                    consequent, unifier
                )
                antecedent_literals[j] = unification.apply_substitution(
                    antecedent_literals[j], unifier
                )
    return Fact(consequent)


def substitute_dterm(datom, value):
    return FunctionApplication[datom.type](
        datom.functor,
        tuple(
            Constant(value) if isinstance(arg, DeltaTerm) else arg
            for arg in datom.args
        )
    )


def is_dterm_constant(dterm):
    return all(isinstance(param, Constant) for param in dterm.args)


def get_constant_dterm_table_cpd(dterm):
    if not is_dterm_constant(dterm):
        raise NeuroLangException('Expected a constant Δ-term')
    if dterm.functor.name == Constant[str]('bernoulli'):
        p = dterm.args[0].value
        return TableDistribution({1: p, 0: 1.0 - p})
    else:
        raise NeuroLangException(f'Unknown distribution {dterm.functor.name}')


InstanceTableCPD = Mapping[Instance, float]
InstanceTableCPDFunctor = Definition[
    Callable[[Iterable[Instance]], InstanceTableCPD]]


class ExtensionalTableCPDFunctor(InstanceTableCPDFunctor):
    def __init__(self, predicate):
        self.predicate = predicate
        self.facts = set()


class IntensionalTableCPDFunctor(InstanceTableCPDFunctor):
    def __init__(self, rule):
        self.rule = rule


class UnionInstanceTableCPDFunctor(InstanceTableCPDFunctor):
    def __init__(self, predicate):
        self.predicate = predicate


class GraphicalModel(Expression):
    def __init__(self):
        self.rv_to_cpd_functor = dict()
        self.parents = defaultdict(frozenset)

    @property
    def random_variables(self):
        return frozenset(self.rv_to_cpd_functor.keys())

    def add_parent(self, child, parent):
        self.parents[child] = self.parents[child].union({parent})

    def get_dependency_sorted_random_variables(
        self,
        result=None,
        rv=None,
        parents=None,
    ):
        if rv is None:
            result = list()
            parents = self.random_variables
        for parent in parents:
            self.get_dependency_sorted_random_variables(
                result, parent, self.parents[parent]
            )
        if rv is not None and rv not in result:
            result.append(rv)
        return result


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

    @add_match(Fact)
    def fact(self, expression):
        predicate = expression.consequent.functor.name
        rv_symbol = Symbol(predicate)
        if rv_symbol not in self.gm.rv_to_cpd_functor:
            self.gm.rv_to_cpd_functor[rv_symbol] = (
                ExtensionalTableCPDFunctor(predicate)
            )
        self.gm.rv_to_cpd_functor[rv_symbol].facts.add(expression)

    @add_match(Implication(Definition, ...))
    def rule(self, rule):
        predicate = rule.consequent.functor.name
        self.intensional_predicate_rule_count[predicate] += 1
        rule_id = self.intensional_predicate_rule_count[predicate]
        rule_rv_symbol = Symbol(f'{predicate}_{rule_id}')
        pred_rv_symbol = Symbol(f'{predicate}')
        if rule_rv_symbol in self.gm.rv_to_cpd_functor:
            raise NeuroLangException(
                f'Random variable {rule_rv_symbol} already defined'
            )
        self.gm.rv_to_cpd_functor[rule_rv_symbol] = (
            IntensionalTableCPDFunctor(rule)
        )
        for antecedent_pred in get_antecedent_predicates(rule):
            self.gm.add_parent(rule_rv_symbol, antecedent_pred)
        if pred_rv_symbol not in self.gm.rv_to_cpd_functor:
            self.gm.rv_to_cpd_functor[pred_rv_symbol] = \
                UnionInstanceTableCPDFunctor(predicate)
            self.gm.add_parent(pred_rv_symbol, rule_rv_symbol)


def gdatalog2gm(program):
    translator = GDatalogToGraphicalModelTranslator()
    translator.walk(program)
    return translator.gm


def delta_infer1(rule, instance):
    if not isinstance(instance, SetInstance):
        raise NeuroLangException('Expected instance to be a SetInstance')
    antecedent_facts = tuple(
        instance[pred] for pred in get_antecedent_predicates(rule)
    )
    inferred_facts = set()
    for fact_list in itertools.product(*antecedent_facts):
        new = produce(rule, fact_list)
        if new is not None:
            inferred_facts.add(new)
    if is_gdatalog_rule(rule):
        table = dict()
        for cpd_entries in itertools.product(
            *[
                get_constant_dterm_table_cpd(get_dterm(dfact.consequent)
                                             ).table.items()
                for dfact in inferred_facts
            ]
        ):
            new_facts = frozenset(
                Fact(substitute_dterm(dfact.consequent, entry[0]))
                for dfact, entry in zip(inferred_facts, cpd_entries)
            )
            prob = np.prod([entry[1] for entry in cpd_entries])
            table[SetInstance(new_facts)] = prob
    else:
        table = {SetInstance(frozenset(inferred_facts)): 1.0}
    return TableDistribution(table)


def check_is_instance(value):
    if not isinstance(value, Instance):
        raise NeuroLangException('Expected an Instance')


def generate_graphical_model_possible_outcomes(
    graphical_model,
    rv_idx=None,
    ordered_rvs=None,
    rv_values=None,
    result_prob=1.0,
    outcomes=None,
):
    if ordered_rvs is None:
        outcomes = dict()
        rv_values = dict()
        ordered_rvs = graphical_model.get_dependency_sorted_random_variables()
        generate_graphical_model_possible_outcomes(
            graphical_model, 0, ordered_rvs, rv_values, result_prob, outcomes
        )
        return Constant[TableDistribution](TableDistribution(outcomes))
    else:
        if rv_idx >= len(ordered_rvs):
            result = SetInstance.union(*rv_values.values())
            if result in outcomes:
                old_prob = outcomes[result]
                new_prob = old_prob + result_prob
                outcomes[result] = new_prob
            else:
                outcomes[result] = result_prob
        else:
            rv_symbol = ordered_rvs[rv_idx]
            cpd_functor = graphical_model.rv_to_cpd_functor[rv_symbol]
            parent_rvs = graphical_model.parents[rv_symbol]
            parent_values = tuple(rv_values[rv] for rv in parent_rvs)
            if isinstance(cpd_functor, ExtensionalTableCPDFunctor):
                cpd = TableDistribution({
                    Instance(frozenset(cpd_functor.facts)):
                    1.0
                })
            elif isinstance(cpd_functor, UnionInstanceTableCPDFunctor):
                cpd = TableDistribution({
                    SetInstance.union(*parent_values): 1.0
                })
            elif isinstance(cpd_functor, IntensionalTableCPDFunctor):
                cpd = delta_infer1(
                    cpd_functor.rule, SetInstance.union(*parent_values)
                )
            for facts, prob in cpd.table.items():
                new_rv_values = rv_values.copy()
                new_rv_values[rv_symbol] = facts
                generate_graphical_model_possible_outcomes(
                    graphical_model, rv_idx + 1, ordered_rvs, new_rv_values,
                    result_prob * prob, outcomes
                )


def solve_conditional_probability_query(graphical_model, evidence):
    check_is_instance(evidence)
    outcomes = generate_graphical_model_possible_outcomes(graphical_model)
    matches_query = lambda outcome: evidence <= outcome
    return Constant[TableDistribution](
        outcomes.value.conditioned_on(matches_query)
    )


def is_valid_query_atom(atom):
    return isinstance(atom, FunctionApplication) and all(
        isinstance(arg, (Symbol, Constant)) for arg in atom.args
    )


def construct_conjunction(atoms):
    if len(atoms) == 1:
        return atoms[0]
    return Constant(operator.and_)(atoms[0], construct_conjunction(atoms[1:]))


def solve_map_query(graphical_model, query_atoms, evidence):
    if not all(is_valid_query_atom(atom) for atom in query_atoms):
        raise NeuroLangException('Invalid query atoms')
    if not isinstance(evidence, Instance):
        raise NeuroLangException('Evidence must be a Datalog instance')
    free_variables = set.union(
        *[extract_datalog_free_variables(atom) for atom in query_atoms]
    )
    query_predicate = Symbol('__q__')
    query_rule = Implication(
        query_predicate(*free_variables), construct_conjunction(query_atoms)
    )
    outcomes = solve_conditional_probability_query(graphical_model, evidence)
    prob_table = defaultdict(float)
    max_prob = 0.
    most_probable_tuple_value = None
    for outcome, prob in outcomes.value.table.items():
        program = ExpressionBlock(
            tuple(fact for fact in outcome) + (query_rule, )
        )
        datalog = DatalogBasic()
        datalog.walk(program)
        chaser = DatalogChase(datalog)
        solution_instance = chaser.build_chase_solution()
        for tuple_value in solution_instance[query_predicate]:
            prob_table[tuple_value] += prob
            if prob_table[tuple_value] >= max_prob:
                most_probable_tuple_value = tuple_value
                max_prob = prob_table[tuple_value]
    substitution = frozenset({
        var: value
        for var, value in zip(free_variables, most_probable_tuple_value)
    })
    return SetInstance({
        atom.functor: frozenset({apply_substitution(atom, substitution).args})
        for atom in query_atoms
    })
