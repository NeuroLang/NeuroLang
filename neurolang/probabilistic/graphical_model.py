import itertools
from collections import defaultdict
from typing import Iterable, Callable, AbstractSet, Mapping

import numpy as np

from ..expressions import (
    Expression, NeuroLangException, FunctionApplication, Constant, Symbol,
    Definition, ExpressionBlock
)
from ..solver_datalog_naive import Implication, Fact
from ..expression_walker import ExpressionWalker
from ..expression_pattern_matching import add_match
from .. import unification
from .ppdl import (
    DeltaTerm, get_antecedent_predicate_names, get_antecedent_literals,
    is_gdatalog_rule, get_dterm
)


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


def group_facts_by_predicate(facts, predicates):
    result = defaultdict(set)
    for fact in facts:
        pred = fact.consequent.functor.name
        if pred in predicates:
            result[pred].add(fact)
    return result


def substitute_dterm(datom, value):
    return FunctionApplication[datom.type](
        datom.functor,
        tuple(
            value if isinstance(arg, DeltaTerm) else arg for arg in datom.args
        )
    )


def is_dterm_constant(dterm):
    return all(isinstance(param, Constant) for param in dterm.args)


def get_constant_dterm_table_cpd(dterm):
    if not is_dterm_constant(dterm):
        raise NeuroLangException('Expected a constant Δ-term')
    if dterm.functor.name == Constant[str]('bernoulli'):
        p = dterm.args[0].value
        return frozenset({
            (Constant[int](1), Constant[float](p)),
            (Constant[int](0), Constant[float](1.0 - p)),
        })


FactSet = AbstractSet[Fact]
FactSetSymbol = Symbol[FactSet]
FactSetTableCPD = Mapping[FactSet, float]
FactSetTableCPDFunctor = Definition[Callable[[Iterable[FactSet]],
                                             FactSetTableCPD]]


class ExtensionalTableCPDFunctor(FactSetTableCPDFunctor):
    def __init__(self, predicate):
        self.predicate = predicate
        self.facts = set()


class IntensionalTableCPDFunctor(FactSetTableCPDFunctor):
    def __init__(self, rule):
        self.rule = rule


class UnionFactSetTableCPDFunctor(FactSetTableCPDFunctor):
    def __init__(self, predicate):
        self.predicate = predicate


class GraphicalModel(Expression):
    def __init__(self):
        self.rv_to_cpd_functor = dict()
        self.parents = defaultdict(frozenset)

    def add_parent(self, child, parent):
        self.parents[child] = self.parents[child].union({parent})


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
        rv_symbol = FactSetSymbol(predicate)
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
        rule_rv_symbol = FactSetSymbol(f'{predicate}_{rule_id}')
        pred_rv_symbol = FactSetSymbol(f'{predicate}')
        if rule_rv_symbol in self.gm.rv_to_cpd_functor:
            raise NeuroLangException(
                f'Random variable {rule_rv_symbol} already defined'
            )
        self.gm.rv_to_cpd_functor[rule_rv_symbol
                                  ] = (IntensionalTableCPDFunctor(rule))
        for antecedent_pred in get_antecedent_predicate_names(rule):
            antecedent_rv_symbol = FactSetSymbol(f'{antecedent_pred}')
            self.gm.add_parent(rule_rv_symbol, antecedent_rv_symbol)
        if pred_rv_symbol not in self.gm.rv_to_cpd_functor:
            self.gm.rv_to_cpd_functor[pred_rv_symbol] = \
                UnionFactSetTableCPDFunctor(predicate)
            self.gm.add_parent(pred_rv_symbol, rule_rv_symbol)


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
    if is_gdatalog_rule(rule):
        result = dict()
        for cpd_entries in itertools.product(
            *[
                get_constant_dterm_table_cpd(get_dterm(dfact.consequent))
                for dfact in inferred_facts
            ]
        ):
            new_facts = set(
                Fact(substitute_dterm(dfact.consequent, entry[0]))
                for dfact, entry in zip(inferred_facts, cpd_entries)
            )
            prob = np.prod([entry[1].value for entry in cpd_entries])
            result[frozenset(new_facts)] = Constant[float](prob)
        return result
    else:
        return {frozenset(inferred_facts): Constant[float](1.0)}


class ConditionalProbabilityQuery(Definition):
    def __init__(self, evidence):
        if not isinstance(evidence, Constant[FactSet]):
            raise NeuroLangException('Expected evidence to be a fact set')
        self.evidence = evidence


class TableCPDGraphicalModelSolver(ExpressionWalker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graphical_model = None
        self.ordered_rvs = None

    @add_match(ExpressionBlock)
    def program(self, program):
        if self.graphical_model is not None:
            raise NeuroLangException('GraphicalModel already constructed')
        self.graphical_model = gdatalog2gm(program)
        self.ordered_rvs = sort_rvs(self.graphical_model)

    @add_match(ConditionalProbabilityQuery)
    def conditional_probability_query_resolution(self, query):
        outcomes = self.generate_possible_outcomes()
        sum_prob = 0.
        filtered = dict()
        for outcome, prob in outcomes.items():
            if query.evidence.value <= outcome:
                sum_prob += prob.value
                filtered[outcome] = prob
        for outcome, prob in filtered.items():
            filtered[outcome] = Constant[float](prob.value / sum_prob)
        return filtered

    def generate_possible_outcomes(self):
        if self.graphical_model is None:
            raise NeuroLangException(
                'No GraphicalModel generated. Try walking a program'
            )
        results = dict()
        self.generate_possible_outcomes_aux(
            0, dict(), Constant[float](1.0), results
        )
        return results

    def generate_possible_outcomes_aux(
        self, rv_idx, rv_values, result_prob, results
    ):
        if rv_idx >= len(self.ordered_rvs):
            result = frozenset.union(*rv_values.values())
            if result in results:
                old_prob = results[result].value
                new_prob = old_prob + result_prob.value
                results[result] = Constant[float](new_prob)
            else:
                results[result] = result_prob
        else:
            rv_symbol = self.ordered_rvs[rv_idx]
            cpd_functor = self.graphical_model.rv_to_cpd_functor[rv_symbol]
            parent_rvs = self.graphical_model.parents[rv_symbol]
            parent_values = tuple(
                Constant[FactSet](rv_values[rv]) for rv in parent_rvs
            )
            cpd = self.walk(cpd_functor(*parent_values)).value
            for facts, prob in cpd.items():
                new_rv_values = rv_values.copy()
                new_rv_values[rv_symbol] = facts
                self.generate_possible_outcomes_aux(
                    rv_idx + 1, new_rv_values,
                    Constant[float](result_prob.value * prob.value), results
                )

    @add_match(FunctionApplication(ExtensionalTableCPDFunctor, ...))
    def extensional_table_cpd(self, expression):
        return Constant[FactSetTableCPD]({
            frozenset(expression.functor.facts):
            Constant[float](1.0)
        })

    @add_match(FunctionApplication(UnionFactSetTableCPDFunctor, ...))
    def union_table_cpd(self, expression):
        parent_facts = frozenset(
        ).union(*[arg.value for arg in expression.args])
        return Constant[FactSetTableCPD]({parent_facts: Constant[float](1.0)})

    @add_match(FunctionApplication(IntensionalTableCPDFunctor, ...))
    def intensional_table_cpd(self, expression):
        parent_facts = frozenset(
        ).union(*[arg.value for arg in expression.args])
        rule = expression.functor.rule
        result = Constant[FactSetTableCPD](delta_infer1(rule, parent_facts))
        return result
