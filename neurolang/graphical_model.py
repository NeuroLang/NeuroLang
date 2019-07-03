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
    Definition
)
from .solver_datalog_naive import Implication, Fact
from .expression_walker import ExpressionWalker
from .expression_pattern_matching import add_match
from . import unification
from .generative_datalog import DeltaAtom, DeltaTerm


def replace_fa_functor_name(fa, new_functor_name):
    return FunctionApplication[fa.type](Symbol(new_functor_name), fa.args)


def sample_from_distribution(dist_name):
    if dist_name == 'bernoulli':
        return np.random.randint(2)
    else:
        raise NeuroLangException(f'Unknown distribution: {dist_name}')


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


def repr_fact(fact):
    return '{}({})'.format(
        fact.functor.name, ', '.join([str(arg.value) for arg in fact.args])
    )


def repr_factset(factset):
    return '{{ {} }}'.format(', '.join([repr_fact(fact) for fact in factset]))


def pprint_dist(dist):
    for factset, prob in dist:
        print('{} \t {}'.format(repr_factset(factset), prob))


class ProbabilityDistrlibution(Expression):
    def prob(self, value):
        raise NeuroLangException('Not implemented in this abstract class')


class DiscreteCPD(ProbabilityDistrlibution):
    def __init__(self, cpd_map):
        self.cpd_map = cpd_map

    def prob(self, value):
        if value not in self.cpd_map:
            raise NeuroLangException(
                f'Value {value} not in range of distribution'
            )
        return self.cpd_map[value]

    @property
    def support(self):
        return set(self.cpd_map.keys())


def const_repr(exp):
    if isinstance(exp, Constant):
        return str(exp.value)
    elif isinstance(exp, Symbol):
        return exp.name
    else:
        return '( {} )'.format(repr(exp))


class ArithmeticOperation(Expression):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs


class Addition(ArithmeticOperation):
    def __repr__(self):
        return '{} + {}'.format(const_repr(self.lhs), const_repr(self.rhs))


class Subtraction(ArithmeticOperation):
    def __repr__(self):
        return '{} - {}'.format(const_repr(self.lhs), const_repr(self.rhs))


class Multiplication(ArithmeticOperation):
    def __repr__(self):
        return '{} · {}'.format(const_repr(self.lhs), const_repr(self.rhs))


def to_sympy_aux(exp):
    if isinstance(exp, Constant):
        return sympy.sympify(exp.value)
    elif isinstance(exp, Symbol):
        return sympy.sympify(exp.name)
    elif isinstance(exp, ArithmeticOperation):
        return to_sympy(exp)


def to_sympy(exp):
    if isinstance(exp, Addition):
        return to_sympy_aux(exp.lhs) + to_sympy_aux(exp.rhs)
    elif isinstance(exp, Subtraction):
        return to_sympy_aux(exp.lhs) - to_sympy_aux(exp.rhs)
    elif isinstance(exp, Multiplication):
        return to_sympy_aux(exp.lhs) * to_sympy_aux(exp.rhs)
    else:
        raise NeuroLangException('invalide type: {}'.format(type(exp)))


def arithmetic_eq(exp1, exp2):
    return (to_sympy(exp1) - to_sympy(exp2)) == 0


def multiply_n_expressions(*expressions):
    if len(expressions) == 1:
        return expressions[0]
    else:
        return Multiplication(
            expressions[0], multiply_n_expressions(*expressions[1:])
        )


def get_dterm_dist(dterm):
    if dterm.dist_name == 'bernoulli':
        return DiscreteCPD({
            Constant[int](1):
            dterm.dist_params[0],
            Constant[int](0):
            Subtraction(Constant[int](1), dterm.dist_params[0])
        })
    else:
        raise NeuroLangException(
            f'Unknown probability distribution: {dterm.dist_name}'
        )


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
                get_dterm_dist(dfact.delta_term).cpd_map.items()
                for dfact in inferred_facts
            ]
        ):
            new_facts = set(
                substitute_dterm(dfact, entry[0])
                for dfact, entry in zip(inferred_facts, cpd_entries)
            )
            prob = multiply_n_expressions(*[entry[1] for entry in cpd_entries])
            new_result.add((frozenset(new_facts), prob))
        return new_result
    else:
        return {(frozenset(inferred_facts), Constant[int](1))}


class PossibleOutcome(Expression):
    def __init__(self, fact_set, probability):
        self.fact_set = fact_set
        self.probability = probability


class GraphicalModelSolver(ExpressionWalker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_variables = set()
        self.facts = defaultdict(set)
        self.parents = defaultdict(set)
        self.intensional_predicate_rule_count = defaultdict(int)
        self.samplers = dict()

    def make_edb_sampler(self, predicate):
        def f(parents):
            if len(parents) != 0:
                raise NeuroLangException(
                    'Expected empty set of parents for extensional predicate'
                )
            return self.facts[predicate]

        return f

    def make_rule_sampler(self, rule_var_name, rule):
        new_rule = Implication[rule.type](
            replace_fa_functor_name(rule.consequent, rule_var_name),
            rule.antecedent
        )

        def f(parents):
            facts = set.union(*[self.sample(p) for p in parents])
            return infer(new_rule, facts)

        return f

    def make_union_sampler(self, predicate):
        def f(parents):
            if len(parents) == 0:
                return set()
            return set.union(
                *[
                    set([
                        replace_fa_functor_name(x, predicate)
                        for x in self.sample(p)
                    ])
                    for p in parents
                ]
            )

        return f

    def make_gdatalog_rule_sampler(self, rule_var_name, rule):
        def f(parents):
            # set of facts sampled from parents
            facts = set.union(*[self.sample(p) for p in parents])
            # create temporary rule without delta term
            tmp_rule = Implication(
                FunctionApplication(
                    Symbol('TMP'), rule.consequent.terms_without_dterm
                ), rule.antecedent
            )
            tmp_facts = infer(tmp_rule, facts)
            facts_with_dterm_sample = set()
            dterm_index = rule.consequent.delta_term_index
            dterm = rule.consequent.terms[dterm_index]
            for fact in tmp_facts:
                sample = sample_from_distribution(dterm.dist_name)
                args_with_dterm = list(fact.args)
                args_with_dterm.insert(dterm_index, Constant(sample))
                facts_with_dterm_sample.add(
                    FunctionApplication[fact.type](
                        Symbol(rule_var_name), tuple(args_with_dterm)
                    )
                )
            return facts_with_dterm_sample

        return f

    @add_match(Fact)
    def fact(self, expression):
        predicate = expression.consequent.functor.name
        if predicate not in self.random_variables:
            self.random_variables.add(predicate)
            self.samplers[predicate] = self.make_edb_sampler(predicate)
        self.facts[predicate].add(expression.consequent)
        return expression

    @add_match(Implication(Definition, ...))
    def add_gdatalog_rule(self, rule):
        predicate = rule.consequent.functor.name
        self.intensional_predicate_rule_count[predicate] += 1
        count = self.intensional_predicate_rule_count[predicate]
        rule_var_name = f'{predicate}_{count}'
        self.random_variables.add(rule_var_name)
        self.random_variables.add(predicate)
        self.parents[predicate].add(rule_var_name)
        self.samplers[predicate] = self.make_union_sampler(predicate)
        antecedent_predicate_names = get_antecedent_predicate_names(rule)
        for pred in antecedent_predicate_names:
            self.parents[rule_var_name].add(pred)
        if isinstance(rule.consequent, DeltaAtom):
            dterm = rule.consequent.delta_term
            self.samplers[rule_var_name] = self.make_gdatalog_rule_sampler(
                rule_var_name, rule
            )
        else:
            self.samplers[rule_var_name] = self.make_rule_sampler(
                rule_var_name, rule
            )
        return rule

    def sample(self, variable):
        logging.debug('sampling %(variable)s', {'variable': variable})
        return self.samplers[variable](self.parents[variable])
