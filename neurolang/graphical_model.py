import operator
import itertools
from collections import defaultdict
import logging

import numpy as np

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


def sample_from_bernoulli(params):
    if len(params) != 1:
        raise NeuroLangException(
            f'Expected one and only one parameter for '
            'bernoulli distribution, got {}'.format(len(params))
        )
    if not isinstance(params[0], Constant) and params[0].type is float:
        raise NeuroLangException(
            f'Expected bernoulli param to be a float, '
            'got {}'.format(type(params[0]))
        )
    p = params[0].value
    print(p)
    return int(np.random.choice([1, 0], p=[p, 1 - p]))


def sample_from_distribution(dist_name, dist_params):
    if dist_name == 'bernoulli':
        return sample_from_bernoulli(dist_params)
    else:
        raise NeuroLangException(f'Unknown distribution: {dist_name}')


def get_datom_vars(datom):
    if not isinstance(datom, DeltaAtom):
        raise NeuroLangException(f'Expected DeltaAtom, got: {type(datom)}')
    datom_vars = set()
    for term in datom.terms:
        if isinstance(term, Symbol):
            datom_vars.add(term.name)
        elif isinstance(term, DeltaTerm):
            for param in term.dist_params:
                if isinstance(param, Symbol):
                    datom_vars.add(param.name)
    return datom_vars


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
            'Expected a list/tuple of function applications but got {}'.format(
                type(facts)
            )
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
        pred = fact.consequent.functor.name
        if pred in predicates:
            result[pred].add(fact)
    return [result[pred] for pred in predicates]


def delta_infer1(rule, facts):
    antecedent_predicate_names = set(get_antecedent_predicate_names(rule))
    facts_by_predicate = group_facts_by_predicate(
        facts, antecedent_predicate_names
    )
    result = set()
    for fact_list in itertools.product(*facts_by_predicate):
        new = produce(rule, fact_list)
        if new is not None:
            result = result.union(new)
    if isinstance(rule.consequent, DeltaAtom):
        new_result = set()
        delta_facts = [f for f in result if isinstance(f, DeltaAtom)]
        normal_facts = {f for f in result if not isinstance(f, DeltaAtom)}
        possible_substitutions_and_probs = [
            get_dterm_cpd(f.dterm) for f in delta_facts
        ]
        for possible_values in itertools.product(
            *[get_dterm_cpd(f.dterm) for f in delta_facts]
        ):
            substituted_facts = {f for f, _ in poss
            new_result.add((normal_facts.union(substituted_facts))
    else:
        return {(result, 1)}


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
        datom = rule.consequent
        if not isinstance(datom, DeltaAtom):
            raise NeuroLangException(
                f'Expected DeltaAtom: got: {type(datom)}'
            )
        dterm = datom.delta_term

        def f(parents):
            # set of facts sampled from parents
            facts = set.union(*[self.sample(p) for p in parents])
            csqt_vars = list(get_datom_vars(datom))
            # create temporary rule without delta term
            tmp_rule = Implication(
                FunctionApplication(
                    Symbol('TMP'), tuple(Symbol(s) for s in csqt_vars)
                ), rule.antecedent
            )
            tmp_facts = infer(tmp_rule, facts)
            facts_with_dterm_sample = set()
            for fact in tmp_facts:
                var_map = {
                    var: value
                    for var, value in zip(csqt_vars, fact.args)
                }
                dist_params = [
                    var_map[arg.name] if isinstance(arg, Symbol) else arg
                    for arg in dterm.dist_params
                ]
                sample = sample_from_distribution(dterm.dist_name, dist_params)
                logging.debug(f'{dterm.dist_name}: {sample}')

                def substitute_term(term):
                    if isinstance(term, Symbol):
                        return var_map[term.name]
                    elif isinstance(term, Constant):
                        return term
                    elif isinstance(term, DeltaTerm):
                        return Constant(sample)
                    else:
                        raise NeuroLangException(
                            'Unexpected term type: {type(term)}'
                        )
                facts_with_dterm_sample.add(
                    FunctionApplication[fact.type](
                        Symbol(rule_var_name),
                        tuple(substitute_term(term) for term in datom.terms)
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
