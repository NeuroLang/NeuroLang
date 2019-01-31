import operator
import itertools
import copy
from collections import defaultdict

from .expressions import (
    NeuroLangException, FunctionApplication, Constant, Definition
)
from .solver_datalog_naive import Implication, Fact
from .expression_walker import ExpressionWalker
from .expression_pattern_matching import add_match
from . import unification
from .generative_datalog import DeltaAtom


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


class GraphicalModelSolver(ExpressionWalker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_variables = set()
        self.facts = defaultdict(set)
        self.parents = defaultdict(set)
        self.intensional_predicate_rule_count = defaultdict(int)
        self.cpds = dict()

    def make_edb_cpd(self, predicate):
        def cpd(parents):
            if len(parents) != 0:
                raise NeuroLangException(
                    'Expected empty set of parents for extensional predicate'
                )
            return self.facts[predicate]

        return cpd

    def make_rule_cpd(self, rule):
        def cpd(parents):
            return infer(rule, set.union(*[self.sample(p) for p in parents]))

        return cpd

    def make_predicate_union_cpd(self, predicate):
        def cpd(parents):
            if len(parents) == 0:
                return set()
            return set.union(*[self.sample(p) for p in parents])

        return cpd

    def make_delta_term_cpd(self, delta_term):
        if delta_term.dist_name == 'bernoulli':

            def cpd(parents):
                if len(parents) > 0:
                    raise NeuroLangException(
                        'Expected empty set of parents for delta term'
                    )
                return np.random.randint(2)

            return cpd
        else:
            raise NeuroLangException('Unknown distribution')

    @add_match(Fact)
    def fact(self, expression):
        predicate = expression.consequent.functor.name
        if predicate not in self.random_variables:
            self.random_variables.add(predicate)
            self.cpds[predicate] = self.make_edb_cpd(predicate)
        self.facts[predicate].add(expression.consequent)
        return expression

    @add_match(Implication(Definition, ...))
    def add_gdatalog_rule(self, expression):
        predicate = expression.consequent.functor.name
        self.intensional_predicate_rule_count[predicate] += 1
        count = self.intensional_predicate_rule_count[predicate]
        rule_var_name = f'{predicate}_{count}'
        self.random_variables.add(rule_var_name)
        self.random_variables.add(predicate)
        self.parents[predicate].add(rule_var_name)
        self.cpds[rule_var_name] = self.make_rule_cpd(expression)
        self.cpds[predicate] = self.make_predicate_union_cpd(predicate)
        antecedent_literals = get_antecedent_literals(expression)
        antecedent_predicates = [l.functor.name for l in antecedent_literals]
        for predicate in antecedent_predicates:
            self.parents[rule_var_name].add(predicate)
        if isinstance(expression.consequent, DeltaAtom):
            delta_term = expression.consequenet.delta_term
            x_delta_term = f'\Delta_{predicate}^{count}'
            self.random_variables.add(x_delta_term)
            self.parents[rule_var_name].add(x_delta_term)
            self.cpds[x_delta_term] = self.make_delta_term_cpd(delta_term)
        return expression

    def sample(self, variable):
        return self.cpds[variable](self.parents[variable])
