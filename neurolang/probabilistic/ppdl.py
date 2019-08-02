import operator
from uuid import uuid1
from typing import Iterable

from ..expressions import NeuroLangException
from ..expressions import (
    Expression, ExpressionBlock, FunctionApplication, Symbol, Constant,
    ExistentialPredicate
)
from ..expression_walker import ExpressionBasicEvaluator
from ..expression_pattern_matching import add_match
from ..existential_datalog import Implication
from ..solver_datalog_naive import DatalogBasic, is_conjunctive_expression


def is_conjunction(expression):
    return (
        isinstance(expression, FunctionApplication) and
        isinstance(expression.functor, Constant) and
        expression.functor.value == operator.and_
    )


def get_conjunction_literals(expression):
    if is_conjunction(expression):
        return (
            get_conjunction_literals(expression.args[0]) +
            get_conjunction_literals(expression.args[1])
        )
    else:
        return [expression]


def get_antecedent_literals(rule):
    if not isinstance(rule, Implication):
        raise NeuroLangException('Implication expected')

    return get_conjunction_literals(rule.antecedent)


def get_antecedent_predicate_names(rule):
    antecedent_literals = get_antecedent_literals(rule)
    return [literal.functor.name for literal in antecedent_literals]


def is_gdatalog_rule(exp):
    return (
        isinstance(exp, Implication) and
        isinstance(exp.consequent, FunctionApplication) and
        sum(isinstance(arg, DeltaTerm) for arg in exp.consequent.args) == 1
    )


def get_dterm(datom):
    return next(arg for arg in datom.args if isinstance(arg, DeltaTerm))


def get_dterm_index(datom):
    return next(
        i for i, arg in enumerate(datom.args) if isinstance(arg, DeltaTerm)
    )


def extend_expression_block(block, to_add):
    '''
    Extend `ExpressionBlock` with another `ExpressionBlock` or
    an iterable of `Expression`.

    Parameters
    ----------
    block: ExpressionBlock
        The initial `ExpressionBlock` to which expressions will be added.
    to_add: ExpressionBlock or Expression iterable
        `Expression`s to be added to the `ExpressionBlock`.

    Returns
    -------
    new_block: ExpressionBlock
        A new `ExpressionBlock` containing the new expressions.

    '''
    if isinstance(to_add, ExpressionBlock):
        return ExpressionBlock(block.expressions + to_add.expressions)
    if isinstance(to_add, Iterable):
        if not all(isinstance(item, Expression) for item in to_add):
            raise NeuroLangException('Expected iterable on expressions')
        return ExpressionBlock(block.expressions + tuple(to_add))
    raise NeuroLangException('Expected ExpressionBlock or Expression iterable')


class DeltaSymbol(Symbol):
    def __init__(self, dist_name, n_terms):
        self.dist_name = dist_name
        self.n_terms = n_terms
        super().__init__(f'Result_{self.dist_name}_{self.n_terms}')

    def __repr__(self):
        return (
            f'Δ-Symbol{{{self.name}({self.dist_name}, '
            '{self.n_terms}): {self.type}}}'
        )

    def __hash__(self):
        return hash((self.dist_name, self.n_terms))


class DeltaTerm(FunctionApplication):
    def __repr__(self):
        return f'Δ-term{{{self.functor}({self.args}): {self.type}}}'


class GenerativeDatalog(DatalogBasic):
    @add_match(Implication(FunctionApplication, ...), lambda exp:
               any(isinstance(arg, DeltaTerm) for arg in exp.consequent.args))
    def gdatalog_rule(self, rule):
        if not is_gdatalog_rule(rule):
            raise NeuroLangException(f'Invalid gdatalog rule: {rule}')
        predicate = rule.consequent.functor.name

        if predicate in self.protected_keywords:
            raise NeuroLangException(f'symbol {predicate} is protected')

        if not is_conjunctive_expression(rule.antecedent):
            raise NeuroLangException('Rule antecedent has to be a conjunction')

        if predicate in self.symbol_table:
            eb = self.symbol_table[predicate]
        else:
            eb = ExpressionBlock(tuple())

        self.symbol_table[predicate] = extend_expression_block(eb, [rule])

        return rule


def get_antecedent_constant_indexes(rule):
    '''Get indexes of constants occurring in antecedent predicates.'''
    constant_indexes = dict()
    for antecedent in get_antecedent_literals(rule):
        predicate = antecedent.functor.name
        indexes = {
            i
            for i, arg in enumerate(antecedent.args)
            if isinstance(arg, Constant)
        }
        if len(indexes) > 0:
            constant_indexes[predicate] = indexes
    return constant_indexes


def get_predicate_probabilistic_rules(gdatalog, predicate):
    if predicate not in gdatalog.symbol_table:
        return set()
    return set(
        rule for rule in gdatalog.symbol_table[predicate].expressions
        if is_gdatalog_rule(rule)
    )


def can_lead_to_object_uncertainty(gdatalog):
    '''Figure out if a walked GDatalog[Δ] program can lead to object
    uncertainty.

    Object uncertainty happens when there is a rule in the program such that:
        - there exist an atom in the antecedent of the rule such that the
          predicate of this atom is defined by a distributional rule
        - the argument at the location of that

    Parameters
    ----------
    gdatalog: GenerativeDatalog
        Instance that already walked the GDatalog[Δ] program.

    Returns
    -------
    has_object_uncertainty: bool
        Whether the program can generate object uncertainty or not.
    '''
    for key, value in gdatalog.symbol_table.items():
        if (
            key not in gdatalog.protected_keywords and
            isinstance(value, ExpressionBlock)
        ):
            for rule in value.expressions:
                for antecedent_predicate, constant_indexes in (
                    get_antecedent_constant_indexes(rule).items()
                ):
                    for rule in get_predicate_probabilistic_rules(
                        gdatalog, antecedent_predicate
                    ):
                        dterm_idx = get_dterm_index(rule.consequent)
                        if any(idx == dterm_idx for idx in constant_indexes):
                            return True
    return False


class TranslateGDatalogToEDatalog(ExpressionBasicEvaluator):
    @add_match(
        ExpressionBlock,
        lambda block: any(is_gdatalog_rule(e) for e in block.expressions)
    )
    def convert_expression_block_to_edatalog(self, block):
        expressions = tuple()
        for exp in block.expressions:
            res = self.walk(exp)
            if isinstance(res, ExpressionBlock):
                expressions += res.expressions
            else:
                expressions += (res, )
        return self.walk(ExpressionBlock(expressions))

    @add_match(Implication, is_gdatalog_rule)
    def convert_gdatalog_rule_to_edatalog_rules(self, expression):
        datom = expression.consequent
        dterm = get_dterm(datom)
        y = Symbol[dterm.type]('y_' + str(uuid1()))
        result_args = (
            dterm.args + (Constant(datom.functor.name), ) +
            tuple(arg for arg in datom.args
                  if not isinstance(arg, DeltaTerm)) + (y, )
        )
        result_atom = FunctionApplication(
            DeltaSymbol(dterm.functor.name, len(datom.args)), result_args
        )
        first_rule = Implication(
            ExistentialPredicate(y, result_atom), expression.antecedent
        )
        second_rule = Implication(
            FunctionApplication(
                Symbol(datom.functor.name),
                tuple(
                    arg if not isinstance(arg, DeltaTerm) else y
                    for arg in datom.args
                )
            ), expression.antecedent & result_atom
        )
        return self.walk(ExpressionBlock((first_rule, second_rule)))

    @add_match(Expression)
    def other_expressions(self, expression):
        return expression
