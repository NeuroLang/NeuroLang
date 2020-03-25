import operator
from typing import Iterable

from ..datalog.expression_processing import extract_logic_predicates
from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionBasicEvaluator
from ..expressions import (
    Constant,
    Expression,
    ExpressionBlock,
    FunctionApplication,
    NeuroLangException,
    Symbol,
)
from ..logic import Union, ExistentialPredicate, Implication
from ..solver_datalog_naive import DatalogBasic, is_conjunctive_expression
from .expressions import DeltaTerm, DeltaSymbol
from .expression_processing import (
    is_ppdl_rule,
    get_dterm,
    get_dterm_index,
    get_antecedent_constant_indexes,
    get_predicate_probabilistic_rules,
)


class PPDLProgram(DatalogBasic):
    @add_match(
        Implication(FunctionApplication, ...),
        lambda exp: any(
            isinstance(arg, DeltaTerm) for arg in exp.consequent.args
        ),
    )
    def ppdl_rule(self, rule):
        if not is_ppdl_rule(rule):
            raise NeuroLangException(f"Invalid ppdl rule: {rule}")
        predicate = rule.consequent.functor.name

        if predicate in self.protected_keywords:
            raise NeuroLangException(f"symbol {predicate} is protected")

        if not is_conjunctive_expression(rule.antecedent):
            raise NeuroLangException("Rule antecedent has to be a conjunction")

        if predicate in self.symbol_table:
            disj = self.symbol_table[predicate].formulas
        else:
            disj = tuple()

        self.symbol_table[predicate] = Union(disj + (rule,))

        return rule


def can_lead_to_object_uncertainty(ppdl):
    """
    Figure out if a walked PPDL program can lead to object uncertainty.

    Object uncertainty happens when there is a rule in the program such that:
        - there exist an atom in the antecedent of the rule such that the
          predicate of this atom is defined by a distributional rule
        - the argument at the location of that

    Parameters
    ----------
    ppdl: PPDLProgram
        Instance that already walked the PPDL program's code.

    Returns
    -------
    bool
        Whether the program can generate object uncertainty or not.

    """
    for key, value in ppdl.symbol_table.items():
        if key not in ppdl.protected_keywords and isinstance(value, Union):
            for rule in value.formulas:
                for (
                    antecedent_predicate,
                    constant_indexes,
                ) in get_antecedent_constant_indexes(rule).items():
                    for rule in get_predicate_probabilistic_rules(
                        ppdl, antecedent_predicate
                    ):
                        dterm_idx = get_dterm_index(rule.consequent)
                        if any(idx == dterm_idx for idx in constant_indexes):
                            return True
    return False


class PPDLToExistentialDatalogTranslator(ExpressionBasicEvaluator):
    @add_match(
        ExpressionBlock,
        lambda block: any(is_ppdl_rule(e) for e in block.expressions),
    )
    def convert_expression_block_to_edatalog(self, block):
        expressions = tuple()
        for exp in block.expressions:
            res = self.walk(exp)
            if isinstance(res, ExpressionBlock):
                expressions += res.expressions
            else:
                expressions += (res,)
        return self.walk(ExpressionBlock(expressions))

    @add_match(Implication, is_ppdl_rule)
    def convert_ppdl_rule_to_edatalog_rules(self, expression):
        datom = expression.consequent
        dterm = get_dterm(datom)
        y = Symbol[dterm.type].fresh()
        result_args = (
            dterm.args
            + (Constant(datom.functor.name),)
            + tuple(
                arg for arg in datom.args if not isinstance(arg, DeltaTerm)
            )
            + (y,)
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
                ),
            ),
            expression.antecedent & result_atom,
        )
        return self.walk(ExpressionBlock((first_rule, second_rule)))

    @add_match(Expression)
    def other_expressions(self, expression):
        return expression
