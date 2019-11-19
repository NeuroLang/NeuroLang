from operator import and_, invert, or_
from typing import Any

from ..expression_walker import PatternWalker, add_match
from ..expressions import Constant, ExpressionBlock, FunctionApplication
from . import FALSE, TRUE, Conjunction, Disjunction, Implication, Negation


class LogicSolver(PatternWalker):
    @add_match(Conjunction)
    def evaluate_conjunction(self, expression):
        unsolved_formulas = tuple()
        for formula in expression.formulas:
            solved_formula = self.walk(formula)
            if isinstance(solved_formula, Constant):
                if not bool(solved_formula.value):
                    return FALSE
            else:
                unsolved_formulas += (solved_formula,)

        if len(unsolved_formulas) > 0:
            return TRUE
        else:
            return Conjunction(unsolved_formulas)

    @add_match(Disjunction)
    def evaluate_disjunction(self, expression):
        unsolved_formulas = tuple()
        for formula in expression.formulas:
            solved_formula = self.walk(formula)
            if isinstance(solved_formula, Constant):
                if bool(solved_formula.value):
                    return TRUE
            else:
                unsolved_formulas += (solved_formula,)

        if len(unsolved_formulas) == 0:
            return FALSE
        else:
            return Disjunction(unsolved_formulas)

    @add_match(Negation)
    def evaluate_negation(self, expression):
        solved_formula = self.walk(expression.formula)
        if isinstance(solved_formula, Constant):
            return Constant[bool](not solved_formula.value)
        return expression

    @add_match(Implication)
    def evaluate_implication(self, expression):
        solved_antecedent = self.walk(expression.antecedent)
        if isinstance(solved_antecedent, Constant):
            if bool(solved_antecedent.value):
                return self.walk(expression.consequent)
            else:
                return TRUE
        else:
            solved_consequent = self.walk(expression.consequent)
            if (
                solved_consequent is not expression.consequent or
                solved_antecedent is not expression.antecedent
            ):
                expression = self.walk(
                    Implication(solved_consequent, solved_antecedent)
                )
            return expression


class TranslateToLogic(PatternWalker):
    @add_match(FunctionApplication(Constant[Any](and_), ...))
    def build_conjunction(self, conjunction):
        args = tuple()
        for arg in conjunction.args:
            new_arg = self.walk(arg)
            if isinstance(new_arg, Conjunction):
                args += new_arg.formulas
            else:
                args += (new_arg,)

        return self.walk(Conjunction(args))

    @add_match(FunctionApplication(Constant[Any](or_), ...))
    def build_disjunction(self, disjunction):
        args = tuple()
        for arg in disjunction.args:
            new_arg = self.walk(arg)
            if isinstance(new_arg, Disjunction):
                args += new_arg.formulas
            else:
                args += (new_arg,)

        return self.walk(Disjunction(args))

    @add_match(FunctionApplication(Constant[Any](invert), ...))
    def build_negation(self, inversion):
        arg = self.walk(inversion.args[0])
        return self.walk(Negation(arg))

    @add_match(ExpressionBlock)
    def build_conjunction_from_expression_block(self, expression_block):
        formulas = tuple()
        for expression in expression_block.expressions:
            new_exp = self.walk(expression)
            formulas += (new_exp,)
        return self.walk(Disjunction(formulas))

    @add_match(
        Implication(..., FunctionApplication(Constant, ...)),
        lambda implication: (
            implication.antecedent.functor.value is and_ or
            implication.antecedent.functor.value is or_ or
            implication.antecedent.functor.value is invert
        )
    )
    def translate_implication(self, implication):
        new_consequent = self.walk(implication.consequent)
        new_antecedent = self.walk(implication.antecedent)
        if (
            new_consequent is not implication.consequent or
            new_antecedent is not implication.antecedent
        ):
            implication = self.walk(
                Implication(new_consequent, new_antecedent)
            )

        return implication
