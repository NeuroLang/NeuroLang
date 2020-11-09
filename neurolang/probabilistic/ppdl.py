from ..datalog.basic_representation import (
    DatalogProgram,
    UnionOfConjunctiveQueries,
)
from ..exceptions import ForbiddenExpressionError
from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionWalker, PatternWalker
from ..expressions import FunctionApplication, Symbol
from ..logic import Implication, Union


def is_ppdl_rule(exp):
    return (
        isinstance(exp, Implication)
        and isinstance(exp.consequent, FunctionApplication)
        and sum(isinstance(arg, PPDLDeltaTerm) for arg in exp.consequent.args)
        == 1
    )


def get_dterm(datom):
    return next(arg for arg in datom.args if isinstance(arg, PPDLDeltaTerm))


def get_dterm_index(datom):
    return next(
        i for i, arg in enumerate(datom.args) if isinstance(arg, PPDLDeltaTerm)
    )


class PPDLDeltaSymbol(Symbol):
    def __init__(self, dist_name, n_terms):
        self.dist_name = dist_name
        self.n_terms = n_terms
        super().__init__(f"Result_{self.dist_name}_{self.n_terms}")

    def __repr__(self):
        return (
            f"Δ-Symbol{{{self.name}({self.dist_name}, "
            "{self.n_terms}): {self.type}}}"
        )

    def __hash__(self):
        return hash((self.dist_name, self.n_terms))


class PPDLDeltaTerm(FunctionApplication):
    def __repr__(self):
        return f"Δ-term{{{self.functor}({self.args}): {self.type}}}"


class PPDLMixin(PatternWalker):
    @add_match(
        Implication(FunctionApplication, ...),
        lambda exp: any(
            isinstance(arg, PPDLDeltaTerm) for arg in exp.consequent.args
        ),
    )
    def ppdl_rule(self, rule):
        if not is_ppdl_rule(rule):
            raise ForbiddenExpressionError(f"Invalid PPDL rule: {rule}")
        pred_symb = rule.consequent.functor
        pred_symb = pred_symb.cast(UnionOfConjunctiveQueries)
        if pred_symb in self.protected_keywords:
            raise ForbiddenExpressionError(f"symbol {pred_symb} is protected")
        if pred_symb in self.symbol_table:
            formulas = self.symbol_table[pred_symb].formulas
        else:
            formulas = tuple()
        formulas += (rule,)
        self.symbol_table[pred_symb] = Union(formulas)
        return rule


class PPDL(PPDLMixin, DatalogProgram, ExpressionWalker):
    """
    Class implementing a Probabilistic Programming DataLog (PPDL) program
    """

    pass
