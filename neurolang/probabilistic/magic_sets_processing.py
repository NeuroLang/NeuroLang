import numpy as np

from ..datalog.basic_representation import DatalogProgram
from ..datalog.expression_processing import dependency_matrix
from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionWalker
from ..expressions import FunctionApplication, Symbol
from ..logic import Implication
from ..type_system import get_generic_type
from .expressions import PROB, ProbabilisticQuery
from .stratification import reachable_code_from_query


def probabilistic_postprocess_magic_sets(program: DatalogProgram, query):
    """
    Within language probabilistic queries (wlq) cannot be solved when they
    depend on one another, which is something that Magic Sets rewrite can
    introduce.

    This post processing step transforms a program where a wlq depends on
    another, i.e.

    A(x, PROB) :- B(x), C(x)
    C(x) :- D(x, p)
    D(x, PROB) :- E(x)

    into

    A(x, PROB) :- B(x), C(x)
    C(x) :- D2(x)
    D(x, PROB) :- D2(x)
    D2(x) :- E(x)
    """
    idb = list(reachable_code_from_query(query, program).formulas)
    idb_symbs, dep_mat = dependency_matrix(program, idb)
    wlq_symbs = set(program.within_language_prob_queries()).intersection(
        idb_symbs
    )
    wlq_symb_idxs = {idb_symbs.index(wlq_symb) for wlq_symb in wlq_symbs}
    replaced = set()
    for wlq_symb_idx in wlq_symb_idxs:
        wlq_symb_deps = _wlq_dependencies(wlq_symb_idx, dep_mat, wlq_symb_idxs)
        for to_replace in wlq_symb_deps:
            wlq_symb = idb_symbs[to_replace]
            if wlq_symb not in replaced:
                wlq_expr = program.within_language_prob_queries()[
                    wlq_symb
                ].consequent
                code = reachable_code_from_query(
                    program.within_language_prob_queries()[wlq_symb],
                    program,
                ).formulas

                replaced_code = ReplaceWLQWalker(wlq_expr).walk(code)


def _wlq_dependencies(wlq_symb_idx, dep_mat, wlq_symb_idxs):
    stack = [wlq_symb_idx]
    deps = set()
    seen = set()
    while stack:
        dep_idxs = set(
            np.argwhere(dep_mat[stack.pop()].astype(bool)).flatten()
        )
        deps.update(wlq_symb_idxs.intersection(dep_idxs))
        stack += list(dep_idxs - seen)
        seen.update(dep_idxs)
    return deps


def _is_prob_arg(arg):
    return isinstance(arg, ProbabilisticQuery) or (
        get_generic_type(type(arg)) is FunctionApplication
        and arg.functor == PROB
    )


class ReplaceWLQWalker(ExpressionWalker):
    """
    Initialize the walker with matchings {S{D} : S{D2}}.
    The walker will then match rules that look like
    D(x, y, PROB) :- A(x, y)

    and replace them by

    D(x, y, PROB) :- D2(x, y)
    D2(x, y) :- A(x, y)
    """

    def __init__(self, wlq_expression):
        self.wlq_expression = wlq_expression
        self.symbol = wlq_expression.functor
        self.new_symbol = Symbol(f"{self.symbol.name}_noprob")

    @add_match(
        Implication,
        lambda implication: isinstance(
            implication.consequent, FunctionApplication
        )
        and any(_is_prob_arg(arg) for arg in implication.consequent.args),
    )
    def replace_probabilistic_queries(self, impl):
        functor = impl.consequent.functor
        if functor == self.symbol:
            non_prob_args = tuple(
                arg for arg in impl.consequent.args if not _is_prob_arg(arg)
            )
            new_wlq = Implication(
                impl.consequent.functor(*impl.consequent.args),
                self.new_symbol(*non_prob_args),
            )
            new_non_wlq = Implication(
                self.new_symbol(*non_prob_args), impl.antecedent
            )
            return (new_wlq, new_non_wlq)
        return impl

    @add_match(Implication)
    def implication(self, impl):
        return Implication(impl.consequent, self.walk(impl.antecedent))

    @add_match(FunctionApplication)
    def replace_probabilistic_predicate(self, expr):
        if expr.functor == self.symbol:
            non_prob_args = tuple(
                arg
                for arg, expr_arg in zip(expr.args, self.wlq_expression.args)
                if not _is_prob_arg(expr_arg)
            )
            new_expr = self.new_symbol(*non_prob_args)
            return new_expr
        return self.process_expression(expr)
