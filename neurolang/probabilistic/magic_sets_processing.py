from neurolang.type_system import get_generic_type
from neurolang.probabilistic.expressions import PROB, ProbabilisticQuery
from neurolang.expressions import FunctionApplication, Symbol
from neurolang.logic import Implication
from neurolang.expression_pattern_matching import add_match
from neurolang.expression_walker import PatternWalker
import numpy as np
from .stratification import reachable_code_from_query
from neurolang.datalog.basic_representation import DatalogProgram
from ..datalog.expression_processing import (
    dependency_matrix,
)


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
    replacements = {}
    code = ()
    wlq_expressions = set()
    for wlq_symb_idx in wlq_symb_idxs:
        wlq_symb_deps = _wlq_dependencies(wlq_symb_idx, dep_mat, wlq_symb_idxs)
        for to_replace in wlq_symb_deps:
            wlq_symb = idb_symbs[to_replace]
            if wlq_symb not in replacements:
                wlq_expressions.add(program.within_language_prob_queries()[wlq_symb].consequent)
                new_name = f"{wlq_symb.name}_noprob"
                replacements[wlq_symb] = Symbol(new_name)
                code += reachable_code_from_query(
                    program.within_language_prob_queries()[wlq_symb],
                    program,
                ).formulas

    new_code = ReplaceWLQWalker(replacements, wlq_expressions).walk(code)

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


class ReplaceWLQWalker(PatternWalker):
    """
    Initialize the walker with matchings {S{D} : S{D2}}.
    The walker will then match rules that look like
    D(x, y, PROB) :- A(x, y)

    and replace them by

    D(x, y, PROB) :- D2(x, y)
    D2(x, y) :- A(x, y)
    """

    def __init__(self, wlq_replacements, wlq_expressions):
        self.wlq_replacements = wlq_replacements
        self.wlq_expressions = wlq_expressions
        self.replaced = set()

    @add_match(
        Implication,
        lambda implication: isinstance(
            implication.consequent, FunctionApplication
        )
        and any(_is_prob_arg(arg) for arg in implication.consequent.args),
    )
    def replace_wlq(self, impl):
        functor = impl.consequent.functor
        if (
            functor in self.wlq_replacements
            and functor not in self.replaced
        ):
            new_functor = self.wlq_replacements[functor]
            non_prob_args = tuple(
                arg for arg in impl.consequent.args if not _is_prob_arg(arg)
            )
            new_wlq = Implication(
                impl.consequent.functor, new_functor(non_prob_args)
            )
            new_non_wlq = Implication(
                new_functor(non_prob_args), impl.antecedent
            )
            return (new_wlq, new_non_wlq)
        return impl
