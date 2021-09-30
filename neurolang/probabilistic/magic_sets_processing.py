import numpy as np

from ..datalog.basic_representation import DatalogProgram
from ..datalog.expression_processing import dependency_matrix
from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionWalker
from ..expressions import FunctionApplication, Symbol
from ..logic import Implication, Union
from ..type_system import get_generic_type
from .expressions import PROB, ProbabilisticQuery
from .stratification import reachable_code_from_query


def probabilistic_postprocess_magic_rules(
    program: DatalogProgram, query: Implication, magic_rules: Union
):
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
    wlq_exprs = set()
    rules_to_update = set()
    for wlq_symb_idx in wlq_symb_idxs:
        wlq_symb_deps = _wlq_dependencies(wlq_symb_idx, dep_mat, wlq_symb_idxs)
        for to_replace in wlq_symb_deps:
            wlq_symb = idb_symbs[to_replace]
            if wlq_symb not in replaced:
                wlq_exprs.add(
                    program.within_language_prob_queries()[wlq_symb].consequent
                )
                code = reachable_code_from_query(
                    program.within_language_prob_queries()[wlq_symb],
                    program,
                )
                for rule in code.formulas:
                    try:
                        rules_to_update.add(magic_rules.formulas.index(rule))
                    except ValueError:
                        pass

    processed_idb = update_rules_with_new_prob_expressions(
        magic_rules,
        rules_to_update,
        wlq_exprs,
    )
    query_predicate = query.consequent
    processed_query = next(
        r
        for r in processed_idb
        if r.consequent.functor == query_predicate.functor
    )
    return processed_query, Union(processed_idb)


def update_rules_with_new_prob_expressions(
    magic_rules, rules_to_update, wlq_exprs
):
    processed = []
    walker = ReplaceWLQWalker(wlq_exprs)
    for idx, rule in enumerate(magic_rules.formulas):
        if idx in rules_to_update:
            updated = walker.walk(rule)
            if isinstance(updated, tuple):
                processed.extend(updated)
            else:
                processed.append(updated)
        else:
            processed.append(rule)
    return processed


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

    def __init__(self, wlq_expressions):
        self.wlq_expressions = wlq_expressions
        self.symbols = {expr.functor for expr in self.wlq_expressions}

    @add_match(
        Implication,
        lambda implication: isinstance(
            implication.consequent, FunctionApplication
        )
        and any(_is_prob_arg(arg) for arg in implication.consequent.args),
    )
    def replace_probabilistic_queries(self, impl):
        functor = impl.consequent.functor
        if functor in self.symbols:
            new_symbol = Symbol(f"{functor.name}_noprob")
            non_prob_args = tuple(
                arg for arg in impl.consequent.args if not _is_prob_arg(arg)
            )
            new_wlq = Implication(
                impl.consequent.functor(*impl.consequent.args),
                new_symbol(*non_prob_args),
            )
            new_non_wlq = Implication(
                new_symbol(*non_prob_args), impl.antecedent
            )
            return (new_non_wlq, new_wlq)
        return impl

    @add_match(Implication)
    def implication(self, impl):
        return Implication(impl.consequent, self.walk(impl.antecedent))

    @add_match(FunctionApplication)
    def replace_probabilistic_predicate(self, expr):
        if expr.functor in self.symbols:
            wlq_expr = next(
                e for e in self.wlq_expressions if e.functor == expr.functor
            )
            non_prob_args = tuple(
                arg
                for arg, wlq_arg in zip(expr.args, wlq_expr.args)
                if not _is_prob_arg(wlq_arg)
            )
            new_expr = Symbol(f"{expr.functor.name}_noprob")(*non_prob_args)
            return new_expr
        return self.process_expression(expr)
