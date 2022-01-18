import collections
import itertools
import operator
from typing import AbstractSet

from ..expression_pattern_matching import add_match
from ..expression_walker import (
    ExpressionWalker,
    PatternWalker,
    ReplaceExpressionWalker
)
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import Conjunction, ExistentialPredicate
from ..logic.expression_processing import extract_logic_atoms
from ..logic.transformations import (
    RemoveDuplicatedConjunctsDisjuncts,
    RemoveTrivialOperations,
    convert_to_pnf_with_dnf_matrix
)
from ..relational_algebra import (
    ExtendedProjection,
    FunctionApplicationListMember,
    Product,
    Projection,
    Selection,
    int2columnint_constant
)
from .exceptions import NotEasilyShatterableError
from .probabilistic_ra_utils import ProbabilisticFactSet
from .transforms import convert_to_dnf_ucq

EQ = Constant(operator.eq)
GT = Constant(operator.gt)
LT = Constant(operator.lt)


def is_ranked(query: Constant) -> bool:
    """Check if a query in UCQ format is ranked.
    That means that whenever to variables appear in an atom,
    they always appear in the same order.

    Parameters
    ----------
    query : Expression
        Logic expression in UCQ format. 

    Returns
    -------
    bool
        True if the query is ranked, False if not. 
    """
    prenex = convert_to_pnf_with_dnf_matrix(query)
    atoms = extract_logic_atoms(prenex)
    anterior_variables = {}
    existential_variables = set()
    while isinstance(prenex, ExistentialPredicate):
        anterior_variables[prenex.head] = set()
        existential_variables.add(prenex.head)
        prenex = prenex.body

    for atom in atoms:
        args = atom.args
        for i, arg in enumerate(args):
            if arg not in existential_variables:
                continue
            arg_and_posteriors = set(args[i:]) & existential_variables
            anterior = set(args[: i]) & existential_variables
            anterior_variables[arg].update(anterior)
            if not anterior_variables[arg].isdisjoint(arg_and_posteriors):
                return False
    return True


def _to_rank(conjunction: Conjunction) -> bool:
    """Detects if a conjunction has a self-join that
    needs to be ranked. Specifically if there are two atoms
    `R(x_1, ..., x_n)` and `R(y_1, ..., y_n)` where either
    forall i: either x_i = y_i or there is no j such i != j and y_i = x_j.

    Parameters
    ----------
    conjunction : Conjunction
        conjunction to check for rankeable atoms.

    Returns
    -------
    bool
        [description]
    """
    atoms = collections.defaultdict(lambda: set())
    for formula in conjunction.formulas:
        if (
            isinstance(formula, FunctionApplication),
            isinstance(formula.functor, Symbol)
        ):
            atoms[formula.functor].add(formula)
            if len(atoms[formula.functor]) > 1:
                return True
    return False


class Rank(PatternWalker):
    @add_match(Conjunction, _to_rank)
    def conjunction_rank_expression(self, expression):
        formulas = tuple()
        atoms = collections.defaultdict(lambda: set())
        for formula in expression.formulas:
            if (
                isinstance(formula, FunctionApplication) and
                isinstance(formula.functor, Symbol) and
                isinstance(
                    self.symbol_table[formula.functor], ProbabilisticFactSet
                )
            ):
                atoms[formula.functor].add(formula)
            else:
                formulas += (formula,)
        for relational_symbol in atoms:
            if len(atoms[relational_symbol]) == 1:
                formulas += (atoms[relational_symbol],)
                del atoms[relational_symbol]

        for relational_symbol in atoms:
            ranked_formulas = self.rank_expression(
                atoms[relational_symbol]
            )

            formulas += ranked_formulas

        return Conjunction(formulas)

    def rank_expression(self, expression_set):
        args = {
            int2columnint_constant(i): set()
            for i in range(len(next(iter(expression_set)).args))
        }
        for formula in expression_set:
            for i, arg in enumerate(formula.args):
                args[int2columnint_constant(i)].add(arg)

        conditions = dict()
        for k, v in args.items():
            if len(v) == 1:
                continue
            conditions_ = []
            for a, b in itertools.combinations(v, 2):
                conditions_.append((
                    FunctionApplication(LT, (a, b)),
                    FunctionApplication(GT, (a, b)),
                    FunctionApplication(EQ, (a, b))
                ))
            conditions[k] = conditions_

        global_conditions = []
        all_equal_conditions = []
        for cs in itertools.product(*itertools.chain(*conditions.values())):
            if all(c.functor == EQ for c in cs):
                all_equal_conditions.append(cs)
            else:
                global_conditions.append(cs)

        output_args = sum((a.args for a in expression_set), tuple())
        output_arg_indices = {
            a: int2columnint_constant(i + 1) for i, a in enumerate(output_args)
        }
        rew = ReplaceExpressionWalker(output_arg_indices)
        global_conditions_ix = [rew.walk(c) for c in global_conditions]
        all_equal_conditions_ix = [rew.walk(c) for c in all_equal_conditions]
        rset = self.symbol_table[next(iter(expression_set)).functor]

        prob_args_count = len(output_args) // 2 + 1
        new_sets = tuple(
            ProbabilisticFactSet(ExtendedProjection(
                Selection(Product((rset.relation, rset.relation)), condition),
                (
                    FunctionApplicationListMember(
                        int2columnint_constant(0) * int2columnint_constant(prob_args_count),
                        int2columnint_constant(0)
                    ),
                ) + tuple(
                    FunctionApplicationListMember(
                        int2columnint_constant(i),
                        int2columnint_constant(i)
                    ) for i in range(1, prob_args_count)
                ) + tuple(
                    FunctionApplicationListMember(
                        int2columnint_constant(i),
                        int2columnint_constant(i - 1)
                    ) for i in range(prob_args_count + 1, 2 * prob_args_count)

                )),
            int2columnint_constant(0))
            for condition in global_conditions_ix
        ) + (
            ProbabilisticFactSet(
                ExtendedProjection(
                    rset.relation,
                    tuple(
                        FunctionApplicationListMember(
                            int2columnint_constant(i),
                            int2columnint_constant(i)
                        )
                        for i in range(prob_args_count)
                    ) + tuple(
                        FunctionApplicationListMember(
                            int2columnint_constant(i),
                            int2columnint_constant(i + prob_args_count)
                        )
                        for i in range(1, prob_args_count)
                    )
                )
            )
            for condition in all_equal_conditions_ix
        )
        formulas = Disjunction(tuple(
            FunctionApplication(new_set, output_args)
            for new_set in new_sets
        ))
