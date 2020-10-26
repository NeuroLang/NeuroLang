'''
Implentation of probabilistic query resolution for
hierarchical queries [^1]. Using this we apply the small dichotomy
theorem [^1, ^2]:
Let Q be a conjunctive query without self-joins or a non-repeating relational
algebra expression. Then:
* If Q is hierarchical, then P(Q) is in polynomial time, and can be computed
using only the lifted inference rules for join, negation, union, and
existential quantifier.

* If Q is not hierarchical, then P(Q) is #P-hard in the size of the database.

[^1]: Robert Fink and Dan Olteanu. A dichotomy for non-repeating queries with
negation in probabilistic databases. In Proceedings of the 33rd ACM
SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems, PODS ’14,
pages 144–155, New York, NY, USA, 2014. ACM.

[^2]: Nilesh N. Dalvi and Dan Suciu. Efficient query evaluation on
probabilistic databases. VLDB J., 16(4):523–544, 2007.
'''

import logging
from collections import defaultdict

from ..datalog.expression_processing import flatten_query
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..expression_walker import ExpressionWalker, add_match
from ..expressions import Constant, Symbol
from ..logic import Conjunction, Implication
from ..logic.expression_processing import extract_logic_predicates
from ..utils.orderedset import OrderedSet
from ..relational_algebra import (
    ColumnInt,
    ColumnStr,
    EliminateTrivialProjections,
    ExtendedProjection,
    ExtendedProjectionListMember,
    NameColumns,
    Projection,
    RelationalAlgebraPushInSelections,
    RelationalAlgebraStringExpression,
    str2columnstr_constant,
)
from ..relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceExpressionSemringSolver,
)
from ..utils import log_performance
from .exceptions import NotHierarchicalQueryException
from .expression_processing import (
    lift_optimization_for_choice_predicates,
    project_on_query_head,
)
from .probabilistic_ra_utils import (
    DeterministicFactSet,
    ProbabilisticChoiceSet,
    ProbabilisticFactSet,
    generate_probabilistic_symbol_table_for_query,
)
from .shattering import shatter_easy_probfacts

LOG = logging.getLogger(__name__)


def is_hierarchical_without_self_joins(query):
    '''
    Let Q be first-order formula. For each variable x denote at(x) the
    set of atoms that contain the variable x. We say that Q is hierarchical
    if forall x, y one of the following holds:
    at(x) ⊆ at(y) or at(x) ⊇ at(y) or at(x) ∩ at(y) = ∅.
    '''

    has_self_joins, atom_set = extract_atom_sets_and_detect_self_joins(
        query
    )

    if has_self_joins:
        return False

    variables = list(atom_set)
    for i, v in enumerate(variables):
        at_v = atom_set[v]
        for v2 in variables[i + 1:]:
            at_v2 = atom_set[v2]
            if not (
                at_v <= at_v2 or
                at_v2 <= at_v or
                at_v.isdisjoint(at_v2)
            ):
                LOG.info(
                    "Not hierarchical on variables %s %s",
                    v.name, v2.name
                )
                return False

    return True


def extract_atom_sets_and_detect_self_joins(query):
    has_self_joins = False
    predicates = extract_logic_predicates(query)
    seen_predicate_functor = set()
    atom_set = defaultdict(set)
    for predicate in predicates:
        functor = predicate.functor
        if functor in seen_predicate_functor:
            LOG.info(
                "Not hierarchical self join on variables %s",
                functor
            )
            has_self_joins = True
        seen_predicate_functor.add(functor)
        for variable in predicate.args:
            if not isinstance(variable, Symbol):
                continue
            atom_set[variable].add(functor)
    return has_self_joins, atom_set


class ProbSemiringSolver(RelationalAlgebraProvenanceExpressionSemringSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translated_probfact_sets = dict()

    @add_match(
        Projection,
        lambda exp: (
            isinstance(
                exp.relation,
                (
                    DeterministicFactSet,
                    ProbabilisticFactSet,
                    ProbabilisticChoiceSet
                )
            )
        )
    )
    def eliminate_superfluous_projection(self, expression):
        return self.walk(expression.relation)

    @add_match(DeterministicFactSet(Symbol))
    def deterministic_fact_set(self, deterministic_set):
        relation_symbol = deterministic_set.relation
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        relation = self.walk(relation_symbol)
        named_columns = tuple(
            str2columnstr_constant(f'col_{i}')
            for i in relation.value.columns
        )
        projection_list = [
            ExtendedProjectionListMember(
                Constant[RelationalAlgebraStringExpression](
                    RelationalAlgebraStringExpression(c.value),
                    verify_type=False
                ),
                c
            )
            for c in named_columns
        ]

        prov_column = ColumnStr(Symbol.fresh().name)
        provenance_set = self.walk(
            ExtendedProjection(
                NameColumns(relation, named_columns),
                tuple(projection_list) +
                (
                    ExtendedProjectionListMember(
                        Constant[float](1.),
                        str2columnstr_constant(prov_column)
                    ),
                )
            )
        )

        self.translated_probfact_sets[relation_symbol] = ProvenanceAlgebraSet(
           provenance_set.value, prov_column
        )
        return self.translated_probfact_sets[relation_symbol]

    @add_match(ProbabilisticFactSet(Symbol, ...))
    def probabilistic_fact_set(self, prob_fact_set):
        relation_symbol = prob_fact_set.relation
        if relation_symbol in self.translated_probfact_sets:
            return self.translated_probfact_sets[relation_symbol]

        relation = self.walk(relation_symbol)
        named_columns = tuple(
            str2columnstr_constant(f'col_{i}')
            for i in relation.value.columns
        )
        relation = NameColumns(
            relation, named_columns
        )
        relation = self.walk(relation)
        rap_column = ColumnStr(
            relation.value.columns[prob_fact_set.probability_column.value]
        )

        self.translated_probfact_sets[relation_symbol] = ProvenanceAlgebraSet(
           relation.value, rap_column
        )
        return self.translated_probfact_sets[relation_symbol]

    @add_match(ProbabilisticChoiceSet(Symbol, ...))
    def probabilistic_choice_set(self, prob_choice_set):
        return self.probabilistic_fact_set(prob_choice_set)

    @add_match(ProbabilisticFactSet)
    def probabilistic_fact_set_invalid(self, prob_fact_set):
        raise NotImplementedError()


class RAQueryOptimiser(
    EliminateTrivialProjections,
    RelationalAlgebraPushInSelections,
    ExpressionWalker,
):
    pass


def solve_succ_query(query, cpl_program):
    """
    Solve a SUCC query on a CP-Logic program.

    Parameters
    ----------
    query : Implication
        SUCC query of the form `ans(x) :- P(x)`.
    cpl_program : CPLogicProgram
        CP-Logic program on which the query should be solved.

    Returns
    -------
    ProvenanceAlgebraSet
        Provenance set labelled with probabilities for each tuple in the result
        set.

    """
    with log_performance(
        LOG, "Preparing query %s", init_args=(query.consequent.functor.name,),
    ):
        flat_query_body = flatten_query(query.antecedent, cpl_program)

    with log_performance(LOG, "Translation and lifted optimisation"):
        flat_query_body = lift_optimization_for_choice_predicates(
            flat_query_body, cpl_program
        )
        flat_query = Implication(query.consequent, flat_query_body)
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl_program, flat_query_body
        )
        shattered_query = shatter_easy_probfacts(flat_query, symbol_table)
        prob_pred_symbs = (
            cpl_program.pfact_pred_symbs
            | cpl_program.pchoice_pred_symbs
        )
        # note: this assumes that the shattering process does not change the
        # order of the antecedent's conjuncts
        shattered_query_probabilistic_body = Conjunction(
            tuple(
                shattered_conjunct
                for shattered_conjunct, flat_conjunct in zip(
                    shattered_query.antecedent.formulas,
                    flat_query.antecedent.formulas,
                )
                if flat_conjunct.functor in prob_pred_symbs
            )
        )
        if not is_hierarchical_without_self_joins(
            shattered_query_probabilistic_body
        ):
            LOG.info(
                "Query with conjunctions %s not hierarchical",
                shattered_query_probabilistic_body.formulas,
            )
            raise NotHierarchicalQueryException(
                "Query not hierarchical, algorithm can't be applied"
            )

        ra_query = TranslateToNamedRA().walk(shattered_query.antecedent)
        proj_cols = tuple(
            OrderedSet(
                str2columnstr_constant(arg.name)
                for arg in shattered_query.consequent.args
                if isinstance(arg, Symbol)
            )
        )
        ra_query = Projection(ra_query, proj_cols)
        ra_query = RAQueryOptimiser().walk(ra_query)

    with log_performance(LOG, "Run RAP query"):
        solver = ProbSemiringSolver(symbol_table)
        prob_set_result = solver.walk(ra_query)
        prob_set_result = project_on_query_head(
            shattered_query, prob_set_result
        )

    return prob_set_result
