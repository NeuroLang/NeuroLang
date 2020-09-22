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
from ..relational_algebra import (
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
from .expression_processing import lift_optimization_for_choice_predicates
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
        rap_column = ColumnStr(f'col_{prob_fact_set.probability_column.value}')

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
    ExpressionWalker
):
    pass


def solve_succ_query(query_predicate, cpl_program):
    """
    Obtain the solution of a SUCC query on a CP-Logic program.

    The SUCC query must take the form

        SUCC[ P(x) ]
    """
    with log_performance(LOG, 'Preparing query'):
        if isinstance(query_predicate, Implication):
            conjunctive_query = query_predicate.antecedent
            variables_to_project = tuple(
                str2columnstr_constant(s.name)
                for s in query_predicate.consequent.args
                if isinstance(s, Symbol)
            )
        else:
            conjunctive_query = query_predicate
            variables_to_project = tuple(
                str2columnstr_constant(s.name)
                for s in query_predicate._symbols
                if s not in (
                    p.functor for p in
                    extract_logic_predicates(query_predicate)
                )
            )

        flat_query = flatten_query(conjunctive_query, cpl_program)

    with log_performance(LOG, "Translation and lifted optimisation"):
        flat_query = lift_optimization_for_choice_predicates(
            flat_query, cpl_program
        )
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl_program, flat_query
        )
        shattered_query = shatter_easy_probfacts(flat_query, symbol_table)
        shattered_query_formulas = set(shattered_query.formulas)
        prob_symbols = cpl_program.probabilistic_predicate_symbols
        query_pred_symbs = set(f.functor for f in flat_query.formulas)
        prob_symbols = prob_symbols.intersection(query_pred_symbs)
        prob_symbols = set(
            symbol_table[psymb].relation for psymb in prob_symbols
        )
        shattered_query_probabilistic_section = Conjunction(
            tuple(
                formula
                for formula in shattered_query_formulas
                if formula.functor.relation in prob_symbols
            )
        )
        flat_query = Conjunction(tuple(shattered_query_formulas))

        if not is_hierarchical_without_self_joins(
            shattered_query_probabilistic_section
        ):
            LOG.info(
                'Query with conjunctions %s not hierarchical',
                flat_query.formulas
            )
            raise NotHierarchicalQueryException(
                "Query not hierarchical, algorithm can't be applied"
            )

        ra_query = TranslateToNamedRA().walk(flat_query)
        ra_query = Projection(ra_query, variables_to_project)
        ra_query = RAQueryOptimiser().walk(ra_query)

    with log_performance(LOG, "Run RAP query"):
        solver = ProbSemiringSolver(symbol_table)
        prob_set_result = solver.walk(ra_query)

    return prob_set_result
