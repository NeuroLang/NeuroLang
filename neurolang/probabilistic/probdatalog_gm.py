import itertools
from typing import Mapping, AbstractSet

import numpy as np
import pandas as pd

from ..expression_walker import PatternWalker
from ..expression_pattern_matching import add_match
from ..exceptions import NeuroLangException
from ..expressions import (
    Definition,
    Symbol,
    Constant,
    FunctionApplication,
    ExpressionBlock,
)
from ..datalog.expression_processing import extract_datalog_predicates
from ..relational_algebra import (
    RelationalAlgebraSolver,
    NaturalJoin,
    Projection,
    RenameColumn,
)
from ..utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
)
from .expressions import (
    VectorisedTableDistribution,
    ReindexVector,
    MultiplyVectors,
    SumVectors,
    SubtractVectors,
    RandomVariableVectorPointer,
    ParameterVectorPointer,
    IndexedGrounding,
)
from .probdatalog import (
    Grounding,
    is_probabilistic_fact,
    is_existential_probabilistic_fact,
)


class GraphicalModel(Definition):
    def __init__(self, edges, cpds, groundings):
        self.edges = edges
        self.cpds = cpds
        self.groundings = groundings

    @property
    def random_variables(self):
        return set(self.cpds.value)


def get_extensional_vectorised_table_distribution(grounding):
    return VectorisedTableDistribution(
        Constant[Mapping](
            {
                Constant[bool](False): Constant[float](0),
                Constant[bool](True): Constant[float](1),
            }
        ),
        grounding,
    )


def get_bernoulli_vectorised_table_distribution(p, grounding):
    return VectorisedTableDistribution(
        Constant[Mapping](
            {
                Constant[bool](False): Constant[float](1.0) - p,
                Constant[bool](True): p,
            }
        ),
        grounding,
    )


def get_parameterised_bernoulli_vectorised_table_distribution(
    parameters, grounding
):
    params_vect_symb = Symbol.fresh()
    return VectorisedTableDistribution(
        table=Constant[Mapping](
            {
                Constant[bool](False): SubtractVectors(
                    Constant[float](1.0),
                    ParameterVectorPointer(params_vect_symb),
                ),
                Constant[bool](True): ParameterVectorPointer(params_vect_symb),
            }
        ),
        grounding=grounding,
        parameters=Constant[Mapping]({params_vect_symb: parameters}),
    )


class AlgebraSet(NamedRelationalAlgebraFrozenSet):
    def __init__(self, columns, iterable=None):
        self._columns = tuple(columns)
        self._columns_sort = tuple(pd.Index(columns).argsort())
        if iterable is None:
            iterable = []

        if isinstance(iterable, RelationalAlgebraFrozenSet):
            self._initialize_from_instance_same_class(iterable)
        elif isinstance(iterable, pd.DataFrame):
            self._container = pd.DataFrame(iterable, columns=self.columns)
            self._container = self._renew_index(self._container)
        else:
            self._container = pd.DataFrame(
                list(iterable), columns=self._columns
            )
            self._container = self._renew_index(self._container)


def add_index_column(algebra_set, index_column):
    """Add an integer-location based index to the given algebra set."""
    new_columns = [index_column] + list(algebra_set.value.columns)
    return Constant[AbstractSet](
        AlgebraSet(
            columns=new_columns,
            iterable=pd.DataFrame(
                np.hstack(
                    [
                        np.transpose(
                            np.atleast_2d(np.arange(len(algebra_set.value)))
                        ),
                        algebra_set.value._container.values,
                    ]
                ),
                columns=new_columns,
            ),
        )
    )


def index_and_natural_join(algebra_sets):
    """
    Add index columns to all the given algebra sets and apply a natural join on
    all of them.

    This is useful for keeping track of the tuple locations in the ordered
    containers of the algebra sets.

    """
    solver = RelationalAlgebraSolver()
    index_columns = []
    result_set = None
    for algebra_set in algebra_sets:
        index_column = Symbol.fresh().name
        index_columns.append(index_column)
        indexed_algebra_set = add_index_column(algebra_set, index_column)
        if result_set is None:
            result_set = indexed_algebra_set
        else:
            result_set = solver.walk(
                NaturalJoin(result_set, indexed_algebra_set)
            )
    return result_set, index_columns


def rename_columns(algebra_set, new_columns):
    """Rename all the columns of the given algebra set."""
    if len(algebra_set.value.columns) != len(new_columns):
        raise NeuroLangException("New names for all columns should be passed.")
    if any(not isinstance(column, str) for column in new_columns):
        raise NeuroLangException("All column names should be strings")
    result = algebra_set
    old_columns = result.value.columns
    for old_column, new_column in zip(old_columns, new_columns):
        if new_column != old_column:
            result = RenameColumn(
                result, Symbol(old_column), Symbol(new_column)
            )
    solver = RelationalAlgebraSolver()
    return solver.walk(result)


def get_intensional_vectorised_table_distribution(
    rule_grounding, parent_groundings
):
    solver = RelationalAlgebraSolver()
    consequent = rule_grounding.expression.consequent
    antecedents = list(
        extract_datalog_predicates(rule_grounding.expression.antecedent)
    )
    consequent_algebra_set = rule_grounding.algebra_set
    antecedent_algebra_sets = [
        solver.walk(
            rename_columns(
                parent_groundings[predicate.functor].algebra_set,
                list(arg.name for arg in predicate.args),
            )
        )
        for predicate in antecedents
    ]
    algebra_sets = [consequent_algebra_set] + antecedent_algebra_sets
    indexed_set, index_columns = index_and_natural_join(algebra_sets)
    rv_index_column = index_columns[0]
    parent_rv_index_columns = index_columns[1:]
    truth_probs = None
    for parent_rv_index_column, antecedent in zip(
        parent_rv_index_columns, antecedents
    ):
        vect = ReindexVector(
            RandomVariableVectorPointer(antecedent.functor),
            Projection(indexed_set, parent_rv_index_column),
        )
        if truth_probs is None:
            truth_probs = vect
        else:
            truth_probs = MultiplyVectors(truth_probs, vect)
    truth_probs = ReindexVector(
        truth_probs, Projection(indexed_set, rv_index_column)
    )
    return VectorisedTableDistribution(
        table=Constant[Mapping](
            {
                Constant[bool](False): SubtractVectors(
                    Constant[float](1), truth_probs
                ),
                Constant[bool](True): truth_probs,
            }
        ),
        grounding=IndexedGrounding(
            expression=rule_grounding.expression,
            algebra_set=rule_grounding.algebra_set,
            index_columns=Constant[Mapping](
                {
                    parent_rv_symb: Constant[str](parent_rv_index_column)
                    for parent_rv_symb, parent_rv_index_column in zip(
                        parent_groundings, parent_rv_index_columns
                    )
                }
            ),
        ),
    )


class TranslateGroundedProbDatalogToGraphicalModel(PatternWalker):
    def __init__(self):
        self.edges = dict()
        self.cpds = dict()
        self.groundings = dict()

    @add_match(
        ExpressionBlock,
        lambda block: all(
            isinstance(exp, Grounding) for exp in block.expressions
        ),
    )
    def block_of_groundings(self, block):
        for grounding in block.expressions:
            self.walk(grounding)
        return GraphicalModel(
            Constant[Mapping](self.edges),
            Constant[Mapping](self.cpds),
            Constant[Mapping](self.groundings),
        )

    @add_match(
        Grounding, lambda exp: isinstance(exp.expression, FunctionApplication)
    )
    def extensional_grounding(self, grounding):
        rv_symb = grounding.expression.functor
        self._add_grounding(rv_symb, grounding)
        self._add_random_variable(
            rv_symb, get_extensional_vectorised_table_distribution(grounding)
        )

    @add_match(Grounding, lambda exp: is_probabilistic_fact(exp.expression))
    def probfact_grounding(self, grounding):
        rv_symb = grounding.expression.consequent.body.functor
        self._add_grounding(rv_symb, grounding)
        self._add_random_variable(
            rv_symb,
            get_bernoulli_vectorised_table_distribution(
                grounding.expression.consequent.probability, grounding
            ),
        )

    @add_match(
        Grounding,
        lambda exp: is_existential_probabilistic_fact(exp.expression),
    )
    def existential_probfact_grounding(self, grounding):
        rv_symb = grounding.expression.consequent.body.body.functor
        self._add_grounding(rv_symb, grounding)
        parameters = [
            Symbol.fresh() for _ in range(len(grounding.algebra_set.value))
        ]
        self._add_random_variable(
            rv_symb,
            get_parameterised_bernoulli_vectorised_table_distribution(
                parameters, grounding
            ),
        )

    @add_match(Grounding)
    def rule_grounding(self, rule_grounding):
        rv_symb = rule_grounding.expression.consequent.functor
        self._add_grounding(rv_symb, rule_grounding)
        parent_groundings = {
            predicate.functor: self.groundings[predicate.functor]
            for predicate in extract_datalog_predicates(
                rule_grounding.expression.antecedent
            )
        }
        self._add_random_variable(
            rv_symb,
            get_intensional_vectorised_table_distribution(
                rule_grounding, parent_groundings
            ),
        )
        parent_rv_symbs = {
            pred.functor
            for pred in extract_datalog_predicates(
                rule_grounding.expression.antecedent
            )
        }
        self._add_edges(rv_symb, parent_rv_symbs)

    def _add_edges(self, rv_symb, parent_rv_symbs):
        if rv_symb not in self.edges:
            self.edges[rv_symb] = set()
        self.edges[rv_symb] |= parent_rv_symbs

    def _add_grounding(self, rv_symb, grounding):
        self.groundings[rv_symb] = grounding

    def _add_random_variable(self, rv_symb, cpd_factory):
        self._check_random_variable_not_already_defined(rv_symb)
        self.cpds[rv_symb] = cpd_factory

    def _check_random_variable_not_already_defined(self, rv_symb):
        if rv_symb in self.cpds:
            raise NeuroLangException(
                f"Already processed predicate symbol {rv_symb}"
            )


def _get_predicate_from_grounded_expression(expression):
    if is_probabilistic_fact(expression):
        return expression.consequent.body
    elif is_existential_probabilistic_fact(expression):
        return expression.consequent.body.body
    elif isinstance(expression, FunctionApplication):
        return expression
    else:
        return expression.consequent


class SuccQuery(Definition):
    def __init__(self, predicate):
        self.predicate = predicate


class SuccQueryGraphicalModelSolver(PatternWalker):
    def __init__(self, graphical_model):
        self.graphical_model = graphical_model

    @add_match(SuccQuery)
    def succ_query(self, query):
        actual_predicate = _get_predicate_from_grounded_expression(
            self.graphical_model[query.predicate.functor]
        )
        rv_symb = actual_predicate.functor
        parent_rv_symbs = sorted(list(self.graphical_model.edges[rv_symb]))
        if len(parent_rv_symbs):
            rule = self.graphical_model.groundings.value[rv_symb].expression
            parent_marginal_distribs = {
                pred.functor: self.walk(SuccQuery(pred))
                for pred in extract_datalog_predicates(rule.antecedent)
            }
            result = self.compute_marginal_distribution(
                self.graphical_model.cpds.value[rv_symb],
                parent_marginal_distribs,
            )
        else:
            result = self.compute_cpd(self.graphical_model.cpds[rv_symb], {})
        return RelationalAlgebraSolver().walk(
            NaturalJoin(
                result,
                _build_query_algebra_set(
                    query.predicate, self.graphical_model.groundings[rv_symb]
                ),
            )
        )

    def compute_marginal_distribution(self, rv_symb, parent_marginal_distribs):
        result = None
        parent_symbs = sorted(parent_marginal_distribs)
        for parent_values in itertools.product(
            *[
                (Constant[bool](False), Constant[bool](True))
                for _ in parent_symbs
            ]
        ):
            new_term = self.compute_cpd(
                rv_symb, dict(zip(parent_symbs, parent_values))
            )
            for parent_symb, parent_value in zip(parent_symbs, parent_values):
                new_term = MultiplyVectors(
                    new_term,
                    Projection(
                        parent_marginal_distribs[parent_symb], parent_value
                    ),
                )
            if result is None:
                result = new_term
            else:
                result = SumVectors(result, new_term)
        return result

    def compute_cpd(self, rv_symb, parent_values):
        computer = CPDCalculator(parent_values)
        return computer.walk(self.graphical_model.cpds[rv_symb])


class CPDCalculator(RelationalAlgebraSolver):
    def __init__(self, parent_values):
        self.parent_values = parent_values

    @add_match(VectorisedTableDistribution)
    def vectorised_table_distribution(self, distrib):
        columns, probs = zip(
            *[
                (rv_value.value, self.walk(rv_prob))
                for rv_value, rv_prob in sorted(
                    distrib.value.items(), key=lambda x: x[0].value
                )
            ]
        )
        return Constant[AbstractSet](
            AlgebraSet(
                iterable=pd.DataFrame(np.hstack(probs), columns=columns),
                columns=columns,
            )
        )

    @add_match(RandomVariableVectorPointer)
    def rv_vect_pointer(self, pointer):
        if pointer not in self.parent_values:
            raise NeuroLangException(
                "Pointer to unknown parent random variable"
            )
        return self.parent_values[pointer.name]

    @add_match(ReindexVector)
    def reindex_vector(self, reindex_op):
        _check_is_vector(reindex_op.vector)
        old_values = vect_algebra_set_to_nparray(reindex_op.vector)
        idx = vect_algebra_set_to_nparray(reindex_op.index)
        columns = reindex_op.vector.value.columns
        return nparray_to_vect_algebra_set(old_values[idx], columns)

    @add_match(SumVectors)
    def sum_vectors(self, sum_op):
        return apply_arithmetic_vect_binary_op(sum_op, np.sum)

    @add_match(MultiplyVectors)
    def multiply_vectors(self, multiply_op):
        return apply_arithmetic_vect_binary_op(multiply_op, np.prod)

    @add_match(SubtractVectors)
    def subtract_vectors(self, subtract_op):
        return apply_arithmetic_vect_binary_op(subtract_op, np.subtract)


def apply_arithmetic_vect_binary_op(op, np_fun):
    return nparray_to_vect_algebra_set(
        np_fun(
            np.vstack(
                [
                    vect_algebra_set_to_nparray(op.first),
                    vect_algebra_set_to_nparray(op.second),
                ]
            ),
            axis=0,
        ),
        columns=op.first.value.columns,
    )


def vect_algebra_set_to_nparray(vect_algebra_set):
    _check_is_vector(vect_algebra_set)
    return np.array(vect_algebra_set.value.itervalues())


def nparray_to_vect_algebra_set(numpy_array, columns):
    return Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            iterable=pd.DataFrame(numpy_array, columns=columns),
            columns=columns,
        )
    )


def _check_is_vector(algebra_set):
    if len(algebra_set.value.columns) != 1:
        raise NeuroLangException(
            "Not a vector algebra set. Expected only one column"
        )


def compute_marginal_probability(rv_cpd, parent_marginal_distribs):
    result = None
    for parent_values, rv_cpd_distrib in rv_cpd_distribs.items():
        if result is None:
            result = _multiply_computed_distribs(
                [rv_cpd_distrib]
                + [parent_marginal_distrib.value[parent_value]]
            )


def _build_query_algebra_set(query_predicate, grounding_columns):
    consts, cols = zip(
        *[
            (arg, col)
            for arg, col in zip(query_predicate.args, grounding_columns)
            if isinstance(arg, Constant)
        ]
    )
    return Constant[AbstractSet](
        AlgebraSet(iterable={tuple(consts)}, columns=cols)
    )
