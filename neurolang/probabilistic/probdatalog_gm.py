from typing import Mapping, AbstractSet
from collections import defaultdict

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
    ColumnStr,
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
    SubtractVectors,
    RandomVariablePointer,
)
from .probdatalog import Grounding, is_probabilistic_fact


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
                Constant[int](0): Constant[float](0),
                Constant[int](1): Constant[float](1),
            }
        ),
        grounding,
    )


def get_bernoulli_vectorised_table_distribution(p, grounding):
    return VectorisedTableDistribution(
        Constant[Mapping](
            {Constant[int](0): Constant[float](1.0) - p, Constant[int](1): p}
        ),
        grounding,
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
    """
    Rename all the columns of the given algebra set.

    This will create nested RenameColumn RA operations.

    """
    if len(algebra_set.value.columns) != len(new_columns):
        raise NeuroLangException("New names for all columns should be passed.")
    if any(not isinstance(column, str) for column in new_columns):
        raise NeuroLangException("All column names should be strings")
    result = algebra_set
    for i in range(len(new_columns)):
        if result.value.columns[i] != new_columns[i]:
            result = RenameColumn(
                result, Symbol(result.value.columns[i]), Symbol(new_columns[i])
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
    all_predicates = [consequent] + antecedents
    algebra_sets = [consequent_algebra_set] + antecedent_algebra_sets
    indexed_set, index_columns = index_and_natural_join(algebra_sets)
    rv_index_column = index_columns[0]
    parent_rv_index_columns = index_columns[1:]
    truth_probs = ReindexVector(
        MultiplyVectors(
            ReindexVector(
                RandomVariablePointer(antecedent.functor),
                Projection(indexed_set, parent_rv_index_column),
            )
            for parent_rv_index_column, antecedent in zip(
                parent_rv_index_columns, antecedents
            )
        ),
        Projection(indexed_set, rv_index_column),
    )
    return VectorisedTableDistribution(
        Constant[Mapping](
            {
                Constant[int](0): SubtractVectors(
                    Constant[int](1), truth_probs
                ),
                Constant[int](1): truth_probs,
            }
        ),
        rule_grounding,
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
        if rv_symb not in self.edges:
            self.edges[rv_symb] = set()
        self.edges[rv_symb] |= {
            pred.functor
            for pred in extract_datalog_predicates(
                rule_grounding.expression.antecedent
            )
        }

    def _add_grounding(self, pred_symb, grounding):
        self.groundings[pred_symb] = grounding

    def _add_random_variable(self, pred_symb, cpd_factory):
        self._check_random_variable_not_already_defined(pred_symb)
        self.cpds[pred_symb] = cpd_factory

    def _check_random_variable_not_already_defined(self, pred_symb):
        if pred_symb in self.cpds:
            raise NeuroLangException(
                f"Already processed predicate symbol {pred_symb}"
            )
