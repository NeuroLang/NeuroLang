import itertools
from typing import Mapping, AbstractSet, Tuple

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
    RelationalAlgebraOperation,
    NameColumns,
)
from ..utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
)
from .expressions import (
    GraphicalModel,
    VectorisedTableDistribution,
    ConcatenateColumn,
    AddIndexColumn,
    ArithmeticOperationOnColumns,
    SumColumns,
    MultiplyColumns,
    RandomVariableValuePointer,
    NegateProbability,
    AddRepeatedValueColumn,
    MultipleNaturalJoin,
)
from .probdatalog import (
    Grounding,
    is_probabilistic_fact,
    is_existential_probabilistic_fact,
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


def bernoulli_vect_table_distrib(p, grounding):
    if not isinstance(p, Constant[float]):
        raise NeuroLangException(
            "Bernoulli's parameter must be Constant[float]"
        )
    return VectorisedTableDistribution(
        Constant[Mapping](
            {
                Constant[bool](False): Constant[float](1.0 - p.value),
                Constant[bool](True): p,
            }
        ),
        grounding,
    )


def extensional_vect_table_distrib(grounding):
    return bernoulli_vect_table_distrib(Constant[float](1.0), grounding)


def get_var_columns(function_application):
    return Constant[Tuple](
        tuple(arg.name for arg in function_application.args)
    )


def and_vect_table_distribution(rule_grounding):
    return MultiplyColumns(
        MultipleNaturalJoin(
            NameColumns(
                get_var_columns(antecedent_pred),
                RandomVariableValuePointer(antecedent_pred.functor),
            )
            for antecedent_pred in extract_datalog_predicates(
                rule_grounding.expression.antecedent
            )
        )
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
            rv_symb, extensional_vect_table_distrib(grounding)
        )

    @add_match(Grounding, lambda exp: is_probabilistic_fact(exp.expression))
    def probfact_grounding(self, grounding):
        rv_symb = grounding.expression.consequent.body.functor
        self._add_grounding(rv_symb, grounding)
        self._add_random_variable(
            rv_symb,
            bernoulli_vect_table_distrib(
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
            rv_symb, and_vect_table_distribution(rule_grounding)
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


def _iter_parents(parent_marginal_probabilities, parent_groundings):
    parent_symbs = sorted(list(parent_marginal_probabilities))
    for parent_bool_values in itertools.product(
        *[(True, False) for _ in parent_symbs]
    ):
        parent_values = {
            parent_symb: AddRepeatedValueColumn(
                parent_groundings[parent_symb],
                Constant[bool](parent_bool_value),
            )
            for parent_symb, parent_bool_value in zip(
                parent_symbs, parent_bool_values
            )
        }
        parent_margin_probs = {
            parent_symb: parent_marginal_probabilities[parent_symb]
            if parent_bool_value
            else NegateProbability(parent_marginal_probabilities[parent_symb])
            for parent_symb, parent_bool_value in zip(
                parent_symbs, parent_bool_values
            )
        }
        yield parent_values, parent_margin_probs


def compute_marginal_probability(
    cpd, parent_marginal_probabilities, parent_groundings
):
    if not len(parent_marginal_probabilities):
        return ExtendedRelationalAlgebraSolver({}).walk(cpd)
    else:
        terms = []
        for parent_values, parent_marg_probs in _iter_parents(
            parent_marginal_probabilities, parent_groundings
        ):
            solver = ExtendedRelationalAlgebraSolver(parent_values)
            terms.append(
                MultiplyColumns(
                    MultipleNaturalJoin(
                        [solver.walk(cpd)] + list(parent_marg_probs.values())
                    ),
                    _make_numerical_col_symb(),
                )
            )
        return ExtendedRelationalAlgebraSolver({}).walk(
            SumColumns(MultipleNaturalJoin(terms), _make_numerical_col_symb)
        )


class ExtendedRelationalAlgebraSolver(RelationalAlgebraSolver):
    def __init__(self, rv_values):
        self.rv_values = rv_values

    @add_match(RandomVariableValuePointer)
    def rv_value_pointer(self, pointer):
        if pointer not in self.rv_values:
            raise NeuroLangException(
                f"Unknown value for random variable {pointer}"
            )
        return Constant[AbstractSet](self.rv_values[pointer])

    @add_match(ConcatenateColumn)
    def concatenate_column(self, concat_op):
        new_column_name = _get_column_name_from_expression(concat_op.column)
        return Constant[AbstractSet](
            AlgebraSet(
                iterable=pd.concat(
                    [
                        concat_op.relation.value._container,
                        pd.DataFrame(
                            {
                                new_column_name: np.array(
                                    concat_op.column_values.value
                                )
                            }
                        ),
                    ]
                ),
                columns=list(concat_op.relation.value.columns)
                + [new_column_name],
            )
        )

    @add_match(AddIndexColumn)
    def add_index_column(self, add_index_op):
        return self.walk(
            ConcatenateColumn(
                relation=add_index_op.relation,
                column=add_index_op.index_column,
                column_values=Constant[np.ndarray](
                    np.arange(len(add_index_op.relation.value))
                ),
            )
        )

    @add_match(SumColumns)
    def sum_columns(self, sum_op):
        return _apply_arithmetic_column_op(sum_op, np.sum)

    @add_match(MultiplyColumns)
    def multiply_columns(self, multiply_op):
        return _apply_arithmetic_column_op(multiply_op, np.prod)

    @add_match(AddRepeatedValueColumn)
    def add_repeated_value_column(self, add_op):
        return self.walk(
            ConcatenateColumn(
                relation=add_op.relation,
                column=_make_numerical_col_symb(),
                column_values=Constant[np.ndarray](
                    np.repeat(
                        add_op.repeated_value, len(add_op.relation.value)
                    )
                ),
            )
        )

    @add_match(VectorisedTableDistribution)
    def vectorised_table_distribution(self, distrib):
        truth_prob = distrib.table.value[Constant[bool](True)]
        if isinstance(truth_prob, Constant):
            return self.walk(
                AddRepeatedValueColumn(distrib.grounding.relation, truth_prob)
            )
        else:
            return self.walk(
                NaturalJoin(self.grounding.relation, self.walk(truth_prob))
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


def _make_numerical_col_symb():
    return Symbol("__numerical__" + Symbol.fresh().name)


def _get_relation_numerical_columns(relation):
    return (col for col in relation.columns if col.startswith("__numerical__"))


def _apply_arithmetic_column_op(op, numpy_op):
    numerical_columns = _get_relation_numerical_columns(op.relation)
    non_numerical_columns = [
        col for col in op.relation.columns if col not in numerical_columns
    ]
    new_column = _make_numerical_col_symb().name
    resulting_columns = [non_numerical_columns] + [new_column]
    iterable = pd.concat(
        [
            op.relation.value._container[non_numerical_columns],
            pd.DataFrame(
                {
                    new_column: numpy_op(
                        op.relation.value._container[numerical_columns], axis=1
                    )
                }
            ),
        ]
    )
    return Constant[AbstractSet](AlgebraSet(resulting_columns, iterable))


def _get_column_name_from_expression(column_exp):
    if isinstance(column_exp, Constant[str]):
        return column_exp.value
    elif isinstance(column_exp, Symbol):
        return column_exp.name
    else:
        raise NeuroLangException(
            "Cannot obtain column name from expression of type {}".format(
                type(column_exp)
            )
        )
