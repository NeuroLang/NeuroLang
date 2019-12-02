import itertools
from typing import Mapping, AbstractSet
import operator

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
from ..logic.expression_processing import extract_logic_predicates
from ..datalog.expressions import Conjunction, Implication
from ..relational_algebra import (
    RelationalAlgebraSolver,
    NaturalJoin,
    Projection,
    RenameColumn,
    Selection,
    ColumnStr,
    eq_,
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
    SumColumns,
    MultiplyColumns,
    DivideColumns,
    RandomVariableValuePointer,
    NegateProbability,
    AddRepeatedValueColumn,
    MultipleNaturalJoin,
    RenameColumns,
    Grounding,
    PfactGrounding,
    make_numerical_col_symb,
    Aggregation,
    ExtendedProjection,
    ExtendedProjectionListMember,
)
from .probdatalog import (
    ground_probdatalog_program,
    is_probabilistic_fact,
    is_existential_probabilistic_fact,
)


def succ_query(program_code, query_pred):
    grounded = ground_probdatalog_program(program_code)
    gm = TranslateGroundedProbDatalogToGraphicalModel().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    return solver.walk(SuccQuery(query_pred))


def marg_query(code, query_pred, evidence_pred):
    joint_rule = _build_joint_rule([query_pred, evidence_pred])
    extended_code = ExpressionBlock(list(code.expressions) + [joint_rule])
    grounded = ground_probdatalog_program(extended_code)
    gm = TranslateGroundedProbDatalogToGraphicalModel().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    evidence_prob = solver.walk(SuccQuery(evidence_pred))
    joint_prob = solver.walk(SuccQuery(joint_rule.consequent))
    evidence_prob_column = Constant(
        ColumnStr(_split_numerical_cols(evidence_prob)[1][0])
    )
    joint_prob_column = Constant(
        ColumnStr(_split_numerical_cols(joint_prob)[1][0])
    )
    return ExtendedRelationalAlgebraSolver({}).walk(
        DivideColumns(
            NaturalJoin(evidence_prob, joint_prob),
            joint_prob_column,
            evidence_prob_column,
        )
    )


def _build_joint_rule(joint_predicates):
    variables = set()
    for pred in joint_predicates:
        variables |= set(arg for arg in pred.args if isinstance(arg, Symbol))
    consequent = Symbol.fresh()(*sorted(variables, key=lambda arg: arg.name))
    antecedent = Conjunction(joint_predicates)
    return Implication(consequent, antecedent)


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

    def to_numpy(self):
        return self._container.values


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


def multi_bernoulli_vect_table_distrib(grounding):
    if not isinstance(grounding.params_relation, Constant[AbstractSet]):
        raise NeuroLangException(
            "Bernoulli's parameter must be Constant[AbstractSet]"
        )
    return VectorisedTableDistribution(
        Constant[Mapping](
            {
                Constant[bool](True): grounding.params_relation,
                Constant[bool](False): Constant[float](1.0)
                - grounding.params_relation,
            }
        ),
        grounding,
    )


def extensional_vect_table_distrib(grounding):
    return bernoulli_vect_table_distrib(Constant[float](1.0), grounding)


def get_rv_value_pointer(pred, grounding):
    rv_name = pred.functor.name
    result = RandomVariableValuePointer(rv_name)
    for col, arg in zip(grounding.relation.value.columns, pred.args):
        if isinstance(arg, Constant):
            result = Selection(
                result, eq_(Constant[ColumnStr](ColumnStr(col)), arg)
            )
        else:
            result = RenameColumn(result, Constant(ColumnStr(col)), arg)
    return result


def and_vect_table_distribution(rule_grounding, parent_groundings):
    antecedent_preds = extract_logic_predicates(
        rule_grounding.expression.antecedent
    )
    to_join = tuple(
        get_rv_value_pointer(pred, parent_groundings[pred.functor])
        for pred in antecedent_preds
    )
    return MultiplyColumns(MultipleNaturalJoin(to_join))


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

    @add_match(PfactGrounding)
    def ground_probfact_grounding(self, grounding):
        rv_symb = grounding.expression.consequent.body.functor
        self._add_grounding(rv_symb, grounding)
        self._add_random_variable(
            rv_symb, multi_bernoulli_vect_table_distrib(grounding)
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
            for predicate in extract_logic_predicates(
                rule_grounding.expression.antecedent
            )
        }
        self._add_random_variable(
            rv_symb,
            and_vect_table_distribution(rule_grounding, parent_groundings),
        )
        parent_rv_symbs = {
            pred.functor
            for pred in extract_logic_predicates(
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


class SuccQuery(Definition):
    def __init__(self, predicate):
        self.predicate = predicate


class MargQuery(Definition):
    def __init__(self, predicate, evidence):
        self.predicate = predicate
        self.evidence = evidence


class QueryGraphicalModelSolver(PatternWalker):
    def __init__(self, graphical_model):
        self.graphical_model = graphical_model

    @add_match(SuccQuery)
    def succ_query(self, query):
        predicate = _get_predicate_from_grounded_expression(
            self.graphical_model.groundings.value[
                query.predicate.functor
            ].expression
        )
        rv_symb = predicate.functor
        if rv_symb not in self.graphical_model.edges.value:
            marginal = compute_marginal_probability(
                self.graphical_model.cpds.value[rv_symb], {}, {}
            )
        else:
            parent_symbs = self.graphical_model.edges.value[rv_symb]
            rule = self.graphical_model.groundings.value[rv_symb].expression
            parent_marginal_distribs = {
                pred.functor: self.walk(SuccQuery(pred))
                for pred in extract_logic_predicates(rule.antecedent)
            }
            parent_groundings = {
                parent_symb: self.graphical_model.groundings.value[parent_symb]
                for parent_symb in parent_symbs
            }
            marginal = compute_marginal_probability(
                self.graphical_model.cpds.value[rv_symb],
                parent_marginal_distribs,
                parent_groundings,
            )
        result = marginal
        for qpred_arg, marginal_arg in zip(
            query.predicate.args, predicate.args
        ):
            if isinstance(qpred_arg, Constant):
                result = Selection(
                    result,
                    eq_(
                        Constant[ColumnStr](
                            ColumnStr(
                                _get_column_name_from_expression(marginal_arg)
                            )
                        ),
                        qpred_arg,
                    ),
                )
            elif qpred_arg != marginal_arg:
                result = RenameColumn(result, marginal_arg, qpred_arg)

        return ExtendedRelationalAlgebraSolver({}).walk(result)

    @add_match(MargQuery)
    def marg_query(self, query):
        evidence_prob = self.walk(SuccQuery(query.evidence))


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
                        (solver.walk(cpd),) + tuple(parent_marg_probs.values())
                    )
                )
            )
        return ExtendedRelationalAlgebraSolver({}).walk(
            SumColumns(MultipleNaturalJoin(tuple(terms)))
        )


def is_arithmetic_operation(exp):
    return (
        isinstance(exp, FunctionApplication)
        and isinstance(exp.functor, Constant)
        and exp.functor.value
        in {operator.add, operator.sub, operator.mul, operator.truediv}
    )


class ExtendedRelationalAlgebraSolver(RelationalAlgebraSolver):
    def __init__(self, rv_values):
        self.rv_values = rv_values

    @add_match(Constant[AbstractSet])
    def relation(self, relation):
        return relation

    @add_match(RandomVariableValuePointer)
    def rv_value_pointer(self, pointer):
        if pointer not in self.rv_values:
            raise NeuroLangException(
                f"Unknown value for random variable {pointer}"
            )
        return self.walk(self.rv_values[pointer])

    @add_match(ConcatenateColumn)
    def concatenate_column(self, concat_op):
        relation = self.walk(concat_op.relation)
        new_column_name = _get_column_name_from_expression(concat_op.column)
        new_columns = list(relation.value.columns) + [new_column_name]
        iterable = relation.value._container.copy()
        iterable[new_column_name] = concat_op.column_values.value
        iterable = iterable[new_columns]
        relation = Constant[AbstractSet](AlgebraSet(new_columns, iterable))
        return relation

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

    @add_match(RenameColumns)
    def rename_columns(self, op):
        result = op.relation
        for old_col, new_col in zip(op.old_names, op.new_names):
            result = RenameColumn(result, old_col, new_col)
        return self.walk(result)

    @add_match(SumColumns)
    def sum_columns(self, sum_op):
        return _apply_arithmetic_column_op(self.walk(sum_op.relation), np.sum)

    @add_match(MultiplyColumns)
    def multiply_columns(self, multiply_op):
        return _apply_arithmetic_column_op(
            self.walk(multiply_op.relation), np.prod
        )

    @add_match(DivideColumns)
    def divide_columns(self, divide_op):
        return _divide_columns(
            self.walk(divide_op.relation),
            divide_op.numerator_column,
            divide_op.denominator_column,
        )

    @add_match(AddRepeatedValueColumn)
    def add_repeated_value_column(self, add_op):
        col_vals = Constant[np.ndarray](
            np.repeat(add_op.repeated_value.value, len(add_op.relation.value))
        )
        return self.walk(
            ConcatenateColumn(
                relation=add_op.relation,
                column=make_numerical_col_symb(),
                column_values=col_vals,
            )
        )

    @add_match(MultipleNaturalJoin)
    def multiple_natural_join(self, op):
        relations_it = iter(op.relations)
        result = next(relations_it)
        for relation in relations_it:
            result = NaturalJoin(result, relation)
        return self.walk(result)

    @add_match(NegateProbability)
    def negate_probability(self, neg_op):
        non_num_cols, num_cols = _split_numerical_cols(neg_op.relation)
        if len(num_cols) != 1:
            raise NeuroLangException("Expected only one numerical column")
        new_prob_col = make_numerical_col_symb().name
        new_cols = list(non_num_cols) + [new_prob_col]
        iterable = neg_op.relation.value._container.copy()
        iterable[new_prob_col] = 1.0 - iterable[num_cols[0]].values
        iterable.drop(columns=num_cols)
        iterable = iterable[new_cols]
        return Constant[AbstractSet](AlgebraSet(new_cols, iterable))

    @add_match(VectorisedTableDistribution)
    def vectorised_table_distribution(self, distrib):
        truth_prob = distrib.table.value[Constant[bool](True)]
        if isinstance(truth_prob, Constant[float]):
            return self.walk(
                AddRepeatedValueColumn(distrib.grounding.relation, truth_prob)
            )
        else:
            return self.walk(
                NaturalJoin(distrib.grounding.relation, self.walk(truth_prob))
            )

    @add_match(Aggregation)
    def aggregation(self, agg_op):
        relation = self.walk(agg_op.relation)
        if agg_op.agg_column is not None:
            relation = self.walk(
                Projection(
                    relation, agg_op.group_columns + (agg_op.agg_column,)
                )
            )
        group_columns = _cols_as_strings(agg_op.group_columns)
        new_container = relation.value._container.groupby(group_columns).agg(
            agg_op.agg_fun.value
        )
        new_container.rename(
            columns={
                new_container.columns[0]: _get_column_name_from_expression(
                    agg_op.dst_column
                )
            },
            inplace=True,
        )
        new_container.reset_index(inplace=True)
        columns_to_drop = (
            set(new_container.columns)
            - set(group_columns)
            - {agg_op.dst_column.value}
        )
        if columns_to_drop:
            new_container.drop(columns=list(columns_to_drop), inplace=True)
        new_relation = AlgebraSet(
            iterable=new_container, columns=list(new_container.columns)
        )
        return Constant[AbstractSet](new_relation)

    @add_match(ExtendedProjection)
    def extended_projection(self, proj_op):
        relation = self.walk(proj_op.relation)
        str_arithmetic_walker = StringArithmeticWalker()
        pandas_eval_expressions = []
        for member in proj_op.projection_list:
            pandas_eval_expressions.append(
                "{} = {}".format(
                    member.dst_column.value,
                    str_arithmetic_walker.walk(self.walk(member.fun_exp)),
                )
            )
        new_container = relation.value._container.eval(
            "/n".join(pandas_eval_expressions)
        )
        return Constant[AbstractSet](
            AlgebraSet(columns=new_container.columns, iterable=new_container)
        )

    @add_match(FunctionApplication, is_arithmetic_operation)
    def arithmetic_operation(self, arithmetic_op):
        return FunctionApplication[arithmetic_op.type](
            arithmetic_op.functor,
            tuple(self.walk(arg) for arg in arithmetic_op.args),
        )


class StringArithmeticWalker(PatternWalker):
    @add_match(Constant[ColumnStr])
    def column(self, column):
        return column.value

    @add_match(FunctionApplication(Constant(len), ...))
    def len(self, fa):
        if not isinstance(fa.args[0], Constant[AbstractSet]):
            raise NeuroLangException("Expected constant RA relation")
        return str(len(fa.args[0].value))

    @add_match(FunctionApplication, is_arithmetic_operation)
    def arithmetic_operation(self, fa):
        return "({} {} {})".format(
            self.walk(fa.args[0]),
            arithmetic_operator_string(fa.functor.value),
            self.walk(fa.args[1]),
        )


def arithmetic_operator_string(op):
    return {
        operator.add: "+",
        operator.sub: "-",
        operator.mul: "*",
        operator.truediv: "/",
    }[op]


def _split_numerical_cols(relation):
    non_num = []
    num = []
    for col in relation.value.columns:
        if col.startswith("__numerical__"):
            num.append(col)
        else:
            non_num.append(col)
    return non_num, num


def _cols_as_strings(columns):
    return [_get_column_name_from_expression(col) for col in columns]


def _apply_arithmetic_column_op(relation, numpy_op):
    non_num_cols, num_cols = _split_numerical_cols(relation)
    new_col = make_numerical_col_symb().name
    new_cols = list(non_num_cols) + [new_col]
    new_col_vals = numpy_op(relation.value._container[num_cols], axis=1)
    if non_num_cols:
        iterable = relation.value._container[non_num_cols].copy()
        iterable[new_col] = new_col_vals
    else:
        iterable = pd.DataFrame({new_col: new_col_vals})
    iterable = iterable[new_cols]
    return Constant[AbstractSet](AlgebraSet(new_cols, iterable))


def _divide_columns(relation, numerator_col, denominator_col):
    new_col = make_numerical_col_symb()
    new_col_vals = (
        relation.value._container[numerator_col.name].values
        / relation.value._container[denominator_col.name].values
    )
    non_num_cols = _split_numerical_cols(relation)[0]
    new_cols = non_num_cols + [new_col.name]
    iterable = relation.value._container[non_num_cols].copy()
    iterable[new_col.name] = new_col_vals
    iterable = iterable[new_cols]
    return Constant[AbstractSet](AlgebraSet(new_cols, iterable))


def _get_column_name_from_expression(column_exp):
    if isinstance(column_exp, Constant):
        return column_exp.value
    elif isinstance(column_exp, Symbol):
        return column_exp.name
    else:
        raise NeuroLangException(
            "Cannot obtain column name from expression of type {}".format(
                type(column_exp)
            )
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


def _get_valued_rv(rv_symb, rv_grounding, bool_value):
    return AddRepeatedValueColumn(
        rv_grounding.relation, Constant[int](int(bool_value))
    )


def _iter_parents(parent_marg_probs, parent_groundings):
    parent_symbs = sorted(parent_marg_probs, key=lambda symb: symb.name)
    for parent_bool_values in itertools.product(
        *[(True, False) for _ in parent_symbs]
    ):
        parent_values = {
            symb: _get_valued_rv(symb, parent_groundings[symb], bool_value)
            for symb, bool_value in zip(parent_symbs, parent_bool_values)
        }
        parent_margin_probs = {
            parent_symb: parent_marg_probs[parent_symb]
            if parent_bool_value
            else NegateProbability(parent_marg_probs[parent_symb])
            for parent_symb, parent_bool_value in zip(
                parent_symbs, parent_bool_values
            )
        }
        yield parent_values, parent_margin_probs


def infer_pfact_params(pfact_grounding, interpretations_ra_set):
    """
    Compute the estimate of the parameters associated with a specific
    probabilistic fact predicate symbol from the facts with that same predicate
    symbol found in interpretations.

    """
    if "__interpretation_id__" not in interpretations_ra_set.value.columns:
        raise NeuroLangException("Column __interpretation_id__ is missing")
    tuple_counts = Aggregation(
        agg_fun=Constant[str]("count"),
        relation=NaturalJoin(
            pfact_grounding.params_relation, interpretations_ra_set
        ),
        group_columns=tuple(
            Constant(ColumnStr(arg.name))
            for arg in pfact_grounding.expression.consequent.body.args
        )
        + (
            Constant(
                ColumnStr(
                    pfact_grounding.expression.consequent.probability.name
                )
            ),
        ),
        agg_column=Constant(ColumnStr("__interpretation_id__")),
        dst_column=Constant(ColumnStr("__tuple_counts__")),
    )
    substitution_counts = Aggregation(
        agg_fun=Constant[str]("count"),
        relation=pfact_grounding.params_relation,
        group_columns=(
            Constant(
                ColumnStr(
                    pfact_grounding.expression.consequent.probability.name
                )
            ),
        ),
        agg_column=None,
        dst_column=Constant(ColumnStr("__substitution_counts__")),
    )
    n_interpretations = Constant(len)(
        Aggregation(
            agg_fun=Constant[str]("count"),
            relation=interpretations_ra_set,
            group_columns=(Constant(ColumnStr("__interpretation_id__")),),
            agg_column=None,
            dst_column=Constant(ColumnStr("__n_tuples_per_interpretation__")),
        )
    )
    probabilities = ExtendedProjection(
        NaturalJoin(tuple_counts, substitution_counts),
        tuple(
            [
                ExtendedProjectionListMember(
                    fun_exp=Constant(ColumnStr("__tuple_counts__"))
                    / (
                        Constant(ColumnStr("__substitution_counts__"))
                        * n_interpretations
                    ),
                    dst_column=Constant(ColumnStr("__probability__")),
                )
            ]
        ),
    )
    parameter_estimations = Aggregation(
        agg_fun=Constant[str]("mean"),
        relation=probabilities,
        group_columns=(
            Constant(
                ColumnStr(
                    pfact_grounding.expression.consequent.probability.name
                )
            ),
        ),
        agg_column=Constant(ColumnStr("__probability__")),
        dst_column=Constant(ColumnStr("__parameter_estimate__")),
    )
    solver = ExtendedRelationalAlgebraSolver({})
    return solver.walk(parameter_estimations)
