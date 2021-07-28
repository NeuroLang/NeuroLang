import math
import operator
from typing import AbstractSet

from .exceptions import (
    RelationalAlgebraError,
    RelationalAlgebraNotImplementedError
)
from .expression_walker import ExpressionWalker, PatternWalker, add_match
from .expressions import (
    Constant,
    FunctionApplication,
    Symbol,
    sure_is_not_pattern
)
from .relational_algebra import (
    Column,
    ColumnInt,
    ColumnStr,
    ConcatenateConstantColumn,
    Difference,
    EquiJoin,
    ExtendedProjection,
    FunctionApplicationListMember,
    GroupByAggregation,
    LeftNaturalJoin,
    NameColumns,
    NAryRelationalAlgebraOperation,
    NaturalJoin,
    Product,
    Projection,
    RelationalAlgebraOperation,
    RelationalAlgebraSolver,
    RenameColumn,
    RenameColumns,
    Selection,
    UnaryRelationalAlgebraOperation,
    Union,
    eq_,
    str2columnstr_constant
)

ADD = Constant(operator.add)
MUL = Constant(operator.mul)
SUB = Constant(operator.sub)


class ProvenanceAlgebraSet(Constant):
    def __init__(self, relations, provenance_column):
        self.relations = relations
        self.provenance_column = provenance_column
        if not isinstance(provenance_column, Column):
            raise ValueError("Provenance column needs to be of Column type")

    @property
    def value(self):
        return self.relations

    @property
    def non_provenance_columns(self):
        return tuple(
            column
            for column in self.value.columns
            if column != self.provenance_column
        )


def check_do_not_share_non_prov_col(prov_set_1, prov_set_2):
    shared_columns = set(prov_set_1.non_provenance_columns) & set(
        prov_set_2.non_provenance_columns
    )
    if len(shared_columns) > 0:
        raise RelationalAlgebraError(
            "Provenance sets should not share non-provenance columns. "
            "Shared columns found: {}".format(
                ", ".join(repr(col) for col in shared_columns)
            )
        )


def is_provenance_operation(operation):
    stack = list(operation.unapply())
    while stack:
        stack_element = stack.pop()
        if isinstance(
            stack_element,
            (ProvenanceAlgebraSet, BuildProvenanceAlgebraSet)
        ):
            return True
        if isinstance(stack_element, tuple):
            stack += list(stack_element)
        elif isinstance(stack_element, RelationalAlgebraOperation):
            stack += list(stack_element.unapply())
    return False


class BuildProvenanceAlgebraSet(UnaryRelationalAlgebraOperation):
    def __init__(self, relation, provenance_column):
        self.relation = relation
        self.provenance_column = provenance_column

    @property
    def non_provenance_columns(self):
        return self.relation.columns() - {self.provenance_column}

    def __repr__(self):
        return (f"RAP[{self.relation}, {self.provenance_column}")


class NaturalJoinInverse(NaturalJoin):
    def __repr__(self):
        return (
            f"[{self.relation_left}"
            f"INVERSE \N{JOIN}"
            f"{self.relation_right}]"
        )


class WeightedNaturalJoin(NAryRelationalAlgebraOperation):
    def __init__(self, relations, weights):
        self.relations = relations
        self.weights = weights

    def __repr__(self):
        return (
            "\N{Greek Capital Letter Sigma}"
            f"_{self.weights}({self.relations})"
        )


class BuildProvenanceAlgebraSetMixin(PatternWalker):
    """
    Mixin to build Provnance Algebra Sets from 
    the `BuildProvenanceAlgebraSet` operation.
    """

    @add_match(BuildProvenanceAlgebraSet(Constant, Constant))
    def build_provenance_algebra_set(self, expression):
        res = ProvenanceAlgebraSet(
            expression.relation.value,
            expression.provenance_column.value
        )
        return self.walk(res)

    @add_match(BuildProvenanceAlgebraSet)
    def cycle_in_build_provenance_algebra_set(self, expression):
        relation = self.walk(expression.relation)
        provenance_column = self.walk(expression.provenance_column)
        if (
            (relation is not expression.relation) or
            (provenance_column is not expression.provenance_column)
        ):
            return self.walk(
                BuildProvenanceAlgebraSet(relation, provenance_column)
            )
        else:
            return expression


class ProvenanceExtendedProjectionMixin(PatternWalker):
    """
    Mixin that implements specific cases of extended projections on provenance
    sets for which the semantics are not modified.

    An extended projection on a provenance set is allowed if all the
    non-provenance columns `c` are projected _as is_ (i.e. `c -> c`), which
    ensures the number of tuples is going to be exactly the same in the
    resulting provenance set, and the provenance label is still going to be
    semantically valid. This is useful in particular for projecting a constant
    column, which can happen when dealing with rules with constant terms or
    variable equalities.

    """
    @add_match(ExtendedProjection(BuildProvenanceAlgebraSet, ...))
    def prov_extended_projection(self, extended_proj):
        relation = extended_proj.relation
        if any(
            proj_list_member.dst_column == relation.provenance_column
            for proj_list_member in extended_proj.projection_list
        ):
            new_prov_col = str2columnstr_constant(Symbol.fresh().name)
            relation = RenameColumn(
                relation,
                relation.provenance_column,
                new_prov_col
            )
        else:
            new_prov_col = relation.provenance_column

        new_proj_list = extended_proj.projection_list + (
            FunctionApplicationListMember(
                fun_exp=new_prov_col, dst_column=new_prov_col
            ),
        )
        return self.walk(BuildProvenanceAlgebraSet(
                ExtendedProjection(
                    relation.relation, new_proj_list,
                ),
                new_prov_col
        ))


class RelationalAlgebraProvenanceCountingSolver(
    ProvenanceExtendedProjectionMixin,
    BuildProvenanceAlgebraSetMixin,
    RelationalAlgebraSolver,
):
    """
    Mixing that walks through relational algebra expressions and
    executes the operations and provenance calculations.

    """

    def __init__(self, symbol_table=None):
        self.symbol_table = symbol_table

    @add_match(
        Selection(BuildProvenanceAlgebraSet, ...)
    )
    def selection_between_column_int(self, selection):
        res = BuildProvenanceAlgebraSet(
            Selection(
                selection.relation.relation,
                selection.formula
            ),
            selection.relation.provenance_column
        )
        return self.walk(res)

    @add_match(
        Product,
        lambda product: all(
            isinstance(ras, BuildProvenanceAlgebraSet)
            for ras in product.relations
        )
    )
    def prov_product(self, product):
        rel_res = product.relations[0]
        for relation in product.relations[1:]:
            check_do_not_share_non_prov_col(rel_res, relation)
            rel_res = self._apply_provenance_join_operation(
                rel_res, relation, Product, MUL,
            )
        return self.walk(rel_res)

    @add_match(ConcatenateConstantColumn(BuildProvenanceAlgebraSet, ..., ...))
    def prov_concatenate_constant_column(self, concat_op):
        if concat_op.column_name == concat_op.relation.provenance_column:
            new_prov_col = str2columnstr_constant(Symbol.fresh().name)
            relation = RenameColumn(
                concat_op.relation.relation,
                concat_op.relation.provenance_column,
                new_prov_col,
            )
        else:
            relation = concat_op.relation.relation
            new_prov_col = concat_op.relation.provenance_column

        res = BuildProvenanceAlgebraSet(
            ConcatenateConstantColumn(
                relation,
                concat_op.column_name,
                concat_op.column_value
            ),
            new_prov_col
        )
        return self.walk(res)

    @add_match(Projection(BuildProvenanceAlgebraSet, ...))
    def prov_projection(self, projection):
        prov_set = projection.relation
        prov_col = prov_set.provenance_column
        group_columns = projection.attributes

        # aggregate the provenance column grouped by the projection columns
        aggregate_functions = [
            FunctionApplicationListMember(
                FunctionApplication(
                    Constant(sum),
                    (prov_col,),
                    validate_arguments=False,
                    verify_type=False,
                ),
                prov_col,
            )
        ]
        operation = GroupByAggregation(
            prov_set.relation,
            group_columns,
            aggregate_functions,
        )
        return self.walk(
            BuildProvenanceAlgebraSet(operation, prov_col)
        )

    @add_match(EquiJoin(ProvenanceAlgebraSet, ..., ProvenanceAlgebraSet, ...))
    def prov_equijoin(self, equijoin):
        raise NotImplementedError("EquiJoin is not implemented.")

    @add_match(RenameColumn(BuildProvenanceAlgebraSet, ..., ...))
    def prov_rename_column(self, rename_column):
        if rename_column.src == rename_column.relation.provenance_column:
            provenance_column = rename_column.dst
        else:
            provenance_column = rename_column.relation.provenance_column
        res = BuildProvenanceAlgebraSet(
            RenameColumn(
                rename_column.relation.relation,
                rename_column.src,
                rename_column.dst
            ),
            provenance_column
        )
        return self.walk(res)

    @add_match(RenameColumns(BuildProvenanceAlgebraSet, ...))
    def prov_rename_columns(self, rename_columns):
        res = BuildProvenanceAlgebraSet(
            RenameColumns(
                rename_columns.relation.relation,
                rename_columns.renames
            ),
            rename_columns.relation.provenance_column
        )
        return self.walk(res)

    @add_match(Difference(ProvenanceAlgebraSet, ProvenanceAlgebraSet))
    def prov_difference(self, diff):
        left = self.walk(diff.relation_left)
        right = self.walk(diff.relation_right)

        res_columns = set(left.value.columns)
        res_columns.symmetric_difference(set(right.value.columns))

        res_columns = tuple(str2columnstr_constant(col) for col in res_columns)
        res_prov_col = str2columnstr_constant(left.provenance_column)
        tmp_left_prov_col = Constant[ColumnStr](
            ColumnStr(f"{left.provenance_column}1"),
            verify_type=False,
            auto_infer_type=False,
        )
        tmp_right_prov_col = Constant[ColumnStr](
            ColumnStr(f"{right.provenance_column}2"),
            verify_type=False,
            auto_infer_type=False,
        )
        tmp_left = RenameColumn(
            Constant[AbstractSet](left.value),
            str2columnstr_constant(left.provenance_column),
            tmp_left_prov_col,
        )
        tmp_right = RenameColumn(
            Constant[AbstractSet](right.value),
            str2columnstr_constant(right.provenance_column),
            tmp_right_prov_col,
        )
        tmp_np_op_args = (tmp_left, tmp_right)
        tmp_non_prov_result = LeftNaturalJoin(*tmp_np_op_args)

        isnan = Constant(lambda x: 0 if math.isnan(x) else x)

        result = ExtendedProjection(
            tmp_non_prov_result,
            (
                FunctionApplicationListMember(
                    fun_exp=MUL(
                        tmp_left_prov_col,
                        SUB(
                            Constant(1),
                            isnan(tmp_right_prov_col)
                        )
                    ),
                    dst_column=res_prov_col,
                ),
            )
            + tuple(
                FunctionApplicationListMember(fun_exp=col, dst_column=col)
                for col in set(res_columns) - {res_prov_col}
            ),
        )
        return ProvenanceAlgebraSet(
            self.walk(result).value, res_prov_col.value
        )

    @add_match(NaturalJoinInverse)
    def prov_naturaljoin_inverse(self, naturaljoin):
        return self.walk(self._apply_provenance_join_operation(
            naturaljoin.relation_left,
            naturaljoin.relation_right,
            NaturalJoin,
            Constant(operator.truediv),
        ))

    @add_match(NaturalJoin(
        BuildProvenanceAlgebraSet, BuildProvenanceAlgebraSet
    ))
    def prov_naturaljoin(self, naturaljoin):
        return self.walk(self._apply_provenance_join_operation(
            naturaljoin.relation_left,
            naturaljoin.relation_right,
            NaturalJoin,
            MUL,
        ))

    def _apply_provenance_join_operation(
        self, left, right, np_op, prov_binary_op
    ):
        res_columns = set(left.columns()) | (
            set(right.columns()) - {right.provenance_column}
        )
        res_columns = tuple(col for col in res_columns)
        res_prov_col = left.provenance_column
        # provenance columns are temporarily renamed for executing the
        # non-provenance operation on the relations
        tmp_left_col = str2columnstr_constant(f"{left.provenance_column.value}1")
        tmp_right_col = str2columnstr_constant(f"{right.provenance_column.value}2")
        tmp_left = RenameColumn(
            left.relation,
            left.provenance_column,
            tmp_left_col,
        )
        tmp_right = RenameColumn(
            right.relation,
            right.provenance_column,
            tmp_right_col,
        )
        tmp_np_op_args = (tmp_left, tmp_right)
        if np_op is Product:
            tmp_non_prov_result = np_op(tmp_np_op_args)
        elif np_op is NaturalJoin:
            tmp_non_prov_result = np_op(*tmp_np_op_args)
        else:
            raise RelationalAlgebraError(
                "Cannot apply non-provenance operation: {}".format(np_op)
            )
        result = ExtendedProjection(
            tmp_non_prov_result,
            (
                FunctionApplicationListMember(
                    fun_exp=prov_binary_op(tmp_left_col, tmp_right_col),
                    dst_column=res_prov_col,
                ),
            )
            + tuple(
                FunctionApplicationListMember(fun_exp=col, dst_column=col)
                for col in set(res_columns) - {res_prov_col}
            ),
        )
        return BuildProvenanceAlgebraSet(
            result, res_prov_col
        )

    @add_match(WeightedNaturalJoin)
    def prov_weighted_join(self, join_op):
        relations = self.walk(join_op.relations)
        weights = self.walk(join_op.weights)

        prov_columns = [
            str2columnstr_constant(Symbol.fresh().name)
            for _ in relations
        ]

        dst_columns = set(sum(
            (
                relation.non_provenance_columns
                for relation in relations
            ),
            tuple()
        ))

        relations = [
            ExtendedProjection(
                Constant[AbstractSet](relation.relations),
                (FunctionApplicationListMember(
                    weight * str2columnstr_constant(
                        relation.provenance_column
                    ),
                    prov_column
                ),) + tuple(
                    FunctionApplicationListMember(
                        str2columnstr_constant(c), str2columnstr_constant(c)
                    )
                    for c in relation.non_provenance_columns
                )
            )
            for relation, weight, prov_column in
            zip(relations, weights, prov_columns)
        ]

        relation = relations[0]
        for relation_ in relations[1:]:
            relation = NaturalJoin(relation, relation_)

        prov_col = str2columnstr_constant(Symbol.fresh().name)
        dst_prov_expr = sum(prov_columns[1:], prov_columns[0])
        relation = ExtendedProjection(
            relation,
            (FunctionApplicationListMember(
                dst_prov_expr,
                prov_col
            ),) + tuple(
                FunctionApplicationListMember(
                    str2columnstr_constant(c), str2columnstr_constant(c)
                )
                for c in dst_columns
            )
        )

        return ProvenanceAlgebraSet(
            self.walk(relation).value,
            prov_col.value
        )

    @add_match(Union(BuildProvenanceAlgebraSet, BuildProvenanceAlgebraSet))
    def prov_union(self, union_op):
        left = union_op.relation_left
        right = union_op.relation_right
        if left.non_provenance_columns != right.non_provenance_columns:
            raise RelationalAlgebraError(
                "All non-provenance columns must be the same: {} != {}".format(
                    left.non_provenance_columns, right.non_provenance_columns,
                )
            )
        prov_col = left.provenance_column
        result_columns = left.non_provenance_columns
        # move to non-provenance relational algebra
        np_left = left.relation
        np_right = right.relation
        # make the the provenance columns match
        np_right = RenameColumn(
            np_right,
            right.provenance_column,
            prov_col,
        )
        # add a dummy column with different values for each relation
        # this ensures that all the tuples will be part of the result
        dummy_col = str2columnstr_constant(Symbol.fresh().name)
        np_left = ConcatenateConstantColumn(
            np_left, dummy_col, Constant[int](0)
        )
        np_right = ConcatenateConstantColumn(
            np_right, dummy_col, Constant[int](1)
        )
        ra_union_op = Union(np_left, np_right)
        result = BuildProvenanceAlgebraSet(ra_union_op, prov_col)
        # provenance projection that removes the dummy column and sums the
        # provenance of matching tuples
        result = Projection(result, result_columns)
        return self.walk(result)

    @add_match(Constant[AbstractSet])
    def constant_relation_or_provenance_set(self, relation):
        return relation


class RelationalAlgebraProvenanceExpressionSemringSolver(
    RelationalAlgebraSolver
):
    @add_match(NaturalJoin(BuildProvenanceAlgebraSet, BuildProvenanceAlgebraSet))
    def natural_join_rap(self, expression):
        rap_left = expression.relation_left
        rap_right = expression.relation_right
        rap_right_r = rap_right.relations
        rap_right_pc = str2columnstr_constant(rap_right.provenance_column)

        cols_to_keep = [
            FunctionApplicationListMember(
                str2columnstr_constant(c), str2columnstr_constant(c)
            )
            for c in (rap_left.columns + rap_right.columns)
            if c not in (
                rap_left.provenance_column,
                rap_right.provenance_column,
            )
        ]
        if rap_left.provenance_column == rap_right.provenance_column:
            rap_right_pc = str2columnstr_constant(Symbol.fresh().name)
            rap_right_r = RenameColumn(
                rap_right_r,
                rap_right.provenance_column,
                rap_right_pc
            )
        new_pc = str2columnstr_constant(Symbol.fresh().name)
        operation = ExtendedProjection(
            NaturalJoin(
                self._build_relation_constant(rap_left.relations),
                rap_right_r
            ),
            cols_to_keep + [
                FunctionApplicationListMember(
                    self._semiring_mul(
                        str2columnstr_constant(rap_left.provenance_column),
                        rap_right_pc
                    ),
                    new_pc
                )
            ]
        )

        with sure_is_not_pattern():
            res = self.walk(
                BuildProvenanceAlgebraSet(
                    operation,
                    new_pc
                )
            )
        return res

    def _semiring_mul(self, left, right):
        return left * right

    @add_match(
        Projection(ProvenanceAlgebraSet, ...),
        lambda proj: any(
            issubclass(att.type, ColumnInt) for att in proj.attributes
        )
    )
    def projection_rap_columnint(self, projection):
        columns = projection.relation.non_provenance_columns
        new_attributes = tuple()
        for att in projection.attributes:
            if issubclass(att.type, ColumnInt):
                att = str2columnstr_constant(columns[att.value])
            new_attributes += (att,)
        return self.walk(Projection(projection.relation, new_attributes))

    @add_match(
        Selection(
            BuildProvenanceAlgebraSet,
            FunctionApplication(
                eq_, (Constant[ColumnInt], Constant[ColumnInt])
            )
        )
    )
    def selection_rap_eq_columnint_columnint(self, selection):
        columns = selection.relation.non_provenance_columns
        formula = selection.formula
        new_formula = FunctionApplication(
            eq_, (
                str2columnstr_constant(columns[formula.args[0].value]),
                str2columnstr_constant(columns[formula.args[1].value]),
            )
        )
        return self.walk(Selection(selection.relation, new_formula))

    @add_match(
        Selection(
            ProvenanceAlgebraSet,
            FunctionApplication(eq_, (Constant[ColumnInt], ...))
        )
    )
    def selection_rap_eq_columnint(self, selection):
        columns = selection.relation.non_provenance_columns
        formula = selection.formula
        new_formula = FunctionApplication(
            eq_, (
                str2columnstr_constant(columns[formula.args[0].value]),
                formula.args[1]
            )
        )
        return self.walk(Selection(selection.relation, new_formula))

    @add_match(Projection(ProvenanceAlgebraSet, ...))
    def projection_rap(self, projection):
        cols = tuple(v.value for v in projection.attributes)
        if projection.relation.relations.is_dum() or (
            cols
            == tuple(
                c
                for c in projection.relation.relations.columns
                if c != projection.relation.provenance_column
            )
        ):
            return projection.relation

        aggregate_functions = [
            FunctionApplicationListMember(
                self._semiring_agg_sum(
                    (str2columnstr_constant(
                        projection.relation.provenance_column
                    ),)
                ),
                str2columnstr_constant(projection.relation.provenance_column),
            )
        ]
        operation = GroupByAggregation(
            self._build_relation_constant(projection.relation.relations),
            projection.attributes,
            aggregate_functions,
        )

        with sure_is_not_pattern():
            res = ProvenanceAlgebraSet(
                self.walk(operation).value,
                projection.relation.provenance_column,
            )
        return res

    def _semiring_agg_sum(self, args):
        return FunctionApplication(
            Constant(sum), args, validate_arguments=False, verify_type=False
        )

    @add_match(RenameColumn(ProvenanceAlgebraSet, ..., ...))
    def rename_column_rap(self, expression):
        ne = RenameColumn(
            self._build_relation_constant(expression.relation.relations),
            expression.src, expression.dst
        )
        return ProvenanceAlgebraSet(
            self.walk(ne).value,
            expression.relation.provenance_column
        )

    @add_match(RenameColumns(ProvenanceAlgebraSet, ...))
    def rename_columns_rap(self, expression):
        ne = RenameColumns(
            self._build_relation_constant(expression.relation.relations),
            expression.renames
        )
        return ProvenanceAlgebraSet(
            self.walk(ne).value,
            expression.relation.provenance_column
        )

    @add_match(NameColumns(ProvenanceAlgebraSet, ...))
    def name_columns_rap(self, expression):
        relation = self._build_relation_constant(expression.relation.relations)
        columns_to_name = tuple(
            c for c in relation.value.columns
            if c != expression.relation.provenance_column
        )
        ne = RenameColumns(
            relation,
            tuple(
                (Constant(src), dst)
                for src, dst in zip(
                    columns_to_name, expression.column_names
                )
            )
        )
        return ProvenanceAlgebraSet(
            self.walk(ne).value,
            expression.relation.provenance_column
        )

    @add_match(Selection(ProvenanceAlgebraSet, ...))
    def selection_rap(self, selection):
        ne = Selection(
            self._build_relation_constant(selection.relation.relations),
            selection.formula
        )
        return ProvenanceAlgebraSet(
            self.walk(ne).value,
            selection.relation.provenance_column
        )

    @add_match(Union(ProvenanceAlgebraSet, ProvenanceAlgebraSet))
    def union_rap(self, union):
        prov_column_left = union.relation_left.provenance_column
        prov_column_right = union.relation_right.provenance_column
        relation_left = self._build_relation_constant(
            union.relation_left.relations
        )
        relation_right = self._build_relation_constant(
            union.relation_right.relations
        )

        if prov_column_left != prov_column_right:
            relation_right = RenameColumn(
                relation_right,
                str2columnstr_constant(prov_column_right),
                str2columnstr_constant(prov_column_left),
            )

        columns_to_keep = tuple(
            str2columnstr_constant(c) for c in
            union.relation_left.non_provenance_columns
        )

        dummy_col = str2columnstr_constant(Symbol.fresh().name)
        relation_left = ConcatenateConstantColumn(
            relation_left, dummy_col, Constant[int](0)
        )
        relation_right = ConcatenateConstantColumn(
            relation_right, dummy_col, Constant[int](1)
        )

        with sure_is_not_pattern():
            ra_union = self.walk(Union(relation_left, relation_right))
        rap_projection = Projection(
            ProvenanceAlgebraSet(
                ra_union.value,
                prov_column_left
            ),
            columns_to_keep
        )

        with sure_is_not_pattern():
            res = self.walk(rap_projection)

        return res

    @add_match(Difference(ProvenanceAlgebraSet, ProvenanceAlgebraSet))
    def difference(self, diff):
        left = self.walk(diff.relation_left)
        right = self.walk(diff.relation_right)
        res_columns = tuple(
            str2columnstr_constant(col) for col in left.value.columns
        )
        res_prov_col = str2columnstr_constant(left.provenance_column)
        tmp_left_prov_col = str2columnstr_constant(Symbol.fresh().name)
        tmp_right_prov_col = str2columnstr_constant(Symbol.fresh().name)
        tmp_left = RenameColumn(
            Constant[AbstractSet](left.value),
            str2columnstr_constant(left.provenance_column),
            tmp_left_prov_col,
        )
        tmp_right = RenameColumn(
            Constant[AbstractSet](right.value),
            str2columnstr_constant(right.provenance_column),
            tmp_right_prov_col,
        )
        tmp_np_op_args = (tmp_left, tmp_right)
        tmp_non_prov_result = LeftNaturalJoin(*tmp_np_op_args)
        isnan = Constant(lambda x: 0 if math.isnan(x) else x)
        result = ExtendedProjection(
            tmp_non_prov_result,
            (
                FunctionApplicationListMember(
                    fun_exp=MUL(
                        tmp_left_prov_col,
                        SUB(Constant(1), isnan(tmp_right_prov_col)),
                    ),
                    dst_column=res_prov_col,
                ),
            )
            + tuple(
                FunctionApplicationListMember(fun_exp=col, dst_column=col)
                for col in set(res_columns) - {res_prov_col}
            ),
        )
        return ProvenanceAlgebraSet(
            self.walk(result).value, res_prov_col.value
        )

    # Raise Exception for non-implemented RAP operations
    @add_match(
        RelationalAlgebraOperation,
        lambda x: any(
            isinstance(arg, ProvenanceAlgebraSet)
            for arg in x.unapply()
        )
    )
    def rap_not_implemented(self, ra_operation):
        raise RelationalAlgebraNotImplementedError(
            f"Relational Algebra with Provenance "
            f"operation {type(ra_operation)} not implemented"
        )
