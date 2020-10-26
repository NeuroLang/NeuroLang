import operator
from typing import AbstractSet

from .exceptions import (
    RelationalAlgebraError,
    RelationalAlgebraNotImplementedError
)
from .expression_walker import ExpressionWalker, add_match
from .expressions import (
    Constant,
    Expression,
    FunctionApplication,
    Symbol,
    sure_is_not_pattern
)
from .relational_algebra import (
    Column,
    ColumnInt,
    ColumnStr,
    ConcatenateConstantColumn,
    EquiJoin,
    ExtendedProjection,
    ExtendedProjectionListMember,
    NameColumns,
    NaturalJoin,
    Product,
    Projection,
    RelationalAlgebraOperation,
    RelationalAlgebraSolver,
    RenameColumn,
    RenameColumns,
    Selection,
    Union,
    eq_,
    str2columnstr_constant
)

ADD = Constant(operator.add)
MUL = Constant(operator.mul)


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
        if isinstance(stack_element, ProvenanceAlgebraSet):
            return True
        if isinstance(stack_element, tuple):
            stack += list(stack_element)
        elif isinstance(stack_element, RelationalAlgebraOperation):
            stack += list(stack_element.unapply())
    return False


class NaturalJoinInverse(NaturalJoin):
    def __repr__(self):
        return (
            f"[{self.relation_left}"
            f"INVERSE \N{JOIN}"
            f"{self.relation_right}]"
        )


class RelationalAlgebraProvenanceCountingSolver(ExpressionWalker):
    """
    Mixing that walks through relational algebra expressions and
    executes the operations and provenance calculations.

    """

    def __init__(self, symbol_table=None):
        self.symbol_table = symbol_table

    @add_match(
        RelationalAlgebraOperation,
        lambda exp: not is_provenance_operation(exp),
    )
    def non_provenance_operation(self, operation):
        return RelationalAlgebraSolver(self.symbol_table).walk(operation)

    @add_match(
        Selection(
            ...,
            FunctionApplication(
                eq_, (Constant[ColumnInt], Constant[ColumnInt])
            ),
        )
    )
    def selection_between_column_int(self, selection):
        relation = self.walk(selection.relation)
        columns = relation.non_provenance_columns
        lhs_col = str2columnstr_constant(
            columns[selection.formula.args[0].value]
        )
        rhs_col = str2columnstr_constant(
            columns[selection.formula.args[1].value]
        )
        new_formula = Constant(operator.eq)(lhs_col, rhs_col)
        new_selection = Selection(relation, new_formula)
        return self.walk(new_selection)

    @add_match(
        Selection(
            ...,
            FunctionApplication(eq_, (Constant[Column], Constant[Column])),
        )
    )
    def selection_between_columns(self, selection):
        relation = self.walk(selection.relation)
        if any(
            col == relation.provenance_column for col in selection.formula.args
        ):
            raise RelationalAlgebraError("Cannot select on provenance column")
        return ProvenanceAlgebraSet(
            self.walk(
                Selection(
                    Constant[AbstractSet](relation.value), selection.formula
                )
            ).value,
            relation.provenance_column,
        )

    @add_match(
        Selection(..., FunctionApplication(eq_, (Constant[Column], Constant)),)
    )
    def selection_by_constant(self, selection):
        relation = self.walk(selection.relation)
        if selection.formula.args[0] == relation.provenance_column:
            raise RelationalAlgebraError("Cannot select on provenance column")
        return ProvenanceAlgebraSet(
            self.walk(
                Selection(
                    Constant[AbstractSet](relation.value), selection.formula
                )
            ).value,
            relation.provenance_column,
        )

    @add_match(Product)
    def prov_product(self, product):
        rel_res = self.walk(product.relations[0])
        for relation in product.relations[1:]:
            rel_temp = self.walk(relation)
            check_do_not_share_non_prov_col(rel_res, rel_temp)
            rel_res = self._apply_provenance_join_operation(
                rel_res, rel_temp, Product, MUL,
            )
        return rel_res

    @add_match(ConcatenateConstantColumn)
    def prov_concatenate_constant_column(self, concat_op):
        relation = self.walk(concat_op.relation)
        if concat_op.column_name == relation.provenance_column:
            new_prov_col = Constant[ColumnStr](
                ColumnStr(Symbol.fresh().name),
                auto_infer_type=False,
                verify_type=False,
            )
        else:
            new_prov_col = relation.provenance_column
        relation = RenameColumn(
            Constant[AbstractSet](relation.value),
            str2columnstr_constant(relation.provenance_column),
            str2columnstr_constant(new_prov_col),
        )
        relation = ConcatenateConstantColumn(
            relation, concat_op.column_name, concat_op.column_value,
        )
        return ProvenanceAlgebraSet(self.walk(relation).value, new_prov_col)

    @add_match(Projection)
    def prov_projection(self, projection):
        prov_set = self.walk(projection.relation)
        prov_col = prov_set.provenance_column
        relation = prov_set.value
        # aggregate the provenance column grouped by the projection columns
        group_columns = [col.value for col in projection.attributes]
        agg_relation = relation.aggregate(group_columns, {prov_col: sum})
        # project the provenance column and the desired projection columns
        proj_columns = [prov_col] + group_columns
        projected_relation = agg_relation.projection(*proj_columns)
        return ProvenanceAlgebraSet(
            projected_relation, prov_set.provenance_column
        )

    @add_match(EquiJoin(ProvenanceAlgebraSet, ..., ProvenanceAlgebraSet, ...))
    def prov_equijoin(self, equijoin):
        raise NotImplementedError("EquiJoin is not implemented.")

    @add_match(RenameColumn)
    def prov_rename_column(self, rename_column):
        prov_relation = self.walk(rename_column.relation)
        new_prov_col = prov_relation.provenance_column
        if rename_column.src.value == prov_relation.provenance_column:
            new_prov_col = rename_column.dst.value
        return ProvenanceAlgebraSet(
            self.walk(
                RenameColumn(
                    Constant[AbstractSet](prov_relation.value),
                    rename_column.src,
                    rename_column.dst,
                )
            ).value,
            new_prov_col,
        )

    @add_match(RenameColumns)
    def prov_rename_columns(self, rename_columns):
        prov_relation = self.walk(rename_columns.relation)
        new_prov_col = prov_relation.provenance_column
        prov_col_rename = dict(rename_columns.renames).get(
            str2columnstr_constant(prov_relation.provenance_column), None
        )
        if prov_col_rename is not None:
            new_prov_col = prov_col_rename
        return ProvenanceAlgebraSet(
            self.walk(
                RenameColumns(
                    Constant[AbstractSet](prov_relation.value),
                    rename_columns.renames,
                )
            ).value,
            new_prov_col,
        )

    @add_match(ExtendedProjection)
    def prov_extended_projection(self, extended_proj):
        relation = self.walk(extended_proj.relation)
        if any(
            proj_list_member.dst_column == relation.provenance_column
            for proj_list_member in extended_proj.projection_list
        ):
            new_prov_col = str2columnstr_constant(Symbol.fresh().name)
        else:
            new_prov_col = str2columnstr_constant(relation.provenance_column)
        relation = self.walk(
            RenameColumn(
                relation,
                str2columnstr_constant(relation.provenance_column),
                new_prov_col
            )
        )
        new_proj_list = extended_proj.projection_list + (
            ExtendedProjectionListMember(
                fun_exp=new_prov_col, dst_column=new_prov_col
            ),
        )
        return ProvenanceAlgebraSet(
            self.walk(
                ExtendedProjection(
                    Constant[AbstractSet](relation.value), new_proj_list,
                )
            ).value,
            new_prov_col.value,
        )

    @add_match(NaturalJoinInverse)
    def prov_naturaljoin_inverse(self, naturaljoin):
        return self._apply_provenance_join_operation(
            naturaljoin.relation_left,
            naturaljoin.relation_right,
            NaturalJoin,
            Constant(operator.truediv),
        )

    @add_match(NaturalJoin)
    def prov_naturaljoin(self, naturaljoin):
        return self._apply_provenance_join_operation(
            naturaljoin.relation_left,
            naturaljoin.relation_right,
            NaturalJoin,
            MUL,
        )

    def _apply_provenance_join_operation(
        self, left, right, np_op, prov_binary_op
    ):
        left = self.walk(left)
        right = self.walk(right)
        res_columns = set(left.value.columns) | (
            set(right.value.columns) - {right.provenance_column}
        )
        res_columns = tuple(str2columnstr_constant(col) for col in res_columns)
        res_prov_col = str2columnstr_constant(left.provenance_column)
        # provenance columns are temporarily renamed for executing the
        # non-provenance operation on the relations
        tmp_left_col = Constant[ColumnStr](
            ColumnStr(f"{left.provenance_column}1"),
            verify_type=False,
            auto_infer_type=False,
        )
        tmp_right_col = Constant[ColumnStr](
            ColumnStr(f"{right.provenance_column}2"),
            verify_type=False,
            auto_infer_type=False,
        )
        tmp_left = RenameColumn(
            Constant[AbstractSet](left.value),
            str2columnstr_constant(left.provenance_column),
            tmp_left_col,
        )
        tmp_right = RenameColumn(
            Constant[AbstractSet](right.value),
            str2columnstr_constant(right.provenance_column),
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
                ExtendedProjectionListMember(
                    fun_exp=prov_binary_op(tmp_left_col, tmp_right_col),
                    dst_column=res_prov_col,
                ),
            )
            + tuple(
                ExtendedProjectionListMember(fun_exp=col, dst_column=col)
                for col in set(res_columns) - {res_prov_col}
            ),
        )
        return ProvenanceAlgebraSet(
            self.walk(result).value, res_prov_col.value
        )

    @add_match(Union)
    def prov_union(self, union_op):
        left = self.walk(union_op.relation_left)
        right = self.walk(union_op.relation_right)
        if left.non_provenance_columns != right.non_provenance_columns:
            raise RelationalAlgebraError(
                "All non-provenance columns must be the same: {} != {}".format(
                    left.non_provenance_columns, right.non_provenance_columns,
                )
            )
        prov_col = left.provenance_column
        result_columns = tuple(
            str2columnstr_constant(col) for col in left.non_provenance_columns
        )
        # move to non-provenance relational algebra
        np_left = Constant[AbstractSet](left.value)
        np_right = Constant[AbstractSet](right.value)
        # make the the provenance columns match
        np_right = RenameColumn(
            np_right,
            str2columnstr_constant(right.provenance_column),
            str2columnstr_constant(prov_col),
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
        new_relation = self.walk(ra_union_op)
        result = ProvenanceAlgebraSet(new_relation.value, prov_col)
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
    @add_match(NaturalJoin(ProvenanceAlgebraSet, ProvenanceAlgebraSet))
    def natural_join_rap(self, expression):
        rap_left = expression.relation_left
        rap_right = expression.relation_right
        rap_right_r = self._build_relation_constant(rap_right.relations)
        rap_right_pc = str2columnstr_constant(rap_right.provenance_column)

        cols_to_keep = [
            ExtendedProjectionListMember(
                str2columnstr_constant(c), str2columnstr_constant(c)
            )
            for c in (rap_left.relations.columns + rap_right.relations.columns)
            if c not in (
                rap_left.provenance_column,
                rap_right.provenance_column,
            )
        ]
        if rap_left.provenance_column == rap_right.provenance_column:
            rap_right_pc = str2columnstr_constant(Symbol.fresh().name)
            rap_right_r = RenameColumn(
                rap_right_r,
                str2columnstr_constant(rap_right.provenance_column),
                rap_right_pc
            )
        new_pc = str2columnstr_constant(Symbol.fresh().name)
        operation = ExtendedProjection(
            NaturalJoin(
                self._build_relation_constant(rap_left.relations),
                rap_right_r
            ),
            cols_to_keep + [
                ExtendedProjectionListMember(
                    self._semiring_mul(
                        str2columnstr_constant(rap_left.provenance_column),
                        rap_right_pc
                    ),
                    new_pc
                )
            ]
        )

        with sure_is_not_pattern():
            res = ProvenanceAlgebraSet(
                self.walk(operation).value,
                new_pc.value
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
            ProvenanceAlgebraSet,
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
        if cols == tuple(
            c for c in projection.relation.relations.columns
            if c != projection.relation.provenance_column
        ):
            return projection.relation

        with sure_is_not_pattern():
            projected_relation = projection.relation.relations.aggregate(
                cols,
                {
                    projection.relation.provenance_column:
                    self._semiring_agg_sum
                }
            )
        return ProvenanceAlgebraSet(
            projected_relation,
            projection.relation.provenance_column
        )

    @staticmethod
    def _semiring_agg_sum(x):
        args = tuple(x)
        if len(args) == 1:
            r = args[0]
        if isinstance(args[0], Expression):
            r = FunctionApplication(
                ADD, args, validate_arguments=False, verify_type=False
            )
        else:
            r = sum(args)
        return r

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
