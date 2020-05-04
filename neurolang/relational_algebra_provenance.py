from typing import AbstractSet

from .exceptions import NeuroLangException
from .expression_walker import ExpressionWalker, add_match
from .expressions import Constant, FunctionApplication, Symbol
from .relational_algebra import (
    Column,
    ColumnStr,
    ConcatenateConstantColumn,
    EquiJoin,
    ExtendedProjection,
    ExtendedProjectionListMember,
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
    str2columnstr_constant,
)


class ProvenanceAlgebraSet(Constant):
    def __init__(self, relations, provenance_column):
        self.relations = relations
        self.provenance_column = provenance_column

    @property
    def value(self):
        return self.relations

    @property
    def non_provenance_columns(self):
        non_prov_cols = set(self.value.columns) - {
            self.provenance_column.value
        }
        return tuple(
            Constant[ColumnStr](
                ColumnStr(col), verify_type=False, auto_infer_type=False
            )
            for col in sorted(non_prov_cols)
        )


def check_do_not_share_non_prov_col(prov_set_1, prov_set_2):
    shared_columns = set(prov_set_1.non_provenance_columns) & set(
        prov_set_2.non_provenance_columns
    )
    if len(shared_columns) > 0:
        raise NeuroLangException(
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
            FunctionApplication(eq_, (Constant[Column], Constant[Column])),
        )
    )
    def selection_between_columns(self, selection):
        relation = self.walk(selection.relation)
        if any(
            col == relation.provenance_column for col in selection.formula.args
        ):
            raise NeuroLangException("Cannot select on provenance column")
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
            raise NeuroLangException("Cannot select on provenance column")
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
                rel_res, rel_temp, Product
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
            relation.provenance_column,
            new_prov_col,
        )
        relation = ConcatenateConstantColumn(
            relation, concat_op.column_name, concat_op.column_value,
        )
        return ProvenanceAlgebraSet(self.walk(relation).value, new_prov_col)

    @add_match(Projection)
    def prov_projection(self, projection):
        prov_set = self.walk(projection.relation)
        prov_col = prov_set.provenance_column.value
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
        if rename_column.src == prov_relation.provenance_column:
            new_prov_col = rename_column.dst
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
            prov_relation.provenance_column, None
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
            new_prov_col = Constant(ColumnStr(Symbol.fresh().name))
        else:
            new_prov_col = relation.provenance_column
        relation = self.walk(
            RenameColumn(relation, relation.provenance_column, new_prov_col)
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
            new_prov_col,
        )

    @add_match(NaturalJoin)
    def prov_naturaljoin(self, naturaljoin):
        return self._apply_provenance_join_operation(
            naturaljoin.relation_left, naturaljoin.relation_right, NaturalJoin,
        )

    def _apply_provenance_join_operation(self, left, right, np_op):
        left = self.walk(left)
        right = self.walk(right)
        res_columns = set(left.value.columns) | (
            set(right.value.columns) - {right.provenance_column.value}
        )
        res_columns = tuple(Constant(ColumnStr(col)) for col in res_columns)
        res_prov_col = left.provenance_column
        # provenance columns are temporarily renamed for executing the
        # non-provenance operation on the relations
        tmp_left_col = Constant[ColumnStr](
            ColumnStr(f"{left.provenance_column.value}1"),
            verify_type=False,
            auto_infer_type=False,
        )
        tmp_right_col = Constant[ColumnStr](
            ColumnStr(f"{right.provenance_column.value}2"),
            verify_type=False,
            auto_infer_type=False,
        )
        tmp_left = RenameColumn(
            Constant[AbstractSet](left.value),
            left.provenance_column,
            tmp_left_col,
        )
        tmp_right = RenameColumn(
            Constant[AbstractSet](right.value),
            right.provenance_column,
            tmp_right_col,
        )
        tmp_np_op_args = (tmp_left, tmp_right)
        if np_op is Product:
            tmp_non_prov_result = np_op(tmp_np_op_args)
        elif np_op is NaturalJoin:
            tmp_non_prov_result = np_op(*tmp_np_op_args)
        else:
            raise NeuroLangException(
                "Cannot apply non-provenance operation: {}".format(np_op)
            )
        result = ExtendedProjection(
            tmp_non_prov_result,
            (
                ExtendedProjectionListMember(
                    fun_exp=tmp_left_col * tmp_right_col,
                    dst_column=res_prov_col,
                ),
            )
            + tuple(
                ExtendedProjectionListMember(fun_exp=col, dst_column=col)
                for col in set(res_columns) - {res_prov_col}
            ),
        )
        return ProvenanceAlgebraSet(self.walk(result).value, res_prov_col)

    @add_match(Union)
    def prov_union(self, union_op):
        left = self.walk(union_op.relation_left)
        right = self.walk(union_op.relation_right)
        if left.non_provenance_columns != right.non_provenance_columns:
            raise NeuroLangException(
                "All non-provenance columns must be the same: {} != {}".format(
                    left.non_provenance_columns, right.non_provenance_columns,
                )
            )
        prov_col = left.provenance_column
        result_columns = left.non_provenance_columns
        # move to non-provenance relational algebra
        np_left = Constant[AbstractSet](left.value)
        np_right = Constant[AbstractSet](right.value)
        # make the the provenance columns match
        np_right = RenameColumn(np_right, right.provenance_column, prov_col)
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
