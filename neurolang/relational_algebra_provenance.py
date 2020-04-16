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
    Selection,
    Union,
    eq_,
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
        if concat_op == relation.provenance_column:
            new_prov_col = Constant[ColumnStr](
                ColumnStr(Symbol.fresh().name),
                auto_infer_type=False,
                verify_type=False,
            )
        else:
            new_prov_col = relation.provenance_column
        return ProvenanceAlgebraSet(
            self.walk(
                ConcatenateConstantColumn(
                    Constant[AbstractSet](relation.value),
                    concat_op.column_name,
                    concat_op.column_value,
                )
            ).value,
            new_prov_col,
        )

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
        return ProvenanceAlgebraSet(
            self.walk(
                ExtendedProjection(
                    Constant[AbstractSet](relation.value),
                    extended_proj.projection_list,
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
            ),
        )
        result = Projection(result, res_columns)
        return ProvenanceAlgebraSet(self.walk(result).value, res_prov_col)

    @add_match(Union)
    def prov_union(self, union_op):
        left = self.walk(union_op.relation_left)
        right = self.walk(union_op.relation_right)

        left_cols = set(left.value.columns)
        left_cols.discard(left.provenance_column.value)
        right_cols = set(right.value.columns)
        right_cols.discard(right.provenance_column.value)

        if (
            len(left_cols.difference(right_cols)) > 0
            or len(right_cols.difference(left_cols)) > 0
        ):
            raise NeuroLangException(
                "At the union, both sets must have the same columns"
            )

        proj_columns = tuple([Constant(ColumnStr(name)) for name in left_cols])

        res1 = ConcatenateConstantColumn(
            Projection(left, proj_columns),
            Constant(ColumnStr("__new_col_union__")),
            Constant[str]("union_temp_value_1"),
        )
        res2 = ConcatenateConstantColumn(
            Projection(right, proj_columns),
            Constant(ColumnStr("__new_col_union__")),
            Constant[str]("union_temp_value_2"),
        )

        left = self.walk(res1)
        right = self.walk(res2)

        new_relation = ProvenanceAlgebraSet(
            left.value | right.value, union_op.relation_left.provenance_column
        )

        return self.walk(Projection(new_relation, proj_columns))

    @add_match(Constant[AbstractSet])
    def constant_relation_or_provenance_set(self, relation):
        return relation
