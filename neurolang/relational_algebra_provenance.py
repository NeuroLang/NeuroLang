import operator
from typing import AbstractSet

from .exceptions import NeuroLangException
from .expression_walker import ExpressionWalker, PatternWalker, add_match
from .expressions import Constant, Definition, FunctionApplication
from .relational_algebra import (
    Column,
    ColumnStr,
    EquiJoin,
    NaturalJoin,
    Product,
    Projection,
    RelationalAlgebraOperation,
    RenameColumn,
    Selection,
    Union,
    eq_,
)
from .utils.relational_algebra_set import RelationalAlgebraExpression


def arithmetic_operator_string(op):
    """
    Get the string representation of an arithmetic operator.

    Parameters
    ----------
    op : builting operator
        Python builtin operator (add, sub, mul or truediv).

    Returns
    -------
    str
        String representation of the operator (e.g. operator.add is "+").

    """
    return {
        operator.add: "+",
        operator.sub: "-",
        operator.mul: "*",
        operator.truediv: "/",
    }[op]


def is_arithmetic_operation(exp):
    """
    Whether the expression is an arithmetic operation function application.

    Parameters
    ----------
    exp : Expression

    Returns
    -------
    bool

    """
    return (
        isinstance(exp, FunctionApplication)
        and isinstance(exp.functor, Constant)
        and exp.functor.value
        in {operator.add, operator.sub, operator.mul, operator.truediv}
    )


class CrossProductNonProvenance(RelationalAlgebraOperation):
    def __init__(self, relations):
        self.relations = tuple(relations)

    def __repr__(self):
        return (
            "["
            + f"\N{n-ary times operator}".join(repr(r) for r in self.relations)
            + "]"
        )


class ProjectionNonProvenance(RelationalAlgebraOperation):
    def __init__(self, relation, attributes):
        self.relation = relation
        self.attributes = attributes

    def __repr__(self):
        return (
            f"\N{GREEK CAPITAL LETTER PI}"
            f"_{self.attributes}({self.relation})"
        )


class NaturalJoinNonProvenance(RelationalAlgebraOperation):
    def __init__(self, relation_left, relation_right):
        self.relation_left = relation_left
        self.relation_right = relation_right

    def __repr__(self):
        return f"[{self.relation_left}" f"\N{JOIN}" f"{self.relation_right}]"


class ExtendedProjection(RelationalAlgebraOperation):
    """
    General operation defining string-based relational algebra projections
    allowing flexible computations on a relation's columns.

    Attributes
    ----------
    relation : Expression[AbstractSet]
        Relation on which the projections are applied.
    projection_list : Tuple[ExtendedProjectionListMember]
        List of projections to apply.

    Notes
    -----
    The concept of extended projection is formally defined in section 5.2.5
    of [1]_.

    .. [1] Garcia-Molina, Hector, Jeffrey D. Ullman, and Jennifer Widom.
       "Database systems: the complete book." (2009).

    """

    def __init__(self, relation, projection_list):
        self.relation = relation
        self.projection_list = tuple(projection_list)

    def __repr__(self):
        join_str = "," if len(self.projection_list) < 2 else ",\n"
        return "π_[{}]({})".format(
            join_str.join([repr(member) for member in self.projection_list]),
            repr(self.relation),
        )


class ExtendedProjectionListMember(Definition):
    """
    Member of a projection list.

    Attributes
    ----------
    fun_exp : `Constant[str]`
        Constant string representation of the extended projection operation.
    dst_column : `Constant[ColumnStr]` or `Symbol[ColumnStr]`
        Constant column string of the destination column.

    Notes
    -----
    As described in [1]_, a projection list member can either be
        - a single attribute (column) name in the relation, resulting in a
          normal non-extended projection,
        - an expression `x -> y` where `x` and `y` are both attribute (column)
          names, `x` effectively being rename as `y`,
        - or an expression `E -> z` where `E` is an expression involving
          attributes of the relation, arithmetic operators, and string
          operators, and `z` is a new name for the attribute that results from
          the calculation implied by `E`. For example, `a + b -> x` represents
          the sum of the attributes `a` and `b`, renamed `x`.

    .. [1] Garcia-Molina, Hector, Jeffrey D. Ullman, and Jennifer Widom.
       "Database systems: the complete book." (2009).

    """

    def __init__(self, fun_exp, dst_column):
        self.fun_exp = fun_exp
        self.dst_column = dst_column

    def __repr__(self):
        return "{} -> {}".format(self.fun_exp, self.dst_column)


class ConcatenateConstantColumn(RelationalAlgebraOperation):
    """
    Add a column with a repeated constant value to a relation.

    Attributes
    ----------
    relation : Constant[RelationalAlgebraSet]
        Relation to which the column will be added.

    column_name : Constant[ColumnStr] or Symbol[ColumnStr]
        Name of the newly added column.

    column_value : Constant
        Constant value repeated in the new column.

    """

    def __init__(self, relation, column_name, column_value):
        self.relation = relation
        self.column_name = column_name
        self.column_value = column_value


class ProvenanceAlgebraSet(Constant):
    def __init__(self, relations, provenance_column):
        self.relations = relations
        self.provenance_column = provenance_column

    @property
    def value(self):
        return self.relations


class RelationalAlgebraProvenanceCountingSolver(ExpressionWalker):
    """
    Mixing that walks through relational algebra expressions and
    executes the operations and provenance calculations.
    """

    @add_match(
        Selection(
            ..., FunctionApplication(eq_, (Constant[Column], Constant[Column]))
        )
    )
    def selection_between_columns(self, selection):
        col1, col2 = selection.formula.args
        selected_relation = self.walk(
            selection.relation
        ).value.selection_columns({col1.value: col2.value})

        return ProvenanceAlgebraSet(
            selected_relation, selection.relation.provenance_column
        )

    @add_match(
        Selection(..., FunctionApplication(eq_, (Constant[Column], ...)))
    )
    def selection_by_constant(self, selection):
        col, val = selection.formula.args
        selected_relation = self.walk(selection.relation).value.selection(
            {col.value: val.value}
        )

        return ProvenanceAlgebraSet(
            selected_relation, selection.relation.provenance_column
        )

    @add_match(Product)
    def prov_product(self, product):
        rel_res = self.walk(product.relations[0])
        for relation in product.relations[1:]:
            rel_temp = self.walk(relation)
            rel_res, rel_temp = self._remove_common_columns(rel_res, rel_temp)
            rel_res = self._apply_provenance_join_operation(
                rel_res, rel_temp, CrossProductNonProvenance
            )
        return rel_res

    def _remove_common_columns(self, rel_res, rel_temp):
        set_res_cols = set(rel_res.value.columns)
        set_res_cols.discard(rel_res.provenance_column.value)
        set_temp_cols = set(rel_temp.value.columns)
        set_temp_cols.discard(rel_temp.provenance_column.value)
        common = set_res_cols & set_temp_cols
        if common:
            cols = set_res_cols.difference(common.union())
            if len(cols) == 0:
                cols = set_temp_cols.difference(common)
                cols.add(rel_temp.provenance_column.value)
                temp_provenance = rel_temp.provenance_column
                rel_temp = ProjectionNonProvenance(
                    rel_temp,
                    tuple([Constant(ColumnStr(name)) for name in cols]),
                )
                rel_temp = self.walk(rel_temp)
                rel_temp = ProvenanceAlgebraSet(rel_temp, temp_provenance)
            else:
                res_provenance = rel_res.provenance_column
                cols.add(rel_res.provenance_column.value)
                rel_res = ProjectionNonProvenance(
                    rel_res,
                    tuple([Constant(ColumnStr(name)) for name in cols]),
                )
                rel_res = self.walk(rel_res)
                rel_res = ProvenanceAlgebraSet(rel_res, res_provenance)

        return rel_res, rel_temp

    @add_match(Projection)
    def prov_projection(self, projection):
        relation = self.walk(projection.relation)

        group_columns = [col.value for col in projection.attributes]
        new_container = relation.value.aggregate(
            group_columns, {relation.provenance_column.value: sum}
        )

        return ProvenanceAlgebraSet(new_container, relation.provenance_column)

    @add_match(CrossProductNonProvenance)
    def ra_product(self, product):
        if len(product.relations) == 0:
            return ProvenanceAlgebraSet(set(), product.provenance_column)

        res = self.walk(product.relations[0])
        res = res.value
        for relation in product.relations[1:]:
            res = res.cross_product(self.walk(relation).value)
        return res

    @add_match(ProjectionNonProvenance)
    def ra_projection(self, projection):
        relation = self.walk(projection.relation)
        cols = tuple(v.value for v in projection.attributes)
        projected_relation = relation.value.projection(*cols)
        return projected_relation

    @add_match(EquiJoin(ProvenanceAlgebraSet, ..., ProvenanceAlgebraSet, ...))
    def prov_equijoin(self, equijoin):
        raise NotImplementedError("EquiJoin is not implemented.")

    @add_match(RenameColumn)
    def prov_rename_column(self, rename_column):
        relation = self.walk(rename_column.relation)
        src = rename_column.src.value
        dst = rename_column.dst.value
        new_set = relation.value
        if len(new_set) > 0:
            new_set = new_set.rename_column(src, dst)

        if src == relation.provenance_column.value:
            new_prov = dst
        else:
            new_prov = relation.provenance_column.value

        return ProvenanceAlgebraSet(new_set, new_prov)

    @add_match(NaturalJoin)
    def prov_naturaljoin(self, naturaljoin):
        return self._apply_provenance_join_operation(
            naturaljoin.relation_left,
            naturaljoin.relation_right,
            NaturalJoinNonProvenance,
        )

    def _apply_provenance_join_operation(self, left, right, np_op):
        res_columns = set(left.value.columns) | (
            set(right.value.columns) - {right.provenance_column.value}
        )
        res_columns = tuple(Constant(ColumnStr(col)) for col in res_columns)
        res_prov_col = left.provenance_column
        # provenance columns are temporarily renamed for executing the
        # non-provenance operation on the relations
        tmp_left_col = Constant[ColumnStr](
            ColumnStr(f"{left.provenance_column.value}1"), verify_type=False,
        )
        tmp_right_col = Constant[ColumnStr](
            ColumnStr(f"{right.provenance_column.value}2"), verify_type=False,
        )
        tmp_left = RenameColumn(left, left.provenance_column, tmp_left_col)
        tmp_right = RenameColumn(right, right.provenance_column, tmp_right_col)
        tmp_np_op_args = (tmp_left, tmp_right)
        if np_op is CrossProductNonProvenance:
            tmp_non_prov_result = ProvenanceAlgebraSet(
                np_op(tmp_np_op_args), right.provenance_column
            )
        elif np_op is NaturalJoinNonProvenance:
            tmp_non_prov_result = ProvenanceAlgebraSet(
                np_op(*tmp_np_op_args), tmp_left_col
            )
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
        result = ProjectionNonProvenance(result, res_columns)
        return ProvenanceAlgebraSet(self.walk(result), res_prov_col)

    @add_match(NaturalJoinNonProvenance)
    def ra_naturaljoin(self, naturaljoin):
        left = self.walk(naturaljoin.relation_left)
        right = self.walk(naturaljoin.relation_right)
        res = left.value.naturaljoin(right.value)
        return res

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

    @add_match(ConcatenateConstantColumn)
    def concatenate_column(self, concat_op):
        relation = self.walk(concat_op.relation)
        new_column_name = concat_op.column_name
        new_column_value = concat_op.column_value
        res = ExtendedProjection(
            relation,
            (
                ExtendedProjectionListMember(
                    fun_exp=new_column_value, dst_column=new_column_name
                ),
            ),
        )
        new_relation = self.walk(res)

        return new_relation

    @add_match(Constant[AbstractSet])
    def prov_relation(self, relation):
        return relation

    @add_match(ExtendedProjection)
    def prov_extended_projection(self, proj_op):
        relation = self.walk(proj_op.relation)
        str_arithmetic_walker = StringArithmeticWalker()
        eval_expressions = {}
        for member in proj_op.projection_list:
            eval_expressions[
                member.dst_column.value
            ] = str_arithmetic_walker.walk(self.walk(member.fun_exp))
        new_container = relation.value.extended_projection(eval_expressions)
        return ProvenanceAlgebraSet(new_container, relation.provenance_column)

    @add_match(FunctionApplication, is_arithmetic_operation)
    def prov_arithmetic_operation(self, arithmetic_op):
        return FunctionApplication[arithmetic_op.type](
            arithmetic_op.functor,
            tuple(self.walk(arg) for arg in arithmetic_op.args),
        )


class StringArithmeticWalker(PatternWalker):
    """
    Walker translating an Expression with basic arithmetic operations on a
    relation's columns to its equivalent string representation.

    The expression can refer to the names a relation's columns or to the
    length of an other constant relation.

    """

    @add_match(Constant)
    def constant(self, cst):
        return cst.value

    @add_match(FunctionApplication(Constant(len), ...))
    def len(self, fa):
        if not isinstance(fa.args[0], Constant[AbstractSet]):
            raise NeuroLangException("Expected constant RA relation")
        return str(len(fa.args[0].value))

    @add_match(FunctionApplication, is_arithmetic_operation)
    def arithmetic_operation(self, fa):
        return RelationalAlgebraExpression(
            "({} {} {})".format(
                self.walk(fa.args[0]),
                arithmetic_operator_string(fa.functor.value),
                self.walk(fa.args[1]),
            )
        )
