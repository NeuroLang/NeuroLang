import math
import operator

from .exceptions import RelationalAlgebraError
from .expression_walker import (
    PatternWalker,
    add_match
)
from .expressions import (
    Constant,
    FunctionApplication,
    Symbol,
    sure_is_not_pattern,
)
from .relational_algebra import (
    ColumnInt,
    ConcatenateConstantColumn,
    Difference,
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
    ReplaceNull,
    Selection,
    UnaryRelationalAlgebraOperation,
    Union,
    eq_,
    int2columnint_constant,
    str2columnstr_constant
)
from .utils import OrderedSet

ADD = Constant(operator.add)
MUL = Constant(operator.mul)
SUB = Constant(operator.sub)
SUM = Constant(sum)
LOG = Constant(math.log)
EXP = Constant(math.exp)
ONE = Constant[float](1.)


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


class ProvenanceAlgebraSet(UnaryRelationalAlgebraOperation):
    def __init__(self, relation, provenance_column):
        self.relation = relation
        self.provenance_column = provenance_column
        self._columns = None
        self._non_provenance_columns = None

    def columns(self):
        if self._columns is None:
            if isinstance(self.relation, Constant):
                columns = OrderedSet(
                    str2columnstr_constant(c) if isinstance(c, str)
                    else int2columnint_constant(c)
                    for c in self.relation.value.columns
                )
            else:
                columns = self.relation.columns()
            self._columns = columns
        return self._columns

    @property
    def non_provenance_columns(self):
        if self._non_provenance_columns is None:
            self._non_provenance_columns = (
                self.columns() - {self.provenance_column}
            )
        return self._non_provenance_columns

    def __repr__(self):
        return (f"RAP[{self.relation}, {self.provenance_column}]")


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


class LiftedPlanProjection(RelationalAlgebraOperation):
    def __init__(self, relation, attributes):
        self.relation = relation
        self.attributes = attributes


class IndependentProjection(LiftedPlanProjection):
    def __repr__(self) -> str:
        if self.attributes is Ellipsis:
            attributes_repr = "..."
        else:
            attributes_repr = ",".join(repr(attr) for attr in self.attributes)
        return "ind-π_[{}]({})".format(attributes_repr, repr(self.relation))


class DisjointProjection(LiftedPlanProjection):
    def __repr__(self) -> str:
        if self.attributes is Ellipsis:
            attributes_repr = "..."
        else:
            attributes_repr = ",".join(repr(attr) for attr in self.attributes)
        return "disj-π_[{}]({})".format(attributes_repr, repr(self.relation))


class WeightedNaturalJoinSolverMixin(PatternWalker):
    @add_match(
        WeightedNaturalJoin,
        lambda join_op: all(
            isinstance(relation, ProvenanceAlgebraSet)
            for relation in join_op.relations
        )
    )
    def prov_weighted_join(self, join_op):
        relations = join_op.relations
        weights = join_op.weights

        prov_columns = [
            str2columnstr_constant(Symbol.fresh().name)
            for _ in relations
        ]

        dst_columns = relations[0].non_provenance_columns
        for relation in relations[1:]:
            dst_columns = dst_columns | relation.non_provenance_columns
        dst_columns = tuple(dst_columns)

        relations = [
            ExtendedProjection(
                relation.relation,
                (FunctionApplicationListMember(
                    weight * relation.provenance_column,
                    prov_column
                ),) + tuple(
                    FunctionApplicationListMember(
                        c, c
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
                    c, c
                )
                for c in dst_columns
            )
        )

        return self.walk(ProvenanceAlgebraSet(
            relation,
            prov_col
        ))


class IndependentDisjointProjectionsAndUnionMixin(PatternWalker):
    @add_match(IndependentProjection(ProvenanceAlgebraSet, ...))
    def independent_projection(self, proj_op):
        prov_set = proj_op.relation
        prov_col = prov_set.provenance_column
        proj_list = [
            FunctionApplicationListMember(col, col)
            for col in prov_set.non_provenance_columns
        ]
        proj_list.append(
            FunctionApplicationListMember(
                LOG(ONE - prov_col),
                prov_col,
            )
        )
        relation = ExtendedProjection(prov_set.relation, proj_list)
        relation = GroupByAggregation(
            relation,
            groupby=proj_op.attributes,
            aggregate_functions=(
                FunctionApplicationListMember(
                    FunctionApplication(Constant(sum), (prov_col,)),
                    prov_col,
                ),
            ),
        )
        proj_list = [
            FunctionApplicationListMember(col, col)
            for col in relation.groupby
        ]
        proj_list.append(
            FunctionApplicationListMember(
               ONE - EXP(prov_col),
               prov_col,
            )
        )
        relation = ExtendedProjection(relation, proj_list)
        return ProvenanceAlgebraSet(relation, prov_col)

    @add_match(DisjointProjection(ProvenanceAlgebraSet, ...))
    def disjoint_projection(self, proj_op):
        prov_set = proj_op.relation
        prov_col = prov_set.provenance_column
        aggregate_functions = [
            FunctionApplicationListMember(
                FunctionApplication(Constant(sum), (prov_col,)),
                prov_col,
            ),
        ]
        operation = GroupByAggregation(
            prov_set.relation,
            proj_op.attributes,
            aggregate_functions,
        )
        res = ProvenanceAlgebraSet(operation, prov_col)
        return res

    @add_match(Union(ProvenanceAlgebraSet, ProvenanceAlgebraSet))
    def union_rap(self, union):
        prov_column_left = union.relation_left.provenance_column
        prov_column_right = union.relation_right.provenance_column
        relation_left = union.relation_left.relation
        relation_right = union.relation_right.relation
        if prov_column_left != prov_column_right:
            relation_right = RenameColumn(
                relation_right,
                prov_column_right,
                prov_column_left,
            )
        columns_to_keep = union.relation_left.non_provenance_columns
        operation = IndependentProjection(
            ProvenanceAlgebraSet(
                Union(relation_left, relation_right),
                prov_column_left
            ),
            columns_to_keep
        )
        return self.walk(operation)


class BuildProvenanceAlgebraSetWalkIntoMixin(PatternWalker):
    @add_match(
        ProvenanceAlgebraSet,
        lambda exp: any(not isinstance(arg, Constant) for arg in exp.unapply())
    )
    def cycle_in_build_provenance_algebra_set(self, expression):
        relation = self.walk(expression.relation)
        provenance_column = self.walk(expression.provenance_column)
        if (
            (relation is not expression.relation) or
            (provenance_column is not expression.provenance_column)
        ):
            return self.walk(
                ProvenanceAlgebraSet(relation, provenance_column)
            )
        else:
            return expression


class ProvenanceSelectionMixin(PatternWalker):
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
                columns[formula.args[0].value],
                columns[formula.args[1].value],
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
                columns[formula.args[0].value],
                formula.args[1]
            )
        )
        return self.walk(Selection(selection.relation, new_formula))

    @add_match(
        Selection(ProvenanceAlgebraSet, ...)
    )
    def selection_provenance_set(self, selection):
        res = ProvenanceAlgebraSet(
            Selection(
                selection.relation.relation,
                selection.formula
            ),
            selection.relation.provenance_column
        )
        return self.walk(res)


class ProvenanceColumnManipulationMixin(PatternWalker):
    @add_match(ConcatenateConstantColumn(ProvenanceAlgebraSet, ..., ...))
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

        res = ProvenanceAlgebraSet(
            ConcatenateConstantColumn(
                relation,
                concat_op.column_name,
                concat_op.column_value
            ),
            new_prov_col
        )
        return self.walk(res)

    @add_match(NameColumns(ProvenanceAlgebraSet, ...))
    def name_columns_rap(self, expression):
        relation = expression.relation.relation
        columns_to_name = (
            c for c in expression.relation.columns()
            if c != expression.relation.provenance_column
        )
        ne = RenameColumns(
            relation,
            tuple(
                (src, dst)
                for src, dst in zip(
                    columns_to_name, expression.column_names
                )
            )
        )
        return self.walk(ProvenanceAlgebraSet(
            ne,
            expression.relation.provenance_column
        ))

    @add_match(RenameColumn(ProvenanceAlgebraSet, ..., ...))
    def prov_rename_column(self, rename_column):
        if rename_column.src == rename_column.relation.provenance_column:
            provenance_column = rename_column.dst
        else:
            provenance_column = rename_column.relation.provenance_column
        res = ProvenanceAlgebraSet(
            RenameColumn(
                rename_column.relation.relation,
                rename_column.src,
                rename_column.dst
            ),
            provenance_column
        )
        return self.walk(res)

    @add_match(RenameColumns(ProvenanceAlgebraSet, ...))
    def prov_rename_columns(self, rename_columns):
        res = ProvenanceAlgebraSet(
            RenameColumns(
                rename_columns.relation.relation,
                rename_columns.renames
            ),
            rename_columns.relation.provenance_column
        )
        return self.walk(res)


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
    @add_match(ExtendedProjection(ProvenanceAlgebraSet, ...))
    def prov_extended_projection(self, extended_proj):
        relation = extended_proj.relation
        equality_columns = {
            c.dst_column for c in extended_proj.projection_list
            if c.fun_exp == c.dst_column
        }
        if equality_columns != relation.non_provenance_columns:
            raise ValueError("Invalid provenance ExtendedProjection")

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
        return self.walk(ProvenanceAlgebraSet(
                ExtendedProjection(
                    relation.relation, new_proj_list,
                ),
                new_prov_col
        ))


class ProvenanceSetOperationsMixin(PatternWalker):
    @add_match(Union(ProvenanceAlgebraSet, ProvenanceAlgebraSet))
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
        result = ProvenanceAlgebraSet(ra_union_op, prov_col)
        # provenance projection that removes the dummy column and sums the
        # provenance of matching tuples
        result = Projection(result, result_columns)
        return self.walk(result)


class RelationalAlgebraProvenanceExpressionSemringSolverMixin(
    ProvenanceSelectionMixin,
    ProvenanceExtendedProjectionMixin,
    ProvenanceColumnManipulationMixin,
    ProvenanceSetOperationsMixin,
):
    @add_match(NaturalJoin(
        ProvenanceAlgebraSet, ProvenanceAlgebraSet
    ))
    def prov_naturaljoin(self, naturaljoin):
        return self.walk(self._apply_provenance_join_operation(
            naturaljoin.relation_left,
            naturaljoin.relation_right,
            NaturalJoin,
            self._semiring_mul,
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
        tmp_left_col = str2columnstr_constant(
            f"{left.provenance_column.value}1"
        )
        tmp_right_col = str2columnstr_constant(
            f"{right.provenance_column.value}2"
        )
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
        return ProvenanceAlgebraSet(
            result, res_prov_col
        )

    @add_match(
        Product,
        lambda product: all(
            isinstance(ras, ProvenanceAlgebraSet)
            for ras in product.relations
        )
    )
    def prov_product(self, product):
        rel_res = product.relations[0]
        for relation in product.relations[1:]:
            check_do_not_share_non_prov_col(rel_res, relation)
            rel_res = self._apply_provenance_join_operation(
                rel_res, relation, Product, self._semiring_mul,
            )
        return self.walk(rel_res)

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
                att = columns[att.value]
            new_attributes += (att,)
        return self.walk(Projection(projection.relation, new_attributes))

    @add_match(Projection(ProvenanceAlgebraSet, ...))
    def projection_rap(self, projection):
        if (
            set(projection.attributes) ==
            projection.relation.non_provenance_columns
        ):
            return self.walk(projection.relation)

        aggregate_functions = (
            FunctionApplicationListMember(
                self._semiring_agg_sum(
                    (projection.relation.provenance_column,)
                ),
                projection.relation.provenance_column
            ),
        )
        operation = GroupByAggregation(
            projection.relation.relation,
            projection.attributes,
            aggregate_functions,
        )

        with sure_is_not_pattern():
            res = self.walk(
                ProvenanceAlgebraSet(
                    operation,
                    projection.relation.provenance_column
                )
            )
        return res

    def _semiring_agg_sum(self, args):
        return FunctionApplication(
            SUM, args, validate_arguments=False, verify_type=False
        )

    @add_match(
        Difference(ProvenanceAlgebraSet, ProvenanceAlgebraSet)
    )
    def difference(self, diff):
        left = diff.relation_left
        right = diff.relation_right
        res_columns = tuple(
            col for col in left.columns()
        )
        res_prov_col = left.provenance_column
        tmp_left_prov_col = str2columnstr_constant(Symbol.fresh().name)
        tmp_right_prov_col = str2columnstr_constant(Symbol.fresh().name)
        tmp_left = RenameColumn(
            left.relation,
            left.provenance_column,
            tmp_left_prov_col,
        )
        tmp_right = RenameColumn(
            right.relation,
            right.provenance_column,
            tmp_right_prov_col,
        )
        tmp_np_op_args = (tmp_left, tmp_right)
        tmp_non_prov_result = LeftNaturalJoin(*tmp_np_op_args)
        tmp_non_prov_result = ReplaceNull(
            tmp_non_prov_result,
            tmp_right_prov_col,
            Constant(0.)
        )
        result = ExtendedProjection(
            tmp_non_prov_result,
            (
                FunctionApplicationListMember(
                    fun_exp=self._semiring_monus(
                        tmp_left_prov_col, tmp_right_prov_col
                    ),
                    dst_column=res_prov_col,
                ),
            )
            + tuple(
                FunctionApplicationListMember(fun_exp=col, dst_column=col)
                for col in set(res_columns) - {res_prov_col}
            ),
        )
        return self.walk(ProvenanceAlgebraSet(
            result, res_prov_col
        ))

    def _semiring_monus(self, left, right):
        return MUL(
            left,
            SUB(Constant(1), right)
        )


class RelationalAlgebraProvenanceCountingSolverMixin(
    RelationalAlgebraProvenanceExpressionSemringSolverMixin,
    PatternWalker
):
    """
    Mixing that walks through relational algebra expressions and
    executes the operations and provenance calculations.

    """

    @add_match(NaturalJoinInverse)
    def prov_naturaljoin_inverse(self, naturaljoin):
        return self.walk(self._apply_provenance_join_operation(
            naturaljoin.relation_left,
            naturaljoin.relation_right,
            NaturalJoin,
            Constant(operator.truediv),
        ))


class RelationalAlgebraProvenanceCountingSolver(
    BuildProvenanceAlgebraSetWalkIntoMixin,
    RelationalAlgebraProvenanceCountingSolverMixin,
    RelationalAlgebraSolver,
):
    pass


class RelationalAlgebraProvenanceExpressionSemringSolver(
    BuildProvenanceAlgebraSetWalkIntoMixin,
    RelationalAlgebraProvenanceExpressionSemringSolverMixin,
    RelationalAlgebraSolver,
):
    pass
