"""Optimisers eliminating, simplifying and pushing projections."""

from itertools import permutations

from nl_expr import expression_walker as ew
from nl_expr.expressions import Constant, FunctionApplication, Symbol
from nl_expr.utils.orderedset import OrderedSet
from ..relational_algebra import (
    COUNT,
    Column,
    ColumnInt,
    ColumnStr,
    ExtendedProjection,
    FullOuterNaturalJoin,
    FunctionApplicationListMember,
    GroupByAggregation,
    LeftNaturalJoin,
    NaturalJoin,
    NaryNaturalJoin,
    Product,
    Projection,
    RelationalAlgebraStringExpression,
    RenameColumns,
    ReplaceColumns,
    ReplaceNull,
    Selection,
    eq_,
    str2columnstr_constant
)
from .columns import _project_onto_operand, _ra_columns


def _is_trivial_nary_projection(expression):
    """Guard: True if a Projection keeps every column of its relation."""
    projected = {
        str(a.name) if hasattr(a, 'name') else str(a)
        for a in expression.attributes
    }
    return projected == {str(c) for c in expression.relation.columns()}


class EliminateTrivialProjections(ew.PatternWalker):
    @ew.add_match(
        Projection(Constant, ...),
        lambda expression: (
            tuple(c.value for c in expression.attributes)
            == tuple(c for c in expression.relation.value.columns)
        ) or (
            tuple(str(c.value) for c in expression.attributes)
            == tuple(c for c in expression.relation.value.columns)
        )
    )
    def eliminate_trivial_projection(self, expression):
        return expression.relation

    @ew.add_match(
        Projection(NaryNaturalJoin, ...),
        _is_trivial_nary_projection
    )
    def eliminate_trivial_nary_projection(self, expression):
        return self.walk(expression.relation)

    @ew.add_match(
        Projection(Projection, ...),
        lambda expression: all(
            isinstance(attribute, Constant[ColumnInt])
            for attribute in (
                expression.attributes + expression.relation.attributes
            )
        )
    )
    def eliminate_trivial_nested_unnamed_projection(self, expression):
        inner_attributes = expression.relation.attributes
        outer_attributes = expression.attributes

        resulting_attributes = tuple(
            inner_attributes[attribute.value]
            for attribute in outer_attributes
        )

        return self.walk(
            Projection(expression.relation.relation, resulting_attributes)
        )

    @ew.add_match(Projection(Projection, ...))
    def eliminate_trivial_nested_projection(self, expression):
        return self.walk(
            Projection(expression.relation.relation, expression.attributes)
        )

    @ew.add_match(Projection(ExtendedProjection, ...))
    def simplify_projection_extended_projection(self, expression):
        new_projections = tuple(
            proj for proj in expression.relation.projection_list
            if proj.dst_column in expression.attributes
        )
        new_relation = self.walk(
            ExtendedProjection(
                expression.relation.relation,
                new_projections
            )
        )
        return new_relation

    @ew.add_match(
        ExtendedProjection,
        lambda e: all(
            isinstance(p.fun_exp, Constant[ColumnStr]) and
            (p.fun_exp == p.dst_column)
            for p in e.projection_list
        )
    )
    def convert_extended_projection_2_projection(self, expression):
        return self.walk(Projection(
            expression.relation,
            tuple(p.dst_column for p in expression.projection_list)
        ))

    @ew.add_match(
        ExtendedProjection,
        lambda e: all(
            isinstance(p.fun_exp, Constant[ColumnStr])
            for p in e.projection_list
        ) and (
            len(set(p.fun_exp for p in e.projection_list))
            == len(e.projection_list)
            == len(e.relation.columns())
        )
    )
    def convert_extended_projection_2_rename(self, expression):
        return self.walk(RenameColumns(
            expression.relation,
            tuple(
                (p.fun_exp, p.dst_column)
                for p in expression.projection_list
            )
        ))


class SimplifyExtendedProjectionsWithConstants(ew.PatternWalker):
    @ew.add_match(
        ExtendedProjection(
            ExtendedProjection,
            ...
        ),
        lambda expression: all(
            isinstance(int_proj.fun_exp, (Constant, FunctionApplication))
            for int_proj in expression.relation.projection_list
        )
    )
    def nested_extended_projection_constant(self, expression):
        rew = ReplaceColumns({
            p.dst_column: p.fun_exp
            for p in expression.relation.projection_list
        })
        new_projections = (
            FunctionApplicationListMember(rew.walk(p.fun_exp), p.dst_column)
            for p in expression.projection_list
        )
        return self.walk(
            ExtendedProjection(
                expression.relation.relation,
                new_projections
            )
        )

    @ew.add_match(
        ExtendedProjection(
            NaturalJoin(ExtendedProjection, ...),
            ...
        ),
        lambda expression: any(
            _function_application_list_member_has_constant_exp(int_proj)
            for int_proj in expression.relation.relation_left.projection_list
        )
    )
    def nested_extended_projection_naturaljoin_constant_l(self, expression):
        return self._nested_ep_join_constant_left(expression)

    @ew.add_match(
        ExtendedProjection(
            LeftNaturalJoin(ExtendedProjection, ...),
            ...
        ),
        lambda expression: any(
           _function_application_list_member_has_constant_exp(int_proj)
           for int_proj in expression.relation.relation_left.projection_list
        )
    )
    def nested_extended_projection_leftnaturaljoin_constant(self, expression):
        return self._nested_ep_join_constant_left(expression)

    @ew.add_match(
        ExtendedProjection(
            FullOuterNaturalJoin(ExtendedProjection, ...),
            ...
        ),
        lambda expression: any(
           _function_application_list_member_has_constant_exp(int_proj)
           for int_proj in expression.relation.relation_left.projection_list
        )
    )
    def nested_extended_projection_outernaturaljoin_constant(self, expression):
        return self._nested_ep_join_constant_left(expression)

    @ew.add_match(
        ExtendedProjection(
            NaturalJoin(..., ExtendedProjection),
            ...
        ),
        lambda expression: any(
            _function_application_list_member_has_constant_exp(int_proj)
            for int_proj in expression.relation.relation_right.projection_list
        )
    )
    def nested_extended_projection_naturaljoin_constantr(self, expression):
        return self._nested_ep_join_constant_left(expression, flip=True)

    @ew.add_match(
        ExtendedProjection(
            LeftNaturalJoin(..., ExtendedProjection),
            ...
        ),
        lambda expression: any(
           _function_application_list_member_has_constant_exp(int_proj)
           for int_proj in expression.relation.relation_right.projection_list
        )
    )
    def nested_extended_projection_leftnaturaljoin_constantr(self, expression):
        return self._nested_ep_join_constant_left(expression, flip=True)

    @ew.add_match(
        ExtendedProjection(
            FullOuterNaturalJoin(..., ExtendedProjection),
            ...
        ),
        lambda expression: any(
           _function_application_list_member_has_constant_exp(int_proj)
           for int_proj in expression.relation.relation_right.projection_list
        )
    )
    def nested_extended_projection_outernaturaljoin_constantr(self, expression):
        return self._nested_ep_join_constant_left(expression, flip=True)

    def _nested_ep_join_constant_left(self, expression, flip=False):
        if flip:
            right, left = expression.relation.unapply()
        else:
            left, right = expression.relation.unapply()
        new_inner_projections = []
        replacements = {}
        selections = []
        right_columns = right.columns()
        for proj in left.projection_list:
            if (
                isinstance(proj.fun_exp, Constant) and
                not isinstance(proj.fun_exp, Constant[Column])
            ):
                replacements[proj.dst_column] = proj.fun_exp
                if proj.dst_column in right_columns:
                    selections.append(eq_(proj.dst_column, proj.fun_exp))
            else:
                new_inner_projections.append(proj)

        rew = ReplaceColumns(replacements)
        new_external_projections = tuple(
            FunctionApplicationListMember(rew.walk(p.fun_exp), p.dst_column)
            for p in expression.projection_list
        )
        for selection in selections:
            right = Selection(right, selection)

        left = ExtendedProjection(left.relation, tuple(new_inner_projections))

        if flip:
            new_join = expression.relation.apply(right, left)
        else:
            new_join = expression.relation.apply(left, right)

        return self.walk(
            ExtendedProjection(
                new_join,
                new_external_projections
            )
        )

    @ew.add_match(
        NaturalJoin(ExtendedProjection, ...),
        lambda expression: (
            set(
                proj.dst_column
                for proj in expression.relation_left.projection_list
                if not isinstance(proj.fun_exp, Constant[Column])
            ) - expression.relation_right.columns()
        )
    )
    def push_computed_columns_up(self, expression, flip=False):
        if flip:
            right, left = expression.unapply()
        else:
            left, right = expression.unapply()

        right_columns = right.columns()
        projections_to_push_up = []
        projections_to_keep = []
        new_columns = {
            proj.dst_column: proj.dst_column
            for proj in left.projection_list
            if proj.dst_column == proj.fun_exp
        }
        for proj in left.projection_list:
            if (
                proj.dst_column not in right_columns and
                not isinstance(proj.fun_exp, Constant[Column])
            ):
                fun_columns = [
                    c
                    for _, c in ew.expression_iterator(proj.fun_exp)
                    if isinstance(c, Constant[Column])
                ]
                for c in fun_columns:
                    if c not in new_columns:
                        new_columns[c] = str2columnstr_constant(
                            Symbol.fresh().name
                        )

                projections_to_push_up.append(FunctionApplicationListMember(
                    ReplaceColumns(new_columns).walk(proj.fun_exp),
                    proj.dst_column
                ))
                projections_to_keep += [
                    FunctionApplicationListMember(c, new_columns[c])
                    for _, c in ew.expression_iterator(proj.fun_exp)
                    if isinstance(c, Constant[Column])
                ]
            else:
                projections_to_keep.append(proj)

        if len(projections_to_keep) > 0:
            left = ExtendedProjection(
                left.relation,
                tuple(projections_to_keep)
            )
        if flip:
            new_join = expression.apply(right, left)
        else:
            new_join = expression.apply(left, right)

        res = ExtendedProjection(
            new_join,
            tuple(
                FunctionApplicationListMember(c, c)
                for c in new_join.columns()
            ) + tuple(projections_to_push_up)
        )
        return self.walk(res)

    @ew.add_match(
        LeftNaturalJoin(ExtendedProjection, ...),
        lambda expression: (
            set(
                proj.dst_column
                for proj in expression.relation_left.projection_list
                if not isinstance(proj.fun_exp, Constant[Column])
            ) - expression.relation_right.columns()
        )
    )
    def push_computed_columns_up_left(self, expression):
        return self.push_computed_columns_up(expression)

    @ew.add_match(
        FullOuterNaturalJoin(ExtendedProjection, ...),
        lambda expression: (
            set(
                proj.dst_column
                for proj in expression.relation_left.projection_list
                if not isinstance(proj.fun_exp, Constant[Column])
            ) - expression.relation_right.columns()
        )
    )
    def push_computed_columns_up_outer(self, expression):
        return self.push_computed_columns_up(expression)

    @ew.add_match(
        NaturalJoin(..., ExtendedProjection),
        lambda expression: (
            set(
                proj.dst_column
                for proj in expression.relation_right.projection_list
                if not isinstance(proj.fun_exp, Constant[Column])
            ) - expression.relation_left.columns()
        )
    )
    def push_computed_columns_up_flip(self, expression):
        return self.push_computed_columns_up(expression, True)

    @ew.add_match(
        GroupByAggregation(
            ExtendedProjection, ...,
            (
                FunctionApplicationListMember(
                    FunctionApplication(Constant(sum), ...),
                    ...
                ),
            )
        ),
        lambda expression: any(
            (
                proj.dst_column ==
                expression.aggregate_functions[0].fun_exp.args[0]
            ) &
            (proj.fun_exp == Constant[float](1))
            for proj in expression.relation.projection_list
        )
    )
    def replace_trivial_agg_groupby(self, expression):
        new_projections = tuple(
            proj for proj in expression.relation.projection_list
            if (
                proj.dst_column !=
                expression.aggregate_functions[0].fun_exp.args[0]
            )
        )
        return self.walk(
            GroupByAggregation(
                ExtendedProjection(
                    expression.relation.relation,
                    new_projections
                ),
                expression.groupby,
                (
                    FunctionApplicationListMember(
                        COUNT(),
                        expression.aggregate_functions[0].dst_column
                    ),
                )

            )
        )

    @ew.add_match(
        ReplaceNull(ExtendedProjection, ..., ...),
        lambda exp: all(
            isinstance(proj.fun_exp, Constant[ColumnStr])
            for proj in exp.relation.projection_list
            if exp.column == proj.dst_column
        )
    )
    def push_replace_null_in_ext_proj(self, expression):
        for proj in expression.relation.projection_list:
            if proj.dst_column == expression.column:
                replacement = proj.fun_exp
                break

        return self.walk(ExtendedProjection(
            ReplaceNull(
                expression.relation.relation,
                replacement,
                expression.value
            ),
            expression.relation.projection_list
        ))


def _function_application_list_member_has_constant_exp(int_proj):
    return (
        isinstance(int_proj.fun_exp, Constant) and
        not isinstance(
            int_proj.fun_exp,
            (Constant[Column], Constant[RelationalAlgebraStringExpression])
        )
    )


class PushProjectionThroughProduct(ew.PatternWalker):
    """
    Push Projection through Product.

    Push Projection through Product so that the GOO pass can see
    individual Product operands without a Projection barrier.

    π[cols](A × B) → π[cols∩A](A) × π[cols∩B](B)
    """

    @ew.add_match(Projection, lambda e: isinstance(e.relation, Product))
    def push_through_product(self, expression):
        return Product(tuple(
            self.walk(_project_onto_operand(rel, expression.attributes))
            for rel in expression.relation.relations
        ))


def _shared_columns(col_sets):
    """Return, for each column set, the columns it shares with any other."""
    shared = [set() for _ in col_sets]
    for i, j in permutations(range(len(col_sets)), 2):
        shared[i] |= col_sets[i] & col_sets[j]
    return shared


def _project_if_narrower(operand, operand_cols, needed):
    """Project ``operand`` on ``needed`` if it drops any of its columns."""
    if needed and needed != operand_cols:
        return Projection(operand, tuple(needed))
    return operand


class PushProjectionsDown(ew.PatternWalker):
    """
    Push projections through NaturalJoins.

    Push projections through NaturalJoins to reduce the columns flowing
    into each join operand, minimising intermediate data size.

    π[cols](A ⋈ B) → π[cols](π[cols_A](A) ⋈ π[cols_B](B))

    where:
      cols_A = (cols ∩ A.cols) ∪ (A.cols ∩ B.cols)
      cols_B = (cols ∩ B.cols) ∪ (A.cols ∩ B.cols)

    Each child retains only the columns it needs for the join (shared
    columns) and for the final output (projected columns).  This is
    safe because NaturalJoin only needs shared column names to match
    rows — any column not in the projection or the join condition is
    dropped from the child early, reducing memory and computation.

    Only fires when at least one child can drop a column.
    """

    @ew.add_match(Projection(NaturalJoin, ...))
    def push_through_join(self, expression):
        result = self._push_through_pair(expression, NaturalJoin)
        if result is not expression:
            return result
        # Fall through to walk children so inner patterns
        # (NaturalJoin(Projection, …), NaturalJoin(NaturalJoin, …)) are
        # reachable — without this, the matched projection acts as a
        # barrier that stops the walker from descending.
        join = expression.relation
        left = self.walk(join.relation_left)
        right = self.walk(join.relation_right)
        if left is not join.relation_left or right is not join.relation_right:
            return Projection(NaturalJoin(left, right), expression.attributes)
        return expression

    @ew.add_match(Projection(NaryNaturalJoin, ...))
    def push_through_nary_join(self, expression):
        cols = expression.attributes
        cols_set = OrderedSet(cols)
        operands = expression.relation.relations
        col_sets = [_ra_columns(op) for op in operands]
        new_operands = tuple(
            _project_if_narrower(
                op, op_cols, (cols_set & op_cols) | (shared & op_cols)
            )
            for op, op_cols, shared in zip(
                operands, col_sets, _shared_columns(col_sets)
            )
        )
        if all(a is b for a, b in zip(new_operands, operands)):
            # Walk operands so inner patterns are reachable
            new_operands = tuple(self.walk(op) for op in operands)
            if all(a is b for a, b in zip(new_operands, operands)):
                return expression
        return Projection(NaryNaturalJoin(new_operands), cols)

    def _push_through_pair(self, expression, join_cls):
        cols = expression.attributes
        join = expression.relation
        A = join.relation_left
        B = join.relation_right

        cols_set = OrderedSet(cols)
        A_cols = _ra_columns(A)
        B_cols = _ra_columns(B)

        # Columns each child must provide:
        #   (a) columns needed in the projection output
        #   (b) columns shared with the other side (required for the ⋈)
        A_needed = (cols_set & A_cols) | (A_cols & B_cols)
        B_needed = (cols_set & B_cols) | (A_cols & B_cols)

        new_A = _project_if_narrower(A, A_cols, A_needed)
        new_B = _project_if_narrower(B, B_cols, B_needed)
        # Return the expression itself when neither side is narrowed (for
        # instance when a side needs no column at all): rebuilding an
        # equal expression makes the walker rewrite it forever.
        if new_A is A and new_B is B:
            return expression

        return Projection(NaturalJoin(new_A, new_B), cols)
