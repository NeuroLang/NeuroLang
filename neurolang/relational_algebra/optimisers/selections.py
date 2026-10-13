"""Optimisers rewriting and pushing selections."""

import operator

from ... import expression_walker as ew
from ...expressions import Constant, FunctionApplication
from ..relational_algebra import (
    Column,
    ColumnInt,
    ColumnStr,
    EquiJoin,
    ExtendedProjection,
    FullOuterNaturalJoin,
    FunctionApplicationListMember,
    GroupByAggregation,
    LeftNaturalJoin,
    NameColumns,
    NaturalJoin,
    Product,
    Projection,
    RenameColumn,
    ReplaceNull,
    Selection,
    eq_,
    get_expression_columns,
    int2columnint_constant
)
from .columns import _ra_columns


AND = Constant(operator.and_)
EQ = Constant(operator.eq)


class RewriteSelections(ew.ExpressionWalker):
    """
    Mixing that optimises through relational algebra expressions.

    The pushing selections (:obj:`Selection`) down and compositions
    and reduces the reach of selections. Then it converts
    equi-selection/product compositions into equijoins.
    """

    @ew.add_match(
        Selection(..., FunctionApplication(eq_, (..., Constant[Column]))),
        lambda s: not issubclass(s.formula.args[0].type, Column),
    )
    def swap_formula_args(self, selection):
        new_selection = Selection(
            self.walk(selection.relation), eq_(*selection.formula.args[::-1])
        )
        return self.walk(new_selection)

    @ew.add_match(
        Selection(
            ..., FunctionApplication(eq_, (Constant[Column], Constant[Column]))
        ),
        lambda s: s.formula.args[0].value > s.formula.args[1].value,
    )
    def sort_formula_args(self, selection):
        new_selection = Selection(
            self.walk(selection.relation), eq_(*selection.formula.args[::-1])
        )
        return self.walk(new_selection)

    @ew.add_match(
        Selection(
            Selection(..., FunctionApplication(eq_, (Constant[Column], ...))),
            FunctionApplication(eq_, (Constant[Column], ...)),
        ),
        lambda s: s.formula.args[0].value > s.relation.formula.args[0].value,
    )
    def selection_selection_swap(self, selection):
        new_selection = Selection(
            Selection(selection.relation.relation, selection.formula),
            selection.relation.formula,
        )
        return self.walk(new_selection)

    @ew.add_match(
        Selection(
            Product, FunctionApplication(eq_, (Constant[ColumnInt], ...))
        ),
        lambda s: (
            s.formula.args[0].value
            >= RewriteSelections.get_arity(
                s.relation.relations[0]
            )
        ),
    )
    def selection_push_right(self, selection):
        relations = selection.relation.relations
        column = int(selection.formula.args[0].value)

        i, accum_arity, column = self.split_relations_column(relations, column)

        left_relations = relations[:i]
        relations = relations[i:]

        arg_right = selection.formula.args[1]
        if issubclass(arg_right.type, Column):
            arg_right = Constant[ColumnInt](
                ColumnInt(-accum_arity + arg_right.value)
            )

        res = Product(
            left_relations
            + (
                Selection(
                    Product(relations),
                    eq_(Constant[ColumnInt](ColumnInt(column)), arg_right),
                ),
            )
        )
        return self.walk(res)

    @ew.add_match(
        Selection(
            Product, FunctionApplication(eq_, (Constant[ColumnInt], ...))
        ),
        lambda s: (
            s.formula.args[1].value
            < (
                RewriteSelections.get_arity(s.relation)
                - RewriteSelections.get_arity(
                    s.relation.relations[-1]
                )
            )
        ),
    )
    def selection_shorten_right(self, selection):
        relations = selection.relation.relations
        column = int(selection.formula.args[1].value)

        i, accum_arity, column = self.split_relations_column(relations, column)
        column += accum_arity
        inner_relations = tuple(relations[: i + 1])
        if len(inner_relations) == 1:
            inner_relations = inner_relations[0]
        else:
            inner_relations = Product(inner_relations)

        outer_relations = tuple(relations[i + 1:])

        arg_left = selection.formula.args[0]

        res = Product(
            (
                Selection(
                    inner_relations,
                    eq_(arg_left, Constant[ColumnInt](ColumnInt(column))),
                ),
            )
            + outer_relations
        )
        return self.walk(res)

    @ew.add_match(
        Selection(
            EquiJoin,
            FunctionApplication(
                eq_, (Constant[ColumnInt], Constant[ColumnInt])
            ),
        )
    )
    def selection_on_equijoin_columns(self, selection):
        column_left = selection.formula.args[0].value
        column_right = selection.formula.args[1].value
        column_min, column_max = sorted((column_left, column_right))
        relation_left = selection.relation.relation_left
        relation_right = selection.relation.relation_right
        left_arity = self.get_arity(relation_left)
        if column_max < left_arity:
            relation_left = Selection(relation_left, selection.formula)
        elif column_min >= left_arity:
            column_min -= left_arity
            column_max -= left_arity
            new_formula = eq_(
                Constant[ColumnInt](ColumnInt(column_min)),
                Constant[ColumnInt](ColumnInt(column_max)),
            )
            relation_right = Selection(relation_right, new_formula)
        else:
            return selection

        return self.walk(
            EquiJoin(
                relation_left,
                selection.relation.columns_left,
                relation_right,
                selection.relation.columns_right,
            )
        )

    @ew.add_match(
        Selection(
            EquiJoin, FunctionApplication(eq_, (Constant[ColumnInt], ...))
        )
    )
    def selection_on_equijoin(self, selection):
        column = selection.formula.args[0].value
        relation_left = selection.relation.relation_left
        relation_right = selection.relation.relation_right
        left_arity = self.get_arity(relation_left)
        if column < left_arity:
            relation_left = Selection(relation_left, selection.formula)
        else:
            relation_right = Selection(
                relation_right,
                eq_(
                    Constant(
                        ColumnInt(
                            int(selection.formula.args[0].value) - left_arity
                        )
                    ),
                    selection.formula.args[1],
                ),
            )

        return self.walk(
            EquiJoin(
                relation_left,
                selection.relation.columns_left,
                relation_right,
                selection.relation.columns_right,
            )
        )

    @ew.add_match(
        Selection(
            Product,
            FunctionApplication(
                eq_, (Constant[ColumnInt], Constant[ColumnInt])
            ),
        )
    )
    def selection_between_columns_product(self, selection):
        relations = selection.relation.relations
        if len(relations) == 1:
            res = Selection(relations[0], selection.formula)
        else:
            column_left = (selection.formula.args[0],)

            column_right = int(selection.formula.args[1].value)
            left_arity = self.get_arity(relations[0])
            column_right -= left_arity
            column_right = (Constant[ColumnInt](ColumnInt(column_right)),)

            relations_right = relations[1:]
            if len(relations_right) == 1:
                relation_right = relations_right[0]
            else:
                relation_right = Product(relations_right)

            res = EquiJoin(
                relations[0], column_left, relation_right, column_right
            )

        return self.walk(res)

    @ew.add_match(
        Selection(
            Product, FunctionApplication(eq_, (Constant[ColumnInt], ...))
        )
    )
    def selection_by_constant_on_product(self, selection):
        return self.walk(
            Product(
                (
                    Selection(
                        selection.relation.relations[0], selection.formula
                    ),
                )
                + selection.relation.relations[1:]
            )
        )

    @ew.add_match(
       Selection(Selection, ...),
       lambda exp: all(
           isinstance(col, Constant[ColumnStr])
           for col in (
               get_expression_columns(exp.formula) |
               get_expression_columns(exp.relation.formula)
           )
       )
    )
    def merge_selections(self, expression):
        return self.walk(
            Selection(
                expression.relation.relation,
                AND(expression.relation.formula, expression.formula)
            )
        )

    @staticmethod
    def split_relations_column(relations, column):
        accum_arity = 0
        for i, relation in enumerate(relations):
            current_arity = RewriteSelections.get_arity(
                relation
            )
            if column < current_arity:
                break
            accum_arity += current_arity
            column -= current_arity
        return i, accum_arity, column

    @staticmethod
    def get_arity(expression):
        if isinstance(expression, Constant):
            return expression.value.arity
        elif isinstance(expression, Product):
            return sum(
                RewriteSelections.get_arity(r)
                for r in expression.relations
            )
        elif isinstance(expression, EquiJoin):
            return RewriteSelections.get_arity(
                expression.relation_left
            ) + RewriteSelections.get_arity(
                expression.relation_right
            )
        else:
            return RewriteSelections.get_arity(
                expression.relation
            )


def _formula_misses_relation(formula, relation):
    """
    Check that a selection formula uses no column of a relation.

    The columns of ``relation`` are the ones it mentions and, when known,
    the ones it outputs: a constant relation mentions no column.
    """
    columns = get_expression_columns(relation)
    try:
        columns = columns | _ra_columns(relation)
    except NotImplementedError:
        pass
    return not (get_expression_columns(formula) & columns)


class PushInSelections(ew.PatternWalker):
    @ew.add_match(
        Selection(NaturalJoin, ...),
        lambda exp: _formula_misses_relation(
            exp.formula, exp.relation.relation_right
        )
    )
    def push_selection_in_left(self, expression):
        return self.walk(
            NaturalJoin(
                Selection(
                    expression.relation.relation_left,
                    expression.formula
                ),
                expression.relation.relation_right
            )
        )

    @ew.add_match(
        Selection(NaturalJoin, ...),
        lambda exp: _formula_misses_relation(
            exp.formula, exp.relation.relation_left
        )
    )
    def push_selection_in_right(self, expression):
        return self.walk(
            NaturalJoin(
                expression.relation.relation_left,
                Selection(
                    expression.relation.relation_right,
                    expression.formula
                )
            )
        )

    @ew.add_match(
        Selection(
            NaturalJoin,
            FunctionApplication(EQ, (Constant[ColumnStr], Constant[ColumnStr]))
        ),
        lambda exp: (
            (
                get_expression_columns(exp.formula) &
                get_expression_columns(exp.relation.relation_left)
            ) and (
                get_expression_columns(exp.formula) &
                get_expression_columns(exp.relation.relation_right)
            )
        )
    )
    def push_and_rename_in_naturaljoin(self, expression):
        column_left = (
            get_expression_columns(expression.formula) &
            (
                get_expression_columns(expression.relation.relation_left) -
                get_expression_columns(expression.relation.relation_right)
            )
        ).pop()
        column_right = (
            get_expression_columns(expression.formula) &
            get_expression_columns(expression.relation.relation_right)
        ).pop()

        if column_left != column_right:
            relation_right = RenameColumn(
                expression.relation.relation_right,
                column_right, column_left
            )
            projections = tuple(
                FunctionApplicationListMember(c, c)
                for c in expression.columns()
                if c != column_right
            )
            projections += (
                FunctionApplicationListMember(column_left, column_right),
            )
            expression = ExtendedProjection(
                NaturalJoin(
                    expression.relation.relation_left,
                    relation_right
                ),
                projections
            )
        return expression

    @ew.add_match(
        Selection(LeftNaturalJoin, ...),
        lambda exp: (
            len(
                get_expression_columns(exp.formula) &
                get_expression_columns(exp.relation.relation_right)
            ) == 0
        )
    )
    def push_selection_in_leftnaturaljoin_left(self, expression):
        return self.walk(
            LeftNaturalJoin(
                Selection(
                    expression.relation.relation_left,
                    expression.formula
                ),
                expression.relation.relation_right
            )
        )

    @ew.add_match(
        Selection(LeftNaturalJoin, ...),
        lambda exp: (
            len(
                get_expression_columns(exp.formula) &
                get_expression_columns(exp.relation.relation_left)
            ) == 0
        )
    )
    def push_selection_in_leftnaturaljoin_right(self, expression):
        return self.walk(
            LeftNaturalJoin(
                expression.relation.relation_left,
                Selection(
                    expression.relation.relation_right,
                    expression.formula
                )
            )
        )

    @ew.add_match(
        Selection(FullOuterNaturalJoin, ...),
        lambda exp: (
            len(
                get_expression_columns(exp.formula) &
                get_expression_columns(exp.relation.relation_right)
            ) == 0
        )
    )
    def push_selection_in_outernaturaljoin_left(self, expression):
        return self.walk(
            FullOuterNaturalJoin(
                Selection(
                    expression.relation.relation_left,
                    expression.formula
                ),
                expression.relation.relation_right
            )
        )

    @ew.add_match(
        Selection(FullOuterNaturalJoin, ...),
        lambda exp: (
            len(
                get_expression_columns(exp.formula) &
                get_expression_columns(exp.relation.relation_left)
            ) == 0
        )
    )
    def push_selection_in_outernaturaljoin_right(self, expression):
        return self.walk(
            FullOuterNaturalJoin(
                expression.relation.relation_left,
                Selection(
                    expression.relation.relation_right,
                    expression.formula
                )
            )
        )

    @ew.add_match(
        Selection(Projection, ...),
        lambda exp: (
            all(
                isinstance(column, Constant[ColumnStr])
                for column in get_expression_columns(exp.formula)
            ) and
            get_expression_columns(exp.formula) <=
            set(exp.relation.attributes)
        )
    )
    def push_selection_in_projection(self, expression):
        return self.walk(Projection(
            Selection(expression.relation.relation, expression.formula),
            expression.relation.attributes
        ))

    @ew.add_match(
        Selection(ExtendedProjection, ...),
        lambda exp: (
            get_expression_columns(exp.formula) <=
            {
                projection.dst_column
                for projection in exp.relation.projection_list
                if isinstance(projection.fun_exp, Constant[Column])
            }
        )
    )
    def push_selection_in_extended_projection(self, expression):
        rew = ew.ReplaceExpressionWalker({
            projection.dst_column: projection.fun_exp
            for projection in expression.relation.projection_list
            if isinstance(projection.fun_exp, Constant[Column])
        })
        new_formula = rew.walk(expression.formula)
        return self.walk(ExtendedProjection(
            Selection(expression.relation.relation, new_formula),
            expression.relation.projection_list
        ))

    @ew.add_match(
        Selection(GroupByAggregation, ...),
        lambda exp: (
            get_expression_columns(exp.formula) <=
            set(exp.relation.groupby)
        )
    )
    def push_selection_in_groupby(self, expression):
        return self.walk(
            GroupByAggregation(
                Selection(
                    expression.relation.relation,
                    expression.formula
                ),
                expression.relation.groupby,
                expression.relation.aggregate_functions
            )
        )

    @ew.add_match(
        Selection(ReplaceNull, ...),
        lambda exp: (
            exp.relation.column not in get_expression_columns(exp.formula)
        )
    )
    def push_selection_in_replace_null(self, expression):
        return ReplaceNull(
            Selection(
                expression.relation.relation,
                expression.formula
            ),
            expression.relation.column,
            expression.relation.value
        )


class PushUnnamedSelectionsUp(ew.PatternWalker):

    @staticmethod
    def _extract_column_int_constants(expression):
        return {
            e for _, e in ew.expression_iterator(expression)
            if isinstance(e, Constant[ColumnInt])
        }

    @ew.add_match(
        Projection(
            Selection(..., eq_(Constant[ColumnInt], Constant[ColumnInt])),
            ...
        ),
        lambda expression: (
            max(expression.relation.formula.args, key=lambda x: x.value)
            in set(expression.attributes)
        ) and (
            min(expression.relation.formula.args, key=lambda x: x.value)
            not in set(expression.attributes)
        )
    )
    def standardize_projected_column(self, expression):
        rew = ew.ReplaceExpressionWalker(
            {
                max(expression.relation.formula.args, key=lambda x: x.value):
                min(expression.relation.formula.args, key=lambda x: x.value)
            }
        )

        return Projection(
            expression.relation,
            rew.walk(expression.attributes)
        )

    @ew.add_match(
        Selection(Selection, eq_(Constant[ColumnInt], Constant[ColumnInt])),
        lambda expression: not (
            isinstance(expression.relation.formula, FunctionApplication) and
            expression.relation.formula.functor == eq_ and
            isinstance(
                expression.relation.formula.args[0],
                Constant[ColumnInt]
            ) and
            isinstance(
                expression.relation.formula.args[1],
                Constant[ColumnInt]
            )
        )
    )
    def invert_selection_order(self, expression):
        rew = ew.ReplaceExpressionWalker(
            {
                max(expression.formula.args, key=lambda x: x.value):
                min(expression.formula.args, key=lambda x: x.value)
            }
        )
        return Selection(
            Selection(
                expression.relation.relation,
                expression.formula
            ),
            rew.walk(expression.relation.formula)
        )

    @ew.add_match(
        Projection(Selection, ...),
        lambda expression: (
            (
                PushUnnamedSelectionsUp
                ._extract_column_int_constants(expression.relation.formula)
            ) and (
                PushUnnamedSelectionsUp
                ._extract_column_int_constants(expression.relation.formula)
                <= set(expression.attributes)
            )
        )
    )
    def push_selection_above_projection(self, expression):
        return Selection(
            Projection(
                expression.relation.relation,
                expression.attributes
            ),
            expression.relation.formula
        )

    @ew.add_match(
        NameColumns(Selection, ...),
        lambda expression: (
            PushUnnamedSelectionsUp
            ._extract_column_int_constants(expression.relation.formula)
        )
    )
    def push_selection_above_name_columns(self, expression):
        rew = ew.ReplaceExpressionWalker({
            int2columnint_constant(i): name
            for i, name in enumerate(expression.column_names)
        })
        new_formula = rew.walk(expression.relation.formula)
        return Selection(
            NameColumns(
                expression.relation.relation,
                expression.column_names
            ),
            new_formula
        )
