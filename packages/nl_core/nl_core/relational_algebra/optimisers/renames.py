"""Optimisers simplifying and pushing column renamings."""

from itertools import chain

from nl_expr import expression_walker as ew
from nl_expr.expressions import Constant
from ..relational_algebra import (
    ColumnStr,
    ExtendedProjection,
    FullOuterNaturalJoin,
    FunctionApplicationListMember,
    GroupByAggregation,
    LeftNaturalJoin,
    NameColumns,
    NaturalJoin,
    NumberColumns,
    Projection,
    RelationalAlgebraOperation,
    RenameColumn,
    RenameColumns,
    ReplaceNull,
    Selection,
    get_expression_columns
)
from .columns import _ra_columns


def _rename_past_projection_is_safe(expression):
    """
    Guard: True if a renaming can be pushed below a projection.

    Below the projection, a column renamed into one the projection drops
    would be duplicated. Relations of unknown columns are not checked.
    """
    try:
        columns = _ra_columns(expression.relation.relation)
    except NotImplementedError:
        return True
    sources = {src for src, _ in expression.renames}
    return all(
        dst not in columns or dst in sources
        for _, dst in expression.renames
    )


class RenameOptimizations(ew.PatternWalker):
    @ew.add_match(RenameColumn)
    def convert_rename_column(self, expression):
        return self.walk(RenameColumns(
            expression.relation,
            ((expression.src, expression.dst),)
        ))

    @ew.add_match(
        RenameColumns,
        lambda exp: (
            all(s == d for s, d in exp.renames)
        )
    )
    def remove_trivial_rename(self, expression):
        return self.walk(expression.relation)

    @ew.add_match(RenameColumns(RenameColumns, ...))
    def merge_nested_rename_columns(self, expression):
        outer_renames = {src: dst for src, dst in expression.renames}
        new_renames = []
        for src, dst in expression.relation.renames:
            if dst in outer_renames:
                new_renames.append(
                    (src, outer_renames[dst])
                )
                del outer_renames[dst]
            else:
                new_renames.append((src, dst))
        new_renames += [
            (src, dst)
            for src, dst in outer_renames.items()
        ]

        return self.walk(
            RenameColumns(expression.relation.relation, tuple(new_renames))
        )

    @ew.add_match(RenameColumns(ExtendedProjection, ...))
    def merge_rename_columns_extended_projection(self, expression):
        renames = {src: dst for src, dst in expression.renames}
        projection_list = expression.relation.projection_list
        new_projection_list = []
        for falm in projection_list:
            if falm.dst_column in renames:
                falm = FunctionApplicationListMember(
                    falm.fun_exp,
                    renames[falm.dst_column]
                )
            new_projection_list.append(falm)
        return self.walk(ExtendedProjection(
            expression.relation.relation,
            tuple(new_projection_list)
        ))

    @ew.add_match(
        RenameColumns(GroupByAggregation, ...),
        lambda exp: len({src for src, _ in exp.renames} & {
            f.dst_column for f in exp.relation.aggregate_functions
        }) > 0
    )
    def merge_rename_column_group_by(self, expression):
        renames = {src: dst for src, dst in expression.renames}
        aggregate_functions = expression.relation.aggregate_functions
        new_aggregate_functions = []
        for falm in aggregate_functions:
            if falm.dst_column in renames:
                old_dst = falm.dst_column
                falm = FunctionApplicationListMember(
                    falm.fun_exp,
                    renames[falm.dst_column]
                )
                del renames[old_dst]
            new_aggregate_functions.append(falm)
        return self.walk(
            RenameColumns(
                GroupByAggregation(
                    expression.relation.relation,
                    expression.relation.groupby,
                    tuple(new_aggregate_functions)
                ),
                tuple((src, dst) for src, dst in renames.items())
            )
        )

    @ew.add_match(
        RenameColumns(GroupByAggregation, ...),
        lambda exp: set(chain(*exp.renames)).isdisjoint(
            chain(*(
                get_expression_columns(agg.fun_exp)
                for agg in exp.relation.aggregate_functions
            ))
        )
    )
    def push_rename_past_groupby(self, expression):
        renames = {src: dst for src, dst in expression.renames}
        new_groupby = tuple(
            renames.get(col, col) for col in expression.relation.groupby
        )

        return self.walk(
            GroupByAggregation(
                RenameColumns(
                    expression.relation.relation,
                    expression.renames
                ),
                new_groupby,
                expression.relation.aggregate_functions
            )
        )

    @ew.add_match(
        RenameColumns(Projection, ...), _rename_past_projection_is_safe
    )
    def push_rename_past_projection(self, expression):
        renames = {src: dst for src, dst in expression.renames}
        new_attributes = tuple(
            renames.get(col, col) for col in expression.relation.attributes
        )

        return self.walk(
            Projection(
                RenameColumns(
                    expression.relation.relation,
                    expression.renames
                ),
                new_attributes
            )
        )

    @ew.add_match(RenameColumns(Selection, ...))
    def push_rename_past_selection(self, expression):
        selection = expression.relation
        renames = {src: dst for src, dst in expression.renames}
        rsw = ew.ReplaceExpressionWalker(renames)
        new_formula = rsw.walk(selection.formula)
        return self.walk(
            Selection(
                RenameColumns(selection.relation, expression.renames),
                new_formula
            )
        )

    @ew.add_match(RenameColumns(NameColumns, ...))
    def simplify_renames_name(self, expression):
        renames = {src: dst for src, dst in expression.renames}
        new_names = []
        for name in expression.relation.column_names:
            name = renames.get(name, name)
            new_names.append(name)
        return self.walk(
            NameColumns(expression.relation.relation, tuple(new_names))
        )

    @ew.add_match(RenameColumns(NaturalJoin(RelationalAlgebraOperation, RelationalAlgebraOperation), ...))
    def split_rename_naturaljoin(self, expression):
        return self._rename_joinop(expression, NaturalJoin)

    @ew.add_match(RenameColumns(LeftNaturalJoin(RelationalAlgebraOperation, RelationalAlgebraOperation), ...))
    def split_rename_left_naturaljoin(self, expression):
        return self._rename_joinop(expression, LeftNaturalJoin)

    @ew.add_match(RenameColumns(FullOuterNaturalJoin(RelationalAlgebraOperation, RelationalAlgebraOperation), ...))
    def split_rename_outer_naturaljoin(self, expression):
        return self._rename_joinop(expression, FullOuterNaturalJoin)

    def _rename_joinop(self, expression, joinop):
        left_renames = []
        right_renames = []
        left_columns = expression.relation.relation_left.columns()
        right_columns = expression.relation.relation_right.columns()
        for src, dst in expression.renames:
            if src in left_columns:
                left_renames.append((src, dst))
            if src in right_columns:
                right_renames.append((src, dst))
        return self.walk(joinop(
            RenameColumns(
                expression.relation.relation_left,
                tuple(left_renames)
            ),
            RenameColumns(
                expression.relation.relation_right,
                tuple(right_renames)
            )
        ))

    @ew.add_match(RenameColumns(ReplaceNull, ...))
    def switch_rename_replace_null(self, expression):
        renames = {src: dst for src, dst in expression.renames}
        return self.walk(
            ReplaceNull(
                RenameColumns(
                    expression.relation.relation,
                    expression.renames
                ),
                renames.get(
                    expression.relation.column,
                    expression.relation.column
                ),
                expression.relation.value
            )
        )

    @ew.add_match(
        NameColumns(NumberColumns(RelationalAlgebraOperation, ...), ...),
        lambda expression: all(
            isinstance(column, Constant[ColumnStr])
            for column in expression.relation.relation.columns()
        )
    )
    def eliminate_trivial_number_columns(self, expression):
        column_renames = tuple(
            (src, dst)
            for src, dst in zip(
                expression.relation.column_names,
                expression.column_names
            )
        )

        return self.walk(
            RenameColumns(
                expression.relation.relation, column_renames
            )
        )

    @ew.add_match(
        Projection(NumberColumns(RelationalAlgebraOperation, ...), ...),
        lambda expression: all(
            isinstance(column, Constant[ColumnStr])
            for column in expression.relation.relation.columns()
        )
    )
    def eliminate_trivial_projection_number_columns(self, expression):
        attributes = tuple(
            expression.relation.relation.columns()[
                column.value
            ]
            for column in expression.relation.columns()
        )
        new_relation = Projection(
            expression.relation.relation,
            attributes
        )
        return self.walk(new_relation)

    @ew.add_match(
        Selection(NumberColumns(RelationalAlgebraOperation, ...), ...),
        lambda expression: all(
            isinstance(column, Constant[ColumnStr])
            for column in expression.relation.relation.columns()
        )
    )
    def eliminate_trivial_selection_number_columns(self, expression):
        replacements = {
            number: name for number, name in
            zip(
                expression.relation.columns(),
                expression.relation.column_names
            )
        }
        new_formula = (
            ew.ReplaceExpressionWalker(replacements)
            .walk(expression.formula)
        )
        new_relation = NumberColumns(
            Selection(expression.relation.relation, new_formula),
            expression.relation.column_names
        )

        return self.walk(new_relation)
