import operator

from .. import expression_walker as ew
from ..expressions import Constant, FunctionApplication, Symbol
from .relational_algebra import (
    COUNT,
    Column,
    ColumnInt,
    ColumnStr,
    EquiJoin,
    ExtendedProjection,
    FunctionApplicationListMember,
    GroupByAggregation,
    LeftNaturalJoin,
    NameColumns,
    NaturalJoin,
    Product,
    Projection,
    RelationalAlgebraStringExpression,
    RenameColumn,
    RenameColumns,
    ReplaceNull,
    Selection,
    eq_,
    get_expression_columns,
    str2columnstr_constant
)


AND = Constant(operator.and_)


class RelationalAlgebraSimplification(ew.ExpressionWalker):
    @ew.add_match(Product, lambda x: len(x.relations) == 1)
    def single_product(self, product):
        return self.walk(product.relations[0])


class RelationalAlgebraRewriteSelections(ew.ExpressionWalker):
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
            Selection(..., FunctionApplication(eq_, ...)),
            FunctionApplication(eq_, ...),
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
            >= RelationalAlgebraRewriteSelections.get_arity(
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
                RelationalAlgebraRewriteSelections.get_arity(s.relation)
                - RelationalAlgebraRewriteSelections.get_arity(
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

    # @ew.add_match(
    #    Selection(Selection, ...),
    #    lambda exp: all(
    #        isinstance(col, Constant[ColumnStr])
    #        for col in (
    #            get_expression_columns(exp.formula) |
    #            get_expression_columns(exp.relation.formula)
    #        )
    #    )
    # )
    def merge_selections(self, expression):
        return self.walk(
            Selection(
                expression.relation.relation,
                AND(expression.formula, expression.relation.formula)
            )
        )

    @staticmethod
    def split_relations_column(relations, column):
        accum_arity = 0
        for i, relation in enumerate(relations):
            current_arity = RelationalAlgebraRewriteSelections.get_arity(
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
                RelationalAlgebraRewriteSelections.get_arity(r)
                for r in expression.relations
            )
        elif isinstance(expression, EquiJoin):
            return RelationalAlgebraRewriteSelections.get_arity(
                expression.relation_left
            ) + RelationalAlgebraRewriteSelections.get_arity(
                expression.relation_right
            )
        else:
            return RelationalAlgebraRewriteSelections.get_arity(
                expression.relation
            )


class EliminateTrivialProjections(ew.PatternWalker):
    @ew.add_match(Projection(Constant, ...))
    def eliminate_trivial_projection(self, expression):
        if (
            tuple(c.value for c in expression.attributes)
            == tuple(c for c in expression.relation.value.columns)
        ) or (
            tuple(str(c.value) for c in expression.attributes)
            == tuple(c for c in expression.relation.value.columns)
        ):
            return expression.relation
        else:
            return expression

    @ew.add_match(
        Projection(Projection, ...),
        lambda e: set(e.attributes) <= set(e.relation.attributes)
    )
    def eliminate_trivial_nested_projection(self, expression):
        return self.walk(
            Projection(expression.relation.relation, expression.attributes)
        )

    @ew.add_match(Projection(ExtendedProjection, ...))
    def try_simplify_projection_extended_projection(self, expression):
        new_relation = self.walk(expression.relation)
        if new_relation is not expression.relation:
            return self.walk(Projection(new_relation, expression.attributes))
        else:
            return expression

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
        rew = ew.ReplaceExpressionWalker({
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

        rew = ew.ReplaceExpressionWalker(replacements)
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
                    ew.ReplaceExpressionWalker(new_columns).walk(proj.fun_exp),
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
        LeftNaturalJoin(..., ExtendedProjection),
        lambda expression: (
            set(
                proj.dst_column
                for proj in expression.relation_right.projection_list
                if not isinstance(proj.fun_exp, Constant[Column])
            ) - expression.relation_left.columns()
        )
    )
    def push_computed_columns_up_flip_left(self, expression):
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


class RenameOptimizations(ew.PatternWalker):
    @ew.add_match(RenameColumn)
    def convert_rename_column(self, expression):
        return self.walk(RenameColumns(
            expression.relation,
            ((expression.src, expression.dst),)
        ))

    @ew.add_match(RenameColumns, lambda exp: len(exp.renames) == 0)
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
        lambda exp: {src for src, _ in exp.renames} <= set(
            exp.relation.groupby
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

    @ew.add_match(RenameColumns(Projection, ...))
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

    @ew.add_match(RenameColumns(NaturalJoin, ...))
    def split_rename_naturaljoin(self, expression):
        return self._rename_joinop(expression, NaturalJoin)

    @ew.add_match(RenameColumns(LeftNaturalJoin, ...))
    def split_rename_left_naturaljoin(self, expression):
        return self._rename_joinop(expression, LeftNaturalJoin)

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


class RelationalAlgebraPushInSelections(ew.PatternWalker):
    @ew.add_match(
        Selection(NaturalJoin, ...),
        lambda exp: (
            len(
                get_expression_columns(exp.formula) &
                get_expression_columns(exp.relation.relation_right)
            ) == 0
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
        lambda exp: (
            len(
                get_expression_columns(exp.formula) &
                get_expression_columns(exp.relation.relation_left)
            ) == 0
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
        Selection(Projection, ...),
        lambda exp: (
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


class RelationalAlgebraOptimiser(
    RelationalAlgebraRewriteSelections,
    RelationalAlgebraSimplification,
    EliminateTrivialProjections,
    RelationalAlgebraPushInSelections,
    RenameOptimizations,
    ew.ExpressionWalker,
):
    """
    Mixing that optimises through relational algebra expressions by
    rewriting.
    equi-selection/product compositions into equijoins.
    """

    pass
