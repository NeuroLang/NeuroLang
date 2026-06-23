import operator
from itertools import chain

from .. import expression_walker as ew
from ..expressions import Constant, FunctionApplication, Symbol
from ..utils import OrderedSet
from .relational_algebra import (
    COUNT,
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
    NaryNaturalJoin,
    NumberColumns,
    Product,
    Projection,
    RelationalAlgebraOperation,
    RelationalAlgebraStringExpression,
    RenameColumn,
    RenameColumns,
    ReplaceNull,
    Selection,
    eq_,
    get_expression_columns,
    int2columnint_constant,
    str2columnstr_constant
)

AND = Constant(operator.and_)
EQ = Constant(operator.eq)


class ProductSimplification(ew.ExpressionWalker):
    @ew.add_match(Product, lambda x: len(x.relations) == 1)
    def single_product(self, product):
        return self.walk(product.relations[0])


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


def _proj_attrs_as_str_set(expr):
    """Return projection attribute names as a set of plain strings."""
    return {str(a.name) if hasattr(a, 'name') else str(a) for a in expr.attributes}


def _ra_columns_as_str_set(expr):
    """Return RA operation column names as a set of plain strings."""
    return {str(c) for c in expr.columns()}


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
        lambda expression: (
            _proj_attrs_as_str_set(expression)
            == _ra_columns_as_str_set(expression.relation)
        )
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


class PushInSelections(ew.PatternWalker):
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


def _ra_columns(expr):
    """Safely get output columns from any RA node or leaf Constant."""
    if isinstance(expr, RelationalAlgebraOperation):
        return expr.columns()
    from .relational_algebra import _get_columns_for_RA_or_constant as _colsfb
    try:
        return _colsfb(expr)
    except (NotImplementedError, AttributeError):
        return set()


def _flatten_joins(expr):
    """Collect all join operands into a flat list.

    Recurse through both NaturalJoin (binary) and NaryNaturalJoin (n-ary)
    so that nested join trees are fully unwound in one pass.
    """
    if isinstance(expr, NaturalJoin):
        return _flatten_joins(expr.relation_left) + _flatten_joins(expr.relation_right)
    if isinstance(expr, NaryNaturalJoin):
        result = []
        for r in expr.relations:
            result.extend(_flatten_joins(r))
        return result
    return [expr]


def _sort_join_operands(operands):
    """
    Greedy sort to minimise cross products.

    Repeatedly pick the operand with the most column overlap with the
    cumulative column set of already-placed operands. This maximises
    shared columns at each join boundary, eliminating cross products
    unless the join graph is truly disconnected.
    """
    if len(operands) <= 2:
        return operands

    col_sets = [_ra_columns(op) for op in operands]
    n = len(operands)

    remaining = set(range(n))
    first = max(remaining, key=lambda i: sum(
        len(col_sets[i] & col_sets[j]) for j in remaining if j != i
    ))
    result = [operands[first]]
    remaining.remove(first)
    current_cols = set(col_sets[first])

    while remaining:
        best = max(remaining, key=lambda j: len(col_sets[j] & current_cols))
        result.append(operands[best])
        remaining.remove(best)
        current_cols |= col_sets[best]

    return result


def _flatten_join_or_product(expr):
    """
    Flatten any chain of NaturalJoin, NaryNaturalJoin, and Product,
    recursing through Projection wrappers that sit between join nodes.

    This lets a mixed tree such as
        NaturalJoin(Projection(Product(A, B)), C)
    be reordered globally when the operands share column names.

    Projections wrapping a join/product are pushed down to the individual
    operands so the GOO pass sees the full operand list.
    """
    if isinstance(expr, Projection):
        inner = _flatten_join_or_product(expr.relation)
        if len(inner) > 1:
            result = []
            for op in inner:
                op_cols = _ra_columns(op)
                pushed_attrs = tuple(
                    a for a in expr.attributes
                    if a in op_cols
                )
                if pushed_attrs and pushed_attrs != tuple(op_cols):
                    result.append(Projection(op, pushed_attrs))
                else:
                    result.append(op)
            return result
        return [expr]
    if isinstance(expr, NaturalJoin):
        return (
            _flatten_join_or_product(expr.relation_left)
            + _flatten_join_or_product(expr.relation_right)
        )
    if isinstance(expr, NaryNaturalJoin):
        result = []
        for r in expr.relations:
            result.extend(_flatten_join_or_product(r))
        return result
    if isinstance(expr, Product):
        result = []
        for r in expr.relations:
            result.extend(_flatten_join_or_product(r))
        return result
    return [expr]


def _group_operands_by_overlap(operands):
    """
    Greedy operator ordering (GOO) by pairwise column-name overlap.

    Operands are grouped into connected components based on shared column
    names. Within each connected component, operands are merged greedily:
    at each step the pair/sub-plan with the largest column intersection is
    joined, producing a left-deep chain whose order maximises adjacent
    overlap. The merge is represented by a new sub-plan (a tuple entry) so
    that the final component can be emitted as a single NaryNaturalJoin.

    The cost model uses only column names: two sub-plans overlap when their
    column sets intersect. This requires no cardinality estimates.
    """
    if len(operands) <= 1:
        return [operands]

    col_sets = [set(_ra_columns(op)) for op in operands]
    n = len(operands)

    # Build adjacency by shared columns, grouping into connected components.
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if col_sets[i] & col_sets[j]:
                union(i, j)

    groups = OrderedSet()
    for i in range(n):
        groups.add(find(i))
    groups = list(groups)

    component_map = {root: [] for root in groups}
    for i in range(n):
        component_map[find(i)].append(i)

    grouped = []
    for root in groups:
        idxs = component_map[root]
        if len(idxs) == 1:
            grouped.append(([operands[idxs[0]]], col_sets[idxs[0]]))
        else:
            grouped.append((
                _order_component_greedily(
                    [operands[i] for i in idxs],
                    [col_sets[i] for i in idxs],
                ),
                set().union(*[col_sets[i] for i in idxs]),
            ))

    return grouped


def _order_component_greedily(operands, col_sets):
    """
    Greedily order operands in one connected component.

    Builds a path through the overlap graph using nearest-neighbour
    expansion from a low-degree endpoint. This places bridge operands
    between otherwise disconnected sub-graphs, avoiding adjacent operands
    that share no columns.
    """
    if len(operands) <= 2:
        return list(operands)

    n = len(operands)
    remaining = set(range(n))

    total_overlap = {
        i: sum(len(col_sets[i] & col_sets[j]) for j in range(n) if j != i)
        for i in range(n)
    }
    first = min(remaining, key=lambda i: total_overlap[i])

    ordered = [operands[first]]
    remaining.remove(first)
    tail_cols = set(col_sets[first])

    while remaining:
        best = max(
            remaining,
            key=lambda j: len(col_sets[j] & tail_cols),
        )
        ordered.append(operands[best])
        remaining.remove(best)
        tail_cols = set(col_sets[best])

    return ordered


class FlattenAndReorderJoins(ew.PatternWalker):
    """
    Flatten NaturalJoins and NaryNaturalJoins into a flat
    NaryNaturalJoin, then greedily reorder to minimise cross products.

    1. **Flatten**: Collect all join operands by recursing through
       NaturalJoin and NaryNaturalJoin structure.
    2. **Sort**: Greedy nearest-neighbour ordering maximising column
       overlap at each join boundary (avoids cross products by keeping
       related tables adjacent).
    3. **Wrap**: Produce an NaryNaturalJoin holding all operands.
    """

    @ew.add_match(NaturalJoin)
    def flatten_and_reorder(self, expression):
        # Walk children first so PushProjectionsDown can process inner
        # Projection(NaturalJoin, ...) patterns before we flatten.
        left = self.walk(expression.relation_left)
        right = self.walk(expression.relation_right)
        walked = NaturalJoin(left, right)
        operands = _flatten_joins(walked)
        if len(operands) <= 1:
            return operands[0] if operands else walked
        sorted_ops = _sort_join_operands(operands)
        return NaryNaturalJoin(sorted_ops)

    @ew.add_match(NaryNaturalJoin)
    def reorder_nary(self, expression):
        operands = list(expression.relations)
        if len(operands) <= 1:
            return operands[0] if operands else expression
        flat_ops = []
        for op in operands:
            if isinstance(op, NaryNaturalJoin):
                flat_ops.extend(op.relations)
            else:
                flat_ops.append(op)
        sorted_ops = _sort_join_operands(flat_ops)
        if all(a is b for a, b in zip(flat_ops, sorted_ops)):
            # Walk operands so inner Projection patterns are reachable
            new_ops = tuple(self.walk(op) for op in flat_ops)
            if any(a is not b for a, b in zip(new_ops, flat_ops)):
                return NaryNaturalJoin(new_ops)
            return expression
        return NaryNaturalJoin(sorted_ops)


def _build_binary_join_tree(operands):
    """
    Build a left-deep binary NaturalJoin tree from a list of operands.
    """
    if len(operands) == 1:
        return operands[0]
    result = operands[0]
    for op in operands[1:]:
        result = NaturalJoin(result, op)
    return result


def _tree_contains_product(expr):
    """Check whether an RA expression tree contains a Product node.

    Does NOT recurse into Selection nodes — a Product inside a Selection
    is already handled by RewriteSelections (eq-selection/product → EquiJoin)
    and must not be reordered by GOO.
    """
    if isinstance(expr, Product):
        return True
    if isinstance(expr, Selection):
        return False
    if isinstance(expr, ew.Expression):
        for child in expr.unapply():
            if isinstance(child, ew.Expression) and _tree_contains_product(child):
                return True
    return False


def _has_eliminable_cross_product(operands):
    """
    Check whether a flat operand list has a cross product that can be
    eliminated by reordering: i.e., there exist at least two operands
    that share columns AND at least one pair that doesn't.

    If all operands share columns with each other, FlattenAndReorderJoins
    already handles it. If no operands share columns, reordering won't help.
    """
    col_sets = [set(_ra_columns(op)) for op in operands]
    n = len(operands)
    has_overlap = False
    has_disjoint = False
    for i in range(n):
        for j in range(i + 1, n):
            if col_sets[i] & col_sets[j]:
                has_overlap = True
            else:
                has_disjoint = True
    return has_overlap and has_disjoint


def _degenerate_natural_join(expression):
    """Guard: True if NaturalJoin(A, B) where A and B share no columns."""
    left_cols = _ra_columns(expression.relation_left)
    right_cols = _ra_columns(expression.relation_right)
    return bool(left_cols) and bool(right_cols) and not (left_cols & right_cols)


def _product_has_eliminable_cross_product(expression):
    """Guard: True if Product has operands with both overlapping and
    disjoint column sets — reordering can eliminate the cross product."""
    if not isinstance(expression, Product):
        return False
    operands = list(expression.relations)
    return _has_eliminable_cross_product(operands)


class DegenerateNaturalJoinToProduct(ew.PatternWalker):
    """
    Convert NaturalJoin(A, B) to Product(A, B) when A and B share no
    column names. A degenerate NaturalJoin is semantically a cross product;
    making it explicit lets the GOO pass detect and reorder it.
    """

    @ew.add_match(NaturalJoin, _degenerate_natural_join)
    def degenerate_join_to_product(self, expression):
        left = self.walk(expression.relation_left)
        right = self.walk(expression.relation_right)
        return Product((left, right))


class PushProjectionThroughProduct(ew.PatternWalker):
    """
    Push Projection through Product so that the GOO pass can see
    individual Product operands without a Projection barrier.

    π[cols](A × B) → π[cols∩A](A) × π[cols∩B](B)
    """

    @ew.add_match(Projection, lambda e: isinstance(e.relation, Product))
    def push_through_product(self, expression):
        product = expression.relation
        new_rels = []
        for rel in product.relations:
            rel_cols = _ra_columns(rel)
            pushed_attrs = tuple(a for a in expression.attributes if a in rel_cols)
            if pushed_attrs and pushed_attrs != tuple(rel_cols):
                new_rels.append(self.walk(Projection(rel, pushed_attrs)))
            else:
                new_rels.append(self.walk(rel))
        return Product(tuple(new_rels))

class GreedyJoinOrdering(ew.PatternWalker):
    """
    Flatten mixed Product/NaturalJoin chains and reorder by pairwise
    ColumnStr overlap using a greedy operator ordering (GOO).

    Only triggers when the expression is a NaturalJoin that contains a
    Product, or a Product whose operands have an eliminable cross product.
    This prevents infinite re-entry: once the cross product is eliminated,
    the guard no longer matches.

    Emits nested NaturalJoin trees (not NaryNaturalJoin) for connected
    groups and Product for disconnected groups.
    """

    @ew.add_match(NaturalJoin, _tree_contains_product)
    def reorder_join(self, expression):
        left = self.walk(expression.relation_left)
        right = self.walk(expression.relation_right)
        expression = NaturalJoin(left, right)

        operands = _flatten_join_or_product(expression)
        if len(operands) <= 1:
            return expression

        if not _has_eliminable_cross_product(operands):
            return expression

        return self._build_reordered(operands)

    @ew.add_match(Product, _product_has_eliminable_cross_product)
    def reorder_product(self, expression):
        new_rels = tuple(self.walk(r) for r in expression.relations)
        expression = Product(new_rels)
        operands = _flatten_join_or_product(expression)
        return self._build_reordered(operands)

    @ew.add_match(ew.Expression)
    def fallthrough(self, expression):
        """Recursively walk children for expressions that don't match
        the reorder patterns. This replaces the old walk() override."""
        args = expression.unapply()
        new_args = tuple()
        changed = False
        for arg in args:
            if isinstance(arg, ew.Expression):
                new_arg = self.walk(arg)
                changed |= new_arg is not arg
            elif isinstance(arg, tuple) and len(arg) > 0 and isinstance(arg[0], ew.Expression):
                new_arg = tuple(self.walk(sub) for sub in arg)
                changed |= any(a is not b for a, b in zip(new_arg, arg))
            else:
                new_arg = arg
            new_args += (new_arg,)
        if changed:
            return expression.apply(*new_args)
        return expression

    def _build_reordered(self, operands):
        grouped = _group_operands_by_overlap(operands)

        join_groups = []
        product_groups = []
        for group, _ in grouped:
            if len(group) > 1:
                join_groups.append(_build_binary_join_tree(group))
            else:
                product_groups.append(group[0])

        if not join_groups:
            return Product(tuple(product_groups))
        if not product_groups:
            if len(join_groups) == 1:
                return join_groups[0]
            return Product(tuple(join_groups))
        return Product(tuple(join_groups + product_groups))


class PushProjectionsDown(ew.PatternWalker):
    """
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
        nary = expression.relation
        cols_set = OrderedSet(cols)
        operands = list(nary.relations)

        col_sets = [_ra_columns(op) for op in operands]
        n = len(operands)

        shared_with_any = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    shared_with_any[i] |= col_sets[i] & col_sets[j]

        new_operands = []
        changed = False
        for i, op in enumerate(operands):
            needed = (cols_set & col_sets[i]) | (shared_with_any[i] & col_sets[i])
            if needed and needed != col_sets[i]:
                new_operands.append(Projection(op, tuple(needed)))
                changed = True
            else:
                new_operands.append(op)

        if not changed:
            # Walk operands so inner patterns are reachable
            new_ops = tuple(self.walk(op) for op in operands)
            if any(a is not b for a, b in zip(new_ops, operands)):
                return Projection(NaryNaturalJoin(new_ops), cols)
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

        if A_needed == A_cols and B_needed == B_cols:
            return expression

        if A_needed and A_needed != A_cols:
            A = Projection(A, tuple(A_needed))
        if B_needed and B_needed != B_cols:
            B = Projection(B, tuple(B_needed))

        return Projection(NaturalJoin(A, B), cols)


class RelationalAlgebraOptimiser(
    RewriteSelections,
    ProductSimplification,
    EliminateTrivialProjections,
    PushInSelections,
    RenameOptimizations,
    PushUnnamedSelectionsUp,
    PushProjectionsDown,
    FlattenAndReorderJoins,
    ew.ExpressionWalker,
):
    """
    Mixing that optimises through relational algebra expressions by
    rewriting.
    equi-selection/product compositions into equijoins.

    Pipeline order within each convergence pass:
      1. Existing optimisers (selection pushdown, projection elimination, …)
      2. PushProjectionsDown            — push projections through
         NaryNaturalJoin operands to reduce join sizes
      3. FlattenAndReorderJoins        — flatten join tree into
         NaryNaturalJoin, greedily reorder to minimise cross products
    """

    pass
