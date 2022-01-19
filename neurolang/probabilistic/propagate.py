import operator
from re import A, M
from tkinter import W

from ..expression_walker import (
    ExpressionWalker,
    PatternMatcher,
    PatternWalker,
    add_match
)
from ..expressions import Constant, Symbol
from ..relational_algebra import (
    BinaryRelationalAlgebraOperation,
    ColumnInt,
    ColumnStr,
    NameColumns,
    NaturalJoin,
    Product,
    Projection,
    RelationalAlgebraSolver,
    Selection,
    UnaryRelationalAlgebraOperation,
    int2columnint_constant,
    str2columnstr_constant
)
from ..relational_algebra_provenance import LiftedPlanProjection, WeightedNaturalJoin
from .probabilistic_ra_utils import (
    ProbabilisticChoiceSet,
    ProbabilisticFactSet,
)

EQ = Constant(operator.eq)


class PushUp(UnaryRelationalAlgebraOperation):
    def __init__(self, relation, push_up):
        self.relation = relation
        self.push_up = push_up


class PushUpWalker(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(
        PushUp(PushUp, ...),
        lambda expression: expression.push_up == expression.relation.push_up
    )
    def simplify_push_up(self, expression):
        return self.walk(expression.relation)

    @add_match(PushUp(..., EQ(Constant[ColumnStr], Constant[ColumnInt])))
    def sym_push_up(self, expression):
        args = expression.push_up.args
        return self.walk(
            PushUp(expression.relation, EQ(args[1], args[0]))
        )

    #@add_match(
    #    NameColumns(
    #        Projection(
    #            PushUp(..., EQ(Constant[ColumnInt], Constant[ColumnStr])),
    #            ...
    #        ),
    #        ...
    #    ),
    #    lambda expression: (
    #        (
    #            expression.relation.relation.push_up.args[1]
    #            not in set(expression.column_names)
    #        ) and (
    #            expression.relation.relation.push_up.args[0]
    #            not in set(expression.relation.attributes)
    #        )
    #    )
    #)
    def unnamed_case(self, expression):
        column_int, column_str = expression.relation.relation.push_up.args
        inner_relation = expression.relation.relation.relation

        new_expression = NameColumns(
            Projection(
                inner_relation,
                expression.relation.attributes + (column_int,)
            ),
            expression.column_names + (column_str,)
        )

        new_expression = PushUp(new_expression, column_str)

        return self.walk(new_expression)

    @add_match(
             Projection(
                PushUp(..., EQ(Constant[ColumnInt], Constant[ColumnStr])),
                ...
            )
    )
    def projection_unnamed(self, expression):
        push_up = self.walk(expression.relation)
        new_expression = PushUp(
            Projection(
                push_up.relation,
                expression.attributes + (push_up.push_up.args[0],)
            ),
            EQ(int2columnint_constant(len(expression.attributes)), push_up.push_up.args[1])
        )

        return self.walk(new_expression)

    @add_match(
            NameColumns(
                PushUp(..., EQ(Constant[ColumnInt], Constant[ColumnStr])),
                ...
            )
    )
    def namecolumns(self, expression):
        push_up = self.walk(expression.relation)
        new_expression = PushUp(
            NameColumns(
                push_up.relation,
                expression.column_names + (push_up.push_up.args[1],)
            ),
            push_up.push_up.args[1]
        )
        return self.walk(new_expression)

    @add_match(
        PushUp(
            PushUp(
                ...,
                EQ(Constant[ColumnInt], Constant[ColumnStr])
            ),
            EQ(Constant[ColumnInt], Constant[ColumnStr])
        ),
        lambda exp: (
            exp.push_up.args[0].value > exp.relation.push_up.args[0].value
        )
    )
    def swap_push_ups(self, expression):
        new_expression = PushUp(
            PushUp(
                expression.relation.relation,
                expression.push_up
            ),
            expression.relation.push_up
        )
        return self.walk(new_expression)

    @add_match(
        PushUp(PushUp, ...),
        lambda exp: exp.push_up == exp.relation.push_up
    )
    def simplify_push_ups(self, expression):
        return self.walk(expression.relation)

    @add_match(Projection(PushUp, ...))
    def projection(self, expression):
        relation = self.walk(expression.relation)
        attributes = expression.attributes
        if relation.push_up not in set(attributes):
            attributes = attributes + (relation.push_up,)

        return self.walk(PushUp(
            expression.apply(
                relation.relation,
                attributes
            ),
            relation.push_up
        ))

    @add_match(LiftedPlanProjection(PushUp, ...))
    def lifted_plan_projection(self, expression):
        return self.projection(expression)

    @add_match(ProbabilisticFactSet(PushUp, Constant[ColumnInt](ColumnInt(0))))
    def prob_fact_set(self, expression):
        push_up = expression.relation.push_up.args
        relation = expression.relation.relation
        if not isinstance(relation, (PushUp, Symbol)):
            new_symbol = Symbol.fresh()
            self.symbol_table[new_symbol] = relation
            relation = new_symbol
        return self.walk(PushUp(
            expression.apply(relation, expression.probability_column),
            EQ(int2columnint_constant(push_up[0].value - 1), push_up[1])
        ))

    @add_match(ProbabilisticChoiceSet(PushUp, Constant[ColumnInt](ColumnInt(0))))
    def prob_choice_set(self, expression):
        return self.prob_fact_set(expression)

    @add_match(
        BinaryRelationalAlgebraOperation,
        lambda expression: (
            isinstance(expression.relation_left, PushUp) or
            isinstance(expression.relation_right, PushUp)
        )
    )
    def binary_ra_operation(self, expression):
        pushes = []
        relations = []
        for relation in expression.unapply():
            if isinstance(relation, PushUp):
                pushes.append(relation.push_up)
                relations.append(relation.relation)
            else:
                relations.append(relation)
        new_expression = expression.apply(*relations)
        for push in set(pushes):
            new_expression = PushUp(new_expression, push)

        return self.walk(new_expression)

    @add_match(
        WeightedNaturalJoin,
        lambda exp: any(isinstance(r, PushUp) for r in exp.relation)
    )
    def weighted_natural_join(self, expression):
        pushes = []
        relations = []
        for relation in expression.relations:
            if isinstance(relation, PushUp):
                pushes.append(relation.push_up)
                relations.append(relation.relation)
        new_expression = WeightedNaturalJoin(relations, expression.weights)
        for push in pushes:
            new_expression = PushUp(new_expression, push)

        return self.walk(new_expression)


class ReplaceSelectionsByPushUps(ExpressionWalker):
    def __init__(self, replacements, symbol_table):
        self.replacements = replacements
        self.symbol_table = symbol_table

    @add_match(Selection(..., EQ(..., ...)))
    def replace_selection(self, expression):
        args = expression.formula.args
        if args[1] in self.replacements:
            new_arg = self.replacements[args[1]]
            expression = self.walk(
                PushUp(expression.relation, EQ(args[0], new_arg))
            )
        return expression

    @add_match(ProbabilisticFactSet(Symbol, ...))
    def prob_fact_set_symbol(self, expression):
        relation = expression.relation
        relation, new_relation = self._solve_walk_symbol(relation)
        if new_relation is not relation:
            return expression.apply(
                new_relation,
                expression.probability_column
            )
        else:
            return expression

    @add_match(ProbabilisticChoiceSet(Symbol, ...))
    def prob_choice_set_symbol(self, expression):
        return self.prob_fact_set_symbol(expression)

    @add_match(
        UnaryRelationalAlgebraOperation,
        lambda expression: isinstance(expression.relation, Symbol)
    )
    def replace_unary_symbol(self, expression):
        relation = expression.relation
        relation, new_relation = self._solve_walk_symbol(relation)
        if new_relation is not relation:
            return expression.apply(
                new_relation,
                expression.unapply()[1]
            )
        else:
            return expression

    def _solve_walk_symbol(self, relation):
        while isinstance(relation, Symbol):
            relation_candidate = self.symbol_table[relation]
            if isinstance(relation_candidate, Constant):
                break
            relation = relation_candidate
        new_relation = self.walk(relation)
        return relation, new_relation

    @add_match(
        BinaryRelationalAlgebraOperation,
        lambda expression: any(
            isinstance(relation, Symbol)
            for relation in expression.unapply()
        )
    )
    def replace_binary_symbol(self, expression):
        relations = expression.unapply()
        new_relations = []
        changed = False
        for relation in relations:
            relation, new_relation = self._solve_walk_symbol(relation)
            if new_relation is not relation:
                changed = True
                new_relations.append(relation)

        if changed:
            return self.walk(expression.apply(*new_relations))
        else:
            return expression


def remove_push_up_from_top(expression):
    while isinstance(expression, PushUp):
        expression = expression.relation
    return expression
