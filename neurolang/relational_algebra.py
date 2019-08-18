from operator import eq
from typing import AbstractSet, Tuple

from . import expressions
from . import solver_datalog_naive as sdb
from . import expression_walker as ew

eq_ = expressions.Constant(eq)


class Column(int):
    pass


C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = sdb.Implication
Fact_ = sdb.Fact
Eb_ = expressions.ExpressionBlock
FA_ = expressions.FunctionApplication


class Selection(expressions.Definition):
    def __init__(self, relation, formula):
        self.relation = relation
        self.formula = formula

    def __repr__(self):
        return f'\N{GREEK SMALL LETTER SIGMA}_{self.formula}({self.relation})'


class Projection(expressions.Definition):
    def __init__(self, relation, attributes):
        self.relation = relation
        self.attributes = attributes

    def __repr__(self):
        return (
            f'\N{GREEK CAPITAL LETTER PI}'
            f'_{self.attributes}({self.relation})'
        )


class EquiJoin(expressions.Definition):
    def __init__(
        self, relation_left, columns_left, relation_right, columns_right
    ):
        self.relation_left = relation_left
        self.columns_left = columns_left
        self.relation_right = relation_right
        self.columns_right = columns_right

    def __repr__(self):
        return (
            f'{self.relation_left}'
            f'\N{JOIN}_{self.columns_left}'
            f'={self.columns_right}{self.relation_right}'
        )


class Product(expressions.Definition):
    def __init__(self, relations):
        self.relations = relations

    def __repr__(self):
        return f'\N{n-ary times operator}'.join(
            repr(r) for r in self.relations
        )


class RelationAlgebraSolver(ew.ExpressionWalker):
    @ew.add_match(C_)
    def constant(self, constant):
        return constant

    @ew.add_match(Projection)
    def projection(self, projection):
        relation = self.walk(projection.relation)
        cols = tuple(v.value for v in projection.attributes)
        return C_[AbstractSet](relation.value.projection(*cols))

    @ew.add_match(Product)
    def product(self, product):
        res = self.walk(product.relations[0]).value
        for relation in product.relations[1:]:
            res = res.cross_product(self.walk(relation).value)
        return C_[AbstractSet](res)

    @ew.add_match(Selection(..., FA_(eq_, (C_[Column], C_[Column]))))
    def selection_between_columns(self, selection):
        col1, col2 = selection.formula.args
        return C_[AbstractSet](
            self.walk(selection.relation
                      ).value.selection_columns({col1.value: col2.value})
        )

    @ew.add_match(Selection(..., FA_(eq_, (C_[Column], ...))))
    def selection_by_constant(self, selection):
        col, val = selection.formula.args
        return C_[AbstractSet](
            self.walk(selection.relation
                      ).value.selection({col.value: val.value})
        )

    @ew.add_match(EquiJoin)
    def equijoin(self, equijoin):
        left = self.walk(equijoin.relation_left).value
        columns_left = (c.value for c in equijoin.columns_left.value)
        right = self.walk(equijoin.relation_right).value
        columns_right = (c.value for c in equijoin.columns_right.value)
        res = left.equijoin(right, list(zip(columns_left, columns_right)))
        return C_[AbstractSet](res)


class RelationAlgebraRewriteOptimiser(ew.ExpressionWalker):
    @ew.add_match(Projection)
    def projection(self, projection):
        return Projection(
            self.walk(projection.relation),
            projection.attributes
        )

    @ew.add_match(Selection(Product, FA_(eq_, (C_[Column], C_[Column]))))
    def selection_between_columns_product(self, selection):
        column_left = int(selection.formula.args[0].value)
        column_right = int(selection.formula.args[1].value)
        relations = selection.relation.relations
        column_min = min(column_left, column_right)
        column_max = max(column_left, column_right)

        accum_arity = 0

        for i, relation in enumerate(relations):
            arity_relation = self.get_arity(relation)
            accum_arity += arity_relation
            if column_min < accum_arity:
                if column_right >= accum_arity:
                    relations_left = relations[:i + 1]
                    relations_right = relations[i + 1:]
                    res = self.walk(
                        EquiJoin(
                            Product(relations_left),
                            C_[Tuple[Column]](
                                (C_[Column](Column(column_min)),)
                            ),
                            Product(relations_right),
                            C_[Tuple[Column]](
                                (C_[Column](Column(column_max - accum_arity)),)
                            )
                        )
                    )
                else:
                    relations_left = relations[:i]
                    relations_right = relations[i + 1:]
                    arity = accum_arity - arity_relation
                    column_min -= arity
                    column_max -= arity
                    relation_c = Selection(
                        relation,
                        eq_(
                            C_[Column](Column(column_min)),
                            C_[Column](Column(column_max)),
                        )
                    )

                    res = self.walk(
                        Product(
                            relations_left + (relation_c, ) + relations_right
                        )
                    )
                break
        else:
            res = Selection(
                Product(tuple(self.walk(r) for r in selection.relations)),
                selection.formula
            )
        return res

    @ew.add_match(Selection(Product, FA_(eq_, (C_[Column], ...))))
    def selection_by_constant_on_product(self, selection):
        new_relations = tuple()
        accum_arity = 0
        column = selection.formula.args[0].value
        for i, relation in enumerate(selection.relation.relations):
            accum_arity += self.get_arity(relation)
            if column < accum_arity:
                relation = Selection(relation, selection.formula)
                new_relations += (relation, )
                break
            new_relations += (relation, )

        new_relations += selection.relation.relations[i + 1:]
        return self.walk(Product(new_relations))

    @ew.add_match(Selection(EquiJoin, FA_(eq_, (C_[Column], ...))))
    def selection_on_equijoin(self, selection):
        column = selection.formula.args[0].value
        relation_left = selection.relation.relation_left
        relation_right = selection.relation.relation_right
        if column < self.get_arity(relation_left):
            relation_left = Selection(relation_left, selection.formula)
        else:
            relation_right = Selection(relation_right, selection.formula)

        return self.walk(
            EquiJoin(
                relation_left, selection.relation.columns_left, relation_right,
                selection.relation.columns_right
            )
        )

    @ew.add_match(Selection(C_[AbstractSet], ...))
    def selection_on_relation(self, selection):
        return selection

    @ew.add_match(...)
    def any_other_case(self, expression):
        return expression

    @staticmethod
    def get_arity(expression):
        if isinstance(expression, C_):
            return expression.value.arity
        elif isinstance(expression, Product):
            return sum(
                RelationAlgebraRewriteOptimiser.get_arity(r)
                for r in expression.relations
            )
        else:
            return RelationAlgebraRewriteOptimiser.get_arity(
                expression.relation
            )
