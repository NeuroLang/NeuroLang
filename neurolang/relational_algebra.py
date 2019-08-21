from operator import eq
from typing import AbstractSet

from .expressions import Constant, Symbol, FunctionApplication, Definition
from . import expression_walker as ew
from .utils import RelationalAlgebraSet

eq_ = Constant(eq)


class Column(int):
    pass


C_ = Constant
S_ = Symbol
FA_ = FunctionApplication


class Selection(Definition):
    def __init__(self, relation, formula):
        self.relation = relation
        self.formula = formula

    def __repr__(self):
        return f'\N{GREEK SMALL LETTER SIGMA}_{self.formula}({self.relation})'


class Projection(Definition):
    def __init__(self, relation, attributes):
        self.relation = relation
        self.attributes = attributes

    def __repr__(self):
        return (
            f'\N{GREEK CAPITAL LETTER PI}'
            f'_{self.attributes}({self.relation})'
        )


class EquiJoin(Definition):
    def __init__(
        self, relation_left, columns_left, relation_right, columns_right
    ):
        self.relation_left = relation_left
        self.columns_left = columns_left
        self.relation_right = relation_right
        self.columns_right = columns_right

    def __repr__(self):
        return (
            f'[{self.relation_left}'
            f'\N{JOIN}_{self.columns_left}'
            f'={self.columns_right}{self.relation_right}]'
        )


class Product(Definition):
    def __init__(self, relations):
        self.relations = relations

    def __repr__(self):
        return '[' + f'\N{n-ary times operator}'.join(
            repr(r) for r in self.relations
        ) + ']'


class RelationalAlgebraWalker(ew.PatternWalker):
    """
    Mixing that walks through relational algebra expressions.

    Columns referred in projections, selections, and joins are
    expected to be instances of :obj:`Column`.
    """

    @ew.add_match(Selection)
    def selection_on_relation(self, selection):
        relation = self.walk(selection.relation)
        if relation is selection.relation:
            return selection
        else:
            return self.walk(Selection(relation, selection.formula))

    @ew.add_match(Projection)
    def ra_projection(self, projection):
        return Projection(
            self.walk(projection.relation),
            projection.attributes
        )

    @ew.add_match(Product)
    def product(self, product):
        if len(product.relations) == 1:
            return product.relations[0]
        else:
            new_relations = []
            changed = False
            for relation in product.relations:
                new_relation = self.walk(relation)
                changed |= new_relation is not relation
                new_relations.append(new_relation)
            if changed:
                res = self.walk(Product(new_relations))
            else:
                res = product
            return res

    @ew.add_match(EquiJoin)
    def equijoin(self, equijoin):
        left = self.walk(equijoin.relation_left)
        right = self.walk(equijoin.relation_right)
        if (
            left is equijoin.relation_left and
            right is equijoin.relation_right
        ):
            return equijoin
        else:
            return self.walk(EquiJoin(
                left, equijoin.columns_left,
                right, equijoin.columns_right
            ))

    @ew.add_match(Constant)
    def constant(self, constant):
        return constant


class RelationalAlgebraSolver(RelationalAlgebraWalker):
    """
    Mixing that walks through relational algebra expressions and
    executes the operations.

    Relations are expected to be represented
    as objects with the same interface as :obj:`RelationalAlgebraSet`.
    """

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

    @ew.add_match(Projection)
    def ra_projection(self, projection):
        relation = self.walk(projection.relation)
        cols = tuple(v.value for v in projection.attributes)
        return C_[AbstractSet](relation.value.projection(*cols))

    @ew.add_match(Product)
    def ra_product(self, product):
        if len(product.relations) == 0:
            return C_[AbstractSet](RelationalAlgebraSet(set()))

        res = self.walk(product.relations[0]).value
        for relation in product.relations[1:]:
            res = res.cross_product(self.walk(relation).value)
        return C_[AbstractSet](res)

    @ew.add_match(EquiJoin)
    def ra_equijoin(self, equijoin):
        left = self.walk(equijoin.relation_left).value
        columns_left = (c.value for c in equijoin.columns_left)
        right = self.walk(equijoin.relation_right).value
        columns_right = (c.value for c in equijoin.columns_right)
        res = left.equijoin(right, list(zip(columns_left, columns_right)))

        return C_[AbstractSet](res)


class RelationalAlgebraRewriteSelections(ew.PatternWalker):
    """
    Mixing that optimises through relational algebra expressions.

    The pushing selections (:obj:`Selection`) down and compositions
    and reduces the reach of selections. Then it converts
    equi-selection/product compositions into equijoins.
    """
    @ew.add_match(
        Selection(..., FA_(eq_, (..., C_[Column]))),
        lambda s: s.formula.args[0].type is not Column
    )
    def swap_formula_args(self, selection):
        new_selection = Selection(
            self.walk(selection.relation),
            eq_(*selection.formula.args[::-1])
        )
        return self.walk(new_selection)

    @ew.add_match(
        Selection(..., FA_(eq_, (C_[Column], C_[Column]))),
        lambda s: s.formula.args[0].value > s.formula.args[1].value
    )
    def sort_formula_args(self, selection):
        new_selection = Selection(
            self.walk(selection.relation),
            eq_(*selection.formula.args[::-1])
        )
        return self.walk(new_selection)

    @ew.add_match(
        Selection(Selection(..., FA_(eq_, ...)), FA_(eq_, ...)),
        lambda s: s.formula.args[0].value > s.relation.formula.args[0].value
    )
    def selection_selection_swap(self, selection):
        new_selection = Selection(
            Selection(
                selection.relation.relation,
                selection.formula
            ),
            selection.relation.formula
        )
        return self.walk(new_selection)

    @ew.add_match(
        Selection(Product, FA_(eq_, (C_[Column], ...))),
        lambda s: (
            s.formula.args[0].value >=
            RelationalAlgebraRewriteSelections.
            get_arity(s.relation.relations[0])
        )
    )
    def selection_push_right(self, selection):
        relations = selection.relation.relations
        column = int(selection.formula.args[0].value)

        i, accum_arity, column = self.split_relations_column(relations, column)

        left_relations = relations[:i]
        relations = relations[i:]

        arg_right = selection.formula.args[1]
        if arg_right.type is Column:
            arg_right = C_[Column](Column(-accum_arity + arg_right.value))

        res = Product(
            left_relations +
            (Selection(
                Product(relations),
                eq_(C_[Column](Column(column)), arg_right)
            ),)
        )
        return self.walk(res)

    @ew.add_match(
        Selection(Product, FA_(eq_, (C_[Column], ...))),
        lambda s: (
            s.formula.args[1].value <
            (
                RelationalAlgebraRewriteSelections.get_arity(s.relation) -
                RelationalAlgebraRewriteSelections.
                get_arity(s.relation.relations[-1])
            )
        )
    )
    def selection_shorten_right(self, selection):
        relations = selection.relation.relations
        column = int(selection.formula.args[1].value)

        i, accum_arity, column = self.split_relations_column(relations, column)
        column += accum_arity
        inner_relations = relations[:i + 1]
        if len(inner_relations) == 1:
            inner_relations = inner_relations[0]
        else:
            inner_relations = Product(inner_relations)

        outer_relations = relations[i + 1:]

        arg_left = selection.formula.args[0]

        res = Product(
            (Selection(
                inner_relations,
                eq_(arg_left, C_[Column](Column(column)))
            ),) + outer_relations
        )
        return self.walk(res)

    @ew.add_match(Selection(EquiJoin, FA_(eq_, (C_[Column], C_[Column]))))
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
                C_[Column](Column(column_min)),
                C_[Column](Column(column_max))
            )
            relation_right = Selection(relation_right, new_formula)
        else:
            return selection

        return self.walk(
            EquiJoin(
                relation_left,
                selection.relation.columns_left, relation_right,
                selection.relation.columns_right
            )
        )

    @ew.add_match(Selection(EquiJoin, FA_(eq_, (C_[Column], ...))))
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
                    C_(Column(
                        int(selection.formula.args[0].value) -
                        left_arity
                    )),
                    selection.formula.args[1]
                )
            )

        return self.walk(
            EquiJoin(
                relation_left, selection.relation.columns_left,
                relation_right, selection.relation.columns_right
            )
        )

    @ew.add_match(
        Selection(Product, FA_(eq_, (C_[Column], C_[Column])))
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
            column_right = (C_[Column](Column(column_right)),)

            relations_right = relations[1:]
            if len(relations_right) == 1:
                relation_right = relations_right[0]
            else:
                relation_right = Product(relations_right)

            res = EquiJoin(
                relations[0], column_left,
                relation_right, column_right
            )

        return self.walk(res)

    @ew.add_match(Selection(Product, FA_(eq_, (C_[Column], ...))))
    def selection_by_constant_on_product(self, selection):
        return self.walk(Product(
            (Selection(selection.relation.relations[0], selection.formula),) +
            selection.relation.relations[1:]
        ))

    @staticmethod
    def split_relations_column(relations, column):
        accum_arity = 0
        for i, relation in enumerate(relations):
            current_arity = (
                RelationalAlgebraRewriteSelections.
                get_arity(relation)
            )
            if column < current_arity:
                break
            accum_arity += current_arity
            column -= current_arity
        return i, accum_arity, column

    @staticmethod
    def get_arity(expression):
        if isinstance(expression, C_):
            return expression.value.arity
        elif isinstance(expression, Product):
            return sum(
                RelationalAlgebraRewriteSelections.get_arity(r)
                for r in expression.relations
            )
        elif isinstance(expression, EquiJoin):
            return (
                RelationalAlgebraRewriteSelections.get_arity(
                    expression.relation_left
                ) +
                RelationalAlgebraRewriteSelections.get_arity(
                    expression.relation_right
                )
            )
        else:
            return RelationalAlgebraRewriteSelections.get_arity(
                expression.relation
            )


class RelationalAlgebraOptimiser(
    RelationalAlgebraRewriteSelections,
    RelationalAlgebraWalker,
):
    """
    Mixing that optimises through relational algebra expressions by
    rewriting.
    equi-selection/product compositions into equijoins.
    """
    pass
