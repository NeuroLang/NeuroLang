from operator import eq
from typing import AbstractSet, Tuple

from . import expression_walker as ew
from .exceptions import NeuroLangException
from .expressions import Constant, Definition, FunctionApplication, Symbol
from .utils import NamedRelationalAlgebraFrozenSet, RelationalAlgebraSet

eq_ = Constant(eq)


class Column:
    pass


class ColumnInt(int, Column):
    pass


class ColumnStr(str, Column):
    pass


C_ = Constant
S_ = Symbol
FA_ = FunctionApplication


class RelationalAlgebraOperation(Definition):
    pass


class Selection(RelationalAlgebraOperation):
    def __init__(self, relation, formula):
        self.relation = relation
        self.formula = formula

    def __repr__(self):
        return f'\N{GREEK SMALL LETTER SIGMA}_{self.formula}({self.relation})'


class Projection(RelationalAlgebraOperation):
    def __init__(self, relation, attributes):
        self.relation = relation
        self.attributes = attributes

    def __repr__(self):
        return (
            f'\N{GREEK CAPITAL LETTER PI}'
            f'_{self.attributes}({self.relation})'
        )


class EquiJoin(RelationalAlgebraOperation):
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
            f'\N{JOIN}\N{SUBSCRIPT EQUALS SIGN}_{self.columns_left}'
            f'={self.columns_right}{self.relation_right}]'
        )


class NaturalJoin(RelationalAlgebraOperation):
    def __init__(
        self, relation_left, relation_right
    ):
        self.relation_left = relation_left
        self.relation_right = relation_right

    def __repr__(self):
        return (
            f'[{self.relation_left}'
            f'\N{JOIN}'
            f'{self.relation_right}]'
        )


class Product(RelationalAlgebraOperation):
    def __init__(self, relations):
        self.relations = tuple(relations)

    def __repr__(self):
        return '[' + f'\N{n-ary times operator}'.join(
            repr(r) for r in self.relations
        ) + ']'


class Difference(RelationalAlgebraOperation):
    def __init__(
        self, relation_left, relation_right
    ):
        self.relation_left = relation_left
        self.relation_right = relation_right

    def __repr__(self):
        return (
            f'[{self.relation_left}'
            f'-'
            f'{self.relation_right}]'
        )


class NameColumns(RelationalAlgebraOperation):
    def __init__(self, relation, column_names):
        self.relation = relation
        self.column_names = column_names

    def __repr__(self):
        return (
            f'\N{GREEK SMALL LETTER DELTA}'
            f'_{self.column_names}({self.relation})'
        )


class RenameColumn(RelationalAlgebraOperation):
    def __init__(self, relation, src, dst):
        self.relation = relation
        self.src = src
        self.dst = dst

    def __repr__(self):
        return (
            f'\N{GREEK SMALL LETTER DELTA}'
            f'_({self.src}\N{RIGHTWARDS ARROW}{self.dst})'
            f'({self.relation})'
        )


class RelationalAlgebraSolver(ew.ExpressionWalker):
    """
    Mixing that walks through relational algebra expressions and
    executes the operations.

    Relations are expected to be represented
    as objects with the same interface as :obj:`RelationalAlgebraSet`.
    """

    def __init__(
        self, symbol_table=None,
        relational_algebra_set_class=RelationalAlgebraSet,
        named_relational_algebra_set_class=NamedRelationalAlgebraFrozenSet
    ):
        self.symbol_table = symbol_table
        self.relational_algebra_set_class = relational_algebra_set_class
        self.named_relational_algebra_set_class =\
            named_relational_algebra_set_class

    @ew.add_match(Selection(..., FA_(eq_, (C_[Column], C_[Column]))))
    def selection_between_columns(self, selection):
        col1, col2 = selection.formula.args
        selected_relation = self.walk(selection.relation)\
            .value.selection_columns({col1.value: col2.value})

        return self._build_relation_constant(selected_relation)

    def _build_relation_constant(self, relation):
        if relation.arity > 0 and len(relation) > 0:
            if hasattr(relation, 'row_type'):
                row_type = relation.row_type
            else:
                row_type = Tuple[tuple(
                    type(arg) for arg in next(iter(relation._container))
                )]

            relation_type = AbstractSet[row_type]
        else:
            relation_type = AbstractSet[Tuple]

        return C_[relation_type](
            relation, verify_type=False
        )

    @ew.add_match(Selection(..., FA_(eq_, (C_[Column], ...))))
    def selection_by_constant(self, selection):
        col, val = selection.formula.args
        selection_relation = self.walk(selection.relation).value
        selected_relation = selection_relation.selection({col.value: val.value})

        return self._build_relation_constant(selected_relation)

    @ew.add_match(Projection)
    def ra_projection(self, projection):
        relation = self.walk(projection.relation)
        cols = tuple(v.value for v in projection.attributes)
        projected_relation = relation.value.projection(*cols)
        return self._build_relation_constant(
            projected_relation
        )

    @ew.add_match(Product)
    def ra_product(self, product):
        if len(product.relations) == 0:
            return C_[AbstractSet](self.relational_algebra_set_class(set()))

        res = self.walk(product.relations[0]).value
        for relation in product.relations[1:]:
            res = res.cross_product(self.walk(relation).value)
        return self._build_relation_constant(res)

    @ew.add_match(EquiJoin)
    def ra_equijoin(self, equijoin):
        left = self.walk(equijoin.relation_left).value
        columns_left = (c.value for c in equijoin.columns_left)
        right = self.walk(equijoin.relation_right).value
        columns_right = (c.value for c in equijoin.columns_right)
        res = left.equijoin(right, list(zip(columns_left, columns_right)))

        return self._build_relation_constant(res)

    @ew.add_match(NaturalJoin)
    def ra_naturaljoin(self, naturaljoin):
        left = self.walk(naturaljoin.relation_left).value
        right = self.walk(naturaljoin.relation_right).value
        res = left.naturaljoin(right)
        return self._build_relation_constant(res)

    @ew.add_match(Difference)
    def ra_difference(self, difference):
        left = self.walk(difference.relation_left).value
        right = self.walk(difference.relation_right).value
        res = left - right
        return self._build_relation_constant(res)

    @ew.add_match(NameColumns)
    def ra_name_columns(self, name_columns):
        relation = self.walk(name_columns.relation)
        relation_set = relation.value
        unwrapped_relation_set = relation_set.unwrap()

        column_names = []
        if not relation_set.is_null():
            for col in name_columns.column_names:
                if isinstance(col, Symbol):
                    column_names.append(col.name)
                elif isinstance(col, Constant):
                    column_names.append(col.value)
                else:
                    raise NeuroLangException(
                        "Column name must be a Constant or Symbol"
                    )
        new_set = self.named_relational_algebra_set_class(
            column_names,
            unwrapped_relation_set
        )
        return self._build_relation_constant(new_set)

    @ew.add_match(RenameColumn)
    def ra_rename_column(self, rename_column):
        relation = self.walk(rename_column.relation)
        src = rename_column.src.name
        dst = rename_column.dst.name
        new_set = relation.value

        if len(new_set) > 0:
            new_set = new_set.rename_column(src, dst)
        return self._build_relation_constant(new_set)

    @ew.add_match(Constant)
    def ra_constant(self, constant):
        return constant

    @ew.add_match(Symbol)
    def ra_symbol(self, symbol):
        try:
            constant = self.symbol_table[symbol]
        except KeyError:
            raise NeuroLangException(f'Symbol {symbol} not in table')
        return constant


class RelationalAlgebraSimplification(ew.ExpressionWalker):
    @ew.add_match(
        Product,
        lambda x: len(x.relations) == 1
    )
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
        Selection(..., FA_(eq_, (..., C_[Column]))),
        lambda s: not issubclass(s.formula.args[0].type, Column)
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
        Selection(Product, FA_(eq_, (C_[ColumnInt], ...))),
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
        if issubclass(arg_right.type, Column):
            arg_right = C_[ColumnInt](
                ColumnInt(-accum_arity + arg_right.value)
            )

        res = Product(
            left_relations +
            (Selection(
                Product(relations),
                eq_(C_[ColumnInt](ColumnInt(column)), arg_right)
            ),)
        )
        return self.walk(res)

    @ew.add_match(
        Selection(Product, FA_(eq_, (C_[ColumnInt], ...))),
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
        inner_relations = tuple(relations[:i + 1])
        if len(inner_relations) == 1:
            inner_relations = inner_relations[0]
        else:
            inner_relations = Product(inner_relations)

        outer_relations = tuple(relations[i + 1:])

        arg_left = selection.formula.args[0]

        res = Product(
            (Selection(
                inner_relations,
                eq_(arg_left, C_[ColumnInt](ColumnInt(column)))
            ),) + outer_relations
        )
        return self.walk(res)

    @ew.add_match(Selection(
        EquiJoin,
        FA_(eq_, (C_[ColumnInt], C_[ColumnInt])))
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
                C_[ColumnInt](ColumnInt(column_min)),
                C_[ColumnInt](ColumnInt(column_max))
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

    @ew.add_match(Selection(EquiJoin, FA_(eq_, (C_[ColumnInt], ...))))
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
                    C_(ColumnInt(
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
        Selection(Product, FA_(eq_, (C_[ColumnInt], C_[ColumnInt])))
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
            column_right = (C_[ColumnInt](ColumnInt(column_right)),)

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

    @ew.add_match(Selection(Product, FA_(eq_, (C_[ColumnInt], ...))))
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
    RelationalAlgebraSimplification,
    ew.ExpressionWalker
):
    """
    Mixing that optimises through relational algebra expressions by
    rewriting.
    equi-selection/product compositions into equijoins.
    """
    pass
