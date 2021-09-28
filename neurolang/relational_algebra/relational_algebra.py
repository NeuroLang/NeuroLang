import math
import operator
from typing import AbstractSet, Callable, Tuple

import numpy
import pandas.core.computation.ops

from .. import expression_walker as ew, type_system
from ..exceptions import ProjectionOverMissingColumnsError, NeuroLangException
from ..expression_pattern_matching import NeuroLangPatternMatchingNoMatch
from ..expressions import (
    Constant,
    Definition,
    Expression,
    FunctionApplication,
    Symbol,
    Unknown,
    sure_is_not_pattern
)
from ..utils import (
    NamedRelationalAlgebraFrozenSet,
    OrderedSet,
    RelationalAlgebraFrozenSet,
    RelationalAlgebraSet
)
from ..utils.relational_algebra_set import (
    RelationalAlgebraColumnInt,
    RelationalAlgebraColumnStr,
    RelationalAlgebraStringExpression
)

eq_ = Constant(operator.eq)
COUNT = Constant(len)


class Column:
    pass


class ColumnInt(int, Column):
    """Refer to a relational algebra set's column by its index."""


class ColumnStr(str, Column):
    """Refer to a named relational algebra set's column by its name."""


def str2columnstr_constant(name):
    return Constant[ColumnStr](
        ColumnStr(name), auto_infer_type=False, verify_type=False,
    )


def int2columnint_constant(col):
    return Constant[ColumnInt](
        ColumnInt(col), auto_infer_type=False, verify_type=False,
    )


def get_expression_columns(expression):
    columns = OrderedSet()
    args = list(expression.unapply())
    while args:
        arg = args.pop()
        if isinstance(arg, Constant[Column]):
            columns.add(arg)
        elif isinstance(arg, Constant):
            continue
        elif isinstance(arg, Expression):
            args += arg.unapply()
        elif isinstance(arg, tuple):
            args += list(arg)

    return columns


class RelationalAlgebraOperation(Definition):
    def __init__(self):
        self._columns = set()

    def columns(self):
        if not hasattr(self, '_columns'):
            raise NotImplementedError()
        return self._columns


def _get_columns_for_RA_or_constant(expression):
    if isinstance(expression, RelationalAlgebraOperation):
        return expression.columns()
    elif isinstance(expression, Constant):
        if isinstance(expression.value, NamedRelationalAlgebraFrozenSet):
            return OrderedSet(
                str2columnstr_constant(c)
                for c in expression.value.columns
            )
        elif isinstance(expression.value, RelationalAlgebraFrozenSet):
            return OrderedSet(
                int2columnint_constant(c)
                for c in expression.value.columns
            )
    raise NotImplementedError()


class NAryRelationalAlgebraOperation(RelationalAlgebraOperation):
    def columns(self):
        if not hasattr(self, '_columns'):
            self._columns = _get_columns_for_RA_or_constant(self.relations[0])
            for r in self.relations[1:]:
                self._columns = (
                    self._columns |
                    _get_columns_for_RA_or_constant(r)
                )
        return self._columns


class BinaryRelationalAlgebraOperation(RelationalAlgebraOperation):
    def columns(self):
        if not hasattr(self, '_columns'):
            self._columns = (
                _get_columns_for_RA_or_constant(self.relation_left) |
                _get_columns_for_RA_or_constant(self.relation_right)
            )
        return self._columns


class UnaryRelationalAlgebraOperation(RelationalAlgebraOperation):
    def columns(self):
        if not hasattr(self, '_columns'):
            self._columns = (
                _get_columns_for_RA_or_constant(self.relation)
            )
        return self._columns


class Selection(UnaryRelationalAlgebraOperation):
    def __init__(self, relation, formula):
        self.relation = relation
        self.formula = formula

    def __repr__(self):
        return f"\N{GREEK SMALL LETTER SIGMA}_{self.formula}({self.relation})"


class Projection(UnaryRelationalAlgebraOperation):
    def __init__(self, relation, attributes):
        self.relation = relation
        self.attributes = attributes
        self._columns = OrderedSet(self.attributes)

    def __repr__(self):
        return (
            f"\N{GREEK CAPITAL LETTER PI}"
            f"_{self.attributes}({self.relation})"
        )


class EquiJoin(BinaryRelationalAlgebraOperation):
    def __init__(
        self, relation_left, columns_left, relation_right, columns_right
    ):
        self.relation_left = relation_left
        self.columns_left = columns_left
        self.relation_right = relation_right
        self.columns_right = columns_right

    def __repr__(self):
        return (
            f"[{self.relation_left}"
            f"\N{JOIN}\N{SUBSCRIPT EQUALS SIGN}_{self.columns_left}"
            f"={self.columns_right}{self.relation_right}]"
        )


class NaturalJoin(BinaryRelationalAlgebraOperation):
    def __init__(self, relation_left, relation_right):
        self.relation_left = relation_left
        self.relation_right = relation_right

    def __repr__(self):
        return f"[{self.relation_left}" f"\N{JOIN}" f"{self.relation_right}]"


class LeftNaturalJoin(BinaryRelationalAlgebraOperation):
    def __init__(self, relation_left, relation_right):
        self.relation_left = relation_left
        self.relation_right = relation_right

    def __repr__(self):
        return (
            f"[{self.relation_left}"
            f"\N{LEFT OUTER JOIN}"
            f"{self.relation_right}]"
        )


class ReplaceNull(UnaryRelationalAlgebraOperation):
    def __init__(self, relation, column, value):
        self.relation = relation
        self.column = column
        self.value = value

    def __repr__(self):
        return (
            f"\N{GREEK SMALL LETTER KAPPA}_"
            f"[{self.column}, NULL={self.value}][{self.relation}]"
        )


class Product(NAryRelationalAlgebraOperation):
    def __init__(self, relations):
        self.relations = tuple(relations)

    def __repr__(self):
        return (
            "["
            + "\N{n-ary times operator}".join(repr(r) for r in self.relations)
            + "]"
        )


class Difference(BinaryRelationalAlgebraOperation):
    def __init__(self, relation_left, relation_right):
        self.relation_left = relation_left
        self.relation_right = relation_right

    def __repr__(self):
        return f"[{self.relation_left}" f"-" f"{self.relation_right}]"


class Union(BinaryRelationalAlgebraOperation):
    def __init__(self, relation_left, relation_right):
        self.relation_left = relation_left
        self.relation_right = relation_right

    def __repr__(self):
        return f"{self.relation_left} ∪ {self.relation_right}"


class Intersection(BinaryRelationalAlgebraOperation):
    def __init__(self, relation_left, relation_right):
        self.relation_left = relation_left
        self.relation_right = relation_right

    def __repr__(self):
        return f"{self.relation_left} & {self.relation_right}"


class NameColumns(UnaryRelationalAlgebraOperation):
    """
    Give names to the columns of a relational algebra set.

    All columns must be named at once. Each column name must either be a
    `Constant[ColumnStr]` or a `Symbol[ColumnStr]` pointing to a symbolic
    column name resolved when the expression is compiled.
    """

    def __init__(self, relation, column_names):
        self.relation = relation
        self.column_names = column_names

    def columns(self):
        return OrderedSet(self.column_names)

    def __repr__(self):
        return (
            f"\N{GREEK SMALL LETTER RHO}"
            f"_{self.column_names}({self.relation})"
        )


class RenameColumn(UnaryRelationalAlgebraOperation):
    def __init__(self, relation, src, dst):
        self.relation = relation
        self.src = src
        self.dst = dst

    def columns(self):
        columns = _get_columns_for_RA_or_constant(self.relation).copy()
        columns.replace(self.src, self.dst)
        return columns

    def __repr__(self):
        return (
            f"\N{GREEK SMALL LETTER RHO}"
            f"_({self.src}\N{RIGHTWARDS ARROW}{self.dst})"
            f"({self.relation})"
        )


class RenameColumns(UnaryRelationalAlgebraOperation):
    """
    Convenient operation for renaming multiple columns at the same time.

    Attributes
    ----------
    relation : NamedRelationalAlgebraFrozenSet
        The relation whose columns shall be renamed.
    renames : tuple of pairs of Constant[ColumnStr] or Symbol[ColumnStr]
        The renamings that should happen, represented as tuples (src, dst).

    """

    def __init__(self, relation, renames):
        self.relation = relation
        self.renames = renames

    def columns(self):
        columns = _get_columns_for_RA_or_constant(self.relation).copy()
        for rename in self.renames:
            if rename[0] in columns:
                columns.replace(rename[0], rename[1])
        return columns

    def __repr__(self):
        return (
            "\N{GREEK SMALL LETTER RHO}"
            + "_({})".format(
                ", ".join(
                    "{}\N{RIGHTWARDS ARROW}{}".format(src, dst)
                    for src, dst in self.renames
                )
            )
            + f"({self.relation})"
        )


class GroupByAggregation(RelationalAlgebraOperation):
    """
    General representation of a groupby operation with aggregate functions.

    Attributes
    ----------
    relation : Expression[AbstractSet]
        Relation on which the groupby is applied.
    groupby : `Tuple[Union[Constant[ColumnStr], Symbol[ColumnStr]]]`
        The list of columns on which to group.
    aggregate_functions : Tuple[FunctionApplicationListMember]
        List of aggregate functions to apply.

    """

    def __init__(self, relation, groupby, aggregate_functions):
        self.relation = relation
        self.groupby = groupby
        self.aggregate_functions = aggregate_functions

    def columns(self):
        return (
            OrderedSet(self.groupby) |
            OrderedSet([
                aggregate.dst_column
                for aggregate in self.aggregate_functions
            ])
        )

    def __repr__(self):
        join_str = ","
        return "γ_[{}, {}]({})".format(
            repr(self.groupby),
            join_str.join(
                [repr(member) for member in self.aggregate_functions]
            ),
            repr(self.relation),
        )


class ExtendedProjection(RelationalAlgebraOperation):
    """
    General operation defining string-based relational algebra projections
    allowing flexible computations on a relation's columns.

    Attributes
    ----------
    relation : Expression[AbstractSet]
        Relation on which the projections are applied.
    projection_list : Tuple[FunctionApplicationListMember]
        List of projections to apply.

    Notes
    -----
    The concept of extended projection is formally defined in section 5.2.5
    of [1]_.

    .. [1] Garcia-Molina, Hector, Jeffrey D. Ullman, and Jennifer Widom.
       "Database systems: the complete book." (2009).

    """

    def __init__(self, relation, projection_list):
        self.relation = relation
        self.projection_list = tuple(projection_list)

    def columns(self):
        return OrderedSet([
            projection.dst_column for projection in
            self.projection_list
        ])

    def __repr__(self):
        join_str = ","
        return "π_[{}]({})".format(
            join_str.join([repr(member) for member in self.projection_list]),
            repr(self.relation),
        )

    def __eq__(self, other):
        if not isinstance(other, ExtendedProjection):
            return False

        return (
            self.relation == other.relation and
            len(self.projection_list) == len(other.projection_list) and
            all(
                any(
                    element_self == element_other
                    for element_other in other.projection_list
                )
                for element_self in self.projection_list
            )
        )


class FunctionApplicationListMember(Definition):
    """
    Representation of a function application to a column. Can be used to
    represent either the application of an extended projection, or the
    application of an aggregate function, to a column.


    Attributes
    ----------
    fun_exp : `Union[Constant[str], FunctionApplication]`
        Constant string representation of an extended projection operation,
        or `FunctionApplication` representation of an aggregate function.
    dst_column : `Constant[ColumnStr]` or `Symbol[ColumnStr]`
        Constant column string of the destination column.

    Notes
    -----
    In the case of an extended projection operation, as described in [1]_,
    a function application list member can either be
        - a single attribute (column) name in the relation, resulting in a
          normal non-extended projection,
        - an expression `x -> y` where `x` and `y` are both attribute (column)
          names, `x` effectively being rename as `y`,
        - or an expression `E -> z` where `E` is an expression involving
          attributes of the relation, arithmetic operators, and string
          operators, and `z` is a new name for the attribute that results from
          the calculation implied by `E`. For example, `a + b -> x` represents
          the sum of the attributes `a` and `b`, renamed `x`.

    .. [1] Garcia-Molina, Hector, Jeffrey D. Ullman, and Jennifer Widom.
       "Database systems: the complete book." (2009).

    """

    def __init__(self, fun_exp, dst_column):
        self.fun_exp = fun_exp
        self.dst_column = dst_column

    def __repr__(self):
        return "{} -> {}".format(self.fun_exp, self.dst_column)


class Destroy(UnaryRelationalAlgebraOperation):
    """
    Operation to map a column of a collection of elements into
    a new column with all collections concatenated

    Attributes
    ----------
    relation : Expression[AbstractSet]
        Relation on which the projections are applied.
    src_column : Constant[ColumnStr]
        Column to destroy in the operation.

    dst_column : Constant[ColumnStr]
        Column to map onto the new set.

    Notes
    -----
    The concept of set destruction is formally defined in chapter 20
    of [1]_.

    .. [1] Abiteboul, S., Hull, R. & Vianu, V. "Foundations of databases". (
           Addison Wesley, 1995).
    """

    def __init__(self, relation, src_column, dst_column):
        self.relation = relation
        self.src_column = src_column
        self.dst_column = dst_column

    def __repr__(self):
        return (
            f"set_destroy[{self.relation}, "
            "{self.src_column} -> {src.dst_column}]"
        )


class ConcatenateConstantColumn(UnaryRelationalAlgebraOperation):
    """
    Add a column with a repeated constant value to a relation.

    Attributes
    ----------
    relation : Constant[RelationalAlgebraSet]
        Relation to which the column will be added.

    column_name : Constant[ColumnStr] or Symbol[ColumnStr]
        Name of the newly added column.

    column_value : Constant
        Constant value repeated in the new column.

    """

    def __init__(self, relation, column_name, column_value):
        self.relation = relation
        self.column_name = column_name
        self.column_value = column_value

    def columns(self):
        return (
            _get_columns_for_RA_or_constant(self.relation) |
            {self.column_name}
        )


OPERATOR_STRING = {
    operator.add: "+",
    operator.sub: "-",
    operator.mul: "*",
    operator.truediv: "/",
    operator.eq: "==",
    operator.gt: ">",
    operator.lt: "<",
    operator.ge: ">=",
    operator.le: "<=",
    operator.pow: "**",
    operator.and_: "and",
    operator.or_: "or"
}


def _get_evaluatable_operations_and_string_translations():
    """
    Get all operations that can be translated for fast evaluations in pandas
    `eval` expressions.

    The operations include some Python builtins, mathematical operations in
    the Python standard library's `math` module, and numpy operations.

    """
    eval_op_to_str = {
        max: "max",
        min: "min",
        sum: "sum",
        abs: "abs",
        numpy.max: "max",
        numpy.min: "min",
        numpy.mean: "mean",
    }
    math_module_equivalent_name = {
        "arcsin": "asin",
        "arccos": "acos",
        "arctan": "atan",
        "arcsinh": "asinh",
        "arccosh": "acosh",
        "arctanh": "atanh",
        "exp": "exp",
        "log": "log",
        "log10": "log10"
    }
    for op_name in pandas.core.computation.ops._unary_math_ops:
        eval_op_to_str[getattr(numpy, op_name)] = op_name
        if hasattr(math, op_name):
            eval_op_to_str[getattr(math, op_name)] = op_name
        elif op_name in math_module_equivalent_name:
            op = getattr(math, math_module_equivalent_name[op_name])
            eval_op_to_str[op] = op_name
    return eval_op_to_str


EVAL_OP_TO_STR = _get_evaluatable_operations_and_string_translations()


def is_translatable_operation(exp):
    return (
        isinstance(exp, FunctionApplication)
        and isinstance(exp.functor, Constant)
        and exp.functor.value in EVAL_OP_TO_STR
    )


def is_arithmetic_operation(exp):
    """
    Whether the expression is an arithmetic operation function application.

    Parameters
    ----------
    exp : Expression

    Returns
    -------
    bool

    """
    return (
        isinstance(exp, FunctionApplication)
        and isinstance(exp.functor, Constant)
        and exp.functor.value in OPERATOR_STRING
    )


class StringArithmeticWalker(ew.PatternWalker):
    """
    Walker translating an Expression with basic arithmetic operations on a
    relation's columns to its equivalent string representation.

    The expression can refer to the names a relation's columns or to the
    length of an other constant relation.

    """

    @ew.add_match(FunctionApplication, is_arithmetic_operation)
    def arithmetic_operation(self, fa):
        return Constant[RelationalAlgebraStringExpression](
            RelationalAlgebraStringExpression(
                "({} {} {})".format(
                    self.walk(fa.args[0]).value,
                    OPERATOR_STRING[fa.functor.value],
                    self.walk(fa.args[1]).value,
                ),
            ),
            auto_infer_type=False,
        )

    @ew.add_match(
        FunctionApplication(Constant, ...),
        is_translatable_operation,
    )
    def translatable_numpy_operation(self, fa):
        return Constant[RelationalAlgebraStringExpression](
            RelationalAlgebraStringExpression(
                "{}({})".format(
                    EVAL_OP_TO_STR[fa.functor.value],
                    self.walk(fa.args[0]).value,
                )
            ),
            auto_infer_type=False,
        )

    @ew.add_match(
        FunctionApplication(Constant, ...),
        lambda fa: (
            isinstance(fa.functor.value, Callable)
            and sum is fa.functor.value
        ),
    )
    def operation_sum(self, fa):
        return Constant[RelationalAlgebraStringExpression](
            RelationalAlgebraStringExpression(
                "sum({})".format(self.walk(fa.args[0]).value)
            ),
            auto_infer_type=False,
        )

    @ew.add_match(
        FunctionApplication(Constant, ...),
        lambda fa: fa.functor.value == operator.neg,
    )
    def negative_value(self, fa):
        return Constant[RelationalAlgebraStringExpression](
            RelationalAlgebraStringExpression(
                "-({})".format(self.walk(fa.args[0]).value)
            ),
            auto_infer_type=False,
        )

    @ew.add_match(Constant[ColumnStr])
    def process_constant_column_str(self, cst_col_str):
        return Constant[RelationalAlgebraStringExpression](
            RelationalAlgebraStringExpression(cst_col_str.value),
            auto_infer_type=False,
        )

    @ew.add_match(Constant[int])
    def process_constant_int(self, cst):
        return Constant[RelationalAlgebraStringExpression](
            RelationalAlgebraStringExpression(cst.value),
            auto_infer_type=False,
        )

    @ew.add_match(Constant[float])
    def process_constant_float(self, cst):
        return Constant[RelationalAlgebraStringExpression](
            RelationalAlgebraStringExpression(cst.value),
            auto_infer_type=False,
        )

    @ew.add_match(Constant[str])
    def process_constant_str(self, cst):
        return Constant[RelationalAlgebraStringExpression](
            RelationalAlgebraStringExpression(cst.value),
            auto_infer_type=False,
        )


class ReplaceConstantColumnStrBySymbol(ew.ExpressionWalker):
    @ew.add_match(Constant[ColumnStr])
    def column_str(self, expression):
        return Symbol[ColumnStr](expression.value)


class RelationalAlgebraSolver(ew.ExpressionWalker):
    """
    Mixing that walks through relational algebra expressions and
    executes the operations.

    Relations are expected to be represented
    as objects with the same interface as :obj:`RelationalAlgebraSet`.
    """

    _rccsbs = ReplaceConstantColumnStrBySymbol()
    _saw = StringArithmeticWalker()
    _fa_2_lambda = ew.FunctionApplicationToPythonLambda()

    def __init__(self, symbol_table=None):
        self.symbol_table = symbol_table

    @ew.add_match(
        Selection(
            Constant, FunctionApplication(
                eq_,
                (Constant[Column], Constant[Column])
            )
        )
    )
    def selection_between_columns(self, selection):
        col1, col2 = selection.formula.args
        selected_relation = (
            selection.relation
        ).value.selection_columns({col1.value: col2.value})

        return self._build_relation_constant(selected_relation)

    def _build_relation_constant(self, relation, type_=Unknown):
        if type_ is not Unknown:
            relation_type = type_
        else:
            relation_type = _infer_relation_type(relation)
        return Constant[relation_type](relation, verify_type=False)

    @ew.add_match(
        Selection(Constant, FunctionApplication(eq_, (Constant[Column], ...)))
    )
    def selection_by_constant(self, selection):
        col, val = selection.formula.args
        selected_relation = selection.relation.value.selection(
            {col.value: val.value}
        )

        return self._build_relation_constant(selected_relation)

    @ew.add_match(
        Selection(Constant, FunctionApplication)
    )
    def selection_general_selection_by_constant(self, selection):
        relation = selection.relation
        compiled_formula = self._compile_function_application_to_sql_fun_exp(
            selection.formula
        )
        selected_relation = relation.value.selection(compiled_formula)
        return self._build_relation_constant(selected_relation)

    @ew.add_match(Projection(Constant, ...))
    def ra_projection(self, projection):
        try:
            relation = projection.relation
            cols = tuple(v.value for v in projection.attributes)
            projected_relation = relation.value.projection(*cols)
        except KeyError:
            raise ProjectionOverMissingColumnsError(
                f"Not all columns {projection.attributes} are present in "
                "the RelationalAlgebra set"
            )
        return self._build_relation_constant(projected_relation)

    @ew.add_match(
        Product,
        lambda product: all(
            isinstance(relation, Constant)
            for relation in product.relations
        )
    )
    def ra_product(self, product):
        if len(product.relations) == 0:
            return Constant[AbstractSet](RelationalAlgebraSet(set()))

        res = self.walk(product.relations[0]).value
        for relation in product.relations[1:]:
            res = res.cross_product(self.walk(relation).value)
        return self._build_relation_constant(res)

    @ew.add_match(EquiJoin(Constant, ..., Constant, ...))
    def ra_equijoin(self, equijoin):
        left = equijoin.relation_left.value
        columns_left = (c.value for c in equijoin.columns_left)
        right = equijoin.relation_right.value
        columns_right = (c.value for c in equijoin.columns_right)
        res = left.equijoin(right, list(zip(columns_left, columns_right)))

        return self._build_relation_constant(res)

    @ew.add_match(NaturalJoin(Constant, Constant))
    def ra_naturaljoin(self, naturaljoin):
        left = naturaljoin.relation_left.value
        right = naturaljoin.relation_right.value
        res = left.naturaljoin(right)
        row_type = self._compute_join_type(naturaljoin, res, left, right)

        return self._build_relation_constant(res, type_=AbstractSet[row_type])

    def _compute_join_type(self, join, res, left, right):
        left_type_args = type_system.get_args(type_system.get_args(
            join.relation_left.type
        )[0])
        right_type_args = type_system.get_args(type_system.get_args(
            join.relation_right.type
        )[0])

        row_type = tuple()
        for column in res.columns:
            column_type = Unknown
            if column in left.columns and len(left_type_args) > 0:
                column_type = left_type_args[left.columns.index(column)]
            if column in right.columns and len(right_type_args) > 0:
                right_type = right_type_args[right.columns.index(column)]
                column_type = type_system.unify_types(
                    column_type, right_type
                )
            row_type += (column_type,)
        return Tuple[row_type]

    @ew.add_match(LeftNaturalJoin(Constant, Constant))
    def ra_left_naturaljoin(self, naturaljoin):
        left = naturaljoin.relation_left.value
        right = naturaljoin.relation_right.value
        res = left.left_naturaljoin(right)
        row_type = self._compute_join_type(naturaljoin, res, left, right)
        return self._build_relation_constant(res, type_=AbstractSet[row_type])

    @ew.add_match(Difference(Constant, Constant))
    def ra_difference(self, difference):
        return self._type_preserving_binary_operation(difference)

    @ew.add_match(Union(Constant, Constant))
    def ra_union(self, union):
        return self._type_preserving_binary_operation(union)

    @ew.add_match(Intersection(Constant, Constant))
    def ra_intersection(self, intersection):
        return self._type_preserving_binary_operation(intersection)

    @ew.add_match(NameColumns(Constant, ...))
    def ra_name_columns(self, name_columns):
        relation = name_columns.relation
        relation_set = relation.value
        column_names = tuple(
            self.walk(column_name).value
            for column_name in name_columns.column_names
        )
        new_set = NamedRelationalAlgebraFrozenSet(column_names, relation_set)
        return self._build_relation_constant(new_set)

    @ew.add_match(RenameColumn(Constant, ..., ...))
    def ra_rename_column(self, rename_column):
        relation = rename_column.relation
        src = rename_column.src.value
        dst = rename_column.dst.value
        new_set = relation.value
        new_set = new_set.rename_column(src, dst)
        return self._build_relation_constant(new_set)

    @ew.add_match(RenameColumns(Constant, ...))
    def ra_rename_columns(self, rename_columns):
        if len(set(c for c, _ in rename_columns.renames)) < len(
            rename_columns.renames
        ):
            raise ValueError("Cannot have duplicated source columns")
        relation = rename_columns.relation
        new_set = relation.value
        renames = {
            src.value: dst.value for src, dst in rename_columns.renames
        }
        new_set = new_set.rename_columns(renames)
        return self._build_relation_constant(new_set)

    @ew.add_match(ConcatenateConstantColumn(Constant, ..., ...))
    def concatenate_constant_column(self, concat_op):
        relation = concat_op.relation
        new_column = concat_op.column_name
        new_column_value = concat_op.column_value
        if new_column.value in relation.value.columns:
            raise NeuroLangException(
                "Cannot concatenate column to a relation that already "
                "has a column with that name. Same column name: {}".format(
                    new_column
                )
            )
        ext_proj_list_members = []
        for column in relation.value.columns:
            cst_column = Constant[ColumnStr](
                column, auto_infer_type=False, verify_type=False
            )
            ext_proj_list_members.append(
                FunctionApplicationListMember(
                    fun_exp=cst_column, dst_column=cst_column,
                )
            )
        ext_proj_list_members.append(
            FunctionApplicationListMember(
                fun_exp=new_column_value, dst_column=new_column
            )
        )
        return self.walk(ExtendedProjection(relation, ext_proj_list_members))

    @ew.add_match(ExtendedProjection(Constant, ...))
    def extended_projection(self, proj_op):
        relation = proj_op.relation
        eval_expressions = {}
        for member in proj_op.projection_list:
            fun_exp = self.walk(member.fun_exp)
            eval_expressions[
                member.dst_column.value
            ] = self._compile_function_application_to_sql_fun_exp(fun_exp)
        with sure_is_not_pattern():
            result = relation.value.extended_projection(eval_expressions)
        return self._build_relation_constant(result)

    def _compile_function_application_to_sql_fun_exp(self, fun_exp):
        if isinstance(fun_exp, FunctionApplication):
            try:
                return self._saw.walk(fun_exp).value
            except NeuroLangPatternMatchingNoMatch:
                fun, args = self._fa_2_lambda.walk(self._rccsbs.walk(fun_exp))
                return lambda t: fun(
                    **{arg: getattr(t, arg) for arg in args}
                )
        elif isinstance(fun_exp, Constant[ColumnInt]):
            return RelationalAlgebraColumnInt(fun_exp.value)
        elif isinstance(fun_exp, Constant[ColumnStr]):
            return RelationalAlgebraColumnStr(fun_exp.value)
        else:
            return fun_exp.value

    @ew.add_match(GroupByAggregation(Constant, ..., ...))
    def aggregate(self, agg_op):
        relation = agg_op.relation
        groupby = list(c.value for c in agg_op.groupby)
        aggregate_functions = []
        for member in agg_op.aggregate_functions:
            fun_args = tuple(arg.value for arg in member.fun_exp.args)
            if len(fun_args) == 1:
                fun_args = fun_args[0]
            aggregate_functions.append(
                (
                    member.dst_column.value,
                    fun_args,
                    member.fun_exp.functor.value,
                )
            )
        with sure_is_not_pattern():
            result = relation.value.aggregate(groupby, aggregate_functions)
        return self._build_relation_constant(result)

    @ew.add_match(FunctionApplication, is_arithmetic_operation)
    def prov_arithmetic_operation(self, arithmetic_op):
        args = self.walk(arithmetic_op.args)
        if any(
            arg_new is not arg_old
            for arg_new, arg_old in zip(args, arithmetic_op.args)
        ):
            return FunctionApplication[arithmetic_op.type](
                arithmetic_op.functor, args
            )
        else:
            return arithmetic_op

    @ew.add_match(Destroy(Constant, ..., ...))
    def set_destroy(self, destroy):
        relation = destroy.relation.value
        src_column = self.walk(destroy.src_column).value
        dst_columns = self.walk(destroy.dst_column).value
        if src_column not in relation.columns:
            raise NeuroLangException(
                f"source column {src_column} not present in "
                "set's columns"
            )

        cols = relation.columns
        set_type = type(relation)
        if not isinstance(dst_columns, tuple):
            dst_columns = (dst_columns,)
        dst_cols = cols + tuple(d for d in dst_columns if d not in cols)
        result_set = set_type(columns=dst_cols)
        if len(cols) > 0:
            row_group_iterator = (t for _, t in relation.groupby(cols))
        else:
            row_group_iterator = (relation,)
        for t in row_group_iterator:
            destroyed_set = set_type(columns=dst_columns)
            for row in t:
                row_set = set_type(
                    columns=dst_columns,
                    iterable=getattr(row, src_column)
                )
                destroyed_set = destroyed_set | row_set
            new_set = (
                t
                .projection(*cols)
                .naturaljoin(destroyed_set)
            )
            result_set = result_set | new_set
        return self._build_relation_constant(result_set)

    @ew.add_match(ReplaceNull(Constant, Constant, Constant))
    def replace_null(self, expression):
        relation = expression.relation.value
        column = expression.column.value
        value = expression.value.value

        new_relation = relation.replace_null(column, value)
        return self._build_relation_constant(
            new_relation, type_=expression.relation.type
        )

    @ew.add_match(Constant)
    def ra_constant(self, constant):
        return constant

    @ew.add_match(Symbol)
    def ra_symbol(self, symbol):
        try:
            constant = self.symbol_table[symbol]
        except KeyError:
            raise NeuroLangException(f"Symbol {symbol} not in table")
        return constant

    @ew.add_match(Constant[RelationalAlgebraStringExpression])
    def arithmetic_string_expression(self, expression):
        return expression

    def _type_preserving_binary_operation(self, ra_op):
        """
        Generic function to apply binary operations (A <op> B) where A and B's
        tuples have the same type, and whose results's tuples have the same
        type as A and B's tuples. This includes, but is not necessarily limited
        to, the Union, Intersection and Difference operations.
        """
        left = self.walk(ra_op.relation_left)
        right = self.walk(ra_op.relation_right)
        left_type = _get_const_relation_type(left)
        right_type = _get_const_relation_type(right)
        type_ = type_system.unify_types(left_type, right_type)
        binary_op_fun_name = {
            Union: "__or__",
            Intersection: "__and__",
            Difference: "__sub__",
        }.get(type(ra_op))
        new_relation = getattr(left.value, binary_op_fun_name)(right.value)
        return self._build_relation_constant(new_relation, type_=type_)


def _const_relation_type_is_known(const_relation):
    """
    Returns whether `T` in `Constant[T]` matches `AbstractSet[Tuple[type_1,
    ..., type_n]]`, in which case we consider the type of the relation's tuples
    to be known.

    """
    type_args = type_system.get_args(const_relation.type)
    if len(type_args) != 1:
        return False
    tuple_type_args = type_system.get_args(type_args[0])
    return len(tuple_type_args) > 0


def _sort_typed_const_named_relation_tuple_type_args(const_named_relation):
    """
    Given a typed `Constant[NamedRelationalAlgebraFrozenSet]`, `R`, with
    columns `c_1, ..., c_n` and whose tuples have the type `Tuple[x_1, ...,
    x_n]`, this function obtains the new type of the relation
    `AbstractSet[Tuple[y_1, ..., y_n]]` such that `y_1, ..., y_n` are the same
    initial types `x_1, ..., x_n` but sorted based on the alphabetical sort of
    columns `c_1, ..., c_n`.

    Notes
    -----
    This is useful when comparing or unifying the types of two named relations
    that have a different column sorting.

    """
    tuple_args = const_named_relation.type.__args__[0].__args__
    columns = const_named_relation.value.columns
    sorted_tuple_args = tuple(
        tuple_args[i]
        for i, _ in sorted(enumerate(columns), key=lambda x: x[1])
    )
    return AbstractSet[Tuple[sorted_tuple_args]]


def _infer_relation_type(relation):
    """
    Infer the type of the tuples in the relation based on its first tuple. If
    the relation is empty, just return `Abstract[Tuple]`.
    """
    if hasattr(relation, "set_row_type"):
        return AbstractSet[relation.set_row_type]
    if relation.is_empty() or relation.arity == 0:
        return AbstractSet[Tuple]
    if hasattr(relation, "row_type"):
        return AbstractSet[relation.row_type]
    tuple_type = Tuple[tuple(type(arg) for arg in relation.fetch_one())]
    return AbstractSet[tuple_type]


def _get_const_relation_type(const_relation):
    if _const_relation_type_is_known(const_relation):
        if isinstance(const_relation.value, NamedRelationalAlgebraFrozenSet):
            return _sort_typed_const_named_relation_tuple_type_args(
                const_relation
            )
        else:
            return const_relation.type
    else:
        return _infer_relation_type(const_relation.value)
