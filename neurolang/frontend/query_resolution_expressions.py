import operator as op
from functools import wraps
from typing import (
    AbstractSet,
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from .. import datalog as dl
from .. import expressions as ir
from .. import neurolang as nl
from ..datalog import constraints_representation as cr
from ..expression_pattern_matching import NeuroLangPatternMatchingNoMatch
from ..expression_walker import (
    ExpressionWalker,
    ReplaceExpressionsByValues,
    add_match,
)
from ..type_system import is_leq_informative
from ..utils import RelationalAlgebraFrozenSet


class Expression(object):
    """Generic class representing expressions in the front end
    An expression can be anything, from a symbol to an operation
    to a query"""

    def __init__(
        self, query_builder: "QueryBuilderBase", expression: ir.Expression
    ) -> "Expression":
        """Returns frontend expression, containing backend expression
        and associated query_builder as attributes

        Parameters
        ----------
        query_builder : QueryBuilderBase
            used to build the current program
        expression : ir.Expression
            backend expression

        Returns
        -------
        Expression
            frontend expression

        Example
        -------
        >>> nl = NeurolangDL()
        >>> from neurolang.expressions import Symbol as BeSymbol
        >>> BeA = BeSymbol("A")
        >>> FeA = FeExpression(nl, BeA)
        >>> FeA
        A
        """
        self.query_builder = query_builder
        self.expression = expression

    @property
    def type(self) -> Type:
        """Returns backend expression's type"""
        return self.expression.type

    def do(self, name=None):
        # ! what is this method ?
        return self.query_builder.execute_expression(
            self.expression, name=name
        )

    def __call__(self, *args, **kwargs) -> "Operation":
        """Returns a FunctionApplication expression, applied
        to the *args cast as Constant if not Symbol or Expression
        **kwargs are ignored

        Returns
        -------
        Operation
            FunctionApplication of self to *args

        Example
        -------
        >>> nl = NeurolangDL()
        >>> A = nl.new_symbol(name="A")
        >>> x = nl.new_symbol(name="x")
        >>> A(x)
        <class 'neurolang.frontend.query_resolution_expressions.Operation'>
        """
        new_args = []
        for a in args:
            if a is Ellipsis:
                new_args.append(nl.Symbol.fresh())
            elif isinstance(a, Expression):
                new_args.append(a.expression)
            else:
                new_args.append(nl.Constant(a))
        new_args = tuple(new_args)

        if self.query_builder.logic_programming and isinstance(self, Symbol):
            functor = self.neurolang_symbol
        else:
            functor = self.expression

        new_expression = ir.FunctionApplication(functor, new_args)
        return Operation(self.query_builder, new_expression, self, args)

    def __setitem__(
        self,
        key: Union[Tuple["Expression"], "Expression"],
        value: Union["Expression", Any],
    ) -> None:
        """Sets items using a frontend Tuple[Expression] key
        and an Expression value.
        1- If logic programming is enabled, self[key] will be
        interpreted as self(*key) (see __call__ method)
        2- ! If logic programming is not enabled by the builder,
        will set item in the general Python sense !

        Parameters
        ----------
        key : Union[Tuple["Expression"], "Expression"]
            will be interpreted a a *tuple if logic programming
        value : Union["Expression", Any]
            If not a frontend expression, will be cast as
            one with a Constant value

        Example
        -------
        >>> nl = NeurolangDL()
        >>> A = nl.new_symbol(name="A")
        >>> B = nl.new_symbol(name="B")
        >>> x = nl.new_symbol(name="x")
        >>> for i in range(3):
        ...     A[i] = True
        >>> B[x] = A[x] & (x == 1)
        """
        if not isinstance(value, Expression):
            value = Expression(self.query_builder, nl.Constant(value))

        if self.query_builder.logic_programming:
            if not isinstance(key, tuple):
                key = (key,)
            self.query_builder.declare_implication(self(*key), value)
        else:
            super().__setitem__(key, value)

    def __getitem__(
        self, key: Union[Tuple["Expression"], "Expression"]
    ) -> Union["Expression", Any]:
        """Gets Expression value.
        1- If logic programming is enabled, self[key] will be
        interpreted as (see __call__ method):
            a- self(*key) if key is a tuple
            b- self(key) if not
        2- ! If logic programming is not enabled by the builder,
        will get item in the general Python sense !

        Parameters
        ----------
        key : Union[Tuple["Expression"], "Expression"]
            if tuple, will be interpreted as *tuple

        Returns
        -------
        Union["Expression", Any]
            see description

        Example
        -------
        >>> nl = NeurolangDL()
        >>> A = nl.new_symbol(name="A")
        >>> B = nl.new_symbol(name="B")
        >>> x = nl.new_symbol(name="x")
        >>> for i in range(3):
        ...     A[i] = True
        >>> B[x] = A[x] & (x == 1)
        """
        if self.query_builder.logic_programming:
            if isinstance(key, tuple):
                return self(*key)
            else:
                return self(key)
        else:
            super().__getitem__(key)

    def __repr__(self) -> str:
        """Represents expression

        Returns
        -------
        str
            representation of expression

        Example
        -------
        >>> nl = NeurolangDL()
        >>> A = nl.new_symbol(name="nameA")
        >>> x = nl.new_symbol(name="x")
        >>> y = nl.add_symbol(2, name="y")
        >>> print(A[x])
        nameA(x)
        >>> print(y)
        y: <class 'int'> = 2
        """
        if isinstance(self.expression, nl.Constant):
            return repr(self.expression.value)
        elif isinstance(self.expression, dl.magic_sets.AdornedExpression):
            name = f"{self.expression.expression.name}"
            if self.expression.adornment:
                name += f"^{self.expression.adornment}"
            if self.expression.number:
                name += f"_{self.expression.number}"
            return name
        elif isinstance(self.expression, nl.Symbol):
            if self.expression.is_fresh and not (
                hasattr(self, "in_ontology") and self.in_ontology
            ):
                return "..."
            else:
                return f"{self.expression.name}"
        else:
            return object.__repr__(self)

    def __getattr__(self, name: Union["Expression", str]) -> "Operation":
        if isinstance(name, Expression):
            name_ = name.expression
        else:
            name_ = nl.Constant[str](name)
        new_expression = ir.FunctionApplication(
            nl.Constant(getattr),
            (
                self.expression,
                name_,
            ),
        )
        return Operation(self.query_builder, new_expression, self, (name,))

    def help(self) -> str:
        """Returns help based on Expression's subclass"""
        expression = self.expression
        if isinstance(expression, nl.Constant):
            if is_leq_informative(expression.type, Callable):
                return expression.value.__doc__
            elif is_leq_informative(expression.type, AbstractSet):
                return "Set of tuples"
            else:
                return "Constant value"
        elif isinstance(expression, nl.FunctionApplication):
            return "Evaluation of function to parameters"
        elif isinstance(expression, nl.Symbol):
            return "Unlinked symbol"
        else:
            return "Help not defined yet"


binary_operations = (
    op.add,
    op.sub,
    op.mul,
    op.ge,
    op.le,
    op.gt,
    op.lt,
    op.eq,
    op.contains,
)


def op_bind(op):
    @wraps(op)
    def fun(self, *args):
        new_args = tuple(
            (
                arg.expression
                if isinstance(arg, Expression)
                else nl.Constant(arg)
                for arg in (self,) + args
            )
        )
        arg_types = [a.type for a in new_args]
        functor = nl.Constant[Callable[arg_types, nl.Unknown]](
            op, auto_infer_type=False
        )
        new_expression = functor(*new_args)
        res = Operation(
            self.query_builder,
            new_expression,
            op,
            (self,) + args,
            infix=len(args) > 0,
        )
        return res

    return fun


def rop_bind(op):
    @wraps(op)
    def fun(self, value):
        raise NotImplementedError()
        original_value = value
        if isinstance(value, Expression):
            value = value.expression
        else:
            value = nl.Constant(value)

        return Operation(
            self.query_builder,
            op(self.expression, value),
            op,
            (self, original_value),
            infix=True,
        )

    return fun


force_linking = [op.eq, op.ne, op.gt, op.lt, op.ge, op.le]

for operator_name in dir(op):
    operator = getattr(op, operator_name)
    if operator_name.startswith("_"):
        continue

    name = f"__{operator_name}__"
    if name.endswith("___"):
        name = name[:-1]

    if operator in force_linking or not hasattr(Expression, name):
        setattr(Expression, name, op_bind(operator))


for operator in [
    op.add,
    op.sub,
    op.mul,
    op.matmul,
    op.truediv,
    op.floordiv,
    op.mod,  # op.divmod,
    op.pow,
    op.lshift,
    op.rshift,
    op.and_,
    op.xor,
    op.or_,
]:
    name = f"__r{operator.__name__}__"
    if name.endswith("___"):
        name = name[:-1]

    setattr(Expression, name, rop_bind(operator))


class Operation(Expression):
    """An Operation is an Expression representing the
    application of an operator to a tuple of arguments

    Example
    -------
    >>> nl = NeurolangDL()
    >>> A = nl.new_symbol(name="A")
    >>> x = nl.new_symbol(name="x")
    >>> A(x)
    <class 'neurolang.frontend.query_resolution_expressions.Operation'>
    >>> A == x
    <class 'neurolang.frontend.query_resolution_expressions.Operation'>
    """

    operator_repr = {
        op.and_: "\u2227",
        op.or_: "\u2228",
        op.invert: "\u00ac",
    }

    def __init__(
        self,
        query_builder: "QueryBuilderBase",
        expression: ir.Expression,
        operator: ir.FunctionApplication,
        arguments: Tuple[Expression, ...],
        infix: bool = False,
    ) -> "Operation":
        self.query_builder = query_builder
        self.expression = expression
        self.operator = operator
        self.arguments = arguments
        self.infix = infix

    def __repr__(self) -> str:
        if isinstance(self.operator, Symbol):
            op_repr = self.operator.symbol_name
        elif isinstance(self.operator, Operation):
            op_repr = "({})".format(repr(self.operator))
        elif self.operator in self.operator_repr:
            op_repr = self.operator_repr[self.operator]
        elif isinstance(self.operator, Expression):
            op_repr = repr(self.operator)
        elif hasattr(self.operator, "__qualname__"):
            op_repr = self.operator.__qualname__
        else:
            op_repr = repr(self.operator)

        return self.__repr_arguments(op_repr)

    def __repr_arguments(self, op_repr: str) -> str:
        arguments_repr = []
        for a in self.arguments:
            arg_repr = self.__repr_arguments_arg(a)
            arguments_repr.append(arg_repr)
        if self.infix:
            return " {} ".format(op_repr).join(arguments_repr)
        else:
            return "{}({})".format(op_repr, ", ".join(arguments_repr))

    def __repr_arguments_arg(self, a: Expression) -> str:
        if isinstance(a, Operation):
            arg_repr = "( {} )".format(repr(a))
        elif isinstance(a, Symbol):
            arg_repr = a.symbol_name
        else:
            arg_repr = repr(a)
        return arg_repr


class Symbol(Expression):
    """A Symbol represents an atomic Expression. Its is
    the most recurrent element of queries

    Example
    -------
    >>> nl = NeurolangDL()
    >>> A = nl.new_symbol(name="nameA")
    >>> x = nl.new_symbol(name="x")
    >>> y = nl.add_symbol(2, name="y")
    >>> l = nl.add_tuple_set([1, 2, 3], name="l")
    >>> type(y)
    <class 'neurolang.frontend.query_resolution_expressions.Symbol'>
    >>> y.value
    2
    >>> y.expression
    C{2: int}
    >>> A[x].arguments
    (x,)
    >>> [x for x in l]
    [(1,), (2,), (3,)]
    """

    def __init__(
        self, query_builder: "QueryBuilderBase", symbol_name: str
    ) -> "Symbol":
        self.symbol_name = symbol_name
        self.query_builder = query_builder
        self._rsbv = ReplaceExpressionsByValues(
            self.query_builder.solver.symbol_table
        )

    def __repr__(self) -> str:
        symbol = self.symbol
        if isinstance(symbol, Symbol):
            return f"{self.symbol_name}: {symbol.type}"
        elif isinstance(symbol, nl.Constant):
            if ir.is_leq_informative(symbol.type, AbstractSet):
                value = list(self)
            else:
                value = symbol.value

            return f"{self.symbol_name}: {symbol.type} = {value}"
        else:
            return f"{self.symbol_name}: {symbol.type}"

    def _repr_iterable_value(self, symbol: "Symbol") -> List[str]:
        # ! symbol isn't used ?
        contained = []
        for v in self:
            contained.append(repr(v))
        return contained

    def __iter__(self) -> Iterable:
        symbol = self.symbol
        if not (
            isinstance(symbol, nl.Constant)
            and (
                ir.is_leq_informative(symbol.type, AbstractSet)
                or ir.is_leq_informative(symbol.type, Tuple)
            )
        ):
            raise TypeError(
                f"Symbol of type {self.symbol.type} is not iterable"
            )

        if self.query_builder.logic_programming:
            return self.__iter_logic_programming(symbol)
        else:
            return self.__iter_non_logic_programming(symbol)

    def __iter_logic_programming(self, symbol: ir.Symbol) -> Iterable:
        for v in symbol.value:
            if isinstance(v, nl.Constant):
                yield self._rsbv.walk(v.value)
            elif isinstance(v, nl.Symbol):
                yield Symbol(self.query_builder, v)
            else:
                raise nl.NeuroLangException(f"element {v} invalid in set")

    def __iter_non_logic_programming(self, symbol: ir.Symbol) -> Iterable:
        all_symbols = self.query_builder.solver.symbol_table.symbols_by_type(
            symbol.type.__args__[0]
        )

        for s in symbol.value:
            if not isinstance(s, nl.Constant):
                yield Symbol(self.query_builder, s.name)
                continue
            for k, v in all_symbols.items():
                if isinstance(v, nl.Constant) and s is v.value:
                    yield Symbol(self.query_builder, k.name)
                    break
                yield Expression(self.query_builder, nl.Constant(s))

    def __len__(self) -> Optional[int]:
        symbol = self.symbol
        if isinstance(symbol, nl.Constant) and (
            ir.is_leq_informative(symbol.type, AbstractSet)
            or ir.is_leq_informative(symbol.type, Tuple)
        ):
            return len(symbol.value)

    def __eq__(self, other: Union[Expression, Any]) -> bool:
        if isinstance(other, Expression):
            return self.expression == other.expression
        else:
            return self.expression == other

    def __hash__(self) -> int:
        return hash(self.expression)

    @property
    def symbol(self) -> ir.Symbol:
        """Returns symbol from symbol_table"""
        return self.query_builder.solver.symbol_table[self.symbol_name]

    @property
    def neurolang_symbol(self) -> ir.Symbol:
        """Returns backend symbol"""
        return nl.Symbol[self.type](self.symbol_name)

    @property
    def expression(self) -> ir.Symbol:
        """Overloads symbol property"""
        return self.symbol

    @property
    def value(self) -> Any:
        """If any, returns value corresponding to the symbol

        Returns
        -------
        Any
            see description

        Raises
        ------
        ValueError
            if Symbol doesn't have a python value
        """
        constant = self.query_builder.solver.symbol_table[self.symbol_name]
        if isinstance(constant, ir.Constant) and isinstance(
            constant.value, RelationalAlgebraFrozenSet
        ):
            return RelationalAlgebraFrozenSet(constant.value)
        else:
            try:
                return self._rsbv.walk(constant)
            except NeuroLangPatternMatchingNoMatch:
                raise ValueError("Expression doesn't have a python value")

    @property
    def parameter_names(self):
        # ! Doesn't seem to be any trace of this in code base,
        # ! what does it do ?
        return self.query_builder.parameter_names(self)


class Query(Expression):
    """A query represents the symbols that verify a given
    predicate:
    x | x%2 == 0
    with x an int represents even numbers"""

    def __init__(
        self,
        query_builder: "QueryBuilderBase",
        expression: ir.Expression,
        symbol: Symbol,
        predicate: Expression,
    ) -> "Query":
        self.query_builder = query_builder
        self.expression = expression
        self.symbol = symbol
        self.predicate = predicate

    def __repr__(self) -> str:
        return "{{{s} | {p}}}".format(
            s=repr(self.symbol), p=repr(self.predicate)
        )


class Exists(Expression):
    """Corresponds to the logical ∃
    ∃x: x == 1
    enunciates a truth"""

    def __init__(
        self,
        query_builder: "QueryBuilderBase",
        expression: ir.Expression,
        symbol: Symbol,
        predicate: Expression,
    ) -> "Exists":
        self.query_builder = query_builder
        self.expression = expression
        self.symbol = symbol
        self.predicate = predicate

    def __repr__(self) -> str:
        return "\u2203{s}: {p}".format(
            s=repr(self.symbol), p=repr(self.predicate)
        )


class All(Expression):
    """Corresponds to the logical ∀
    ∀x: x == x
    enunciates a truth"""

    def __init__(
        self,
        query_builder: "QueryBuilderBase",
        expression: ir.Expression,
        symbol: Symbol,
        predicate: Expression,
    ) -> "All":
        self.query_builder = query_builder
        self.expression = expression
        self.symbol = symbol
        self.predicate = predicate

    def __repr__(self) -> str:
        return "\u2200{s}: {p}".format(
            s=repr(self.symbol), p=repr(self.predicate)
        )


class Implication(Expression):
    """Corresponds to the logical implication:
    consequent ← antecedent
    or alternatively
    consequent if antecedent"""

    def __init__(
        self,
        query_builder: "QueryBuilderBase",
        expression: ir.Expression,
        consequent: Expression,
        antecedent: Expression,
    ) -> "Implication":
        self.expression = expression
        self.query_builder = query_builder
        self.antecedent = antecedent
        self.consequent = consequent

    def __repr__(self) -> str:
        return "{c} \u2190 {a}".format(
            a=repr(self.antecedent), c=repr(self.consequent)
        )


class RightImplication(Expression):
    """Corresponds to the logical implication:
    antecedent → consequent
    or alternatively
    if antecedent then consequent"""

    def __init__(
        self,
        query_builder: "QueryBuilderBase",
        expression: ir.Expression,
        antecedent: Expression,
        consequent: Expression,
    ) -> "RightImplication":
        self.expression = expression
        self.query_builder = query_builder
        self.antecedent = antecedent
        self.consequent = consequent

    def __repr__(self) -> str:
        return "{a} \u2192 {c}".format(
            a=repr(self.antecedent), c=repr(self.consequent)
        )


class Fact(Expression):
    """A Fact reprsents an information considered
    as True. It can be seen as the Implication:
    fact ← True"""

    def __init__(
        self,
        query_builder: "QueryBuilderBase",
        expression: ir.Expression,
        consequent: Expression,
    ) -> "Fact":
        self.expression = expression
        self.query_builder = query_builder
        self.consequent = consequent

    def __repr__(self) -> str:
        return "{c}".format(
            c=repr(self.consequent),
        )


class TranslateExpressionToFrontEndExpression(ExpressionWalker):
    """Walks through a backend Expression to translate it to a
    frontend Expression"""

    def __init__(
        self, query_builder: "QueryBuilderBase"
    ) -> "TranslateExpressionToFrontEndExpression":
        self.query_builder = query_builder
        self.right_implication_mode = False

    @add_match(ir.Symbol)
    def symbol(self, expression: ir.Expression) -> Expression:
        ret = Expression(self.query_builder, expression)
        ret.in_ontology = self.right_implication_mode
        return ret

    @add_match(ir.Constant)
    def constant(self, expression: ir.Expression) -> Any:
        return expression.value

    @add_match(ir.FunctionApplication)
    def function_application(self, expression: ir.Expression) -> Any:
        functor = self.walk(expression.functor)
        args = tuple(self.walk(arg) for arg in expression.args)
        return functor(*args)

    @add_match(dl.Implication(..., True))
    def fact(self, expression: ir.Expression) -> Fact:
        return Fact(
            self.query_builder, expression, self.walk(expression.consequent)
        )

    @add_match(dl.Implication)
    def implication(self, expression: ir.Expression) -> Implication:
        return Implication(
            self.query_builder,
            expression,
            self.walk(expression.consequent),
            self.walk(expression.antecedent),
        )

    @add_match(cr.RightImplication)
    def right_implication(self, expression):
        self.right_implication_mode = True
        ret = RightImplication(
            self.query_builder,
            expression,
            self.walk(expression.antecedent),
            self.walk(expression.consequent),
        )
        self.right_implication_mode = False
        return ret

    @add_match(dl.Conjunction)
    def conjunction(self, expression: ir.Expression) -> Expression:
        formulas = list(expression.formulas[::-1])
        current_expression = self.walk(formulas.pop())
        while len(formulas) > 0:
            current_expression = current_expression & self.walk(formulas.pop())
        return current_expression
