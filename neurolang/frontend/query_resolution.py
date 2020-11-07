r"""
Query Builder Base
================================================
Base classes to construct Datalog programs.
Capabilities to declare, manage and manipulate symbols.
"""
from contextlib import contextmanager
from typing import (
    AbstractSet,
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from uuid import uuid1

from .. import expressions as ir
from ..type_system import Unknown, is_leq_informative
from ..typed_symbol_table import TypedSymbolTable
from . import query_resolution_expressions as fe
from ..datalog import DatalogProgram


class QueryBuilderBase:
    """
    Base class to construct Datalog programs.
    Capabilities to declare, manage and manipulate symbols.
    """

    def __init__(
        self, program_ir: DatalogProgram, logic_programming: bool = False
    ) -> "QueryBuilderBase":
        """
        Query Builder with symbol management capabilities

        Parameters
        ----------
        program_ir : DatalogProgram
            Datalog program's intermediate representation,
            usually blank
        logic_programming : bool, optional
            defines if symbols can be dynamically
            declared, by default False

        Returns
        -------
        QueryBuilderBase
            see description
        """
        self.program_ir = program_ir
        self.set_type = AbstractSet[self.program_ir.type]
        self.logic_programming = logic_programming

        for k, v in self.program_ir.included_functions.items():
            self.symbol_table[ir.Symbol[v.type](k)] = v

        self._symbols_proxy = QuerySymbolsProxy(self)

    def get_symbol(self, symbol_name: Union[str, fe.Expression]) -> fe.Symbol:
        """
        Retrieves symbol via its name, either providing a
        fe.Expression with the correct name or the name itself

        Parameters
        ----------
        symbol_name : Union[str, fe.Expression]
            name of the symbol to be retrieved. If of type fe.Expression,
            expression's name is used as the name

        Returns
        -------
        fe.Symbol
            symbol corresponding to the input name

        Raises
        ------
        ValueError
            if no symbol could be found with given name

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> nl.add_symbol(3, "x")
        >>> nl.get_symbol("x")
        x: <class 'int'> = 3
        """
        if isinstance(symbol_name, fe.Expression):
            symbol_name = symbol_name.expression.name
        if symbol_name not in self.symbol_table:
            raise ValueError(f"fe.Symbol {symbol_name} not defined")
        return fe.Symbol(self, symbol_name)

    def __getitem__(
        self, symbol_name: Union[fe.Symbol, str, fe.Expression]
    ) -> fe.Symbol:
        """
        Retrieves symbol via its name, either providing a
        fe.Expression with the correct name or the name itself
        Points towards .get_symbol method

        Parameters
        ----------
        symbol_name : Union[fe.Symbol, str, fe.Expression]
            if of type fe.Symbol, symol's name will be passed
            to the .get_symbol method

        Returns
        -------
        fe.Symbol
            output from .get_symbol method
        """
        if isinstance(symbol_name, fe.Symbol):
            symbol_name = symbol_name.symbol_name
        return self.get_symbol(symbol_name)

    def __contains__(self, symbol: fe.Symbol) -> bool:
        """
        Checks if symbol exists in current symbol_table

        Parameters
        ----------
        symbol : fe.Symbol
            symbol to check

        Returns
        -------
        bool
            does symbol exist in current symbol_table
        """
        return symbol in self.symbol_table

    @property
    def types(self) -> List[Type]:
        """
        Returns a list of the types of the symbols currently
        in the table (type can be Unknown)

        Returns
        -------
        List[Union[fe.Expression, Type]]
            List of types or Unknown if symbol has no known type
        """
        return self.symbol_table.types

    @property
    def symbol_table(self) -> TypedSymbolTable:
        """Projector to the program_ir's symbol_table"""
        return self.program_ir.symbol_table

    @property
    def symbols(self) -> Iterator[str]:
        """Iterator through the symbol's names"""
        return self._symbols_proxy

    @property
    @contextmanager
    def environment(self) -> "QuerySymbolsProxy":
        """
        Dynamic context that can be used to create
        symbols to write a Datalog program.
        Contrary to a scope, symbols stay in the symbol_table
        when exiting the environment context

        Yields
        -------
        QuerySymbolsProxy
            in dynamic mode, can be used to create symbols on-the-fly

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> with nl.environment as e:
        ...     e.x = 3
        >>> "x" in nl
        True
        >>> nl.symbols.x == 3
        True
        """
        old_dynamic_mode = self._symbols_proxy._dynamic_mode
        self._symbols_proxy._dynamic_mode = True
        try:
            yield self._symbols_proxy
        finally:
            self._symbols_proxy._dynamic_mode = old_dynamic_mode

    @property
    @contextmanager
    def scope(self) -> "QuerySymbolsProxy":
        """
        Dynamic context that can be used to create
        symbols to write a Datalog program.
        Contrary to an environment, symbols disappear from the symbol_table
        when exiting the scope context

        Yields
        -------
        QuerySymbolsProxy
            in dynamic mode, can be used to create symbols on-the-fly

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> with nl.scope as e:
        ...     e.x = 3
        >>> "x" in nl
        False
        """
        old_dynamic_mode = self._symbols_proxy._dynamic_mode
        self._symbols_proxy._dynamic_mode = True
        self.program_ir.push_scope()
        try:
            yield self._symbols_proxy
        finally:
            self.program_ir.pop_scope()
            self._symbols_proxy._dynamic_mode = old_dynamic_mode

    def new_symbol(
        self,
        type_: Union[Any, Tuple[Any, ...], List[Any]] = Unknown,
        name: str = None,
    ) -> fe.Expression:
        """
        Creates a symbol and associated expression, optionally
        specifying it's type and/or name

        Parameters
        ----------
        type_ : Union[Any, Tuple[Any, ...], List[Any]], optional
            type of the created symbol, by default Unknown
            if Iterable, will be cast to a Tuple
        name : str, optional
            name of the created symbol, by default None

        Returns
        -------
        fe.Expression
            associated to the created symbol
        """
        if isinstance(type_, (tuple, list)):
            type_ = tuple(type_)
            type_ = Tuple[type_]

        if name is None:
            name = str(uuid1())
        return fe.Expression(self, ir.Symbol[type_](name))

    @property
    def functions(self) -> List[str]:
        """
        Returns the list of symbols corresponding to callables

        Returns
        -------
        List[str]
            list of symbols of type leq Callable

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> def f(x: int) -> int:
        ...     return x+2
        >>> nl.add_symbol(f, "f")
        f: typing.Callable[[int], int] = <function f at ...>
        >>> "f" in nl.functions
        True
        >>> nl.add_symbol(3, "x")
        x: <class 'int'> = 3
        >>> "x" in nl.functions
        False
        """
        return [
            s.name
            for s in self.symbol_table
            if is_leq_informative(s.type, Callable)
        ]

    def add_symbol(
        self, value: Union[fe.Expression, ir.Constant, Any], name: str = None
    ) -> fe.Symbol:
        """
        Creates a symbol with given value and adds it to the
        current symbol_table.
        Can typicaly be used to decorate callables, or add an
        ir.Constant to the program.

        Parameters
        ----------
        value : Union[fe.Expression, ir.Constant, Any]
            value of the symbol to add. If not an fe.Expression,
            will be cast as an ir.Constant
        name : str, optional
            overrides automatic naming of the symbol, by default None

        Returns
        -------
        fe.Symbol
            created symbol

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> @nl.add_symbol
        ... def g(x: int) -> int:
        ...     return x + 2
        >>> nl.get_symbol("g")
        g: typing.Callable[[int], int] = <function g at ...>
        >>> nl.add_symbol(3, "x")
        >>> nl.get_symbol("x")
        x: <class 'int'> = 3
        """
        name = self._obtain_symbol_name(name, value)

        if isinstance(value, fe.Expression):
            value = value.expression
        elif isinstance(value, ir.Constant):
            pass
        else:
            value = ir.Constant(value)

        symbol = ir.Symbol[value.type](name)
        self.symbol_table[symbol] = value

        return fe.Symbol(self, name)

    def _obtain_symbol_name(self, name: Optional[str], value: Any) -> str:
        if name is not None:
            return name

        if not hasattr(value, "__qualname__"):
            return str(uuid1())

        if "." in value.__qualname__:
            ix = value.__qualname__.rindex(".")
            name = value.__qualname__[ix + 1 :]
        else:
            name = value.__qualname__

        return name

    def del_symbol(self, name: str) -> None:
        """
        Deletes the symbol with parameter name
        from the symbol_table

        Parameters
        ----------
        name : str
            Name of the symbol to delete

        Raises
        ------
        ValueError
            if no symbol could be found with given name

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> nl.add_symbol(3, "x")
        x: <class 'int'> = 3
        >>> nl.get_symbol("x")
        x: <class 'int'> = 3
        >>> nl.del_symbol("x")
        >>> nl.get_symbol("x")
        ValueError: Symbol x not defined
        """
        del self.symbol_table[name]

    def add_tuple_set(
        self,
        iterable: Union[Iterable, Iterable[Tuple[Any, ...]]],
        type_: Type = Unknown,
        name: str = None,
    ) -> fe.Symbol:
        """
        Creates an AbstractSet fe.Symbol containing the elements specified in
        the iterable with a List[Tuple[Any, ...]] format (see examples).
        Typically used to create extensional facts from existing databases

        Parameters
        ----------
        iterable : Union[Iterable, Iterable[Tuple[Any, ...]]]
            typically a list of tuples of values, other formats will
            be interpreted as the latter (see examples)
        type_ : Type, optional
            type of elements for the tuples, if not specified
            will be inferred from the first element, by default Unknown
        name : str, optional
            name for the AbstractSet symbol, by default None

        Returns
        -------
        fe.Symbol
            see description

        Examples
        --------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (3, 4)], name="l1")
        l1: typing.AbstractSet[typing.Tuple[int, int]] = \
            [(1, 2), (3, 4)]
        >>> nl.add_tuple_set([[1, 2, 3], (3, 4)], name="l2")
        l2: typing.AbstractSet[typing.Tuple[int, int, float]] = \
            [(1, 2, 3.0), (3, 4, nan)]
        >>> nl.add_tuple_set((1, 2, 3), name="l3")
        l3: typing.AbstractSet[typing.Tuple[int]] = \
            [(1,), (2,), (3,)]
        """
        if not isinstance(type_, tuple) or len(type_) == 1:
            if isinstance(type_, tuple) and len(type_) == 1:
                type_ = type_[0]
                iterable = (e[0] for e in iterable)

            set_type = AbstractSet[type_]
        else:
            type_ = tuple(type_)
            set_type = AbstractSet[Tuple[type_]]

        constant = self._add_tuple_set_elements(iterable, set_type)
        if name is None:
            name = str(uuid1())

        symbol = ir.Symbol[set_type](name)
        self.symbol_table[symbol] = constant

        return fe.Symbol(self, name)

    def _add_tuple_set_elements(self, iterable, set_type):
        element_type = set_type.__args__[0]
        new_set = []
        for e in iterable:
            if not (isinstance(e, fe.Symbol)):
                s, c = self._create_symbol_and_get_constant(e, element_type)
                self.symbol_table[s] = c
            else:
                s = e.neurolang_symbol
            new_set.append(s)

        return ir.Constant[set_type](self.program_ir.new_set(new_set))

    @staticmethod
    def _create_symbol_and_get_constant(element, element_type):
        symbol = ir.Symbol[element_type].fresh()
        if isinstance(element, ir.Constant):
            constant = element.cast(element_type)
        elif is_leq_informative(element_type, Tuple):
            constant = ir.Constant[element_type](
                tuple(ir.Constant(ee) for ee in element)
            )
        else:
            constant = ir.Constant[element_type](element)
        return symbol, constant


class QuerySymbolsProxy:
    """
    Class useful to create symbols on-the-fly
    Typically used in QueryBuilderBase contexts as the yielded value
    to write a program.
    Various methods are projectors to QueryBuilderBase methods
    """

    def __init__(self, query_builder):
        self._dynamic_mode = False
        self._query_builder = query_builder

    def __getattr__(self, name):
        """See QueryBuilderBase.get_symbol"""
        if name in self.__getattribute__("_query_builder"):
            return self._query_builder.get_symbol(name)

        try:
            return super().__getattribute__(name)
        except AttributeError:
            if self._dynamic_mode:
                return self._query_builder.new_symbol(Unknown, name=name)
            else:
                raise

    def __setattr__(self, name, value):
        """See QueryBuilderBase.add_symbol"""
        if name == "_dynamic_mode":
            return super().__setattr__(name, value)
        elif self._dynamic_mode:
            return self._query_builder.add_symbol(value, name=name)
        else:
            return super().__setattr__(name, value)

    def __delattr__(self, name):
        """See QueryBuilderBase.del_symbol"""
        if self._dynamic_mode and name:
            self._query_builder.del_symbol(name)
        else:
            super().__delattr__(name)

    def __getitem__(self, attr):
        """See QueryBuilderBase.get_symbol"""
        return self._query_builder.get_symbol(attr)

    def __setitem__(self, key, value):
        """See QueryBuilderBase.add_symbol"""
        return self._query_builder.add_symbol(value, name=key)

    def __contains__(self, symbol):
        """See QueryBuilderBase.__contains__"""
        return symbol in self._query_builder.symbol_table

    def __iter__(self) -> List[str]:
        """
        Iterates through the names of the symbols
        currently in the symbol_table, ordered in ascending name

        Returns
        -------
        List[str]
            list of symbol names
        """
        return iter(
            sorted(set(s.name for s in self._query_builder.symbol_table))
        )

    def __len__(self) -> int:
        """
        Returns number of symbols currently in symbol_table

        Returns
        -------
        int
            see description
        """
        return len(self._query_builder.symbol_table)

    def __dir__(self):
        """Descibes self and lists symbols in current symbol_table"""
        init = object.__dir__(self)
        init += [symbol.name for symbol in self._query_builder.symbol_table]
        return init

    def __repr__(self):
        """Describes symbols currently in symbol_table"""
        init = [symbol.name for symbol in self._query_builder.symbol_table]
        return f"QuerySymbolsProxy with symbols {init}"
