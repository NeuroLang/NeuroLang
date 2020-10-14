from collections import defaultdict
from typing import AbstractSet, Tuple, List, Dict, Iterable, Any, Union
from uuid import uuid1

from .. import datalog
from .. import expressions as exp
from ..datalog import aggregation
from ..datalog.expression_processing import (
    TranslateToDatalogSemantics,
    reachable_code,
)
from ..type_system import Unknown
from ..utils import NamedRelationalAlgebraFrozenSet, RelationalAlgebraFrozenSet
from .datalog import parser as datalog_parser
from .datalog.natural_syntax_datalog import parser as nat_datalog_parser
from .query_resolution import NeuroSynthMixin, QueryBuilderBase, RegionMixin
from .query_resolution_expressions import (
    Expression,
    Operation,
    Symbol,
    TranslateExpressionToFrontEndExpression,
)

__all__ = ["QueryBuilderDatalog"]


class QueryBuilderDatalog(RegionMixin, NeuroSynthMixin, QueryBuilderBase):
    def __init__(self, solver, chase_class=aggregation.Chase):
        super().__init__(solver, logic_programming=True)
        self.chase_class = chase_class
        self.frontend_translator = TranslateExpressionToFrontEndExpression(
            self
        )
        self.translate_expression_to_datalog = TranslateToDatalogSemantics()
        self.datalog_parser = datalog_parser
        self.nat_datalog_parser = nat_datalog_parser

    @property
    def current_program(self) -> List[Expression]:
        """Returns the list of expressions that have currently been declared in the
        program

        Returns
        -------
        List[Expression]
            see description

        Example
        -------
        >>> nl = QueryBuilderDatalog(...)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     e.l2[e.x] = e.l[e.x, e.y] & (e.x == e.y)
        ...     cp = nl.current_program
        >>> cp
        [
            l2(x) ← ( l(x, y) ) ∧ ( x eq y )
        ]
        """
        cp = []
        for rules in self.solver.intensional_database().values():
            for rule in rules.formulas:
                cp.append(self.frontend_translator.walk(rule))
        return cp

    def assign(self, consequent: Expression, antecedent: Expression) -> Expression:
        """Creates an implication of the consequent by the antecedent
        and adds the rule to the current program:
            consequent <- antecedent

        Parameters
        ----------
        consequent : Expression
            see description, will be processed to a logic form before
            creating the implication rule
        antecedent : Expression
            see description, will be processed to a logic form before
            creating the implication rule

        Returns
        -------
        Expression
            see description

        Example
        -------
        >>> nl = QueryBuilderDatalog(...)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     nl.assign(e.l2[e.x], e.l2[e.x, e.y])
        ...     cp = nl.current_program
        >>> cp
        [
            l2(x) ← l(x, y)
        ]
        """
        consequent = self.translate_expression_to_datalog.walk(consequent.expression)
        antecedent = self.translate_expression_to_datalog.walk(antecedent.expression)
        rule = datalog.Implication(consequent, antecedent)
        self.solver.walk(rule)
        return rule

    def execute_datalog_program(self, code: str) -> None:
        """Execute a datalog program in classical syntax

        Parameters
        ----------
        code : string
            datalog program.
        """
        ir = self.datalog_parser(code)
        self.solver.walk(ir)

    def execute_nat_datalog_program(self, code: str) -> None:
        """Execute a natural language datalog program in classical syntax

        Parameters
        ----------
        code : string
            datalog program.
        """
        ir = self.nat_datalog_parser(code)
        self.solver.walk(ir)

    def query(self, *args) -> Union[bool, RelationalAlgebraFrozenSet, Symbol]:
        """Performs an inferential query on the database.
        There are three modalities
        1. If there is only one argument, the query returns `True` or `False`
        depending on wether the query could be inferred.
        2. If there are two arguments and the first is a tuple of `Symbol`, it
        returns the set of results meeting the query in the second argument.
        # ! How to write this thirs modality ?
        3. If the first argument is a predicate (e.g. `Q(x)`) it performs the
        query adds it to the engine memory and returns the
        corresponding symbol.

        Returns
        -------
        Union[bool, RelationalAlgebraFrozenSet, Symbol]
            read the descrpition.

        Example
        -------
        Note: example ran with pandas backend
        >>> nl = QueryBuilderDatalog(...)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     e.l2[e.x, e.y] = e.l[e.x, e.y] & (e.x == e.y)
        ...     s1 = nl.query(e.l2[e.x, e.y])
        ...     s2 = nl.query((e.x,), e.l2[e.x, e.y])
        >>> s1
        True
        >>> s2
            x
        0   2
        """

        if len(args) == 1:
            predicate = args[0]
            head = tuple()
        elif len(args) == 2:
            head, predicate = args
        else:
            raise ValueError("query takes 1 or 2 arguments")

        solution_set, functor_orig = self.execute_query(head, predicate)

        if not isinstance(head, tuple):
            out_symbol = exp.Symbol[solution_set.type](functor_orig.name)
            self.add_tuple_set(solution_set.value, name=functor_orig.name)
            return Symbol(self, out_symbol.name)
        elif len(head) == 0:
            return len(solution_set.value) > 0
        else:
            return RelationalAlgebraFrozenSet(solution_set.value)

    def execute_query(
        self, head: Tuple[Expression, ...], predicate: Expression
    ) -> Union[bool, RelationalAlgebraFrozenSet]:
        """Performs an inferential query:
        1- If head is an empty Tuple, will verify if the predicate query
        can be inferred, returning a bool
        2- If head is a tuple of expressions, will return a
        RelationalAlgebraFrozenSet listing the results meeting the query

        Parameters
        ----------
        head : Tuple[Expression, ...]
            see description
        predicate : Expression
            see description

        Returns
        -------
        Union[bool, RelationalAlgebraFrozenSet]
            see description

        Examples
        --------
        Note: example ran with pandas backend
        >>> nl = QueryBuilderDatalog(...)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     e.l2[e.x, e.y] = e.l[e.x, e.y] & (e.x == e.y)
        ...     s1 = nl.execute_query(tuple(), e.l2[e.x, e.y])
        ...     s2 = nl.execute_query((e.x,), e.l2[e.x, e.y])
        >>> s1
        True
        >>> s2
            x
        0   2
        """
        functor_orig = None
        self.solver.symbol_table = self.symbol_table.create_scope()
        if isinstance(head, Operation):
            functor_orig = head.expression.functor
            new_head = self.new_symbol()(*head.arguments)
            functor = new_head.expression.functor
        elif isinstance(head, tuple):
            new_head = self.new_symbol()(*head)
            functor = new_head.expression.functor
        query_expression = self.assign(new_head, predicate)

        reachable_rules = reachable_code(query_expression, self.solver)
        solution = self.chase_class(
            self.solver, rules=reachable_rules
        ).build_chase_solution()

        solution_set = solution.get(functor.name, exp.Constant(set()))
        self.solver.symbol_table = self.symbol_table.enclosing_scope
        return solution_set, functor_orig

    def solve_all(self) -> Dict:
        """
        Returns a dictionary of "predicate_name": "Content"
        for all elements in the solution of the datalog program.

        Returns
        -------
        Dict
            extensional and intentional facts that have been derived
            through the current program

        Example
        -------
        Note: example ran with pandas backend
        >>> nl = QueryBuilderDatalog(...)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     e.l2[e.x] = e.l[e.x, e.y] & (e.x == e.y)
        ...     solution = nl.solve_all()
        >>> solution
        {
            'l':
                0   1
            0   1   2
            1   2   2
            'l2':
                x
            0   2
        }
        """
        solution_ir = self.chase_class(self.solver).build_chase_solution()

        solution = {}
        for k, v in solution_ir.items():
            solution[k.name] = NamedRelationalAlgebraFrozenSet(
                self.predicate_parameter_names(k.name), v.value.unwrap()
            )
            solution[k.name].row_type = v.value.row_type
        return solution

    def reset_program(self) -> None:
        """Clears current symbol table"""
        self.symbol_table.clear()

    def add_tuple_set(
        self, iterable: Iterable, type_: Any = Unknown, name: str = None
    ) -> Symbol:
        """Creates an AbstractSet Symbol containing the elements specified in the
        iterable with a List[Tuple[Any]] format (see examples).
        Typically used to crate extensional facts from existing databases

        Parameters
        ----------
        iterable : Iterable
            typically a list of tuples of values, other formats will
            be interpreted as the latter
        type_ : Any, optional
            type of elements for the tuples, if not specified
            will be inferred from the first element, by default Unknown
        name : str, optional
            name for the AbstractSet symbol, by default None

        Returns
        -------
        Symbol
            see description

        Examples
        --------
        >>> nl = pfe.ProbabilisticFrontend()
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
        if name is None:
            name = str(uuid1())

        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        symbol = exp.Symbol[AbstractSet[type_]](name)
        self.solver.add_extensional_predicate_from_tuples(symbol, iterable, type_=type_)

        return Symbol(self, name)

    def predicate_parameter_names(self, predicate_name: str) -> Tuple[str]:
        """Get the names of the parameters for the given predicate

        Parameters
        ----------
        predicate_name : str
            predicate to obtain the names from

        Returns
        -------
        tuple[str]
            parameter names
        """
        predicate_name = self._get_predicate_name(predicate_name)
        parameter_names = []
        pcount = defaultdict(lambda: 0)
        for s in self.solver.predicate_terms(predicate_name):
            param_name = self._obtain_parameter_name(s)
            pcount[param_name] += 1
            if pcount[param_name] > 1:
                param_name = f"{param_name}_{pcount[param_name] - 1}"
            parameter_names.append(param_name)
        return tuple(parameter_names)

    def _obtain_parameter_name(self, parameter_expression):
        if hasattr(parameter_expression, "name"):
            param_name = parameter_expression.name
        elif hasattr(parameter_expression, "functor") and hasattr(
            parameter_expression.functor, "name"
        ):
            param_name = parameter_expression.functor.name
        else:
            param_name = exp.Symbol.fresh().name
        return param_name

    def _get_predicate_name(self, predicate_name):
        if isinstance(predicate_name, Symbol):
            predicate_name = predicate_name.neurolang_symbol
        elif isinstance(predicate_name, Expression) and isinstance(
            predicate_name.expression, exp.Symbol
        ):
            predicate_name = predicate_name.expression
        elif not isinstance(predicate_name, str):
            raise ValueError(f"{predicate_name} is not a string or symbol")
        return predicate_name
