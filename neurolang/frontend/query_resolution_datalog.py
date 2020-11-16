r"""
Query Builder Datalog
=====================
Complements QueryBuilderBase with query capabilities,
as well as Region and Neurosynth capabilities
"""
from collections import defaultdict
from typing import (
    AbstractSet,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from uuid import uuid1

from .. import datalog
from .. import expressions as ir
from ..datalog import aggregation
from ..datalog.constraints_representation import RightImplication
from ..datalog.expression_processing import (
    TranslateToDatalogSemantics,
    reachable_code,
)
from ..type_system import Unknown
from ..utils import NamedRelationalAlgebraFrozenSet, RelationalAlgebraFrozenSet
from .datalog.standard_syntax import parser as datalog_parser
from .datalog.natural_syntax import parser as nat_datalog_parser
from .query_resolution import NeuroSynthMixin, QueryBuilderBase, RegionMixin
from ..datalog import DatalogProgram
from . import query_resolution_expressions as fe

__all__ = ["QueryBuilderDatalog"]


class QueryBuilderDatalog(RegionMixin, NeuroSynthMixin, QueryBuilderBase):
    """
    Complements QueryBuilderBase with query capabilities,
    as well as Region and Neurosynth capabilities
    """

    def __init__(
        self,
        program_ir: DatalogProgram,
        chase_class: Type[aggregation.Chase] = aggregation.Chase,
    ) -> "QueryBuilderDatalog":
        """
        Query builder with query, Region, Neurosynth capabilities

        Parameters
        ----------
        program_ir : DatalogProgram
            Datalog program's intermediate representation,
            usually blank
        chase_class : Type[aggregation.Chase], optional
            used to compute deterministic solutions,
            by default aggregation.Chase

        Returns
        -------
        QueryBuilderDatalog
            see description
        """
        super().__init__(program_ir, logic_programming=True)
        self.chase_class = chase_class
        self.frontend_translator = fe.TranslateExpressionToFrontEndExpression(
            self
        )
        self.translate_expression_to_datalog = TranslateToDatalogSemantics()
        self.datalog_parser = datalog_parser
        self.nat_datalog_parser = nat_datalog_parser

    @property
    def current_program(self) -> List[fe.Expression]:
        """
        Returns the list of expressions that have currently been
        declared in the program

        Returns
        -------
        List[fe.Expression]
            see description

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
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
        for rules in self.program_ir.intensional_database().values():
            for rule in rules.formulas:
                cp.append(self.frontend_translator.walk(rule))
        return cp

    def _declare_implication(
        self, consequent: fe.Expression, antecedent: fe.Expression
    ) -> fe.Expression:
        """
        Creates an implication of the consequent by the antecedent
        and adds the rule to the current program:
            consequent <- antecedent

        Parameters
        ----------
        consequent : fe.Expression
            see description, will be processed to a logic form before
            creating the implication rule
        antecedent : fe.Expression
            see description, will be processed to a logic form before
            creating the implication rule

        Returns
        -------
        fe.Expression
            see description

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     nl._declare_implication(e.l2[e.x], e.l2[e.x, e.y])
        ...     cp = nl.current_program
        >>> cp
        [
            l2(x) ← l(x, y)
        ]
        """
        consequent = self.translate_expression_to_datalog.walk(
            consequent.expression
        )
        antecedent = self.translate_expression_to_datalog.walk(
            antecedent.expression
        )
        rule = datalog.Implication(consequent, antecedent)
        self.program_ir.walk(rule)
        return rule

    def add_constraint(
        self, antecedent: fe.Expression, consequent: fe.Expression
    ) -> fe.Expression:
        """
        Creates an right implication of the consequent by the antecedent
        and adds the rule to the current program:
            antecedent -> consequent

        Parameters
        ----------
        antecedent : fe.Expression
            see description, will be processed to a logic form before
            creating the right implication rule
        consequent : fe.Expression
            see description, will be processed to a logic form before
            creating the right implication rule

        Returns
        -------
        fe.Expression
            see description

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     nl.add_constraint(e.l2[e.x, e.y], e.l2[e.x])
        """
        consequent = self.translate_expression_to_datalog.walk(
            consequent.expression
        )
        antecedent = self.translate_expression_to_datalog.walk(
            antecedent.expression
        )
        rule = RightImplication(antecedent, consequent)
        self.program_ir.walk(rule)
        return rule

    def execute_datalog_program(self, code: str) -> None:
        """
        Execute a Datalog program in classical syntax

        Parameters
        ----------
        code : string
            Datalog program.
        """
        intermediate_representation = self.datalog_parser(code)
        self.program_ir.walk(intermediate_representation)

    def execute_nat_datalog_program(self, code: str) -> None:
        """Execute a natural language Datalog program in classical syntax

        Parameters
        ----------
        code : string
            Datalog program.
        """
        intermediate_representation = self.nat_datalog_parser(code)
        self.program_ir.walk(intermediate_representation)

    def query(
        self, *args
    ) -> Union[bool, RelationalAlgebraFrozenSet, fe.Symbol]:
        """
        Performs an inferential query on the database.
        There are three modalities
        1. If there is only one argument, the query returns `True` or `False`
        depending on wether the query could be inferred.
        2. If there are two arguments and the first is a tuple of `fe.Symbol`,
        it returns the set of results meeting the query in the second argument.
        3. If the first argument is a predicate (e.g. `Q(x)`) it performs the
        query, adds it to the engine memory, and returns the
        corresponding symbol.
        See example for 3 modalities

        Returns
        -------
        Union[bool, RelationalAlgebraFrozenSet, fe.Symbol]
            read the descrpition.

        Example
        -------
        Note: example ran with pandas backend
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.environment as e:
        ...     e.l2[e.x, e.y] = e.l[e.x, e.y] & (e.x == e.y)
        ...     s1 = nl.query(e.l2[e.x, e.y])
        ...     s2 = nl.query((e.x,), e.l2[e.x, e.y])
        ...     s3 = nl.query(e.l3[e.x], e.l2[e.x, e.y])
        >>> s1
        True
        >>> s2
            x
        0   2
        >>> s3
        l3: typing.AbstractSet[typing.Tuple[int]] = [(2,)]
        """
        if len(args) == 1:
            predicate = args[0]
            head = tuple()
        elif len(args) == 2:
            head, predicate = args
            if isinstance(head, fe.Symbol):
                head = (head,)
        else:
            raise ValueError("query takes 1 or 2 arguments")

        solution_set, functor_orig = self._execute_query(head, predicate)

        if not isinstance(head, tuple):
            out_symbol = ir.Symbol[solution_set.type](functor_orig.name)
            self.add_tuple_set(solution_set.value, name=functor_orig.name)
            return fe.Symbol(self, out_symbol.name)
        elif len(head) == 0:
            return len(solution_set.value) > 0
        else:
            return RelationalAlgebraFrozenSet(solution_set.value)

    def _execute_query(
        self,
        head: Union[fe.Symbol, Tuple[fe.Expression, ...]],
        predicate: fe.Expression,
    ) -> Tuple[AbstractSet, Optional[ir.Symbol]]:
        """
        [Internal usage - documentation for developpers]

        Performs an inferential query. Will return as first output
        an AbstractSet with as many elements as solutions of the
        predicate query. The AbstractSet's columns correspond to
        the expressions in the head.
        If head expressions are arguments of a functor, the latter will
        be returned as the second output, defaulted as None.

        Parameters
        ----------
        head : Union[fe.Symbol, Tuple[fe.Expression, ...]]
            see description
        predicate : fe.Expression
            see description

        Returns
        -------
        Tuple[AbstractSet, Optional[fe.Symbol]]
            see description

        Examples
        --------
        Note: example ran with pandas backend
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     e.l2[e.x, e.y] = e.l[e.x, e.y] & (e.x == e.y)
        ...     s1 = nl._execute_query(tuple(), e.l2[e.x, e.y])
        ...     s2 = nl._execute_query((e.x,), e.l2[e.x, e.y])
        ...     s3 = nl._execute_query(e.l2[e.x, e.y], e.l2[e.x, e.y])
        >>> s1
        (
            C{
                Empty DataFrame
                Columns: []
                Index: [0]
                : typing.AbstractSet
            },
            None
        )
        >>> s2
        (
            C{
                    x
                0   2
                : typing.AbstractSet
            },
            None
        )
        >>> s3
        (
            C{
                    x   y
                0   2   2
                : typing.AbstractSet
            },
            S{
                l2: Unknown
            }
        )
        """
        functor_orig = None
        self.program_ir.symbol_table = self.symbol_table.create_scope()
        if isinstance(head, fe.Operation):
            functor_orig = head.expression.functor
            new_head = self.new_symbol()(*head.arguments)
            functor = new_head.expression.functor
        elif isinstance(head, tuple):
            new_head = self.new_symbol()(*head)
            functor = new_head.expression.functor
        query_expression = self._declare_implication(new_head, predicate)

        reachable_rules = reachable_code(query_expression, self.program_ir)
        solution = self.chase_class(
            self.program_ir, rules=reachable_rules
        ).build_chase_solution()

        solution_set = solution.get(functor.name, ir.Constant(set()))
        self.program_ir.symbol_table = self.symbol_table.enclosing_scope
        return solution_set, functor_orig

    def solve_all(self) -> Dict[str, NamedRelationalAlgebraFrozenSet]:
        """
        Returns a dictionary of "predicate_name": "Content"
        for all elements in the solution of the Datalog program.

        Returns
        -------
        Dict[str, NamedRelationalAlgebraFrozenSet]
            extensional and intentional facts that have been derived
            through the current program

        Example
        -------
        Note: example ran with pandas backend
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
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
        solution_ir = self.chase_class(self.program_ir).build_chase_solution()

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
        self, iterable: Iterable, type_: Type = Unknown, name: str = None
    ) -> fe.Symbol:
        """
        Creates an AbstractSet fe.Symbol containing the elements specified in
        the iterable with a List[Tuple[Any, ...]] format (see examples).
        Typically used to crate extensional facts from existing databases

        Parameters
        ----------
        iterable : Iterable
            typically a list of tuples of values, other formats will
            be interpreted as the latter
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
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
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
        symbol = ir.Symbol[AbstractSet[type_]](name)
        self.program_ir.add_extensional_predicate_from_tuples(
            symbol, iterable, type_=type_
        )

        return fe.Symbol(self, name)

    def predicate_parameter_names(
        self, predicate_name: Union[str, fe.Symbol, fe.Expression]
    ) -> Tuple[str]:
        """
        Get the names of the parameters for the given predicate

        Parameters
        ----------
        predicate_name : Union[str, fe.Symbol, fe.Expression]
            predicate to obtain the names from

        Returns
        -------
        tuple[str]
            parameter names
        """
        predicate_name = self._get_predicate_name(predicate_name)
        parameter_names = []
        pcount = defaultdict(lambda: 0)
        for s in self.program_ir.predicate_terms(predicate_name):
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
            param_name = ir.Symbol.fresh().name
        return param_name

    def _get_predicate_name(self, predicate_name):
        if isinstance(predicate_name, fe.Symbol):
            predicate_name = predicate_name.neurolang_symbol
        elif isinstance(predicate_name, fe.Expression) and isinstance(
            predicate_name.expression, ir.Symbol
        ):
            predicate_name = predicate_name.expression
        elif not isinstance(predicate_name, str):
            raise ValueError(f"{predicate_name} is not a string or symbol")
        return predicate_name
