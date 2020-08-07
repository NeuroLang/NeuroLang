
from typing import AbstractSet, Tuple
from uuid import uuid1

from .. import datalog
from .. import expressions as exp
from ..datalog import aggregation
from ..datalog.expression_processing import (TranslateToDatalogSemantics,
                                             reachable_code)
from ..probabilistic.expression_processing import is_within_language_succ_query
from ..type_system import Unknown
from ..utils import RelationalAlgebraFrozenSet
from .datalog import parser as datalog_parser
from .datalog.natural_syntax_datalog import parser as nat_datalog_parser
from .query_resolution import NeuroSynthMixin, QueryBuilderBase, RegionMixin
from .query_resolution_expressions import (
    Expression, Operation, Symbol, TranslateExpressionToFrontEndExpression)

__all__ = ['QueryBuilderDatalog']


class QueryBuilderDatalog(RegionMixin, NeuroSynthMixin, QueryBuilderBase):
    def __init__(self, solver, chase_class=aggregation.Chase):
        super().__init__(
            solver, logic_programming=True
        )
        self.chase_class = chase_class
        self.frontend_translator = \
            TranslateExpressionToFrontEndExpression(self)
        self.translate_expression_to_datalog = TranslateToDatalogSemantics()
        self.datalog_parser = datalog_parser
        self.nat_datalog_parser = nat_datalog_parser

    @property
    def current_program(self):
        cp = []
        for rules in self.solver.intensional_database().values():
            for rule in rules.formulas:
                cp.append(self.frontend_translator.walk(rule))
        return cp

    def assign(self, consequent, antecedent):
        consequent = self.translate_expression_to_datalog.walk(
            consequent.expression
        )
        antecedent = self.translate_expression_to_datalog.walk(
            antecedent.expression
        )
        rule = datalog.Implication(consequent, antecedent)
        self.solver.walk(rule)
        return rule

    def execute_datalog_program(self, code):
        """Execute a datalog program in classical syntax

        Parameters
        ----------
        code : string
            datalog program.
        """
        ir = self.datalog_parser(code)
        self.solver.walk(ir)

    def execute_nat_datalog_program(self, code):
        """Execute a natural language datalog program in classical syntax

        Parameters
        ----------
        code : string
            datalog program.
        """
        ir = self.nat_datalog_parser(code)
        self.solver.walk(ir)

    def query(self, *args):
        """Performs an inferential query on the database.
        There are three modalities
        1. If there is only one argument, the query returns `True` or `False`
        depending on wether the query could be inferred.
        2. If there are two arguments and the first is a tuple of `Symbol`, it
        returns the set of results meeting the query in the second argument.
        3. If the first argument is a predicate (e.g. `Q(x)`) it performs the
        query adds it to the engine memory and returns the
        corresponding symbol.

        Returns
        -------
        bool, frozenset, Symbol
            read the descrpition.
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
            self.add_tuple_set(
                solution_set.value, name=functor_orig.name
            )
            return Symbol(self, out_symbol.name)
        elif len(head) == 0:
            return len(solution_set.value) > 0
        else:
            return RelationalAlgebraFrozenSet(solution_set.value)

    def execute_query(self, head, predicate):
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
        solution = (
            self.chase_class(self.solver, rules=reachable_rules)
            .build_chase_solution()
        )

        solution_set = solution.get(functor.name, exp.Constant(set()))
        self.solver.symbol_table = self.symbol_table.enclosing_scope
        return solution_set, functor_orig

    def solve_all(self):
        '''
        Returns a dictionary of "predicate_name": "Content"
        for all elements in the solution of the datalog program.
        '''
        solution_ir = (
            self.chase_class(self.solver)
            .build_chase_solution()
        )

        solution = {
            k.name: v.value for k, v in solution_ir.items()
        }
        return solution

    def reset_program(self):
        self.symbol_table.clear()

    def add_tuple_set(self, iterable, type_=Unknown, name=None):
        if name is None:
            name = str(uuid1())

        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        symbol = exp.Symbol[AbstractSet[type_]](name)
        self.solver.add_extensional_predicate_from_tuples(
            symbol, iterable, type_=type_
        )

        return Symbol(self, name)

    def predicate_parameter_names(self, predicate_name):
        if isinstance(predicate_name, Symbol):
            predicate_name = predicate_name.neurolang_symbol
        elif (
            isinstance(predicate_name, Expression) and
            isinstance(predicate_name.expression, exp.Symbol)
        ):
            predicate_name = predicate_name.expression
        elif not isinstance(predicate_name, str):
            raise ValueError(f'{predicate_name} is not a string or symbol')
        return tuple(
            s.name if hasattr(s, 'name') else exp.Symbol.fresh().name
            for s in self.solver.predicate_terms(predicate_name)
        )
