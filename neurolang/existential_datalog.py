from .expressions import NeuroLangException as NLE
from .expressions import (
    Definition, Statement, ExistentialPredicate, Symbol, ExpressionBlock,
    Query, FunctionApplication
)
from .solver_datalog_naive import NaiveDatalog
from .expression_pattern_matching import add_match


class ExistentialDatalog(NaiveDatalog):
    @add_match(
        Statement(ExistentialPredicate, ...),
        # ensure the predicate is a simple function application on symbols
        lambda expression: all(
            isinstance(arg, Symbol) and
            not isinstance(arg, FunctionApplication)
            for arg in expression.lhs.body.args
        )
    )
    def existential_predicate_in_head(self, expression):
        '''
        Add statement with a lhs existential predicate to symbol table
        '''
        eq_variable = expression.lhs.head
        if not isinstance(eq_variable, Symbol):
            raise NLE('\u2203-quantified variable must be a symbol')
        if eq_variable in expression.rhs._symbols:
            raise NLE('\u2203-quantified variable cannot occur in rhs')
        if eq_variable not in expression.lhs.body._symbols:
            raise NLE('\u2203-quantified variable must occur in lhs body')
        predicate_name = expression.lhs.body.functor.name
        if predicate_name in self.symbol_table:
            expressions = self.symbol_table.get(predicate_name).expressions
        else:
            expressions = tuple()
        expressions += (expression, )
        self.symbol_table[predicate_name] = ExpressionBlock(expressions)
        return expression


class SolverExistentialDatalog(ExistentialDatalog):
    @add_match(Query)
    def query_resolution(self, expression):
        pass
