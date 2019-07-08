from .expressions import (
    FunctionApplication, Constant, NeuroLangException, is_leq_informative,
    Symbol, Lambda, ExpressionBlock, Expression, Definition,
    Query, ExistentialPredicate, Quantifier,
)

from .solver_datalog_naive import (
    SolverNonRecursiveDatalogNaive, DatalogBasic, Implication, Fact,
)

from .expression_walker import (
    add_match, PatternWalker,
)

from operator import invert, and_, or_
from typing import AbstractSet
from collections import OrderedDict

class StratifiedDatalog():

    _idb_symbols = []
    _imp_symbols = []
    _negative_graph = {}

    def solve(self, expression_block):

        for rule in expression_block.expressions:
            self._add_idb_symbol(rule)

        stratifiable = self._check_stratification(expression_block)

        if stratifiable:
            return self._stratify(expression_block)
        else:
            raise NeuroLangException(
                f'The program cannot be stratifiable'
            )


    def _add_idb_symbol(self, implication):
        if hasattr(implication.consequent.functor, 'value') and implication.consequent.functor == invert:
            raise NeuroLangException(
                f'Symbol in the consequent can not be negated: {implication.consequent}'
            )

        for symbol in implication.consequent._symbols:
            self._idb_symbols.append(symbol)


    def _check_stratification(self, expression_block):
        sEval = SymbolEvaluator()
        self._imp_symbols = sEval.walk(expression_block)
        for k, v in enumerate(self._idb_symbols):
            for s in self._imp_symbols[k]:
                if hasattr(s, 'functor') and s.functor == invert:
                    name = s.args[0].functor.name
                    if name in self._idb_symbols:
                        if k in self._negative_graph:
                            rel = self._negative_graph[v]
                            rel.append(name)
                            self._negative_graph[v.name] = rel
                        else:
                            self._negative_graph[v.name] = [name]
        
        for key, values in self._negative_graph.items():
            for in_value in values:
                if in_value in self._negative_graph and key in self._negative_graph[in_value]:
                    return False

        return True

    def _stratify(self, expression_block):
        pass


class SymbolEvaluator(PatternWalker):


    @add_match(Implication)
    def eval_implication(self, expression):
        ant = self.walk(expression.antecedent)
        return ant

    
    @add_match(Fact)
    def eval_fact(self, expression):
        pass


    @add_match(FunctionApplication(Constant(invert), ...))
    def eval_function_invert_application(self, expression):
        return [expression]


    @add_match(FunctionApplication(Constant(...), ...))
    def eval_function_application_comb(self, expression):
        res = []
        for arg in expression.args:
            res.append(self.walk(arg)[0])
        
        return res


    @add_match(FunctionApplication(Symbol, ...))
    def eval_function_application(self, expression):
        return [self.walk(expression.functor)]

    
    @add_match(Symbol)
    def eval_symbol(self, expression):
        return expression


    @add_match(Constant)
    def eval_constant(self, expression):
        return expression


    @add_match(ExpressionBlock)
    def eval_expression_block(self, expression):
        eval_block = []
        for exp in expression.expressions:
            eval_exp = self.walk(exp)
            eval_block.append(eval_exp)

        return eval_block