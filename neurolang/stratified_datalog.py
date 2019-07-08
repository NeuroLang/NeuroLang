from .expressions import (
    FunctionApplication, Constant, NeuroLangException,
    Symbol, ExpressionBlock, Expression,
)

from .solver_datalog_naive import (
    Implication, Fact,
)

from .expression_walker import (
    add_match, PatternWalker,
)

from operator import invert, and_, or_

class StratifiedDatalog():

    _idb_symbols = []
    _imp_symbols = []
    _negative_graph = {}


    def solve(self, expression_block):
        """Main function. Given an expression block, calculates 
        and returns the stratified program if possible.

        Parameters
        ----------
        expression_block : ExpressionBlock
            The expression block defining the program to be solved.

        Returns
        -------
        ExpressionBlock
            The stratified version of the program.
        """
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
        """Given an implication, this function validates that the consequent
        is not denied and saves all the symbols of the intentional database.

        Parameters
        ----------
        expression_block : ExpressionBlock
            The implication to be checked.
        """
        if (
                hasattr(implication.consequent.functor, 'value') and 
                implication.consequent.functor == invert
        ):
            raise NeuroLangException(
                f'Symbol in the consequent can not be \
                    negated: {implication.consequent}'
            )

        for symbol in implication.consequent._symbols:
            self._idb_symbols.append(symbol)


    def _check_stratification(self, expression_block):
        """Given an expression block, this function construct 
        a graph of negative relations and check for stratifiability.

        Parameters
        ----------
        expression_block : ExpressionBlock
            The expression block defining the program to be solved.

        Returns
        -------
        bool
            True if the program is or can be stratified. False otherwise.
        """
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
                if (
                    in_value in self._negative_graph and 
                    key in self._negative_graph[in_value]
                ):
                    return False

        return True


    def _stratify(self, expression_block):
        """Given an expression block, this function reorder it
        in an stratified way. No change is made if it is not necessary. 

        Parameters
        ----------
        expression_block : ExpressionBlock
            The expression block defining the program to be solved.

        Returns
        -------
        ExpressionBlock
            The stratified version of the program.
        """
        stratified_index = []

        moved = 0
        for key, _ in enumerate(expression_block.expressions):
            imp_symbols = self._imp_symbols[key]
            new_pos = key
            for ith_idb in range(key+1, len(self._idb_symbols)):
                for imp_symbol in imp_symbols:
                    if (
                        hasattr(imp_symbol, 'functor') and
                        imp_symbol.functor == invert
                    ):
                        if (
                            self._idb_symbols[ith_idb] == 
                            imp_symbol.args[0].functor
                        ):
                            new_pos += 1
            
            if new_pos != key:
                moved += 1
                stratified_index.append(new_pos)
            else:
                stratified_index.append(new_pos-moved)

        new_order = sorted(zip(stratified_index, expression_block.expressions))
        stratified_rules = [x for i, x in new_order]

        return ExpressionBlock(tuple(stratified_rules))


class SymbolEvaluator(PatternWalker):


    @add_match(Implication)
    def eval_implication(self, expression):
        ant = self.walk(expression.antecedent)
        return ant


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