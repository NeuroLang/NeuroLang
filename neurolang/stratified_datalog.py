'''Implementation of negation by stratification'''

from .expressions import (
    FunctionApplication,
    Constant,
    Symbol,
    ExpressionBlock,
)

from .solver_datalog_naive import (
    Implication,
)

from .expression_walker import (
    add_match,
    PatternWalker,
)

from .exceptions import (NeuroLangDataLogNonStratifiable, NeuroLangException)

from operator import invert


class StratifiedDatalog():
    '''Main class for solving stratifications in Datalog. Given an
    expression block, this class checks that there is a stratification
    and if so, calculates it, returning a new expression block with
    the respective stratified program.'''

    def __init__(self):
        self._idb_symbols = []
        self._imp_symbols = []
        self._negative_graph = {}

    def stratify(self, expression_block: ExpressionBlock):
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
        self._idb_symbols = []
        self._imp_symbols = []
        self._negative_graph = {}

        for rule in expression_block.expressions:
            self._add_idb_symbol(rule)

        stratifiable = self._check_stratification(expression_block)

        if stratifiable:
            return self._solve(expression_block)
        else:
            raise NeuroLangDataLogNonStratifiable(
                f'The program cannot be stratifiable'
            )

    def _add_idb_symbol(self, implication: Implication):
        """Given an implication, this function validates that the consequent
        is not denied and saves all the symbols of the intentional database.

        Parameters
        ----------
        implication : Implication
            The implication to be checked.
        """
        if self._is_negation(implication.consequent):
            raise NeuroLangException(
                f'Symbol in the consequent can not be \
                    negated: {implication.consequent}'
            )

        self._idb_symbols.append(implication.consequent)

    def _check_stratification(self, expression_block: ExpressionBlock):
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
        cons_symb = ConsequentSymbols()
        self._imp_symbols = cons_symb.walk(expression_block)
        self._create_graph()

        for key, values in self._negative_graph.items():
            for in_value in values:
                if (
                    in_value in self._negative_graph and
                    key in self._negative_graph[in_value]
                ):
                    return False

        return True

    def _create_graph(self):
        """This function iterate over every expression in the main block
        and create a graph with the negative symbols.
        """
        for k, v in enumerate(self._idb_symbols):
            for s in self._imp_symbols[k]:
                if (
                    StratifiedDatalog._is_negation(s) and
                    s.args[0] in self._idb_symbols
                ):
                    name = hash(v)
                    self._set_negative_graph(name, s)

    def _set_negative_graph(self, name, s):
        if name in self._negative_graph:
            rel = self._negative_graph[name]
            rel.append(hash(s.args[0]))
            self._negative_graph[name] = rel
        else:
            self._negative_graph[name] = [hash(s.args[0])]

    def _solve(self, expression_block: ExpressionBlock):
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
        new_block = ExpressionBlock(expression_block.expressions)
        temp_block = ExpressionBlock(())

        while new_block.expressions != temp_block.expressions:
            temp_block.expressions = new_block.expressions
            stratified_index = self._calc_index(temp_block)

            new_block = self._reorder(stratified_index, temp_block)

        return new_block

    def _calc_index(self, temp_block):
        stratified_index = []
        moved = 0
        for key, _ in enumerate(temp_block.expressions):
            new_pos = self._calculate_new_position(key)
            if new_pos == key and key <= max(stratified_index, default=0):
                stratified_index.append(new_pos - moved)
            elif new_pos == key and key > max(stratified_index):
                stratified_index.append(new_pos)
            else:
                moved += 1
                stratified_index.append(new_pos)

        return stratified_index

    def _reorder(self, new_positions, block):
        """Given a list of position and an expression block, this function
        reorder the block and also update the order of the global variables.

        Parameters
        ----------
        new_positions : list
            The list with the new positions
        expression_block : ExpressionBlock
            The expression block defining the program to be reordered.

        Returns
        -------
        ExpressionBlock
            The reordered version of the program.
        """

        new_block_order = sorted(zip(new_positions, block.expressions))
        reordered_block = [x for _, x in new_block_order]
        new_block = ExpressionBlock(tuple(reordered_block))

        new_idb_order = sorted(zip(new_positions, self._idb_symbols))
        self._idb_symbols = [x for _, x in new_idb_order]

        new_imp_order = sorted(zip(new_positions, self._imp_symbols))
        self._imp_symbols = [x for _, x in new_imp_order]

        return new_block

    def _calculate_new_position(self, actual_position):
        """Given the actual position of the expression, this function return a
        new position that try accomplish the constraints of the stratification.
        No change is made if it is not necessary.

        Parameters
        ----------
        actual_position : int
            The actual position of the expression in the main block.

        Returns
        -------
        int
            The new position of the expression.
        """
        imp_symbols = self._imp_symbols[actual_position]
        new_pos = actual_position
        for ith_idb in range(actual_position + 1, len(self._idb_symbols)):
            for imp_symbol in imp_symbols:
                if (
                    StratifiedDatalog._is_negation(imp_symbol) and
                    self._idb_symbols[ith_idb] == imp_symbol.args[0]
                ):
                    new_pos = ith_idb

        return new_pos

    @staticmethod
    def _is_negation(expression):
        if isinstance(
            expression, FunctionApplication
        ) and expression.functor == invert:
            return True

        return False


class ConsequentSymbols(PatternWalker):
    '''This class implements a PatternWalker who is in charge of going through
    an expression block and obtaining the symbols present in the antecedent of
    each one of the available implications.'''

    @add_match(Implication)
    def eval_implication(self, expression):
        return self.walk(expression.antecedent)

    @add_match(FunctionApplication(Constant(invert), ...))
    def eval_function_invert_application(self, expression):
        return [expression]

    @add_match(FunctionApplication(Constant(...), ...))
    def eval_function_application_comb(self, expression):
        res = []
        for arg in expression.args:
            out = self.walk(arg)
            res += out

        return res

    @add_match(FunctionApplication(Symbol, ...))
    def eval_function_application(self, expression):
        return [expression]

    @add_match(Symbol)
    def eval_symbol(self, expression):
        return [expression]

    @add_match(Constant)
    def eval_constant(self, expression):
        return []

    @add_match(ExpressionBlock)
    def eval_expression_block(self, expression):
        eval_block = []
        for exp in expression.expressions:
            eval_exp = self.walk(exp)
            eval_block.append(eval_exp)

        return eval_block
