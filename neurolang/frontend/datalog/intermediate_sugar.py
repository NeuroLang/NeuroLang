"""
Set of syntactic sugar processors at the intermediate level.
"""


from typing import AbstractSet, Callable, DefaultDict

from ... import expression_walker as ew, expressions as ir
from ...datalog.expression_processing import (
    conjunct_formulas,
    extract_logic_atoms,
)
from ...exceptions import SymbolNotFoundError
from ...expression_walker import ReplaceExpressionWalker
from ...logic import Conjunction, Implication
from ...type_system import get_args, is_leq_informative


class Column(ir.Definition):
    def __init__(self, set_symbol, column_position):
        self.set_symbol = set_symbol
        self.column_position = column_position
        self._symbols = (
            self.set_symbol._symbols | self.column_position._symbols
        )


def _has_column_sugar(conjunction, marker):
    return any(
        isinstance(arg, marker)
        for atom in extract_logic_atoms(conjunction)
        for arg in atom.args
    )


def _concatenate_flatten_tuples(*args):
    res = tuple()
    for arg in args:
        if isinstance(arg, tuple):
            res += arg
        else:
            res += (arg,)
    return res


class TranslateColumnsToAtoms(ew.PatternWalker):
    def __init__(self, symbol_table=None):
        self.symbol_table = symbol_table

    @ew.add_match(
        ir.FunctionApplication,
        lambda exp: any(isinstance(arg, Column) for arg in exp.args),
    )
    def function_application_column_sugar(self, expression):
        return self.walk(Conjunction((expression,)))

    @ew.add_match(Conjunction, lambda exp: _has_column_sugar(exp, Column))
    def conjunction_column_sugar(self, expression):
        (
            replacements,
            new_atoms,
        ) = self._obtain_new_atoms_and_column_replacements(expression)

        replaced_expression = ew.ReplaceExpressionWalker(replacements).walk(
            expression
        )
        new_formulas = replaced_expression.formulas + tuple(new_atoms)
        return self.walk(Conjunction(new_formulas))

    @ew.add_match(Implication, lambda exp: _has_column_sugar(exp, Column))
    def implication_column_sugar(self, expression):
        (
            replacements,
            new_atoms,
        ) = self._obtain_new_atoms_and_column_replacements(expression)

        replaced_expression = ew.ReplaceExpressionWalker(replacements).walk(
            expression
        )
        new_antecedent = conjunct_formulas(
            replaced_expression.antecedent, Conjunction(new_atoms)
        )

        return self.walk(
            Implication(replaced_expression.consequent, new_antecedent)
        )

    def _obtain_new_atoms_and_column_replacements(self, expression):
        sugared_columns = DefaultDict(dict)
        for atom in extract_logic_atoms(expression):
            for arg in atom.args:
                if isinstance(arg, Column):
                    sugared_columns[arg.set_symbol][arg] = ir.Symbol.fresh()

        new_atoms = []
        replacements = {}
        for k, v in sugared_columns.items():
            k_constant = ew.ReplaceSymbolsByConstants(self.symbol_table).walk(
                k
            )
            args = (
                v.get(Column(k, ir.Constant[int](i)), ir.Symbol.fresh())
                for i in range(k_constant.value.arity)
            )
            new_atoms.append(k(*args))
            replacements.update(v)
        return replacements, tuple(new_atoms)


class SelectByFirstColumn(ir.Definition):
    def __init__(self, set_symbol, selector):
        self.set_symbol = set_symbol
        self.selector = selector
        self._symbols = self.set_symbol._symbols | self.selector._symbols


class TranslateSelectByFirstColumn(ew.PatternWalker):
    @ew.add_match(
        ir.FunctionApplication,
        lambda exp: any(
            isinstance(arg, SelectByFirstColumn) for arg in exp.args
        ),
    )
    def function_application_column_sugar(self, expression):
        return self.walk(Conjunction((expression,)))

    @ew.add_match(
        Conjunction, lambda exp: _has_column_sugar(exp, SelectByFirstColumn)
    )
    def conjunction_column_sugar(self, expression):
        (
            replacements,
            new_atoms,
        ) = self._obtain_new_atoms_and_select_by_first_column_replacements(
            expression
        )

        replaced_expression = ew.ReplaceExpressionWalker(replacements).walk(
            expression
        )
        new_formulas = replaced_expression.formulas + tuple(new_atoms)
        return self.walk(Conjunction(new_formulas))

    @ew.add_match(Implication(SelectByFirstColumn, ...))
    def implication_select_by_left_head(self, expression):
        consequent = expression.consequent
        head_fresh = ir.Symbol.fresh()
        new_consequent = consequent.set_symbol(consequent.selector, head_fresh)

        replacements = {consequent: new_consequent}
        for atom in extract_logic_atoms(expression.consequent):
            if len(atom.args) == 1 and self._theoretical_arity(atom) == 2:
                replacements[atom] = atom.functor((head_fresh, atom.args[0]))

        new_rule = Implication(
            new_consequent,
            ReplaceExpressionWalker(replacements).walk(expression.antecedent),
        )

        return new_rule

    def _theoretical_arity(self, atom):
        functor = atom.functor
        if isinstance(functor, ir.Symbol):
            try:
                functor = self.symbol_table[functor]
            except KeyError:
                raise SymbolNotFoundError(f"Symbol {functor} not found")
        if is_leq_informative(Callable, functor.type):
            arity = len(get_args(functor.type))
        elif is_leq_informative(AbstractSet, functor.type):
            arity = len(get_args(get_args(functor.type)[0]))
        else:
            arity = None
        return arity

    @ew.add_match(
        Implication, lambda exp: _has_column_sugar(exp, SelectByFirstColumn)
    )
    def implication_column_sugar(self, expression):
        (
            replacements,
            new_atoms,
        ) = self._obtain_new_atoms_and_select_by_first_column_replacements(
            expression
        )

        replaced_expression = ew.ReplaceExpressionWalker(replacements).walk(
            expression
        )
        new_antecedent = conjunct_formulas(
            replaced_expression.antecedent, Conjunction(new_atoms)
        )

        return self.walk(
            Implication(replaced_expression.consequent, new_antecedent)
        )

    def _obtain_new_atoms_and_select_by_first_column_replacements(
        self, expression
    ):
        sugared_columns = DefaultDict(dict)
        for atom in extract_logic_atoms(expression):
            for arg in atom.args:
                if isinstance(arg, SelectByFirstColumn):
                    sugared_columns[arg] = ir.Symbol.fresh()

        new_atoms = []
        for k, v in sugared_columns.items():
            new_atom = k.set_symbol(k.selector, v)
            new_atoms.append(new_atom)
        return sugared_columns, tuple(new_atoms)
