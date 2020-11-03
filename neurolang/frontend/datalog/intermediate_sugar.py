"""
Set of syntactic sugar processors at the intermediate level.
"""


from typing import DefaultDict

from ... import expression_walker as ew, expressions as ir
from ...datalog.expression_processing import extract_logic_atoms
from ...logic import Conjunction, Implication


class Column(ir.Definition):
    def __init__(self, set_symbol, column_position):
        self.set_symbol = set_symbol
        self.column_position = column_position
        self._symbols = (
            self.set_symbol._symbols | self.column_position._symbols
        )


def has_column_sugar(conjunction):
    return any(
        isinstance(arg, Column)
        for atom in extract_logic_atoms(conjunction)
        for arg in atom.args
    )


class TranslateColumnsToAtoms(ew.ExpressionWalker):
    def __init__(self, symbol_table=None):
        self.symbol_table = symbol_table

    @ew.add_match(
        ir.FunctionApplication,
        lambda exp: any(isinstance(arg, Column) for arg in exp.args),
    )
    def function_application(self, expression):
        return self.walk(Conjunction((expression,)))

    @ew.add_match(Conjunction, has_column_sugar)
    def conjunction_column_sugar(self, expression):
        replacements, new_atoms = self._obtain_new_atoms_and_replacements(
            expression
        )

        replaced_expression = ew.ReplaceExpressionWalker(replacements).walk(
            expression
        )
        new_formulas = replaced_expression.formulas + tuple(new_atoms)
        return self.walk(Conjunction(new_formulas))

    @ew.add_match(Implication, has_column_sugar)
    def implication_column_sugar(self, expression):
        replacements, new_atoms = self._obtain_new_atoms_and_replacements(
            expression
        )

        replaced_expression = ew.ReplaceExpressionWalker(replacements).walk(
            expression
        )
        new_antecedent = replaced_expression.antecedent.formulas + tuple(
            new_atoms
        )
        return self.walk(
            Implication(replaced_expression.consequent, new_antecedent)
        )

    def _obtain_new_atoms_and_replacements(self, expression):
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
        return replacements, new_atoms
