from .. import expression_walker
from .. import symbols_and_types
from .. import neurolang as nl

S_ = nl.Symbol
C_ = nl.Constant


def test_symbol_table_as_parameter():
    symbol_table = symbols_and_types.TypedSymbolTable()
    solver = expression_walker.SymbolTableEvaluator(symbol_table)
    s = S_('S1')
    c = C_[str]('S')
    symbol_table[s] = c
    assert solver.symbol_table[s] is c
