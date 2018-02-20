from .. import neurolang as nl

associative_command = '''
    a = (element1 {op} element2) {op} element3
    b = element1 {op} (element2 {op} element3)
'''

commutative_command = '''
    a = element1 {op} element2
    b = element2 {op} element1
'''

identity_command = '''
    a = element1
    b = element1 {op} null
'''

inverse_command = '''
    a = null
    b = element1 {op} element1
'''

left_distributy_command = '''
    a = element1 {op_mul} (element2 {op_add} element3)
    b = (element1 {op_mul} element2) {op_add} (element1 {op_mul} element3)
'''

right_distributy_command = '''
    a = (element2 {op_add} element3) {op_mul} element1
    b = (element2 {op_mul} element1) {op_add} (element3 {op_mul} element1)
'''


def test_abelian_group_under_addition(
    type_=int,
    elements=(1, 2, 3), null_element=0,
    operation='+', inverse_operation='-'
):

    symbols = {
        'element{}'.format(i + 1): nl.TypedSymbol(type_, e)
        for i, e in enumerate(elements)
    }
    symbols['null'] = nl.TypedSymbol(type_, null_element)
    nli = nl.NeuroLangInterpreter(symbols=symbols, types=[(type_, 'dummy')])

    nli.evaluate(nl.parser(associative_command.format(op=operation)))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    nli.evaluate(nl.parser(commutative_command.format(op=operation)))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    nli.evaluate(nl.parser(identity_command.format(op=operation)))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    nli.evaluate(nl.parser(inverse_command.format(op=inverse_operation)))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value


def test_monoid_under_multiplication(
        type_=int,
        elements=(1, 2, 3), null_element=1,
        operation='*', inverse_operation='/'
):
    symbols = {
        'element{}'.format(i + 1): nl.TypedSymbol(type_, e)
        for i, e in enumerate(elements)
    }
    symbols['null'] = nl.TypedSymbol(type_, null_element)
    nli = nl.NeuroLangInterpreter(symbols=symbols, types=[(type_, 'dummy')])

    nli.evaluate(nl.parser(associative_command.format(op=operation)))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    nli.evaluate(nl.parser(identity_command.format(op=operation)))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value


def test_multiplication_is_distributive_with_respect_to_addition(
        type_=int,
        elements=(1, 2, 3), null_element=1,
        add='+', mul='*'
):
    symbols = {
        'element{}'.format(i + 1): nl.TypedSymbol(type_, e)
        for i, e in enumerate(elements)
    }
    nli = nl.NeuroLangInterpreter(symbols=symbols, types=[(type_, 'dummy')])

    command = left_distributy_command.format(op_add=add, op_mul=mul)
    nli.evaluate(nl.parser(command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    command = right_distributy_command.format(op_add=add, op_mul=mul)
    nli.evaluate(nl.parser(command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value
