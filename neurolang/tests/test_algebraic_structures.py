from .. import neurolang as nl

associative_command = '''
    a = (element1 {0} element2) {0} element3
    b = element1 {0} (element2 {0} element3)
'''

commutative_command = '''
    a = element1 {0} element2
    b = element2 {0} element1
'''

identity_command = '''
    a = element1 {0} null
    b = null {0} element1
'''

inverse_command = '''
    a = element1 {0} element1
'''

left_distributy_command = '''
    a = 2 {0} (3 {1} 4)
    b = (2 {0} 3) {1} (2 {0} 4)
'''

right_distributy_command = '''
    a = (3 {1} 4) {0} 2
    b = (3 {0} 2) {1} (4 {0} 2)
'''

# left_distributy_command = '''
#     a = 2 * (3 + 4)
#     b = (2 * 3) + (2 * 4)
# '''

# right_distributy_command = '''
#     a = (3 + 4) * 2
#     b = (3 * 2) + (4 * 2)
# '''


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

    nli.evaluate(nl.parser(associative_command.format(operation)))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    nli.evaluate(nl.parser(commutative_command.format(operation)))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    nli.evaluate(nl.parser(identity_command.format(operation)))
    assert nli.symbol_table['a'].value == nli.symbol_table['element1'].value
    assert nli.symbol_table['b'].value == nli.symbol_table['element1'].value

    nli.evaluate(nl.parser(inverse_command.format(inverse_operation)))
    assert nli.symbol_table['a'].value == nli.symbol_table['null'].value


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

    nli.evaluate(nl.parser(associative_command.format(operation)))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    nli.evaluate(nl.parser(identity_command.format(operation)))
    assert nli.symbol_table['a'].value == nli.symbol_table['element1'].value
    assert nli.symbol_table['b'].value == nli.symbol_table['element1'].value


def test_multiplication_is_distributive_with_respect_to_addition(
        first_order_operation='*',
        second_order_operation='+'
):
    nli = nl.NeuroLangInterpreter()

    command = left_distributy_command.format(first_order_operation,
                                             second_order_operation)
    nli.evaluate(nl.parser(command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    command = right_distributy_command.format(first_order_operation,
                                              second_order_operation)
    nli.evaluate(nl.parser(command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value
