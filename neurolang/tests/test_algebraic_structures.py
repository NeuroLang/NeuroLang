from .. import neurolang as nl


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

    associative_command = '''
        a = (element1 {0} element2) {0} element3
        b = element1 {0} (element2 {0} element3)
    '''.format(operation)
    nli.evaluate(nl.parser(associative_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    commutative_command = '''
        a = element1 {0} element2
        b = element2 {0} element1
    '''.format(operation)
    nli.evaluate(nl.parser(commutative_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    identity_command = '''
        a = element1 {0} null
    '''.format(operation)
    nli.evaluate(nl.parser(identity_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['element1'].value

    inverse_command = '''
        a = element1 {0} element1
    '''.format(inverse_operation)
    nli.evaluate(nl.parser(inverse_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['null'].value


def test_monoid_under_multiplication():
    nli = nl.NeuroLangInterpreter()

    associative_command = '''
        a = (1 * 2) * 3
        b = 1 * (2 * 3)
    '''
    nli.evaluate(nl.parser(associative_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    identity_command = '''
        b = 42
        a = b * 1
    '''
    nli.evaluate(nl.parser(identity_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    identity_command = '''
        b = 42
        a = 1 * b
    '''
    nli.evaluate(nl.parser(identity_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value


def test_multiplication_is_distributive_with_respect_to_addition():
    nli = nl.NeuroLangInterpreter()

    left_distributy_command = '''
        a = 2 * (3 + 4)
        b = (2 * 3) + (2 * 4)
    '''
    nli.evaluate(nl.parser(left_distributy_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    right_distributy_command = '''
        a = (3 + 4) * 2
        b = (3 * 2) + (4 * 2)
    '''
    nli.evaluate(nl.parser(right_distributy_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value
