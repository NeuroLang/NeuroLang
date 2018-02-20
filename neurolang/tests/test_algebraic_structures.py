from .. import neurolang as nl


def test_abelian_group_under_addition():
    nli = nl.NeuroLangInterpreter()

    associative_command = '''
        a = (1 + 2) + 3
        b = 1 + (2 + 3)
    '''
    nli.evaluate(nl.parser(associative_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    commutative_command = '''
        a = 1 + 2
        b = 2 + 1
    '''
    nli.evaluate(nl.parser(commutative_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    identity_command = '''
        b = 42
        a = b + 0
    '''
    nli.evaluate(nl.parser(identity_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    inverse_command = '''
        a = 42 - 42
    '''
    nli.evaluate(nl.parser(inverse_command))
    assert nli.symbol_table['a'].value == 0


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
