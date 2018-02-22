from .. import neurolang as nl
from .. import solver

from typing import Set


def test_xxxx():
    class FourInts(int, solver.FiniteDomain):
        pass

    class FourIntsSetSolver(solver.SetBasedSolver):
        type_name = 'four_int'
        type = FourInts

        def predicate_equal_to(self, value: int)->FourInts:
            return FourInts(value)

        def predicate_singleton_set(self, value: int)->Set[FourInts]:
            return {FourInts(value)}

    construction_script = '''
        one is a four_int equal_to 1
        two is a four_int equal_to 2
        three is a four_int equal_to 3
        four is a four_int equal_to 4
        one_two_set are four_ints singleton_set 1 or singleton_set 2
        two_three_set are four_ints singleton_set 1 or singleton_set 2
        three_four_set are four_ints singleton_set 3 or singleton_set 4
        one_two_three_four are four_ints singleton_set 1 or singleton_set 2 or singleton_set 3 or singleton_set 4
        one_set are four_ints singleton_set 1
        empty_set are four_ints singleton_set 42
        full_set are four_ints singleton_set 1 or singleton_set 2 or singleton_set 3 or singleton_set 4
    '''
    # nli.compile(nl.parser('full_set = one_two_three_four_set'))

    nli = nl.NeuroLangInterpreter(
        category_solvers=[FourIntsSetSolver()],
    )
    nli.compile(nl.parser(construction_script))

    ####
    # Actually testing somethng
    ##

    commutative_or_command = '''
    a are four_ints in one_two_set or in three_four_set
    b are four_ints in three_four_set or in one_two_set
    '''
    nli.compile(nl.parser(commutative_or_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    commutative_and_command = '''
    a are four_ints in one_two_set and in two_three_set
    b are four_ints in two_three_set and in one_two_set
    '''
    nli.compile(nl.parser(commutative_and_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    # associative_and_command = '''
    # a are four_ints (in one_two_set and in two_three_set) and in one_two_three_four_set
    # b are four_ints in one_two_set and (in two_three_set and in one_two_three_four_set)
    # '''
    associative_and_command = '''
    xx are four_ints in one_two_set and in two_three_set
    yy are four_ints in two_three_set and in one_two_three_four_set
    a are four_ints in xx and in one_two_three_four_set
    b are four_ints in one_two_set and in yy
    '''
    nli.compile(nl.parser(associative_and_command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    # associative_or_command = '''
    # a are four_ints (in one_two_set or in two_three_set) or in one_two_three_four_set
    # b are four_ints in one_two_set or (in two_three_set or in one_two_three_four_set)
    # '''
    # nli.compile(nl.parser(associative_or_command))
    # assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    # distributive_or_of_ands_command = '''
    # a are four_ints in one_two_set or (in two_three_set and in three_four_set)
    # b are four_ints (in one_two_set and in two_three_set) or (in one_two_set and in one_two_three_four_set)
    # '''
    # nli.compile(nl.parser(distributive_or_of_ands_command))
    # assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    # distributive_and_of_ors_command = '''
    # a are four_ints in one_two_set and (in two_three_set or in three_four_set)
    # b are four_ints (in one_two_set or in two_three_set) and (in one_two_set or in one_two_three_four_set)
    # '''
    # nli.compile(nl.parser(distributive_and_of_ors_command))
    # assert nli.symbol_table['a'].value == nli.symbol_table['b'].value

    # identity_command = '''
    # a are four_ints in one_set or in empty_set
    # b are four_ints in one_set and in full_set
    # '''
    # nli.compile(nl.parser(identity_command))
    # assert nli.symbol_table['a'].value == nli.symbol_table['one_set'].value
    # assert nli.symbol_table['b'].value == nli.symbol_table['one_set'].value
