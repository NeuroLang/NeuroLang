from pytest import raises

from .. import neurolang as nl
from .. import solver
from typing import Set, Tuple


def test_assignment_values():
    command = '''
        a = 1
        b = "a"
        c = 1.2
        d = 1 + 1.2 + 1.
        e = 1 + 2 * 5.
        f = 2. ** 3.
        g = f
        h = double(f)
        i = 5 - 2
        l = 5. / 2.
        m = 5 // 2
    '''

    def double(v: float)->float:
        return 2 * v

    def bad_double(v: float)->float:
        return {}

    nli = nl.NeuroLangInterpreter(functions=[double, bad_double])
    ast = nl.parser(command)
    nli.compile(ast)

    assert type(nli.symbol_table['a'].value) == int
    assert nli.symbol_table['a'].value == 1
    assert nli.symbol_table['b'].value == "a"
    assert nli.symbol_table['c'].value == 1.2
    assert nli.symbol_table['d'].value == 3.2
    assert nli.symbol_table['e'].value == 11
    assert nli.symbol_table['f'].value == 8.
    assert nli.symbol_table['g'].value == 8.
    assert nli.symbol_table['h'].value == 16.
    assert nli.symbol_table['i'].value == 3
    assert nli.symbol_table['l'].value == 2.5
    assert nli.symbol_table['m'].value == 2

    with raises(nl.NeuroLangTypeException):
        nli.compile(nl.parser('t = double("a")'))

    with raises(nl.NeuroLangTypeException):
        nli.compile(nl.parser('t = bad_double(1.)'))

    with raises(nl.NeuroLangTypeException):
        nli.compile(nl.parser('t = a("a")'))


def test_tuples():
    command = '''
        a = (1, "a")
        b = a[0]
    '''
    nli = nl.NeuroLangInterpreter()
    ast = nl.parser(command)
    nli.compile(ast)

    assert nli.symbol_table['a'].type == Tuple[int, str]
    assert nli.symbol_table['a'].value == (1, "a")

    assert nli.symbol_table['b'].type == int
    assert nli.symbol_table['b'].value == 1

    with raises(nl.NeuroLangTypeException):
        nli.compile(nl.parser('a[2]'))


def test_queries():

    class FourInts(int, solver.FiniteDomain):
        pass

    class FourIntsSetSolver(solver.SetBasedSolver):
        type_name = 'four_int'
        type = FourInts

        def predicate_equal_to(self, value: int)->FourInts:
            return FourInts(value)

        def predicate_singleton_set(self, value: int)->Set[FourInts]:
            return {FourInts(value)}

    nli = nl.NeuroLangInterpreter(
        category_solvers=[FourIntsSetSolver()],
    )

    script = '''
    one is a four_int equal_to 1
    two is a four_int equal_to 2
    three is a four_int equal_to 3
    oneset are four_ints singleton_set 1
    oneset_ are four_ints in oneset
    onetwo are four_ints singleton_set 1 or singleton_set 2
    twoset are four_ints in onetwo and singleton_set 2
    twothree are four_ints not in oneset
    '''

    ast = nl.parser(script)
    nli.compile(ast)

    assert nli.symbol_table['one'].value == 1
    assert nli.symbol_table['one'].type == FourInts
    assert nli.symbol_table['two'].value == 2
    assert nli.symbol_table['three'].value == 3
    assert nli.symbol_table['oneset'].value == {1}
    assert nli.symbol_table['oneset_'].value == {1}
    assert nli.symbol_table['onetwo'].value == {1, 2}
    assert nli.symbol_table['twoset'].value == {2}
    assert nli.symbol_table['twothree'].value == {2, 3}

    with raises(nl.NeuroLangException):
        nli.compile(nl.parser("fail is a four_int singleton_set 1"))

    with raises(nl.NeuroLangException):
        nli.compile(nl.parser("fail are four_int singleton_set 1"))

    with raises(nl.NeuroLangException):
        nli.compile(nl.parser("fail is a four_ints singleton_set 1"))


def test_mix_queries_and_algebra():
    command = '''
        foo is a String foo
        bar = foo
    '''
    nli = nl.NeuroLangInterpreter()
    nli.compile(nl.parser(command))

    assert nli.symbol_table['foo'].value == "foo"
    assert nli.symbol_table['bar'].value == "foo"
