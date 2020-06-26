import operator as op

import numpy as np
import pandas as pd

from ... import expressions, logic
from ...datalog import Fact
from ...expressions import Constant, Symbol
from ...logic import Conjunction, Implication, Union
from .. import weighted_model_counting
from ..cplogic.program import CPLogicProgram


def test_one():
    random = np.random.RandomState(0)
    P = pd.DataFrame(
        np.arange(int(10)), columns=['x']
    )
    P['prob'] = random.rand(len(P))
    sP = expressions.Symbol('P')
    sQ = expressions.Symbol('Q')
    x = expressions.Symbol('x')
    y = expressions.Symbol('y')
    zero = expressions.Constant(0)
    cplp = CPLogicProgram()
    cplp.add_probabilistic_facts_from_tuples(
        sP, P[['prob', 'x']].itertuples(index=False, name=None)
    )

    query = logic.Implication(
        sQ(y),
        logic.Conjunction((sP(x), sP(y), expressions.Constant(op.eq)(x, zero)))
    )

    res = weighted_model_counting.solve_succ_query(
        query, cplp
    )

    print(res)


P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
H = Symbol("H")
A = Symbol("A")
B = Symbol("B")
C = Symbol("C")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")

a = Constant("a")
b = Constant("b")


def test_deterministic():
    """
    We define the program

        P(x) <- Q(x)

    And we expect the provenance set resulting from the
    marginalisation of P(x) to be

        _p_ | x
        ====|===
        1.0 | a
        1.0 | b

    """
    code = Union((Fact(Q(a)), Fact(Q(b)), Implication(P(x), Q(x)),))
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    query_pred = P(x)
    query_pred = (
        cpl_program.intensional_database()
        [query_pred.functor]
        .formulas[0]
    )
    result = weighted_model_counting.solve_succ_query(query_pred, cpl_program)
    expected = testing.make_prov_set([(1.0, "a"), (1.0, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_simple_bernoulli():
    """
    We define the program

        P(a) : 0.7 <- T
        P(b) : 0.8 <- T

    And expect the provenance set resulting from the
    marginalisation of P(x) to be

        _p_ | x
        ====|===
        0.7 | a
        0.8 | b

    """
    code = Union(())
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    cpl_program.add_probabilistic_facts_from_tuples(
        P, {(0.7, "a"), (0.8, "b")}
    )
    result = weighted_model_counting.solve_succ_query(P(x), cpl_program)
    expected = testing.make_prov_set([(0.7, "a"), (0.8, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)