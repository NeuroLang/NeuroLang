import numpy as np
import pandas as pd
import operator as op
from ... import expressions, logic
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
