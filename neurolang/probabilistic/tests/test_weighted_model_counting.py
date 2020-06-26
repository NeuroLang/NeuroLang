import operator as op

import numpy as np
import pandas as pd
from neurolang import expression_walker as ew
from neurolang import expressions, logic
from neurolang import relational_algebra as ras
from neurolang import relational_algebra_provenance as rap
from neurolang.probabilistic import weighted_model_counting
from neurolang.probabilistic.cplogic.program import CPLogicProgram
from neurolang.utils.relational_algebra_set import pandas as ra
from pysdd import sdd


def test_one():
    random = np.random.RandomState(0)
    P = pd.DataFrame(
        np.arange(int(500)), columns=['x']
    )
    P['prob'] = random.rand(len(P))
    sP = expressions.Symbol('P')
    x = expressions.Symbol('x')
    y = expressions.Symbol('y')
    zero = expressions.Constant(0)
    cplp = CPLogicProgram()
    cplp.add_probabilistic_facts_from_tuples(
        sP, P[['prob', 'x']].itertuples(index=False, name=None)
    )
    res = weighted_model_counting.solve_succ_query(
        logic.Conjunction((sP(zero), sP(y))), cplp
    )

    print(res)
