from ....logic import Conjunction, Constant, Implication, Union
from ... import dalvi_suciu_lift
from ...cplogic.program import Condition, CPLogicProgram
from ...expressions import PROB, ProbabilisticQuery, Symbol
from ..program import CausalProgram
from .test_expression_processing import CausalIntervention


def test_simpsons_paradox():
    '''         | Drug                            | No drug
    ___________________________________________________________________
    Men         | 81 out of 87 recovered (93%)    | 234 out of 270 recovered (87%)
    Women       | 192 out of 263 recovered (73%)  | 55 out of 80 recovered (69%)
    Combined    | 273 out of 350 recovered (78%)  | 289 out of 350 recovered (83%)
    '''

    Gender = Symbol("Gender")
    Drug = Symbol("Drug")
    Recovery = Symbol("Recovery")

    g = Symbol("g")
    d = Symbol("d")
    r = Symbol("r")

    U_Recovery = Symbol("U_Recovery")

    ans = Symbol("ans")
    P = Symbol("P")

    men = Constant("men")

    cpl_program = CPLogicProgram()

    cpl_program.add_probabilistic_choice_from_tuples(
        Gender, {
            (0.51, "men"),
            (0.49, "women")
        }
    )

    cpl_program.add_probabilistic_choice_from_tuples(
        U_Recovery, {
            (81/700, "men", "drug", "recovery"),
            (6/700, "men", "drug", "no recovery"),
            (234/700, "men", "no drug", "recovery"),
            (36/700, "men", "no drug", "no recovery"),
            (192/700, "women", "drug", "recovery"),
            (71/700, "women", "drug", "no recovery"),
            (55/700, "women", "no drug", "recovery"),
            (25/700, "women", "no drug", "no recovery"),
        }
    )

    code = Union(
        (
            Implication(
                Drug(g, d),
                Conjunction(
                    (Gender(g), U_Recovery(g, d, r))
                )),
            Implication(
                Recovery(g, d, r),
                Conjunction(
                    (Drug(g, d), U_Recovery(g, d, r), Gender(g))
                )),
            Implication(
                ans(r, ProbabilisticQuery(PROB, (r,))), Condition(
                    Recovery(g, d, r),
                    Conjunction(
                        (CausalIntervention((Gender(men),)),)
                    )
                )
            )
        )
    )

    new_program, new_query = CausalProgram().rewrite_program(code)
    new_program.remove(new_query)

    cpl_program.walk(Union(tuple(new_program)))
    cpl_program.walk(new_query)

    res = dalvi_suciu_lift.solve_marg_query(new_query, cpl_program)
    a = 1

