from ....expressions import Symbol, Constant
from ..program import CPLogicProgram
from ..problog import cplogic_to_problog

P = Symbol("P")


def test_convert_cpl_to_pl():
    cpl_program = CPLogicProgram()
    cpl_program.add_probabilistic_facts_from_tuples(
        P, {(0.2, "a"), (1.0, "b")}
    )
    pl = cplogic_to_problog(cpl_program)
