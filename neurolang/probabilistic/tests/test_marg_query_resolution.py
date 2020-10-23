import pytest

from ...expressions import Constant, Symbol
from ...logic import Implication
from ...relational_algebra import ColumnStr, NamedRelationalAlgebraFrozenSet
from ...relational_algebra_provenance import ProvenanceAlgebraSet
from .. import dichotomy_theorem_based_solver, weighted_model_counting
from ..cplogic import testing
from ..cplogic.program import CPLogicProgram
from ..expressions import PROB, Condition, ProbabilisticQuery

ans = Symbol("ans")
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
c = Constant("c")


@pytest.fixture(
    params=((weighted_model_counting, dichotomy_theorem_based_solver)),
    ids=["SDD-WMC", "dichotomy-Safe query"],
)
def solver(request):
    return request.param


def test_marg_query_ground_conditioning(solver):
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        R,
        [
            (0.2, "a", "a"),
            (0.4, "b", "a"),
            (0.1, "b", "c"),
            (0.9, "c", "a"),
        ],
    )
    cpl.add_probabilistic_choice_from_tuples(
        Z,
        [
            (0.2, "c"),
            (0.5, "b"),
            (0.3, "a"),
        ],
    )
    query = Implication(
        Q(x, ProbabilisticQuery(PROB, (x,))), Condition(Z(x), R(x, a))
    )
    cpl.walk(query)
    result = solver.solve_marg_query(query, cpl)
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"),
            [
                (0.5, "b"),
                (0.3, "a"),
                (0.2, "c"),
            ],
        ),
        ColumnStr("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)
