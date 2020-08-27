from ...datalog.expressions import Fact
from ...expressions import Constant, Symbol
from ...logic import TRUE, Conjunction, Implication, Union
from ..cplogic.program import CPLogicProgram
from ..expressions import PROB, ProbabilisticPredicate, ProbabilisticQuery
from ..stratification import stratify_program

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
S = Symbol("S")
T = Symbol("T")
A = Symbol("A")
B = Symbol("B")
C = Symbol("C")
WLQ = Symbol("WLQ")
WLQ1 = Symbol("WLQ1")
WLQ2 = Symbol("WLQ2")
Query = Symbol("Query")
x = Symbol("x")
y = Symbol("y")
p = Symbol("p")
p1 = Symbol("p1")
p2 = Symbol("p2")
a = Constant("a")
b = Constant("b")


def test_stratify_deterministic():
    code = Union(
        (
            Fact(Q(a)),
            Fact(R(b)),
            Fact(S(a)),
            Implication(P(x, y), Conjunction((Q(x), R(y)))),
            Implication(Z(x), S(x)),
        )
    )
    program = CPLogicProgram()
    program.walk(code)
    query = Implication(Query(y), Conjunction((Z(x), P(x, y))))
    det_idb, prob_idb, ppq_det_idb = stratify_program(query, program)
    assert len(det_idb.formulas) == 2
    assert len(prob_idb.formulas) == 0
    assert len(ppq_det_idb.formulas) == 0


def test_stratify_deterministic_probabilistic():
    code = Union(
        (
            Fact(Q(a)),
            Fact(R(b)),
            Implication(ProbabilisticPredicate(Constant(0.5), S(a)), TRUE),
            Implication(P(x, y), Conjunction((Q(x), R(y)))),
            Implication(Z(x), S(x)),
        )
    )
    program = CPLogicProgram()
    program.walk(code)
    query = Implication(Query(y), Conjunction((Z(x), P(x, y))))
    det_idb, prob_idb, ppq_det_idb = stratify_program(query, program)
    assert len(det_idb.formulas) == 1
    assert det_idb.formulas[0].consequent.functor == P
    assert len(prob_idb.formulas) == 1
    assert prob_idb.formulas[0].consequent.functor == Z
    assert len(ppq_det_idb.formulas) == 0


def test_stratify_deterministic_probabilistic_wlq():
    det_idb = [
        Implication(P(x, y), Conjunction((Q(x), R(y)))),
    ]
    prob_idb = [
        Implication(Z(x), S(x)),
        Implication(
            WLQ(x, y, ProbabilisticQuery(PROB, (x, y)),),
            Conjunction((P(x, y), Z(x))),
        ),
    ]
    ppq_det_idb = [
        Implication(B(x, p), Conjunction((A(x), A(y), WLQ(x, y, p)))),
    ]
    code = Union(
        [
            Fact(Q(a)),
            Fact(R(b)),
            Implication(ProbabilisticPredicate(Constant(0.5), S(a)), TRUE),
            Fact(A(b)),
            Fact(C(b)),
        ]
        + det_idb
        + prob_idb
        + ppq_det_idb
    )
    program = CPLogicProgram()
    program.walk(code)
    query = Implication(Query(x, p), Conjunction((B(x, p), C(x))))
    res_det_idb, res_prob_idb, res_ppq_det_idb = stratify_program(
        query, program
    )
    assert set(res_det_idb.formulas) == set(det_idb)
    assert set(res_prob_idb.formulas) == set(prob_idb)
    assert set(res_ppq_det_idb.formulas) == set(ppq_det_idb)


def test_stratify_multiple_wlqs():
    det_idb = [
        Implication(P(x, y), Conjunction((Q(x), R(y)))),
    ]
    prob_idb = [
        Implication(Z(x), Conjunction((S(x), T(x, y)))),
        Implication(
            WLQ1(x, y, ProbabilisticQuery(PROB, (x, y)),),
            Conjunction((P(x, y), Z(x))),
        ),
        Implication(
            WLQ2(y, ProbabilisticQuery(PROB, (y,)),),
            Conjunction((P(y, y), Z(y))),
        ),
    ]
    ppq_det_idb = [
        Implication(
            B(x, p1, p2),
            Conjunction((A(x), A(y), WLQ1(x, y, p1), WLQ2(y, p2))),
        ),
    ]
    code = Union(
        [
            Fact(Q(a)),
            Fact(R(b)),
            Implication(ProbabilisticPredicate(Constant(0.5), S(a)), TRUE),
            Implication(ProbabilisticPredicate(Constant(0.5), T(a)), TRUE),
            Fact(A(b)),
            Fact(C(b)),
        ]
        + det_idb
        + prob_idb
        + ppq_det_idb
    )
    program = CPLogicProgram()
    program.walk(code)
    query = Implication(Query(x, p1, p2), Conjunction((B(x, p1, p2), C(x))))
    res_det_idb, res_prob_idb, res_ppq_det_idb = stratify_program(
        query, program
    )
    assert set(res_det_idb.formulas) == set(det_idb)
    assert set(res_prob_idb.formulas) == set(prob_idb)
    assert set(res_ppq_det_idb.formulas) == set(ppq_det_idb)
