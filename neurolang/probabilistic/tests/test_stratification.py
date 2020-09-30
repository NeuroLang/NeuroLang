import pytest

from ...datalog.expressions import Fact
from ...exceptions import ForbiddenRecursivityError, UnsupportedProgramError
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
c = Constant("c")


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
    idbs = stratify_program(query, program)
    assert len(idbs["deterministic"].formulas) == 2
    assert "probabilistic" not in idbs
    assert "post_probabilistic" not in idbs


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
    idbs = stratify_program(query, program)
    assert len(idbs["deterministic"].formulas) == 1
    assert idbs["deterministic"].formulas[0].consequent.functor == P
    assert len(idbs["probabilistic"].formulas) == 1
    assert idbs["probabilistic"].formulas[0].consequent.functor == Z
    assert "post_probabilistic" not in idbs


def test_stratify_deterministic_probabilistic_wlq():
    det_idb = [
        Implication(P(x, y), Conjunction((Q(x), R(y)))),
    ]
    prob_idb = [
        Implication(Z(x), S(x)),
        Implication(
            WLQ(x, y, ProbabilisticQuery(PROB, (x, y))),
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
    idbs = stratify_program(query, program)
    assert set(idbs["deterministic"].formulas) == set(det_idb)
    assert set(idbs["probabilistic"].formulas) == set(prob_idb)
    assert set(idbs["post_probabilistic"].formulas) == set(ppq_det_idb)


def test_stratify_multiple_wlqs():
    det_idb = [
        Implication(P(x, y), Conjunction((Q(x), R(y)))),
    ]
    prob_idb = [
        Implication(Z(x), Conjunction((S(x), T(x, y)))),
        Implication(
            WLQ1(x, y, ProbabilisticQuery(PROB, (x, y))),
            Conjunction((P(x, y), Z(x))),
        ),
        Implication(
            WLQ2(y, ProbabilisticQuery(PROB, (y,))),
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
    idbs = stratify_program(query, program)
    assert set(idbs["deterministic"].formulas) == set(det_idb)
    assert set(idbs["probabilistic"].formulas) == set(prob_idb)
    assert set(idbs["post_probabilistic"].formulas) == set(ppq_det_idb)


def test_wlq_dependence_on_other_wlq():
    det_idb = [
        Implication(P(x, y), Conjunction((Q(x), R(y)))),
    ]
    prob_idb = [
        Implication(Z(x), Conjunction((S(x), T(x, y)))),
        Implication(
            WLQ1(x, y, ProbabilisticQuery(PROB, (x, y))),
            Conjunction((P(x, y), Z(x))),
        ),
        Implication(
            WLQ2(y, p, ProbabilisticQuery(PROB, (y, p))),
            Conjunction((WLQ1(y, y, p), Z(y))),
        ),
    ]
    code = Union(
        [
            Fact(Q(a)),
            Fact(R(b)),
            Implication(ProbabilisticPredicate(Constant(0.5), S(a)), TRUE),
            Implication(ProbabilisticPredicate(Constant(0.5), T(a)), TRUE),
        ]
        + det_idb
        + prob_idb
    )
    program = CPLogicProgram()
    program.walk(code)
    query = Implication(Query(x, p), Conjunction((WLQ2(x, p), C(x))))
    with pytest.raises(UnsupportedProgramError):
        stratify_program(query, program)


def test_cannot_stratify_recursive_program():
    prob_idb = [
        Implication(B(x), Conjunction((A(x), C(x)))),
        Implication(A(x), B(x)),
    ]
    code = Union(
        [
            Fact(C(a)),
            Fact(C(b)),
        ]
        + prob_idb
    )
    program = CPLogicProgram()
    program.walk(code)
    query = Implication(Query(x), A(x))
    with pytest.raises(ForbiddenRecursivityError):
        stratify_program(query, program)


def test_stratification_multiple_post_probabilistic_rules():
    facts = [Fact(R(a)), Fact(R(b)), Fact(B(a)), Fact(B(c)), Fact(B(b))]
    det_idb = [Implication(A(x, y), Conjunction((B(y), R(x))))]
    pfacts = [
        Implication(ProbabilisticPredicate(Constant(0.2), C(a, b)), TRUE)
    ]
    prob_idb = [
        Implication(
            WLQ(x, y, ProbabilisticQuery(PROB, (x, y))),
            Conjunction((A(x, y), C(x, y))),
        )
    ]
    post_prob_idb = [
        Implication(Z(x, p), Conjunction((WLQ(x, y, p), R(y)))),
        Implication(T(p), Z(x, p)),
    ]
    code = Union(tuple(facts + pfacts + prob_idb + det_idb + post_prob_idb))
    program = CPLogicProgram()
    program.walk(code)
    query = Implication(Query(x), T(x))
    idbs = stratify_program(query, program)
    assert set(idbs["deterministic"].formulas) == set(det_idb)
    assert set(idbs["probabilistic"].formulas) == set(prob_idb)
    assert set(idbs["post_probabilistic"].formulas) == set(post_prob_idb)
