from ....expressions import (
    Symbol,
    Constant,
)
from ....expression_walker import ExpressionBasicEvaluator
from ....logic import Implication, Conjunction, UniversalPredicate, Union
from ....datalog.basic_representation import DatalogProgram
from ....datalog.expressions import TranslateToLogic
from ....datalog.chase import (
    ChaseGeneral,
    ChaseMGUMixin,
    ChaseNaive,
)
from ..translate_to_dl import CnlFrontendMixin, IntoConjunctionOfSentences
from ... import QueryBuilderDatalog

from itertools import product


x = Symbol("X")
y = Symbol("Y")
Jones = Constant("Jones")
Ulysses = Constant("Ulysses")
Odyssey = Constant("Odyssey")


class Datalog(TranslateToLogic, DatalogProgram, ExpressionBasicEvaluator):
    pass


class Chase(ChaseNaive, ChaseMGUMixin, ChaseGeneral):
    pass


class NeurolangCNL(CnlFrontendMixin, QueryBuilderDatalog):
    def __init__(self, solver=None):
        super().__init__(Datalog(), chase_class=Chase)


def test_translate_to_dl():
    intersects = Symbol("intersects")
    nl = NeurolangCNL()
    nl.execute_cnl_code("if Y intersects X then X intersects Y")
    expected = Union((Implication(intersects(x, y), intersects(y, x)),))
    assert nl.solver.symbol_table["intersects"] == expected


def test_translate_to_dl_2():
    region = Symbol("region")
    intersects = Symbol("intersects")
    nl = NeurolangCNL()
    nl.execute_cnl_code(
        "if a region Y intersects a region X then X intersects Y"
    )
    expected = Union(
        (
            Implication(
                intersects(x, y),
                Conjunction((intersects(y, x), region(x), region(y),)),
            ),
        )
    )
    assert nl.solver.symbol_table["intersects"] == expected


def test_distribute_implication_conjunctive_head():
    x = Symbol("x")
    y = Symbol("y")
    A = Symbol("A")
    B = Symbol("B")
    C = Symbol("C")

    exp = IntoConjunctionOfSentences().walk(
        UniversalPredicate(
            x,
            UniversalPredicate(
                y, Implication(Conjunction((B(x), C(y),)), A(x, y),)
            ),
        )
    )

    assert exp == Conjunction(
        (
            UniversalPredicate(
                x, UniversalPredicate(y, Implication(B(x), A(x, y))),
            ),
            UniversalPredicate(
                x, UniversalPredicate(y, Implication(C(y), A(x, y))),
            ),
        )
    )


def test_solver_integration():
    nl = NeurolangCNL()
    nl.execute_cnl_code(
        """
        if a book X references Odyssey then Jones likes X.
        Ulysses references Odyssey.
        """
    )
    nl.add_tuple_set({(Odyssey,), (Ulysses,)}, name="book")
    res = nl.solve_all()
    assert (Jones, Ulysses) in res["likes"].unwrap()


def test_conjunctions():
    nl = NeurolangCNL()
    nl.execute_cnl_code(
        """
        if `edge(X, Y)` then X reaches Y.
        if X reaches Y then Y reaches X.
        if A reaches B and B reaches C then A reaches C.
        """
    )
    nl.add_tuple_set({(1, 2), (2, 3), (3, 4)}, name="edge")
    res = nl.solve_all()

    for a, b in product((1, 2, 3, 4), (1, 2, 3, 4)):
        if a == b:
            continue
        assert (Constant(a), Constant(b)) in res["reaches"].unwrap()


def test_conjunctions_2():
    nl = NeurolangCNL()
    nl.execute_cnl_code(
        """
        if `edge(X, Y)` then X reaches Y.
        if X reaches Y then Y reaches X.
        if X reaches Y and X has "label" then Y has "label".
        if `start(X)` then X has "label".
        """
    )
    nl.add_tuple_set(
        {("A", "B"), ("B", "C"), ("C", "D"), ("F", "G")}, name="edge"
    )
    nl.add_tuple_set({("A",)}, name="start")
    res = nl.solve_all()

    for x in ("A", "B", "C", "D"):
        assert (Constant(x), Constant("label")) in res["has"].unwrap()


def test_conjunction_3():
    edge = Symbol("edge")
    has = Symbol("has")

    nl = NeurolangCNL()
    nl.execute_cnl_code(
        """
        if `edge(X, Y)` then X reaches Y.
        if X reaches Y and Y reaches Z then X reaches Z.
        if X reaches X then X has "cycle".
        """
    )
    nl.add_tuple_set(
        {
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "E"),
            ("E", "B"),
            ("E", "F"),
        },
        name=edge,
    )
    res = nl.solve_all()

    for x in ("B", "C", "D", "E"):
        assert (Constant(x), Constant("cycle")) in res[has].unwrap()
    for x in ("A", "F"):
        assert (Constant(x), Constant("cycle")) not in res[has].unwrap()


def test_define_a_verb():
    nl = NeurolangCNL()
    nl.this_is_it = None
    nl.execute_cnl_code(
        """
        `singular_verb("activates")`.
        if `edge(X, Y)` then X activates Y.
        if X activates Y then Y activates X.
        if A activates B and B activates C then A activates C.
        """
    )
    nl.add_tuple_set({(1, 2), (2, 3), (3, 4)}, name="edge")
    res = nl.solve_all()
    for a, b in product((1, 2, 3, 4), (1, 2, 3, 4)):
        if a == b:
            continue
        assert (Constant(a), Constant(b)) in res["activates"].unwrap()


def test_define_synonyms():
    nl = NeurolangCNL()
    nl.this_is_it = None
    nl.execute_cnl_code(
        """
        `singular_verb("activates")`.
        `singular_verb("actuates")`.
        `singular_verb("triggers")`.
        if X triggers Y then X activates Y.
        if X activates Y then X actuates Y.
        if X actuates Y then X triggers Y.

        if `edge(X, Y)` then X triggers Y.
        if X actuates Y then Y activates X.
        if A triggers B and B activates C then A actuates C.
        """
    )
    nl.add_tuple_set({(1, 2), (2, 3), (3, 4)}, name="edge")
    res = nl.solve_all()
    for a, b in product((1, 2, 3, 4), (1, 2, 3, 4)):
        if a == b:
            continue
        assert (Constant(a), Constant(b)) in res["activates"].unwrap()


def test_guess_verb():
    nl = NeurolangCNL()
    nl.debug = True
    nl.this_is_it = None
    nl.execute_cnl_code(
        """
        if `edge(X, Y)` then X activates Y.
        if X activates Y then Y activates X.
        if A activates B and B activates C then A activates C.
        """
    )
    nl.add_tuple_set({(1, 2), (2, 3), (3, 4)}, name="edge")
    res = nl.solve_all()
    for a, b in product((1, 2, 3, 4), (1, 2, 3, 4)):
        if a == b:
            continue
        assert (Constant(a), Constant(b)) in res["activates"].unwrap()
