from ....expressions import (
    Symbol,
    Constant,
)
from ....expression_walker import ExpressionBasicEvaluator
from ....logic import Implication, Conjunction, UniversalPredicate, Union
from ....datalog.basic_representation import DatalogProgram
from ....datalog.negation import DatalogProgramNegation
from ....datalog.expressions import TranslateToLogic
from ....datalog.chase import (
    ChaseGeneral,
    ChaseMGUMixin,
    ChaseNaive,
)
from ....datalog.chase.negation import (
    DatalogChaseNegation,
)
from ... import QueryBuilderDatalog
from ..translate_to_dl import (
    CnlFrontendMixin,
    TransformIntoConjunctionOfDatalogSentences,
)

from itertools import product


x = Symbol("X")
y = Symbol("Y")
Jones = Constant("Jones")
Ulysses = Constant("Ulysses")
Odyssey = Constant("Odyssey")


class Datalog(TranslateToLogic, DatalogProgram, ExpressionBasicEvaluator):
    pass


class Chase(
    ChaseNaive, ChaseMGUMixin, ChaseGeneral,
):
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

    exp = TransformIntoConjunctionOfDatalogSentences().walk(
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


class DatalogNeg(
    TranslateToLogic, DatalogProgramNegation, ExpressionBasicEvaluator
):
    pass


class NeurolangNegCNL(CnlFrontendMixin, QueryBuilderDatalog):
    def __init__(self, solver=None):
        super().__init__(DatalogNeg(), chase_class=DatalogChaseNegation)


def test_sentence_negation():
    nl = NeurolangNegCNL()
    nl.execute_cnl_code(
        """
        if `pair(X, Y)` then X accompanies Y and Y accompanies X.  if
        `number(X)` and is not the case that X accompanies "3" then `sol(X)`.
        """
    )
    nl.add_tuple_set({("1", "2"), ("3", "4"), ("5", "6")}, name="pair")
    nl.add_tuple_set(
        {("1",), ("2",), ("3",), ("4",), ("5",), ("6",)}, name="number"
    )
    res = nl.solve_all()
    for n in ("1", "2", "3", "5", "6"):
        assert (Constant(n),) in res["sol"].unwrap()


def test_multiple_inferred_words():
    nl = NeurolangNegCNL()
    nl.execute_cnl_code(
        """
        if X affects_in_some_way Y then Y reacts_somehow_to X.
        """
    )
    nl.add_tuple_set({("a", "b")}, name="affects_in_some_way")
    res = nl.solve_all()
    assert (Constant("b"), Constant("a"),) in res["reacts_somehow_to"].unwrap()
