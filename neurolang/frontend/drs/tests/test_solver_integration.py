from ....expressions import (
    Symbol,
    Constant,
    ExpressionBlock,
)
from ....expression_walker import ExpressionBasicEvaluator
from ....logic import Implication, Conjunction, UniversalPredicate
from ....datalog.basic_representation import DatalogProgram
from ....datalog.expressions import TranslateToLogic
from ....datalog.chase import (
    ChaseGeneral,
    ChaseMGUMixin,
    ChaseNaive,
)
from ..translate_to_dl import (
    TranslateToDatalog,
    TransformIntoConjunctionOfDatalogSentences,
)

from itertools import product


intersects = Symbol("intersects")
region = Symbol("region")
book = Symbol("book")
likes = Symbol("likes")
references = Symbol("references")
x = Symbol("X")
y = Symbol("Y")
Jones = Constant("Jones")
Ulysses = Constant("Ulysses")
Odyssey = Constant("Odyssey")


def test_translate_to_dl():
    ttdl = TranslateToDatalog()
    program = ttdl.translate_sentence("if Y intersects X then X intersects Y")

    assert program == ExpressionBlock(
        (Implication(intersects(x, y), intersects(y, x)),)
    )


def test_translate_to_dl_2():
    ttdl = TranslateToDatalog()
    program = ttdl.translate_sentence(
        "if a region Y intersects a region X then X intersects Y"
    )

    expected = ExpressionBlock(
        (
            Implication(
                intersects(x, y),
                Conjunction((intersects(y, x), region(x), region(y),)),
            ),
        )
    )
    assert program == expected


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


class Datalog(TranslateToLogic, DatalogProgram, ExpressionBasicEvaluator):
    pass


class Chase(ChaseNaive, ChaseMGUMixin, ChaseGeneral):
    pass


def test_solver_integration():
    """
    Given an extensional base having the predicates:
      `likes(x, y), book(x), references(x, y)`

    And the facts:
      `book(Odyssey), book(Ulysses)`

    Adding the next sentences to the base:
      `Jones likes a book if it references Odyssey.`
      `Ulysses references Odyssey.`

    Should answer:
      `Does Jones like Ulysses`
    """

    ttdl = TranslateToDatalog()
    program = ttdl.translate_block(
        """
        if a book X references Odyssey then Jones likes X.
        Ulysses references Odyssey.
        """
    )

    dl = Datalog()
    dl.add_extensional_predicate_from_tuples(book, {(Odyssey,), (Ulysses,)})
    dl.walk(program)

    dc = Chase(dl)
    solution = dc.build_chase_solution()

    assert (Jones, Ulysses) in solution[likes].value


def test_conjunctions():
    reaches = Symbol("reaches")
    edge = Symbol("edge")

    ttdl = TranslateToDatalog()
    program = ttdl.translate_block(
        """
        if `edge(X, Y)` then X reaches Y.
        if X reaches Y then Y reaches X.
        if A reaches B and B reaches C then A reaches C.
        """
    )
    dl = Datalog()
    dl.add_extensional_predicate_from_tuples(edge, {(1, 2), (2, 3), (3, 4)})
    dl.walk(program)

    dc = Chase(dl)
    solution = dc.build_chase_solution()

    for a, b in product((1, 2, 3, 4), (1, 2, 3, 4)):
        if a == b:
            continue
        assert (Constant(a), Constant(b)) in solution[reaches].value


def test_conjunctions_2():
    edge = Symbol("edge")
    start = Symbol("start")
    has = Symbol("has")
    program = TranslateToDatalog().translate_block(
        """
        if `edge(X, Y)` then X reaches Y.
        if X reaches Y then Y reaches X.
        if X reaches Y and X has "label" then Y has "label".
        if `start(X)` then X has "label".
        """
    )
    dl = Datalog()
    dl.add_extensional_predicate_from_tuples(
        edge, {("A", "B"), ("B", "C"), ("C", "D"), ("F", "G")}
    )
    dl.add_extensional_predicate_from_tuples(start, {("A",)})
    dl.walk(program)
    dc = Chase(dl)
    solution = dc.build_chase_solution()

    for x in ("A", "B", "C", "D"):
        assert (Constant(x), Constant("label")) in solution[has].value


def test_conjunction_3():
    edge = Symbol("edge")
    has = Symbol("has")
    program = TranslateToDatalog().translate_block(
        """
        if `edge(X, Y)` then X reaches Y.
        if X reaches Y and Y reaches Z then X reaches Z.
        if X reaches X then X has "cycle".
        """
    )
    dl = Datalog()
    dl.add_extensional_predicate_from_tuples(
        edge,
        {
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "E"),
            ("E", "B"),
            ("E", "F"),
        },
    )
    dl.walk(program)
    dc = Chase(dl)
    solution = dc.build_chase_solution()

    for x in ("B", "C", "D", "E"):
        assert (Constant(x), Constant("cycle")) in solution[has].value
    for x in ("A", "F"):
        assert (Constant(x), Constant("cycle")) not in solution[has].value
