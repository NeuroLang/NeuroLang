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
