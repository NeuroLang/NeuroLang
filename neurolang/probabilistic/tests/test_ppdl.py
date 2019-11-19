import pytest

from ...datalog import Fact
from ...exceptions import NeuroLangException
from ...existential_datalog import (Implication,
                                    SolverNonRecursiveExistentialDatalog)
from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import Constant, ExpressionBlock, Symbol
from ...logic.unification import apply_substitution, most_general_unifier
from ...solver_datalog_extensional_db import ExtensionalDatabaseSolver
from ..ppdl import (DeltaTerm, GenerativeDatalog, TranslateGDatalogToEDatalog,
                    can_lead_to_object_uncertainty,
                    concatenate_to_expression_block, get_dterm)

C_ = Constant
S_ = Symbol

x = S_("x")
y = S_("y")
z = S_("z")
P = S_("P")
Q = S_("Q")
R = S_("R")
W = S_("W")
K = S_("K")
Z = S_("Z")
p = S_("p")
q = S_("q")
a = C_("a")
b = C_("b")

bernoulli = S_("bernoulli")


class GenerativeDatalogTest(GenerativeDatalog, ExpressionBasicEvaluator):
    pass


class GenerativeDatalogTestSolver(
    ExtensionalDatabaseSolver,
    SolverNonRecursiveExistentialDatalog,
    ExpressionBasicEvaluator,
):
    pass


class TranslateGDatalogToEDatalogTestSolver(TranslateGDatalogToEDatalog):
    pass


def test_get_dterm():
    datom = P(x, y, DeltaTerm(S_("Hi"), tuple()))
    assert get_dterm(datom) == DeltaTerm(S_("Hi"), tuple())
    datom = P(x, y, DeltaTerm(S_("Hi"), (C_(2),)))
    assert get_dterm(datom) == DeltaTerm(S_("Hi"), (C_(2),))


def test_apply_substitution_to_delta_term():
    dterm = DeltaTerm(bernoulli, (p,))
    new_dterm = apply_substitution(dterm, {p: q})
    assert new_dterm == DeltaTerm(bernoulli, (q,))

    substitution = {S_("random_symbol"): S_("another_random_symbol")}
    new_dterm = apply_substitution(dterm, substitution)
    assert new_dterm == dterm


def test_apply_substitution_to_delta_atom():
    datom = P(x, DeltaTerm(bernoulli, (p,)))
    new_datom = apply_substitution(datom, {p: q})
    assert new_datom == P(x, DeltaTerm(bernoulli, (q,)))


def test_unification_of_delta_atom():
    a = P(x, DeltaTerm(bernoulli, (p,)))
    b = P(y, DeltaTerm(bernoulli, (q,)))
    mgu = most_general_unifier(a, b)
    assert mgu is not None
    unifier, new_exp = mgu
    assert unifier == {x: y, p: q}


def test_generative_datalog():
    tau_1 = Implication(P(x, DeltaTerm(bernoulli, (C_(0.5),))), Q(x))
    program = ExpressionBlock((tau_1,))
    gdatalog = GenerativeDatalogTest()
    gdatalog.walk(program)
    edb = gdatalog.extensional_database()
    idb = gdatalog.intensional_database()
    assert tau_1 in idb["P"].formulas

    with pytest.raises(NeuroLangException):
        tau_2 = Implication(
            P(
                x,
                DeltaTerm(bernoulli, tuple()),
                DeltaTerm(S_("Flap"), tuple()),
            ),
            Q(x),
        )
        gdatalog = GenerativeDatalogTest()
        gdatalog.walk(ExpressionBlock((tau_2, Fact(Q(C_(2))))))
        edb = gdatalog.extensional_database()
        idb = gdatalog.intensional_database()
        assert tau_2 in idb["P"].formulas
        assert Fact(Q(C_(2))) in edb["Q"]


def test_check_gdatalog_object_uncertainty():
    program = ExpressionBlock(
        (
            P(a),
            P(b),
            Implication(Q(x, DeltaTerm(bernoulli, (C_(0.5),))), P(x)),
            Implication(R(x), Q(x, C_(0))),
        )
    )
    gdatalog = GenerativeDatalogTest()
    gdatalog.walk(program)
    assert can_lead_to_object_uncertainty(gdatalog)

    program = ExpressionBlock(
        (
            P(a),
            P(b),
            Implication(Q(x, DeltaTerm(bernoulli, (C_(0.5),))), P(x)),
            Implication(R(x), Q(x, y)),
        )
    )
    gdatalog = GenerativeDatalogTest()
    gdatalog.walk(program)
    assert not can_lead_to_object_uncertainty(gdatalog)


def test_translation_of_gdatalog_program_to_edatalog_program():
    solver = TranslateGDatalogToEDatalogTestSolver()
    res = solver.walk(
        Implication(P(x, DeltaTerm(bernoulli, (C_(0.5),))), Q(x))
    )
    assert isinstance(res, ExpressionBlock)
    assert len(res.expressions) == 2


def test_non_generative_rule_preserved_when_block_translated():
    block = ExpressionBlock(
        (
            Implication(P(x, DeltaTerm(bernoulli, (C_(0.5),))), Q(x)),
            Fact(Q(a)),
            Fact(Q(b)),
            Implication(Z(x, y, z), W(x, y) & K(y, z)),
        )
    )
    translator = TranslateGDatalogToEDatalogTestSolver()
    translated_block = translator.walk(block)
    assert Fact(Q(a)) in translated_block.expressions
    assert (
        Implication(P(x, DeltaTerm(bernoulli, (C_(0.5),))), Q(x))
        not in translated_block.expressions
    )
    assert (
        Implication(Z(x, y, z), W(x, y) & K(y, z))
        in translated_block.expressions
    )


def test_burglar():
    City = S_("City")
    House = S_("House")
    Business = S_("Business")
    Unit = S_("Unit")
    Earthquake = S_("Earthquake")
    Burglary = S_("Burglary")
    Trig = S_("Trig")
    Alarm = S_("Alarm")
    x, h, b, c, r = S_("x"), S_("h"), S_("b"), S_("c"), S_("r")
    program = ExpressionBlock(
        (
            Implication(Unit(h, c), House(h, c)),
            Implication(Unit(b, c), Business(b, c)),
            Implication(
                Earthquake(c, DeltaTerm(bernoulli, (C_(0.01),))), City(c, r)
            ),
            Implication(
                Burglary(x, c, DeltaTerm(bernoulli, (r,))),
                Unit(x, c) & City(c, r),
            ),
            Implication(
                Trig(x, DeltaTerm(bernoulli, (C_(0.6),))),
                Unit(x, c) & Earthquake(c, C_(1)),
            ),
            Implication(
                Trig(x, DeltaTerm(bernoulli, (C_(0.9),))),
                Burglary(x, c, C_(1)),
            ),
            Implication(Alarm(x), Trig(x, C_(1))),
        )
    )

    translator = TranslateGDatalogToEDatalogTestSolver()
    translated = translator.walk(program)
    assert not any(
        isinstance(e, ExpressionBlock) for e in translated.expressions
    )
    solver = GenerativeDatalogTestSolver()
    solver.walk(translated)


def test_pcs_example():
    Uniform = S_("uniform")
    Gender = S_("Gender")
    Subject = S_("Subject")
    pGender = S_("pGender")
    pHasLPC = S_("pHasLPC")
    pHasRPC = S_("pHasRPC")
    HasLPC = S_("HasLPC")
    HasRPC = S_("HasRPC")
    x = S_("x")
    p = S_("p")
    program = ExpressionBlock(
        (
            Implication(
                Gender(x, DeltaTerm(bernoulli, (p,))), Subject(x) & pGender(p)
            ),
            Implication(
                HasLPC(x, DeltaTerm(bernoulli, (p,))), Subject(x) & pHasLPC(p)
            ),
            Implication(
                HasRPC(x, DeltaTerm(bernoulli, (p,))), Subject(x) & pHasRPC(p)
            ),
            Implication(pGender(DeltaTerm(Uniform, (C_(0), C_(1)))), C_(True)),
            Implication(pHasLPC(DeltaTerm(Uniform, (C_(0), C_(1)))), C_(True)),
            Implication(pHasRPC(DeltaTerm(Uniform, (C_(0), C_(1)))), C_(True)),
        )
    )
    translator = TranslateGDatalogToEDatalogTestSolver()
    translated = translator.walk(program)
    assert not any(
        isinstance(e, ExpressionBlock) for e in translated.expressions
    )
    solver = GenerativeDatalogTestSolver()
    solver.walk(translated)


def test_concatenate_to_expression_block():
    x = S_("x")
    y = S_("y")
    z = S_("z")
    P = S_("P")
    Q = S_("Q")
    block1 = ExpressionBlock((P(x),))
    block2 = ExpressionBlock((Q(z), Q(y), Q(x)))
    block3 = ExpressionBlock((P(z), P(y)))
    assert P(y) in concatenate_to_expression_block(block1, [P(y)]).expressions
    for expression in block2.expressions:
        new_block = concatenate_to_expression_block(block1, block2)
        assert expression in new_block.expressions
        new_block = concatenate_to_expression_block(block1, block2)
        assert expression in new_block.expressions
    for expression in (
        block1.expressions + block2.expressions + block3.expressions
    ):
        new_block = concatenate_to_expression_block(
            concatenate_to_expression_block(block1, block2), block3
        )
        assert expression in new_block.expressions
