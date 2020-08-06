import pytest
from unittest.mock import patch
import operator

from ..exceptions import NeuroLangException
from ..expressions import (
    Symbol,
    Constant,
    FunctionApplication,
    ExpressionBlock,
)
from ..logic.unification import most_general_unifier
from ..logic import (
    Implication,
    Disjunction,
    Conjunction,
    Union,
    Negation,
    UniversalPredicate,
    ExistentialPredicate,
)
from ..logic.transformations import (
    EliminateImplications,
    MoveNegationsToAtoms,
    RemoveUniversalPredicates,
    MoveQuantifiersUp,
    DistributeDisjunctions,
    DesambiguateQuantifiedVariables,
    CollapseDisjunctions,
    CollapseConjunctions,
    convert_to_pnf_with_cnf_matrix,
)
from ..logic.horn_clauses import (
    MoveNegationsToAtomsOrExistentialQuantifiers,
    HornClause,
    HornFact,
    convert_to_srnf,
    convert_srnf_to_horn_clauses,
    range_restricted_variables,
    is_safe_range,
    NeuroLangTranslateToHornClauseException,
    translate_horn_clauses_to_datalog,
    fol_query_to_datalog_program,
    Fol2DatalogMixin,
    Fol2DatalogTranslationException,
)
from ..datalog.negation import DatalogProgramNegation
from ..expression_walker import ExpressionBasicEvaluator
from ..datalog.expressions import TranslateToLogic, Fact
from ..datalog.chase import Chase as Chase_
from ..datalog.chase.negation import NegativeFactConstraints


def test_remove_implication():
    A = Symbol("A")
    B = Symbol("B")
    exp = Implication(B, A)
    res = EliminateImplications().walk(exp)

    assert isinstance(res, Disjunction)
    assert res == Disjunction((B, Negation(A)))


def test_remove_nested_implication():
    A = Symbol("A")
    B = Symbol("B")
    exp = Implication(Implication(Negation(A), Negation(B)), Implication(B, A))
    res = EliminateImplications().walk(exp)

    assert isinstance(res, Disjunction)
    assert res == Disjunction(
        (
            Disjunction((Negation(A), Negation(Negation(B)))),
            Negation(Disjunction((B, Negation(A)))),
        )
    )


def test_remove_implication_in_disjunction():
    A = Symbol("A")
    B = Symbol("B")
    exp = Disjunction((A, Implication(B, A)))
    res = EliminateImplications().walk(exp)
    assert res == Disjunction((A, Disjunction((B, Negation(A)))))


def test_remove_double_negation():
    A = Symbol("A")
    exp = Negation(Negation(A))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == A


def test_negated_universal():
    X = Symbol("X")
    P = Symbol("P")
    exp = Negation(UniversalPredicate(X, P(X)))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == ExistentialPredicate(X, Negation(P(X)))


def test_negated_existensial():
    X = Symbol("X")
    P = Symbol("P")
    exp = Negation(ExistentialPredicate(X, P(X)))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == UniversalPredicate(X, Negation(P(X)))


def test_negated_disjunction():
    A = Symbol("A")
    B = Symbol("B")
    exp = Negation(Disjunction((A, B)))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == Conjunction((Negation(A), Negation(B)))


def test_negated_conjunction():
    A = Symbol("A")
    B = Symbol("B")
    exp = Negation(Conjunction((A, B)))
    res = MoveNegationsToAtoms().walk(exp)
    assert res == Disjunction((Negation(A), Negation(B)))


def test_long_negated_chain():
    A = Symbol("A")
    X = Symbol("X")
    Y = Symbol("Y")
    R = Symbol("R")
    exp = Negation(
        Disjunction(
            (
                A,
                UniversalPredicate(
                    X, ExistentialPredicate(Y, Negation(R(X, Y)))
                ),
            )
        )
    )
    res = MoveNegationsToAtoms().walk(exp)
    assert res == Conjunction(
        (Negation(A), ExistentialPredicate(X, UniversalPredicate(Y, R(X, Y))))
    )


def test_remove_universal_predicate():
    X = Symbol("X")
    P = Symbol("P")
    exp = UniversalPredicate(X, P(X))
    res = RemoveUniversalPredicates().walk(exp)

    assert res == Negation(ExistentialPredicate(X, Negation(P(X))))


def test_remove_nested_universal_predicates():
    Y = Symbol("Y")
    X = Symbol("X")
    P = Symbol("P")
    R = Symbol("R")
    exp = UniversalPredicate(
        X, Disjunction((P(X), UniversalPredicate(Y, R(X, Y))))
    )
    res = RemoveUniversalPredicates().walk(exp)

    assert res == Negation(
        ExistentialPredicate(
            X,
            Negation(
                Disjunction(
                    (
                        P(X),
                        Negation(ExistentialPredicate(Y, Negation(R(X, Y)))),
                    )
                )
            ),
        )
    )


def test_rename_repeated_symbols_in_nested_universal_predicates():
    X = Symbol("X")
    P = Symbol("P")
    exp = UniversalPredicate(
        X, Disjunction((P(X), UniversalPredicate(X, P(X))))
    )
    res = DesambiguateQuantifiedVariables().walk(exp)

    c0 = res.body.formulas[0].args[0]
    c1 = res.body.formulas[1].body.args[0]

    assert c0 != c1


def test_rename_repeated_symbols_in_sibling_quantifiers():
    X = Symbol("X")
    P = Symbol("P")
    Q = Symbol("Q")
    exp = Conjunction(
        (ExistentialPredicate(X, P(X)), ExistentialPredicate(X, Q(X)),)
    )
    res = DesambiguateQuantifiedVariables().walk(exp)

    c0 = res.formulas[0].head
    c1 = res.formulas[1].head
    assert c0 != c1
    assert res.formulas[0].body.args[0] != c1
    assert res.formulas[1].body.args[0] != c0


def test_dont_rename_when_there_is_no_need():
    X = Symbol("X")
    Y = Symbol("Y")
    P = Symbol("P")
    exp = Conjunction(
        (ExistentialPredicate(X, P(X)), ExistentialPredicate(Y, P(Y)),)
    )
    res = DesambiguateQuantifiedVariables().walk(exp)

    assert X == res.formulas[0].head
    assert Y == res.formulas[1].head


def test_rename_complex_repeated_symbols():
    X = Symbol("X")
    Y = Symbol("Y")
    P = Symbol("P")
    exp = Conjunction(
        (
            ExistentialPredicate(X, P(X)),
            Negation(UniversalPredicate(X, P(X))),
            Disjunction(
                (
                    UniversalPredicate(
                        X, UniversalPredicate(Y, Disjunction((P(X), P(Y))))
                    ),
                    ExistentialPredicate(Y, P(Y)),
                )
            ),
        )
    )
    res = DesambiguateQuantifiedVariables().walk(exp)

    X1 = res.formulas[0].head
    X2 = res.formulas[1].formula.head
    X3 = res.formulas[2].formulas[0].head
    Y1 = res.formulas[2].formulas[0].body.head
    Y2 = res.formulas[2].formulas[1].head

    assert X == X1
    assert X1 != X2
    assert X1 != X3
    assert X2 != X3

    assert Y == Y1
    assert Y1 != Y2

    expected = Conjunction(
        (
            ExistentialPredicate(X1, P(X)),
            Negation(UniversalPredicate(X2, P(X2))),
            Disjunction(
                (
                    UniversalPredicate(
                        X3, UniversalPredicate(Y1, Disjunction((P(X3), P(Y1))))
                    ),
                    ExistentialPredicate(Y2, P(Y2)),
                )
            ),
        )
    )
    assert expected == res


def test_remove_sibling_universal_predicates():
    Y = Symbol("Y")
    X = Symbol("X")
    P = Symbol("P")
    R = Symbol("R")
    exp = Disjunction(
        (
            UniversalPredicate(X, P(X)),
            UniversalPredicate(X, Conjunction((P(X), R(X, Y)))),
            P(Y),
        )
    )
    res = RemoveUniversalPredicates().walk(exp)

    assert res == Disjunction(
        (
            Negation(ExistentialPredicate(X, Negation(P(X)))),
            Negation(
                ExistentialPredicate(X, Negation(Conjunction((P(X), R(X, Y)))))
            ),
            P(Y),
        )
    )


def test_move_quantifiers_up():
    Y = Symbol("Y")
    X = Symbol("X")
    P = Symbol("P")
    R = Symbol("R")
    exp = Disjunction(
        (P, UniversalPredicate(X, Negation(ExistentialPredicate(Y, R(X, Y)))))
    )
    res = MoveQuantifiersUp().walk(exp)
    assert res == UniversalPredicate(
        X, UniversalPredicate(Y, Disjunction((P, Negation(R(X, Y)))))
    )


def test_distribute_disjunction():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")
    exp = Disjunction((P, Conjunction((Q, R))))
    res = DistributeDisjunctions().walk(exp)
    assert res == Conjunction((Disjunction((P, Q)), Disjunction((P, R)),))


def test_distribute_non_binary_disjunction():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")
    S = Symbol("S")
    exp = Disjunction((P, Conjunction((Q, R)), S))
    res = DistributeDisjunctions().walk(exp)
    assert res == Conjunction(
        (
            Disjunction((P, Disjunction((Q, S)))),
            Disjunction((P, Disjunction((R, S)))),
        )
    )


def test_collapse_disjunctions():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")
    S = Symbol("S")
    exp = Conjunction(
        (
            Disjunction((P, Disjunction((Q, S)))),
            Disjunction((P, Disjunction((R, S)))),
        )
    )
    res = CollapseDisjunctions().walk(exp)
    assert res == Conjunction(
        (Disjunction((P, Q, S)), Disjunction((P, R, S)),)
    )


def test_collapse_conjunctions():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")
    S = Symbol("S")
    exp = Conjunction((Conjunction((P, Conjunction((Q, R)))), S))
    res = CollapseConjunctions().walk(exp)
    assert res == Conjunction((P, Q, R, S))


def test_transform_to_cnf():
    A = Symbol("A")
    B = Symbol("B")
    C = Symbol("C")
    D = Symbol("D")
    P = Symbol("P")
    Q = Symbol("Q")
    exp = Disjunction((P, Conjunction((A, B, Disjunction((C, D)))), Q))
    res = convert_to_pnf_with_cnf_matrix(exp)
    assert res == Conjunction(
        (
            Disjunction((P, A, Q)),
            Disjunction((P, B, Q)),
            Disjunction((P, C, D, Q)),
        )
    )


def test_transform_to_cnf_idempotence():
    P = Symbol("P")
    Q = Symbol("Q")
    R = Symbol("R")

    exp = Disjunction((P, Q))
    res = convert_to_pnf_with_cnf_matrix(exp)
    assert res == exp

    exp = Conjunction((P, Q))
    res = convert_to_pnf_with_cnf_matrix(exp)
    assert res == exp

    exp = Conjunction((Disjunction((P, Q)), Disjunction((P, R))))
    res = convert_to_pnf_with_cnf_matrix(exp)
    assert res == exp


def test_transform_to_cnf_1():
    Y = Symbol("Y")
    X = Symbol("X")
    P = Symbol("P")
    R = Symbol("R")
    exp = UniversalPredicate(
        Y,
        Implication(P(Y), UniversalPredicate(X, Disjunction((R(Y, X), P(X))))),
    )
    res = convert_to_pnf_with_cnf_matrix(exp)
    Y, X = res.head, res.body.head
    assert res == UniversalPredicate(
        Y,
        ExistentialPredicate(
            X,
            Conjunction(
                (
                    Disjunction((P(Y), Negation(R(Y, X)),)),
                    Disjunction((P(Y), Negation(P(X)),)),
                )
            ),
        ),
    )


def test_transform_to_cnf_2():
    socrates = Symbol("socrates")
    mortal = Symbol("mortal")
    man = Symbol("man")
    X = Symbol("X")
    exp = Conjunction(
        (UniversalPredicate(X, Implication(mortal(X), man(X))), man(socrates))
    )
    res = convert_to_pnf_with_cnf_matrix(exp)
    X = res.head
    assert res == UniversalPredicate(
        X,
        Conjunction(
            (Disjunction((mortal(X), Negation(man(X)))), man(socrates))
        ),
    )


def test_transform_to_cnf_3():
    father = Symbol("father")
    sister = Symbol("sister")
    Alice = Symbol("Alice")
    Bob = Symbol("Bob")
    Carol = Symbol("Carol")
    X = Symbol("X")
    Y = Symbol("Y")
    Z = Symbol("Z")

    exp = Conjunction(
        (
            UniversalPredicate(
                X,
                UniversalPredicate(
                    Y,
                    ExistentialPredicate(
                        Z,
                        Implication(
                            sister(X, Y),
                            Conjunction((father(X, Z), father(Y, Z))),
                        ),
                    ),
                ),
            ),
            father(Alice, Bob),
            father(Carol, Bob),
        )
    )

    res = convert_to_pnf_with_cnf_matrix(exp)

    X, Y, Z = res.head, res.body.head, res.body.body.head
    assert res == UniversalPredicate(
        X,
        UniversalPredicate(
            Y,
            ExistentialPredicate(
                Z,
                Conjunction(
                    (
                        Disjunction(
                            (
                                sister(X, Y),
                                Negation(father(X, Z)),
                                Negation(father(Y, Z)),
                            )
                        ),
                        father(Alice, Bob),
                        father(Carol, Bob),
                    )
                ),
            ),
        ),
    )


def test_horn_clause_validation():
    X = Symbol("X")
    Y = Symbol("Y")
    with pytest.raises(NeuroLangException):
        HornClause(Disjunction((X, Y)), None)
    with pytest.raises(NeuroLangException):
        HornClause(None, None)
    with pytest.raises(NeuroLangException):
        HornFact(None)
    with pytest.raises(NeuroLangException):
        HornClause(None, Conjunction((X, Y)))
    with pytest.raises(NeuroLangException):
        HornClause(None, (X, None))
    with pytest.raises(NeuroLangException):
        HornClause(None, (Conjunction((X, Y)), Y))


def test_move_negations_to_atoms_or_existentials():
    Y = Symbol("Y")
    X = Symbol("X")
    P = Symbol("P")
    Q = Symbol("Q")
    exp = Negation(
        Disjunction(
            (
                P(X),
                ExistentialPredicate(
                    Y, Negation(Conjunction((P(Y), Negation(Q(Y)))))
                ),
            )
        )
    )
    res = MoveNegationsToAtomsOrExistentialQuantifiers().walk(exp)
    assert res == Conjunction(
        (
            Negation(P(X)),
            Negation(
                ExistentialPredicate(Y, Disjunction((Negation(P(Y)), Q(Y))))
            ),
        )
    )


def test_convert_to_srnf():
    Movies = Symbol("Movies")
    H = Symbol("'Hitchcock'")
    Xt = Symbol("Xt")
    Xd = Symbol("Xd")
    Xa = Symbol("Xa")
    Ya = Symbol("Ya")
    Yd = Symbol("Yd")
    Zt = Symbol("Zt")

    exp = Conjunction(
        (
            ExistentialPredicate(
                Xd, ExistentialPredicate(Xa, Movies(Xt, Xd, Xa))
            ),
            UniversalPredicate(
                Ya,
                Implication(
                    ExistentialPredicate(Zt, Movies(Zt, H, Ya)),
                    ExistentialPredicate(Yd, Movies(Xt, Yd, Ya)),
                ),
            ),
        )
    )
    res = convert_to_srnf(exp)
    assert res == Conjunction(
        (
            ExistentialPredicate(
                Xd, ExistentialPredicate(Xa, Movies(Xt, Xd, Xa))
            ),
            Negation(
                ExistentialPredicate(
                    Ya,
                    Conjunction(
                        (
                            Negation(
                                ExistentialPredicate(Zt, Movies(Zt, H, Ya))
                            ),
                            ExistentialPredicate(Yd, Movies(Xt, Yd, Ya)),
                        )
                    ),
                )
            ),
        )
    )


def test_range_restricted_variables():
    Movies = Symbol("Movies")
    H = Constant("'Hitchcock'")
    Xt = Symbol("Xt")
    Xd = Symbol("Xd")
    Xa = Symbol("Xa")
    Ya = Symbol("Ya")
    Yd = Symbol("Yd")
    Zt = Symbol("Zt")

    exp = Conjunction(
        (
            ExistentialPredicate(
                Xd, ExistentialPredicate(Xa, Movies(Xt, Xd, Xa))
            ),
            Negation(
                ExistentialPredicate(
                    Ya,
                    Conjunction(
                        (
                            Negation(
                                ExistentialPredicate(Zt, Movies(Zt, H, Ya))
                            ),
                            ExistentialPredicate(Yd, Movies(Xt, Yd, Ya)),
                        )
                    ),
                )
            ),
        )
    )

    res = range_restricted_variables(exp)
    assert res == {Xt}
    assert is_safe_range(exp)


def test_not_safe_range():
    x = Symbol("x")
    y = Symbol("y")
    P = Symbol("P")

    exp = Conjunction(
        (P(x), Negation(ExistentialPredicate(y, Negation(P(y)))),)
    )

    assert not is_safe_range(exp)


def test_convert_to_srnf_2():
    father = Symbol("father")
    sister = Symbol("sister")
    X = Symbol("X")
    Y = Symbol("Y")
    Z = Symbol("Z")

    # Starting with this sentence:
    exp = UniversalPredicate(
        X,
        UniversalPredicate(
            Y,
            Implication(
                sister(X, Y),
                ExistentialPredicate(
                    Z, Conjunction((father(X, Z), father(Y, Z))),
                ),
            ),
        ),
    )

    # The first thing to do is to remove implications:
    exp = UniversalPredicate(
        X,
        UniversalPredicate(
            Y,
            Disjunction(
                (
                    sister(X, Y),
                    Negation(
                        ExistentialPredicate(
                            Z, Conjunction((father(X, Z), father(Y, Z))),
                        )
                    ),
                )
            ),
        ),
    )

    # Now, this is a universally quantified disjunction
    # with one positive literal so we can treat the sister
    # relation as the answer to:
    exp = ExistentialPredicate(Z, Conjunction((father(X, Z), father(Y, Z))),)
    res = convert_to_srnf(exp)
    assert is_safe_range(res)

    # And also the quantified variables from sister(X, Y) must
    # appear in the query:
    # (At this point we know that range_restricted_variables == free_variables)
    assert {X, Y} == range_restricted_variables(exp)


def test_convert_to_srnf_3():
    n = Symbol("n")
    m = Symbol("m")
    m_ = Symbol("m'")
    r = Symbol("r")
    r_ = Symbol("r'")
    Director = Symbol("Director")
    Actor = Symbol("Actor")
    Equal = Symbol("Equal")

    # Which directors played exactly one role in each of their movies
    exp = Conjunction(
        (
            ExistentialPredicate(m, Director(n, m)),
            UniversalPredicate(
                m_,
                Implication(
                    ExistentialPredicate(
                        r,
                        Conjunction(
                            (
                                Actor(n, m_, r),
                                UniversalPredicate(
                                    r_,
                                    Implication(
                                        Equal(r, r_), Actor(n, m_, r_)
                                    ),
                                ),
                            )
                        ),
                    ),
                    Director(n, m_),
                ),
            ),
        )
    )
    res = convert_to_srnf(exp)
    assert is_safe_range(res)


def test_convert_srnf2horn():
    Father = Symbol("Father")
    X = Symbol("X")
    Y = Symbol("Y")
    Z = Symbol("Z")
    Sister = Symbol("Sister")

    exp = ExistentialPredicate(Z, Conjunction((Father(X, Z), Father(Y, Z))),)
    exp = convert_to_srnf(exp)
    res = convert_srnf_to_horn_clauses(Sister(X, Y), exp)

    expected = Union(
        (HornClause(Sister(X, Y), Conjunction((Father(X, Z), Father(Y, Z)))),)
    )
    assert res == expected


def test_convert_srnf2horn_2():
    Movies = Symbol("Movies")
    Ans = Symbol("Ans")
    H = Constant("'Hitchcock'")
    Xt = Symbol("Xt")
    Xd = Symbol("Xd")
    Xa = Symbol("Xa")
    Ya = Symbol("Ya")
    Yd = Symbol("Yd")
    Zt = Symbol("Zt")

    exp = Conjunction(
        (
            ExistentialPredicate(
                Xd, ExistentialPredicate(Xa, Movies(Xt, Xd, Xa))
            ),
            UniversalPredicate(
                Ya,
                Implication(
                    ExistentialPredicate(Zt, Movies(Zt, H, Ya)),
                    ExistentialPredicate(Yd, Movies(Xt, Yd, Ya)),
                ),
            ),
        )
    )
    exp2 = convert_to_srnf(exp)
    res = convert_srnf_to_horn_clauses(Ans(Xt), exp2)

    Aux2 = res.formulas[0].head.functor
    Aux1 = res.formulas[1].head.functor

    expected = Union(
        (
            HornClause(Aux2(Ya), Movies(Zt, H, Ya)),
            HornClause(
                Aux1(Xt), Conjunction((Movies(Xt, Yd, Ya), Negation(Aux2(Ya))))
            ),
            HornClause(
                Ans(Xt), Conjunction((Movies(Xt, Xd, Xa), Negation(Aux1(Xt))))
            ),
        )
    )
    assert res == expected


def test_convert_srnf2horn_3():
    n = Symbol("n")
    m = Symbol("m")
    m_ = Symbol("m'")
    r = Symbol("r")
    r_ = Symbol("r'")
    Director = Symbol("Director")
    Actor = Symbol("Actor")
    Equal = Symbol("Equal")
    Ans = Symbol("Ans")

    # Which directors played exactly one role in each of their movies
    exp = Conjunction(
        (
            ExistentialPredicate(m, Director(n, m)),
            UniversalPredicate(
                m_,
                Implication(
                    ExistentialPredicate(
                        r,
                        Conjunction(
                            (
                                Actor(n, m_, r),
                                UniversalPredicate(
                                    r_,
                                    Implication(
                                        Equal(r, r_), Actor(n, m_, r_)
                                    ),
                                ),
                            )
                        ),
                    ),
                    Director(n, m_),
                ),
            ),
        )
    )

    exp2 = convert_to_srnf(exp)
    res = convert_srnf_to_horn_clauses(Ans(n), exp2)

    Aux1 = res.formulas[2].head.functor
    Aux2 = res.formulas[1].head.functor
    Aux3 = res.formulas[0].head.functor

    expected = Union(
        (
            HornClause(
                Aux3(r, n, m_),
                Conjunction(
                    (Actor(n, m_, r), Actor(n, m_, r_), Negation(Equal(r, r_)))
                ),
            ),
            HornClause(
                Aux2(n, m_),
                Conjunction((Actor(n, m_, r), Negation(Aux3(r, n, m_)))),
            ),
            HornClause(
                Aux1(n), Conjunction((Director(n, m_), Negation(Aux2(n, m_))))
            ),
            HornClause(
                Ans(n), Conjunction((Director(n, m), Negation(Aux1(n))))
            ),
        )
    )
    assert res == expected


def test_convert_srnf2horn_fails():
    P = Symbol("P")
    X = Symbol("X")
    Y = Symbol("Y")
    Ans = Symbol("Ans")

    exp = Negation(P(X))
    with pytest.raises(NeuroLangTranslateToHornClauseException):
        convert_srnf_to_horn_clauses(Ans(X), exp)

    exp = UniversalPredicate(Y, P(Y))
    with pytest.raises(NeuroLangTranslateToHornClauseException):
        convert_srnf_to_horn_clauses(Ans(X), exp)


def test_convert_srnf2horn_disjunction():
    x = Symbol("x")
    P = Symbol("P")
    Q = Symbol("Q")
    Ans = Symbol("Ans")
    program = fol_query_to_datalog_program(Ans(x), Disjunction((P(x), Q(x))))
    aux = program.expressions[0].consequent.functor
    expected = ExpressionBlock(
        (
            Implication(aux(x), P(x)),
            Implication(aux(x), Q(x)),
            Implication(Ans(x), aux(x)),
        )
    )
    assert program == expected


class Datalog(
    TranslateToLogic, DatalogProgramNegation, ExpressionBasicEvaluator
):
    def function_gt(self, x: int, y: int) -> bool:
        return x > y


class Chase(NegativeFactConstraints, Chase_):
    pass


def test_safe_range_queries_in_datalog_solver():
    x = Symbol("x")
    G = Symbol("G")
    T = Symbol("T")
    V = Symbol("V")

    program = fol_query_to_datalog_program(
        G(x), Conjunction((V(x), Negation(T(x))))
    )

    dl = Datalog()
    dl.walk(program)
    dl.add_extensional_predicate_from_tuples(V, {(1,), (2,), (3,)})
    dl.add_extensional_predicate_from_tuples(T, {(1,), (4,)})

    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    assert solution_instance["V"].value == {(1,), (2,), (3,)}
    assert solution_instance["T"].value == {(1,), (4,)}
    assert solution_instance["G"].value == {(2,), (3,)}


def test_safe_range_queries_in_datalog_solver_2():
    Movies = Symbol("Movies")
    H = Constant("Hitchcock")
    Xt = Symbol("Xt")
    Xd = Symbol("Xd")
    Xa = Symbol("Xa")
    Ya = Symbol("Ya")
    Yd = Symbol("Yd")
    Zt = Symbol("Zt")
    Ans = Symbol("Ans")

    program = fol_query_to_datalog_program(
        Ans(Xt),
        Conjunction(
            (
                ExistentialPredicate(
                    Xd, ExistentialPredicate(Xa, Movies(Xt, Xd, Xa))
                ),
                UniversalPredicate(
                    Ya,
                    Implication(
                        ExistentialPredicate(Zt, Movies(Zt, H, Ya)),
                        ExistentialPredicate(Yd, Movies(Xt, Yd, Ya)),
                    ),
                ),
            )
        ),
    )

    dl = Datalog()
    dl.walk(program)
    dl.add_extensional_predicate_from_tuples(
        Movies,
        {
            ("Psycho", "Hitchcock", "Hitchcock"),
            ("Vertigo", "Hitchcock", "Hitchcock"),
            ("Rope", "Hitchcock", "Hitchcock"),
            ("Rope", "Hitchcock", "X"),
            ("The Apartment", "Wilder", "Lemmon"),
            ("Sabrina", "Wilder", "X"),
            ("Sunset Boulevard", "Wilder", "Holden"),
            ("Sunset Boulevard", "Wilder", "Swanson"),
            ("Mulholland Drive", "Lynch", "Watts"),
            ("Twin Peaks", "Lynch", "MacLachlan"),
            ("Twin Peaks", "Lynch", "Lynch"),
            ("Manhattan", "Allen", "X"),
            ("Everything You Always Wanted to Know", "Allen", "Allen"),
            ("Everything You Always Wanted to Know", "Allen", "Lasser"),
        },
    )

    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    assert solution_instance["Ans"].value == {
        ("Psycho",),
        ("Vertigo",),
        ("Rope",),
        ("Sabrina",),
        ("Manhattan",),
    }


def test_safe_range_queries_in_datalog_solver_3():
    n = Symbol("n")
    m = Symbol("m")
    m_ = Symbol("m_")
    r = Symbol("r")
    r_ = Symbol("r_")
    Director = Symbol("Director")
    Actor = Symbol("Actor")
    equals = Constant(operator.eq)
    Ans = Symbol("Ans")

    # Which directors played exactly one role in each of their movies
    program = fol_query_to_datalog_program(
        Ans(n),
        Conjunction(
            (
                ExistentialPredicate(m, Director(n, m)),
                UniversalPredicate(
                    m_,
                    Implication(
                        ExistentialPredicate(
                            r,
                            Conjunction(
                                (
                                    Actor(n, m_, r),
                                    UniversalPredicate(
                                        r_,
                                        Implication(
                                            equals(r, r_), Actor(n, m_, r_)
                                        ),
                                    ),
                                )
                            ),
                        ),
                        Director(n, m_),
                    ),
                ),
            )
        ),
    )

    dl = Datalog()
    dl.walk(program)
    dl.add_extensional_predicate_from_tuples(
        Director,
        {
            ("Hitchcock", "Psycho"),
            ("Hitchcock", "Vertigo"),
            ("Hitchcock", "Rope"),
            ("Wilder", "The Apartment"),
            ("Wilder", "Sabrina"),
            ("Wilder", "Sunset Boulevard"),
            ("Lynch", "Mulholland Drive"),
            ("Lynch", "Twin Peaks"),
            ("Allen", "Manhattan"),
            ("Allen", "Everything You Always Wanted to Know"),
        },
    )
    dl.add_extensional_predicate_from_tuples(
        Actor,
        {
            ("Hitchcock", "Psycho", "Man Outside Real Estate Office"),
            ("Hitchcock", "Vertigo", "Man Walking Past Elsters Office"),
            ("Hitchcock", "Rope", "Man Walking in Street"),
            ("Lynch", "Twin Peaks", "FBI Chief"),
            ("Allen", "Manhattan", "Isaac"),
            ("Allen", "Everything You Always Wanted to Know", "Victor"),
            ("Allen", "Everything You Always Wanted to Know", "Fabrizzio"),
        },
    )
    roles = [
        "Man Outside Real Estate Office",
        "Man Walking Past Elsters Office",
        "Man Walking in Street",
        "FBI Chief",
        "Isaac",
        "Victor",
        "Fabrizzio",
    ]
    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    assert solution_instance["Ans"].value == {("Hitchcock",)}


class Datalog2(
    TranslateToLogic,
    Fol2DatalogMixin,
    DatalogProgramNegation,
    ExpressionBasicEvaluator,
):
    pass


def test_fol2datalog_mixin_trivial_case():
    x = Symbol("x")
    G = Symbol("G")
    T = Symbol("T")
    V = Symbol("V")

    dl = Datalog2()
    dl.walk(
        ExpressionBlock(
            (
                Fact(T(Constant(1))),
                Fact(T(Constant(4))),
                Fact(V(Constant(1))),
                Fact(V(Constant(2))),
                Fact(V(Constant(3))),
                Implication(G(x), Conjunction((V(x), Negation(T(x))))),
            )
        )
    )
    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    assert solution_instance["V"].value == {(1,), (2,), (3,)}
    assert solution_instance["T"].value == {(1,), (4,)}
    assert solution_instance["G"].value == {(2,), (3,)}


def test_fol2datalog_mixin_disjunction():
    x = Symbol("x")
    G = Symbol("G")
    T = Symbol("T")
    V = Symbol("V")

    dl = Datalog2()
    dl.walk(
        ExpressionBlock(
            (
                Fact(T(Constant(1))),
                Fact(T(Constant(4))),
                Fact(V(Constant(1))),
                Fact(V(Constant(2))),
                Fact(V(Constant(3))),
                Implication(G(x), Disjunction((V(x), T(x)))),
            )
        )
    )
    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    assert solution_instance["V"].value == {(1,), (2,), (3,)}
    assert solution_instance["T"].value == {(1,), (4,)}
    assert solution_instance["G"].value == {(1,), (2,), (3,), (4,)}


def test_fol2datalog_mixin_complex_formula():
    x = Symbol("x")
    y = Symbol("y")
    G = Symbol("G")
    T = Symbol("T")
    R = Symbol("R")
    V = Symbol("V")

    dl = Datalog2()
    dl.walk(
        ExpressionBlock(
            (
                Fact(T(Constant(1))),
                Fact(T(Constant(4))),
                Fact(V(Constant(1))),
                Fact(V(Constant(2))),
                Fact(V(Constant(3))),
                Fact(R(Constant(4), Constant(5))),
                Fact(R(Constant(2), Constant(6))),
                Implication(
                    G(x),
                    Disjunction(
                        (
                            T(x),
                            ExistentialPredicate(
                                y, Conjunction((V(y), R(y, x),))
                            ),
                        )
                    ),
                ),
            )
        )
    )
    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    assert solution_instance["V"].value == {(1,), (2,), (3,)}
    assert solution_instance["T"].value == {(1,), (4,)}
    assert solution_instance["R"].value == {(4, 5), (2, 6)}
    assert solution_instance["G"].value == {(1,), (4,), (6,)}


def test_fol2datalog_unsafe_disjunction():
    x = Symbol("x")
    y = Symbol("y")
    G = Symbol("G")
    T = Symbol("T")
    V = Symbol("V")

    dl = Datalog2()
    with pytest.raises(Fol2DatalogTranslationException):
        dl.walk(
            ExpressionBlock((Implication(G(x), Disjunction((V(y), T(x)))),))
        )


def test_fol2datalog_safe_universal_usage():
    x = Symbol("x")
    y = Symbol("y")
    G = Symbol("G")
    T = Symbol("T")
    R = Symbol("R")
    V = Symbol("V")

    dl = Datalog2()
    dl.walk(
        ExpressionBlock(
            (
                Fact(V(Constant(1))),
                Fact(V(Constant(2))),
                Fact(V(Constant(3))),
                Fact(T(Constant(1))),
                Fact(T(Constant(4))),
                Fact(R(Constant(2), Constant(1))),
                Fact(R(Constant(2), Constant(4))),
                Implication(
                    G(x),
                    Conjunction(
                        (
                            V(x),
                            UniversalPredicate(y, Implication(R(x, y), T(y))),
                        )
                    ),
                ),
            )
        )
    )
    dc = Chase(dl)
    solution_instance = dc.build_chase_solution()

    assert solution_instance["G"].value == {(2,)}


def test_fol2datalog_unsafe_complex_formula():
    x = Symbol("x")
    y = Symbol("y")
    G = Symbol("G")
    T = Symbol("T")
    R = Symbol("R")

    dl = Datalog2()
    with pytest.raises(Fol2DatalogTranslationException):
        dl.walk(
            ExpressionBlock(
                (
                    Implication(
                        G(x),
                        Disjunction(
                            (
                                T(x),
                                UniversalPredicate(y, Conjunction((R(y, x),))),
                            )
                        ),
                    ),
                )
            )
        )
