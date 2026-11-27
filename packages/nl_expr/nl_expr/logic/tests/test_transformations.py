from ...expressions import Symbol
from .. import (
    FALSE,
    TRUE,
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Negation,
    UniversalPredicate
)
from ...expression_walker import ChainedWalker
from ..transformations import (
    AbsorbAssertedConjuncts,
    AbsorbAssertedConjunctsInNegations,
    CollapseConjunctions,
    CollapseDisjunctions,
    DesambiguateQuantifiedVariables,
    DistributeConjunctions,
    DistributeDisjunctions,
    LogicExpressionWalker,
    PushExistentialsDown,
    PushUniversalsDown,
    SimplifyTruthValuesMixin,
    standardise_apart
)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
X = Symbol("X")
Y = Symbol("Y")
Z = Symbol("Z")


PED = PushExistentialsDown()
PUD = PushUniversalsDown()
AACN = AbsorbAssertedConjunctsInNegations()
AAC = AbsorbAssertedConjuncts()


def test_push_existentials_down():
    exp = ExistentialPredicate(
        X,
        Disjunction((P(X), Q(X)))
    )

    assert PED.walk(exp) == Disjunction((
        ExistentialPredicate(X, P(X)),
        ExistentialPredicate(X, Q(X))
    ))

    exp = ExistentialPredicate(
        X, Conjunction((
            P(X), Q(X)
        ))
    )

    assert PED.walk(exp) == exp

    exp = ExistentialPredicate(
        X, Conjunction((
            P(Y), Q(X)
        ))
    )

    assert PED.walk(exp) == Conjunction((
        ExistentialPredicate(X, Q(X)),
        P(Y)
    ))

    exp = ExistentialPredicate(
        X, Conjunction((
            P(Y), Q(X)
        ))
    )

    assert PED.walk(exp) == Conjunction((
        ExistentialPredicate(X, Q(X)),
        P(Y)
    ))

    exp = ExistentialPredicate(
        X, Conjunction((
            P(Y, Z), Negation(Q(X, Z))
        ))
    )

    assert PED.walk(exp) == exp

    exp = ExistentialPredicate(
        X,
        ExistentialPredicate(
            Y,
            Conjunction((
                P(X), Q(Y)
            ))
        )
    )

    assert PED.walk(exp) == Conjunction((
        ExistentialPredicate(X, P(X)),
        ExistentialPredicate(Y, Q(Y))
    ))

    exp = ExistentialPredicate(
        X,
        UniversalPredicate(
            Y,
            Conjunction((P(X), Q(Y)))
        )
    )

    assert PED.walk(exp) == exp


def test_push_existentials_down_keeps_only_the_negations_of_its_variable():
    """The conjuncts without X, a negation among them, are not kept with a
    negation of X when they share no variable with it."""
    exp = ExistentialPredicate(X, Conjunction((
        P(X), Negation(Q(X)), R(Y), Negation(Q(Y))
    )))

    assert PED.walk(exp) == Conjunction((
        ExistentialPredicate(X, Conjunction((P(X), Negation(Q(X))))),
        R(Y), Negation(Q(Y))
    ))


def test_push_existentials_down_a_long_chain_in_polynomial_walks():
    """
    ∃v₁ … ∃v₂₂ ∃X (P(X) ∧ Q(v₁, …, v₂₂, X)), as a rule body with a wide
    atom whose other columns are existential: each vᵢ moves to Q(…), X
    stays. Trying every swap of the chain of the vᵢ, stuck on Q(…), took
    2²² walks.
    """
    class CountingPushExistentialsDown(PushExistentialsDown):
        walks = 0

        def walk(self, expression):
            type(self).walks += 1
            if type(self).walks > 100000:
                raise RecursionError("exponential walk")
            return super().walk(expression)

    variables = tuple(Symbol(f"v{i}") for i in range(22))
    for chain in (variables + (X,), (X,) + variables):
        exp = Conjunction((P(X), Q(*variables, X)))
        for variable in reversed(chain):
            exp = ExistentialPredicate(variable, exp)
        CountingPushExistentialsDown.walks = 0

        res = CountingPushExistentialsDown().walk(exp)

        q_chain = Q(*variables, X)
        for variable in reversed(variables):
            q_chain = ExistentialPredicate(variable, q_chain)
        assert res == ExistentialPredicate(
            X, Conjunction((P(X), q_chain))
        ) or res == ExistentialPredicate(
            X, Conjunction((q_chain, P(X)))
        )
        assert CountingPushExistentialsDown.walks < 100000


def test_push_universals_down():
    exp = UniversalPredicate(
        X,
        Conjunction((P(X), Q(X)))
    )

    assert PUD.walk(exp) == Conjunction((
        UniversalPredicate(X, P(X)),
        UniversalPredicate(X, Q(X))
    ))

    exp = UniversalPredicate(
        X, Disjunction((
            P(X), Q(X)
        ))
    )

    assert PUD.walk(exp) == exp

    exp = UniversalPredicate(
        X, Disjunction((
            P(Y), Q(X)
        ))
    )

    assert PUD.walk(exp) == Disjunction((
        UniversalPredicate(X, Q(X)),
        P(Y)
    ))

    exp = UniversalPredicate(
        X, Disjunction((
            P(Y), Q(X)
        ))
    )

    assert PUD.walk(exp) == Disjunction((
        UniversalPredicate(X, Q(X)),
        P(Y)
    ))

    exp = UniversalPredicate(
        X, Conjunction((
            P(Y, Z), Negation(Q(X, Z))
        ))
    )

    assert PUD.walk(exp) == Conjunction((
        P(Y, Z),
        UniversalPredicate(X, Negation(Q(X, Z)))
    ))

    exp = UniversalPredicate(
        X,
        UniversalPredicate(
            Y,
            Conjunction((
                P(X), Q(Y)
            ))
        )
    )

    assert PUD.walk(exp) == Conjunction((
        UniversalPredicate(X, P(X)),
        UniversalPredicate(Y, Q(Y))
    ))

    exp = UniversalPredicate(
        X,
        ExistentialPredicate(
            Y,
            Conjunction((P(X), Q(Y)))
        )
    )

    assert PUD.walk(exp) == exp


def test_absorb_asserted_conjuncts_in_negation():
    exp = Conjunction((P(X), Negation(Conjunction((P(X), Q(X))))))

    assert AACN.walk(exp) == Conjunction((P(X), Negation(Q(X))))


def test_absorb_asserted_conjuncts_in_negated_existential():
    exp = Conjunction((
        P(X), Q(X, Z),
        Negation(ExistentialPredicate(
            Y, Conjunction((P(X), Q(X, Z), R(X, Y)))
        ))
    ))

    assert AACN.walk(exp) == Conjunction((
        P(X), Q(X, Z),
        Negation(ExistentialPredicate(Y, R(X, Y)))
    ))


def test_absorb_asserted_conjuncts_keeps_bound_variables():
    exp = Conjunction((
        P(Y),
        Negation(ExistentialPredicate(Y, Conjunction((P(Y), Q(X, Y)))))
    ))

    assert AACN.walk(exp) == exp


def test_absorb_asserted_conjuncts_keeps_contradictions():
    exp = Conjunction((P(X), Negation(P(X))))

    assert AACN.walk(exp) == exp

    exp = Conjunction((P(X), Negation(Conjunction((P(X), P(X))))))

    assert AACN.walk(exp) == exp


def test_absorb_asserted_conjuncts_within_quantifiers():
    exp = ExistentialPredicate(X, Conjunction((
        P(X), Negation(Conjunction((P(X), Q(X))))
    )))

    assert AACN.walk(exp) == ExistentialPredicate(
        X, Conjunction((P(X), Negation(Q(X))))
    )


def test_absorb_disjuncts_contradicting_asserted_conjuncts():
    """A ∧ (B ∨ (D ∧ ¬A)) ≡ A ∧ B, the remaining disjunct being
    collapsed into the conjunction."""
    exp = Conjunction((
        P(X), Disjunction((Q(X), Conjunction((R(X), Negation(P(X))))))
    ))

    assert AAC.walk(exp) == Conjunction((P(X), Q(X)))

    exp = Conjunction((
        P(X), Disjunction((
            Conjunction((Q(X), R(X))), Negation(P(X)),
            Conjunction((R(X), Negation(P(X))))
        ))
    ))

    assert AAC.walk(exp) == Conjunction((P(X), Q(X), R(X)))


def test_absorb_self_contradicting_disjuncts():
    exp = Disjunction((Q(X), Conjunction((P(X), R(X), Negation(P(X))))))

    assert AAC.walk(exp) == Q(X)


def test_absorb_contradicting_disjuncts_through_existentials():
    exp = Conjunction((
        P(X), Disjunction((
            Q(X),
            ExistentialPredicate(Y, Conjunction((R(Y), Negation(P(X)))))
        ))
    ))

    assert AAC.walk(exp) == Conjunction((P(X), Q(X)))


def test_absorb_contradicting_disjuncts_keeps_bound_variables():
    exp = Conjunction((
        P(Y), Disjunction((
            Q(X),
            ExistentialPredicate(Y, Conjunction((R(Y), Negation(P(Y)))))
        ))
    ))

    assert AAC.walk(exp) == exp


def test_absorb_contradicting_disjuncts_keeps_contradictions():
    """If every disjunct is a contradiction, the formula is left
    unchanged."""
    exp = Conjunction((
        P(X), Disjunction((
            Negation(P(X)), Conjunction((Q(X), Negation(P(X))))
        ))
    ))

    assert AAC.walk(exp) == exp

    exp = Disjunction((
        Conjunction((P(X), Negation(P(X)))),
        Conjunction((Q(X), Negation(Q(X)))),
    ))

    assert AAC.walk(exp) == exp


def test_absorb_asserted_conjuncts_still_absorbs_in_negations():
    exp = Conjunction((P(X), Negation(Conjunction((P(X), Q(X))))))

    assert AAC.walk(exp) == Conjunction((P(X), Negation(Q(X))))


def test_standardise_apart():
    """Each quantified variable is renamed to a fresh one, the free
    variables are kept."""
    exp = Conjunction((
        P(X),
        ExistentialPredicate(
            X, Conjunction((Q(X), Negation(ExistentialPredicate(X, R(X)))))
        ),
        UniversalPredicate(Y, P(Y)),
    ))
    res = standardise_apart(exp)
    free, outer, universal = res.formulas
    assert free == P(X)
    inner = outer.body.formulas[1].formula
    heads = {outer.head, inner.head, universal.head}
    assert len(heads) == 3 and heads.isdisjoint({X, Y})
    assert outer.body.formulas[0] == Q(outer.head)
    assert inner.body == R(inner.head)
    assert universal.body == P(universal.head)


class SimplifyTruthValues(SimplifyTruthValuesMixin, LogicExpressionWalker):
    pass


def test_simplify_truth_values():
    stv = SimplifyTruthValues()
    assert stv.walk(
        Conjunction((P(X), Negation(Conjunction((Q(X), FALSE)))))
    ) == Conjunction((P(X),))
    assert stv.walk(Negation(TRUE)) == FALSE
    assert stv.walk(
        ExistentialPredicate(Y, Conjunction((Q(Y), FALSE)))
    ) == FALSE
    assert stv.walk(Conjunction((TRUE, TRUE))) == TRUE
    assert stv.walk(Conjunction((P(X), Q(Y)))) == Conjunction((P(X), Q(Y)))


# A formula of this many conjuncts used to exceed the recursion limit
MANY = 1000


_DNF = ChainedWalker(
    DistributeConjunctions, CollapseDisjunctions, CollapseConjunctions
)
_CNF = ChainedWalker(
    DistributeDisjunctions, CollapseConjunctions, CollapseDisjunctions
)


def _atoms(n, symbol=P):
    return tuple(symbol(Symbol(f"x{i}")) for i in range(n))


def test_distributions_of_many_formulas_without_a_dual_are_flat():
    atoms = _atoms(MANY)
    conjunction = Conjunction(atoms)
    disjunction = Disjunction(atoms)
    assert DistributeConjunctions().walk(conjunction) == conjunction
    assert DistributeDisjunctions().walk(disjunction) == disjunction


def test_conjunction_of_many_formulas_distributed_on_a_disjunction():
    atoms = _atoms(MANY)
    exp = Conjunction(atoms[:MANY // 2] + (Disjunction((Q(X), R(X))),)
                      + atoms[MANY // 2:])
    res = _DNF.walk(exp)
    assert res == Disjunction(tuple(
        Conjunction(atoms[:MANY // 2] + (d,) + atoms[MANY // 2:])
        for d in (Q(X), R(X))
    ))


def test_disjunction_of_many_formulas_distributed_on_a_conjunction():
    atoms = _atoms(MANY)
    exp = Disjunction(atoms[:MANY // 2] + (Conjunction((Q(X), R(X))),)
                      + atoms[MANY // 2:])
    res = _CNF.walk(exp)
    assert res == Conjunction(tuple(
        Disjunction(atoms[:MANY // 2] + (c,) + atoms[MANY // 2:])
        for c in (Q(X), R(X))
    ))


def test_grouped_runs_keep_the_order_of_the_distribution():
    """(a ∧ b) ∧ (c ∨ d) ∧ e ∧ (f ∨ g) as the formula folded on the right."""
    a, b, e = P(X), P(Y), P(Z)
    c, d, f, g = Q(X), Q(Y), R(X), R(Y)
    exp = Conjunction((a, b, Disjunction((c, d)), e, Disjunction((f, g))))
    res = _DNF.walk(exp)
    assert res == Disjunction(tuple(
        Conjunction((a, b, x, e, y)) for y in (f, g) for x in (c, d)
    ))


def test_grouped_runs_do_not_hide_what_becomes_a_conjunction():
    """A formula of a run that is walked into the dual is distributed
    before the run is paired with the rest: the clauses are in the order
    the formulas are folded on the right."""
    c1, c2, a, s, t, u = (
        Symbol(n)(X) for n in ("C1", "C2", "A", "S", "T", "U")
    )
    exp = Disjunction((
        Conjunction((c1, c2)), a, Disjunction((s, Conjunction((t, u))))
    ))
    res = _CNF.walk(exp)
    assert res == Conjunction(tuple(
        Disjunction((c, a, s, d)) for d in (t, u) for c in (c1, c2)
    ))


def test_standardising_many_quantified_conjuncts():
    exp = Conjunction(tuple(
        ExistentialPredicate(X, P(X)) for _ in range(400)
    ))
    res = DesambiguateQuantifiedVariables().walk(exp)
    heads = [formula.head for formula in res.formulas]
    assert len(set(heads)) == 400
    assert all(
        formula.body == P(formula.head) for formula in res.formulas
    )
