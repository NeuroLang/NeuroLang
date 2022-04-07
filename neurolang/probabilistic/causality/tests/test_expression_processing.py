import pytest

from ....expressions import Constant, Symbol
from ....logic import Conjunction, Implication, Negation, Union
from ...exceptions import MalformedCausalOperatorError
from ...expressions import Condition
from ..expressions import CausalIntervention, CausalInterventionIdentification, CausalInterventionRewriter

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
S = Symbol("S")
T = Symbol("T")
U = Symbol("U")
ans = Symbol("ans")

x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
a = Constant("a")
b = Constant("b")

def test_symbols_not_allowed():
    with pytest.raises(MalformedCausalOperatorError, match="The atoms intervened by the*"):
        Condition(
            P(x),
            Conjunction((
                CausalIntervention((
                    Q(x),
                )),
            ))
        )

def test_do_instantiation():
    imp = Condition(
        P(x),
        Conjunction((
            CausalIntervention((
                Q(a),
            )),
        ))
    )
    ciw = CausalInterventionIdentification()

    ciw.walk(imp)
    assert ciw.intervention == CausalIntervention((Q(a),))


def test_do_instantiation_more_atoms():
    imp = Condition(
        P(x),
        Conjunction((
            CausalIntervention((
                Q(a),R(b)
            )),
        ))
    )
    ciw = CausalInterventionIdentification()

    ciw.walk(imp)
    assert ciw.intervention == CausalIntervention((Q(a),R(b)))

def test_multiple_instantiation_exception():
    imp = Condition(
        P(x),
        Conjunction((
            CausalIntervention((
                Q(a),
            )),
            CausalIntervention((
                R(a),
            )),
        ))
    )
    ciw = CausalInterventionIdentification()
    with pytest.raises(MalformedCausalOperatorError, match="The use of more than one DO operator*"):
        ciw.walk(imp)

def test_nothing_to_rewrite():
    ciw = CausalInterventionIdentification()
    imp1 = Implication(Q(x), P(x))
    imp2 = Implication(
        ans(x),
        Condition(
            R(x),
            Conjunction((
                    Q(a),
            ))
        )
    )

    _ = ciw.walk(Union((imp1, imp2)))
    assert not hasattr(ciw, 'intervention')


def test_simple_rewrite():
    ciw = CausalInterventionIdentification()
    imp1 = Implication(Q(x), P(x))
    imp2 = Implication(
        ans(x),
        Condition(
            R(x),
            Conjunction((
                CausalIntervention((
                    Q(a),
                )),
            ))
        )
    )

    formulas = ciw.walk(Union((imp1, imp2)))

    cir = CausalInterventionRewriter(ciw.intervention)
    new_formulas = cir.walk(formulas)

    plain_unions = []
    for f1 in new_formulas.formulas:
        if isinstance(f1, Union):
            for f2 in f1.formulas:
                plain_unions.append(f2)
        else:
            plain_unions.append(f1)

    new_formulas = Union(tuple(plain_unions))

    assert len(cir.intervention.formulas) == 1
    intervention = cir.intervention.formulas[0]
    new_f1 = cir.intervention_replacement[intervention.functor]
    new_f2 = cir.new_head_replacement[intervention.functor]

    assert new_f1.is_fresh
    assert new_f2.is_fresh

    new_f1 = new_f1(x)
    new_f2 = new_f2(x)

    new_imp1 = Implication(new_f2, new_f1)
    new_imp2 = Implication(
        ans(x),
        Condition(
            R(x),
            Conjunction((
                Q(a),
            ))
        )
    )
    new_rule = Implication(new_f2, Conjunction((P(x), Negation(new_f1))))
    new_fact = new_f1.functor(*intervention.args)

    assert len(cir.new_facts) == 1
    assert new_fact == list(cir.new_facts)[0]
    assert len(new_formulas.formulas) == 3
    assert new_imp1 in new_formulas.formulas
    assert new_imp2 in new_formulas.formulas
    assert new_rule in new_formulas.formulas


def test_rewrite_two_implications():
    ciw = CausalInterventionIdentification()
    imp1 = Implication(Q(z), P(z))
    imp2 = Implication(
        ans(x),
        Condition(
            R(x),
            Conjunction((
                CausalIntervention((
                    Q(a),
                )),
            ))
        )
    )
    imp3 = Implication(Q(x), S(x, y))

    formulas = ciw.walk(Union((imp1, imp2, imp3)))

    cir = CausalInterventionRewriter(ciw.intervention)
    new_formulas = cir.walk(formulas)

    plain_unions = set()
    for f1 in new_formulas.formulas:
        if isinstance(f1, Union):
            for f2 in f1.formulas:
                plain_unions.add(f2)
        else:
            plain_unions.add(f1)

    new_formulas = Union(tuple(plain_unions))

    assert len(cir.intervention.formulas) == 1
    intervention = cir.intervention.formulas[0]
    new_f1 = cir.intervention_replacement[intervention.functor]
    new_f2 = cir.new_head_replacement[intervention.functor]

    assert new_f1.is_fresh
    assert new_f2.is_fresh

    new_f2_rule_z = new_f2(z)
    new_f2_rule_x = new_f2(x)

    new_f1_rule_z = new_f1(z)
    new_f1_rule_x = new_f1(x)

    new_imp1_x = Implication(new_f2_rule_x, new_f1_rule_x)
    new_imp1_z = Implication(new_f2_rule_z, new_f1_rule_z)
    new_imp2 = Implication(
        ans(x),
        Condition(
            R(x),
            Conjunction((
                Q(a),
            ))
        )
    )
    new_rule1 = Implication(new_f2_rule_z, Conjunction((P(z), Negation(new_f1_rule_z))))
    new_rule2 = Implication(new_f2_rule_x, Conjunction((S(x, y), Negation(new_f1_rule_x))))
    new_fact = new_f1(*intervention.args)

    assert len(cir.new_facts) == 1
    assert new_fact == list(cir.new_facts)[0]
    assert len(new_formulas.formulas) == 4
    assert (new_imp1_x in new_formulas.formulas) or (new_imp1_z in new_formulas.formulas)
    assert new_imp2 in new_formulas.formulas
    assert new_rule1 in new_formulas.formulas
    assert new_rule2 in new_formulas.formulas

def test_rewrite_two_interventions():
    ciw = CausalInterventionIdentification()
    imp1 = Implication(Q(z), P(z))
    imp2 = Implication(
        ans(x),
        Condition(
            T(x),
            Conjunction((
                CausalIntervention((
                    Q(a),R(a, b)
                )),
            ))
        )
    )
    imp3 = Implication(R(x, y), S(y, x))
    imp4 = Implication(T(z), U(y, z))

    formulas = ciw.walk(Union((imp1, imp2, imp3, imp4)))

    cir = CausalInterventionRewriter(ciw.intervention)
    new_formulas = cir.walk(formulas)

    plain_unions = set()
    for f1 in new_formulas.formulas:
        if isinstance(f1, Union):
            for f2 in f1.formulas:
                plain_unions.add(f2)
        else:
            plain_unions.add(f1)

    new_formulas = Union(tuple(plain_unions))

    assert len(cir.intervention.formulas) == 2
    intervention_0 = cir.intervention.formulas[0]
    new_f0s = cir.intervention_replacement[intervention_0.functor]
    new_f1s = cir.new_head_replacement[intervention_0.functor]

    intervention_1 = cir.intervention.formulas[1]
    new_f2s = cir.intervention_replacement[intervention_1.functor]
    new_f3s = cir.new_head_replacement[intervention_1.functor]

    assert new_f0s.is_fresh
    assert new_f1s.is_fresh
    assert new_f2s.is_fresh
    assert new_f3s.is_fresh

    new_f0 = new_f0s(z)
    new_f1 = new_f1s(z)

    new_f2 = new_f2s(x, y)
    new_f3 = new_f3s(x, y)

    new_imp_int1 = Implication(new_f1, new_f0)
    new_imp_int2 = Implication(new_f3, new_f2)
    new_imp2 = Implication(
        ans(x),
        Condition(
            T(x),
            Conjunction((
                Q(a),R(a, b)
            ))
        )
    )
    new_rule1 = Implication(new_f1, Conjunction((P(z), Negation(new_f0))))
    new_rule2 = Implication(new_f3, Conjunction((S(y, x), Negation(new_f2))))

    new_fact1 = new_f0s(*intervention_0.args)
    new_fact2 = new_f2s(*intervention_1.args)

    assert len(cir.new_facts) == 2
    assert new_fact1 in list(cir.new_facts)
    assert new_fact2 in list(cir.new_facts)
    assert len(new_formulas.formulas) == 6
    assert new_imp_int1 in new_formulas.formulas
    assert new_imp_int2 in new_formulas.formulas
    assert new_imp2 in new_formulas.formulas
    assert new_rule1 in new_formulas.formulas
    assert new_rule2 in new_formulas.formulas
    assert imp4 in new_formulas.formulas