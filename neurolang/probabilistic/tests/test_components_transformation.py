
from neurolang.utils.orderedset import OrderedSet
from neurolang.logic.expression_processing import extract_logic_free_variables
from neurolang.logic.transformations import (
    CheckConjunctiveQueryWithNegation, GuaranteeConjunction, ExtractConjunctiveQueryWithNegation,
    PushExistentialsDown, RemoveExistentialPredicates, RemoveTrivialOperations
)
from neurolang.probabilistic.dalvi_suciu_lift import convert_ucq_to_ccq
from neurolang.probabilistic.transforms import convert_rule_to_ucq
from neurolang.logic import (
    Conjunction, Disjunction, ExistentialPredicate, Implication
)
from neurolang.expressions import Symbol

ECQ = ExtractConjunctiveQueryWithNegation()
CPC = CheckConjunctiveQueryWithNegation()
REP = RemoveExistentialPredicates()
PED = PushExistentialsDown()
GC = GuaranteeConjunction()

# Tests for RemoveExistentialPredicate, start here
def test_remove_existential():
    S = Symbol('S')
    x = Symbol('x')
    y = Symbol('y')

    rule = ExistentialPredicate(y, S(x, y))
    rule = REP.walk(rule)

    expected = S(x, y)
    assert rule == expected

def test_remove_inner_existential():
    S = Symbol('S')
    x = Symbol('x')
    y = Symbol('y')

    rule = Conjunction((
        ExistentialPredicate(y, S(x, y)),
    ))

    rule = REP.walk(rule)

    expected = Conjunction((
        S(x, y),
    ))
    assert rule == expected

def test_remove_nested_existentials():
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    z = Symbol('z')

    rule = Conjunction((
        ExistentialPredicate(y2,
            ExistentialPredicate(x2,
                Disjunction((
                    T(z, x2), S(x2, y2)
                ))
            )
        ),
        ExistentialPredicate(y1,
            ExistentialPredicate(x1,
                Conjunction((
                    R(z, x1), S(x1, y1)
                ))
            )
        )
    ))

    rule = REP.walk(rule)

    expected = Conjunction((
        Disjunction((
            T(z, x2), S(x2, y2)
        )),
        Conjunction((
            R(z, x1), S(x1, y1)
        )),
    ))
    assert rule == expected

# Tests for CheckPureConjunction, start here
def test_pure_conjunction():
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    z = Symbol('z')

    rule = Conjunction((
         R(z, x1), S(x1, y1), T(z, x2), S(x2, y2)
    ))

    is_pure = CPC.walk(rule)
    assert is_pure

def test_pure_conjunction_with_existentials():
    S = Symbol('S')
    x = Symbol('x')
    y = Symbol('y')

    rule = Conjunction((
        ExistentialPredicate(y, S(x, y)),
    ))

    is_pure = CPC.walk(rule)
    assert is_pure

def test_nested_conjunctions():
    P = Symbol('P')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x = Symbol('x')
    y = Symbol('y')

    rule = Conjunction((
        Conjunction((
             P(x), S(x), R(y)
        )),
        Conjunction((
            T(x), S(y)
        )),
        R(y)
    ))

    is_pure = CPC.walk(rule)
    assert not is_pure

def test_nested_disjunction():
    P = Symbol('P')
    R = Symbol('R')
    S = Symbol('S')
    x = Symbol('x')
    y = Symbol('y')

    rule = Disjunction((
        Conjunction((
             P(x), S(x), R(y)
        )),
    ))

    is_pure = CPC.walk(rule)
    assert not is_pure

def test_expression():
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    z = Symbol('z')

    rule = Conjunction((
        ExistentialPredicate(y2,
            ExistentialPredicate(x2,
                Conjunction((
                    T(z, x2), S(x2, y2)
                ))
            )
        ),
        ExistentialPredicate(y1,
            ExistentialPredicate(x1,
                Conjunction((
                    R(z, x1), S(x1, y1)
                ))
            )
        )
    ))

    is_pure = CPC.walk(rule)
    assert not is_pure

# Tests for IdentifyPureConjunctions, start here
def test_identify_pure_conjunction():
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    z = Symbol('z')

    rule = Conjunction((
         R(z, x1), S(x1, y1), T(z, x2), S(x2, y2)
    ))

    expected = OrderedSet([rule])

    conjunctions = ECQ.walk(rule)
    assert conjunctions == expected


def test_identify_existential_conjunctions():
    S = Symbol('S')
    x = Symbol('x')
    y = Symbol('y')

    rule = Conjunction((
        ExistentialPredicate(y, S(x, y)),
    ))

    expected = OrderedSet([rule])

    conjunctions = ECQ.walk(rule)
    assert conjunctions == expected


def test_basic_nested_conjunctions():
    P = Symbol('P')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x = Symbol('x')
    y = Symbol('y')

    rule = Conjunction((
        Conjunction((
             P(x), S(x), R(y)
        )),
        Conjunction((
            T(x), S(y)
        )),
        R(y)
    ))

    expected = OrderedSet([rule.formulas[0], rule.formulas[1]])

    conjunctions = ECQ.walk(rule)
    assert conjunctions == expected

def test_basic_inner_disjunctions():
    P = Symbol('P')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x = Symbol('x')
    y = Symbol('y')

    rule = Conjunction((
        Disjunction((
             P(x), S(x), R(y)
        )),
        Disjunction((
            T(x), S(y)
        )),
        R(y)
    ))

    expected = OrderedSet([])

    conjunctions = ECQ.walk(rule)
    assert conjunctions == expected

def test_basic_nested_mixed():
    P = Symbol('P')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x = Symbol('x')
    y = Symbol('y')

    rule = Conjunction((
        Conjunction((
             P(x), S(x), R(y)
        )),
        Disjunction((
            T(x), S(y)
        )),
        R(y)
    ))

    expected = OrderedSet([rule.formulas[0]])

    conjunctions = ECQ.walk(rule)
    assert conjunctions == expected


def test_basic_nested_mixed_2():
    P = Symbol('P')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x = Symbol('x')
    y = Symbol('y')

    rule = Conjunction((
        P(x),
        Disjunction((
            T(x), S(y)
        )),
        R(y)
    ))

    expected = OrderedSet([])

    conjunctions = ECQ.walk(rule)
    assert conjunctions == expected


def test_nested_existentials():
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    z = Symbol('z')

    rule = Conjunction((
        ExistentialPredicate(y2,
            ExistentialPredicate(x2,
                Conjunction((
                    T(z, x2), S(x2, y2)
                ))
            )
        ),
        ExistentialPredicate(y1,
            ExistentialPredicate(x1,
                Conjunction((
                    R(z, x1), S(x1, y1)
                ))
            )
        )
    ))

    expected = OrderedSet([rule.formulas[0], rule.formulas[1]])

    conjunctions = ECQ.walk(rule)
    assert conjunctions == expected


# Tests for CCQ transformation, start here

def test_ccq_transformation_example_2_12():
    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    y3 = Symbol('y3')
    z = Symbol('z')

    rule = Implication(Q(z), Disjunction((
        Conjunction((
            R(x1), S(x1, y1),
        )),
        Conjunction((
            T(y2), S(x2, y2),
        )),
        Conjunction((
            R(x3), T(y3),
        )),
    )))

    rule = convert_rule_to_ucq(rule)
    rule = convert_ucq_to_ccq(rule)

    c0 = ExistentialPredicate(x1,
        Conjunction((
            ExistentialPredicate(
                y1, S(x1, y1),
            ),
            R(x1),
        ))
    )
    c1 = ExistentialPredicate(y2,
        Conjunction((
            ExistentialPredicate(
                x2, S(x2, y2),
            ),
            T(y2),
        ))
    )
    c2 = ExistentialPredicate(x3, R(x3),)
    c3 = ExistentialPredicate(y3, T(y3),)

    expected = Conjunction((
        Disjunction((
            c0, c1, c3
        )),
        Disjunction((
            c0, c1, c2
        )),
    ))

    assert rule == expected


def test_ccq_no_transformation_conjunction():

    Q = Symbol('Q')
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    z = Symbol('z')

    rule = Implication(Q(z), Conjunction((
         R(z, x1), S(x1, y1), T(z, x2), S(x2, y2)
    )))

    rule_ucq = convert_rule_to_ucq(rule)
    rule_ccq = convert_ucq_to_ccq(rule_ucq)

    RTO = RemoveTrivialOperations()
    implication = RTO.walk(rule)
    consequent, antecedent = implication.unapply()
    head_vars = set(consequent.args)
    existential_vars = set(
        extract_logic_free_variables(antecedent) -
        set(head_vars)
    )

    for a in existential_vars:
        antecedent = ExistentialPredicate(a, antecedent)
    expected = RTO.walk(PED.walk(antecedent))

    assert isinstance(rule_ccq, Conjunction)
    assert isinstance(expected, Conjunction)
    assert len(expected.formulas) == len(rule_ccq.formulas)
    # The formulas are not sorted, but they are always the same
    assert (
        (rule_ccq.formulas[0] == expected.formulas[0] and
        rule_ccq.formulas[1] == expected.formulas[1]) or

        (rule_ccq.formulas[0] == expected.formulas[1] and
        rule_ccq.formulas[1] == expected.formulas[0])
    )
