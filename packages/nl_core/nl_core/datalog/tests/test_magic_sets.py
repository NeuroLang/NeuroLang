import operator

from pytest import raises

from nl_expr import expression_walker, expressions
from nl_expr.logic import Negation, Union as LogicUnion
from .. import DatalogProgram, Fact, Implication, Conjunction
from ..aggregation import (
    AGG_COUNT,
    BuiltinAggregationMixin,
    DatalogWithAggregationMixin,
    TranslateToLogicWithAggregation,
)
from ..chase import Chase
from ..expression_processing import extract_logic_free_variables
from ..exceptions import (
    BoundAggregationApplicationError,
    NegationInMagicSetsRewriteError,
    NoConstantPredicateFoundError,
)
from ..magic_sets import (
    AdornedSymbol,
    LeftToRightSIPS,
    ReplaceAdornedSymbolWalker,
    magic_rewrite,
)
from ..negation import DatalogProgramNegationMixin

C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = Implication
Cnj_ = Conjunction
F_ = Fact
Eb_ = expressions.ExpressionBlock


class Datalog(
    TranslateToLogicWithAggregation,
    BuiltinAggregationMixin,
    DatalogWithAggregationMixin,
    DatalogProgramNegationMixin,
    DatalogProgram,
    expression_walker.ExpressionBasicEvaluator,
):
    pass


x = S_("x")
y = S_("y")
z = S_("z")
x1 = S_("x1")
y1 = S_("y1")
rsg = S_("rsg")
up = S_("up")
down = S_("down")
flat = S_("flat")
anc = S_("anc")
par = S_("par")
anc2 = S_("anc2")
anc3 = S_("anc3")
q = S_("q")
a = C_("a")
b = C_("b")
c = C_("c")
d = C_("d")
eq = C_(operator.eq)

aq = AdornedSymbol(q, "f", 0)
aanc2 = AdornedSymbol(anc2, "bf", 0)
aanc = AdornedSymbol(anc, "bf", 0)


def test_adorned_symbol():
    ax = AdornedSymbol(x, 'b', 1)
    ax_ = AdornedSymbol(x, 'b', 1)
    ax__ = AdornedSymbol(a, 'b', 1)

    assert ax.expression == x
    assert ax.adornment == 'b'
    assert ax.number == 1
    assert ax == ax_
    assert ax != ax__

    ax = AdornedSymbol(a, 'b', 1)
    ax_ = AdornedSymbol(a, 'b', 1)
    ax__ = AdornedSymbol(a, 'b', 2)

    assert ax.expression == a
    assert ax.adornment == 'b'
    assert ax.number == 1
    assert ax == ax_
    assert ax != ax__

    with raises(NotImplementedError):
        AdornedSymbol(a, 'b', 1).name


def test_unadorned_expression_walker():
    adorned_code = tuple(
        [
            Imp_(aq(x), aanc2(a, x)),
            Imp_(aanc(x, y), par(x, y)),
            Imp_(aanc(x, y), aanc(x, z) & par(z, y)),
            Imp_(aanc2(x, y), aanc(x, y) & Negation(aanc(x, y))),
        ]
    )
    unadorned_code = ReplaceAdornedSymbolWalker().walk(adorned_code)

    xanc2 = S_("anc2^bf_0")
    xanc = S_("anc^bf_0")
    xq = S_("q^f_0")
    expected_code = tuple(
        [
            Imp_(xq(x), xanc2(a, x)),
            Imp_(xanc(x, y), par(x, y)),
            Imp_(xanc(x, y), xanc(x, z) & par(z, y)),
            Imp_(xanc2(x, y), xanc(x, y) & Negation(xanc(x, y))),
        ]
    )
    assert unadorned_code == expected_code


def test_l2r_sips():
    edb = Eb_(
        [
            F_(par(a, b)),
        ]
    )
    dl = Datalog()
    dl.walk(edb)
    sips = LeftToRightSIPS(dl)

    # Default case
    arcs, _ = sips.creates_arcs(Imp_(q(x), anc(a, x)), aq(x))
    assert arcs == {aanc(a, x): (aq(x),)}

    # par is in EDB and should not be an arc
    arcs, _ = sips.creates_arcs(Imp_(anc(x, y), par(x, y)), aanc(x, y))
    assert arcs == {}

    # par is in EDB and should not be an arc
    arcs, _ = sips.creates_arcs(
        Imp_(anc(x, y), Cnj_((anc(x, z), par(z, y)))), aanc(x, y)
    )
    assert arcs == {aanc(x, z): (aanc(x, y),)}

    # bindings should propagate from predicate to predicate
    arcs, _ = sips.creates_arcs(
        Imp_(anc(x, y), Cnj_((anc(y, x), par(z, y), up(y, z), down(z, x)))),
        aanc(x, y),
    )
    ancfb = AdornedSymbol(anc, "fb", 0)(y, x)
    upbf = AdornedSymbol(up, "bf", 0)(y, z)
    downbb = AdornedSymbol(down, "bb", 0)(z, x)
    assert arcs == {
        ancfb: (aanc(x, y),),
        upbf: (aanc(x, y), ancfb),
        downbb: (aanc(x, y), ancfb, upbf),
    }

    # negative predicates should not be added to the tails of arcs
    arcs, _ = sips.creates_arcs(
        Imp_(anc2(x, y), Cnj_((Negation(anc(x, y)), anc2(y, x)))), aanc2(x, y)
    )
    anc2fb = AdornedSymbol(anc2, "fb", 0)
    assert arcs == {
        Negation(aanc(x, y)): (aanc2(x, y),),
        anc2fb(y, x): (aanc2(x, y),),
    }


def test_resolution_works():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (b, c, d)}


def test_resolution_works_2():
    """
    Reverse Same Generation from Abiteboul et al. 1995.
    Foundations of databases. p.312

    l0:  l  m   n   o   p
    l1:  e  f   g   h   i   j   k
    l2:  a  b   c   d
    """
    edb = Eb_(
        [
            F_(up(C_("a"), C_("e"))),
            F_(up(C_("a"), C_("f"))),
            F_(up(C_("f"), C_("m"))),
            F_(up(C_("g"), C_("n"))),
            F_(up(C_("h"), C_("n"))),
            F_(up(C_("i"), C_("o"))),
            F_(up(C_("j"), C_("o"))),
            F_(flat(C_("g"), C_("f"))),
            F_(flat(C_("m"), C_("n"))),
            F_(flat(C_("m"), C_("o"))),
            F_(flat(C_("p"), C_("m"))),
            F_(down(C_("l"), C_("f"))),
            F_(down(C_("m"), C_("f"))),
            F_(down(C_("g"), C_("b"))),
            F_(down(C_("h"), C_("c"))),
            F_(down(C_("i"), C_("d"))),
            F_(down(C_("p"), C_("k"))),
        ]
    )

    code = Eb_(
        [
            Imp_(q(y), rsg(C_("a"), y)),
            Imp_(rsg(x, y), flat(x, y)),
            Imp_(rsg(x, y), up(x, x1) & rsg(y1, x1) & down(y1, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {(C_("b"),), (C_("c"),), (C_("d"),)}


def test_resolution_works_query_constant():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x, a), anc(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e, a)) for e in (b, c, d)}


def test_resolution_works_query_free():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc2(x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(anc2(x), anc(a, x)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (b, c, d)}


def test_resolution_works_constant_in_edb_atom():
    """
    An atom of the extensional database with a constant binds the
    variables of the atoms after it: ``x`` in ``anc(x, y)``, the magic
    set of ``anc`` holding the children of ``a``.
    """
    edb = Eb_([F_(par(a, b)), F_(par(b, c)), F_(par(c, d))])
    code = Eb_(
        [
            Imp_(q(y), par(a, x) & anc(x, y)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(y), dl)
    assert any(
        rule.consequent.functor.name.startswith("magic_anc^bf")
        and par(a, x) in rule.antecedent.formulas
        for rule in mr.formulas
    )

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)
    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((c,)), C_((d,))}


def test_edb_atom_without_constant_does_not_seed_magic_sets():
    edb = Eb_([F_(par(a, b)), F_(par(b, c)), F_(par(c, d))])
    code = Eb_(
        [
            Imp_(q(y), par(z, x) & anc(x, y)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    with raises(NoConstantPredicateFoundError):
        magic_rewrite(q(y), dl)


def test_magic_rules_leave_out_unconnected_atoms():
    """
    A magic rule keeps only the atoms left of its arc which share a
    variable, maybe through other atoms, with its bound arguments: the
    magic set of ``anc`` does not depend on ``anc2(z, w)``, whose
    product with it would only be projected out.
    """
    edb = Eb_([F_(par(a, b)), F_(par(b, c)), F_(par(c, d))])
    code = Eb_(
        [
            Imp_(q(y), anc2(z, x1) & anc(x, y) & par(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(anc2(x, y), par(x, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(y), dl)
    magic_anc = [
        rule for rule in mr.formulas
        if rule.consequent.functor.name.startswith("magic_anc^")
    ]
    assert len(magic_anc) > 0
    for rule in magic_anc:
        assert all(
            not atom.functor.name.startswith("anc2")
            for atom in rule.antecedent.formulas
        ), rule

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)
    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (c, d)}

def test_resolution_works_builtin():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc2(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(anc2(x, y), anc(x, z) & eq(z, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (b, c, d)}


def test_resolution_works_repeated_predicate_different_constants():
    """
    Calling the same IDB predicate twice with different constants on a
    shared variable adorns the second call with the variable bound, e.g.
    anc^bb(x, d). Its magic set must be derived from the first call,
    not seeded as a non range restricted fact magic_anc^bb(x, d) :- ().
    """
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc(x, c) & anc(x, d)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (a, b)}


def test_resolution_works_repeated_predicate_different_constants_nested():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(y), anc2(a, y)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(anc2(x, y), anc(x, y) & anc(y, c) & anc(y, d)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(y), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((b,))}


def test_resolution_works_constants_in_different_rules():
    """
    Check that every constant of an adorned predicate seeds its magic set.

    anc(a, x) is met once anc^bf was adorned from anc(c, y); its constant
    must still get a magic init rule.
    """
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x, y), anc2(x) & anc3(y)),
            Imp_(anc2(x), anc(a, x)),
            Imp_(anc3(y), anc(c, y)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x, y), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e, d)) for e in (b, c, d)}


def test_resolution_works_aggregation():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc2(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(anc2(x, AGG_COUNT(y)), anc(x, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {(3,)}


def test_bound_aggregation_raises_error():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc2(x, C_(3))),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(anc2(x, AGG_COUNT(y)), anc(x, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    with raises(BoundAggregationApplicationError):
        magic_rewrite(q(x), dl)


def test_resolution_works_negation():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc2(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(up(x, y), par(x, y)),
            Imp_(anc2(x, y), anc(x, y) & ~up(x, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)

    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (c, d)}


def test_complex_negations_raise_errors():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc2(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(anc2(x, y), anc(x, y) & ~(par(y, z) & anc(x, z))),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)

    with raises(NegationInMagicSetsRewriteError):
        magic_rewrite(q(x), dl)


def test_magic_init_rules_are_idb_not_edb():
    """Magic init rules must be registered as IDB, not EDB.

    When ``create_magic_query_inits`` produces a rule like
    ``magic_p(x) :- True``, the ``TranslateToLogic`` layer rewrites
    ``Implication(..., Constant(True))`` into a ``Fact``. Facts are
    stored as ``Constant[AbstractSet]`` (EDB). If two init rules share
    the same functor, the second walk hits ``_is_previously_defined``
    and crashes with "has been previously defined as Fact".

    Regression test for: magic init rules using ``Constant(True)``.
    """
    # Build a program where the magic rewrite produces init rules.
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )
    code = Eb_(
        [
            Imp_(q(x), anc(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )
    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    # Walk the magic rules into a *fresh* solver and inspect the symbol table.
    dl2 = Datalog()
    dl2.walk(mr)

    # Every magic predicate introduced by the init rules must be a Union (IDB),
    # not a Constant (EDB).
    for symb, value in dl2.symbol_table.items():
        if symb.name.startswith("magic_"):
            assert isinstance(value, LogicUnion), (
                f"Expected magic predicate {str(symb)} to be IDB (Union), "
                f"but got {type(value).__name__}"
            )


def test_magic_init_rules_walk_twice_no_crash():
    """Walking the same magic rules twice must not raise.

    When init rules share a functor (e.g. two rules both introduce
    ``magic_mentions``), walking them a second time used to trigger the
    broken ``_is_previously_defined`` error path in the negation mixin.

    Regression test for: double-walk of magic init rules.
    """
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )
    code = Eb_(
        [
            Imp_(q(x), anc(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )
    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl2 = Datalog()
    dl2.walk(mr)          # first walk
    st_after_first = dict(dl2.symbol_table)
    dl2.walk(mr)          # second walk – must not raise
    # Verify idempotency: second walk must not change the symbol table
    assert st_after_first == dict(dl2.symbol_table), (
        "Magic init rules are not idempotent: symbol table changed after "
        "walking magic rules a second time"
    )


def _magic_rewrite_and_solve(code, edb, query):
    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(query, dl)

    for rule in mr.formulas:
        assert (
            extract_logic_free_variables(rule.consequent) <=
            extract_logic_free_variables(rule.antecedent)
        ), f"magic rule {rule} is not range restricted"

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)
    return Chase(dl).build_chase_solution()[goal].value


def test_resolution_works_same_predicate_shared_variable_constants():
    """
    ``q(x) :- anc(x, b), anc(x, c)`` binds ``x`` of the second ``anc`` call
    through the first one: its magic rule must be
    ``magic_anc^bb(x, c) :- anc^fb(x, b)``, not the seed
    ``magic_anc^bb(x, c) :- ()`` with ``x`` free.
    """
    edb = Eb_([F_(par(a, b)), F_(par(a, c)), F_(par(b, c)), F_(par(d, b))])
    code = Eb_(
        [
            Imp_(q(x), anc(x, b) & anc(x, c)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )
    res = _magic_rewrite_and_solve(code, edb, q(x))
    assert res == {C_((e,)) for e in (a, d)}


def test_resolution_works_same_predicate_positive_and_negated():
    edb = Eb_([F_(par(a, b)), F_(par(a, c)), F_(par(b, c)), F_(par(d, b))])
    code = Eb_(
        [
            Imp_(q(x), anc(x, c) & Negation(anc(x, b))),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )
    res = _magic_rewrite_and_solve(code, edb, q(x))
    assert res == {C_((b,))}


def test_adorned_symbol_predicates_keep_their_own_rules():
    """
    Rewrite the rules of a predicate whose symbol is an AdornedSymbol.

    The conditional query sugar defines ``anc^cond_num``, an
    ``AdornedSymbol`` of ``anc``. Adorning it again must neither take the
    rules of ``anc`` for its own nor give it the name of an adornment of
    ``anc``.
    """
    anc_num = AdornedSymbol(anc, "cond_num", None)
    edb = Eb_([F_(par(a, b)), F_(par(b, c)), F_(par(c, d)), F_(par(d, a))])
    code = Eb_(
        [
            Imp_(q(x), anc(x, b)),
            Imp_(anc(x, y), anc_num(x, y) & par(y, z)),
            Imp_(anc_num(x, y), par(x, y)),
        ]
    )
    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)
    heads = {rule.consequent.functor for rule in mr.formulas}
    num_heads = [
        head for head in heads
        if isinstance(head, AdornedSymbol) and head.adornment == "cond_num"
    ]
    assert len(num_heads) == 1, heads
    assert num_heads[0].expression != anc
    dl.walk(mr)
    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {(a,)}


# A function of a variable that only the query binds, as the similarity
# of a query text to a study: the rule is solved only for the query texts
# magic sets pass to it.


