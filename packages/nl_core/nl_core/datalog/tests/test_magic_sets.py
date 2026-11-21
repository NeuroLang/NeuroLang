import operator

import pytest
from pytest import raises

from nl_expr import expression_walker, expressions
from nl_expr.logic import Negation, Union as LogicUnion
from .. import (
    Conjunction, DatalogProgram, Fact, Implication, extract_logic_predicates,
)
from ..aggregation import (
    AGG_COUNT,
    BuiltinAggregationMixin,
    DatalogWithAggregationMixin,
    TranslateToLogicWithAggregation,
)
from ..chase import Chase
from ..expression_processing import (
    extract_logic_free_variables, recursive_components,
)
from ..expressions import AggregationApplication
from ..exceptions import (
    BoundAggregationApplicationError,
    InvalidMagicSetError,
    NegationInMagicSetsRewriteError,
    NewRecursionInMagicSetsRewriteError,
    NoConstantPredicateFoundError,
    UnstratifiedMagicSetsRewriteError,
)
from .. import magic_sets
from ..magic_sets import (
    AdornedSymbol,
    LeftToRightSIPS,
    ReplaceAdornedSymbolWalker,
    check_stratified,
    magic_rewrite,
    needed_positions,
    order_rules_definitions_first,
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
u = S_("u")
v = S_("v")
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


def _magic_and_chase(code, edb, query):
    """The rules magic sets rewrite *code* into, and whether the goal's
    answers are those of the query of *code* without them."""
    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    expected = Chase(dl).build_chase_solution()[query.functor].value
    goal, mr = magic_rewrite(query, dl)
    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)
    assert Chase(dl).build_chase_solution()[goal].value == expected
    return mr


def _heads(rules):
    return {rule.consequent.functor.name for rule in rules.formulas}


sel = S_("sel")
chosen = S_("chosen")
anc_edb = Eb_([
    F_(par(u, v)) for u, v in ((a, b), (b, c), (c, d), (a, d), (b, b))
] + [F_(sel(u)) for u in (a, b, c)])


@pytest.mark.parametrize("body", [
    chosen(x) & anc(x, y),  # bound by a relation
    anc(a, y),  # bound by a constant
    chosen(x) & anc(x, x) & eq(y, x),  # a repeated variable
    chosen(x) & anc(x, y) & anc(y, y),  # used twice
])
def test_restrictions_of_stored_relations_are_unfolded(body):
    """A restriction of a stored relation, ``anc^bf(x, y) :- magic_anc^bf(x),
    par(x, y)``, is not a rule of the rewritten program: its body is in
    the rules that use it, with the answers of the program."""
    code = Eb_([
        Imp_(q(y), body), Imp_(anc(x, y), par(x, y)),
        Imp_(chosen(x), sel(x)),
    ])
    mr = _magic_and_chase(code, anc_edb, q(y))
    assert not any(h.startswith("anc^") for h in _heads(mr))
    assert any(
        atom.functor.name.startswith("magic_anc^")
        for rule in mr.formulas
        for atom in extract_logic_predicates(rule.antecedent)
    )


def test_restrictions_not_unfolded():
    """A restriction used under negation, one of a predicate of several
    rules, of a recursive one, or the goal is kept."""
    negated = Eb_([
        Imp_(q(y), chosen(y) & chosen(x) & Negation(anc(x, y))),
        Imp_(anc(x, y), par(x, y)),
        Imp_(chosen(x), sel(x)),
    ])
    mr = _magic_and_chase(negated, anc_edb, q(y))
    assert any(h.startswith("anc^") for h in _heads(mr))
    recursive = Eb_([
        Imp_(q(y), anc(a, y)),
        Imp_(anc(x, y), par(x, y)),
        Imp_(anc(x, y), anc(x, z) & par(z, y)),
    ])
    mr = _magic_and_chase(recursive, anc_edb, q(y))
    assert any(h.startswith("anc^") for h in _heads(mr))
    dl = Datalog()
    dl.walk(Eb_([Imp_(anc(x, y), par(x, y))]))
    dl.walk(anc_edb)
    goal, mr = magic_rewrite(anc(a, y), dl)
    assert goal.name.startswith("anc^")
    assert goal.name in _heads(mr)


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


@pytest.mark.parametrize("body", [
    lambda: par(z, x) & anc(x, y),
    lambda: anc(x, y) & par(z, x),  # put first
])
def test_edb_atom_of_the_query_rule_binds_without_constant(body):
    """
    In the query's own rule, an atom of the extensional database binds
    its variables, constant or not (the relation the query restricts
    itself to, as ``chosen(t)``), and is put first: ``x`` of
    ``anc(x, y)`` is bound by ``par(z, x)``; the answers are those of the
    program not rewritten.
    """
    edb = Eb_([F_(par(a, b)), F_(par(b, c)), F_(par(c, d))])
    code = Eb_(
        [
            Imp_(q(y), body()),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    expected = Chase(dl).build_chase_solution()[q].value
    goal, mr = magic_rewrite(q(y), dl)
    assert any(
        rule.consequent.functor.name.startswith("magic_anc^bf")
        and par(z, x) in rule.antecedent.formulas
        for rule in mr.formulas
    )

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)
    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == expected


def test_large_edb_atom_of_the_query_rule_does_not_seed_magic_sets():
    """In the query's rule, a stored relation of more than
    ``[MAGIC_SETS] seed_rows`` tuples binds nothing: its magic set would
    restrict little (all the voxels of a mask)."""
    from nl_expr.config import config

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
    previous = config.get("MAGIC_SETS", "seed_rows")
    config.set("MAGIC_SETS", "seed_rows", "2")  # par has 3
    try:
        with raises(NoConstantPredicateFoundError):
            magic_rewrite(q(y), dl)
    finally:
        config.set("MAGIC_SETS", "seed_rows", previous)


def test_edb_atom_of_a_recursive_query_rule_does_not_seed_magic_sets():
    """A query defined recursively (a transitive closure, as a frontend
    unfolds ``ans(x, y) :- reach(x, y)``) keeps no relation of its own:
    its stored relations, steps of the recursion, bind nothing."""
    edb = Eb_([F_(par(a, b)), F_(par(b, c)), F_(par(c, d))])
    code = Eb_(
        [
            Imp_(q(x, y), par(x, y)),
            Imp_(q(x, y), q(x, z) & par(z, y)),
        ]
    )
    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    with raises(NoConstantPredicateFoundError):
        magic_rewrite(q(x, y), dl)


def test_edb_atom_without_constant_does_not_seed_magic_sets():
    """Out of the query's rule, an atom of the extensional database
    without a constant binds nothing: no magic set covers a whole
    relation."""
    r = S_("r")
    edb = Eb_([F_(par(a, b)), F_(par(b, c)), F_(par(c, d))])
    code = Eb_(
        [
            Imp_(q(y), r(y)),
            Imp_(r(y), par(z, x) & anc(x, y)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    with raises(NoConstantPredicateFoundError):
        magic_rewrite(q(y), dl)


def test_constant_in_builtin_does_not_seed_magic_sets():
    """
    ``q(x) :- par(x, y), contains(y, (1, 2))``: the constant is in a
    built-in, a filter, and binds no predicate: the query is not
    rewritten, and is solved as it is.
    """
    edb = Eb_([
        F_(par(a, C_(((1, 2), (3, 4))))), F_(par(b, C_(((3, 4),))))
    ])
    contains = C_(operator.contains)
    code = Eb_([Imp_(q(x), par(x, y) & contains(y, C_((1, 2))))])

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    with raises(NoConstantPredicateFoundError):
        magic_rewrite(q(x), dl)

    solution = Chase(dl).build_chase_solution()
    assert solution[q].value == {C_((a,))}


def test_intensional_atom_without_constant_binds_the_atoms_right_of_it():
    """
    No constant, but ``x`` of ``anc(x, y)`` is bound by ``p2(z, x)``, an
    intensional atom left of it: magic sets restrict ``anc`` to it.
    """
    p2 = S_("p2")
    edb = Eb_([F_(par(a, b)), F_(par(b, c)), F_(par(c, d))])
    code = Eb_(
        [
            Imp_(p2(x, y), par(x, y)),
            Imp_(q(y), p2(z, x) & anc(x, y)),
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
        for rule in mr.formulas
    )

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)
    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((c,)), C_((d,))}


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


def test_rules_ordered_definitions_first():
    """
    A rule comes after the rules defining the predicates of its body,
    taken in passes in the order of the rules; a predicate defined by no
    rule, as ``r``, waits on nothing. When the rules left all wait, as
    those of ``s`` and ``t`` on each other, the first is taken.
    """
    p, q, s, t, r = S_("p"), S_("q"), S_("s"), S_("t"), S_("r")
    x, y = S_("x"), S_("y")
    rules = [
        Imp_(p(x), Cnj_((q(x), s(x)))),
        Imp_(q(x), r(x)),
        Imp_(s(x), Cnj_((t(x, y), q(y)))),
        Imp_(t(x, y), Cnj_((r(x), s(y)))),
        Imp_(q(x), Cnj_((r(x), r(y)))),
        Imp_(s(x), r(x)),
    ]
    ordered = order_rules_definitions_first(rules)
    assert [rules.index(rule) for rule in ordered] == [1, 4, 5, 0, 2, 3]


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




def _cycle_through_an_aggregate_program():
    """A and B both read the aggregate Agg, and R is A then B: the magic
    predicate of B gets its bindings from A, and the one of Agg from B, so
    Agg would depend on A, which depends on Agg."""
    t, s, v, l = S_("t"), S_("s"), S_("v"), S_("l")
    A, B, R, Agg = S_("A"), S_("B"), S_("R"), S_("Agg")  # noqa: N806
    S0, S1, E, Sel = S_("S0"), S_("S1"), S_("E"), S_("Sel")  # noqa: N806
    code = Eb_(
        [
            Imp_(q(s), R(a, s)),
            Imp_(R(t, s), A(t, s) & B(t, s)),
            Imp_(A(t, s), S0(t, s) & Agg(t, v)),
            Imp_(B(t, s), S1(t, s) & Agg(t, v)),
            Imp_(Agg(t, AGG_COUNT(l)), E(t, l) & Sel(t, l)),
        ]
    )
    edb = Eb_(
        [
            F_(S0(a, b)), F_(S0(a, c)), F_(S1(a, c)), F_(S1(a, d)),
            F_(E(a, b)), F_(E(a, c)), F_(Sel(a, b)), F_(Sel(a, c)),
        ]
    )
    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    return dl


def test_magic_rewrite_leaves_unbound_an_aggregation_in_a_cycle():
    """Bound, Agg would be in a cycle of the rewritten program, which would
    not be stratified: the rewrite leaves Agg unbound, computed whole, and
    binds the rest."""
    for seed_from_edb in (True, False):
        dl = _cycle_through_an_aggregate_program()
        goal, mr = magic_rewrite(q(x), dl, seed_from_edb=seed_from_edb)
        magic = {
            rule.consequent.functor.name for rule in mr.formulas
            if rule.consequent.functor.name.startswith("magic_")
        }
        assert not any(name.startswith("magic_Agg^b") for name in magic)
        assert any(name.startswith("magic_B^b") for name in magic)
        dl = _cycle_through_an_aggregate_program()
        dl.walk(mr)
        solution = Chase(dl).build_chase_solution()
        assert solution[goal].value == {(c,)}


def test_the_program_without_the_rewrite_is_solved():
    dl = _cycle_through_an_aggregate_program()
    solution = Chase(dl).build_chase_solution()
    assert solution[S_("R")].value == {(a, c)}


def test_magic_rewrite_keeps_an_aggregation_that_is_not_in_a_cycle():
    """The recursion of anc is a cycle, and the aggregation reads it
    from outside: that is fine, and is rewritten."""
    edb = Eb_([F_(par(a, b)), F_(par(b, c)), F_(par(c, d))])
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
    magic_rewrite(q(x), dl)  # does not raise


@pytest.mark.parametrize("negation_first", [True, False])
def test_an_aggregation_in_a_cycle_is_reported_before_a_negation(
    negation_first,
):
    """
    Of a cycle with an aggregation and a negation in it, the predicate
    reported (which magic sets leave unbound) is the aggregation's, in
    whichever order the rules come: it was the first of a set, which
    changed with the hashes of the symbols, and so did the plan.
    """
    n, g, e, s_ = S_("n"), S_("g"), S_("e"), S_("s")
    negation = Imp_(n(x), Cnj_((s_(x), Negation(g(x, y)))))
    count = AggregationApplication(S_("count"), (y,))
    aggregation = Imp_(g(x, count), Cnj_((n(y), e(x, y))))
    rules = (negation, aggregation) if negation_first else (
        aggregation, negation
    )
    with raises(UnstratifiedMagicSetsRewriteError) as error:
        check_stratified(rules, {n: S_("origin_n"), g: S_("origin_g")})
    assert error.value.predicate == S_("origin_g")


class DatalogWithClose(Datalog):
    def function_close(self, text: str, study: str) -> bool:
        return text[0] == study[0]


def test_a_predicate_bound_through_its_head_is_bound_by_a_stored_relation():
    """
    ``q`` of ``Sim(q, s)`` is bound only through the head: ``Sim`` is a
    relation over the texts its users bind, here ``Def``, a stored
    relation without a constant. It binds ``d`` because ``Sim`` needs it
    (``Sim`` was left unbound, and its rule could not be solved).
    """
    t, s, text = S_("t"), S_("s"), S_("text")
    Sim, Use = S_("Sim"), S_("Use")  # noqa: N806
    Def, Study = S_("Def"), S_("Study")  # noqa: N806
    close = S_("close")
    dl = DatalogWithClose()
    dl.walk(Eb_([
        Imp_(Sim(text, s), Study(s) & close(text, s)),
        Imp_(Use(s), Def(t, text) & Sim(text, s)),
        Imp_(q(s), Use(s)),
    ]))
    dl.walk(Eb_([
        F_(Def(C_("t1"), C_("apple"))), F_(Def(C_("t2"), C_("banana"))),
        F_(Study(C_("a1"))), F_(Study(C_("c1"))), F_(Study(C_("b1"))),
    ]))
    assert needed_positions(dl) == {Sim: {0}}
    goal, mr = magic_rewrite(q(x), dl)
    assert any(
        rule.consequent.functor.name.startswith("Sim^bf")
        for rule in mr.formulas
    )
    dl.walk(mr)
    solution = Chase(dl, rules=mr).build_chase_solution()
    assert solution[goal].value == {(C_("a1"),), (C_("b1"),)}


def _shared_call_sites_program():
    """
    A tract-like program, not recursive: ``A`` reads ``P`` then ``Q``,
    ``B`` reads ``Q`` then ``P``. Bound, ``P`` and ``Q`` each have one
    adorned and one magic predicate for both call sites: ``magic_Q^b``
    reads ``P^bf``, which reads ``magic_P^bf``, which reads ``Q^b``, which
    reads ``magic_Q^b``.
    """
    P, Q, A, B, R = (S_(n) for n in "PQABR")  # noqa: N806
    EP, EQ = S_("EP"), S_("EQ")  # noqa: N806
    dl = Datalog()
    dl.walk(Eb_([
        Imp_(P(x, y), EP(x, y)),
        Imp_(Q(x), EQ(x)),
        Imp_(A(x), P(x, y) & Q(y)),
        Imp_(B(x), Q(x) & P(x, y)),
        Imp_(R(x), A(x) & B(x)),
        Imp_(q(x), R(x) & EQ(x)),
    ]))
    dl.walk(Eb_([
        F_(EP(a, a)), F_(EP(a, b)), F_(EP(b, c)), F_(EQ(a)), F_(EQ(b)),
    ]))
    return dl


def test_magic_rewrite_does_not_put_non_recursive_predicates_in_a_cycle():
    """
    The rewrite of the program of shared call sites would be one
    recursive stratum, rule by rule until nothing changes: it is not done,
    and the program is solved unrewritten.
    """
    for seed_from_edb in (True, False):
        dl = _shared_call_sites_program()
        with raises(NewRecursionInMagicSetsRewriteError) as error:
            magic_rewrite(S_("R")(a), dl, seed_from_edb=seed_from_edb)
        assert "P, Q" in str(error.value)
    solution = Chase(_shared_call_sites_program()).build_chase_solution()
    assert solution[q].value == {(a,)}


@pytest.mark.parametrize("helper", [False, True])
def test_magic_rewrite_keeps_the_recursion_of_the_program(helper):
    """
    The cycles of the rewrite of a transitive closure are its recursion:
    it is rewritten, though the magic predicate of the non-recursive
    ``par2`` is in the cycle of ``magic_anc^bf``.
    """
    par2 = S_("par2")
    edge = par2 if helper else par
    code = [
        Imp_(q(x), anc(a, x)),
        Imp_(anc(x, y), edge(x, y)),
        Imp_(anc(x, y), edge(x, z) & anc(z, y)),
    ]
    if helper:
        code.append(Imp_(par2(x, y), par(x, y)))
    edb = Eb_([F_(par(a, b)), F_(par(b, c)), F_(par(d, a))])
    dl = Datalog()
    dl.walk(Eb_(code))
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)
    assert any(
        rule.consequent.functor.name.startswith("magic_anc^bf")
        for rule in mr.formulas
    )
    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)
    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {(b,), (c,)}


def test_magic_rewrite_stops_at_the_second_cycle_of_its_own(monkeypatch):
    """
    Each aggregation of ``Agg`` and ``Agg2``, bound, is in a cycle of the
    rewrite's own, not the program's: the rewrite leaves the first one
    reported unbound, and gives up at the second rather than leaving
    unbound one predicate after the other, each a whole rewrite; seeding
    from the stored relations or not does not change that.
    """
    t, s, v, w = S_("t"), S_("s"), S_("v"), S_("w")
    A, B, R = S_("A"), S_("B"), S_("R")  # noqa: N806
    Agg, Agg2 = S_("Agg"), S_("Agg2")  # noqa: N806
    S0, S1, E, Sel = S_("S0"), S_("S1"), S_("E"), S_("Sel")  # noqa: N806
    dl = Datalog()
    dl.walk(Eb_([
        Imp_(q(s), R(a, s)),
        Imp_(R(t, s), A(t, s) & B(t, s)),
        Imp_(A(t, s), S0(t, s) & Agg(t, v) & Agg2(t, v)),
        Imp_(B(t, s), S1(t, s) & Agg(t, v) & Agg2(t, v)),
        Imp_(Agg(t, AGG_COUNT(w)), E(t, w) & Sel(t, w)),
        Imp_(Agg2(t, AGG_COUNT(w)), E(t, w)),
    ]))
    dl.walk(Eb_([
        F_(S0(a, b)), F_(S0(a, c)), F_(S1(a, c)), F_(E(a, b)), F_(Sel(a, b)),
    ]))
    calls = []
    rewrite = magic_sets._magic_rewrite

    def counted(*args):
        calls.append(set(args[3]))
        return rewrite(*args)

    monkeypatch.setattr(magic_sets, "_magic_rewrite", counted)
    with raises(NewRecursionInMagicSetsRewriteError):
        magic_rewrite(q(x), dl)
    assert len(calls) == 2
    assert len(calls[1]) == 1 and calls[1] <= {Agg, Agg2}


def test_magic_rewrite_binds_from_constants_a_product_in_a_cycle_of_its_own():
    """
    As in the bundles of tracts (``uf``), the program of shared call sites
    reads ``Above``, the product of ``EV``, a stored relation of more
    than ``[MAGIC_SETS] seed_rows`` tuples, and ``Box`` restricted by a
    comparison: computed whole, it is all the pairs of their rows. The
    rewrite with the bindings of the literals has a cycle of its own; the
    one with the bindings of the constants and the head alone has none,
    binds ``Above`` to its box ``a``, and gives the program's answer.
    """
    P, Q, A, B, R = (S_(n) for n in "PQABR")  # noqa: N806
    EP, EQ, EV, Box, Above = (  # noqa: N806
        S_(n) for n in ("EP", "EQ", "EV", "Box", "Above")
    )
    t, w = S_("t"), S_("w")
    code = Eb_([
        Imp_(P(x, y), EP(x, y)),
        Imp_(Q(x), EQ(x)),
        Imp_(A(x), P(x, y) & Q(y)),
        Imp_(B(x), Q(x) & P(x, y)),
        Imp_(Above(x, t), EV(x, v) & Box(t, w) & C_(operator.gt)(v, w)),
        Imp_(R(x), A(x) & B(x) & Above(x, a)),
        Imp_(q(x), R(x)),
    ])
    edb = Eb_([
        F_(EP(a, a)), F_(EP(a, b)), F_(EP(b, c)), F_(EQ(a)), F_(EQ(b)),
        F_(EV(a, C_(3))), F_(EV(b, C_(1))),
        F_(Box(a, C_(2))), F_(Box(b, C_(0))),
    ] + [F_(EV(C_(f"e{i}"), C_(0))) for i in range(100)])
    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(S_("R")(x), dl)
    assert not list(recursive_components(mr.formulas))
    assert any(
        rule.consequent.functor.name.startswith("Above^fb")
        for rule in mr.formulas
    )
    rewritten = Datalog()
    rewritten.walk(mr)
    rewritten.walk(edb)
    solution = Chase(rewritten).build_chase_solution()
    assert solution[goal].value == {(a,)}
    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    assert Chase(dl).build_chase_solution()[q].value == {(a,)}
