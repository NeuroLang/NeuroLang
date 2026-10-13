"""Tests for the pretty-printer of named relational algebra plans."""
import operator
from typing import AbstractSet

from ...expressions import Constant, FunctionApplication, Symbol
from ...utils import NamedRelationalAlgebraFrozenSet
from ..pretty_printer import PrettyPrinter, pretty_repr, rows_label
from ..relational_algebra import (
    ExtendedProjection,
    FunctionApplicationListMember,
    NameColumns,
    NaturalJoin,
    NaryNaturalJoin,
    Product,
    Projection,
    Selection,
    int2columnint_constant,
    str2columnstr_constant,
)


def named(relation, *names):
    """Return ``ρ[names](π[0, …](relation))`` as TranslateToNamedRA does."""
    return NameColumns(
        Projection(
            relation,
            tuple(int2columnint_constant(i) for i in range(len(names)))
        ),
        tuple(str2columnstr_constant(n) for n in names),
    )


def test_nary_join_one_operand_per_branch():
    plan = three_way_join()
    assert pretty_repr(plan).splitlines() == [
        "⋈",
        "├─ ρ[x, y]",
        "│  └─ π[#0, #1]",
        "│     └─ edge",
        "├─ ρ[y, l]",
        "│  └─ π[#0, #1]",
        "│     └─ label",
        "└─ σ[w > 15]",
        "   └─ ρ[l, w]",
        "      └─ π[#0, #1]",
        "         └─ weight",
    ]


def test_binary_join_and_cross_product():
    join = NaturalJoin(
        named(Symbol("edge"), "x", "y"), named(Symbol("label"), "y", "l")
    )
    cross = NaturalJoin(
        named(Symbol("edge"), "x", "y"), named(Symbol("weight"), "l", "w")
    )
    assert pretty_repr(join).splitlines()[0] == "⋈"
    assert pretty_repr(cross).splitlines()[0] == "×"
    lines = pretty_repr(join).splitlines()
    assert lines[1].startswith("├─ ρ[x, y]")
    assert lines[4].startswith("└─ ρ[y, l]")


def test_product_one_operand_per_branch():
    plan = Product((Symbol("a"), Symbol("b"), Symbol("c")))
    assert pretty_repr(plan).splitlines() == ["×", "├─ a", "├─ b", "└─ c"]


def test_level_indents_every_line():
    plan = Product((Symbol("a"), Symbol("b")))
    printer = PrettyPrinter(indent=2)
    assert printer.format(plan) == "×\n├─ a\n└─ b"
    assert printer.format(plan, level=1) == "  ×\n  ├─ a\n  └─ b"


def test_unnamed_table_leaf():
    table = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(("a",), [(1,), (2,)]),
        verify_type=False,
    )
    assert pretty_repr(table) == "Table(columns=['a']) ‹2 rows›"


def test_named_leaves_show_their_size():
    table = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(("a",), [(1,), (2,), (3,)]),
        verify_type=False,
    )
    plan = Product((table, Symbol("edge"), Symbol("other")))
    printed = pretty_repr(
        plan,
        name_map={id(table): "voxel", Symbol("edge"): "edge"},
        leaf_sizes={Symbol("edge"): 1234567},
    )
    assert printed.splitlines() == [
        "×",
        "├─ voxel ‹3 rows›",
        "├─ edge ‹1,234,567 rows›",
        "└─ other",
    ]


def test_rows_label():
    assert rows_label(1) == "‹1 row›"
    assert rows_label(0) == "‹0 rows›"
    assert rows_label(8675289) == "‹8,675,289 rows›"
    assert rows_label(None) == "‹? rows›"


def three_way_join():
    """Return the plan of ``edge(x, y), label(y, l), weight(l, w), w > 15``."""
    return NaryNaturalJoin((
        named(Symbol("edge"), "x", "y"),
        named(Symbol("label"), "y", "l"),
        Selection(
            named(Symbol("weight"), "l", "w"),
            FunctionApplication(
                Constant(operator.gt),
                (str2columnstr_constant("w"), Constant(15)),
            ),
        ),
    ))


def test_collapse_renamed_base_relations_into_atoms():
    printed = pretty_repr(
        three_way_join(), collapse=True,
        leaf_sizes={Symbol("edge"): 3, Symbol("weight"): 2},
        leaf_arities={Symbol("edge"): 2},
    )
    assert printed.splitlines() == [
        "⋈",
        "├─ edge(x, y) ‹3 rows›",
        "├─ label(y, l)",
        "└─ σ[w > 15]",
        "   └─ weight(l, w) ‹2 rows›",
    ]


def test_no_collapse_prints_full_form():
    printed = pretty_repr(three_way_join(), collapse=False)
    assert "ρ[x, y]" in printed
    assert "π[#0, #1]" in printed
    assert "edge(x, y)" not in printed


def equal_column(column, value):
    """Return the formula ``#column == value``."""
    return FunctionApplication(
        Constant(operator.eq), (int2columnint_constant(column), value)
    )


def test_collapse_selected_and_projected_relation():
    """Check ``ρ[x](π[#0](σ[#1 == 2](edge)))`` is shown as edge(x, 2)."""
    plan = NameColumns(
        Projection(
            Selection(Symbol("edge"), equal_column(1, Constant(2))),
            (int2columnint_constant(0),),
        ),
        (str2columnstr_constant("x"),),
    )
    printed = pretty_repr(
        plan, collapse=True, leaf_sizes={Symbol("edge"): 3},
        leaf_arities={Symbol("edge"): 2},
    )
    assert printed == "edge(x, 2) ‹from 3 rows›"
    repeated = NameColumns(
        Projection(
            Selection(
                Symbol("edge"), equal_column(0, int2columnint_constant(1))
            ),
            (int2columnint_constant(0),),
        ),
        (str2columnstr_constant("x"),),
    )
    assert pretty_repr(repeated, collapse=True) == "edge(x, x)"


def test_collapse_partial_projection_uses_underscores():
    plan = NameColumns(
        Projection(Symbol("edge"), (int2columnint_constant(1),)),
        (str2columnstr_constant("y"),),
    )
    printed = pretty_repr(
        plan, collapse=True, leaf_arities={Symbol("edge"): 3},
        leaf_sizes={Symbol("edge"): 3},
    )
    assert printed == "edge(_, y, _) ‹from 3 rows›"


def test_collapse_skips_trivial_projection_and_identity_members():
    join = NaturalJoin(
        named(Symbol("edge"), "x", "y"), named(Symbol("label"), "y", "l")
    )
    plan = ExtendedProjection(
        Projection(join, tuple(
            str2columnstr_constant(c) for c in ("x", "y", "l")
        )),
        (
            FunctionApplicationListMember(
                str2columnstr_constant("x"), str2columnstr_constant("x")
            ),
            FunctionApplicationListMember(
                FunctionApplication(
                    Constant(operator.mul),
                    (str2columnstr_constant("y"), Constant(2)),
                ),
                str2columnstr_constant("z"),
            ),
        ),
    )
    assert pretty_repr(plan, collapse=True).splitlines() == [
        "π_ext[x, y * 2 -> z]",
        "└─ ⋈",
        "   ├─ edge(x, y)",
        "   └─ label(y, l)",
    ]
    verbose = pretty_repr(plan).splitlines()
    assert verbose[0] == "π_ext[x -> x, y * 2 -> z]"
    assert verbose[1] == "└─ π[x, y, l]"


def test_nary_cross_product():
    plan = NaryNaturalJoin((
        named(Symbol("a"), "x"), named(Symbol("b"), "y"),
        named(Symbol("c"), "z"),
    ))
    assert pretty_repr(plan, collapse=True).splitlines()[0] == "×"
    plan = NaryNaturalJoin((
        named(Symbol("a"), "x"), named(Symbol("b"), "y"),
        named(Symbol("c"), "x"),
    ))
    assert pretty_repr(plan, collapse=True).splitlines()[0] == "⋈"
