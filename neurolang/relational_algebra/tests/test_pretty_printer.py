"""Tests for the pretty-printer of named relational algebra plans."""
import operator
from typing import AbstractSet

from ...expressions import Constant, FunctionApplication, Symbol
from ...utils import NamedRelationalAlgebraFrozenSet
from ..pretty_printer import PrettyPrinter, pretty_repr
from ..relational_algebra import (
    NameColumns,
    NaturalJoin,
    NaryNaturalJoin,
    Product,
    Projection,
    Selection,
    str2columnstr_constant,
)


def named(relation, *names):
    """Return ``ρ[names](π[0, …](relation))`` as TranslateToNamedRA does."""
    return NameColumns(
        Projection(
            relation,
            tuple(Constant(i) for i in range(len(names)))
        ),
        tuple(str2columnstr_constant(n) for n in names),
    )


def test_nary_join_one_operand_per_branch():
    plan = NaryNaturalJoin((
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
    assert pretty_repr(plan).splitlines() == [
        "⋈",
        "├─ ρ[x, y]",
        "│  └─ π[0, 1]",
        "│     └─ edge",
        "├─ ρ[y, l]",
        "│  └─ π[0, 1]",
        "│     └─ label",
        "└─ σ[w > 15]",
        "   └─ ρ[l, w]",
        "      └─ π[0, 1]",
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
    assert pretty_repr(table) == "Table(columns=['a'], rows=2)"
