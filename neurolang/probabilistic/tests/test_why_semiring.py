import operator
import typing

from ...expressions import Constant, Symbol
from ...relational_algebra import ColumnStr, NamedRelationalAlgebraFrozenSet
from ...relational_algebra_provenance import ProvenanceAlgebraSet
from ..why_semiring import WhySemiringCompiler


def test_simple_add():
    relation = NamedRelationalAlgebraFrozenSet(
        ("_p_", "x"),
        [(Constant(operator.add)(Symbol("a"), Symbol("b")), "neurolang",)],
    )
    provset = ProvenanceAlgebraSet(relation, ColumnStr("_p_"))
    compiler = WhySemiringCompiler()
    result = compiler.walk(provset)
    result = result.relations.projection(result.provenance_column)
    assert len(result) == 1
    result = next(iter(result))
    assert result[0] == frozenset(
        [frozenset([Symbol("a")]), frozenset([Symbol("b")])]
    )


def test_simple_mul():
    relation = NamedRelationalAlgebraFrozenSet(
        ("_p_", "x"),
        [(Constant(operator.mul)(Symbol("a"), Symbol("b")), "neurolang",)],
    )
    provset = ProvenanceAlgebraSet(relation, ColumnStr("_p_"))
    compiler = WhySemiringCompiler()
    result = compiler.walk(provset)
    result = result.relations.projection(result.provenance_column)
    assert len(result) == 1
    result = next(iter(result))
    assert result[0] == frozenset([frozenset([Symbol("a"), Symbol("b")])])


def test_nested_ops():
    relation = NamedRelationalAlgebraFrozenSet(
        ("_p_", "x"),
        [
            (
                Constant(operator.mul)(
                    Constant(operator.add)(Symbol("a"), Symbol("b")),
                    Symbol("c"),
                ),
                "neurolang",
            )
        ],
    )
    provset = ProvenanceAlgebraSet(relation, ColumnStr("_p_"))
    compiler = WhySemiringCompiler()
    result = compiler.walk(provset)
    result = result.relations.projection(result.provenance_column)
    assert len(result) == 1
    assert (
        frozenset(
            [
                frozenset({Symbol("a"), Symbol("c")}),
                frozenset({Symbol("b"), Symbol("c")}),
            ]
        ),
    ) in result
