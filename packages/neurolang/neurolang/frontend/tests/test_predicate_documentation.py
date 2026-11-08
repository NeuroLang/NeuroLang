"""The engine keeps the documentation of the predicates of a program."""

import pytest

from nl_expr.exceptions import InvalidCommandExpression
from nl_expr.expressions import Command, Constant, Symbol
from ...commands import PredicateDocumentation
from .. import NeurolangDL, NeurolangPDL

PROGRAM = """
%% The tracts.
tract(1, "left")
tract(2, "right")
other(3, "left")
%% uncinate(x: streamline, h: hemisphere)
%% The uncinate fasciculus.
uncinate(x, h) :- tract(x, h)
uncinate(x, h) :- other(x, h)
"""


@pytest.mark.parametrize("engine", [NeurolangDL, NeurolangPDL])
def test_program_documents_its_predicates(engine):
    nl = engine()
    nl.execute_datalog_program(PROGRAM)
    assert nl.predicate_documentation() == {
        "uncinate": PredicateDocumentation(
            "The uncinate fasciculus.",
            (("x", "streamline"), ("h", "hemisphere")),
        ),
        "tract": PredicateDocumentation("The tracts.", ()),
    }


def test_program_with_a_query_documents_its_rules():
    nl = NeurolangDL()
    result = nl.execute_datalog_program(
        PROGRAM + '%% A query.\nans(x) :- uncinate(x, "left")'
    )
    assert set(result.as_pandas_dataframe().iloc[:, 0]) == {1, 3}
    assert set(nl.predicate_documentation()) == {"uncinate", "tract"}


def test_several_blocks_add_up_once_each():
    nl = NeurolangDL()
    nl.execute_datalog_program(PROGRAM)
    nl.execute_datalog_program(
        "%% Also from other.\nuncinate(x, h) :- other(x, h)"
    )
    nl.execute_datalog_program(PROGRAM)
    documentation = nl.predicate_documentation()["uncinate"]
    assert documentation.description == (
        "The uncinate fasciculus.\n\nAlso from other."
    )
    assert documentation.arguments == (
        ("x", "streamline"), ("h", "hemisphere")
    )


def test_a_description_of_several_paragraphs_is_kept_once():
    nl = NeurolangDL()
    program = "%% First.\n%%\n%% Second.\na(1)"
    nl.execute_datalog_program(program)
    nl.execute_datalog_program(program)
    assert nl.predicate_documentation()["a"].description == (
        "First.\n\nSecond."
    )


def test_documentation_follows_the_scopes():
    nl = NeurolangDL()
    nl.execute_datalog_program("%% The a.\na(1)")
    with nl.scope:
        nl.execute_datalog_program("%% The b.\nb(x) :- a(x)")
        nl.execute_datalog_program("%% More a.\na(2)")
        assert set(nl.predicate_documentation()) == {"a", "b"}
        assert nl.predicate_documentation()["a"].description == (
            "The a.\n\nMore a."
        )
    assert nl.predicate_documentation() == {
        "a": PredicateDocumentation("The a.")
    }


def test_documentation_of_a_removed_predicate_is_gone():
    nl = NeurolangDL()
    nl.execute_datalog_program("%% The a.\na(1)")
    del nl.program_ir.symbol_table[Symbol("a")]
    assert nl.predicate_documentation() == {}


def test_python_api_documents_a_predicate():
    nl = NeurolangDL()
    nl.add_tuple_set([(1,)], name="r")
    nl.document_predicate("r", "Some numbers.", [("n", "number")])
    assert nl.predicate_documentation() == {
        "r": PredicateDocumentation("Some numbers.", (("n", "number"),))
    }


def test_document_command_in_a_program():
    nl = NeurolangDL()
    nl.execute_datalog_program('r(1)\n.document(r, "Some numbers.")')
    assert nl.predicate_documentation()["r"].description == "Some numbers."


def test_document_command_needs_a_name_and_a_description():
    nl = NeurolangDL()
    with pytest.raises(InvalidCommandExpression):
        nl.program_ir.walk(
            Command(Symbol("document"), (Constant("r"),), ())
        )


def test_imported_module_documents_its_predicates(tmp_path):
    from ...commands import CommandsMixin

    if not hasattr(CommandsMixin, "import_module"):
        pytest.skip(".import is not available")
    module = tmp_path / "tracts.dl"
    module.write_text(
        "%% A header, a comment.\n\n"
        "%% tract(x: streamline)\n%% The tracts.\ntract(1)\n"
    )
    nl = NeurolangDL()
    with nl.scope:
        nl.execute_datalog_program(f'.import("{module}")')
        assert nl.predicate_documentation() == {
            "tract": PredicateDocumentation(
                "The tracts.", (("x", "streamline"),)
            )
        }
    assert nl.predicate_documentation() == {}
