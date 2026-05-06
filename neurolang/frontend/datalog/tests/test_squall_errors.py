import pytest

from .... import config
from ..squall_syntax_lark import (
    parser,
    SquallSemanticError,
    UnexpectedCharactersError,
    UnexpectedTokenError,
)

config.disable_expression_type_printing()


class TestSquallAnaphoraErrors:
    def test_the_in_object_position_single_rule(self):
        code = "define as V for every Voxel ?v where a Study reports the Term"
        try:
            parser(code)
            pytest.fail("Expected SquallSemanticError for 'the Term' anaphora")
        except SquallSemanticError as e:
            assert "the term" in str(e).lower()
            assert "term" in str(e).lower()

    def test_the_in_subject_position_single_rule(self):
        code = "define as V for every Voxel ?v where the Term reports a Study"
        try:
            parser(code)
            pytest.fail("Expected SquallSemanticError for 'the Term' anaphora")
        except SquallSemanticError as e:
            assert "the term" in str(e).lower()

    def test_the_resolves_when_in_scope(self):
        code = "define as V for every Voxel ?v where a Study reports the Voxel"
        from ....datalog import Implication
        result = parser(code)
        assert isinstance(result, Implication)

    def test_the_anaphora_with_different_noun(self):
        code = "define as V for every Voxel ?v where the Study reports a Term"
        try:
            parser(code)
            pytest.fail("Expected SquallSemanticError for 'the Study' anaphora")
        except SquallSemanticError as e:
            assert "the study" in str(e).lower()

    def test_the_in_compound_quantifier_subject(self):
        code = (
            "define as V for every Voxel ?v and for every Study ?s "
            "where the Term reports a Study"
        )
        try:
            parser(code)
            pytest.fail("Expected SquallSemanticError for 'the Term' in compound")
        except SquallSemanticError as e:
            assert "the term" in str(e).lower()

    def test_the_in_compound_quantifier_object(self):
        code = (
            "define as V for every Voxel ?v and for every Study ?s "
            "where a Voxel reports the Term"
        )
        try:
            parser(code)
            pytest.fail("Expected SquallSemanticError for 'the Term' in compound object pos")
        except SquallSemanticError as e:
            assert "the term" in str(e).lower()


class TestSquallParseErrors:
    def test_gibberish_input(self):
        try:
            parser("foo bar baz qux")
            pytest.fail("Expected parse error for gibberish input")
        except UnexpectedCharactersError:
            pass

    def test_gibberish_after_define(self):
        try:
            parser("define as Verb ~!@#$%")
            pytest.fail("Expected parse error for gibberish after define")
        except UnexpectedCharactersError:
            pass
