"""Tests for the SQUALL grammar metadata projection."""
import json

import lark

from ..squall_syntax_lark import (
    COMPILED_GRAMMAR,
    GRAMMAR,
    GRAMMAR_PATH,
    SquallTransformer
)


def _metadata():
    return SquallTransformer.squall_grammar_metadata()


def test_squall_grammar_metadata_shape():
    """The metadata dict is JSON-serializable and self-consistent."""
    info = _metadata()
    assert json.dumps(info)  # JSON-serializable

    assert info["name"] == "neurolang_natural.lark"
    assert info["grammar"] == GRAMMAR
    assert info["grammar"]  # non-empty

    parser = info["parser"]
    assert parser["library"] == "lark"
    assert parser["version"] == lark.__version__
    assert parser["mode"] == "earley"
    assert parser["ambiguity"] == "resolve"


def test_squall_grammar_metadata_matches_compiled_grammar():
    """Metadata reflects the grammar actually used by the parser."""
    info = _metadata()
    options = COMPILED_GRAMMAR.options
    assert info["parser"]["mode"] == options.parser
    assert info["parser"]["ambiguity"] == options.ambiguity
    assert info["grammar"] == open(GRAMMAR_PATH).read()
