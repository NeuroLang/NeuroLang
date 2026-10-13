"""
Module implementing datalog and natural-syntax datalog parsers.
"""
from .squall_syntax_lark import SquallTransformer

squall_grammar_metadata = SquallTransformer.squall_grammar_metadata

__all__ = ["squall_grammar_metadata"]
