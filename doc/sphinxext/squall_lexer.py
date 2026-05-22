"""Minimal Pygments lexer for the SQUALL CNL syntax."""

from pygments.lexer import RegexLexer, words
from pygments.token import (
    Comment, Keyword, Name, Number, Operator, Punctuation, String, Text
)


class SquallLexer(RegexLexer):

    """Simple lexer for SQUALL (Semantics with QUantifiers and ALgebra of Logic)."""

    name = "squall"
    aliases = ["squall"]
    filenames = ["*.squall"]

    _keywords = (
        "define", "as", "obtain", "every", "a", "an", "some", "no", "the",
        "that", "if", "and", "or", "not", "with", "for", "is", "are",
        "inferred", "probability", "given", "where",
    )

    tokens = {
        "root": [
            # Comments
            (r"#.*$", Comment.Single),
            # Keywords
            (words(_keywords, suffix=r"\b"), Keyword),
            # Variables: ?var
            (r"\?[A-Za-z_][A-Za-z0-9_]*", Name.Variable),
            # Tilde-prefixed predicates
            (r"~[A-Za-z_][A-Za-z0-9_]*", Name.Function),
            # Capitalized names (concepts)
            (r"[A-Z][A-Za-z0-9_]*", Name.Class),
            # Lowercase names (predicates / relations)
            (r"[a-z_][A-Za-z0-9_]*", Name),
            # Numbers
            (r"\d+(\.\d+)?", Number),
            # Strings
            (r"'[^']*'", String.Single),
            (r'"[^"]*"', String.Double),
            # Operators and punctuation
            (r"[=<>;,.]", Punctuation),
            (r"[+\-*/]", Operator),
            # Whitespace
            (r"\s+", Text),
        ]
    }


def setup(app):
    from sphinx.highlighting import lexers
    lexers["squall"] = SquallLexer()
    return {"parallel_read_safe": True}
