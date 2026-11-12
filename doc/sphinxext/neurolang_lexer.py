"""
Pygments lexer for NeuroLang's Datalog, as the REPL colours it.

The fragments are those of :func:`neurolang_apps.utils.query_highlight.tokens`,
which the terminal and the REPL in the browser colour too, so that a query
looks the same on a page of the documentation.
"""

from pygments.lexer import Lexer
from pygments.token import (
    Comment, Keyword, Name, Number, Operator, String, Text
)


class NeuroLangLexer(Lexer):

    """Datalog as the REPL colours it."""

    name = "NeuroLang Datalog"
    aliases = ["neurolang"]

    KINDS = {
        "comment": Comment.Single, "keyword": Keyword,
        "operator": Operator, "predicate": Name.Function,
        "string": String, "number": Number, "": Text,
    }

    def get_tokens_unprocessed(self, text):
        from neurolang_apps.utils.query_highlight import tokens

        position = 0
        for line in text.splitlines(keepends=True):
            body = line.rstrip("\n")
            for kind, fragment in tokens(body):
                yield position, self.KINDS[kind], fragment
                position += len(fragment)
            if len(body) < len(line):
                yield position, Text, "\n"
                position += 1


def setup(app):
    from sphinx.highlighting import lexers

    lexers["neurolang"] = NeuroLangLexer()
    return {"parallel_read_safe": True}
