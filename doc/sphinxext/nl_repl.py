"""
Queries to run in the REPL, on the pages of the documentation.

``.. nl-repl::`` shows a query, or several statements and dot-commands, as
they are typed in ``neurolang-query -i`` or in the REPL in the browser, and
links it to the REPL in the browser with the query in its box (a share link
of ``doc/_static/repl/share.js``, made by :mod:`repl_links`: the page
runs nothing until Run)::

    .. nl-repl::

       ans(s) :- mentions(s, "pain")

Options: ``:mode: squall`` for SQUALL; ``:engine: NAME`` for another engine
than neurosynth; ``:norun:`` for a block that needs the statements of
earlier blocks, which has no link.

The Datalog is coloured as the REPL colours it, by the ``neurolang`` lexer
(``neurolang_lexer.py``).
"""

from docutils import nodes
from docutils.parsers.rst import Directive, directives

from repl_links import repl_link

DEFAULT_ENGINE = "neurosynth"
RUN_TEXT = "Run in the browser REPL"


class NeuroLangRepl(Directive):

    """A query of the REPL, with a link running it in the browser."""

    has_content = True
    option_spec = {
        "mode": lambda value: directives.choice(value, ("datalog", "squall")),
        "engine": directives.unchanged,
        "norun": directives.flag,
    }

    def run(self):
        code = "\n".join(self.content)
        mode = self.options.get("mode", "datalog")
        block = nodes.literal_block(code, code)
        block["language"] = "squall" if mode == "squall" else "neurolang"
        container = nodes.container(classes=["nl-repl"])
        container += block
        if "norun" not in self.options:
            docname = self.state.document.settings.env.docname
            link = repl_link(
                code, mode, self.options.get("engine", DEFAULT_ENGINE),
                docname.count("/"),
            )
            paragraph = nodes.paragraph(classes=["nl-repl-run"])
            paragraph += nodes.reference(RUN_TEXT, RUN_TEXT, refuri=link)
            container += paragraph
        return [container]


def setup(app):
    app.add_directive("nl-repl", NeuroLangRepl)
    return {"parallel_read_safe": True}
