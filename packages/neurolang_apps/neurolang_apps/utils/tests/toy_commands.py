"""A command for the tests of the commands an engine adds (``solver_mixins``
of ``engines.yaml``): ``.greet("name")`` adds ``greeted(name)``."""

from nl_expr.expression_pattern_matching import add_match
from nl_expr.expression_walker import PatternWalker
from nl_expr.expressions import Command, Constant, Symbol


class GreetingMixin(PatternWalker):
    @add_match(Command(Symbol("greet"), ..., ...))
    def greet(self, command):
        names = self.__dict__.setdefault("_greeted", [])
        names.extend(
            (a.value if isinstance(a, Constant) else a.name,)
            for a in command.args
        )
        self.add_extensional_predicate_from_tuples(Symbol("greeted"), names)


NOT_A_CLASS = 1
