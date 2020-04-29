from ..logic.unification import most_general_unifier, apply_substitution
from ..expressions import Symbol, Constant
from collections import namedtuple


Rule = namedtuple("Rule", "name constructor constituents is_root")


def add_rule(*args, root=False):
    def wrapper(foo):
        foo._rule_pattern = args
        foo._rule_is_root = root
        return foo

    return wrapper


class Grammar:
    def __init__(self):
        self.rules = list(self._get_rules())

    def _get_rules(self):
        for k in dir(self):
            v = getattr(self, k)
            if hasattr(v, "_rule_pattern"):
                yield Rule(k, v, v._rule_pattern, v._rule_is_root)


class ChartParser:
    Edge = namedtuple("Edge", "head rule completed remaining used_edges")

    def __init__(self, grammar):
        self.grammar = grammar

    def recognize(self, string):
        tokens = [Constant[str](t) for t in string.split()]
        chart = self._fill_chart(tokens)
        return any(e.rule.is_root for e in chart[0][len(tokens)])

    def parse(self, string):
        tokens = [Constant[str](t) for t in string.split()]
        chart = self._fill_chart(tokens)
        compl = [
            e
            for e in chart[0][len(tokens)]
            if e.rule.is_root and not e.remaining
        ]
        return [self._build_tree(e) for e in compl]

    def _build_tree(self, edge):
        if edge.used_edges:
            return edge.head(*map(self._build_tree, edge.used_edges))
        return edge.head

    def _fill_chart(self, tokens):
        chart = [
            [[] for _ in range(len(tokens) + 1)]
            for _ in range(len(tokens) + 1)
        ]
        agenda = []
        self._initialize(tokens, chart, agenda)
        while agenda:
            agenda.sort(key=lambda e: (-e[1], -e[2]))
            edge, i, j = agenda.pop()

            self._predict(edge, i, j, chart, agenda)
            self._complete(edge, i, j, chart, agenda)

        return chart

    # For every word wi add the edge
    #   [wi →  • , (i, i+1)]
    #
    def _initialize(self, tokens, chart, agenda):
        for i, t in enumerate(tokens):
            edge = self.Edge(t, None, [], [], [])
            chart[i][i + 1].append(edge)
            agenda.append((edge, i, i + 1))

    # If the chart contains the complete edge
    #   [A → α • , (i, j)]
    # and the grammar contains the production
    #   B → A β
    # then add the self-loop edge
    #   [B →  • A β , (i, i)]
    #
    def _predict(self, edge, i, j, chart, agenda):
        for rule in self.grammar.rules:
            u = _lu.unify(rule.constituents[0], edge.head)
            if u is not None:
                chart[i][i].append(
                    self.Edge(None, rule, [], rule.constituents[:], [])
                )

    # If the chart contains the edges
    #   [A → α • B β , (i, j)]
    #   [B → γ • , (j, k)]
    # then add the new edge
    #   [A → α B • β , (i, k)]
    # where α, β, and γ are (possibly empty) sequences
    # of terminals or non-terminals
    #
    def _complete(self, completed_edge, j, k, chart, agenda):
        for i in range(j + 1):
            for e in chart[i][j]:
                if not e.remaining:
                    continue
                u = _lu.unify(e.remaining[0], completed_edge.head)
                if not u:
                    continue
                # here completed could have the references
                #   to the involved edges
                n_completed = e.completed + [u[1]]
                n_used_edges = e.used_edges + [completed_edge]
                n_remaining = e.remaining[1:]
                if n_remaining:
                    n_remaining = [
                        _lu.substitute(r, u[0]) for r in n_remaining
                    ]
                    new_edge = self.Edge(
                        e.head, e.rule, n_completed, n_remaining, n_used_edges
                    )
                else:
                    new_head = e.rule.constructor(*n_completed)
                    if not new_head:
                        continue
                    new_edge = self.Edge(
                        new_head, e.rule, n_completed, [], n_used_edges
                    )
                    agenda.append((new_edge, i, k))
                chart[i][k].append(new_edge)


class _lu:
    """
    Quick solution to get unification working without function applications.
    """

    @staticmethod
    def unify(a, b):
        w = Symbol("wrapper")
        u = most_general_unifier(w(a), w(b))
        if u is None:
            return
        return u[0], u[1].args[0]

    @staticmethod
    def substitute(exp, sub):
        w = Symbol("wrapper")
        n = apply_substitution(w(exp), sub)
        return n.args[0]
