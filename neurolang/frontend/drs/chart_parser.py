from ...logic.unification import most_general_unifier, apply_substitution
from ...expressions import Symbol, Constant
from collections import namedtuple
import re


Quote = Symbol("Quote")
Rule = namedtuple("Rule", "name constructor constituents is_root")

CODE_QUOTE = '`'
STRING_QUOTE = '"'


def add_rule(*args, root=False):
    def wrapper(foo):
        foo._rule_pattern = args
        foo._rule_is_root = root
        return foo

    return wrapper


class Grammar:
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.rules = list(self._get_rules())

    def _get_rules(self):
        for k in dir(self):
            v = getattr(self, k)
            if hasattr(v, "_rule_pattern"):
                yield Rule(k, v, v._rule_pattern, v._rule_is_root)


class Lexicon:
    def __init__(self):
        pass

    def get_meanings(self, word):
        raise NotImplementedError("")


class DictLexicon(Lexicon):
    """Convenience implementation of lexicon over dict"""

    def __init__(self, d):
        self.dict = d

    def get_meanings(self, token):
        if isinstance(token, Constant):
            if token.value in self.dict:
                return self.dict[token.value]
        return ()


class Chart(list):
    pass


class Tokenizer:
    def __init__(self, grammar, quotes=[CODE_QUOTE, STRING_QUOTE]):
        self.grammar = grammar
        self.matches = []
        for q in quotes:
            self.matches.append(
                (
                    re.compile(f"^{q}.+?{q}\\s"),
                    lambda span, q=q: Quote(
                        Constant(q), Constant[str](span[1:-1])
                    ),
                )
            )
        self.matches.append(
            (re.compile("^\\w+?\\s"), lambda span: Constant[str](span),)
        )

    def tokenize(self, string):
        rem = string.strip() + " "
        tokens = []
        while rem:
            t, rem = self.next_token(rem)
            tokens.append(t)
        return tokens

    def next_token(self, text):
        for r, on_match in self.matches:
            m = r.match(text)
            if m:
                e = m.end()
                span = text[:e].strip()
                rem = text[e:].lstrip()
                return on_match(span), rem

        raise Exception(f"Couldnt match token at: {text}")


class ChartParser:
    Edge = namedtuple(
        "Edge", "head rule completed remaining used_edges unification"
    )

    def __init__(self, grammar):
        self.grammar = grammar
        self.tokenizer = Tokenizer(grammar)

    def recognize(self, string):
        tokens = self.tokenizer.tokenize(string)
        self._fill_chart(tokens)
        return any(e.rule.is_root for e in self.chart[0][len(tokens)])

    def parse(self, string):
        tokens = self.tokenizer.tokenize(string)
        self._fill_chart(tokens)
        compl = [
            e
            for e in self.chart[0][len(tokens)]
            if e.rule.is_root and not e.remaining
        ]
        return [self._build_tree(e, e.unification) for e in compl]

    def _build_tree(self, edge, unif):
        head = _lu.substitute(edge.head, unif)
        args = ()
        for ce in edge.used_edges:
            cu = ce.unification.copy()
            cu.update(unif)
            args += (self._build_tree(ce, cu),)

        if args:
            return head(*args)
        return head

    def _fill_chart(self, tokens):
        self._initialize(tokens)
        while self.agenda:
            self.agenda.sort(key=lambda e: (-e[1], -e[2]))
            edge, i, j = self.agenda.pop()

            self._predict(edge, i, j)
            self._complete(edge, i, j)

    # For every word wi add the edge
    #   [wi →  • , (i, i+1)]
    #
    def _initialize(self, tokens):
        self.agenda = []
        self.chart = Chart(
            [
                [[] for _ in range(len(tokens) + 1)]
                for _ in range(len(tokens) + 1)
            ]
        )
        for i, t in enumerate(tokens):
            word_edge = self.Edge(t, None, [], [], [], dict())
            self.chart[i][i + 1].append(word_edge)
            self.agenda.append((word_edge, i, i + 1))
            for m in self.grammar.lexicon.get_meanings(t):
                edge = self.Edge(
                    m, None, [word_edge.head], [], [word_edge], dict()
                )
                self.chart[i][i + 1].append(edge)
                self.agenda.append((edge, i, i + 1))

    # If the chart contains the complete edge
    #   [A → α • , (i, j)]
    # and the grammar contains the production
    #   B → A β
    # then add the self-loop edge
    #   [B →  • A β , (i, i)]
    #
    def _predict(self, edge, i, j):
        for rule in self.grammar.rules:
            u = _lu.unify(rule.constituents[0], edge.head)
            if u is not None:
                self.chart[i][i].append(
                    self.Edge(None, rule, [], rule.constituents[:], [], dict())
                )

    # If the chart contains the edges
    #   [A → α • B β , (i, j)]
    #   [B → γ • , (j, k)]
    # then add the new edge
    #   [A → α B • β , (i, k)]
    # where α, β, and γ are (possibly empty) sequences
    # of terminals or non-terminals
    #
    def _complete(self, completed_edge, j, k):
        for i, e in self._uncompleted_edges_ending_at(j):
            u = _lu.unify(e.remaining[0], completed_edge.head)
            if not u:
                continue
            new_edge = self._complete_edge(e, completed_edge, u)
            self._add_completed_edge_to_chart(new_edge, i, k)

    def _uncompleted_edges_ending_at(self, j):
        for i in range(j + 1):
            for e in self.chart[i][j]:
                if e.remaining:
                    yield i, e

    def _complete_edge(self, edge_a, edge_b, u):
        # here completed could have the references
        #   to the involved edges
        n_completed = edge_a.completed + [u[1]]
        n_used_edges = edge_a.used_edges + [edge_b]
        n_remaining = edge_a.remaining[1:]
        unif = edge_a.unification.copy()
        unif.update(u[0])
        if n_remaining:
            n_remaining = [_lu.substitute(r, unif) for r in n_remaining]
            return self.Edge(
                edge_a.head,
                edge_a.rule,
                n_completed,
                n_remaining,
                n_used_edges,
                unif,
            )
        else:
            new_head = edge_a.rule.constructor(*n_completed)
            if not new_head:
                raise Exception("No new head in rule:", edge_a.rule)
            new_head = _lu.substitute(new_head, unif)
            return self.Edge(
                new_head, edge_a.rule, n_completed, [], n_used_edges, unif
            )

    def _add_completed_edge_to_chart(self, new_edge, i, k):
        if not new_edge:
            return
        if not new_edge.remaining:
            self.agenda.append((new_edge, i, k))
        self.chart[i][k].append(new_edge)


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
