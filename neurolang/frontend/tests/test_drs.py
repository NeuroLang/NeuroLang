from ...logic.unification import most_general_unifier, apply_substitution
from ...expressions import Symbol, Constant
from collections import namedtuple


def rule(*args, root=False):
    def wrapper(foo):
        foo._rule_pattern = args
        foo._rule_is_root = root
        return foo

    return wrapper


class Rule:
    def __init__(self, name, head, constituents, is_root):
        self.name = name
        self.head = head
        self.is_root = is_root
        self.constituents = constituents

    def __repr__(self):
        return (
            f"<{self.name} := "
            + ", ".join([repr(c) for c in self.constituents])
            + ">"
        )


class Grammar:
    @property
    def rules(self):
        for k in dir(self):
            v = getattr(self, k)
            if hasattr(v, "_rule_pattern"):
                yield Rule(k, v, v._rule_pattern, v._rule_is_root)


Edge = namedtuple("Edge", "head rule completed remaining")


def recognize(grammar, string):
    tokens = [Constant[str](t) for t in string.split()]
    # Create an empty chart spanning the sentence.
    chart = [
        [[] for _ in range(len(tokens) + 1)] for _ in range(len(tokens) + 1)
    ]
    agenda = []
    # Apply the Bottom-Up Initialization Rule to each word.
    _initialize(grammar, tokens, chart, agenda)
    # Until no more edges are added:
    while agenda:
        agenda.sort(key=lambda e: (e[1], e[2]))

        edge, i, j = agenda.pop(0)
        # Apply the Bottom-Up Predict Rule everywhere it applies.
        _predict(grammar, edge, i, j, chart, agenda)
        # Apply the Fundamental Rule everywhere it applies.
        _complete(grammar, edge, i, j, chart, agenda)

    # Return all of the parse trees in the chart.
    return any(e.rule.is_root for e in chart[0][len(tokens)])


# For every word wi add the edge
#   [wi →  • , (i, i+1)]
#
def _initialize(grammar, tokens, chart, agenda):
    for i, t in enumerate(tokens):
        edge = Edge(t, None, [], [])
        chart[i][i + 1].append(edge)
        agenda.append((edge, i, i+1))


def unify(a, b):
    w = Symbol("wrapper")
    u = most_general_unifier(w(a), w(b))
    if u is None:
        return
    return u[0], u[1].args[0]


def substitute(exp, sub):
    w = Symbol("wrapper")
    n = apply_substitution(w(exp), sub)
    return n.args[0]


# If the chart contains the complete edge
#   [A → α • , (i, j)]
# and the grammar contains the production
#   B → A β
# then add the self-loop edge
#   [B →  • A β , (i, i)]
#
def _predict(grammar, edge, i, j, chart, agenda):
    for rule in grammar.rules:
        u = unify(rule.constituents[0], edge.head)
        if u is not None:
            chart[i][i].append(Edge(None, rule, [], rule.constituents[:]))


# If the chart contains the edges
#   [A → α • B β , (i, j)]
#   [B → γ • , (j, k)]
# then add the new edge
#   [A → α B • β , (i, k)]
# where α, β, and γ are (possibly empty) sequences
# of terminals or non-terminals
#
def _complete(grammar, completed_edge, j, k, chart, agenda):
    for i in range(j+1):
        for head, rule, completed, remaining in chart[i][j]:
            if not remaining:
                continue
            u = unify(remaining[0], completed_edge.head)
            if not u:
                continue
            # here completed could have the references to the involved edges
            completed = completed + [u[1]]
            remaining = remaining[1:]
            if remaining:
                remaining = [substitute(r, u[0]) for r in remaining]
                new_edge = Edge(head, rule, completed, remaining)
            else:
                new_head = rule.head(*completed)
                if not new_head:
                    continue
                new_edge = Edge(new_head, rule, completed, [])
                agenda.append((new_edge, i, k))
            chart[i][k].append(new_edge)


#
#
#
#
#
#
#
#
#
#
#
#


S = Symbol("S")
NP = Symbol("NP")
PN = Symbol("PN")
VP = Symbol("VP")
V = Symbol("V")

a = Symbol("a")
b = Symbol("b")

plural = Constant("plural")
singular = Constant("singular")


class DRSGrammar(Grammar):
    @rule(NP(a), VP(a), root=True)
    def s(self, np, vp):
        return S(np.args[0])  # this could look better if unified too

    @rule(V(a), NP(b))
    def vp(self, v, np):
        return VP(v.args[0])

    @rule(NP(a), Constant("and"), NP(b))
    def np_and(self, first, _, second):
        return NP(plural)

    @rule(PN(a))
    def np_proper(self, pn):
        return NP(pn.args[0])

    @rule(a)
    def verb_singular(self, token):
        if token.value in ["owns"]:
            return V(singular)

    @rule(a)
    def verb_plural(self, token):
        if token.value in ["own"]:
            return V(plural)

    @rule(a)
    def proper_name(self, token):
        if token.value in ["Jones", "Smith", "Ulysses"]:
            return PN(singular)


#
#
#
#
#
#
#
#
#
#
#
#

def test_mgu():
    u = unify(a, Constant("hehe"))
    assert u is not None
    u = unify(NP(a), NP(Constant("hoho")))
    assert u is not None


def test_parser():
    g = DRSGrammar()
    assert recognize(g, "Jones owns Ulysses")
    assert not recognize(g, "Jones own Ulysses")
    assert recognize(g, "Jones and Smith own Ulysses")


