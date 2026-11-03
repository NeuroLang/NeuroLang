"""
The graph of a query: where its answer comes from.

:func:`query_graph` reads a program, without running it, and returns the
predicates its query depends on, down to the data of the engine: its
relations, its random choices and probabilistic facts, the rules of the
program and of the engine, the probabilities ``PROB[...]`` asks for, and
the answer, each a :class:`Node`, and which uses which, an :class:`Edge`.
A rule is probabilistic when it depends on a random choice or on
probabilistic facts not through a probability. :func:`graph_page` draws
the graph in an HTML page, from the data on the left to the answer on the
right, with what each kind of node and edge means. ``.graph`` shows the
graph of the last query; the REPL in the browser, that of each query.
"""

import html
import json
from typing import NamedTuple, Optional

from nl_core.datalog.expression_processing import extract_logic_predicates
from nl_core.datalog.pretty_printer import DatalogPrettyPrinter
from nl_core.probabilistic.expressions import (
    PROB, Condition, ProbabilisticPredicate
)
from nl_expr.expressions import Constant, FunctionApplication, Symbol
from nl_core.datalog import Union
from nl_core.datalog.expressions import AdornedSymbol
from nl_expr.logic import Conjunction, Implication, Negation
from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.tui.commands import command
from neurolang_apps.utils.tui.provenance import (
    _datalog_queries, _squall_queries
)


class Node(NamedTuple):
    """
    A predicate of the graph, of a *kind* of :data:`KINDS`, shown as
    *label*, whose *detail* is its description, its size and its rules.
    """

    id: str
    label: str
    kind: str
    detail: str = ""


class Edge(NamedTuple):
    """
    The use of the predicate *source* by *target*, in *atom*: of *kind*
    ``uses``, ``negated`` (``~``) or ``given`` (after ``//``).
    """

    source: str
    target: str
    kind: str
    atom: str = ""


class QueryGraph(NamedTuple):
    nodes: tuple
    edges: tuple


# The kinds of nodes, in the order of the legend, with what they are.
KINDS = {
    "data": ("Data", "a relation of the engine's data, as stored"),
    "choice": (
        "Random choice", "one of its rows is chosen at random, each as "
        "likely as given (e.g. one study of all)",
    ),
    "facts": (
        "Probabilistic facts", "each of its rows is true with its "
        "probability, independently",
    ),
    "rule": ("Rule", "its rows follow from those of its body, for sure"),
    "probabilistic": (
        "Probabilistic rule", "uses a random choice or probabilistic "
        "facts: each of its rows holds with some probability",
    ),
    "probability": (
        "Probability", "PROB[…]: the probability of each row of a "
        "probabilistic body, given the condition after //",
    ),
    "answer": ("Answer", "the rows the query returns"),
    "undefined": ("Undefined", "no relation or rule of this name"),
}

# The kinds of edges, with what they are.
EDGE_KINDS = {
    "uses": "its rows are used by the rule it points to",
    "given": "it is in the condition after // of a probability",
    "negated": "used negated (~): the rows it does not have",
}

_PRINTER = DatalogPrettyPrinter()


def _text(expression) -> str:
    """Return *expression* as Datalog, on one line."""
    return " ".join(_PRINTER.walk(expression).split())


def _head(rule):
    """Return the atom of the head of *rule*, ``p(x)`` of ``p(x) :: 0.5``
    too."""
    head = rule.consequent
    return head.body if isinstance(head, ProbabilisticPredicate) else head


def _is_probability(rule) -> bool:
    """Whether the head of *rule* asks for a probability, ``PROB``."""
    return any(
        isinstance(arg, FunctionApplication) and arg.functor == PROB
        for arg in _head(rule).args
    )


def _atoms(body) -> list:
    """Return the (atom, edge kind) pairs of the rule body *body*."""
    parts = (
        [(body.conditioned, "uses"), (body.conditioning, "given")]
        if isinstance(body, Condition) else [(body, "uses")]
    )
    atoms = []
    for part, kind in parts:
        for predicate in extract_logic_predicates(part):
            if isinstance(predicate, Negation):
                atoms.append((predicate.formula, "negated"))
            else:
                atoms.append((predicate, kind))
    return [
        (atom, kind) for atom, kind in atoms
        if isinstance(atom.functor, Symbol)
    ]


def _is_fresh(name: str) -> bool:
    """Whether *name* is that of a symbol the engine made, of no meaning."""
    return name.startswith("fresh_")


def _conjuncts(formula) -> tuple:
    """Return the conjuncts of *formula*."""
    if isinstance(formula, Conjunction):
        return formula.formulas
    return (formula,)


def _probability_label(rule) -> str:
    """Return the label of the probability a query asks in its body."""
    given = " // …" if isinstance(rule.antecedent, Condition) else ""
    return f"PROB[…{given}]"


def _probability_text(rule) -> str:
    """Return ``PROB[body // condition]`` of a rule asking a probability."""
    body = rule.antecedent
    if isinstance(body, Condition):
        return (
            f"PROB[{_text(body.conditioned)} // {_text(body.conditioning)}]"
        )
    return f"PROB[{_text(body)}]"


class _Builder:
    """Gathers the nodes and edges of the predicates a query depends on."""

    def __init__(self, nl, rules, descriptions):
        self.ir = nl.program_ir
        self.table = nl.program_ir.symbol_table
        self.descriptions = descriptions
        self.program_rules = {}
        for rule in rules:
            name = _head(rule).functor.name
            self.program_rules.setdefault(name, []).append(rule)
        self.nodes = {}
        self.edges = {}

    def rules(self, name: str) -> Optional[list]:
        """Return the rules of *name*, those of the program first."""
        if name in self.program_rules:
            return self.program_rules[name]
        value = self.table.get(Symbol(name))
        if isinstance(value, Union):
            return [self._conditional(rule) for rule in value.formulas]
        return value

    def _conditional(self, rule):
        """
        Return the rule ``P(…, PROB) :- A // C`` the engine compiled into
        *rule*, ``P(…, p) :- P^cond_num(…, p1), P^cond_den(…, p2),
        p == p1 / p2``, where ``P^cond_num`` is the probability of ``A, C``
        and ``P^cond_den`` that of ``C``; or *rule* if it is not such.
        """
        parts = {}
        for formula in _conjuncts(rule.antecedent):
            functor = getattr(formula, "functor", None)
            if isinstance(functor, AdornedSymbol) and functor.adornment in (
                "cond_num", "cond_den"
            ):
                value = self.table.get(functor)
                if isinstance(value, Union) and len(value.formulas) == 1:
                    parts[functor.adornment] = value.formulas[0]
        if len(parts) != 2:
            return rule
        numerator, denominator = parts["cond_num"], parts["cond_den"]
        given = _conjuncts(denominator.antecedent)
        both = _conjuncts(numerator.antecedent)
        if both[len(both) - len(given):] != given:
            return rule  # not as the sugar compiles it: shown as it is
        head = numerator.consequent
        return Implication(
            head.functor.expression(*head.args),
            Condition(
                Conjunction(both[:len(both) - len(given)]),
                Conjunction(given),
            ),
        )

    def _stored_copy(self, rules: list):
        """Return the relation *rules* copy, when they are one rule whose
        body is a stored relation of no name, such as a precomputed one."""
        atoms = _conjuncts(rules[0].antecedent) if len(rules) == 1 else ()
        functor = getattr(atoms[0], "functor", None) if len(atoms) == 1 \
            else None
        if not (isinstance(functor, Symbol) and _is_fresh(functor.name)):
            return None
        value = self.table.get(functor)
        return value if isinstance(value, Constant) else None

    def add(self, name: str) -> Optional[str]:
        """Add the predicate *name* and what it uses; return its id."""
        if name in self.nodes:
            return name
        found = self.rules(name)
        if isinstance(found, list):
            stored = self._stored_copy(found)
            if stored is None:
                self.nodes[name] = None  # being added: a recursive rule
                self._add_rules(name, found)
                return name
            found = stored
        if isinstance(found, Constant) and callable(found.value):
            return None  # a function, such as a builtin
        self.nodes[name] = self._relation(name, found)
        return name

    def _relation(self, name: str, value) -> Node:
        symbol = Symbol(name)
        if value is None:
            return Node(name, name, "undefined", "not defined")
        if symbol in getattr(self.ir, "pchoice_pred_symbs", ()):
            kind = "choice"
        elif symbol in getattr(self.ir, "pfact_pred_symbs", ()):
            kind = "facts"
        else:
            kind = "data"
        lines = [self.descriptions.get(name, "")]
        try:
            lines.append(f"{len(value.value):,} rows")
        except TypeError:
            pass
        return Node(name, name, kind, "\n".join(filter(None, lines)))

    def _add_rules(self, name: str, rules: list) -> None:
        probability = any(_is_probability(rule) for rule in rules)
        fresh = _is_fresh(name)
        for rule in rules:
            for atom, kind in _atoms(rule.antecedent):
                source = self.add(atom.functor.name)
                if source is not None:
                    self.edges.setdefault(
                        (source, name, kind), _text(atom)
                    )
        if fresh and probability:
            label = _probability_label(rules[0])
            detail = "\n".join(_probability_text(rule) for rule in rules)
        else:
            label = name
            detail = "\n\n".join(
                [self.descriptions.get(name, "")]
                + [_PRINTER.walk(rule) for rule in rules]
            ).strip()
        if probability:
            kind = "probability"
        elif any(
            isinstance(rule.consequent, ProbabilisticPredicate)
            for rule in rules
        ):
            kind = "probabilistic"  # p(x) :: 0.5 :- ..., p(x) :~: ...
        else:
            kind = "rule"
        self.nodes[name] = Node(name, label, kind, detail)

    def graph(self) -> QueryGraph:
        nodes = _probabilistic(self.nodes, self.edges)
        edges = tuple(
            Edge(source, target, kind, atom)
            for (source, target, kind), atom in self.edges.items()
        )
        return QueryGraph(tuple(nodes.values()), edges)


def _probabilistic(nodes: dict, edges: dict) -> dict:
    """Return *nodes*, the rules using random choices made probabilistic."""
    uncertain = {n for n, node in nodes.items() if node.kind in (
        "choice", "facts", "probabilistic"
    )}
    changed = True
    while changed:
        changed = False
        for source, target, _ in edges:
            if source in uncertain and target not in uncertain and (
                nodes[target].kind == "rule"
            ):
                uncertain.add(target)
                changed = True
    return {
        name: node._replace(kind="probabilistic")
        if node.kind == "rule" and name in uncertain else node
        for name, node in nodes.items()
    }


def query_graph(
    nl, program: str, squall: bool = False, descriptions=None
) -> QueryGraph:
    """
    Return the graph of the queries of *program*, in SQUALL if *squall*,
    on the engine *nl*, whose predicates are described in *descriptions*.
    """
    if squall:
        rules, queries = _squall_queries(program)
    else:
        rules, queries = _datalog_queries(nl, program)
    builder = _Builder(nl, rules, descriptions or {})
    for key, (head, body) in queries.items():
        name = head.functor.name if isinstance(
            head, FunctionApplication
        ) else key
        for atom, kind in _atoms(body):
            source = builder.add(atom.functor.name)
            if source is not None:
                builder.edges.setdefault((source, name, kind), _text(atom))
        builder.nodes[name] = Node(name, name, "answer", program.strip())
    return builder.graph()


def engine_descriptions(engine_name: str) -> dict:
    """Return the descriptions of the predicates of the engine, by name."""
    try:
        predicates = engine_registry.get_predicates(engine_name)
    except Exception:  # an engine of no configuration
        return {}
    return {
        name: str(info.get("description") or "").strip()
        for name, info in predicates.items()
    }


# The layout of the page, in pixels.
_CHAR = 7.8  # the width of a character of the labels
_HEIGHT = 46
_GAP_X = 90
_GAP_Y = 22
_LANE = 16  # the height an edge takes crossing a column
_MARGIN = 20
_LOOP = 36  # the room above the nodes for the loop of a recursive rule


class _Layout(NamedTuple):
    """
    Where the nodes of a graph go: *columns*, the ids of the nodes in
    each column, from the data to the answer, and *routes*, the ids of
    the nodes each edge goes through, in order. An edge from a column to
    one further than the next goes through a waypoint ``(edge index,
    column)`` in each column between them, which is placed as the nodes,
    so that the edge goes around the nodes rather than over them.
    """

    columns: list
    routes: list


def _depths(graph: QueryGraph) -> dict:
    """Return the column of each node: 0 for those using no other, else
    one more than the column of the furthest node it uses."""
    inputs = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.source != edge.target:
            inputs[edge.target].append(edge.source)
    depth = {}

    def column(node, seen):
        if node not in depth:
            if node in seen:
                return 0  # a recursion: placed by its other inputs
            seen.add(node)
            depth[node] = max(
                (column(n, seen) + 1 for n in inputs[node]), default=0
            )
        return depth[node]

    for node in graph.nodes:
        column(node.id, set())
    return depth


def _layout(graph: QueryGraph) -> _Layout:
    """Return the columns of the nodes of *graph* and the routes of its
    edges, ordered so that edges cross less."""
    depth = _depths(graph)
    columns = [[] for _ in range(max(depth.values(), default=0) + 1)]
    for node in graph.nodes:
        columns[depth[node.id]].append(node.id)
    routes = []
    for n, edge in enumerate(graph.edges):
        start, end = depth[edge.source], depth[edge.target]
        waypoints = [(n, c) for c in range(start + 1, end)]
        for waypoint in waypoints:
            columns[waypoint[1]].append(waypoint)
        routes.append([edge.source, *waypoints, edge.target])
    _order(columns, [
        pair for route, edge in zip(routes, graph.edges)
        if depth[edge.source] < depth[edge.target]
        for pair in zip(route, route[1:])
    ])
    return _Layout(columns, routes)


def _order(columns: list, links: list) -> None:
    """Order each column by the place of what it is linked to in the
    column before (after, sweeping back), so that edges cross less (the
    barycentre heuristic). *links* go from a column to the next."""
    for sweep in (1, -1, 1):
        place = {n: i for column in columns for i, n in enumerate(column)}
        for column in columns[::sweep][1:]:
            def centre(node):
                near = [
                    place[a if sweep == 1 else b] for a, b in links
                    if (b if sweep == 1 else a) == node
                ]
                return sum(near) / len(near) if near else place[node]

            column.sort(key=centre)
            place.update({n: i for i, n in enumerate(column)})


def _subtitle(node: Node) -> str:
    rows = next(
        (line for line in node.detail.splitlines() if line.endswith(" rows")),
        "",
    )
    name = KINDS[node.kind][0].lower()
    return f"{name} · {rows}" if rows else name


def _width(node: Node) -> float:
    return max(len(node.label), len(_subtitle(node)) * .8) * _CHAR + 28


def _positions(graph: QueryGraph, layout: _Layout) -> tuple:
    """
    Return the box (x, y, width) of each node and waypoint, and the size
    of the drawing. A waypoint's box is the lane across its column.
    """
    nodes = {node.id: node for node in graph.nodes}
    top = _MARGIN + (_LOOP if any(
        edge.source == edge.target for edge in graph.edges
    ) else 0)

    def size(item):
        return _HEIGHT + _GAP_Y if item in nodes else _LANE

    tallest = max(sum(map(size, column)) for column in layout.columns)
    boxes, x = {}, _MARGIN
    for column in layout.columns:
        width = max(_width(nodes[n]) for n in column if n in nodes)
        y = top + (tallest - sum(map(size, column))) / 2
        for item in column:
            if item in nodes:
                boxes[item] = (x, y, width)
            else:  # the lane of an edge, through the middle of its slot
                boxes[item] = (x, y + _LANE / 2 - _HEIGHT / 2, width)
            y += size(item)
        x += width + _GAP_X
    return boxes, x - _GAP_X + _MARGIN, top + tallest - _GAP_Y + _MARGIN + 30


def _shape(kind: str, x: float, y: float, w: float) -> str:
    """Return the SVG of the outline of a node of *kind*."""
    h = _HEIGHT
    if kind == "data":  # a cylinder, as a database is drawn
        e = 6
        return (
            f'<path class="shape" d="M{x},{y + e} a{w / 2},{e} 0 0 0 {w},0 '
            f'v{h - 2 * e} a{w / 2},{e} 0 0 1 {-w},0 z"/>'
            f'<path class="rim" d="M{x},{y + e} a{w / 2},{e} 0 0 1 {w},0 '
            f'a{w / 2},{e} 0 0 1 {-w},0"/>'
        )
    if kind == "probability":  # a hexagon
        s = 12
        points = (
            f"{x + s},{y} {x + w - s},{y} {x + w},{y + h / 2} "
            f"{x + w - s},{y + h} {x + s},{y + h} {x},{y + h / 2}"
        )
        return f'<polygon class="shape" points="{points}"/>'
    radius = {"answer": h / 2, "choice": 4, "facts": 4}.get(kind, 8)
    return (
        f'<rect class="shape" x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="{radius}"/>'
    )


def _route_path(route: list, boxes: dict, shift: float) -> tuple:
    """
    Return the SVG path of an edge going through the boxes of *route*,
    from the right of the first to the left of the last, *shift* below
    their middles, and the point where its label goes, at the start of
    its last curve, apart from those of other edges: a curve from a
    column to the next, a straight line across the lane of a waypoint.
    """
    last = len(route) - 1

    def middle(n):
        _, y, _ = boxes[route[n]]
        return y + _HEIGHT / 2 + (shift if n in (0, last) else 0)

    x, _, w = boxes[route[0]]
    px, py = x + w, middle(0)
    tx = boxes[route[last]][0]
    if tx <= px:  # back to a column on the left: around below
        ty = middle(last)
        low = max(py, ty) + _HEIGHT
        d = f"M{px},{py} C{px + 60},{low} {tx - 60},{low} {tx},{ty}"
        return d, ((px + tx) / 2, low - 10)
    d = f"M{px},{py}"
    for n in range(1, last + 1):
        x, _, w = boxes[route[n]]
        y = middle(n)
        c = (x - px) / 2
        d += f" C{px + c},{py} {x - c},{y} {x},{y}"
        label = (px + (x - px) / 4, py + (y - py) / 4)
        if n < last:
            d += f" H{x + w}"
            px, py = x + w, y
    return d, label


def _edge_paths(graph: QueryGraph, layout: _Layout, boxes: dict) -> list:
    """Return the SVG of the edges; those between the same nodes side by
    side, a recursive rule's in a loop above it."""
    pairs = {}
    for n, edge in enumerate(graph.edges):
        pairs.setdefault((edge.source, edge.target), []).append(n)
    svg = []
    for (source, target), indices in pairs.items():
        for i, n in enumerate(indices):
            edge = graph.edges[n]
            if source == target:
                sx, sy, sw = boxes[source]
                d = (
                    f"M{sx + sw * .65},{sy} C{sx + sw * .65},{sy - 34} "
                    f"{sx + sw * .35},{sy - 34} {sx + sw * .35},{sy}"
                )
                mx, my = sx + sw / 2, sy - 28
            else:
                shift = (i - (len(indices) - 1) / 2) * 10
                d, (mx, my) = _route_path(layout.routes[n], boxes, shift)
            title = html.escape(f"{edge.atom} ({EDGE_KINDS[edge.kind]})")
            word = {"given": "given", "negated": "not"}.get(edge.kind)
            label = (
                f'<text class="edge-label" x="{mx}" y="{my - 3}">{word}'
                "</text>" if word else ""
            )
            svg.append(
                f'<g class="edge {edge.kind}" data-source="'
                f'{html.escape(source)}" data-target="{html.escape(target)}">'
                f'<title>{title}</title><path d="{d}" '
                f'marker-end="url(#arrow-{edge.kind})"/>{label}</g>'
            )
    return svg


def _node_svg(node: Node, box: tuple) -> str:
    x, y, w = box
    text = y + (23 if node.kind == "data" else 20)  # below a cylinder's rim
    title = html.escape(node.detail or node.label)
    return (
        f'<g class="node {node.kind}" data-id="{html.escape(node.id)}" '
        f'tabindex="0"><title>{title}</title>{_shape(node.kind, x, y, w)}'
        f'<text class="label" x="{x + w / 2}" y="{text}">'
        f"{html.escape(node.label)}</text>"
        f'<text class="kind" x="{x + w / 2}" y="{text + 16}">'
        f"{html.escape(_subtitle(node))}</text></g>"
    )


def _legend(graph: QueryGraph) -> str:
    """Return the HTML of the legend of the kinds the graph has."""
    shown = {node.kind for node in graph.nodes}
    items = [
        f'<li><span class="swatch {kind}"></span><b>{name}</b>: '
        f"{html.escape(meaning)}</li>"
        for kind, (name, meaning) in KINDS.items() if kind in shown
    ]
    used = {edge.kind for edge in graph.edges}
    items += [
        f'<li><span class="line {kind}"></span><b>{kind}</b>: '
        f"{html.escape(meaning)}</li>"
        for kind, meaning in EDGE_KINDS.items() if kind in used
    ]
    return "<ul>" + "".join(items) + "</ul>"


_STYLE = """
:root { --bg: #fff; --text: #1f2328; --muted: #59636e; --edge: #8c959f;
  --data: #dbeafe; --data-b: #2563eb; --rand: #ffedd5; --rand-b: #ea580c;
  --rule: #e0f2fe; --rule-b: #0369a1; --prob: #ede9fe; --prob-b: #7c3aed;
  --ans: #dcfce7; --ans-b: #15803d; --neg: #dc2626; }
@media (prefers-color-scheme: dark) { :root { --bg: #0d1117;
  --text: #e6edf3; --muted: #9198a1; --edge: #6e7681; --data: #172554;
  --rand: #431407; --rule: #082f49; --prob: #2e1065; --ans: #052e16;
  --neg: #f87171; } }
body { margin: 0; padding: .75rem 1rem; background: var(--bg);
  color: var(--text); font: 14px/1.4 system-ui, sans-serif; }
h1 { font-size: 1rem; margin: 0 0 .25rem; }
p { margin: 0 0 .5rem; color: var(--muted); }
.graph { overflow: auto; }
svg { display: block; }
.shape, .rim { stroke-width: 1.5; }
.data .shape, .data .rim { fill: var(--data); stroke: var(--data-b); }
.choice .shape, .facts .shape { fill: var(--rand); stroke: var(--rand-b); }
.facts .shape { stroke-dasharray: 5 3; }
.rule .shape { fill: var(--rule); stroke: var(--rule-b); }
.probabilistic .shape { fill: var(--rand); stroke: var(--rand-b);
  stroke-dasharray: 5 3; }
.probability .shape { fill: var(--prob); stroke: var(--prob-b); }
.answer .shape { fill: var(--ans); stroke: var(--ans-b); stroke-width: 2.5; }
.undefined .shape { fill: none; stroke: var(--neg); stroke-dasharray: 2 2; }
.node { cursor: pointer; outline: none; }
.node:focus-visible .shape { stroke-width: 3; }
.node text { text-anchor: middle; fill: var(--text); }
.label { font: 600 13px ui-monospace, Menlo, monospace; }
.kind { font-size: 10.5px; fill: var(--muted) !important; }
.edge path { fill: none; stroke: var(--edge); stroke-width: 1.5; }
.edge.given path { stroke-dasharray: 6 4; stroke: var(--prob-b); }
.edge.negated path { stroke: var(--neg); }
.edge-label { font-size: 11px; fill: var(--muted); text-anchor: middle; }
.faded { opacity: .15; }
.selected .shape { stroke-width: 3; }
ul { list-style: none; padding: 0; margin: .5rem 0; display: grid;
  gap: .2rem; grid-template-columns: repeat(auto-fill, minmax(22rem, 1fr)); }
.swatch, .line { display: inline-block; width: 1.4rem; height: .9rem;
  margin-right: .4rem; vertical-align: middle; border: 1.5px solid; }
.swatch.data { background: var(--data); border-color: var(--data-b);
  border-radius: 40%/25%; }
.swatch.choice, .swatch.facts, .swatch.probabilistic {
  background: var(--rand); border-color: var(--rand-b); }
.swatch.facts, .swatch.probabilistic { border-style: dashed; }
.swatch.rule { background: var(--rule); border-color: var(--rule-b); }
.swatch.probability { background: var(--prob); border-color: var(--prob-b);
  clip-path: polygon(20% 0, 80% 0, 100% 50%, 80% 100%, 20% 100%, 0 50%); }
.swatch.answer { background: var(--ans); border-color: var(--ans-b);
  border-radius: 1rem; }
.swatch.undefined { border-color: var(--neg); border-style: dotted; }
.line { height: 0; border-width: 1.5px 0 0; border-color: var(--edge); }
.line.given { border-style: dashed; border-color: var(--prob-b); }
.line.negated { border-color: var(--neg); }
#detail { white-space: pre-wrap; font: 12px/1.5 ui-monospace, Menlo,
  monospace; margin: .5rem 0 0; padding: .5rem; border-radius: 6px;
  border: 1px solid var(--edge); }
#detail:empty { display: none; }
"""

# Clicking a node shows its details and what it depends on, fading the
# rest; clicking elsewhere shows everything again.
_SCRIPT = """
const nodes = JSON.parse(document.getElementById("nodes").textContent);
const svg = document.querySelector("svg");
const detail = document.getElementById("detail");
function upstream(id) {
  const seen = new Set([id]);
  for (let grew = true; grew;) {
    grew = false;
    for (const e of svg.querySelectorAll(".edge"))
      if (seen.has(e.dataset.target) && !seen.has(e.dataset.source)) {
        seen.add(e.dataset.source);
        grew = true;
      }
  }
  return seen;
}
function select(id) {
  const kept = id === null ? null : upstream(id);
  for (const n of svg.querySelectorAll(".node")) {
    n.classList.toggle("faded", kept !== null && !kept.has(n.dataset.id));
    n.classList.toggle("selected", n.dataset.id === id);
  }
  for (const e of svg.querySelectorAll(".edge"))
    e.classList.toggle("faded", kept !== null
      && !(kept.has(e.dataset.source) && kept.has(e.dataset.target)));
  const node = nodes[id];
  detail.textContent = node
    ? `${node.label} (${node.kind})\\n\\n${node.detail}` : "";
}
svg.addEventListener("click", (event) => {
  const node = event.target.closest(".node");
  select(node ? node.dataset.id : null);
});
svg.addEventListener("keydown", (event) => {
  const node = event.target.closest(".node");
  if (node && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    select(node.dataset.id);
  }
});
"""


def graph_page(graph: QueryGraph, title: str) -> str:
    """Return the HTML page drawing *graph*, entitled *title*."""
    layout = _layout(graph)
    boxes, width, height = _positions(graph, layout)
    markers = "".join(
        f'<marker id="arrow-{kind}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" class="arrow-{kind}"/></marker>'
        for kind in EDGE_KINDS
    )
    details = {
        node.id: {
            "label": node.label, "kind": KINDS[node.kind][0].lower(),
            "detail": node.detail,
        } for node in graph.nodes
    }
    data = json.dumps(details).replace("<", "\\u003c")
    body = "".join(
        _edge_paths(graph, layout, boxes)
        + [_node_svg(node, boxes[node.id]) for node in graph.nodes]
    )
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title><style>{_STYLE}"
        ".arrow-uses { fill: var(--edge); } .arrow-given { fill: "
        "var(--prob-b); } .arrow-negated { fill: var(--neg); }</style>"
        f"</head><body><h1>{html.escape(title)}</h1>"
        "<p>Read from left to right: the data on the left go through the "
        "rules to the answer on the right. Click a node to see its rules "
        "and what it depends on.</p>"
        f'<div class="graph"><svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{html.escape(title)}"><defs>{markers}</defs>{body}'
        "</svg></div>"
        f'<pre id="detail"></pre>{_legend(graph)}'
        f'<script type="application/json" id="nodes">{data}</script>'
        f"<script>{_SCRIPT}</script></body></html>"
    )


def last_query_page(app) -> str:
    """Return the page of the graph of the last query of *app*."""
    program, squall = app.last_query
    graph = query_graph(
        app._nl, program, squall,
        engine_descriptions(app.settings.engine_name),
    )
    return graph_page(graph, "The last query")


@command(
    ".graph", "Show where the answer of the last query comes from: its "
    "data, rules and probabilities, in a graph in the browser",
)
def cmd_graph(app, arg: str) -> None:
    """Show the graph of the last query."""
    from neurolang_apps.utils.tui.session import _no_last_query

    if _no_last_query(app):
        return
    try:
        page = last_query_page(app)
    except Exception as exc:
        app._report_error(exc, app.last_query[0])
        return
    app.show_graph(page)
