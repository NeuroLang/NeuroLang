"""
The graph of a query: where its answer comes from, and how.

:func:`query_graph` reads a program, without running it, and returns the
predicates its query depends on, down to the data of the engine: its
relations, its random choices and probabilistic facts, the rules of the
program and of the engine, the probabilities ``PROB[...]`` asks for, and
the answer, each a :class:`Node`. A rule is shown with the atoms of its
body, a :class:`Row` each, which the relations they use point to, an
:class:`Edge` each, and with how it combines them: the variables it joins
them on, those it keeps and those it drops; a probability, with how it
turns the random choices it depends on into a number for each row. A rule
is probabilistic when it depends on a random choice or on probabilistic
facts not through a probability. :func:`graph_page` draws the graph in an
HTML page, from the data on the left to the answer on the right, with the
distribution of the probabilities of the answer when its rows are known.
``.graph`` shows the graph of the last query; the REPL in the browser,
that of each query.
"""

import html
import json
import operator
import textwrap
from typing import NamedTuple, Optional

from nl_core.datalog import Union
from nl_core.datalog.expression_processing import extract_logic_predicates
from nl_core.datalog.expressions import AdornedSymbol
from nl_core.datalog.pretty_printer import DatalogPrettyPrinter
from nl_core.probabilistic.expressions import (
    PROB, Condition, ProbabilisticPredicate
)
from nl_expr.expressions import Constant, FunctionApplication, Symbol
from nl_expr.logic import Conjunction, Implication, Negation
from nl_expr.logic.expression_processing import extract_logic_free_variables
from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.tui.brain_viewer import frame_hooks
from neurolang_apps.utils.tui.commands import command
from neurolang_apps.utils.tui.provenance import (
    _datalog_queries, _squall_queries
)


class Row(NamedTuple):
    """
    A line of the body of a rule: an ``atom``, ``negated`` (``~``), a
    ``filter`` such as ``x < y``, or a ``heading`` (``given``, ``or``).
    """

    text: str
    kind: str


class Node(NamedTuple):
    """
    A predicate of the graph, of a *kind* of :data:`KINDS`, shown as
    *label*, whose *detail* is its description, its size and its rules.

    A rule has the *rows* of its body and *notes* on how it combines
    them; a relation, the names of its *columns*. *probabilities* are the
    columns of the answer that are probabilities.
    """

    id: str
    label: str
    kind: str
    detail: str = ""
    rows: tuple = ()
    notes: tuple = ()
    columns: str = ""
    probabilities: tuple = ()


class Edge(NamedTuple):
    """
    The use of the predicate *source* by *target*, in *atom*, the row
    *row* of *target*: of *kind* ``uses``, ``negated`` or ``given``
    (after ``//``). Its *flow* is what goes through it: ``rows``,
    ``uncertain`` rows or ``probabilities``.
    """

    source: str
    target: str
    kind: str
    atom: str = ""
    row: int = -1
    flow: str = "rows"


class QueryGraph(NamedTuple):
    nodes: tuple
    edges: tuple


# The kinds of nodes, in the order of the legend, with what they are.
KINDS = {
    "data": ("Data", "a relation of the engine's data, as stored"),
    "ontology": (
        "Ontology", "a relation of an ontology (OWL/RDF): its triples and "
        "classes, whose subclasses are included",
    ),
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
        "Probability", "PROB[…]: the probability of the rows of its body, "
        "given the condition after //, as a number for each row",
    ),
    "answer": ("Answer", "the rows the query returns"),
    "undefined": ("Undefined", "no relation or rule of this name"),
}

# What goes through the edges, with what it is.
FLOWS = {
    "rows": "rows that hold for sure",
    "uncertain": (
        "uncertain rows: they hold for some random choices only"
    ),
    "probabilities": "probabilities: a number for each row",
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


def _is_prob(arg) -> bool:
    return isinstance(arg, FunctionApplication) and arg.functor == PROB


def _prob_position(rule) -> Optional[int]:
    """Return the place of ``PROB`` in the head of *rule*, if any."""
    return next(
        (i for i, arg in enumerate(_head(rule).args) if _is_prob(arg)), None
    )


def _is_fresh(name: str) -> bool:
    """Whether *name* is that of a symbol the engine made, of no meaning."""
    return name.startswith("fresh_")


def _conjuncts(formula) -> tuple:
    """Return the conjuncts of *formula*."""
    if isinstance(formula, Conjunction):
        return formula.formulas
    return (formula,)


def _names(variables) -> str:
    return ", ".join(variables)


def _variables(expression) -> list:
    """Return the names of the variables of *expression*, in order."""
    found = []
    for arg in getattr(expression, "args", ()):
        if isinstance(arg, Symbol):
            found.append(arg.name)
        else:
            found += [v for v in _variables(arg) if v not in found]
    return list(dict.fromkeys(found))


class _Builder:
    """Gathers the nodes and edges of the predicates a query depends on."""

    def __init__(self, nl, rules, descriptions, columns):
        self.ir = nl.program_ir
        self.table = nl.program_ir.symbol_table
        self.descriptions = descriptions
        self.ontology = getattr(nl, "ontology_predicates", ())
        self.columns = columns
        self.program_rules = {}
        for rule in rules:
            name = _head(rule).functor.name
            self.program_rules.setdefault(name, []).append(rule)
        self.nodes = {}
        self.edges = []
        self.relations = {}  # the stored relations, by name
        self.conditional = {}  # whether a probability is conditional
        self.positions = {}  # where PROB is in the head of a probability

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
        found = self.table.get(Symbol(name)) if name in self.ontology \
            else self.rules(name)
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
        self.relations[name] = value
        if symbol in getattr(self.ir, "pchoice_pred_symbs", ()):
            kind = "choice"
        elif symbol in getattr(self.ir, "pfact_pred_symbs", ()):
            kind = "facts"
        elif name in self.ontology:
            kind = "ontology"
        else:
            kind = "data"
        lines = [self.descriptions.get(name, "")]
        rows = _size(value)
        if rows is not None:
            lines.append(f"{rows:,} rows")
        return Node(
            name, name, kind, "\n".join(filter(None, lines)),
            columns=_names(self.columns.get(name, ())),
        )

    def label(self, name: str) -> str:
        node = self.nodes.get(name)
        return node.label if node is not None and _is_fresh(name) else name

    def body_rows(self, target: str, body, offset: int = 0) -> tuple:
        """
        Return the rows of *body*, the body of a rule of *target*, and
        its atoms using a relation, not negated; add the edges of its
        atoms to the rows, numbered from *offset*.
        """
        parts = [(body, "uses")]
        if isinstance(body, Condition):
            parts = [
                (body.conditioned, "uses"), (None, "given"),
                (body.conditioning, "given"),
            ]
        rows, positive = [], []
        for part, kind in parts:
            if part is None:
                rows.append(Row("given", "heading"))
                continue
            for conjunct in _conjuncts(part):
                rows.append(self._row(
                    target, conjunct, kind, offset + len(rows), positive
                ))
        return rows, positive

    def _row(self, target, conjunct, kind, row, positive) -> Row:
        """Return the row of *conjunct*, adding the edges of its atoms."""
        used = []
        for predicate in extract_logic_predicates(conjunct):
            negated = isinstance(predicate, Negation)
            atom = predicate.formula if negated else predicate
            if not isinstance(atom.functor, Symbol):
                continue  # a comparison
            source = self.add(atom.functor.name)
            if source is None:
                continue  # a function
            self.edges.append((
                source, target, "negated" if negated else kind,
                self._atom_text(atom), row,
            ))
            used.append(atom)
        if isinstance(conjunct, Negation):
            atom = conjunct.formula
            text = self._atom_text(atom) if used else _text(atom)
            return Row(f"not {text}", "negated")
        if not used:
            return Row(_text(conjunct), "filter")
        if used == [conjunct]:
            positive.append(conjunct)
            return Row(self._atom_text(conjunct), "atom")
        positive += used
        return Row(_text(conjunct), "atom")  # e.g. ∃t. (p(s, t))

    def _atom_text(self, atom) -> str:
        args = ", ".join(_text(arg) for arg in atom.args)
        return f"{self.label(atom.functor.name)}({args})"

    def _add_rules(self, name: str, rules: list) -> None:
        probability = _prob_position(rules[0]) is not None
        rows, notes = [], []
        for n, rule in enumerate(rules):
            if n:
                rows.append(Row("or", "heading"))
            body_rows, positive = self.body_rows(
                name, rule.antecedent, len(rows)
            )
            rows += body_rows
            prefix = f"rule {n + 1}: " if len(rules) > 1 else ""
            notes += [
                prefix + note for note in _combination(
                    _head(rule), rule.antecedent, positive, probability
                )
            ]
        if probability:
            kind = "probability"
            self.conditional[name] = isinstance(
                rules[0].antecedent, Condition
            )
            self.positions[name] = _prob_position(rules[0])
        elif any(
            isinstance(rule.consequent, ProbabilisticPredicate)
            for rule in rules
        ):
            kind = "probabilistic"  # p(x) :: 0.5 :- ..., p(x) :~: ...
        else:
            kind = "rule"
        if _is_fresh(name) and probability:
            given = " // …" if self.conditional[name] else ""
            label = f"PROB[…{given}]"
            detail = "\n".join(_probability_text(rule) for rule in rules)
        else:
            label = _head_text(_head(rules[0]))
            detail = "\n\n".join(
                [self.descriptions.get(name, "")]
                + [_PRINTER.walk(rule) for rule in rules]
            ).strip()
        self.nodes[name] = Node(
            name, label, kind, detail, tuple(rows), tuple(notes)
        )

    def add_answer(self, name: str, head, body, program: str) -> None:
        """Add the answer *name* of the query ``head :- body``."""
        rows, positive = self.body_rows(name, body)
        notes = _combination(head, body, positive, False)
        if not isinstance(head, FunctionApplication):
            notes = notes[:-1]  # a SQUALL query: its columns have no names
        self.nodes[name] = Node(
            name, name, "answer", program.strip(), tuple(rows), tuple(notes),
            probabilities=self._probability_columns(head, positive),
        )

    def _probability_columns(self, head, atoms) -> tuple:
        """Return where the answer *head* has probabilities, the values of
        a ``PROB`` of one of the *atoms* of its body."""
        if not isinstance(head, FunctionApplication):
            return ()
        variables = [getattr(arg, "name", None) for arg in head.args]
        found = []
        for atom in atoms:
            position = self.positions.get(atom.functor.name)
            if position is None or position >= len(atom.args):
                continue
            name = getattr(atom.args[position], "name", None)
            if name in variables:
                found.append(variables.index(name))
        return tuple(sorted(set(found)))

    def graph(self) -> QueryGraph:
        nodes = _probabilistic(self.nodes, self.edges)
        nodes = self._explained(nodes)
        flows = {"choice": "uncertain", "facts": "uncertain",
                 "probabilistic": "uncertain",
                 "probability": "probabilities"}
        edges = tuple(
            Edge(source, target, kind, atom, row,
                 flows.get(nodes[source].kind, "rows"))
            for source, target, kind, atom, row in self.edges
        )
        return QueryGraph(tuple(nodes.values()), edges)

    def _explained(self, nodes: dict) -> dict:
        """Return *nodes*, the probabilities and probabilistic rules with
        notes on the random choices they depend on."""
        explained = {}
        for name, node in nodes.items():
            if node.kind in ("probability", "probabilistic"):
                sources = _sources(name, nodes, self.edges)
                if node.kind == "probability":
                    notes = self._probability_notes(name, node, sources)
                else:
                    notes = node.notes + (
                        f"uncertain: its rows depend on {_names(sources)}",
                    )
                node = node._replace(notes=notes)
            explained[name] = node
        return explained

    def _probability_notes(self, name, node, sources) -> tuple:
        """Return the notes of the probability *name* on how it is
        computed from the random *sources* it depends on."""
        joins = tuple(n for n in node.notes if "joined" in n or (
            "combination" in n
        ))
        output = tuple(n for n in node.notes if n not in joins)
        conditional = self.conditional[name]
        holds = "A and C hold" if conditional else "A holds"
        notes = [
            "p = P(A, C) / P(C): A above given, C below" if conditional
            else "p = P(A)"
        ]
        choices = [s for s in sources if (
            self.nodes[s] is not None and self.nodes[s].kind == "choice"
        )]
        uniform = _uniform(self.relations.get(choices[0])) if (
            len(sources) == 1 and choices
        ) else None
        if uniform is not None:
            notes.append(
                f"{choices[0]} chooses one of its {uniform:,} rows, each "
                f"with probability 1/{uniform:,}"
            )
            among = ", among those where C holds" if conditional else ""
            notes.append(
                f"so p is the share of those choices where {holds}{among}"
            )
        elif sources:
            over = (
                ", over those where C holds" if conditional else ""
            )
            notes.append(
                f"p sums the probabilities of the worlds where {holds}"
                f"{over}: a world is what {_names(sources)} choose or make "
                "true"
            )
        return joins + tuple(notes) + output


def _size(value) -> Optional[int]:
    """Return the number of rows of the relation *value*, if known."""
    try:
        return len(value.value)
    except TypeError:
        return None


def _uniform(value) -> Optional[int]:
    """Return the number of rows of the choice *value* when they are all
    as likely, else None. Its probability is its first column."""
    try:
        probability = value.value.as_pandas_dataframe().iloc[:, 0]
    except (AttributeError, TypeError):
        return None
    if len(probability) and probability.max() - probability.min() <= (
        1e-9 * probability.max()
    ):
        return len(probability)
    return None


def _sources(name: str, nodes: dict, edges: list) -> list:
    """Return the random choices and probabilistic facts *name* depends
    on, not through a probability."""
    inputs = {}
    for source, target, *_ in edges:
        inputs.setdefault(target, set()).add(source)
    found, seen, todo = set(), {name}, [name]
    while todo:
        for source in inputs.get(todo.pop(), ()):
            kind = nodes[source].kind
            if kind in ("choice", "facts"):
                found.add(source)
            elif source not in seen and kind != "probability":
                seen.add(source)
                todo.append(source)
    return sorted(found)


def _head_text(head) -> str:
    """Return the head *head* as shown: ``PROB`` without its arguments."""
    args = ", ".join(
        "PROB" if _is_prob(arg) else _text(arg) for arg in head.args
    )
    return f"{head.functor.name}({args})"


def _combination(head, body, positive: list, probability: bool) -> list:
    """
    Return how a rule ``head :- body`` combines the rows of its *positive*
    atoms: the variables it joins them on, and those it keeps and drops,
    summing over them for a *probability*.
    """
    notes = []
    if len(positive) > 1:
        seen, shared = {}, []
        for atom in positive:
            for variable in set(_variables(atom)):
                seen[variable] = seen.get(variable, 0) + 1
        for atom in positive:
            shared += [
                v for v in _variables(atom)
                if seen[v] > 1 and v not in shared
            ]
        notes.append(
            f"joined on {_names(shared)}" if shared
            else "every combination of their rows"
        )
    args = getattr(head, "args", (head,) if isinstance(head, Symbol) else ())
    kept = [arg.name for arg in args if isinstance(arg, Symbol)]
    computed = [
        arg for arg in args
        if isinstance(arg, FunctionApplication) and not _is_prob(arg)
    ]
    aggregated = [v for arg in computed for v in _variables(arg)]
    # A variable kept, x of x == expression, is computed from those of
    # the expression: they are not dropped.
    for formula in _body_conjuncts(body):
        if (
            isinstance(formula, FunctionApplication)
            and formula.functor == Constant(operator.eq)
            and isinstance(formula.args[0], Symbol)
            and formula.args[0].name in kept
        ):
            expression = formula.args[1]
            notes.append(f"{formula.args[0].name} = {_text(expression)}")
            aggregated += [expression.name] if isinstance(
                expression, Symbol
            ) else _variables(expression)
    used = [
        v.name for v in extract_logic_free_variables(body)
        if isinstance(v, Symbol)
    ]
    dropped = sorted(set(used) - set(kept) - set(aggregated))
    for arg in computed:
        notes.append(f"{_text(arg)} for each ({_names(kept)})")
    if probability:
        note = f"a probability p for each ({_names(kept)})" if kept \
            else "a probability p"
        notes.append(
            f"{note}, summing over {_names(dropped)}" if dropped else note
        )
    elif dropped and not computed:
        notes.append(
            f"a row for each ({_names(kept)}), for some {_names(dropped)}"
        )
    return notes


def _body_conjuncts(body) -> tuple:
    """Return the conjuncts of *body*, on both sides of a condition."""
    if isinstance(body, Condition):
        return _conjuncts(body.conditioned) + _conjuncts(body.conditioning)
    return _conjuncts(body)


def _probability_text(rule) -> str:
    """Return ``PROB[body // condition]`` of a rule asking a probability."""
    body = rule.antecedent
    if isinstance(body, Condition):
        return (
            f"PROB[{_text(body.conditioned)} // {_text(body.conditioning)}]"
        )
    return f"PROB[{_text(body)}]"


def _probabilistic(nodes: dict, edges: list) -> dict:
    """Return *nodes*, the rules using random choices made probabilistic."""
    uncertain = {n for n, node in nodes.items() if node.kind in (
        "choice", "facts", "probabilistic"
    )}
    changed = True
    while changed:
        changed = False
        for source, target, *_ in edges:
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
    nl, program: str, squall: bool = False, descriptions=None, columns=None
) -> QueryGraph:
    """
    Return the graph of the queries of *program*, in SQUALL if *squall*,
    on the engine *nl*, whose predicates are described in *descriptions*
    and whose relations have the *columns*, by name.
    """
    if squall:
        rules, queries = _squall_queries(program)
    else:
        rules, queries = _datalog_queries(nl, program)
    builder = _Builder(nl, rules, descriptions or {}, columns or {})
    for key, (head, body) in queries.items():
        name = head.functor.name if isinstance(
            head, FunctionApplication
        ) else key
        builder.add_answer(name, head, body, program)
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


def engine_columns(engine_name: str) -> dict:
    """Return the sorts of the arguments of the predicates of the engine,
    by name, as the names of their columns."""
    try:
        signatures = engine_registry.get_signatures(engine_name)
    except Exception:  # an engine of no configuration
        return {}
    return {
        name: tuple(argument.sort for argument in arguments)
        for name, arguments in signatures.items()
    }


# The layout of the page, in pixels.
_CHAR = 7.8  # the width of a character of the labels
_MONO = 7.3  # of a row
_SANS = 6.1  # of a note
_TOP = 38  # the height of the label and kind of a card
_ROW = 17
_NOTE = 15
_HISTOGRAM = 78
_RELATION = 60  # the height of a relation
_LONGEST = 48  # the most characters of a row shown
_WRAP = 58  # of a line of a note
_GAP_X = 90
_GAP_Y = 22
_LANE = 24  # the height an edge takes crossing a column
_MARGIN = 20
_LOOP = 36  # the room above the nodes for the loop of a recursive rule


def _is_card(node: Node) -> bool:
    """Whether *node* is drawn with its rows, as a rule is."""
    return node.kind in ("rule", "probabilistic", "probability", "answer")


def _shown(text: str) -> str:
    return text if len(text) <= _LONGEST else text[:_LONGEST - 1] + "…"


def _note_lines(node: Node) -> list:
    """Return the lines of the notes of *node*, wrapped."""
    return [
        line for note in node.notes
        for line in textwrap.wrap(note, _WRAP, subsequent_indent="  ")
    ]


def _subtitle(node: Node) -> str:
    rows = next(
        (line for line in node.detail.splitlines() if line.endswith(" rows")),
        "",
    )
    name = KINDS[node.kind][0].lower()
    return f"{name} · {rows}" if rows else name


def _height(node: Node, histogram: bool) -> float:
    if not _is_card(node):
        return _RELATION if node.columns else _RELATION - 14
    notes = len(_note_lines(node)) * _NOTE + (8 if node.notes else 0)
    return _TOP + len(node.rows) * _ROW + notes + 10 + (
        _HISTOGRAM if histogram else 0
    )


def _width(node: Node) -> float:
    texts = [len(node.label) * _CHAR, len(_subtitle(node)) * _SANS]
    texts += [len(node.columns) * _MONO]
    texts += [(len(_shown(r.text)) + 6) * _MONO for r in node.rows]
    texts += [len(line) * _SANS for line in _note_lines(node)]
    if node.probabilities:  # room for the histogram and its caption
        texts.append(270)
    return max(texts) + 28


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


def _positions(graph: QueryGraph, layout: _Layout, histogram) -> tuple:
    """
    Return the box (x, y, width, height) of each node and waypoint, and
    the size of the drawing. A waypoint's box is the lane across its
    column. The node *histogram* has the distribution of its values.
    """
    nodes = {node.id: node for node in graph.nodes}
    top = _MARGIN + (_LOOP if any(
        edge.source == edge.target for edge in graph.edges
    ) else 0)

    def height(item):
        if item in nodes:
            return _height(nodes[item], item == histogram)
        return _LANE - _GAP_Y

    tallest = max(
        sum(height(item) + _GAP_Y for item in column)
        for column in layout.columns
    )
    boxes, x = {}, _MARGIN
    for column in layout.columns:
        width = max(_width(nodes[n]) for n in column if n in nodes)
        y = top + (tallest - sum(height(i) + _GAP_Y for i in column)) / 2
        for item in column:
            boxes[item] = (x, y, width, height(item))
            y += height(item) + _GAP_Y
        x += width + _GAP_X
    return boxes, x - _GAP_X + _MARGIN, top + tallest + _MARGIN


def _row_y(box: tuple, row: int) -> float:
    """Return the height of the middle of the row *row* of a card."""
    return box[1] + _TOP + row * _ROW + _ROW / 2


def _out_y(node: Node, box: tuple) -> float:
    """Return where the edges leave *node*: its label, for a card."""
    return box[1] + 14 if _is_card(node) else box[1] + box[3] / 2


def _shape(kind: str, box: tuple) -> str:
    """Return the SVG of the outline of a node of *kind*."""
    x, y, w, h = box
    if kind == "data":  # a cylinder, as a database is drawn
        e = 6
        return (
            f'<path class="shape" d="M{x},{y + e} a{w / 2},{e} 0 0 0 {w},0 '
            f'v{h - 2 * e} a{w / 2},{e} 0 0 1 {-w},0 z"/>'
            f'<path class="rim" d="M{x},{y + e} a{w / 2},{e} 0 0 1 {w},0 '
            f'a{w / 2},{e} 0 0 1 {-w},0"/>'
        )
    if kind == "ontology":  # a hexagon, as a concept map's nodes
        s = 14
        points = (
            f"{x + s},{y} {x + w - s},{y} {x + w},{y + h / 2} "
            f"{x + w - s},{y + h} {x + s},{y + h} {x},{y + h / 2}"
        )
        return f'<polygon class="shape" points="{points}"/>'
    if kind == "probability":  # corners cut
        s = 10
        points = (
            f"{x + s},{y} {x + w - s},{y} {x + w},{y + s} {x + w},"
            f"{y + h - s} {x + w - s},{y + h} {x + s},{y + h} {x},"
            f"{y + h - s} {x},{y + s}"
        )
        return f'<polygon class="shape" points="{points}"/>'
    radius = {"answer": 14, "choice": 4, "facts": 4}.get(kind, 8)
    return (
        f'<rect class="shape" x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="{radius}"/>'
    )


def _route_path(points: list) -> tuple:
    """
    Return the SVG path through *points*, the start, the left and right
    ends of the lane of each waypoint, and the end, and where its label
    goes, at the start of its last curve: a curve from a column to the
    next, a straight line across the lane of a waypoint.
    """
    (px, py), (tx, ty) = points[0], points[-1]
    if tx <= px:  # back to a column on the left: around below
        low = max(py, ty) + 60
        d = f"M{px},{py} C{px + 60},{low} {tx - 60},{low} {tx},{ty}"
        return d, ((px + tx) / 2, low - 14)
    d = f"M{px},{py}"
    for n in range(1, len(points), 2):
        x, y = points[n]
        c = (x - px) / 2
        d += f" C{px + c},{py} {x - c},{y} {x},{y}"
        label = (px + (x - px) / 4, py + (y - py) / 4)
        if n + 1 < len(points):
            px, py = points[n + 1]
            d += f" H{px}"
    return d, label


def _edge_paths(graph: QueryGraph, layout: _Layout, boxes: dict) -> list:
    """Return the SVG of the edges, each arriving at the row of its atom;
    a recursive rule's in a loop above it."""
    nodes = {node.id: node for node in graph.nodes}
    svg = []
    for n, edge in enumerate(graph.edges):
        source = nodes[edge.source]
        sbox, tbox = boxes[edge.source], boxes[edge.target]
        end_y = _row_y(tbox, edge.row) if edge.row >= 0 else (
            tbox[1] + tbox[3] / 2
        )
        if edge.source == edge.target:
            x, y, w, _ = sbox
            d = (
                f"M{x + w * .65},{y} C{x + w * .65},{y - 34} "
                f"{x + w * .35},{y - 34} {x + w * .35},{y}"
            )
            mx, my = x + w / 2, y - 28
        else:
            points = [(sbox[0] + sbox[2], _out_y(source, sbox))]
            for waypoint in layout.routes[n][1:-1]:
                x, y, w, h = boxes[waypoint]
                middle = y + (h + _GAP_Y) / 2  # of the lane
                points += [(x, middle), (x + w, middle)]
            points.append((tbox[0], end_y))
            d, (mx, my) = _route_path(points)
        title = html.escape(
            f"{edge.atom}: {FLOWS[edge.flow]}"
            + (", in the condition" if edge.kind == "given" else "")
            + (", negated" if edge.kind == "negated" else "")
        )
        label = (
            f'<text class="edge-label" x="{mx}" y="{my - 3}">not</text>'
            if edge.kind == "negated" else ""
        )
        svg.append(
            f'<g class="edge {edge.kind} {edge.flow}" data-source="'
            f'{html.escape(edge.source)}" data-target="'
            f'{html.escape(edge.target)}"><title>{title}</title>'
            f'<path d="{d}" marker-end="url(#arrow-{edge.flow})"/>'
            f"{label}</g>"
        )
    return svg


def _relation_svg(node: Node, box: tuple) -> str:
    x, y, w, h = box
    lines = [("label", node.label)]
    if node.columns:
        lines.append(("columns", node.columns))
    lines.append(("kind", _subtitle(node)))
    top = y + (h - 15 * len(lines)) / 2 + 13
    return "".join(
        f'<text class="{cls}" x="{x + w / 2}" y="{top + 15 * i}">'
        f"{html.escape(text)}</text>"
        for i, (cls, text) in enumerate(lines)
    )


def _card_svg(node: Node, box: tuple, ports: set, histogram) -> str:
    """Return the SVG of the label, rows, notes and histogram of a card;
    the rows in *ports* get the dot where an edge arrives."""
    x, y, w, h = box
    svg = [
        f'<text class="label start" x="{x + 12}" y="{y + 18}">'
        f"{html.escape(node.label)}</text>"
        f'<text class="kind start" x="{x + 12}" y="{y + 32}">'
        f"{html.escape(_subtitle(node))}</text>"
    ]
    for n, row in enumerate(node.rows):
        middle = _row_y(box, n)
        text = {"filter": f"where {row.text}"}.get(row.kind, row.text)
        svg.append(
            f'<text class="row {row.kind}" x="{x + 18}" y="{middle + 4}">'
            f"{html.escape(_shown(text))}</text>"
        )
        if n in ports:
            svg.append(
                f'<circle class="port" cx="{x}" cy="{middle}" r="3"/>'
            )
    notes_top = y + _TOP + len(node.rows) * _ROW + 6
    if node.notes:
        svg.append(
            f'<line class="rule-line" x1="{x + 10}" x2="{x + w - 10}" '
            f'y1="{notes_top}" y2="{notes_top}"/>'
        )
    for n, note in enumerate(_note_lines(node)):
        svg.append(
            f'<text class="note" x="{x + 12}" y="{notes_top + 14 + n * _NOTE}"'
            f">{html.escape(note)}</text>"
        )
    if histogram is not None:
        svg.append(_histogram_svg(
            histogram, x + 12, y + h - _HISTOGRAM - 4, w - 24
        ))
    return "".join(svg)


def _node_svg(node: Node, box: tuple, ports: set, histogram=None) -> str:
    title = html.escape(node.detail or node.label)
    inner = _card_svg(node, box, ports, histogram) if _is_card(node) \
        else _relation_svg(node, box)
    return (
        f'<g class="node {node.kind}" data-id="{html.escape(node.id)}" '
        f'tabindex="0"><title>{title}</title>{_shape(node.kind, box)}'
        f"{inner}</g>"
    )


def _histogram(graph: QueryGraph, frame) -> Optional[tuple]:
    """Return the answer whose probabilities *frame* has, the name of
    their column and their values, or None."""
    if frame is None or frame.empty:
        return None
    for node in graph.nodes:
        if node.kind == "answer" and node.probabilities:
            column = node.probabilities[0]
            if column < frame.shape[1]:
                values = frame.iloc[:, column].astype(float).dropna()
                if len(values):
                    return node.id, str(frame.columns[column]), values
    return None


def _histogram_svg(histogram: tuple, x: float, y: float, w: float) -> str:
    """Return the SVG of the histogram of the values of a probability, on
    [0, 1] when one of them is above 1/2, else on [0, their largest]."""
    _, name, values = histogram
    bins = 20
    top = 1.0 if values.max() > .5 else float(values.max()) or 1.0
    counts = [0] * bins
    for value in values:
        counts[min(int(value / top * bins), bins - 1)] += 1
    most = max(counts)
    height = _HISTOGRAM - 34
    width = w / bins
    bars = "".join(
        f'<rect class="bar" x="{x + i * width + .5}" '
        f'y="{y + 14 + height - count / most * height}" '
        f'width="{max(width - 1, 1)}" height="{count / most * height}">'
        f"<title>{count:,} rows from {i * top / bins:.3g} to "
        f"{(i + 1) * top / bins:.3g}</title></rect>"
        for i, count in enumerate(counts) if count
    )
    base = y + 14 + height
    return (
        f'<g class="histogram"><text class="note" x="{x}" y="{y + 8}">'
        f"{html.escape(name)} over the {len(values):,} rows: from "
        f"{values.min():.3g} to {values.max():.3g}</text>{bars}"
        f'<line class="axis" x1="{x}" x2="{x + w}" y1="{base}" '
        f'y2="{base}"/><text class="tick" x="{x}" y="{base + 12}">0</text>'
        f'<text class="tick end" x="{x + w}" y="{base + 12}">{top:.3g}'
        "</text></g>"
    )


def _legend(graph: QueryGraph) -> str:
    """Return the HTML of the legend of the kinds the graph has."""
    shown = {node.kind for node in graph.nodes}
    items = [
        f'<li><span class="swatch {kind}"></span><b>{name}</b>: '
        f"{html.escape(meaning)}</li>"
        for kind, (name, meaning) in KINDS.items() if kind in shown
    ]
    flows = {edge.flow for edge in graph.edges}
    items += [
        f'<li><span class="line {flow}"></span>{html.escape(meaning)}</li>'
        for flow, meaning in FLOWS.items() if flow in flows
    ]
    kinds = {edge.kind for edge in graph.edges}
    if "given" in kinds:
        items.append(
            '<li><span class="line given"></span>dashed: in the condition '
            "after // (C)</li>"
        )
    if "negated" in kinds:
        items.append(
            '<li><span class="line negated"></span>not: used negated (~), '
            "the rows it does not have</li>"
        )
    return "<ul>" + "".join(items) + "</ul>"


# The dark colours that differ from the light ones, which are the default:
# those of the theme the REPL sends the page (see frame_hooks.js), else the
# system's.
_DARK = (
    "color-scheme: dark; --bg: #0d1117; --text: #e6edf3; --muted: #9198a1; "
    "--edge: #6e7681; --data: #172554; --onto: #042f2e; --rand: #431407; --rule: #082f49; "
    "--prob: #2e1065; --ans: #052e16; --neg: #f87171;"
)

_STYLE = """
:root { color-scheme: light; --bg: #fff; --text: #1f2328; --muted: #59636e;
  --edge: #8c959f; --data: #dbeafe; --data-b: #2563eb; --onto: #ccfbf1;
  --onto-b: #0f766e; --rand: #ffedd5;
  --rand-b: #ea580c; --rule: #e0f2fe; --rule-b: #0369a1; --prob: #ede9fe;
  --prob-b: #7c3aed; --ans: #dcfce7; --ans-b: #15803d; --neg: #dc2626; }
:root[data-theme=dark] { __DARK__ }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme]) { __DARK__ } }
body { margin: 0; padding: .75rem 1rem; background: var(--bg);
  color: var(--text); font: 14px/1.4 system-ui, sans-serif; }
h1 { font-size: 1rem; margin: 0 0 .25rem; }
p { margin: 0 0 .5rem; color: var(--muted); max-width: 60rem; }
.graph { overflow: auto; }
.graph.fit svg { max-width: 100%; height: auto; }
.zoom { margin: 0 0 .5rem; padding: .2rem .65rem;
  border: 1px solid var(--edge); border-radius: 6px; background: var(--bg); color: var(--text);
  font: 12px system-ui, sans-serif; cursor: pointer; }
svg { display: block; }
.shape, .rim { stroke-width: 1.5; }
.data .shape, .data .rim { fill: var(--data); stroke: var(--data-b); }
.ontology .shape { fill: var(--onto); stroke: var(--onto-b); }
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
.node text.start, .row, .note { text-anchor: start !important; }
.label { font: 600 13px ui-monospace, Menlo, monospace; }
.columns { font: 11px ui-monospace, Menlo, monospace; }
.kind, .tick { font-size: 10.5px; fill: var(--muted) !important; }
.tick.end { text-anchor: end !important; }
.row { font: 12px ui-monospace, Menlo, monospace; }
.row.filter { font-style: italic; }
.row.negated { fill: var(--neg) !important; }
.row.heading { font: 600 11px system-ui, sans-serif;
  fill: var(--muted) !important; }
.note { font: italic 11.5px system-ui, sans-serif; }
.port { fill: var(--text); }
.rule-line, .axis { stroke: var(--edge); stroke-width: 1; }
.bar { fill: var(--prob-b); }
.edge path { fill: none; stroke-width: 1.6; }
.edge.rows path { stroke: var(--edge); }
.edge.uncertain path { stroke: var(--rand-b); }
.edge.probabilities path { stroke: var(--prob-b); stroke-width: 2.2; }
.edge.given path { stroke-dasharray: 6 4; }
.edge-label { font-size: 11px; fill: var(--neg); text-anchor: middle; }
.faded { opacity: .15; }
.selected .shape { stroke-width: 3; }
ul { list-style: none; padding: 0; margin: .5rem 0; display: grid;
  gap: .2rem; grid-template-columns: repeat(auto-fill, minmax(22rem, 1fr)); }
.swatch, .line { display: inline-block; width: 1.4rem; height: .9rem;
  margin-right: .4rem; vertical-align: middle; border: 1.5px solid; }
.swatch.data { background: var(--data); border-color: var(--data-b);
  border-radius: 40%/25%; }
.swatch.ontology { background: var(--onto); border-color: var(--onto-b);
  clip-path: polygon(15% 0, 85% 0, 100% 50%, 85% 100%, 15% 100%, 0 50%); }
.swatch.choice, .swatch.facts, .swatch.probabilistic {
  background: var(--rand); border-color: var(--rand-b); }
.swatch.facts, .swatch.probabilistic { border-style: dashed; }
.swatch.rule { background: var(--rule); border-color: var(--rule-b); }
.swatch.probability { background: var(--prob); border-color: var(--prob-b); }
.swatch.answer { background: var(--ans); border-color: var(--ans-b);
  border-radius: 1rem; }
.swatch.undefined { border-color: var(--neg); border-style: dotted; }
.line { height: 0; border-width: 2px 0 0; }
.line.rows { border-color: var(--edge); }
.line.uncertain { border-color: var(--rand-b); }
.line.probabilities { border-color: var(--prob-b); }
.line.given { border-style: dashed; border-color: var(--muted); }
.line.negated { border-color: var(--neg); }
.arrow-rows { fill: var(--edge); } .arrow-uncertain { fill: var(--rand-b); }
.arrow-probabilities { fill: var(--prob-b); }
#detail { white-space: pre-wrap; font: 12px/1.5 ui-monospace, Menlo,
  monospace; margin: .5rem 0 0; padding: .5rem; border-radius: 6px;
  border: 1px solid var(--edge); }
#detail:empty { display: none; }
""".replace("__DARK__", _DARK)

# Clicking a node shows its details and what it depends on, fading the
# rest; clicking elsewhere shows everything again.
_SCRIPT = """
// The graph fits the page at first, its whole seen at once; or at its size.
const zoom = document.getElementById("zoom");
zoom.addEventListener("click", () => {
  const fit = document.querySelector(".graph").classList.toggle("fit");
  zoom.textContent = fit ? "Actual size" : "Fit to the page";
});
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


def graph_page(graph: QueryGraph, title: str, frame=None) -> str:
    """
    Return the HTML page drawing *graph*, entitled *title*; with *frame*,
    the rows of its answer, the distribution of its probabilities.
    """
    histogram = _histogram(graph, frame)
    layout = _layout(graph)
    boxes, width, height = _positions(
        graph, layout, histogram[0] if histogram else None
    )
    ports = {}
    for edge in graph.edges:
        ports.setdefault(edge.target, set()).add(edge.row)
    markers = "".join(
        f'<marker id="arrow-{flow}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" class="arrow-{flow}"/></marker>'
        for flow in FLOWS
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
        + [
            _node_svg(
                node, boxes[node.id], ports.get(node.id, set()),
                histogram if histogram and histogram[0] == node.id else None,
            )
            for node in graph.nodes
        ]
    )
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style>"
        f"</head><body><h1>{html.escape(title)}</h1>"
        "<p>Read from left to right: the data on the left go through the "
        "rules to the answer on the right. A rule lists the atoms of its "
        "body, where the relations they use arrive, and below them how it "
        "combines them. The colour of an edge tells what goes through it: "
        "rows that hold for sure, rows that hold for some random choices "
        "only, or probabilities. Click a node to see its rules and what "
        "it depends on.</p>"
        '<button id="zoom" class="zoom" type="button">Actual size</button>'
        f'<div class="graph fit"><svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{html.escape(title)}"><defs>{markers}</defs>{body}'
        "</svg></div>"
        f'<pre id="detail"></pre>{_legend(graph)}'
        f'<script type="application/json" id="nodes">{data}</script>'
        f"<script>{frame_hooks()}</script><script>{_SCRIPT}</script>"
        "</body></html>"
    )


def last_query_page(app) -> str:
    """Return the page of the graph of the last query of *app*, with the
    distribution of the probabilities of its result when it has one."""
    program, squall = app.last_query
    engine = app.settings.engine_name
    graph = query_graph(
        app._nl, program, squall, engine_descriptions(engine),
        engine_columns(engine),
    )
    results = app.results
    frame = results[0].frame if len(results) == 1 and (
        app.results_query is app.last_query
    ) else None
    return graph_page(graph, "The last query", frame)


@command(
    ".graph", "Show where the answer of the last query comes from, and "
    "how: its data, rules and probabilities, in a graph in the browser",
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
