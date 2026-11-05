"""A transpiler from WMQL, tract_querier's query language, to Datalog.

WMQL is Python syntax over *query values*. In ``tract_querier`` every value
is a triple, componentwise: the streamlines ``tracts``, a set of ``labels``,
and the streamlines whose first and last point are in the labels,
``end0`` and ``end1``. A WMQL name ``x`` becomes four Datalog predicates:

========================  ==========================================
``q_x(x)``                the streamlines, by their id ``x``
``q_x__label(t, l)``      the labels of the query, for each tractogram
``q_x__end0(x)``          streamlines with the first point in labels
``q_x__end1(x)``          streamlines with the last point in labels
========================  ==========================================

over the relations of DESIGN.md, 3, with the atlas ``a`` fixed to a constant
(labels in WMQL are the ids of an atlas, as ``FreeSurfer.qry`` has them):
``tractogram(t)``, ``streamline(x, t, s)``, ``streamline_label(a, x, l)``,
``endpoint_label(a, x, e, l)``, and for the directional terms
``label_extent(a, l, axis, lo, hi)``, ``streamline_extent(x, axis, lo, hi)``
and ``endpoint_coord(x, e, axis, v)``. A streamline ``x`` is of one
tractogram, so the streamline components are unary; the label sets are per
tractogram, as tract_querier's are per tractography, and a rule that relates
the two reads ``t`` from ``streamline(x, t, s)``. A component that is empty
whatever the data is ``None`` and has no predicate.

The algorithm, in order (DESIGN.md, 11.4): parse the program as Python;
lower-case it, inline the imports and rewrite ``not in`` to its WMQL
precedence; walk the statements, expanding ``.side``, ``for`` and string
patterns on the syntax tree; compile each expression to one
``QueryPredicates`` by set algebra on the four components, emitting a rule
for each non-trivial operation (a component known to be empty is carried as
``None`` and disappears); copy each definition into predicates of its own;
name the predicates once all the definitions are known.

The semantics are those of ``tract_querier.query_processor`` (DESIGN.md, 11),
read from its source and checked against it, on synthetic data, by
``tests/test_wmql_oracle.py``. Names are case-insensitive; ``|=`` and ``=``
define (the second saves the query, as does a bare name); a name defined
again is a new definition, which later statements use and earlier ones do
not; ``.side`` expands to ``.left`` and ``.right``; a string is a
``fnmatch`` over the names defined so far; ``for`` and ``import`` are
expanded here.

Not supported: ``save`` and unknown functions (``WMQLUnsupported``).
``superior_of`` and ``inferior_of`` are tract_querier's, which select the
reverse of what their names say: see ``_DIRECTIONS`` and
``anatomical_vertical``.

This module imports nothing of NeuroLang: it turns text into text.
"""

import ast
import fnmatch
import numbers
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import takewhile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "QueryPredicates",
    "Transpiled",
    "WMQLError",
    "WMQLUnsupported",
    "transpile",
]

RELATIVE_TERMS = (
    "anterior_of", "posterior_of", "medial_of", "lateral_of",
    "inferior_of", "superior_of",
)

_NAME = re.compile(r"[a-z0-9_]+")


class WMQLError(ValueError):
    """A WMQL program that tract_querier would reject too."""


class WMQLUnsupported(WMQLError, NotImplementedError):
    """A construct that is not transpiled (yet)."""


@dataclass(frozen=True)
class QueryPredicates:
    """The predicates of the four components of a query; ``None`` for a
    component that is empty whatever the data."""

    tracts: Optional[str] = None
    labels: Optional[str] = None
    end0: Optional[str] = None
    end1: Optional[str] = None


@dataclass(frozen=True)
class _Atom:
    pred: str
    args: Tuple[str, ...]
    negated: bool = False
    extensional: bool = False

    def render(self, rename) -> str:
        pred = self.pred if self.extensional else rename(self.pred)
        text = f"{pred}({', '.join(self.args)})"
        return f"~{text}" if self.negated else text


@dataclass(frozen=True)
class _Comparison:
    """``(left op right)`` in a rule body; Datalog needs the parentheses."""

    left: str
    op: str
    right: str

    def render(self, rename) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True)
class _Rule:
    head: _Atom
    body: Tuple[Union[_Atom, _Comparison], ...] = ()
    # the statement of the program that the rule comes from, for the comments
    origin: Optional[str] = field(default=None, compare=False)

    def render(self, rename) -> str:
        head = self.head.render(rename)
        if not self.body:
            return head
        return f"{head} :- " + ", ".join(b.render(rename) for b in self.body)

    def dependencies(self) -> List[str]:
        return [
            b.pred for b in self.body
            if isinstance(b, _Atom) and not b.extensional
        ]


@dataclass(frozen=True)
class Transpiled:
    """A transpiled WMQL program.

    Attributes
    ----------
    defined : dict
        WMQL name (lower case, ``x.left`` as such) to the
        :class:`QueryPredicates` of its last definition.
    saved : tuple of str
        The names saved by ``=`` or by naming them, in order.
    skipped : dict
        With ``skip_unsupported``: the names that were left out, with the
        reason (an unsupported construct, or a name that was left out).
    """

    defined: Dict[str, QueryPredicates]
    saved: Tuple[str, ...]
    skipped: Dict[str, str]
    _rules: Tuple[_Rule, ...] = field(repr=False)
    # name -> kind -> the rule that gives the last definition its public name
    _public: Dict[str, Dict[str, _Rule]] = field(repr=False)
    # the rules that give their public names to the earlier definitions
    _earlier: Tuple[_Rule, ...] = field(repr=False)

    @staticmethod
    def _render(rules: Iterable[_Rule], comments: bool = False) -> str:
        out, last = [], None
        for rule in rules:
            if comments and rule.origin and rule.origin != last:
                out.append(f"% {rule.origin}")
            last = rule.origin
            out.append(rule.render(lambda p: p))
        return "\n".join(out) + "\n"

    @property
    def text(self) -> str:
        """The whole program, in the syntax of NeuroLang's Datalog."""
        public = [
            rule for kinds in self._public.values() for rule in kinds.values()
        ]
        return self._render(
            list(self._rules) + list(self._earlier) + public
        )

    def program(
        self,
        *names: str,
        components: Sequence[str] = ("tracts", "labels", "end0", "end1"),
        comments: bool = False,
    ) -> str:
        """The program without the rules that ``names`` do not need.

        Parameters
        ----------
        names : str
            WMQL names; default, the saved ones.
        components : sequence of str
            Which of ``tracts``, ``labels``, ``end0``, ``end1`` of the
            names are wanted; ``("labels",)`` leaves out the rules over the
            streamlines. A component is computed only if it is wanted or
            another wanted component needs it.
        comments : bool
            Put a ``%`` comment with the statement of the WMQL program
            (``.side`` and ``for`` expanded) over the rules that come from
            it, and one over the names.
        """
        names = names or self.saved
        aliases, stack = [], []
        for name in names:
            if name not in self.defined:
                raise WMQLError(f"unknown query {name!r}")
            for kind in components:
                rule = self._public[name].get(kind)
                if rule is not None:
                    aliases.append(rule)
                    stack.extend(rule.dependencies())
        by_head: Dict[str, List[_Rule]] = {}
        for rule in self._rules:
            by_head.setdefault(rule.head.pred, []).append(rule)
        needed = set()
        while stack:
            pred = stack.pop()
            if pred in needed:
                continue
            needed.add(pred)
            for rule in by_head.get(pred, ()):
                stack.extend(rule.dependencies())
        kept = [r for r in self._rules if r.head.pred in needed]
        text = self._render(kept, comments)
        if comments and aliases:
            text += "% the names of the queries\n"
        return text + self._render(aliases)


# --- preprocessing: a port of tract_querier's RewritePreprocess and
# RewriteChangeNotInPrescedence ------------------------------------------


class _Preprocess(ast.NodeTransformer):
    """Lower-case everything and inline the imports."""

    def __init__(self, include_folders: Sequence[str]):
        self.include_folders = list(include_folders)

    def visit_Attribute(self, node):
        return ast.copy_location(
            ast.Attribute(
                value=self.visit(node.value), attr=node.attr.lower()
            ),
            node,
        )

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=node.id.lower()), node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(node.value.lower()), node)
        return node

    def visit_Import(self, node):
        statements = []
        for alias in node.names:
            for folder in self.include_folders or ["."]:
                path = os.path.join(folder, alias.name)
                if os.path.isfile(path):
                    break
            else:
                raise WMQLError(f"imported file not found: {alias.name}")
            with open(path) as f:
                module = _parse(f.read(), path)
            statements.extend(self.visit(module).body)
        return statements


class _NotInPrecedence(ast.NodeTransformer):
    """``a and b not in c`` is ``(a and b) not in c``: ``not in`` binds
    looser than ``and`` and ``or``, in WMQL (not in Python)."""

    def visit_BoolOp(self, node):
        def not_in(value):
            return isinstance(value, ast.Compare) and isinstance(
                value.ops[0], ast.NotIn
            )

        plain = list(takewhile(lambda v: not not_in(v), node.values[1:]))
        if len(plain) == len(node.values) - 1:
            return node
        compare = node.values[len(plain) + 1]
        left = ast.copy_location(
            ast.BoolOp(
                op=node.op, values=[node.values[0]] + plain + [compare.left]
            ),
            node,
        )
        new = ast.copy_location(
            ast.Compare(
                left=left, ops=compare.ops, comparators=compare.comparators
            ),
            node,
        )
        rest = node.values[len(plain) + 2:]
        if not rest:
            return self.visit(new)
        return self.visit(
            ast.copy_location(
                ast.BoolOp(op=node.op, values=[new] + rest), node
            )
        )


def _parse(source: str, filename: str = "<wmql>") -> ast.Module:
    try:
        return ast.parse(source, filename=filename)
    except SyntaxError as e:
        raise WMQLError(
            f"syntax error in {filename}, line {e.lineno}: {e.text}"
        ) from e


# --- code generation ----------------------------------------------------

_X = ("x",)
_TL = ("t", "l")

# The directional terms, as tract_querier's ``process_relative_term`` has
# them: ``term -> (axis, face, operator)``. A streamline is selected if its
# ``face`` (``lo`` the minimum, ``hi`` the maximum, along ``axis``: 0 x, 1 y,
# 2 z, RAS) is ``operator`` than that face of the union of the boxes of the
# labels of the argument; its end is selected, by the same test, on the
# coordinate of the end. NOTE superior_of and inferior_of are the reference's,
# whose tests are the reverse of anterior_of and posterior_of: the streamlines
# whose top is *below* that of the box, and whose bottom is *above* its
# bottom. ``_ANATOMICAL_VERTICAL`` is the same table with them the other way
# round, for ``anatomical_vertical=True`` (DESIGN.md, 11.2, 15.20).
_DIRECTIONS = {
    "anterior_of": (1, "hi", ">"),
    "posterior_of": (1, "lo", "<"),
    "superior_of": (2, "hi", "<"),
    "inferior_of": (2, "lo", ">"),
}
_ANATOMICAL_VERTICAL = {
    "superior_of": (2, "hi", ">"),
    "inferior_of": (2, "lo", "<"),
}
# medial_of and lateral_of depend on the side of the argument, which must be
# written ``x.left`` or ``x.right``: (term, side) -> (axis, face, operator)
_SIDED_DIRECTIONS = {
    ("medial_of", "left"): (0, "hi", ">"),
    ("medial_of", "right"): (0, "lo", "<"),
    ("lateral_of", "left"): (0, "lo", "<"),
    ("lateral_of", "right"): (0, "hi", ">"),
}


class _Transpiler:
    def __init__(
        self,
        atlas: str,
        skip_unsupported: bool = False,
        anatomical_vertical: bool = False,
    ):
        self.skip_unsupported = skip_unsupported
        self.directions = dict(_DIRECTIONS)
        if anatomical_vertical:
            self.directions.update(_ANATOMICAL_VERTICAL)
        self.skipped: Dict[str, str] = {}
        if '"' in atlas:
            raise WMQLError("the atlas name cannot have a double quote")
        self.atlas = f'"{atlas}"'
        self.rules: List[_Rule] = []
        self.defined: Dict[str, QueryPredicates] = {}
        self.versions: Dict[str, List[QueryPredicates]] = {}
        self.saved: List[str] = []
        self._count = 0
        self._universe = None
        self._origin = None

    # predicates and rules
    def fresh(self) -> str:
        self._count += 1
        return f"qi_{self._count}"

    def emit(self, pred, args, *body):
        self.rules.append(
            _Rule(_Atom(pred, args), tuple(body), origin=self._origin)
        )

    @staticmethod
    def atom(pred, args, negated=False):
        return _Atom(pred, args, negated)

    def ext(self, pred, *args):
        return _Atom(pred, args, extensional=True)

    def decoding(self):
        """``streamline(x, t, s)``: the tractogram ``t`` of a streamline
        ``x``, for a rule that relates it to a label set ``(t, l)``."""
        return self.ext("streamline", "x", "t", "s")

    # set operations on one component (None is the empty set)
    def union(self, a, b, args):
        if a is None or a == b:
            return b
        if b is None:
            return a
        p = self.fresh()
        self.emit(p, args, self.atom(a, args))
        self.emit(p, args, self.atom(b, args))
        return p

    def intersection(self, a, b, args):
        if a is None or b is None:
            return None
        if a == b:
            return a
        p = self.fresh()
        self.emit(p, args, self.atom(a, args), self.atom(b, args))
        return p

    def difference(self, a, b, args):
        if a is None or a == b:
            return None
        if b is None:
            return a
        p = self.fresh()
        self.emit(p, args, self.atom(a, args), self.atom(b, args, True))
        return p

    # operations on queries
    def q_union(self, a, b):
        return QueryPredicates(
            self.union(a.tracts, b.tracts, _X),
            self.union(a.labels, b.labels, _TL),
            self.union(a.end0, b.end0, _X),
            self.union(a.end1, b.end1, _X),
        )

    def q_intersection(self, a, b):
        # the streamlines and the endpoints are intersected, the labels
        # are united
        return QueryPredicates(
            self.intersection(a.tracts, b.tracts, _X),
            self.union(a.labels, b.labels, _TL),
            self.intersection(a.end0, b.end0, _X),
            self.intersection(a.end1, b.end1, _X),
        )

    def q_difference(self, a, b):
        return QueryPredicates(
            self.difference(a.tracts, b.tracts, _X),
            self.difference(a.labels, b.labels, _TL),
            self.difference(a.end0, b.end0, _X),
            self.difference(a.end1, b.end1, _X),
        )

    def universe(self):
        """The labels traversed by some streamline of each tractogram."""
        if self._universe is None:
            self._universe = self.fresh()
            self.emit(
                self._universe, _TL,
                self.ext("streamline_label", self.atlas, "x", "l"),
                self.decoding(),
            )
        return self._universe

    def q_complement(self, a):
        """``not`` and unary ``-``: the streamlines that traverse some
        label outside the labels; no endpoints."""
        labels = self.difference(self.universe(), a.labels, _TL)
        tracts = None
        if labels is not None:
            tracts = self.fresh()
            self.emit(
                tracts, _X,
                self.ext("streamline_label", self.atlas, "x", "l"),
                self.decoding(),
                self.atom(labels, _TL),
            )
        return QueryPredicates(tracts, labels, None, None)

    def _only_tracts(self, a):
        """The streamlines of ``a`` whose labels are all in its labels."""
        if a.tracts is None:
            return None
        outside = self.fresh()
        # of the streamlines of ``a`` only: a join that starts from them, and
        # a result of their size, not of all the rows of ``streamline_label``
        body = [
            self.atom(a.tracts, _X),
            self.ext("streamline_label", self.atlas, "x", "l"),
        ]
        if a.labels is not None:
            body += [self.decoding(), self.atom(a.labels, _TL, True)]
        self.emit(outside, _X, *body)
        return self.difference(a.tracts, outside, _X)

    def q_invert(self, a):
        """``~a``: the streamlines of ``a`` with all their labels in the
        labels of ``a``; **no endpoints**."""
        return QueryPredicates(self._only_tracts(a), a.labels, None, None)

    def q_only(self, a):
        """``only(a)``: as ``~a``, and the endpoint sets keep the
        streamlines whose end label is in the labels."""

        def ends(end, e):
            if end is None or a.labels is None:
                return None
            p = self.fresh()
            self.emit(
                p, _X,
                self.atom(end, _X),
                self.ext("endpoint_label", self.atlas, "x", str(e), "l"),
                self.decoding(),
                self.atom(a.labels, _TL),
            )
            return p

        return QueryPredicates(
            self._only_tracts(a), a.labels, ends(a.end0, 0), ends(a.end1, 1)
        )

    def q_endpoints_in(self, a):
        return QueryPredicates(
            self.union(a.end0, a.end1, _X), a.labels, a.end0, a.end1
        )

    def q_both_endpoints_in(self, a):
        return QueryPredicates(
            self.intersection(a.end0, a.end1, _X), a.labels, a.end0, a.end1
        )

    def q_directional(self, term: str, arg: QueryPredicates, side=None):
        """``anterior_of(A)`` and the other directional terms.

        The reference takes the union of the boxes of the labels of ``A``
        (its streamlines are not read), and keeps the streamlines whose
        ``face`` is beyond that face of the box (``_DIRECTIONS``); so do
        their ends, on the coordinate of the end. The labels of the result
        are those that the streamlines traverse. A label without a box
        (the background, or a label absent from the volume) is ignored, where
        the reference raises; no label, or no box, gives an empty query.
        """
        axis, face, op = (
            _SIDED_DIRECTIONS[(term, side)] if side
            else self.directions[term]
        )
        if arg.labels is None:
            return QueryPredicates()
        aggregate = "max" if face == "hi" else "min"
        bound = self.fresh()  # the face of the box of the labels, per t
        self.emit(
            bound, ("t", f"{aggregate}({face})"),
            self.atom(arg.labels, _TL),
            self.ext("label_extent", self.atlas, "l", str(axis), "lo", "hi"),
        )
        edge = {"lo": "lo", "hi": "hi"}[face]

        tracts = self.fresh()
        self.emit(
            tracts, _X,
            self.ext("streamline_extent", "x", str(axis), "lo", "hi"),
            self.decoding(),
            self.atom(bound, ("t", "box")),
            _Comparison(edge, op, "box"),
        )
        ends = []
        for e in (0, 1):
            p = self.fresh()
            self.emit(
                p, _X,
                self.ext("endpoint_coord", "x", str(e), str(axis), "v"),
                self.decoding(),
                self.atom(bound, ("t", "box")),
                _Comparison("v", op, "box"),
            )
            ends.append(p)
        labels = self.fresh()
        self.emit(
            labels, _TL,
            self.atom(tracts, _X),
            self.ext("streamline_label", self.atlas, "x", "l"),
            self.decoding(),
        )
        return QueryPredicates(tracts, labels, *ends)

    def q_label(self, value: int):
        t = self.fresh()
        self.emit(
            t, _X, self.ext("streamline_label", self.atlas, "x", str(value))
        )
        labels = self.fresh()
        self.emit(labels, ("t", str(value)), self.ext("tractogram", "t"))
        ends = []
        for e in (0, 1):
            p = self.fresh()
            self.emit(
                p, _X,
                self.ext("endpoint_label", self.atlas, "x", str(e),
                         str(value)),
            )
            ends.append(p)
        return QueryPredicates(t, labels, *ends)

    # expressions
    def expr(self, node) -> QueryPredicates:
        method = getattr(self, f"expr_{type(node).__name__}", None)
        if method is None:
            raise WMQLError(
                f"invalid operation {type(node).__name__} "
                f"in line {getattr(node, 'lineno', '?')}"
            )
        return method(node)

    def expr_Name(self, node):
        return self.lookup(node.id, node)

    def expr_Attribute(self, node):
        if not isinstance(node.value, ast.Name):
            raise WMQLError(f"invalid query in line {node.lineno}")
        return self.lookup(f"{node.value.id}.{node.attr}", node)

    def lookup(self, name, node):
        if name in self.skipped:
            raise WMQLUnsupported(f"{name} was skipped: {self.skipped[name]}")
        if name not in self.defined:
            raise WMQLError(
                f"invalid query name in line {node.lineno}: {name}"
            )
        return self.defined[name]

    def expr_Constant(self, node):
        value = node.value
        if isinstance(value, str):
            matches = fnmatch.filter(list(self.defined), value)
            result = QueryPredicates()
            for name in matches:
                result = self.q_union(result, self.defined[name])
            return result
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            return self.q_label(int(value))
        raise WMQLUnsupported(f"{value!r} is not a label id or a pattern")

    def expr_BoolOp(self, node):
        if isinstance(node.op, ast.Or):
            op = self.q_union
        elif isinstance(node.op, ast.And):
            op = self.q_intersection
        else:
            raise WMQLError(f"invalid operation in line {node.lineno}")
        result = self.expr(node.values[0])
        for value in node.values[1:]:
            result = op(result, self.expr(value))
        return result

    def expr_BinOp(self, node):
        ops = {
            ast.Add: self.q_union,
            ast.Mult: self.q_intersection,
            ast.Sub: self.q_difference,
        }
        if type(node.op) not in ops:
            raise WMQLError(f"invalid operation in line {node.lineno}")
        return ops[type(node.op)](self.expr(node.left), self.expr(node.right))

    def expr_UnaryOp(self, node):
        operand = self.expr(node.operand)
        if isinstance(node.op, ast.Invert):
            return self.q_invert(operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, (ast.USub, ast.Not)):
            return self.q_complement(operand)
        raise WMQLError(f"syntax error in query line {node.lineno}")

    def expr_Compare(self, node):
        if any(not isinstance(op, ast.NotIn) for op in node.ops):
            raise WMQLError(f"invalid syntax in query line {node.lineno}")
        result = self.expr(node.left)
        for value in node.comparators:
            result = self.q_difference(result, self.expr(value))
        return result

    def expr_Call(self, node):
        if not (
            isinstance(node.func, ast.Name)
            and len(node.args) == 1
            and not node.keywords
        ):
            raise WMQLError(f"invalid call in line {node.lineno}")
        name = node.func.id
        if name in RELATIVE_TERMS:
            return self.directional(name, node)
        calls = {
            "only": self.q_only,
            "endpoints_in": self.q_endpoints_in,
            "both_endpoints_in": self.q_both_endpoints_in,
        }
        if name not in calls:
            raise WMQLUnsupported(f"{name} (line {node.lineno})")
        return calls[name](self.expr(node.args[0]))

    def directional(self, name, node):
        """The argument must be a name, or ``name.left`` / ``name.right``
        (the side matters to ``medial_of`` and ``lateral_of``, which cannot
        have a bare name: the reference fails on it); nothing else, as in the
        reference."""
        arg = node.args[0]
        side = None
        if isinstance(arg, ast.Attribute):
            if arg.attr not in ("left", "right"):
                raise WMQLError(
                    f"attribute not recognized for {name} in line "
                    f"{node.lineno}: use .left or .right"
                )
            side = arg.attr
        elif not isinstance(arg, ast.Name):
            raise WMQLError(
                f"the argument of {name} (line {node.lineno}) must be a "
                "query name, not an expression"
            )
        sided = name in ("medial_of", "lateral_of")
        if sided and side is None:
            raise WMQLError(
                f"{name} (line {node.lineno}) needs an argument ending in "
                ".left or .right"
            )
        return self.q_directional(
            name, self.expr(arg), side if sided else None
        )

    # statements
    def statement(self, node):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Expr)):
            self._origin = " ".join(ast.unparse(node).split())
        if isinstance(node, ast.Assign):
            if len(node.targets) > 1:
                raise WMQLError(f"invalid assignment in line {node.lineno}")
            self.assignment(node, node.targets[0], saved=True)
        elif isinstance(node, ast.AugAssign):
            if not isinstance(node.op, ast.BitOr):
                raise WMQLError(f"invalid assignment in line {node.lineno}")
            self.assignment(node, node.target, saved=False)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Name):
            if node.value.id not in self.defined:
                raise WMQLError(
                    f"query {node.value.id} not known, line {node.lineno}"
                )
            self.save(node.value.id)
        elif isinstance(node, ast.For):
            self.loop(node)
        else:
            raise WMQLError(
                f"invalid operation {type(node).__name__} "
                f"in line {getattr(node, 'lineno', '?')}"
            )

    def assignment(self, node, target, saved):
        if isinstance(target, ast.Name):
            self.define(target.id, node.value, saved)
        elif isinstance(target, ast.Attribute) and target.attr == "side":
            for side, opposite in (("left", "right"), ("right", "left")):
                copy = deepcopy(node)
                for sub in ast.walk(copy):
                    if isinstance(sub, ast.Attribute):
                        if sub.attr == "side":
                            sub.attr = side
                        elif sub.attr == "opposite":
                            sub.attr = opposite
                self.statement(copy)
        elif isinstance(target, ast.Attribute) and isinstance(
            target.value, ast.Name
        ):
            self.define(f"{target.value.id}.{target.attr}", node.value, saved)
        else:
            raise WMQLError(f"invalid assignment in line {node.lineno}")

    def define(self, name, value, saved):
        try:
            query = self.expr(value)
        except WMQLUnsupported as e:
            if not self.skip_unsupported:
                raise
            self.skipped[name] = str(e)
            self.defined.pop(name, None)
            return
        self.skipped.pop(name, None)
        own = query  # no copy: the public name is an alias, at the output
        self.defined[name] = own
        self.versions.setdefault(name, []).append(own)
        if saved:
            self.save(name)

    def save(self, name):
        if name not in self.saved:
            self.saved.append(name)

    def loop(self, node):
        if not isinstance(node.target, ast.Name):
            raise WMQLError(f"invalid for in line {node.lineno}")
        variable = node.target.id
        iterable = node.iter
        if isinstance(iterable, ast.Constant) and isinstance(
            iterable.value, str
        ):
            items = fnmatch.filter(list(self.defined), iterable.value)
        elif isinstance(iterable, ast.List):
            items = []
            for item in iterable.elts:
                if not isinstance(item, ast.Name):
                    raise WMQLError(
                        f"error in for, line {node.lineno}: the elements "
                        "of the list must be query names"
                    )
                items.append(item.id)
        else:
            raise WMQLError(f"invalid for in line {node.lineno}")
        for item in items:
            body = deepcopy(node.body)
            for statement in body:
                for sub in ast.walk(statement):
                    if isinstance(sub, ast.Name) and sub.id == variable:
                        sub.id = item
            for statement in body:
                self.statement(statement)

    # the names of the predicates, once the definitions are known
    def finish(self) -> Transpiled:
        self._origin = None
        mangled: Dict[str, str] = {}
        public: Dict[str, Dict[str, _Rule]] = {}
        earlier: List[_Rule] = []
        suffixes = {
            "tracts": "", "labels": "__label", "end0": "__end0",
            "end1": "__end1",
        }
        names = {}  # name -> the public QueryPredicates of its last version
        for name, versions in self.versions.items():
            base = name.replace(".", "_")
            if not _NAME.fullmatch(base):
                raise WMQLError(f"cannot make a predicate of {name!r}")
            if mangled.setdefault(base, name) != name:
                raise WMQLError(
                    f"queries {mangled[base]!r} and {name!r} would both "
                    f"be q_{base}"
                )
            for k, query in enumerate(versions, 1):
                last = k == len(versions)
                tag = "" if last else f"__v{k}"
                kinds = {}
                for kind, pred in vars(query).items():
                    if pred is None:
                        continue
                    args = _TL if kind == "labels" else _X
                    name_ = f"q_{base}{tag}{suffixes[kind]}"
                    rule = _Rule(
                        _Atom(name_, args), (_Atom(pred, args),)
                    )
                    kinds[kind] = rule
                    if not last:
                        earlier.append(rule)
                if last:
                    public[name] = kinds
                    names[name] = QueryPredicates(
                        **{k_: r.head.pred for k_, r in kinds.items()}
                    )
        defined = {
            name: names[name] for name in self.defined if name in names
        }
        return Transpiled(
            defined=defined,
            saved=tuple(self.saved),
            skipped=dict(self.skipped),
            _rules=tuple(self.rules),
            _public=public,
            _earlier=tuple(earlier),
        )


def transpile(
    source: str,
    atlas: str,
    include_folders: Sequence[str] = (),
    skip_unsupported: bool = False,
    anatomical_vertical: bool = False,
) -> Transpiled:
    """Transpile a WMQL program to Datalog.

    Parameters
    ----------
    source : str
        The WMQL program, as tract_querier reads it.
    atlas : str
        The atlas whose label ids the program uses: the constant in the
        first column of ``streamline_label`` and ``endpoint_label``.
    include_folders : sequence of str
        Where ``import`` looks for files.
    skip_unsupported : bool
        Leave out the queries that use an unsupported construct, and those
        that use them (``Transpiled.skipped`` says which and why), instead
        of raising :class:`WMQLUnsupported`. For programs that use a
        construct that is not transpiled (``save``, an unknown function).
    anatomical_vertical : bool
        ``superior_of`` and ``inferior_of`` as their names say (the streamlines
        whose top is above the top of the box, whose bottom is below its
        bottom) instead of as tract_querier has them, the other way round
        (``_DIRECTIONS``).

    Returns
    -------
    Transpiled
        ``.text`` is the Datalog program; ``.defined`` maps each WMQL name
        to the predicates of its four components.

    Raises
    ------
    WMQLError
        On what tract_querier rejects. :class:`WMQLUnsupported`, a
        subclass, on what it accepts and this does not (directional terms).
    """
    module = _parse(source)
    module = _NotInPrecedence().visit(
        _Preprocess(include_folders).visit(module)
    )
    transpiler = _Transpiler(atlas, skip_unsupported, anatomical_vertical)
    for statement in module.body:
        try:
            transpiler.statement(statement)
        except WMQLUnsupported:
            if not skip_unsupported:
                raise
    return transpiler.finish()
