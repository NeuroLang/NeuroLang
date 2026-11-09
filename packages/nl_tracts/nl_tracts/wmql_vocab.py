"""WMQL, tract_querier's query language, in the vocabulary of tractography.

:func:`to_vocabulary` translates a ``.qry`` program into rules over the
predicates of ``vocabulary.dl`` (DESIGN.md, 16 and 17), such as a person
writes them: one predicate per query, ``UF.side`` becomes ``uf(x, h)``, of
a streamline ``x`` and a hemisphere ``h``. Its semantics is the natural one
of the vocabulary, not tract_querier's exactly (DESIGN.md, 16.5 and 17.3).

* A name defined by labels, with ``|=`` (``frontal.side |= ...``, the label
  ids of ``FreeSurfer.qry``, a wildcard ``'*orpus_callosum*'``), is a
  **term** of the atlas's mapping, the string ``"frontal"``: it has no
  predicate. With ``terms``, a name that is not one of them is replaced by
  the terms it is made of.
* Any other definition, and one by labels with ``=``, which tract_querier
  saves, is a **predicate**: ``name.side`` is ``name(x, h)`` (``.side`` the
  variable ``h``, ``.opposite`` the variable ``o`` of
  ``opposite_hemisphere(h, o)``), ``name.left`` the rule
  ``name(x, "left")``, a name without a side ``name(x)``.
* ``or`` is ``|``, ``and`` a conjunction, ``not in`` and ``not`` a
  negation; ``endpoints_in(E)`` the conditions of ``E`` on one end ``e``,
  ``both_endpoints_in`` on both; ``only`` ``only_in``, or for several terms
  an auxiliary predicate, ``<name>_leaves``; the directional terms those of
  the vocabulary, anatomical (``superior_of`` is above).
* ``HEMISPHERE.side``, the union of every ``.left`` (``.right``) name by
  ``'*.left'``, is ``in_hemisphere(x, h)``; a term without a side is of any
  hemisphere, ``_``.

What the vocabulary cannot say is not translated: the definition is left
out, with the reason (:class:`WMQLUnsupported`, or ``Translation.skipped``
with ``skip_unsupported``), and so are those that use it.

The parsing is that of tract_querier, as ``nl_tracts.wmql`` of the exact
transpiler had it: Python syntax, lower-cased, ``import`` inlined, ``not in``
binding looser than ``and`` and ``or``, ``for`` loops expanded. This module
imports nothing of NeuroLang: it turns text into text.
"""

import ast
import csv
import fnmatch
import os
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.resources import files
from itertools import takewhile
from typing import Dict, List, Optional, Sequence, Tuple

from .vocabulary import predicates as _vocabulary_predicates

__all__ = [
    "BUNDLES",
    "Translation",
    "WMQLError",
    "WMQLUnsupported",
    "bundle",
    "to_vocabulary",
    "tract_names",
    "translate",
]

#: The shipped translations of tract_querier's ``.qry`` files, in
#: ``nl_tracts/bundles`` (made by ``tools/make_bundle_modules.py``).
BUNDLES = (
    "freesurfer_queries", "freesurfer_queries_bsf2016",
    "freesurfer_queries_bsf2016_toad", "mori_queries",
)

DIRECTIONS = (
    "anterior_of", "posterior_of", "superior_of", "inferior_of",
    "medial_of", "lateral_of",
)
#: Names a predicate cannot have: those of the vocabulary, of the stored
#: relations and functions of nl_tracts (DESIGN.md, 3.1), and ``ans``. A
#: query named so is prefixed ``wmql_``.
RESERVED = frozenset(_vocabulary_predicates()) | {
    "tractogram", "streamline", "atlas", "label", "label_term",
    "opposite_hemisphere", "streamline_label", "endpoint_label",
    "streamline_extent", "endpoint_coord", "endpoint_point",
    "label_extent", "streamline_length", "streamline_id", "subject_of",
    "index_of", "ans",
}
_WIDTH = 79
_HEMISPHERE = "%hemisphere"  # the term of '*.left': in_hemisphere


class WMQLError(ValueError):
    """A WMQL program that tract_querier rejects too."""


class WMQLUnsupported(WMQLError, NotImplementedError):
    """What the vocabulary cannot say."""


# --- parsing: tract_querier's preprocessing ---------------------------------


def _parse(source: str, filename: str) -> ast.Module:
    try:
        module = ast.parse(source, filename=filename)
    except SyntaxError as e:
        raise WMQLError(
            f"syntax error in {filename}, line {e.lineno}: {e.text}"
        ) from e
    for node in module.body:  # the file and text of each statement, for
        node.wmql_file = os.path.basename(filename)  # its documentation
        node.wmql_source = [
            line.rstrip() for line in (
                ast.get_source_segment(source, node) or ""
            ).expandtabs(4).splitlines()
        ]
    return module


class _Preprocess(ast.NodeTransformer):
    """Lower-case everything and inline the imports."""

    def __init__(self, include_folders: Sequence[str]):
        self.include_folders = list(include_folders)

    def visit_Attribute(self, node):
        return ast.copy_location(ast.Attribute(
            value=self.visit(node.value), attr=node.attr.lower()), node)

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
    looser than ``and`` and ``or`` in WMQL (not in Python)."""

    def visit_BoolOp(self, node):
        def not_in(value):
            return isinstance(value, ast.Compare) and isinstance(
                value.ops[0], ast.NotIn)

        plain = list(takewhile(lambda v: not not_in(v), node.values[1:]))
        if len(plain) == len(node.values) - 1:
            return node
        compare = node.values[len(plain) + 1]
        left = ast.copy_location(ast.BoolOp(
            op=node.op, values=[node.values[0]] + plain + [compare.left]),
            node)
        new = ast.copy_location(ast.Compare(
            left=left, ops=compare.ops, comparators=compare.comparators), node)
        rest = node.values[len(plain) + 2:]
        if not rest:
            return self.visit(new)
        return self.visit(ast.copy_location(
            ast.BoolOp(op=node.op, values=[new] + rest), node))


def _statements(source, filename, include_folders):
    """The statements of the program, ``for`` loops expanded."""
    module = _NotInPrecedence().visit(
        _Preprocess(include_folders).visit(_parse(source, filename)))
    out = []
    for node in module.body:
        out.extend(_expand_for(node))
    return out


def _expand_for(node):
    if not isinstance(node, ast.For):
        return [node]
    if not isinstance(node.target, ast.Name) or not isinstance(
            node.iter, ast.List) or not all(
            isinstance(i, ast.Name) for i in node.iter.elts):
        raise WMQLUnsupported(
            f"for in line {node.lineno}: only over a list of names")
    statements = []
    for item in node.iter.elts:
        for statement in deepcopy(node.body):
            for sub in ast.walk(statement):
                if isinstance(sub, ast.Name) and sub.id == node.target.id:
                    sub.id = item.id
            statement.wmql_source = [ast.unparse(statement)]
            statement.wmql_file = getattr(node, "wmql_file", None)
            statements.append(statement)
    return statements


# --- formulas ----------------------------------------------------------------
# Arguments are rendered strings: "x", "h", "o", "e", '"temporal"', "_", "0".
# A hemisphere is "h" (.side), "o" (.opposite), '"left"', '"right"' or "_".


@dataclass(frozen=True)
class _Atom:
    pred: object  # a str, or the _Definition / _Leaves it calls
    args: Tuple[str, ...]


@dataclass(frozen=True)
class _Not:
    formula: object


@dataclass(frozen=True)
class _And:
    formulas: Tuple[object, ...]


@dataclass(frozen=True)
class _Or:
    formulas: Tuple[object, ...]


@dataclass(frozen=True)
class _Exists:
    variable: str
    formula: object


def _and(*formulas):
    flat = []
    for f in formulas:
        flat.extend(f.formulas if isinstance(f, _And) else (f,))
    return flat[0] if len(flat) == 1 else _And(tuple(flat))


def _or(formulas):
    flat = []
    for f in formulas:
        for g in (f.formulas if isinstance(f, _Or) else (f,)):
            if g not in flat:
                flat.append(g)
    return flat[0] if len(flat) == 1 else _Or(tuple(flat))


def _not(f):
    """The negation, pushed to the atoms (De Morgan) but for ``∃``."""
    if isinstance(f, _Not):
        return f.formula
    if isinstance(f, _Or):
        return _and(*(_not(g) for g in f.formulas))
    if isinstance(f, _And):
        return _or([_not(g) for g in f.formulas])
    return _Not(f)


def _replace(f, old, new):
    """``f`` with the argument ``old`` replaced by ``new``."""
    if isinstance(f, _Atom):
        return _Atom(f.pred, tuple(new if a == old else a for a in f.args))
    if isinstance(f, _Not):
        return _Not(_replace(f.formula, old, new))
    if isinstance(f, _Exists):
        return _Exists(f.variable, _replace(f.formula, old, new))
    return type(f)(tuple(_replace(g, old, new) for g in f.formulas))


def _variables(f, positive=True):
    """The variables of ``f``; with ``positive``, those a positive atom
    binds (in every disjunct of a disjunction)."""
    if isinstance(f, _Atom):
        return {a for a in f.args if a[:1].isalpha()}
    if isinstance(f, _Not):
        return set() if positive else _variables(f.formula, False)
    if isinstance(f, _Exists):
        return _variables(f.formula, positive) - {f.variable}
    sets = [_variables(g, positive) for g in f.formulas]
    if positive and isinstance(f, _Or):
        return set.intersection(*sets)
    return set().union(*sets)


def _opposite(hemi):
    try:
        return {"h": "o", "o": "h", '"left"': '"right"',
                '"right"': '"left"'}[hemi]
    except KeyError:
        raise WMQLUnsupported("the opposite of a name without a side")


# --- the program -------------------------------------------------------------


@dataclass(eq=False)
class _Definition:
    """A statement ``name[.side] = value``: a label set (``leaves``), a
    predicate (``group``), or both."""

    base: str
    side: Optional[str]  # "side", "left", "right" or None
    source: List[str]  # the lines of the statement
    saved: bool
    # ("id", n) | ("ref", definition, hemi) | ("hemi", hemi)
    leaves: Optional[list] = None
    group: Optional["_Group"] = None
    body: object = None
    auxiliaries: list = field(default_factory=list)
    file: Optional[str] = None  # the .qry file of the statement

    def head(self):
        hemi = {"side": ("h",), None: ()}.get(self.side, (f'"{self.side}"',))
        return ("x",) + hemi


@dataclass(eq=False)
class _Group:
    """The definitions that are rules of one predicate."""

    base: str
    sided: bool
    keys: set = field(default_factory=set)
    name: str = ""


@dataclass(eq=False)
class _Leaves:
    """``<name>_leaves``: ``x`` goes through a label outside the terms of an
    ``only`` of several terms."""

    owner: _Definition
    terms: Tuple[Tuple[str, str], ...]
    name: str = ""

    def body(self):
        sided = self.owner.side == "side"
        return _and(
            *((_Atom("in_hemisphere", ("x", "h")),) if sided else ()),
            _Atom("streamline_label", ("a", "x", "l")),
            *(_Not(_Atom("term_label", ("a", "l", f'"{t}"', hemi)))
              for t, hemi in self.terms),
        )


@dataclass(frozen=True)
class Translation:
    """A WMQL program in the vocabulary of tractography.

    Attributes
    ----------
    text : str
        The rules, with a header and the WMQL statement over each.
    queries : dict
        Each WMQL query translated, as tract_querier names it (``uf.left``,
        ``cc``), to the atom that gives its streamlines ``x``
        (``uf(x, "left")``, ``cc(x)``): the last definition of the name.
    skipped : dict
        The WMQL names left out, with the reason.
    terms : tuple of str
        The terms the rules use: the atlas's mapping must have them.
    """

    text: str
    queries: Dict[str, str]
    skipped: Dict[str, str]
    terms: Tuple[str, ...]


class _Translator:
    def __init__(self, terms, skip_unsupported):
        self.terms = None if terms is None else set(terms)
        self.skip_unsupported = skip_unsupported
        self.env: Dict[str, _Definition] = {}  # WMQL name -> definition
        self.skipped: Dict[str, str] = {}
        self.groups: Dict[str, List[_Group]] = {}
        self.definitions: List[_Definition] = []

    # statements
    def statement(self, node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            self.assign(node, node.targets[0], saved=True)
        elif isinstance(node, ast.AugAssign) and isinstance(
                node.op, ast.BitOr):
            self.assign(node, node.target, saved=False)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Name):
            pass  # saves a query: every query has its predicate already
        else:
            raise WMQLError(f"invalid statement in line {node.lineno}")

    def assign(self, node, target, saved):
        if isinstance(target, ast.Name):
            base, side = target.id, None
        elif isinstance(target, ast.Attribute) and isinstance(
                target.value, ast.Name) and target.attr in (
                "side", "left", "right"):
            base, side = target.value.id, target.attr
        else:
            raise WMQLError(f"invalid assignment in line {node.lineno}")
        keys = ([f"{base}.left", f"{base}.right"] if side == "side"
                else [base if side is None else f"{base}.{side}"])
        d = _Definition(base, side, getattr(node, "wmql_source", []), saved,
                        file=getattr(node, "wmql_file", None))
        self.hemis, self.n_ends, self.current = side == "side", 0, d
        try:
            d.leaves = self.label_set(node.value)
            if d.leaves is None or saved:
                d.body = (self.labels_formula(d, d.head()[1:] or ("_",))
                          if d.leaves is not None
                          else self.formula(node.value))
                self.check_safe(d)
                d.group = self.group(base, side, keys)
        except WMQLError as e:
            if not self.skip_unsupported:
                raise type(e)(f"{' '.join(keys)}: {e}") from e
            for key in keys:
                self.skipped[key] = str(e)
                self.env.pop(key, None)
            return
        for key in keys:
            self.skipped.pop(key, None)
            self.env[key] = d
        self.definitions.append(d)

    def group(self, base, side, keys):
        sided = side is not None
        groups = self.groups.setdefault(base, [])
        if not groups or groups[-1].sided != sided or groups[-1].keys & set(
                keys):
            groups.append(_Group(base, sided))
        groups[-1].keys |= set(keys)
        return groups[-1]

    def check_safe(self, d):
        bound = _variables(d.body)
        if "x" not in bound:
            raise WMQLUnsupported(
                "no condition that selects streamlines, only negations")
        if d.side == "side" and not bound & {"h", "o"}:
            raise WMQLUnsupported("no condition on the hemisphere .side")

    # names
    def lookup(self, node):
        """``(definition, hemi)`` of a name; ``.side`` gives the two
        definitions of ``.left`` and ``.right``, which must agree."""
        if isinstance(node, ast.Name):
            return self.get(node.id, node), "_"
        if not (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)):
            raise WMQLError(f"invalid name in line {node.lineno}")
        base, attr = node.value.id, node.attr
        if attr in ("left", "right"):
            return self.get(f"{base}.{attr}", node), f'"{attr}"'
        if attr not in ("side", "opposite") or not self.hemis:
            raise WMQLError(
                f"invalid name in line {node.lineno}: {base}.{attr}")
        for key in (f"{base}.left", f"{base}.right"):
            if key in self.skipped:
                raise WMQLUnsupported(f"uses {base}.{attr}, skipped")
        left = self.get(f"{base}.left", node)
        right = self.get(f"{base}.right", node)
        if left is not right and not (
                left.leaves is not None and right.leaves is not None) and not (
                left.group is not None and left.group is right.group):
            raise WMQLUnsupported(
                f"{base}.left and {base}.right are not of one kind")
        return left, "h" if attr == "side" else "o"

    def ref(self, key):
        """``(definition, hemi)`` of a name matched by a wildcard."""
        side = key.rsplit(".", 1)[-1] if "." in key else None
        return self.env[key], f'"{side}"' if side in ("left", "right") else "_"

    def get(self, name, node):
        if name in self.skipped:
            raise WMQLUnsupported(f"uses {name}, skipped")
        if name not in self.env:
            raise WMQLUnsupported(
                f"{name} is not defined (line {node.lineno}): tract_querier "
                "rejects it")
        return self.env[name]

    def is_term(self, d):
        if d.leaves and all(leaf[0] == "hemi" for leaf in d.leaves):
            return True
        return not d.saved and (self.terms is None or d.base in self.terms)

    # label sets
    def label_set(self, node):
        """The leaves of a union of labels, or None for any other query."""
        if isinstance(node, (ast.Name, ast.Attribute)):
            try:
                d, hemi = self.lookup(node)
            except WMQLError:  # raised again by formula
                return None
            return [("ref", d, hemi)] if d.leaves is not None else None
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return None
            if isinstance(node.value, int):
                return [("id", node.value)]
            if node.value in ("*.left", "*.right"):
                return [("hemi", f'"{node.value[2:]}"')]
            matches = [
                self.ref(k) for k in fnmatch.filter(self.env, node.value)]
            if matches and all(d.leaves is not None for d, _ in matches):
                return [("ref", d, hemi) for d, hemi in matches]
            return None
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            parts = node.values
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            parts = [node.left, node.right]
        else:
            return None
        sets = [self.label_set(p) for p in parts]
        return None if None in sets else [leaf for s in sets for leaf in s]

    def term_leaves(self, d, hemi):
        """The (term, hemi) of a label set ``d`` named with ``hemi``."""
        if self.is_term(d):
            if d.leaves and d.leaves[0][0] == "hemi":
                return [(_HEMISPHERE, d.leaves[0][1] if hemi == "_" else hemi)]
            return [(d.base, hemi)]
        if d.side in ("left", "right") and hemi in ("h", "o"):
            raise WMQLUnsupported(
                f"{d.base} is not a term, and {d.base}.left and "
                f"{d.base}.right are two definitions")
        out = []
        for leaf in d.leaves:
            if leaf[0] == "id":
                raise WMQLUnsupported(
                    f"{d.base} is made of label ids and is not a term")
            if leaf[0] == "hemi":
                out.append((_HEMISPHERE, leaf[1]))
                continue
            _, ref, tok = leaf
            if d.side == "side":
                tok = {"h": hemi, "o": _opposite(hemi)}.get(tok, tok)
            for t in self.term_leaves(ref, tok):
                if t not in out:
                    out.append(t)
        return out

    def leaves_of(self, node):
        """The (term, hemi) of an argument made of label sets with ``or`` and
        ``and`` (whose labels tract_querier unites), or None."""
        if isinstance(node, ast.BoolOp) or isinstance(node, ast.BinOp) and \
                isinstance(node.op, (ast.Add, ast.Mult)):
            parts = (node.values if isinstance(node, ast.BoolOp)
                     else [node.left, node.right])
            sets = [self.leaves_of(p) for p in parts]
            if None in sets:
                return None
            return list(dict.fromkeys(t for s in sets for t in s))
        if isinstance(node, (ast.Name, ast.Attribute)):
            d, hemi = self.lookup(node)  # raises what makes it unusable
            return None if d.leaves is None else self.term_leaves(d, hemi)
        leaves = self.label_set(node)
        if leaves is None:
            return None
        return self.term_leaves(_Definition("", None, [], False, leaves), "_")

    def labels_formula(self, d, hemis):
        return _or([
            _Atom("in_hemisphere", ("x", hemi)) if term == _HEMISPHERE
            else _Atom("traverses", ("x", f'"{term}"', hemi))
            for term, hemi in self.term_leaves(d, hemis[0])
        ])

    # queries over streamlines
    def formula(self, node):
        if isinstance(node, (ast.Name, ast.Attribute)):
            return self.call(*self.lookup(node))
        if isinstance(node, ast.Constant):
            leaves = self.label_set(node)
            if isinstance(node.value, str) and leaves is None:
                keys = fnmatch.filter(self.env, node.value)
                if not keys:
                    raise WMQLUnsupported(f"{node.value!r} matches no name")
                return _or([self.call(*self.ref(k)) for k in keys])
            if leaves is None or leaves[0][0] == "id":
                raise WMQLUnsupported(
                    f"{node.value!r} in a query: a label id is not a term")
            return self.labels_formula(
                _Definition("", None, [], False, leaves), ("_",))
        if isinstance(node, ast.BoolOp):
            parts = [self.formula(v) for v in node.values]
            return (_or(parts) if isinstance(node.op, ast.Or)
                    else _and(*parts))
        if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Mult, ast.Sub)):
            left, right = self.formula(node.left), self.formula(node.right)
            return {ast.Add: lambda: _or([left, right]),
                    ast.Mult: lambda: _and(left, right),
                    ast.Sub: lambda: _and(left, _not(right))}[
                type(node.op)]()
        if isinstance(node, ast.Compare) and all(
                isinstance(op, ast.NotIn) for op in node.ops):
            return _and(self.formula(node.left), *(
                _not(self.formula(c)) for c in node.comparators))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.UAdd):
                return self.formula(node.operand)
            if isinstance(node.op, (ast.Not, ast.USub)):
                return _not(self.formula(node.operand))
            return self.only(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and len(node.args) == 1 and not node.keywords:
            name, arg = node.func.id, node.args[0]
            if name in DIRECTIONS:
                term, hemi = self.box(name, arg)
                return _Atom(name, ("x", f'"{term}"', hemi))
            if name == "only":
                return self.only(arg)
            if name == "endpoints_in":
                return self.endpoints_in(arg)
            if name == "both_endpoints_in":
                f = self.ends(arg)
                return _and(_replace(f, "§e", "0"), _replace(f, "§e", "1"))
            raise WMQLUnsupported(f"{name} (line {node.lineno})")
        raise WMQLError(f"invalid query in line {node.lineno}")

    def call(self, d, hemi):
        """A name in a query: its predicate, or its terms."""
        if d.group is not None:
            return _Atom(d.group, ("x",) + ((hemi,) if d.group.sided else ()))
        return self.labels_formula(d, (hemi,))

    def box(self, name, arg):
        """The term of a directional term: one term, of a side."""
        if not isinstance(arg, (ast.Name, ast.Attribute)):
            raise WMQLError(f"the argument of {name} must be a name")
        leaves = self.leaves_of(arg)
        if leaves is None or len(leaves) != 1 or leaves[0][0] == _HEMISPHERE:
            raise WMQLUnsupported(
                f"{name} of a query that is not one term: the box of a "
                "union of terms is not in the vocabulary")
        if leaves[0][1] == "_":
            raise WMQLUnsupported(
                f"{name} of a name without a side: the box of a term over "
                "both hemispheres is not in the vocabulary")
        return leaves[0]

    def only(self, arg):
        leaves = self.leaves_of(arg)
        if leaves is None or any(t == _HEMISPHERE for t, _ in leaves):
            raise WMQLUnsupported(
                "only of a query that is not made of terms with or and and")
        if len(leaves) == 1 and leaves[0][1] != "_":
            term, hemi = leaves[0]
            return _Atom("only_in", ("x", f'"{term}"', hemi))
        aux = _Leaves(self.current, tuple(leaves))
        self.current.auxiliaries.append(aux)
        return _and(self.formula(arg), _Not(
            _Atom(aux, ("x", "h") if self.current.side == "side" else ("x",))))

    def endpoints_in(self, arg):
        f = self.ends(arg)
        atoms = f.formulas if isinstance(f, _Or) else (f,)
        if all(isinstance(a, _Atom) and a.pred == "endpoint_in"
               for a in atoms):
            return _or([_Atom("ends_in", ("x",) + a.args[2:]) for a in atoms])
        self.n_ends += 1
        e = "e" if self.n_ends == 1 else f"e{self.n_ends}"
        return _Exists(e, _replace(f, "§e", e))

    def ends(self, node):
        """The conditions on an end ``§e`` of the streamline."""
        if isinstance(node, (ast.Name, ast.Attribute)):
            leaves = self.leaves_of(node)
            if leaves is None or any(t == _HEMISPHERE for t, _ in leaves):
                raise WMQLUnsupported(
                    "endpoints_in of a query that is not made of terms")
            return _or([_Atom("endpoint_in", ("x", "§e", f'"{t}"', hemi))
                        for t, hemi in leaves])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in DIRECTIONS and len(node.args) == 1:
            term, hemi = self.box(node.func.id, node.args[0])
            return _Atom(f"endpoint_{node.func.id}",
                         ("x", "§e", f'"{term}"', hemi))
        if isinstance(node, ast.BoolOp):
            parts = [self.ends(v) for v in node.values]
            return (_or(parts) if isinstance(node.op, ast.Or)
                    else _and(*parts))
        if isinstance(node, ast.Compare) and all(
                isinstance(op, ast.NotIn) for op in node.ops):
            return _and(self.ends(node.left), *(
                _not(self.ends(c)) for c in node.comparators))
        raise WMQLUnsupported(
            f"endpoints_in of {ast.unparse(node)}: not terms, directions, "
            "and, or, not in")

    # the names of the predicates, and the text
    def finish(self, title, qry_file=None) -> Translation:
        taken = set(RESERVED)
        for base, groups in self.groups.items():
            for k, group in enumerate(groups, 1):
                name = base if k == len(groups) else f"{base}_{k}"
                if name in RESERVED or not name[:1].isalpha():
                    name = f"wmql_{name}"
                while name in taken:
                    name += "_"
                group.name = name
                taken.add(name)
        blocks, terms, named = [], set(), set()
        for d in self.definitions:
            if d.group is None:
                continue
            lines = []
            for k, aux in enumerate(d.auxiliaries, 1):
                aux.name = f"{d.group.name}_leaves" + (
                    f"_{k}" if k > 1 else "")
                while aux.name in taken:
                    aux.name += "_"
                taken.add(aux.name)
                head = _Atom(aux, d.head()[:1] + (
                    ("h",) if d.side == "side" else ()))
                lines.append(_leaves_doc(head, aux, d.group.name))
                lines.append(_rule(head, aux.body(), terms))
            lines.append(_definition_doc(
                d, qry_file, first=d.group.name not in named))
            named.add(d.group.name)
            lines.append(_rule(_Atom(d.group, d.head()), d.body, terms))
            blocks.append("\n".join(lines))
        queries = {
            key: _render(_Atom(d.group, ("x",) + (
                (f'"{key.rsplit(".", 1)[1]}"',) if d.group.sided else ())))
            for key, d in self.env.items() if d.group is not None
        }
        header = _header(title, self.skipped, sorted(terms))
        text = header + "\n\n" + "\n\n".join(blocks) + "\n"
        return Translation(text, queries, dict(self.skipped),
                           tuple(sorted(terms)))


def tract_names(qry_file: Optional[str]) -> Dict[str, Tuple[str, str]]:
    """The full name and a sentence describing each bundle of a ``.qry``
    file of tract_querier, by its WMQL name, lower case: ``uf`` of
    ``freesurfer_queries.qry`` is the uncinate fasciculus. None for a file
    that is not one of them (``tract_names.tsv``)."""
    text = files("nl_tracts").joinpath("tract_names.tsv").read_text()
    rows = csv.DictReader(
        (line for line in text.splitlines() if not line.startswith("#")),
        delimiter="\t")
    return {
        row["query"]: (row["name"], row["description"])
        for row in rows if qry_file in row["files"].split()
    }


def _doc(lines):
    """The ``%%`` documentation block of ``lines``."""
    return "\n".join(f"%% {line}".rstrip() for line in lines)


def _fill(text):
    return textwrap.wrap(text, _WIDTH - 3, break_on_hyphens=False)


def _signature(atom):
    sorts = {"x": "streamline"}
    args = [f"{a}: {sorts.get(a, 'hemisphere')}" for a in atom.args]
    return f"{_name(atom.pred)}({', '.join(args)})"


def _leaves_doc(head, aux, owner):
    terms = ", ".join(f'"{t}"' for t, _ in aux.terms)
    where = " of hemisphere h" if "h" in head.args else ""
    return _doc([_signature(head)] + _fill(
        f"Auxiliary of {owner}: x goes through a label{where} that is of "
        f"none of the terms of its only(...), {terms}."))


def _definition_doc(d, qry_file, first):
    """The documentation of the rule of the definition ``d``: its
    signature; for the ``first`` of its predicate, the full name of the
    bundle and what it is, if the table of ``qry_file`` has it, a paragraph;
    then the WMQL statement it is translated from."""
    file = d.file if d.file not in (None, "<wmql>") else qry_file
    shipped = file is not None and os.path.splitext(file)[0] in BUNDLES
    source = f"tract_querier's {file}" if shipped else file or "WMQL"
    lines = [_signature(_Atom(d.group, ("x",) + (
        ("h",) if d.group.sided else ())))]
    if first:
        name = tract_names(file).get(d.base) if shipped else None
        if name is not None:
            full, sentence = name
            if not full.startswith("tract_querier"):
                full = full[:1].upper() + full[1:]
            lines += _fill(f"{full}: {sentence}.")
        else:
            lines += _fill(f"The query {d.base} of {source}.")
        lines.append("")  # the first paragraph, the one the assistant lists
    return _doc(lines + [f"Translated from {source}:"] + d.source)


def _name(pred):
    return pred if isinstance(pred, str) else pred.name


def _render(f):
    if isinstance(f, _Atom):
        return f"{_name(f.pred)}({', '.join(f.args)})"
    if isinstance(f, _Not):
        return "~" + _render(f.formula)
    if isinstance(f, _Exists):
        body = f.formula.formulas if isinstance(f.formula, _And) else (
            f.formula,)
        return f"exists({f.variable}; {', '.join(map(_render, body))})"
    if isinstance(f, _Or):
        return "(" + " | ".join(_render(g) for g in f.formulas) + ")"
    return "(" + ", ".join(_render(g) for g in f.formulas) + ")"


def _conjuncts(body):
    """The top-level conjuncts: ``∃e`` of a positive end is the rule's."""
    out = []
    for f in (body.formulas if isinstance(body, _And) else (body,)):
        if isinstance(f, _Exists):
            out.extend(_conjuncts(f.formula))
        else:
            out.append(f)
    return out


def _rule(head, body, terms):
    items = _conjuncts(body)
    # the positive conditions first, so that a negation's variables are bound
    items = ([f for f in items if _variables(f)]
             + [f for f in items if not _variables(f)])
    if any("o" in _variables(f, False) for f in items):
        # before the first condition on o, if h is bound by then; else
        # after the first that binds o
        bound, at = set(), None
        for i, f in enumerate(items):
            if "o" in _variables(f, False) and "h" in bound:
                at = i
                break
            if "o" in _variables(f):
                at = i + 1
                break
            bound |= _variables(f)
        items.insert(len(items) if at is None else at,
                     _Atom("opposite_hemisphere", ("h", "o")))
    _terms(items, terms)
    return "\n".join([f"{_render(head)} :-"] + _pack(items, 4, "", ""))


def _pack(items, indent, close, tail):
    """Lines of ``items``, a conjunction (``close`` empty: the body of a
    rule) or the formulas of an ``(a, b)`` or ``(a | b)`` that ends with
    ``close``: filled up to the width, a formula too long broken in turn."""
    disjunction = close == "|)"
    lines, line, broken = [], None, False
    for i, f in enumerate(items):
        last = i == len(items) - 1
        end = (close[-1:] + tail) if last else ("" if disjunction else ",")
        text = _render(f) + end
        if line is not None:
            joined = line + (" | " if disjunction else " ") + text
            if len(joined) <= _WIDTH and not broken:
                line = joined
                continue
            lines.append(line)
            start = " " * (indent + 1) + "| " if disjunction else \
                " " * (indent + (1 if close else 0))
        else:
            start = " " * indent + ("(" if close else "")
        line = start + text
        broken = len(line) > _WIDTH and isinstance(f, (_And, _Or))
        if broken:
            *full, line = _pack(f.formulas, len(start),
                                "|)" if isinstance(f, _Or) else ")", end)
            full[:1] = [start + full[0][len(start):]] if full else []
            if not full:
                line = start + line[len(start):]
            lines.extend(full)
    return lines + [line]


def _terms(items, terms):
    for f in items:
        if isinstance(f, _Atom):
            terms.update(a[1:-1] for a in f.args
                         if a.startswith('"') and a not in (
                             '"left"', '"right"', '"none"'))
        elif isinstance(f, (_Not, _Exists)):
            _terms([f.formula], terms)
        elif isinstance(f, (_And, _Or)):
            _terms(f.formulas, terms)


def _header(title, skipped, terms):
    lines = [
        f"% The bundles of {title}, in the vocabulary of tractography",
        "% (nl_tracts, DESIGN.md, 17): written by nl_tracts.wmql_vocab, one",
        "% predicate per query of tract_querier's language, WMQL, with the",
        "% statement of WMQL over it. x is a streamline, h a hemisphere,",
        '% "left" or "right" (name.side is name(x, h)); a name of labels is a',
        '% term of the atlas\'s mapping ("frontal"). The semantics is the',
        "% vocabulary's (DESIGN.md, 16.5); superior_of and inferior_of are",
        "% anatomical (above, below the box of the term): the reverse of",
        "% tract_querier's (DESIGN.md, 15.20).",
        "%",
        "% The terms it uses: " + ", ".join(terms) + ".",
    ]
    if skipped:
        lines += ["%", "% Not translated (DESIGN.md, 17.2):"]
        for name, why in skipped.items():
            base, _, side = name.rpartition(".")
            if side == "left" and skipped.get(f"{base}.right") == why:
                name = f"{base}.side"
            elif side == "right" and skipped.get(f"{base}.left") == why:
                continue
            lines.append(f"%   {name}: {why}")
    out = []
    for line in lines:  # the term list may be long
        while len(line) > _WIDTH:
            cut = line.rindex(" ", 0, _WIDTH)
            out.append(line[:cut])
            line = "%   " + line[cut + 1:]
        out.append(line)
    return "\n".join(out)


def translate(
    source: str,
    include_folders: Sequence[str] = (),
    terms: Optional[Sequence[str]] = None,
    skip_unsupported: bool = False,
    title: str = "a WMQL program",
    qry_file: Optional[str] = None,
) -> Translation:
    """Translate a WMQL program to rules over the vocabulary.

    Parameters
    ----------
    source : str
        The program, as tract_querier reads it.
    include_folders : sequence of str
        Where ``import`` looks for files.
    terms : sequence of str, optional
        The terms of the atlas's mapping. A name of labels that is not one
        of them stands for the terms it is made of; made of label ids, it
        cannot be used. None: every name of labels is a term.
    skip_unsupported : bool
        Leave out what the vocabulary cannot say, and what uses it
        (``Translation.skipped``), instead of raising
        :class:`WMQLUnsupported`.
    title : str
        What the header calls the program.
    qry_file : str, optional
        The name of the ``.qry`` file of the program. Each predicate is
        documented by a ``%%`` block: its signature, ``uf(x: streamline,
        h: hemisphere)``, the full name of the bundle and what it is, for
        the files of tract_querier (:func:`tract_names`), and the WMQL
        statement it is translated from, of this file or of one it
        imports.
    """
    translator = _Translator(terms, skip_unsupported)
    for node in _statements(source, qry_file or "<wmql>", include_folders):
        translator.statement(node)
    return translator.finish(title, qry_file)


def to_vocabulary(source: str, include_folders: Sequence[str] = (),
                  **kwargs) -> str:
    """The text of :func:`translate`."""
    return translate(source, include_folders, **kwargs).text


def bundle(name: str) -> str:
    """The rules of a shipped translation (:data:`BUNDLES`), such as
    ``bundle("freesurfer_queries")``, which has ``uf(x, h)``."""
    if name not in BUNDLES:
        raise ValueError(
            f"unknown bundle module {name!r}: one of {', '.join(BUNDLES)}")
    return files("nl_tracts").joinpath("bundles", f"{name}.dl").read_text()
