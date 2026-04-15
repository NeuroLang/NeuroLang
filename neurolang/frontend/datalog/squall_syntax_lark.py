"""
SQUALL (Semantically controlled Query-Answerable Logical Language) parser.

Translates controlled English sentences into NeuroLang logical expressions
using a Lark grammar and Transformer. Uses Continuation-Passing Style (CPS)
for correct quantifier scoping following Montague semantics.

Supported CNL constructs
------------------------
Basic sentences::

    squall ?s reports             →  reports(s)
    squall every study plays      →  ∀x. plays(x) ← study(x)
    squall a study plays          →  ∃x. study(x) ∧ plays(x)

Relative clauses::

    every study that plays        →  ∀x. study(x) ∧ plays(x)
    every study that ~reports X   →  ∀x. study(x) ∧ reports(x, X)

Negation-as-failure (Datalog \\+)::

    every study that does not activate   →  ∀x. study(x) ∧ ¬activate(x)

Aggregation::

    define as max_items for every Item ?i ;
        where every Max of the Quantity where ?i item_count per ?i.
    →  max_items(i, max({q : item_count(i,q)})) :- item(i)

Probabilistic rules::

    every study probably activates        →  ProbabilisticFact(p̃, activates(?s))
    every study activates with probability 0.3
                                          →  ProbabilisticFact(0.3, activates(?s))

Queries::

    obtain every item that activates      →  SquallProgram with Query expression

Known stubs (parsed but not semantically implemented)
------------------------------------------------------
- **Conditioned rules** — ``rule_body1_cond`` / ``rule_body1_cond_prior`` /
  ``rule_body1_cond_posterior``: the conditioning NP is silently discarded;
  only the sentence ``s`` is returned.
- **``rule_body2_cond``** — present in the grammar but has no transformer
  handler; falls through to ``_default`` and returns a raw list.
- **Inverse transitive prefix ``~``** — ``transitive_inv`` and
  ``transitive_multiple_inv`` now return ``_InverseVerbSymbol`` instances
  which emit ``InvertedFunctionApplication`` when called; argument-order
  inversion is applied by ``ResolveInvertedFunctionApplicationMixin``.

References
----------
S. Ferré, "SQUALL: The expressiveness of SPARQL 1.1 made available
as a controlled natural language", Data & Knowledge Engineering, 2014.
"""
import os

import numpy as np
from lark import Lark, Transformer
from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

from ...datalog import Conjunction, Fact, Implication, Negation, Union
from ...datalog.aggregation import AggregationApplication
from ...datalog.expressions import AggregationApplication as _AggApp
from ...expressions import (
    Constant,
    FunctionApplication,
    Query,
    Symbol,
)
from ...logic import Disjunction, ExistentialPredicate, UniversalPredicate
from ...probabilistic.expressions import ProbabilisticFact
from .squall import InvertedFunctionApplication


class SquallProgram:
    """
    Container for a parsed SQUALL program that contains both rule definitions
    and ``obtain`` queries.

    When a SQUALL program contains only rule definitions (``define as …``),
    the parser returns a plain ``Union`` or ``Implication`` for backward
    compatibility.  When ``obtain`` clauses are present, a ``SquallProgram``
    is returned so the handler can separate rules from queries.

    Attributes
    ----------
    rules : list
        ``Implication`` / ``Fact`` / ``ProbabilisticFact`` objects.
    queries : list
        CPS noun-phrase callables, one per ``obtain`` clause.
    """

    def __init__(self, rules, queries):
        self.rules = list(rules)
        self.queries = list(queries)


GRAMMAR_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "neurolang_natural.lark"
)

with open(GRAMMAR_PATH, "r") as f:
    GRAMMAR = f.read()


COMPILED_GRAMMAR = Lark(
    GRAMMAR,
    parser="earley",
    ambiguity="resolve",
)


COMPARISON_OPS = {
    ("greater",): gt,
    ("greater", "equal"): ge,
    ("lower",): lt,
    ("lower", "equal"): le,
    ("equal",): eq,
    ("not", "equal"): ne,
}

# Mapping from aggregation function names (as they appear in SQUALL nouns or
# the AGG_FUNC grammar terminal) to their corresponding Constant-wrapped
# callables.  Grammar-valid names: count, sum, max, min, average.
_AGG_FUNC_MAP = {
    "count":   Constant(len),
    "sum":     Constant(sum),
    "max":     Constant(max),
    "min":     Constant(min),
    "average": Constant(np.mean),
}


class _InverseVerbSymbol:
    """
    Thin callable wrapper returned by ``transitive_inv`` /
    ``transitive_multiple_inv``.  Calling it with ``(subject, *objects)``
    produces an :class:`InvertedFunctionApplication` so that
    :class:`ResolveInvertedFunctionApplicationMixin` can reverse the argument
    order at walk time, without any changes to ``_apply_ops`` or ``rel_vpn``.
    """

    def __init__(self, symbol):
        self.symbol = symbol
        self.name = symbol.name  # lets downstream code treat it like a Symbol

    def __call__(self, *args):
        return InvertedFunctionApplication(self.symbol, args)


class SquallTransformer(Transformer):
    """
    Transforms a SQUALL parse tree into NeuroLang logical expressions.

    Noun phrases and quantified expressions use CPS: they are represented
    as functions that take a continuation (predicate over an individual)
    and produce a logical formula with the quantifier in scope.
    """

    def start(self, args):
        return args[0]

    def squall(self, args):
        rules = []
        queries = []
        for a in args:
            if a is None:
                continue
            if isinstance(a, tuple) and len(a) == 2 and a[0] == '_query':
                queries.append(a[1])
            else:
                rules.append(a)

        if queries:
            return SquallProgram(rules=rules, queries=queries)

        # Backward compat: no queries → return Union or single rule
        if len(rules) == 1:
            return rules[0]
        return Union(tuple(rules))

    def sentence(self, args):
        return args[0]

    # ---- Rules ----

    def rule_op(self, args):
        items = [a for a in args if a is not None and not (isinstance(a, str) and a.lower() == "probably")]
        verb = items[0]  # verb1 (Symbol = head predicate name)
        body_result = items[1]  # from rule_body1: (head_args, body_formula)

        if isinstance(body_result, tuple) and len(body_result) == 2 and body_result[0] == '_rule_body':
            head_args, body_formula = body_result[1]
            if head_args:
                if isinstance(head_args, (list, tuple)):
                    head = verb(*head_args)
                else:
                    head = verb(head_args)
            else:
                head = verb()
            return Implication(head, body_formula)
        else:
            return Implication(verb(), body_result if isinstance(body_result, Conjunction) else Constant(True))

    def rule_op_prob(self, args):
        # "define as verb with probability np rule_body1"
        # args: [verb1, np_prob_cps, body_result]
        items = [a for a in args if a is not None]
        verb = items[0]
        np_prob = items[1]
        body_result = items[2]
        if isinstance(body_result, tuple) and body_result[0] == '_rule_body':
            head_args, body_formula = body_result[1]
        else:
            head_args, body_formula = [], Constant(True)
        head = verb(*head_args) if head_args else verb()
        # Extract probability value from CPS NP
        if callable(np_prob) and not isinstance(np_prob, (Symbol, Constant)):
            prob_val = np_prob(lambda x: x)
        else:
            prob_val = np_prob
        return Implication(ProbabilisticFact(prob_val, head), body_formula)

    def rule_opnn(self, args):
        """Build an n-ary Datalog rule from ``define as verbn rule_body1 ops``.

        ``rule_body1`` supplies the primary subject variable(s) and body
        formula.  ``ops`` is a CPS noun phrase for the object position(s);
        ``_extract_datalog_body`` is called to append the object variable(s)
        to ``head_args`` and any restriction predicates to the body.
        """
        items = [a for a in args if a is not None]
        verb = items[0]  # Symbol for the head predicate
        body_result = items[1]  # from rule_body1: ('_rule_body', (args, formula))
        ops = items[2] if len(items) > 2 else None

        # Extract body args and formula
        if isinstance(body_result, tuple) and body_result[0] == '_rule_body':
            body_args, body_formula = body_result[1]
        else:
            body_args = []
            body_formula = body_result

        # Extract ops args - ops is a CPS NP
        all_body_parts = [body_formula]
        head_args = list(body_args)

        if ops is not None:
            if callable(ops) and not isinstance(ops, (Symbol, Constant)):
                # CPS NP: collect bound variables into head_args and body
                # predicates into all_body_parts via _extract_datalog_body.
                all_body_parts.extend(_extract_datalog_body(ops, head_args))
            elif isinstance(ops, (list, tuple)):
                head_args.extend(ops)

        if len(all_body_parts) == 1:
            full_body = all_body_parts[0]
        else:
            full_body = Conjunction(tuple(all_body_parts))

        consequent = verb(*head_args) if head_args else verb()
        return Implication(consequent, full_body)

    def rule_opnn_per(self, args):
        # "define as probably verbn rule_body1 [conditioned?] [break?] ops"
        # The VP now carries ProbabilisticFact via vpdo_prob_vn, so this
        # rule_opnn_per handler collects the result similarly to rule_opnn.
        items = [a for a in args if a is not None and not (isinstance(a, str) and a.lower() in ('probably', 'conditioned'))]
        verb = items[0]
        body_result = items[1] if len(items) > 1 else None
        ops = items[2] if len(items) > 2 else None

        if isinstance(body_result, tuple) and body_result[0] == '_rule_body':
            body_args, body_formula = body_result[1]
        else:
            body_args, body_formula = [], Constant(True)

        head_args = list(body_args)
        all_body_parts = [body_formula]
        if ops is not None and callable(ops) and not isinstance(ops, (Symbol, Constant)):
            _extract_datalog_body(ops, head_args)

        fresh_prob = Symbol.fresh()
        consequent = verb(*head_args) if head_args else verb()
        full_body = Conjunction(tuple(all_body_parts)) if len(all_body_parts) > 1 else all_body_parts[0]
        return Implication(ProbabilisticFact(fresh_prob, consequent), full_body)

    def rule_body1(self, args):
        items = [a for a in args if a is not None]
        # items: [optional_prep, det, ng1]
        # det is a function: ng -> (d -> formula)
        # ng1 is a function: x -> formula (restriction)
        # Compose them to get a CPS NP, then extract rule components

        det = None
        ng1 = None
        for item in items:
            if callable(item) and not isinstance(item, (Symbol, Constant)):
                if det is None:
                    det = item
                else:
                    ng1 = item
            elif isinstance(item, str):
                pass  # prep

        if det is None and ng1 is None and len(items) == 1:
            return items[0]
        if det is None or ng1 is None:
            return ('_rule_body', ([], Constant(True)))

        # Get var_info from ng1 to extract the variable
        var_info = getattr(ng1, '_var_info', None)
        if var_info is not None:
            head_args = list(var_info) if isinstance(var_info, tuple) else [var_info]
        else:
            var_info = Symbol.fresh()
            head_args = [var_info]

        # Get the body formula from ng1
        body_formula = ng1(var_info if isinstance(var_info, tuple) else var_info)

        return ('_rule_body', (head_args, body_formula))

    def rule_body1_cond_prior(self, args):
        det = args[0]
        ng1 = args[1]
        s = args[2]
        return s

    def rule_body1_cond_posterior(self, args):
        s = args[0]
        det = args[1]
        ng1 = args[2]
        return s

    # ---- Sentences ----

    def s_np_vp(self, args):
        np = args[0]
        vp = args[1]
        return np(vp)

    def s_for(self, args):
        np = args[0]
        s = args[1]
        return np(lambda x: s)

    def s_be(self, args):
        np = args[0]
        return np(lambda x: Constant(True))

    # ---- Noun Phrases ----

    def expr_np(self, args):
        val = args[0]
        if callable(val) and not isinstance(val, (Symbol, Constant, FunctionApplication)):
            return val
        # Wrap raw values (Constant, Symbol) in CPS
        return lambda d: d(val)

    def np_quantified(self, args):
        det = args[0]
        ng1 = args[1]
        return det(ng1)

    def np_np2(self, args):
        np2 = args[0]
        np = args[1]
        return lambda d: np(lambda x: np2(x)(d))

    def np2(self, args):
        det = args[0]
        ng2 = args[1]
        # np2 takes an individual x (the "possessor") and returns
        # a CPS NP: (x -> (d -> formula))
        def np2_func(x):
            def ng_for_x(y):
                return ng2(x, y)
            return det(ng_for_x)
        return np2_func

    # ---- Determiners ----

    def det(self, args):
        return args[0]

    def det_some(self, args):
        return args[0]

    def det_every(self, args):
        """Return the universal-quantifier CPS determinant function.

        When the ng1 carries ``_agg_info`` (set by ``ng1_agg_npc``), a
        specialised path builds ``AggregationApplication`` in the head and
        extracts body predicates from the npc, bypassing the normal
        ``UniversalPredicate`` construction.
        """
        def every(ng):
            def apply_d(d):
                # Special handling for aggregation ng1
                # (e.g. "every Max of the Quantity where ?i item_count per ?i")
                agg_info = getattr(ng, '_agg_info', None)
                if agg_info is not None:
                    agg_func_const, npc_cps, per_vars = agg_info
                    # Apply the npc with a capturing continuation to extract
                    # the witness variable q.  We use a mutable container to
                    # capture x from inside the closure, while returning
                    # Constant(True) so no spurious conjuncts appear in the body.
                    captured = []

                    def capturing_cont(v, _cap=captured):
                        _cap.append(v)
                        return Constant(True)

                    npc_formula = npc_cps(capturing_cont)
                    if captured and isinstance(captured[0], Symbol):
                        q = captured[0]
                    elif isinstance(npc_formula, ExistentialPredicate):
                        q = npc_formula.head
                    else:
                        q = Symbol.fresh()

                    # Build AggregationApplication and hand it to the scope
                    # collector so it ends up in the rule head.
                    agg_expr = _AggApp(agg_func_const, (q,))
                    d(agg_expr)  # adds agg_expr to head_args

                    # Return the npc formula so _flatten_to_datalog extracts
                    # the body predicates (quantity(q), item_count(i, q), …).
                    return npc_formula

                var_info = getattr(ng, '_var_info', None)
                if var_info is not None and isinstance(var_info, tuple):
                    syms = var_info
                    body = ng(syms)
                    scope = _apply_to_vars(d, syms)
                    result = Implication(scope, body)
                    for sym in syms:
                        result = UniversalPredicate(sym, result)
                    return result
                elif var_info is not None:
                    x = var_info
                    body = ng(x)
                    scope = d(x)
                    return UniversalPredicate(x, Implication(scope, body))
                else:
                    x = Symbol.fresh()
                    body = ng(x)
                    scope = d(x)
                    return UniversalPredicate(x, Implication(scope, body))
            return apply_d
        return every

    def det_the(self, args):
        def the(ng):
            def apply_d(d):
                var_info = getattr(ng, '_var_info', None)
                if var_info is not None and isinstance(var_info, tuple):
                    syms = var_info
                    body = ng(syms)
                    scope = _apply_to_vars(d, syms)
                    result = Conjunction((body, scope))
                    for sym in syms:
                        result = ExistentialPredicate(sym, result)
                    return result
                elif var_info is not None:
                    x = var_info
                    body = ng(x)
                    scope = d(x)
                    return ExistentialPredicate(x, Conjunction((body, scope)))
                else:
                    x = Symbol.fresh()
                    body = ng(x)
                    scope = d(x)
                    return ExistentialPredicate(x, Conjunction((body, scope)))
            return apply_d
        return the

    def det1_some(self, args):
        def some(ng):
            def apply_d(d):
                var_info = getattr(ng, '_var_info', None)
                if var_info is not None and isinstance(var_info, tuple):
                    syms = var_info
                    body = ng(syms)
                    scope = _apply_to_vars(d, syms)
                    result = Conjunction((body, scope))
                    for sym in syms:
                        result = ExistentialPredicate(sym, result)
                    return result
                elif var_info is not None:
                    x = var_info
                    body = ng(x)
                    scope = d(x)
                    return ExistentialPredicate(x, Conjunction((body, scope)))
                else:
                    x = Symbol.fresh()
                    body = ng(x)
                    scope = d(x)
                    return ExistentialPredicate(x, Conjunction((body, scope)))
            return apply_d
        return some

    def det1_no(self, args):
        def no(ng):
            def apply_d(d):
                var_info = getattr(ng, '_var_info', None)
                if var_info is not None:
                    x = var_info
                else:
                    x = Symbol.fresh()
                body = ng(x)
                scope = d(x)
                return Negation(
                    ExistentialPredicate(x, Conjunction((body, scope)))
                )
            return apply_d
        return no

    # ---- Noun Groups ----

    def ng1_noun(self, args):
        items = [a for a in args if a is not None]
        noun1 = items[0]
        app = None
        rel = None
        for item in items[1:]:
            if isinstance(item, tuple) and len(item) == 2 and item[0] == '_rel':
                rel = item[1]
            elif isinstance(item, Symbol):
                app = item
            elif isinstance(item, tuple) and all(isinstance(s, Symbol) for s in item):
                app = item
            elif callable(item) and not isinstance(item, (Symbol, Constant)):
                # Could be a CPS function from app_label or something else
                if app is None:
                    app = item

        def ng(x):
            if isinstance(x, tuple):
                noun_app = noun1(*x)
            else:
                noun_app = noun1(x)
            parts = [noun_app]
            if rel is not None:
                rel_val = rel(x)
                if isinstance(rel_val, Conjunction):
                    parts.extend(rel_val.formulas)
                elif rel_val is not None:
                    parts.append(rel_val)
            if len(parts) == 1:
                return parts[0]
            return Conjunction(tuple(parts))

        if app is not None:
            ng._var_info = app
        return ng

    def ng1_agg_npc(self, args):
        """Handle ``noun1 OF npc [dims]`` aggregation noun groups.

        When ``noun1`` is an aggregation function name (count, sum, max, min,
        average) **and** an npc is present, the returned ng1 function carries
        an ``_agg_info = (agg_func_const, npc_cps, per_vars)`` attribute.
        ``det_every`` inspects this attribute to build an
        ``AggregationApplication`` head argument rather than a plain variable.
        """
        noun1 = args[0]
        app = None
        # Last argument before dims is the npc; dims (if present) is tagged
        dims = None
        npc = None
        for a in args[1:]:
            if a is None:
                continue
            if isinstance(a, tuple) and a[0] == '_dims':
                dims = a[1]
            elif callable(a) and not isinstance(a, (Symbol, Constant)):
                npc = a
            elif isinstance(a, (Symbol, Constant)):
                app = a

        per_vars = []    # groupby variables (from _per dims)
        agg_specs = []   # (agg_func_constant, npc_cps) pairs (from _agg dims)
        if dims is not None:
            for d in dims:
                if isinstance(d, tuple) and d[0] == '_per':
                    per_vars.append(d[1])
                elif isinstance(d, tuple) and d[0] == '_agg':
                    agg_specs.append((d[1], d[2]))

        # Map noun1 name to an aggregation function constant.
        # e.g. noun1 = Symbol('max') → Constant(max)
        noun_name = noun1.name.lower() if isinstance(noun1, Symbol) else None
        agg_func_from_noun = _AGG_FUNC_MAP.get(noun_name) if noun_name else None

        # If the noun is an aggregation function and we have an npc
        # (pattern: "every Max of the Quantity where ?i item_count per ?i"),
        # produce a tagged ng1 with _agg_info so det_every can build
        # AggregationApplication in the head.
        if agg_func_from_noun is not None and npc is not None:
            def ng_agg(x):
                # Fallback: used only if det_every does not intercept _agg_info.
                # Introduce a fresh aggregation variable and build body from npc.
                q = Symbol.fresh()
                body_formula = npc(lambda v: Constant(True))
                agg_expr = _AggApp(agg_func_from_noun, (q,))
                return agg_expr

            ng_agg._agg_info = (agg_func_from_noun, npc, list(per_vars))
            if app is not None:
                ng_agg._var_info = app
            return ng_agg

        if agg_specs:
            agg_func_const, agg_npc = agg_specs[0]

            def ng_agg(x):
                # x is the result variable (the aggregated dimension)
                # agg_npc gives the predicate for what is aggregated
                agg_var = Symbol.fresh()
                if callable(agg_npc) and not isinstance(
                    agg_npc, (Symbol, Constant)
                ):
                    agg_body = agg_npc(lambda v: v)
                    if isinstance(agg_body, (Symbol, Constant)):
                        agg_var = agg_body
                    # else leave agg_var fresh

                # Build AggregationApplication:
                # AggregationApplication(func, (agg_var, *per_vars))
                agg_args = (agg_var,) + tuple(per_vars)
                return _AggApp(agg_func_const, agg_args)

            if app is not None:
                ng_agg._var_info = app
            return ng_agg

        # No aggregation dims: fall back to plain noun1
        def ng(x):
            if isinstance(x, tuple):
                return noun1(*x)
            return noun1(x)

        if app is not None:
            ng._var_info = app
        return ng

    def ng2(self, args):
        noun2 = args[0]
        return noun2

    # ---- Relative Clauses ----

    def rel_vp(self, args):
        vp = args[0]
        return ('_rel', vp)

    def rel_vpn(self, args):
        np = args[0]
        verb = args[1]
        ops = args[2] if len(args) > 2 else None
        def rel(x):
            if isinstance(x, tuple):
                return np(lambda y: verb(y, *x))
            else:
                return np(lambda y: verb(y, x))
        return ('_rel', rel)

    def rel_np2(self, args):
        np2 = args[0]
        vp = args[1]
        def rel(x):
            return np2(x)(vp)
        return ('_rel', rel)

    def rel_ng2(self, args):
        """Handle ``whose NG2 VP`` possessive relative clauses.

        Semantics: given binary noun ``ng2`` and verb phrase ``vp``,
        produces ``∃y. ng2(x, y) ∧ vp(y)`` — the subject ``x`` possesses
        a ``y`` related by ``ng2`` that satisfies ``vp``.

        Example: ``every person whose writer plays``
        →  ``∀x. person(x) → ∃y. writer(x, y) ∧ plays(y)``
        """
        ng2 = args[0]
        vp = args[1]
        def rel(x):
            y = Symbol.fresh()
            ng2_atom = ng2(x, y)
            vp_atom = vp(y)
            return ExistentialPredicate(y, Conjunction((ng2_atom, vp_atom)))
        return ('_rel', rel)

    def rel_s(self, args):
        s = args[0]
        return ('_rel', lambda x: s)

    def rel_comp(self, args):
        comparison = args[0]
        op = args[1]
        def rel(x):
            return op(lambda y: Constant(comparison)(x, y))
        return ('_rel', rel)

    def rel_adj1(self, args):
        adj = args[0]
        return ('_rel', adj)

    def comparison(self, args):
        tokens = []
        for a in args:
            if hasattr(a, 'value'):
                tokens.append(a.value.lower())
            elif isinstance(a, str):
                tokens.append(a.lower())
        key = tuple(tokens)
        return COMPARISON_OPS.get(key, eq)

    # ---- Verb Phrases ----

    def vpdo_v1(self, args):
        verb1 = args[0]
        cp = args[1] if len(args) > 1 and args[1] is not None else None
        if cp is not None:
            return lambda x: verb1(x)
        return verb1

    def vpdo_vn(self, args):
        verb = args[0]
        ops = args[1]
        def vp(x):
            return _apply_ops(ops, verb, x)
        return vp

    def vpdo_prob_v1(self, args):
        # "probably verb1 [cp]" → ProbabilisticFact with fresh prob symbol
        verb1 = args[0]
        fresh_prob = Symbol.fresh()
        return lambda x: ProbabilisticFact(fresh_prob, verb1(x))

    def vpdo_prob_vn(self, args):
        # "probably verbn opn" → ProbabilisticFact with fresh prob symbol
        verb = args[0]
        ops = args[1]
        fresh_prob = Symbol.fresh()
        return lambda x: ProbabilisticFact(fresh_prob, _apply_ops(ops, verb, x))

    def vpdo_explicit_prob_v1(self, args):
        # "verb1 [cp] with probability number"
        verb1 = args[0]
        prob_value = args[-1]   # Constant(float)
        return lambda x: ProbabilisticFact(prob_value, verb1(x))

    def vpdo_explicit_prob_vn(self, args):
        # "verbn opn with probability number"
        verb = args[0]
        prob_value = args[-1]   # Constant(float)
        ops = args[1] if len(args) > 2 else None
        if ops is not None:
            return lambda x: ProbabilisticFact(
                prob_value, _apply_ops(ops, verb, x)
            )
        return lambda x: ProbabilisticFact(prob_value, verb(x))

    def vp_aux(self, args):
        aux = args[0]
        vp_inner = args[1]
        if aux == 'not':
            return lambda x: Negation(vp_inner(x))
        return vp_inner

    def vpbe_there(self, args):
        return lambda x: Constant(True)

    def vpbe_rel(self, args):
        rel = args[0]
        if isinstance(rel, tuple) and rel[0] == '_rel':
            return rel[1]
        return rel

    def vpbe_npc(self, args):
        npc = args[0]
        if callable(npc) and not isinstance(npc, (Symbol, Constant)):
            return lambda x: npc(lambda y: Constant(eq)(x, y))
        return lambda x: Constant(eq)(x, npc)

    def vphave_noun2(self, args):
        noun2 = args[0]
        op = args[1]
        def vp(x):
            return op(lambda y: noun2(x, y))
        return vp

    def vphave_np2(self, args):
        np2 = args[0]
        rel = None
        for a in args[1:]:
            if isinstance(a, tuple) and a[0] == '_rel':
                rel = a[1]
            elif a is not None:
                rel = a
        if rel is not None:
            return lambda x: np2(x)(lambda y: rel(y) if callable(rel) else rel)
        return lambda x: np2(x)(lambda y: Constant(True))

    def aux_id(self, args):
        return 'id'

    def aux_not(self, args):
        return 'not'

    # ---- NPC (noun phrase complement) ----

    def npc_term(self, args):
        return args[0]

    def npc_det(self, args):
        items = [a for a in args if a is not None]
        if len(items) >= 2:
            det_val = items[0]
            ng1 = items[1]
            if callable(det_val) and not isinstance(det_val, (Symbol, Constant)):
                return det_val(ng1)
            return self.det_the([])(ng1)
        elif len(items) == 1:
            ng1 = items[0]
            return self.det_the([])(ng1)
        return self.det1_some([])(lambda x: Constant(True))

    # ---- Operations / Prepositional Phrases ----

    def op_np(self, args):
        items = [a for a in args if a is not None]
        if len(items) == 2:
            return items[1]
        return items[0]

    def ops_base(self, args):
        return args[0]

    def ops_rec(self, args):
        prev = args[0]
        prep = args[1] if len(args) > 2 else None
        new_op = args[-1]
        if callable(prev) and callable(new_op):
            def combined(f):
                return lambda x: _conj_flat(
                    prev(f)(x) if callable(prev(f)) else prev(f),
                    new_op(f)(x) if callable(new_op(f)) else new_op(f)
                )
            return combined
        return new_op

    def pp_np(self, args):
        prep = args[0]
        np = args[1]
        return np

    def prep(self, args):
        return args[0].value if hasattr(args[0], 'value') else str(args[0])

    # ---- Booleans ----

    def bool_disjunction(self, args):
        if len(args) == 1:
            return args[0]
        formulas = []
        for a in args:
            if isinstance(a, tuple) and a[0] == '_rel':
                formulas.append(a[1])
            else:
                formulas.append(a)
        if all(callable(f) and not isinstance(f, (Symbol, Constant)) for f in formulas):
            return lambda x: Disjunction(tuple(f(x) for f in formulas))
        return Disjunction(tuple(formulas))

    def bool_conjunction(self, args):
        if len(args) == 1:
            return args[0]
        formulas = []
        for a in args:
            if isinstance(a, tuple) and a[0] == '_rel':
                formulas.append(a[1])
            else:
                formulas.append(a)
        if all(callable(f) and not isinstance(f, (Symbol, Constant)) for f in formulas):
            return lambda x: Conjunction(tuple(f(x) for f in formulas))
        return Conjunction(tuple(formulas))

    def bool_negation(self, args):
        inner = args[0]
        if isinstance(inner, tuple) and inner[0] == '_rel':
            return ('_rel', lambda x: Negation(inner[1](x)))
        if callable(inner) and not isinstance(inner, (Symbol, Constant)):
            return lambda x: Negation(inner(x))
        return Negation(inner)

    def bool_atom(self, args):
        if len(args) == 1:
            return args[0]
        return args

    # ---- Expressions / Arithmetic ----

    def expr_atom(self, args):
        if len(args) == 1:
            return args[0]
        return args

    def expr_sum(self, args):
        if len(args) == 1:
            return args[0]
        left = args[0]
        op_token = args[1]
        right = args[2]
        op = Constant(add) if op_token.value == '+' else Constant(sub)
        return op(left, right)

    def expr_mul(self, args):
        if len(args) == 1:
            return args[0]
        left = args[0]
        op_token = args[1]
        right = args[2]
        op = Constant(mul) if op_token.value == '*' else Constant(truediv)
        return op(left, right)

    def expr_pow(self, args):
        if len(args) == 1:
            return args[0]
        return Constant(pow)(args[0], args[1])

    def expr_atom_par(self, args):
        return args[0]

    def expr_atom_tuple(self, args):
        return tuple(args)

    def expr_atom_fun(self, args):
        func = args[0]
        func_args = args[1:]
        return FunctionApplication(func, tuple(func_args))

    def expr_atom_term(self, args):
        return args[0]

    # ---- Pass-through handlers for template-expanded rules ----

    def noun1(self, args):
        return args[0]

    def noun2(self, args):
        return args[0]

    def verb1(self, args):
        return args[0]

    def verbn(self, args):
        return args[0]

    def adj1(self, args):
        return args[0]

    def a_an_the(self, args):
        return self.det1_some([])

    def in_(self, args):
        return "in"

    def np_b(self, args):
        if len(args) == 1:
            return args[0]
        return args

    def s_b(self, args):
        if len(args) == 1:
            return args[0]
        return args

    def vp_b(self, args):
        if len(args) == 1:
            return args[0]
        return args

    def rel_b(self, args):
        if len(args) == 1:
            return args[0]
        return args

    # ---- Terminals / Atoms ----

    def intransitive(self, args):
        if isinstance(args[0], Symbol):
            return args[0]
        name = args[0].value
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        return Symbol(name.lower())

    def transitive(self, args):
        if isinstance(args[0], Symbol):
            return args[0]
        name = args[0].value
        if name.startswith("~"):
            name = name[1:]
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        return Symbol(name)

    def transitive_inv(self, args):
        token = args[0]
        name = token.value if hasattr(token, 'value') else token.name
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        if name.startswith('~'):
            name = name[1:]
        return _InverseVerbSymbol(Symbol(name))

    def transitive_multiple(self, args):
        if isinstance(args[0], Symbol):
            return args[0]
        name = args[0].value
        if name.startswith("~"):
            name = name[1:]
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        return Symbol(name)

    def transitive_multiple_inv(self, args):
        token = args[0]
        name = token.value if hasattr(token, 'value') else token.name
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        if name.startswith('~'):
            name = name[1:]
        return _InverseVerbSymbol(Symbol(name))

    def upper_identifier(self, args):
        name = args[0].value
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        return Symbol(name.lower())

    def identifier(self, args):
        name = args[0].value
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        return Symbol(name)

    def label_identifier(self, args):
        return Symbol(args[0].value)

    def label(self, args):
        if len(args) == 1:
            marker = args[0]
            if hasattr(marker, 'value') and marker.value == '_':
                return lambda d: d(Symbol.fresh())
            sym = marker
            return lambda d: d(sym)
        labels = list(args)
        if len(labels) == 1:
            sym = labels[0]
            return lambda d: d(sym)
        return lambda d: d(tuple(labels))

    def ANONYMOUS_LABEL(self, token):
        return lambda d: d(Symbol.fresh())

    def term(self, args):
        return args[0]

    def literal(self, args):
        return args[0]

    def string(self, args):
        return Constant(args[0].value)

    def number(self, args):
        token = args[0]
        val = token.value
        if '.' in val:
            return Constant(float(val))
        return Constant(int(val))

    def external_literal(self, args):
        return Symbol(args[0].value)

    def command(self, args):
        name = args[0]
        terms = [a for a in args[1:] if a is not None]
        return FunctionApplication(name, tuple(terms))

    def query(self, args):
        # "obtain ops" — convert the CPS noun phrase into a Query expression.
        # ops is a CPS callable: (d -> formula) representing the NP.
        # Apply it to (lambda x: x) to get a quantified formula, then
        # extract the restriction body and wrap in Query(head_var, body).
        ops = args[0]
        if not callable(ops) or isinstance(ops, (Symbol, Constant)):
            # Fallback: wrap raw value
            x = Symbol.fresh()
            return ('_query', Query(x, ops if not isinstance(ops, Symbol) else ops(x)))

        formula = ops(lambda x: x)
        return ('_query', _cps_formula_to_query(formula))

    # ---- Dimension / Aggregation ----

    def app_dimension(self, args):
        return lambda x: x

    def app_label(self, args):
        label = args[0]
        if callable(label) and not isinstance(label, (Symbol, Constant)):
            # CPS label - extract the symbol from it
            result = label(lambda x: x)
            if isinstance(result, tuple):
                return result
            return result
        return label

    def dims_base(self, args):
        return ('_dims', [args[0]])

    def dims_rec(self, args):
        dim = args[0]
        rest = args[1]
        if isinstance(rest, tuple) and rest[0] == '_dims':
            return ('_dims', [dim] + rest[1])
        return ('_dims', [dim, rest])

    def dim_ng2(self, args):
        # "per region" → groupby variable from the noun
        noun2 = args[0]
        groupby_var = Symbol.fresh()
        return ('_per', groupby_var)

    def dim_npc(self, args):
        # "per the region" / "per ?i" → groupby variable from the NPC
        npc = args[0]
        if isinstance(npc, (Symbol, Constant)):
            return ('_per', npc)
        if callable(npc) and not isinstance(npc, (Symbol, Constant)):
            # CPS NP / label: applying with identity extracts the underlying symbol
            result = npc(lambda x: x)
            if isinstance(result, (Symbol, Constant)):
                return ('_per', result)
        # fallback: fresh variable
        groupby_var = Symbol.fresh()
        return ('_per', groupby_var)

    def dim_agg(self, args):
        # "count of the activations" → ('_agg', Constant(len), npc_cps)
        agg_func_const = args[0]   # already a Constant from agg_func()
        npc = args[1]              # CPS NP or Symbol
        return ('_agg', agg_func_const, npc)

    def agg_func(self, args):
        token = args[0]
        name = token.value if hasattr(token, 'value') else str(token)
        return _AGG_FUNC_MAP.get(name.lower(), Constant(len))

    # ---- Probabilistic ----

    def PROBABLY(self, token):
        return "probably"

    def _default(self, data, children, meta):
        if len(children) == 1:
            return children[0]
        return children if children else None


def _extract_datalog_body(cps_np, head_args):
    """Extract body predicates from a CPS NP for use in Datalog rules.

    Calls ``cps_np`` with a ``collect_scope`` continuation that appends each
    bound variable to ``head_args`` (side-effect) and returns
    ``Constant(True)``.  The resulting formula is then passed to
    ``_flatten_to_datalog`` to harvest body predicates.

    Parameters
    ----------
    cps_np:
        A CPS noun-phrase callable ``(d -> formula)``.
    head_args:
        Mutable list; bound variables are appended here as a side-effect.

    Returns
    -------
    list
        Flat list of body predicate expressions (``FunctionApplication``
        and similar) with no ``Constant(True)`` entries.
    """
    body_parts = []

    def collect_scope(x):
        if isinstance(x, tuple):
            head_args.extend(x)
        else:
            head_args.append(x)
        return Constant(True)

    result = cps_np(collect_scope)

    # Walk the result to extract body predicates from quantifiers
    extracted = _flatten_to_datalog(result)
    body_parts.extend(extracted)
    return body_parts


def _flatten_to_datalog(expr):
    """Flatten quantified expressions into a flat list of Datalog body atoms.

    Strips ``ExistentialPredicate`` and ``UniversalPredicate`` wrappers,
    unpacks ``Conjunction`` formulas recursively, and extracts both sides of
    an ``Implication`` (antecedent = restriction body, consequent = scope).
    ``Constant(True)`` entries are filtered out by the caller.

    Returns
    -------
    list
        Flat list of atomic expressions (``FunctionApplication`` instances
        and similar) ready to be used as Datalog body literals.
    """
    if isinstance(expr, ExistentialPredicate):
        return _flatten_to_datalog(expr.body)
    elif isinstance(expr, UniversalPredicate):
        return _flatten_to_datalog(expr.body)
    elif isinstance(expr, Conjunction):
        parts = []
        for f in expr.formulas:
            parts.extend(_flatten_to_datalog(f))
        return parts
    elif isinstance(expr, Implication):
        # Universal: scope ← restriction
        parts = _flatten_to_datalog(expr.antecedent)
        scope_parts = _flatten_to_datalog(expr.consequent)
        # Filter out True constants
        parts = [p for p in parts if p != Constant(True)]
        scope_parts = [p for p in scope_parts if p != Constant(True)]
        return parts + scope_parts
    elif expr == Constant(True):
        return []
    else:
        return [expr]


def _apply_to_vars(d, syms):
    if isinstance(d, Symbol):
        return d(*syms)
    if callable(d):
        try:
            return d(*syms)
        except TypeError:
            return d(syms)
    return d


def _conj_flat(a, b):
    """Flatten nested conjunctions into a single Conjunction."""
    formulas = []
    if isinstance(a, Conjunction):
        formulas.extend(a.formulas)
    else:
        formulas.append(a)
    if isinstance(b, Conjunction):
        formulas.extend(b.formulas)
    else:
        formulas.append(b)
    return Conjunction(tuple(formulas))


def _apply_ops(ops, verb, subject):
    """Apply operation arguments to a transitive verb."""
    if callable(ops) and not isinstance(ops, (Symbol, Constant)):
        return ops(lambda obj: verb(subject, obj))
    elif isinstance(ops, tuple):
        return verb(subject, *ops)
    else:
        return verb(subject, ops)


def _cps_formula_to_query(formula):
    """Convert a CPS-produced quantified formula into a Query expression.

    The CPS NP for an "obtain" clause produces a universally or existentially
    quantified formula.  Extract the bound variable and restriction body so we
    can build ``Query(head_var, body)``.

    The resulting ``Query`` has its body passed through ``LogicSimplifier`` so
    that trivial ``Constant(True)`` conjuncts are removed before execution.

    Parameters
    ----------
    formula : Expression
        Result of applying the CPS NP to ``lambda x: x``.

    Returns
    -------
    Query
        A NeuroLang Query expression ready to be walked into an engine.
    """
    from .squall import LogicSimplifier

    if isinstance(formula, UniversalPredicate):
        x = formula.head
        body = formula.body
        if isinstance(body, Implication):
            # Universal: head ← restriction  →  Query(x, restriction)
            raw = Query(x, body.antecedent)
        else:
            raw = Query(x, body)
    elif isinstance(formula, ExistentialPredicate):
        x = formula.head
        raw = Query(x, formula.body)
    else:
        # Fallback: the formula IS the body; introduce a fresh variable
        x = Symbol.fresh()
        raw = Query(x, formula)

    return LogicSimplifier().walk(raw)


def parser(code, locals=None, globals=None):
    """
    Parse SQUALL controlled English and return a Union of logical expressions.

    Parameters
    ----------
    code : str
        SQUALL code to parse.

    Returns
    -------
    Union or Expression
        Parsed logical expression(s).
    """
    tree = COMPILED_GRAMMAR.parse(code.strip())
    result = SquallTransformer().transform(tree)
    if isinstance(result, Union):
        return result
    return result
