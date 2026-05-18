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

Implemented constructs (previously stubs)
-----------------------------------------
- **Conditioned rules** — ``rule_body1_cond_prior`` / ``rule_body1_cond_posterior`` /
  ``rule_body2_cond``: all three handlers now produce
  ``Condition(conditioned, conditioning)`` in the rule body, which
  ``TranslateProbabilisticQueryMixin.rewrite_conditional_query`` rewrites into
  the three-rule conditional probability form.
- **Inverse transitive prefix ``~``** — ``transitive_inv`` and
  ``transitive_multiple_inv`` return ``_InverseVerbSymbol``, which emits
  ``InvertedFunctionApplication`` nodes resolved by
  ``ResolveInvertedFunctionApplicationMixin`` to reversed argument order.

References
----------
S. Ferré, "SQUALL: The expressiveness of SPARQL 1.1 made available
as a controlled natural language", Data & Knowledge Engineering, 2014.
"""
import os

import numpy as np
from lark import Lark, Transformer
from lark.exceptions import (
    LarkError, UnexpectedCharacters, UnexpectedToken,
    VisitError,
)
from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

from ...datalog import Conjunction, Fact, Implication, Negation, Union
from ...datalog.aggregation import AggregationApplication
from ...datalog.expressions import AggregationApplication as _AggApp
from ...exceptions import (
    NeuroLangException,
    SquallSemanticError,
    UnexpectedCharactersError,
    UnexpectedTokenError,
)
from ...expressions import (
    Constant,
    FunctionApplication,
    Query,
    Symbol,
)
from ...logic import Disjunction, ExistentialPredicate, UniversalPredicate
from ...logic.expression_processing import extract_logic_free_variables
from ...probabilistic.expressions import (
    Condition, ProbabilisticFact, ProbabilisticQuery, PROB
)
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

    def __init__(self, rules, queries, query_names=None):
        self.rules = list(rules)
        self.queries = list(queries)
        # Mapping from query index (0-based position in self.queries) to the
        # user-supplied name for 'obtain … as Name' clauses.  Unnamed queries
        # have no entry here.
        self.query_names = dict(query_names) if query_names is not None else {}


class _AnonymousVar:
    """Sentinel for a ``_`` wildcard in a tuple label.

    When encountered in a tuple label ``(?i; ?j; _)``, this placeholder
    causes the corresponding argument position to be bound to a fresh
    existential variable in the *body* of the rule but **not** to appear
    in the *head*.  The existential variable is materialised on demand via
    :meth:`as_symbol`.
    """

    def __init__(self):
        # Allocate one fresh symbol per anonymous slot.
        self._sym = Symbol.fresh()

    def as_symbol(self):
        return self._sym


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


def _resolve_var_info(var_info):
    """Return (body_args, head_args) from a _var_info value.

    ``body_args`` is the tuple of all symbols (including fresh symbols for
    anonymous ``_`` wildcards) used to build the rule *body* predicate.
    ``head_args`` is the tuple of *named* symbols that appear in the rule
    *head*; anonymous wildcards are excluded.

    Parameters
    ----------
    var_info : Symbol, tuple, or None
        The ``_var_info`` attribute attached to an ng1 function.

    Returns
    -------
    body_args : tuple or Symbol or None
        The argument tuple / symbol for calling ``ng(body_args)``.
    head_args : list
        The list of head variables (no anonymous vars).
    """
    if var_info is None:
        return None, []
    if isinstance(var_info, Symbol):
        return var_info, [var_info]
    if isinstance(var_info, tuple):
        body_args = tuple(
            item.as_symbol() if isinstance(item, _AnonymousVar) else item
            for item in var_info
        )
        head_args = [
            item for item in var_info if not isinstance(item, _AnonymousVar)
        ]
        return body_args, head_args
    # Fallback (callable _var_info — not expected in normal use)
    return var_info, [var_info]


class SquallTransformer(Transformer):
    """
    Transforms a SQUALL parse tree into NeuroLang logical expressions.

    Noun phrases and quantified expressions use CPS: they are represented
    as functions that take a continuation (predicate over an individual)
    and produce a logical formula with the quantifier in scope.
    """

    def __init__(self, source_lines=None):
        super().__init__()
        self._symbol_scope = {}
        self._source_lines = source_lines or []
        self._current_line = None
        self._current_column = None

    def _clear_scope(self):
        self._symbol_scope.clear()

    def _capture_pos(self, token):
        if hasattr(token, 'line') and hasattr(token, 'column'):
            self._current_line = token.line
            self._current_column = token.column

    def _make_error(self, message):
        source_line = None
        if (
            self._current_line is not None
            and 1 <= self._current_line <= len(self._source_lines)
        ):
            source_line = self._source_lines[self._current_line - 1]
        return SquallSemanticError(
            message,
            line=self._current_line,
            column=self._current_column,
            source_line=source_line,
        )

    def start(self, args):
        return args[0]

    def squall(self, args):
        rules = []
        queries = []
        query_names = {}  # index → name for 'obtain … as Name' clauses
        for a in args:
            if a is None:
                continue
            if isinstance(a, tuple) and len(a) == 2 and a[0] == '_query':
                queries.append(a[1])
            elif isinstance(a, tuple) and len(a) == 2 and a[0] == '_query_as':
                impl, q, name_str = a[1]
                rules.append(impl)
                query_names[len(queries)] = name_str
                queries.append(q)
            else:
                rules.append(a)

        if queries:
            return SquallProgram(rules=rules, queries=queries, query_names=query_names)

        # Backward compat: no queries → return Union or single rule
        if len(rules) == 1:
            return rules[0]
        return Union(tuple(rules))

    def sentence(self, args):
        return args[0]

    # ---- Rules ----

    def rule_op(self, args):
        self._clear_scope()
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
            raise self._make_error(
                "Cannot construct rule body — expected a noun phrase after "
                "the verb.  For compound-quantifier rules, use "
                "'for every X and for every Y where …'."
            )

    def rule_op_prob(self, args):
        self._clear_scope()
        # "define as verb with probability np rule_body1"
        # args: [verb1, np_prob_cps, body_result]
        items = [a for a in args if a is not None]
        verb = items[0]
        np_prob = items[1]
        body_result = items[2]
        if isinstance(body_result, tuple) and body_result[0] == '_rule_body':
            head_args, body_formula = body_result[1]
        else:
            raise self._make_error(
                "Cannot construct rule body — expected a noun phrase after "
                "the verb."
            )
        head = verb(*head_args) if head_args else verb()
        # Extract probability value from CPS NP
        if callable(np_prob) and not isinstance(np_prob, (Symbol, Constant)):
            prob_val = np_prob(lambda x: x)
        else:
            prob_val = np_prob
        return Implication(ProbabilisticFact(prob_val, head), body_formula)

    def rule_op_marg(self, args):
        self._clear_scope()
        """Build a MARG query from ``define as verb with probability rule_body1_cond``.

        Emits:
            Implication(
                verb(head_vars..., ProbabilisticQuery(PROB, (head_vars...))),
                Condition(conditioned, conditioning)
            )

        TranslateProbabilisticQueryMixin.rewrite_conditional_query then rewrites
        this into the standard three-rule conditional probability form, adding
        the probability as the last column of the output relation.
        """
        items = [a for a in args if a is not None]
        verb = items[0]
        body_result = items[1]

        if isinstance(body_result, tuple) and body_result[0] == '_rule_body':
            head_args, body_formula = body_result[1]
        else:
            raise self._make_error(
                "Cannot construct MARG rule body — expected a conditioned "
                "noun phrase after 'with probability'."
            )

        prob_query_arg = ProbabilisticQuery(PROB, tuple(head_args))
        head = verb(*(list(head_args) + [prob_query_arg]))
        return Implication(head, body_formula)

    def rule_op_prob_agg(self, args):
        self._clear_scope()
        """Build a probabilistic aggregation rule.

        Grammar: ``define as verb1 WITH A PROBABILITY OF np``

        The ``np`` must be an aggregation noun phrase (``ng1_agg_npc``), e.g.
        ``the Agg_max_proximity of the Focus_reported (…) per ?i1 …
            such that Voxel(…) and ?d equal to euclidean(…) and ?d lower than 1``.

        Produces::

            Implication(
                ProbabilisticFact(
                    AggApp(agg_max_proximity, (d,)),
                    voxel_reported(i1, j1, k1, s)
                ),
                Conjunction([focus_reported(i2,j2,k2,s), voxel(i1,j1,k1), ...])
            )

        The ``per`` variables from the aggregation NP become the head predicate
        arguments.  The aggregation variable(s) inside ``AggApp`` are the
        non-``per`` free variables of the body.
        """
        items = [a for a in args if a is not None]
        verb = items[0]      # Symbol — head predicate name
        np_cps = items[1]    # CPS NP carrying ng1._agg_info

        # Materialise the NP via a capturing continuation that collects the
        # AggApp expression produced by det_every / det_the for agg_info NPs.
        captured = []

        def capturing_cont(v, *extra, _cap=captured):
            if extra:
                _cap.append((v,) + extra)
            else:
                _cap.append(v)
            return Constant(True)

        result = np_cps(capturing_cont)

        # Strip existential wrappers to get the bare body formula.
        bare_body = result
        while isinstance(bare_body, ExistentialPredicate):
            bare_body = bare_body.body
        if (
            isinstance(bare_body, Conjunction)
            and len(bare_body.formulas) == 2
            and bare_body.formulas[1] == Constant(True)
        ):
            bare_body = bare_body.formulas[0]

        if not captured:
            raise ValueError(
                "rule_op_prob_agg: the noun phrase did not produce an "
                "aggregation expression.  Make sure it uses "
                "'the <Agg_func> of the <Relation> per ?var …' form."
            )
        agg_expr = captured[0]
        if isinstance(agg_expr, tuple):
            agg_expr = agg_expr[0]

        # The per_vars (groupby variables) become the head predicate arguments.
        # Prefer the _per_vars tag set by det_the/_agg_info branch (exact);
        # fall back to: free vars in body that are NOT the aggregated variable(s).
        if hasattr(agg_expr, '_per_vars') and agg_expr._per_vars:
            per_vars = agg_expr._per_vars
        else:
            agg_vars = (
                set(agg_expr.args)
                if hasattr(agg_expr, 'args')
                else set()
            )
            body_free = extract_logic_free_variables(bare_body)
            per_vars = sorted(
                (v for v in body_free if v not in agg_vars),
                key=lambda s: s.name,
            )

        head = verb(*per_vars) if per_vars else verb()
        return Implication(ProbabilisticFact(agg_expr, head), bare_body)

    def rule_opnn(self, args):
        self._clear_scope()
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
        self._clear_scope()
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
            raise self._make_error(
                "Cannot construct n-ary rule body — expected a noun phrase "
                "after the verb."
            )

        head_args = list(body_args)
        all_body_parts = [body_formula]
        if ops is not None and callable(ops) and not isinstance(ops, (Symbol, Constant)):
            _extract_datalog_body(ops, head_args)

        fresh_prob = Symbol.fresh()
        consequent = verb(*head_args) if head_args else verb()
        full_body = Conjunction(tuple(all_body_parts)) if len(all_body_parts) > 1 else all_body_parts[0]
        return Implication(ProbabilisticFact(fresh_prob, consequent), full_body)

    def rule_opnn_compound(self, args):
        """Build an n-ary Datalog rule from compound quantifiers.

        Grammar: `define as verbn rule_body2`
        """
        self._clear_scope()
        items = [a for a in args if a is not None]
        verb = items[0]
        body_result = items[1] if len(items) > 1 else None

        if isinstance(body_result, tuple) and body_result[0] == '_rule_body2':
            head_vars, body_formula = body_result[1]
        else:
            raise self._make_error(
                "Cannot construct compound-quantifier rule body — expected "
                "'for every X and for every Y where ...'."
            )

        consequent = verb(*head_vars) if head_vars else verb()
        return Implication(consequent, body_formula)

    def rule_opnn_prob(self, args):
        """`define as verbn with probability np rule_body2`"""
        self._clear_scope()
        items = [a for a in args if a is not None]
        verb = items[0]
        np_prob = items[1]
        body_result = items[2] if len(items) > 2 else None

        if isinstance(body_result, tuple) and body_result[0] == '_rule_body2':
            head_vars, body_formula = body_result[1]
        else:
            raise self._make_error(
                "Cannot construct compound-quantifier probabilistic rule "
                "body — expected 'for every X and for every Y where ...'."
            )

        if callable(np_prob) and not isinstance(np_prob, (Symbol, Constant)):
            prob_val = np_prob(lambda x: x)
        else:
            prob_val = np_prob

        head = verb(*head_vars) if head_vars else verb()
        return Implication(ProbabilisticFact(prob_val, head), body_formula)

    def rule_opnn_marg(self, args):
        """`define as verbn with inferred probability rule_body2`"""
        self._clear_scope()
        items = [a for a in args if a is not None]
        verb = items[0]
        body_result = items[1] if len(items) > 1 else None

        if isinstance(body_result, tuple) and body_result[0] == '_rule_body2':
            head_vars, body_formula = body_result[1]
        else:
            raise self._make_error(
                "Cannot construct compound-quantifier MARG rule body — "
                "expected 'for every X and for every Y where ...'."
            )

        prob_query_arg = ProbabilisticQuery(PROB, tuple(head_vars))
        head = verb(*(list(head_vars) + [prob_query_arg]))
        return Implication(head, body_formula)

    def rule_opnn_per_compound(self, args):
        """`define as probably verbn rule_body2`"""
        self._clear_scope()
        items = [a for a in args if a is not None and not (isinstance(a, str) and a.lower() in ('probably', 'conditioned'))]
        verb = items[0]
        body_result = items[1] if len(items) > 1 else None

        if isinstance(body_result, tuple) and body_result[0] == '_rule_body2':
            head_vars, body_formula = body_result[1]
        else:
            raise self._make_error(
                "Cannot construct compound-quantifier 'probably' rule body "
                "— expected 'for every X and for every Y where ...'."
            )

        fresh_prob = Symbol.fresh()
        consequent = verb(*head_vars) if head_vars else verb()
        return Implication(ProbabilisticFact(fresh_prob, consequent), body_formula)

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
            raise self._make_error(
                "Cannot identify determiner or noun group in rule body.  "
                "Expected a noun phrase such as 'every Voxel that fires' "
                "or 'a Study'."
            )
        if det is None or ng1 is None:
            raise self._make_error(
                "Incomplete rule body — missing determiner (every/a/the) or "
                "noun group.  Expected a full noun phrase after the verb."
            )

        # Handle aggregation ng1 (e.g. "every Create_overlay of the Prob_map")
        agg_info = getattr(ng1, '_agg_info', None)
        if agg_info is not None:
            agg_func_const, npc_cps, per_vars, extra_rel = agg_info
            captured = []

            def capturing_cont(v, *extra, _cap=captured):
                # multi-arg call when _apply_to_vars expands a tuple label
                if extra:
                    _cap.append((v,) + extra)
                else:
                    _cap.append(v)
                return Constant(True)

            npc_formula = npc_cps(capturing_cont)

            # Strip nested ExistentialPredicates so the body is safe-range.
            bare_body = npc_formula
            while isinstance(bare_body, ExistentialPredicate):
                bare_body = bare_body.body
            if (
                isinstance(bare_body, Conjunction)
                and len(bare_body.formulas) == 2
                and bare_body.formulas[1] == Constant(True)
            ):
                bare_body = bare_body.formulas[0]

            # Determine aggregation arguments from captured variables.
            # tuple-labeled npc (e.g. "of the Foo (?i;?j;?k;?p)") captures
            # all column variables as a tuple in the first entry.
            if captured and isinstance(captured[0], tuple):
                agg_args = tuple(captured[0])
            elif captured and isinstance(captured[0], Symbol):
                agg_args = tuple(captured)
            else:
                agg_args = (Symbol.fresh(),)

            agg_expr = _AggApp(agg_func_const, agg_args)
            head_args = [agg_expr]
            # Conjoin any trailing relative clause (e.g. "such that Voxel(…) and …")
            if extra_rel is not None:
                extra_formula = extra_rel(None)
                bare_body = Conjunction((bare_body, extra_formula))
            return ('_rule_body', (head_args, bare_body))

        var_info = getattr(ng1, '_var_info', None)
        body_args, head_args = _resolve_var_info(var_info)
        if body_args is None:
            body_args = Symbol.fresh()
            head_args = [body_args]

        noun_name = getattr(ng1, '_noun_name', None)
        if noun_name:
            self._symbol_scope[noun_name] = body_args

        body_formula = ng1(body_args)

        return ('_rule_body', (head_args, body_formula))

    def quant_clause_ng1(self, args):
        """Handle `for every Region [?r]` in a compound quantifier list.

        Returns `('_quant_clause', (var, type_predicate))`.
        """
        items = [a for a in args if a is not None]
        ng1 = items[0]
        app = items[1] if len(items) > 1 else None

        var_info = app if app is not None else getattr(ng1, '_var_info', None)
        if var_info is not None:
            body_args, head_args = _resolve_var_info(var_info)
            if body_args is None:
                body_args = Symbol.fresh()
                head_args = [body_args]
        else:
            body_args = Symbol.fresh()
            head_args = [body_args]

        # Register noun name in scope for anaphora
        noun_name = getattr(ng1, '_noun_name', None)
        if noun_name:
            self._symbol_scope[noun_name] = body_args

        type_predicate = ng1(body_args)
        return ('_quant_clause', (body_args, type_predicate))

    def quant_list_single(self, args):
        return [args[0]]

    def quant_list_rec(self, args):
        prev_list = args[0]
        new_clause = args[1]
        return prev_list + [new_clause]

    def rule_body2_where(self, args):
        """Handle `for every X and for every Y where ...`.

        Returns `('_rule_body2', (head_vars, body_formula))`.
        """
        quant_list = args[0]
        where_sentence = args[1]

        head_vars = []
        type_preds = []
        for clause in quant_list:
            _, (var, type_pred) = clause
            head_vars.append(var)
            type_preds.append(type_pred)

        body_parts = type_preds + [where_sentence]
        body_formula = Conjunction(tuple(body_parts))
        return ('_rule_body2', (head_vars, body_formula))

    def rule_body1_cond_prior(self, args):
        # Grammar: det ng1 _CONDITIONED _TO s
        # args = [det, ng1, s]
        det, ng1, s = args[0], args[1], args[2]
        var_info = getattr(ng1, '_var_info', None)
        body_args, head_args = _resolve_var_info(var_info)
        if body_args is None:
            body_args = Symbol.fresh()
            head_args = [body_args]
        conditioned_body = ng1(body_args)
        return ('_rule_body', (head_args, Condition(conditioned_body, s)))

    def rule_body1_cond_posterior(self, args):
        # Grammar: s _CONDITIONED _TO det ng1
        # args = [s, det, ng1]
        s, det, ng1 = args[0], args[1], args[2]
        var_info = getattr(ng1, '_var_info', None)
        body_args, head_args = _resolve_var_info(var_info)
        if body_args is None:
            body_args = Symbol.fresh()
            head_args = [body_args]
        conditioned_body = ng1(body_args)
        return ('_rule_body', (head_args, Condition(s, conditioned_body)))

    def rule_body2_cond(self, args):
        # Grammar: det ng1_left _CONDITIONED _TO det ng1_right
        # args = [det1, ng1_left, det2, ng1_right]
        _, ng1_left, _, ng1_right = args
        var_info_left = getattr(ng1_left, '_var_info', None)
        body_args_left, head_args = _resolve_var_info(var_info_left)
        if body_args_left is None:
            body_args_left = Symbol.fresh()
            head_args = [body_args_left]
        conditioned_body = ng1_left(body_args_left)
        # Use the right ng1's own _var_info — do NOT pass the left var_info.
        # The right side is an independent noun phrase (e.g. "every Term_association ?t
        # that is 'auditory'") with its own bound variable(s).
        var_info_right = getattr(ng1_right, '_var_info', None)
        body_args_right, _ = _resolve_var_info(var_info_right)
        if body_args_right is None:
            body_args_right = Symbol.fresh()
        conditioning_body = ng1_right(body_args_right)
        return ('_rule_body', (head_args, Condition(conditioned_body, conditioning_body)))

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
        if isinstance(val, (Symbol, Constant, FunctionApplication, tuple)):
            return lambda d: d(val)
        raise self._make_error(
            f"Cannot interpret '{type(val).__name__}' as a noun phrase."
        )

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
                    agg_func_const, npc_cps, per_vars, extra_rel = agg_info

                    # Build the npc body formula to discover free variables.
                    captured = []

                    def capturing_cont(v, *extra, _cap=captured):
                        # Called with (v,) for scalar npc or (v, *rest) when
                        # _apply_to_vars expands a tuple label: d(*syms).
                        if extra:
                            # multi-arg call from a tuple-labeled npc
                            _cap.append((v,) + extra)
                        else:
                            _cap.append(v)
                        return Constant(True)

                    npc_formula = npc_cps(capturing_cont)

                    # Flatten captured entries: a tuple entry means a tuple-labeled
                    # npc was used (e.g. "the Foo (?i;?j;?k;?p)").
                    if captured and isinstance(captured[0], tuple):
                        captured_flat = list(captured[0])
                    else:
                        captured_flat = [c for c in captured if isinstance(c, Symbol)]

                    # Strip nested ExistentialPredicates to recover the bare body
                    # predicate, which avoids "not safe range" errors when all
                    # variables are existentially bound inside the npc formula.
                    bare_body = npc_formula
                    while isinstance(bare_body, ExistentialPredicate):
                        bare_body = bare_body.body
                    # Collapse Conjunction(pred, True) → pred
                    if (
                        isinstance(bare_body, Conjunction)
                        and len(bare_body.formulas) == 2
                        and bare_body.formulas[1] == Constant(True)
                    ):
                        bare_body = bare_body.formulas[0]

                    if per_vars:
                        # Explicit groupby: use the captured witness variable
                        # (same as the original single-var path).
                        if captured_flat:
                            agg_args = (captured_flat[0],)
                        elif isinstance(npc_formula, ExistentialPredicate):
                            agg_args = (npc_formula.head,)
                        else:
                            agg_args = (Symbol.fresh(),)
                    else:
                        # No explicit groupby: use all captured vars (covers
                        # tuple-labeled npcs such as "the Foo (?i;?j;?k;?p)").
                        if captured_flat:
                            agg_args = tuple(captured_flat)
                        else:
                            free_vars = extract_logic_free_variables(bare_body)
                            if free_vars:
                                agg_args = tuple(
                                    sorted(free_vars, key=lambda s: s.name)
                                )
                            elif isinstance(npc_formula, ExistentialPredicate):
                                agg_args = (npc_formula.head,)
                            else:
                                agg_args = (Symbol.fresh(),)

                    agg_expr = _AggApp(agg_func_const, agg_args)
                    d(agg_expr)  # adds agg_expr to head_args

                    # Conjoin any trailing relative clause (e.g. "such that Voxel(…)")
                    if extra_rel is not None:
                        extra_formula = extra_rel(None)
                        bare_body = Conjunction((bare_body, extra_formula))

                    # Return the bare body predicate (existentials stripped) so
                    # that _flatten_to_datalog can extract a safe-range body.
                    return bare_body

                var_info = getattr(ng, '_var_info', None)
                if var_info is not None and isinstance(var_info, tuple):
                    body_syms, head_syms = _resolve_var_info(var_info)
                    body = ng(body_syms)
                    scope = _apply_to_vars(d, head_syms)
                    result = Implication(scope, body)
                    for sym in head_syms:
                        result = UniversalPredicate(sym, result)
                    return result
                elif var_info is not None:
                    x = var_info
                    noun_name = getattr(ng, '_noun_name', None)
                    if noun_name:
                        self._symbol_scope[noun_name] = x
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
                noun_name = getattr(ng, '_noun_name', None)
                if noun_name and noun_name in self._symbol_scope:
                    x = self._symbol_scope[noun_name]
                    return d(x)

                # Special handling for aggregation ng1 — mirrors det_every.
                agg_info = getattr(ng, '_agg_info', None)
                if agg_info is not None:
                    agg_func_const, npc_cps, per_vars, extra_rel = agg_info

                    captured = []

                    def capturing_cont(v, *extra, _cap=captured):
                        if extra:
                            _cap.append((v,) + extra)
                        else:
                            _cap.append(v)
                        return Constant(True)

                    npc_formula = npc_cps(capturing_cont)

                    if captured and isinstance(captured[0], tuple):
                        captured_flat = list(captured[0])
                    else:
                        captured_flat = [c for c in captured if isinstance(c, Symbol)]

                    bare_body = npc_formula
                    while isinstance(bare_body, ExistentialPredicate):
                        bare_body = bare_body.body
                    if (
                        isinstance(bare_body, Conjunction)
                        and len(bare_body.formulas) == 2
                        and bare_body.formulas[1] == Constant(True)
                    ):
                        bare_body = bare_body.formulas[0]

                    if per_vars:
                        agg_args = (captured_flat[0],) if captured_flat else (Symbol.fresh(),)
                    else:
                        if captured_flat:
                            agg_args = tuple(captured_flat)
                        else:
                            free_vars = extract_logic_free_variables(bare_body)
                            if free_vars:
                                agg_args = tuple(sorted(free_vars, key=lambda s: s.name))
                            else:
                                agg_args = (Symbol.fresh(),)

                    agg_expr = _AggApp(agg_func_const, agg_args)
                    agg_expr._per_vars = list(per_vars)

                    if extra_rel is not None:
                        extra_formula = extra_rel(None)
                        extended_body = Conjunction((bare_body, extra_formula))
                        per_set = set(per_vars)
                        npc_free = extract_logic_free_variables(bare_body)
                        extra_free = extract_logic_free_variables(extended_body)
                        new_agg_vars = sorted(
                            (
                                v for v in extra_free
                                if v not in per_set and v not in npc_free
                            ),
                            key=lambda sv: sv.name,
                        )
                        if new_agg_vars:
                            agg_expr = _AggApp(agg_func_const, tuple(new_agg_vars))
                            agg_expr._per_vars = list(per_vars)
                        bare_body = extended_body

                    d(agg_expr)
                    return bare_body

                if noun_name and self._symbol_scope:
                    raise self._make_error(
                        f"Cannot resolve 'the {noun_name}' — "
                        f"'{noun_name}' was not introduced by any "
                        f"preceding 'for every {noun_name}' clause.  "
                        f"Add 'for every {noun_name}' before 'where', "
                        f"or use 'a {noun_name}' instead."
                    )

                var_info = getattr(ng, '_var_info', None)
                if var_info is not None and isinstance(var_info, tuple):
                    body_syms, head_syms = _resolve_var_info(var_info)
                    body = ng(body_syms)
                    scope = _apply_to_vars(d, head_syms)
                    result = Conjunction((body, scope))
                    for sym in head_syms:
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
                    body_syms, head_syms = _resolve_var_info(var_info)
                    body = ng(body_syms)
                    scope = _apply_to_vars(d, head_syms)
                    result = Conjunction((body, scope))
                    for sym in head_syms:
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
                if var_info is not None and isinstance(var_info, tuple):
                    body_syms, head_syms = _resolve_var_info(var_info)
                    body = ng(body_syms)
                    scope = _apply_to_vars(d, head_syms)
                    result = Conjunction((body, scope))
                    for sym in head_syms:
                        result = ExistentialPredicate(sym, result)
                    return Negation(result)
                elif var_info is not None:
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
            elif isinstance(item, tuple) and all(
                isinstance(s, (Symbol, _AnonymousVar)) for s in item
            ):
                app = item
            elif callable(item) and not isinstance(item, (Symbol, Constant)):
                # Could be a CPS function from app_label or something else
                if app is None:
                    app = item

        def ng(x):
            # Resolve _AnonymousVar instances to their fresh symbols for body.
            if isinstance(x, tuple):
                body_args = tuple(
                    item.as_symbol() if isinstance(item, _AnonymousVar) else item
                    for item in x
                )
                noun_app = noun1(*body_args)
            else:
                noun_app = noun1(x)
            parts = [noun_app]
            if rel is not None:
                rel_x = x
                if isinstance(x, tuple):
                    # Apply rel to the first named (non-anonymous) var or the tuple
                    named = [
                        item.as_symbol() if isinstance(item, _AnonymousVar) else item
                        for item in x
                    ]
                    rel_x = tuple(named) if len(named) > 1 else named[0]
                rel_val = rel(rel_x)
                if isinstance(rel_val, Conjunction):
                    parts.extend(rel_val.formulas)
                elif rel_val is not None:
                    parts.append(rel_val)
            if len(parts) == 1:
                return parts[0]
            return Conjunction(tuple(parts))

        ng._noun_name = noun1.name if isinstance(noun1, Symbol) else None

        if app is not None:
            ng._var_info = app
        return ng

    def ng1_agg_npc(self, args):
        """Handle ``noun1 OF npc [dims] [rel]`` aggregation noun groups.

        When ``noun1`` is an aggregation function name (count, sum, max, min,
        average) **and** an npc is present, the returned ng1 function carries
        an ``_agg_info = (agg_func_const, npc_cps, per_vars, extra_rel)``
        attribute.  ``det_every`` inspects this attribute to build an
        ``AggregationApplication`` head argument rather than a plain variable.

        The optional ``extra_rel`` (from a trailing ``such that …`` / ``and …``
        relative clause) is conjoined with the npc body in ``det_every`` and
        ``rule_body1``, allowing cross-join filters such as
        ``such that Voxel(?i1;?j1;?k1) and ?d equal to euclidean(…)``.
        """
        noun1 = args[0]
        app = None
        # Last argument before dims is the npc; dims (if present) is tagged
        dims = None
        npc = None
        extra_rel = None
        for a in args[1:]:
            if a is None:
                continue
            if isinstance(a, tuple) and a[0] == '_dims':
                dims = a[1]
            elif isinstance(a, tuple) and a[0] == '_rel':
                extra_rel = a[1]
            elif callable(a) and not isinstance(a, (Symbol, Constant)):
                if npc is None:
                    npc = a
                else:
                    # Second callable: this is the [rel] from bool{rel_b}.
                    # The grammar's transparent ?rel rule means the callable is
                    # not wrapped in ('_rel', ...), so we detect it positionally.
                    extra_rel = lambda x, _r=a: _r(x)
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
        if npc is not None:
            # Enter aggregation path whenever "noun OF the npc" is present.
            # Built-in names (count/sum/max/min/average) map to Python callables;
            # arbitrary names become Symbol(noun_name) — TranslateToLogicWithAggregation
            # promotes any FunctionApplication in a rule head to AggregationApplication.
            agg_func = (
                agg_func_from_noun if agg_func_from_noun is not None
                else (Symbol(noun_name) if noun_name else None)
            )
            if agg_func is not None:
                def ng_agg(x):
                    # Fallback body — only used if det_every does not intercept _agg_info.
                    q = Symbol.fresh()
                    npc(lambda v: Constant(True))
                    return _AggApp(agg_func, (q,))

                ng_agg._agg_info = (agg_func, npc, list(per_vars), extra_rel)
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
            if ops is not None:
                return np(lambda y: ops(lambda z: verb(y, z)))
            elif isinstance(x, tuple):
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

    def rel_fun_call(self, args):
        """Handle ``identifier(label, label, ...)`` as a body predicate atom."""
        func_sym = args[0]   # Symbol from identifier handler
        label_cps_list = args[1:]  # list of CPS lambdas from label handler

        # Materialise each label CPS into a concrete Symbol.
        label_vars = []
        for lbl_cps in label_cps_list:
            if callable(lbl_cps) and not isinstance(lbl_cps, (Symbol, Constant)):
                label_vars.append(lbl_cps(lambda v: v))
            elif isinstance(lbl_cps, Symbol):
                label_vars.append(lbl_cps)
            else:
                label_vars.append(Symbol.fresh())

        if len(label_vars) == 1:
            # Binary predicate: subject + one explicit arg
            y = label_vars[0]
            return ('_rel', lambda x: func_sym(x, y))
        else:
            # N-ary: prepend subject only when it is a scalar Symbol.
            # When the enclosing noun binds a tuple (e.g. Foo (?a;?b;?c)),
            # the labels already cover every argument position and the
            # tuple must NOT be prepended — it would produce wrong arity.
            def rel(x, _vars=label_vars):
                if isinstance(x, Symbol):
                    return func_sym(x, *_vars)
                return func_sym(*_vars)
            return ('_rel', rel)

    def rel_tuple_noun(self, args):
        """Handle 'where (?i; ?j; ?k) is a Noun' → Noun(i, j, k) in rule body.

        args[0] : CPS label lambda (from `label` handler) or Symbol
        args[1] : Symbol from `noun1` handler (already lowercased)

        The enclosing subject variable x is ignored — the tuple provides all
        argument positions explicitly, just like rel_fun_call with explicit labels.
        """
        label_val = args[0]
        noun = args[1]   # Symbol, e.g. Symbol('voxel')

        # Materialise the label CPS to get the underlying tuple of Symbols.
        if callable(label_val) and not isinstance(label_val, (Symbol, Constant)):
            raw = label_val(lambda x: x)
        else:
            raw = label_val

        tup = raw if isinstance(raw, tuple) else (raw,)

        # Resolve _AnonymousVar markers to fresh symbols.
        body_syms = tuple(
            item.as_symbol() if isinstance(item, _AnonymousVar) else item
            for item in tup
        )

        return ('_rel', lambda x: noun(*body_syms))

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
        op = COMPARISON_OPS.get(key)
        if op is None:
            raise self._make_error(
                f"Unrecognised comparison: '{'/'.join(tokens)}'.  "
                "Valid comparisons: greater [equal] than, "
                "lower [equal] than, [not] equal to."
            )
        return op

    # ---- Verb Phrases ----

    def vpdo_v1(self, args):
        verb1 = args[0]
        cp = args[1] if len(args) > 1 and args[1] is not None else None
        if cp is not None:
            def vp(x):
                return _apply_ops(cp, verb1, x)
            return vp
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
        # "verb1 [cp] with probability op"
        verb1 = args[0]
        prob_raw = args[-1]
        if callable(prob_raw) and not isinstance(prob_raw, (Symbol, Constant)):
            prob_value = prob_raw(lambda v: v)
        else:
            prob_value = prob_raw
        return lambda x: ProbabilisticFact(prob_value, verb1(x))

    def vpdo_explicit_prob_vn(self, args):
        # "verbn opn with probability op"
        verb = args[0]
        prob_raw = args[-1]
        ops = args[1] if len(args) > 2 else None
        if callable(prob_raw) and not isinstance(prob_raw, (Symbol, Constant)):
            prob_value = prob_raw(lambda v: v)
        else:
            prob_value = prob_raw
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
        raise self._make_error(
            "Empty noun phrase complement — expected a noun or determiner "
            "+ noun after the verb."
        )

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
        if all(callable(f) and not isinstance(f, (Symbol, Constant, FunctionApplication)) for f in formulas):
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
        if all(callable(f) and not isinstance(f, (Symbol, Constant, FunctionApplication)) for f in formulas):
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
        left = _unwrap_label(args[0])
        op_token = args[1]
        right = _unwrap_label(args[2])
        op = Constant(add) if op_token.value == '+' else Constant(sub)
        return op(left, right)

    def expr_mul(self, args):
        if len(args) == 1:
            return args[0]
        left = _unwrap_label(args[0])
        op_token = args[1]
        right = _unwrap_label(args[2])
        op = Constant(mul) if op_token.value == '*' else Constant(truediv)
        return op(left, right)

    def expr_pow(self, args):
        if len(args) == 1:
            return args[0]
        return Constant(pow)(_unwrap_label(args[0]), _unwrap_label(args[1]))

    def expr_atom_par(self, args):
        return args[0]

    def expr_atom_tuple(self, args):
        return tuple(args)

    def expr_atom_fun(self, args):
        func = args[0]
        func_args = [_unwrap_label(a) for a in args[1:]]
        return FunctionApplication(func, tuple(func_args))

    def expr_atom_fun_upper(self, args):
        """Handle UPPERCASE_NAME(...) function calls, preserving the original case.

        Unlike ``expr_atom_fun`` (which uses ``identifier`` → ``Symbol(name.lower())``),
        this handler is triggered by the ``UPPER_NAME(...)`` grammar alternative and
        keeps the symbol name exactly as written.  This allows SQUALL programs to call
        reserved uppercase symbols such as ``EUCLIDEAN`` so the spatial-prior sugar in
        :mod:`~neurolang.frontend.datalog.sugar.spatial` can recognise them.
        """
        token = args[0]
        # token is a Lark Token (UPPER_NAME terminal) — preserve original case.
        name = token.value if hasattr(token, 'value') else str(token)
        func = Symbol(name)
        func_args = [_unwrap_label(a) for a in args[1:]]
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
        self._capture_pos(args[0])
        if isinstance(args[0], Symbol):
            return args[0]
        name = args[0].value
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        return Symbol(name.lower())

    def transitive(self, args):
        self._capture_pos(args[0])
        if isinstance(args[0], Symbol):
            return args[0]
        name = args[0].value
        if name.startswith("~"):
            name = name[1:]
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        return Symbol(name)

    def transitive_inv(self, args):
        self._capture_pos(args[0])
        token = args[0]
        name = token.value if hasattr(token, 'value') else token.name
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        if name.startswith('~'):
            name = name[1:]
        return _InverseVerbSymbol(Symbol(name))

    def transitive_multiple(self, args):
        self._capture_pos(args[0])
        if isinstance(args[0], Symbol):
            return args[0]
        name = args[0].value
        if name.startswith("~"):
            name = name[1:]
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        return Symbol(name)

    def transitive_multiple_inv(self, args):
        self._capture_pos(args[0])
        token = args[0]
        name = token.value if hasattr(token, 'value') else token.name
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        if name.startswith('~'):
            name = name[1:]
        return _InverseVerbSymbol(Symbol(name))

    def upper_identifier(self, args):
        self._capture_pos(args[0])
        name = args[0].value
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        return Symbol(name.lower())

    def identifier(self, args):
        self._capture_pos(args[0])
        name = args[0].value
        if name.startswith('`') and name.endswith('`'):
            name = name[1:-1]
        return Symbol(name)

    def label_identifier(self, args):
        self._capture_pos(args[0])
        return Symbol(args[0].value)

    def label_tuple_item(self, args):
        """Handle a single item in a tuple label: ?var or _ (anonymous)."""
        if len(args) == 1:
            item = args[0]
            if isinstance(item, Symbol):
                return item
            # ANONYMOUS_LABEL token ("_") → anonymous wildcard sentinel
            if hasattr(item, 'type') and item.type == 'ANONYMOUS_LABEL':
                return _AnonymousVar()
            if hasattr(item, 'value') and item.value == '_':
                return _AnonymousVar()
        # Fallback: anonymous
        return _AnonymousVar()

    def label(self, args):
        if len(args) == 1:
            marker = args[0]
            if isinstance(marker, _AnonymousVar):
                return lambda d: d(marker.as_symbol())
            if hasattr(marker, 'value') and marker.value == '_':
                return lambda d: d(Symbol.fresh())
            sym = marker
            return lambda d: d(sym)
        # Tuple label: args are Symbols or _AnonymousVar instances.
        labels = list(args)
        if len(labels) == 1:
            item = labels[0]
            sym = item.as_symbol() if isinstance(item, _AnonymousVar) else item
            return lambda d: d(sym)
        # Build the tuple, preserving _AnonymousVar markers so ng1_noun can
        # exclude anonymous positions from the rule head.
        return lambda d: d(tuple(labels))

    def ANONYMOUS_LABEL(self, token):
        return _AnonymousVar()

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

    def query_unnamed(self, args):
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

    def query_as(self, args):
        self._clear_scope()
        """Handle 'obtain ops as Name'.

        Builds Implication(name_sym(*free_vars), body) as an IDB rule,
        then returns ('_query_as', (impl, query)) so squall() can
        register both the rule and the query.
        """
        ops, name_sym = args[0], args[1]   # ops: CPS NP; name_sym: Symbol

        # Use a capturing continuation to harvest the head variables and the
        # body formula from the CPS noun phrase, handling both the single-var
        # and tuple-label (multi-var) cases.
        captured_vars = []

        def capturing_d(*var_args):
            """Continuation that records head variables and returns a sentinel."""
            if len(var_args) == 1:
                captured_vars.append(var_args[0])
            else:
                captured_vars.extend(var_args)
            return Constant(True)  # sentinel scope value

        if callable(ops) and not isinstance(ops, (Symbol, Constant)):
            body_formula = ops(capturing_d)
        else:
            body_formula = ops

        # Strip nested quantifiers and Implication wrappers to reach the bare body.
        while isinstance(body_formula, (UniversalPredicate, ExistentialPredicate)):
            body_formula = body_formula.body
        if isinstance(body_formula, Implication):
            # Implication(scope, restriction) — scope is the captured sentinel (True),
            # restriction is the actual body predicate (antecedent).
            body_formula = body_formula.antecedent

        # Strip sentinel True from top-level Conjunction
        def _strip_true(f):
            if isinstance(f, Conjunction):
                parts = [p for p in f.formulas if p != Constant(True)]
                if not parts:
                    return Constant(True)
                if len(parts) == 1:
                    return parts[0]
                return Conjunction(tuple(parts))
            return f

        body_formula = _strip_true(body_formula)

        # If no vars were captured, use free variables from the body.
        if not captured_vars:
            free = sorted(extract_logic_free_variables(body_formula), key=lambda s: s.name)
            head_vars = free
        else:
            head_vars = [v for v in captured_vars if isinstance(v, Symbol)]

        head = name_sym(*head_vars) if head_vars else name_sym()
        impl = Implication(head, body_formula)
        q = Query(head, body_formula)

        return ('_query_as', (impl, q, name_sym.name.lower()))

    # ---- Dimension / Aggregation ----

    def app_dimension(self, args):
        return None

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
        d = args[0]
        # dim_npc_list returns a plain list; other dims return a single tagged tuple.
        return ('_dims', d if isinstance(d, list) else [d])

    def dims_rec(self, args):
        dim = args[0]
        rest = args[1]
        rest_list = rest[1] if (isinstance(rest, tuple) and rest[0] == '_dims') else [rest]
        dim_list = dim if isinstance(dim, list) else [dim]
        return ('_dims', dim_list + rest_list)

    def dim_ng2(self, args):
        # "per region" → fresh groupby variable from the noun
        groupby_var = Symbol.fresh()
        return ('_per', groupby_var)

    def dim_npc_list(self, args):
        """Handle 'per ?i, ?j, ?k' — multiple per-variables under one 'per' keyword.

        Returns a plain list of ('_per', sym) tuples so that dims_base / dims_rec
        can flatten it into the _dims list without any change to ng1_agg_npc.
        """
        per_syms = []
        for npc in args:
            if isinstance(npc, (Symbol, Constant)):
                per_syms.append(npc)
            elif callable(npc) and not isinstance(npc, (Symbol, Constant)):
                result = npc(lambda x: x)
                per_syms.append(
                    result if isinstance(result, (Symbol, Constant)) else Symbol.fresh()
                )
            else:
                per_syms.append(Symbol.fresh())
        return [('_per', s) for s in per_syms]

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
        agg_func_const = args[0]
        npc = args[1]
        if npc is None:
            raise self._make_error(
                "'of the <Relation>' missing after aggregation function.  "
                "Expected: 'the count of the Items' or similar."
            )
        if not isinstance(npc, (Symbol, Constant)) and not callable(npc):
            raise self._make_error(
                f"Cannot use '{type(npc).__name__}' as a relation reference "
                f"after 'of the'."
            )
        return ('_agg', agg_func_const, npc)

    def agg_func(self, args):
        token = args[0]
        name = token.value if hasattr(token, 'value') else str(token)
        func = _AGG_FUNC_MAP.get(name.lower())
        if func is None:
            raise self._make_error(
                f"Unknown aggregation function '{name}'.  "
                "Valid: count, sum, max, min, average."
            )
        return func

    # ---- Probabilistic ----

    def PROBABLY(self, token):
        return "probably"

    def _default(self, data, children, meta):
        if hasattr(meta, 'line') and hasattr(meta, 'column'):
            self._current_line = meta.line
            self._current_column = meta.column
        if len(children) == 1:
            return children[0]
        return children if children else None


def _unwrap_label(val):
    """Materialise a CPS label lambda to a plain Symbol / expression.

    When a label variable (``?x``) appears as an argument inside an
    arithmetic expression or function call, the transformer produces a CPS
    lambda ``lambda d: d(Symbol('x'))`` rather than the Symbol directly.
    This helper applies the identity continuation to recover the value.
    """
    if callable(val) and not isinstance(val, (Symbol, Constant, FunctionApplication)):
        return val(lambda v: v)
    return val


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
        except TypeError as e:
            if "positional argument" in str(e) or "takes " in str(e):
                return d(syms)
            raise
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
    # When verb is an _InverseVerbSymbol (from the ``~`` prefix), the
    # ResolveInvertedFunctionApplicationMixin walker will reverse the
    # argument tuple.  In relative-clause context the caller already
    # supplies (object, subject) so the reversal is correct, but in
    # ``such that`` sentence context the caller supplies (subject, object)
    # and the reversal would swap them.  We detect the wrapper here and
    # pre-swap so the final result is (subject, object) in both cases.
    is_inverse = isinstance(verb, _InverseVerbSymbol)
    if callable(ops) and not isinstance(ops, (Symbol, Constant)):
        if is_inverse:
            return ops(lambda obj: verb(obj, subject))
        return ops(lambda obj: verb(subject, obj))
    elif isinstance(ops, tuple):
        if is_inverse:
            return verb(*ops, subject)
        return verb(subject, *ops)
    else:
        if is_inverse:
            return verb(ops, subject)
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

    Raises
    ------
    UnexpectedTokenError
        A token was encountered that the grammar did not expect.
    UnexpectedCharactersError
        The lexer could not match the input to any terminal.
    SquallSemanticError
        The input parsed successfully but represents an invalid or
        unsupported construction (empty body, unresolved anaphora, etc.).
    """
    source_code = code.strip()
    try:
        tree = COMPILED_GRAMMAR.parse(source_code)
    except UnexpectedToken as e:
        raise UnexpectedTokenError(
            str(e), line=e.line - 1, column=e.column - 1
        ) from e
    except UnexpectedCharacters as e:
        raise UnexpectedCharactersError(
            str(e), line=e.line - 1, column=e.column - 1
        ) from e
    except LarkError as e:
        raise NeuroLangException(
            f"Parse error: {e}"
        ) from e

    try:
        result = SquallTransformer(
            source_lines=source_code.splitlines()
        ).transform(tree)
    except VisitError as e:
        orig = e.orig_exc
        if isinstance(orig, SquallSemanticError):
            raise orig from e
        raise SquallSemanticError(
            f"Internal error while processing '{e.rule}': {orig}"
        ) from e

    from .squall import LogicSimplifier, ResolveInvertedFunctionApplicationMixin
    from neurolang.expression_walker import ExpressionWalker

    class _SquallSimplifier(ResolveInvertedFunctionApplicationMixin, LogicSimplifier, ExpressionWalker):
        pass

    simplifier = _SquallSimplifier()
    if isinstance(result, SquallProgram):
        result = SquallProgram(
            rules=[simplifier.walk(r) for r in result.rules],
            queries=result.queries,
            query_names=result.query_names,
        )
    elif isinstance(result, Union):
        result = Union(tuple(simplifier.walk(f) for f in result.formulas))
    else:
        result = simplifier.walk(result)
    return result
