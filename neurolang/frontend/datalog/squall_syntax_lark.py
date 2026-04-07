"""
SQUALL (Semantically controlled Query-Answerable Logical Language) parser.

Translates controlled English sentences into NeuroLang logical expressions
using a Lark grammar and Transformer. Uses Continuation-Passing Style (CPS)
for correct quantifier scoping following Montague semantics.

References:
    S. Ferré, "SQUALL: The expressiveness of SPARQL 1.1 made available
    as a controlled natural language", Data & Knowledge Engineering, 2014.
"""
import os

from lark import Lark, Transformer
from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

from ...datalog import Conjunction, Fact, Implication, Negation, Union
from ...datalog.aggregation import AggregationApplication
from ...expressions import (
    Constant,
    FunctionApplication,
    Query,
    Symbol,
)
from ...logic import ExistentialPredicate, UniversalPredicate
from ...probabilistic.expressions import ProbabilisticFact


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
        rules = [a for a in args if a is not None]
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
        verb = args[0]
        np_prob = args[1]
        body = args[2]
        head_pred, consequent_args = verb
        head = head_pred(*consequent_args) if consequent_args else head_pred
        prob_val = np_prob(lambda x: x)
        return Implication(
            ProbabilisticFact(prob_val, head),
            body,
        )

    def rule_opnn(self, args):
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
                # CPS NP from the ops: extract var and restriction
                ops_var = Symbol.fresh()
                ops_body = ops(lambda x: x)
                if isinstance(ops_body, (Symbol, Constant)):
                    head_args.append(ops_body)
                else:
                    # The CPS application gives a formula/quantified expr
                    # We need to flatten it into Datalog
                    ops_extracted = _extract_datalog_body(ops, head_args)
                    all_body_parts.extend(ops_extracted)
            elif isinstance(ops, (list, tuple)):
                head_args.extend(ops)

        if len(all_body_parts) == 1:
            full_body = all_body_parts[0]
        else:
            full_body = Conjunction(tuple(all_body_parts))

        consequent = verb(*head_args) if head_args else verb()
        return Implication(consequent, full_body)

    def rule_opnn_per(self, args):
        verb = args[0]
        body = args[1]
        ops = args[2] if len(args) > 2 else None
        head_symbol = verb
        body_formula = body
        if ops is not None:
            all_args = ops
            consequent = head_symbol(*all_args)
        else:
            consequent = head_symbol
        return Implication(
            ProbabilisticFact(Constant(True), consequent),
            body_formula,
        )

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
        def every(ng):
            def apply_d(d):
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
        noun1 = args[0]
        app = None
        npc = args[-1]
        dims = None
        for a in args[1:-1]:
            if a is not None:
                if isinstance(a, tuple) and a[0] == '_dims':
                    dims = a[1]
                else:
                    app = a
        if dims is not None:
            return lambda x: noun1(x)
        return lambda x: noun1(x)

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
        ng2 = args[0]
        vp = args[1]
        def rel(x):
            return vp(x)
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
        from ...logic import Disjunction
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
        sym = args[0] if isinstance(args[0], Symbol) else Symbol(args[0].value)
        sym._inverse = True
        return sym

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
        sym = args[0] if isinstance(args[0], Symbol) else Symbol(args[0].value)
        sym._inverse = True
        return sym

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
        ops = args[0]
        return ops

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
        return args[0]

    def dim_npc(self, args):
        return args[0]

    # ---- Probabilistic ----

    def PROBABLY(self, token):
        return "probably"

    def _default(self, data, children, meta):
        if len(children) == 1:
            return children[0]
        return children if children else None


def _extract_datalog_body(cps_np, head_args):
    """
    Extract body predicates from a CPS NP for use in Datalog rules.
    Also adds bound variables to head_args.
    Returns a list of body formula parts.
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
    """
    Flatten quantified expressions into a list of Datalog body predicates.
    Strips existential/universal quantifiers and extracts conjuncts.
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
    """Apply a continuation d to a tuple of symbols, splatting if d is a Symbol."""
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
