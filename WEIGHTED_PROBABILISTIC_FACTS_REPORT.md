# NeuroLang Weighted Probabilistic Facts: IR-Level Analysis

## Executive Summary

This report details how weighted probabilistic facts are expressed at the IR (Intermediate Representation) level in NeuroLang, specifically focusing on:
- The `ProbabilisticFact` IR node and its probability parameter handling
- SQUALL syntax handlers that produce `ProbabilisticFact` instances
- How the CPS (Continuation-Passing Style) architecture enables correct scoping
- Aggregation handling through `ng1_agg_npc` and `det_every`

## 1. ProbabilisticFact IR Node

### Definition (neurolang/probabilistic/expressions.py)

```python
class ProbabilisticPredicate(Definition):
    def __init__(self, probability, body):
        self.probability = probability
        self.body = body
        self._symbols = body._symbols | self.probability._symbols

    @property
    def functor(self):
        return self.body.functor

    @property
    def args(self):
        return self.body.args


class ProbabilisticFact(ProbabilisticPredicate):
    def __repr__(self):
        return "ProbabilisticFact{{{} :: {} : {}}}".format(
            self.body, self.probability, self.type
        )
```

### Key Properties

1. **Probability Parameter**: Can be ANY expression (not just scalar)
   - `Constant(0.7)` — explicit scalar probability
   - `Symbol('p1')` — symbolic probability (fresh or named)
   - Any expression in `probability` is tracked via `self.probability._symbols`

2. **Body Parameter**: Must be a callable predicate
   - Typically `FunctionApplication` (e.g., `pred(x, y)`)
   - Tracks all symbols in both probability and body via `self._symbols`

3. **Symbol Fusion**: The combined symbol set includes both probability variables and body variables
   - Enables proper scoping of probabilistic rules with free variables

## 2. SQUALL Syntax Handlers Producing ProbabilisticFact

### Handler: vpdo_explicit_prob_v1

**Grammar Rule:**
```lark
vpdo : verb1 [ cp ] _WITH _PROBABILITY number -> vpdo_explicit_prob_v1
```

**Handler Code (squall_syntax_lark.py:861-865):**
```python
def vpdo_explicit_prob_v1(self, args):
    # "verb1 [cp] with probability number"
    verb1 = args[0]
    prob_value = args[-1]   # Constant(float)
    return lambda x: ProbabilisticFact(prob_value, verb1(x))
```

**Input:** SQUALL text like `activates with probability 0.7`
**Output:** CPS function `lambda x: ProbabilisticFact(Constant(0.7), activates(x))`

**Flow:**
1. Parser extracts `verb1` (e.g., `activates` → `Symbol('activates')`)
2. Parser extracts `prob_value` (e.g., `0.7` → `Constant(0.7)`)
3. Returns a CPS function that wraps the verb in `ProbabilisticFact`
4. When applied to subject variable `x`, produces: `ProbabilisticFact(Constant(0.7), activates(x))`

### Handler: vpdo_explicit_prob_vn

**Grammar Rule:**
```lark
vpdo : verbn opn _WITH _PROBABILITY number -> vpdo_explicit_prob_vn
```

**Handler Code (squall_syntax_lark.py:867-876):**
```python
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
```

**Input:** SQUALL text like `activates voxel with probability 0.3`
**Output:** CPS function that wraps n-ary predicate call in `ProbabilisticFact`

### Handler: vpdo_prob_v1 (Probabilistic with fresh symbol)

**Grammar Rule:**
```lark
vpdo : PROBABLY verb1 [ cp ] -> vpdo_prob_v1
```

**Handler Code (squall_syntax_lark.py:848-852):**
```python
def vpdo_prob_v1(self, args):
    # "probably verb1 [cp]" → ProbabilisticFact with fresh prob symbol
    verb1 = args[0]
    fresh_prob = Symbol.fresh()
    return lambda x: ProbabilisticFact(fresh_prob, verb1(x))
```

**Output:** `ProbabilisticFact(Symbol('_fresh_1234'), verb1(x))`
- Uses fresh probability symbol instead of explicit value
- Later inference must assign probability to the symbol

### Handler: vpdo_prob_vn (N-ary probabilistic)

**Grammar Rule:**
```lark
vpdo : PROBABLY verbn opn -> vpdo_prob_vn
```

**Handler Code (squall_syntax_lark.py:854-859):**
```python
def vpdo_prob_vn(self, args):
    # "probably verbn opn" → ProbabilisticFact with fresh prob symbol
    verb = args[0]
    ops = args[1]
    fresh_prob = Symbol.fresh()
    return lambda x: ProbabilisticFact(fresh_prob, _apply_ops(ops, verb, x))
```

## 3. Rule-Level Probabilistic Fact Handlers

### Handler: rule_op_prob

**Grammar Rule:**
```lark
rule_op : _rule_start verb1 _WITH _PROBABILITY np rule_body1 -> rule_op_prob
```

**Handler Code (squall_syntax_lark.py:212-229):**
```python
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
```

**Output Example:**
```
Input:  "define as activates with probability 0.8 every study."
Output: Implication(
          ProbabilisticFact(Constant(0.8), activates(s)),
          study(s)
        )
```

### Handler: rule_op_marg (Marginal/Conditional)

**Grammar Rule:**
```lark
rule_op : _rule_start verb1 _WITH _PROBABILITY rule_body1_cond -> rule_op_marg
```

**Handler Code (squall_syntax_lark.py:231-255):**
```python
def rule_op_marg(self, args):
    """Build a MARG query from ``define as verb with probability rule_body1_cond``."""
    items = [a for a in args if a is not None]
    verb = items[0]
    body_result = items[1]

    if isinstance(body_result, tuple) and body_result[0] == '_rule_body':
        head_args, body_formula = body_result[1]
    else:
        head_args, body_formula = [], Constant(True)

    prob_query_arg = ProbabilisticQuery(PROB, tuple(head_args))
    head = verb(*(list(head_args) + [prob_query_arg]))
    return Implication(head, body_formula)
```

**Output Example:**
```
Input:  "define as published with probability every voxel 
          conditioned to every study activates."
Output: Implication(
          published(v, ProbabilisticQuery(PROB, (v,))),
          Condition(voxel(v), study(s) & activates(s))
        )
```

**Note:** This returns a standard `Implication` with a `ProbabilisticQuery(PROB, ...)` in the head.
Later rewrites by `TranslateProbabilisticQueryMixin.rewrite_conditional_query` expand this into
the three-rule conditional probability form.

### Handler: rule_opnn_per (N-ary Probabilistic Rule)

**Grammar Rule:**
```lark
rulen : _rule_start PROBABLY verbn rule_body1 _CONDITIONED? _BREAK? ops -> rule_opnn_per
```

**Handler Code (squall_syntax_lark.py:297-319):**
```python
def rule_opnn_per(self, args):
    items = [a for a in args if a is not None 
             and not (isinstance(a, str) and a.lower() in ('probably', 'conditioned'))]
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
```

**Output Example:**
```
Input:  "define as probably activates every study voxel."
Output: Implication(
          ProbabilisticFact(Symbol('_fresh_5678'), activates(s, v)),
          study(s) & voxel(v)
        )
```

## 4. Grammar Rules (neurolang_natural.lark)

```lark
vpdo : PROBABLY verb1 [ cp ]                  -> vpdo_prob_v1
     | PROBABLY verbn opn                     -> vpdo_prob_vn
     | verb1 [ cp ] _WITH _PROBABILITY number -> vpdo_explicit_prob_v1
     | verbn opn _WITH _PROBABILITY number    -> vpdo_explicit_prob_vn
     | verb1 [ cp ]                           -> vpdo_v1
     | verbn opn                              -> vpdo_vn

rule_op : _rule_start [ PROBABLY ] verb1 rule_body1
        | _rule_start PROBABLY verb1 rule_body1_cond
        | _rule_start verb1 _WITH _PROBABILITY np rule_body1 -> rule_op_prob
        | _rule_start verb1 _WITH _PROBABILITY rule_body1_cond -> rule_op_marg
```

## 5. Label Marker @ vs Probability

**Grammar Extract (neurolang_natural.lark:195-201):**
```lark
_LABEL_MARKER : "?"
              | "@"

ANONYMOUS_LABEL : "_"
```

**Finding:** `@` is used ONLY as an alternate label marker (like `?`), NOT as a probability operator.
- `?x` and `@x` both introduce labeled variables
- No probability weight operator `@` exists in the SQUALL grammar
- Probability is expressed exclusively via:
  1. `with probability <number>` (explicit)
  2. `probably` keyword (fresh symbol)

## 6. Aggregation Chain: ng1_agg_npc → det_every

### ng1_agg_npc Handler (squall_syntax_lark.py:663-758)

**Grammar Context:**
```lark
ng1 : noun1 OF npc [dims]  → ng1_agg_npc (via the larger grammar)
```

**Handler Code (Key Section):**
```python
def ng1_agg_npc(self, args):
    """Handle ``noun1 OF npc [dims]`` aggregation noun groups."""
    noun1 = args[0]
    app = None
    dims = None
    npc = None
    # ... extract app, npc, dims ...

    per_vars = []
    agg_specs = []
    if dims is not None:
        for d in dims:
            if isinstance(d, tuple) and d[0] == '_per':
                per_vars.append(d[1])
            elif isinstance(d, tuple) and d[0] == '_agg':
                agg_specs.append((d[1], d[2]))

    # Map noun1 to aggregation function constant
    noun_name = noun1.name.lower() if isinstance(noun1, Symbol) else None
    agg_func_from_noun = _AGG_FUNC_MAP.get(noun_name) if noun_name else None

    if npc is not None:
        agg_func = (
            agg_func_from_noun if agg_func_from_noun is not None
            else (Symbol(noun_name) if noun_name else None)
        )
        if agg_func is not None:
            def ng_agg(x):
                # Fallback body — only used if det_every does not intercept _agg_info
                q = Symbol.fresh()
                npc(lambda v: Constant(True))
                return _AggApp(agg_func, (q,))

            ng_agg._agg_info = (agg_func, npc, list(per_vars))
            if app is not None:
                ng_agg._var_info = app
            return ng_agg
    # ... handle agg_specs ...
```

**Output:**
- Returns a callable `ng_agg(x)` with attached `_agg_info` tuple
- `_agg_info = (agg_func_const, npc_cps, per_vars_list)`
- `agg_func_const` is either `Constant(builtin_func)` or `Symbol(custom_name)`
- `npc_cps` is the CPS noun phrase for the aggregated content
- `per_vars_list` contains groupby variables from `_per` dimensions

### det_every Handler (squall_syntax_lark.py:480-556)

**Handler Code (Key Section for Aggregation):**
```python
def det_every(self, args):
    """Return the universal-quantifier CPS determinant function."""
    def every(ng):
        def apply_d(d):
            # Special handling for aggregation ng1
            agg_info = getattr(ng, '_agg_info', None)
            if agg_info is not None:
                agg_func_const, npc_cps, per_vars = agg_info

                # Build the npc body formula to discover free variables
                captured = []

                def capturing_cont(v, _cap=captured):
                    _cap.append(v)
                    return Constant(True)

                npc_formula = npc_cps(capturing_cont)

                if per_vars:
                    # Explicit groupby: use the captured witness variable
                    if captured and isinstance(captured[0], Symbol):
                        agg_args = (captured[0],)
                    elif isinstance(npc_formula, ExistentialPredicate):
                        agg_args = (npc_formula.head,)
                    else:
                        agg_args = (Symbol.fresh(),)
                else:
                    # No explicit groupby: aggregate over all free variables
                    # in the npc body, sorted by name for determinism
                    free_vars = extract_logic_free_variables(npc_formula)
                    if free_vars:
                        agg_args = tuple(
                            sorted(free_vars, key=lambda s: s.name)
                        )
                    elif captured and isinstance(captured[0], Symbol):
                        agg_args = (captured[0],)
                    elif isinstance(npc_formula, ExistentialPredicate):
                        agg_args = (npc_formula.head,)
                    else:
                        agg_args = (Symbol.fresh(),)

                agg_expr = _AggApp(agg_func_const, agg_args)
                d(agg_expr)  # adds agg_expr to head_args

                # Return the npc formula for _flatten_to_datalog to extract
                # the body predicates
                return npc_formula

            # Standard (non-aggregation) path: universal quantification
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
```

**CPS Chain Execution:**
1. `det_every([])` returns the `every` function
2. `every(ng)` returns the `apply_d` function (CPS continuation)
3. `apply_d(d)` where `d` is a continuation:
   - Checks for `_agg_info` on the ng
   - If present: builds `AggregationApplication` and returns npc body formula
   - If absent: builds `UniversalPredicate(x, Implication(...))`

**Aggregation CPS Path:**
1. `npc_cps` is applied to `capturing_cont` to build formula
2. Free variables are extracted from formula
3. `agg_args` is determined (either explicit `per_vars` or all free vars)
4. `AggregationApplication(agg_func_const, agg_args)` is created
5. Continuation `d(agg_expr)` is called to add to head arguments
6. Returns `npc_formula` for body extraction

## 7. Example: Complete Flow

### Input
```
define as result every max of the quantity where item_count per item.
```

### Parse Tree
```
rule
  ├─ rule_op
  │  ├─ verb1: Symbol('result')
  │  ├─ rule_body1
  │     ├─ det: every (returns det_every([]))
  │     ├─ ng1: ng1_agg_npc
  │        ├─ noun1: Symbol('max')
  │        ├─ npc: CPS formula
  │        ├─ dims: [('_per', Symbol('item'))]
```

### Semantic Flow
1. `rule_body1` calls `det_every([])` to get determinant function
2. `det_every([])` returns `every` function
3. `rule_body1` creates continuation that expects variables and body
4. `every(ng)` inspects `ng._agg_info`:
   - `agg_func_const = Constant(max)`
   - `npc_cps = <CPS formula for quantity>`
   - `per_vars = [Symbol('item')]`
5. `npc_formula = npc_cps(capturing_cont)` builds body formula
6. `agg_args = (Symbol('item'),)` (from explicit per_vars)
7. Creates `AggregationApplication(Constant(max), (Symbol('item'),))`
8. Returns `npc_formula` for body

### Output IR
```
Implication(
  result(AggregationApplication(Constant(max), (Symbol('item'),))),
  Conjunction((item_count(...), ...))
)
```

## 8. Type System Integration

From `neurolang/frontend/datalog/sugar/__init__.py`:

```python
from ....probabilistic.expressions import (
    PROB,
    Condition,
    ProbabilisticFact,
    ProbabilisticChoice,
    ProbabilisticQuery,
)
```

**Sugar Processing:**
- `TranslateColumnsToAtoms` handles `Column` sugar
- `TranslateSelectByFirstColumn` handles `SelectByFirstColumn` sugar
- Probabilistic structures are preserved through transformation

## 9. Key Insights

1. **Probability is an Expression**: The `probability` field in `ProbabilisticFact` can be:
   - `Constant(float)` from explicit `with probability N`
   - `Symbol` from `probably` keyword or complex expressions
   - Any IR expression via `np` (noun phrase) CPS in `rule_op_prob`

2. **CPS enables Correct Scoping**: 
   - Variables are bound through continuation chains
   - Probability expressions properly track free variables
   - Aggregation dimensions are resolved through `_agg_info` attribute

3. **No Weight Operator (@)**:
   - `@` is strictly a label marker (alternative to `?`)
   - Weights/probabilities are expressed via:
     - `with probability <expr>` (explicit)
     - `probably` (fresh symbol)

4. **Aggregation is Deterministic**:
   - Free variables sorted by name for determinism
   - Fallback to captured variables or explicit per_vars
   - Extracted via `capture_cont` CPS continuation

5. **Conditional Probabilities**:
   - `Condition(conditioned, conditioning)` wraps conditional formulas
   - `ProbabilisticQuery(PROB, vars)` marks marginal/conditional queries
   - Later rewrite passes expand into three-rule form

## 10. Integration Points

- **IR Builder (nl.scope)**: Uses `e.PROB[vars]` to construct probabilistic queries
- **Sugar Layer**: `ProbabilisticFact` preserved through expression walking
- **Solver Integration**: `RegionFrontendCPLogicSolver` interprets `ProbabilisticFact` in probabilistic logic programs
