# NeuroLang Weighted Probabilistic Facts: Quick Reference

## What is ProbabilisticFact?

A NeuroLang IR node that wraps a predicate with a probability (weight) expression:

```python
class ProbabilisticFact(ProbabilisticPredicate):
    def __init__(self, probability, body):
        # probability: ANY expression (Constant, Symbol, etc.)
        # body: FunctionApplication (the predicate)
```

**Examples:**
- `ProbabilisticFact(Constant(0.7), activates(s))` — explicit weight
- `ProbabilisticFact(Symbol('p1'), pred(x, y))` — symbolic weight
- Both probability and body variables are tracked in `self._symbols`

---

## How to Create Probabilistic Facts in SQUALL

### 1. Explicit Probability (Verb Phrase)
```
every study activates with probability 0.7
↓
vpdo_explicit_prob_v1 returns:
lambda x: ProbabilisticFact(Constant(0.7), activates(x))
```

**Handler:** `vpdo_explicit_prob_v1` (lines 861-865)
```python
def vpdo_explicit_prob_v1(self, args):
    verb1 = args[0]
    prob_value = args[-1]   # Constant(float)
    return lambda x: ProbabilisticFact(prob_value, verb1(x))
```

### 2. Probabilistic with Fresh Symbol (Verb Phrase)
```
every study probably activates
↓
vpdo_prob_v1 returns:
lambda x: ProbabilisticFact(Symbol('_fresh_1234'), activates(x))
```

**Handler:** `vpdo_prob_v1` (lines 848-852)
```python
def vpdo_prob_v1(self, args):
    verb1 = args[0]
    fresh_prob = Symbol.fresh()
    return lambda x: ProbabilisticFact(fresh_prob, verb1(x))
```

### 3. Rule with Explicit Probability
```
define as activates with probability 0.8 every study.
↓
rule_op_prob returns:
Implication(
  ProbabilisticFact(Constant(0.8), activates(s)),
  study(s)
)
```

**Handler:** `rule_op_prob` (lines 212-229)
```python
def rule_op_prob(self, args):
    verb = items[0]
    np_prob = items[1]  # CPS noun phrase
    body_result = items[2]
    head_args, body_formula = extract_from(body_result)
    head = verb(*head_args) if head_args else verb()
    # Extract probability from CPS if needed
    prob_val = np_prob(lambda x: x) if callable(np_prob) else np_prob
    return Implication(ProbabilisticFact(prob_val, head), body_formula)
```

### 4. N-ary Probabilistic Rule
```
define as probably activates every study voxel.
↓
rule_opnn_per returns:
Implication(
  ProbabilisticFact(Symbol('_fresh_5678'), activates(s, v)),
  study(s) & voxel(v)
)
```

**Handler:** `rule_opnn_per` (lines 297-319)
- Creates fresh probability symbol
- Collects body arguments from subject and objects

---

## Aggregation Chain: ng1_agg_npc → det_every

### ng1_agg_npc (Create aggregation noun with _agg_info)

```
"every max of the quantity where item_count per item"
↓
ng1_agg_npc creates ng with attached _agg_info:
  ng._agg_info = (Constant(max), npc_cps, [Symbol('item')])
```

**Key Insight:** Marks the noun group for aggregation handling.

### det_every (Execute aggregation via CPS)

When `every(ng)` is called with `ng._agg_info` present:

1. Extracts `_agg_info = (agg_func_const, npc_cps, per_vars)`
2. Applies `npc_cps(capturing_cont)` to build body formula
3. Extracts free variables (sorted by name for determinism)
4. Creates `AggregationApplication(agg_func_const, agg_args)`
5. Returns body formula for extraction into rule antecedent

**CPS Chain:**
```python
det_every([]) → every → apply_d(d)
                           ├─ Check _agg_info
                           ├─ Build AggregationApplication
                           ├─ Call d(agg_expr) to add to head
                           └─ Return npc_formula for body
```

---

## Probability in Different Forms

| Form | Type | Handler | Example |
|------|------|---------|---------|
| Explicit | `Constant(float)` | `vpdo_explicit_prob_v1` | `with probability 0.7` |
| Fresh | `Symbol` | `vpdo_prob_v1` | `probably` |
| Rule explicit | `Constant(float)` | `rule_op_prob` | `define as ... with probability 0.8 ...` |
| Rule fresh | `Symbol` | `rule_opnn_per` | `define as probably ... ` |
| Conditional | `ProbabilisticQuery(PROB, vars)` | `rule_op_marg` | `with probability ... conditioned to ...` |

---

## Important: @ is NOT a weight operator

```lark
_LABEL_MARKER : "?"
              | "@"
```

The `@` symbol is an **alternate label marker** (like `?`), NOT a probability operator.
- `?x` and `@x` both introduce labeled variables
- Probability is expressed ONLY via:
  - `with probability <expr>` (explicit)
  - `probably` keyword (fresh symbol)

---

## Aggregation Determinism

When no explicit `per` dimension is given, `det_every` collects **all free variables** from the npc body:

```python
free_vars = extract_logic_free_variables(npc_formula)
if free_vars:
    agg_args = tuple(sorted(free_vars, key=lambda s: s.name))  # SORTED!
```

**Why sorted?** Ensures deterministic argument order regardless of parse order.

---

## End-to-End Flow: "define as result every max of the quantity per item"

```
Input:  "define as result every max of the quantity per item."
                                   ↓
Grammar: rule_op + rule_body1 + ng1_agg_npc

Transform Chain:
1. ng1_agg_npc creates ng with:
   ng._agg_info = (Constant(max), npc_for_quantity, [Symbol('item')])

2. rule_body1 calls det_every([])

3. det_every returns every function

4. every(ng) is called with ng carrying _agg_info

5. apply_d(continuation) is invoked:
   - Detects _agg_info
   - Applies npc_cps to capturing_cont to build formula
   - Extracts free variable (item) from per_vars
   - Creates AggregationApplication(Constant(max), (Symbol('item'),))
   - Calls continuation to add to head_args
   - Returns npc_formula for body

Output IR:
Implication(
  result(AggregationApplication(Constant(max), (Symbol('item'),))),
  Conjunction((quantity(...), ...))
)
```

---

## Symbol Tracking

All symbols (from probability and body) are combined:

```python
self._symbols = body._symbols | self.probability._symbols
```

This enables:
- Proper scoping of probabilistic rules with free variables in both
- Solver to know which variables must be instantiated
- Type system to correctly infer variable types

---

## Key Files

- **IR Definition:** `neurolang/probabilistic/expressions.py`
- **Syntax Handlers:** `neurolang/frontend/datalog/squall_syntax_lark.py`
- **Grammar Rules:** `neurolang/frontend/datalog/neurolang_natural.lark`
- **Sugar Processing:** `neurolang/frontend/datalog/sugar/__init__.py`
- **Tests:** `neurolang/frontend/datalog/tests/test_squall_parser.py`

---

## Summary

1. **ProbabilisticFact** = `(probability_expr, predicate_expr)`
2. **Probability** can be any IR expression, not just scalar
3. **SQUALL syntax** provides multiple ways to express weighted predicates
4. **CPS (Continuation-Passing Style)** enables correct variable scoping
5. **Aggregation** is handled through `_agg_info` attribute inspection in `det_every`
6. **@ is a label marker**, not a probability operator
7. **Fresh symbols** allow symbolic probabilities to be resolved later by the solver
