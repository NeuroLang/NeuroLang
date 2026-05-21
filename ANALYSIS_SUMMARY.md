# NeuroLang Weighted Probabilistic Facts Analysis: Summary

## Investigation Scope

Analyzed the NeuroLang codebase to understand how weighted probabilistic facts work at the IR level:

1. **ProbabilisticFact IR node definition** → `neurolang/probabilistic/expressions.py`
2. **SQUALL syntax handlers** → `neurolang/frontend/datalog/squall_syntax_lark.py`
3. **Grammar rules** → `neurolang/frontend/datalog/neurolang_natural.lark`
4. **Aggregation chain** → `ng1_agg_npc` + `det_every` flow
5. **@ symbol usage** → Grammar verification
6. **Examples** → `examples/plot_neurosynth_implementation.py`

## Key Findings

### 1. ProbabilisticFact Takes Any Expression as Probability

```python
class ProbabilisticFact(ProbabilisticPredicate):
    def __init__(self, probability, body):
        self.probability = probability  # ← ANY IR expression!
        self.body = body
        self._symbols = body._symbols | self.probability._symbols
```

**Types of probabilities observed:**
- `Constant(0.7)` — explicit scalar from `with probability 0.7`
- `Symbol('p1')` — fresh symbol from `probably` keyword
- Any expression via CPS noun phrase extraction in `rule_op_prob`

### 2. Four Main IR Paths to Create ProbabilisticFact

| Path | SQUALL Syntax | Handler | Probability Type | When Used |
|------|---------------|---------|------------------|-----------|
| **VP Explicit** | `subject activates with probability 0.7` | `vpdo_explicit_prob_v1` | `Constant(float)` | In verb phrases |
| **VP Fresh** | `subject probably activates` | `vpdo_prob_v1` | `Symbol(fresh)` | In verb phrases |
| **Rule Explicit** | `define as activates with probability 0.8 every study` | `rule_op_prob` | `Constant(float)` or complex expr | Rule heads |
| **Rule Fresh** | `define as probably activates every study` | `rule_opnn_per` | `Symbol(fresh)` | Rule heads |

### 3. @ Symbol is NOT a Weight Operator

```lark
_LABEL_MARKER : "?"
              | "@"
```

**Finding:** `@` is an alternate label marker (like `?`), NOT a probability operator.

- `?x` and `@x` both introduce labeled variables
- No `@` weight syntax exists in SQUALL
- Probability expressed exclusively via:
  - `with probability <expr>` (explicit)
  - `probably` keyword (fresh symbol)

### 4. Aggregation Chain: ng1_agg_npc → det_every

**Pattern:** `every max of the quantity per item`

**Flow:**
1. **ng1_agg_npc** marks noun group with `_agg_info = (func, npc_cps, per_vars)`
2. **det_every** inspects `_agg_info` and:
   - Extracts body formula from npc via `capturing_cont`
   - Determines aggregation arguments (explicit `per_vars` or all free variables)
   - **Sorts variables by name for determinism**
   - Creates `AggregationApplication(func, sorted_vars)`
   - Returns body formula for antecedent

**Determinism Mechanism:**
```python
free_vars = extract_logic_free_variables(npc_formula)
agg_args = tuple(sorted(free_vars, key=lambda s: s.name))
```

### 5. CPS (Continuation-Passing Style) Enables Correct Scoping

**Architecture:** All expressions and quantifiers flow through CPS continuations:

```
det_every([]) → every(ng) → apply_d(continuation)
```

- Variables bound through continuation chains
- Probability expressions properly track symbols
- Aggregation dimensions resolved via `_agg_info` attribute

**Example:**
```python
rule_body1 creates continuation:
  lambda (head_args, body_formula): 
    det_every(ng)(lambda head_var: 
      ... use head_var in head and body ...)
```

### 6. Conditional Probabilities (rule_op_marg)

**SQUALL:** `define as published with probability every voxel conditioned to ...`

**Produces:**
```python
Implication(
  head=published(v, ProbabilisticQuery(PROB, (v,))),
  antecedent=Condition(conditioned, conditioning)
)
```

**Key:** Uses `ProbabilisticQuery(PROB, vars)` in head to mark as conditional.
Later passes expand this into three-rule form via `TranslateProbabilisticQueryMixin.rewrite_conditional_query`.

## Handler Catalog

### VP-Level (Return CPS Functions)

1. **vpdo_explicit_prob_v1** (861-865)
   - Input: `verb1 with probability number`
   - Output: `lambda x: ProbabilisticFact(Constant(num), verb1(x))`

2. **vpdo_explicit_prob_vn** (867-876)
   - Input: `verbn opn with probability number`
   - Output: `lambda x: ProbabilisticFact(Constant(num), _apply_ops(ops, verb, x))`

3. **vpdo_prob_v1** (848-852)
   - Input: `probably verb1`
   - Output: `lambda x: ProbabilisticFact(Symbol.fresh(), verb1(x))`

4. **vpdo_prob_vn** (854-859)
   - Input: `probably verbn opn`
   - Output: `lambda x: ProbabilisticFact(Symbol.fresh(), _apply_ops(ops, verb, x))`

### Rule-Level (Return Implications with ProbabilisticFact Head)

5. **rule_op_prob** (212-229)
   - Input: `define as verb with probability np rule_body1`
   - Output: `Implication(ProbabilisticFact(prob, head), body)`
   - Note: Extracts probability from CPS noun phrase

6. **rule_opnn_per** (297-319)
   - Input: `define as probably verbn rule_body1 ops`
   - Output: `Implication(ProbabilisticFact(fresh_prob, head), body)`
   - Note: Always fresh probability symbol

7. **rule_op_marg** (231-255)
   - Input: `define as verb with probability rule_body1_cond`
   - Output: `Implication(head(vars, ProbabilisticQuery(PROB, vars)), Condition(...))`
   - Note: Marks as marginal/conditional query

### Aggregation (Set _agg_info Attribute)

8. **ng1_agg_npc** (663-758)
   - Input: `noun1 of npc [dims]` where noun1 is max/min/sum/count/average
   - Output: Returns callable ng with `ng._agg_info = (agg_func_const, npc_cps, per_vars)`
   - Note: Marks for special handling by det_every

9. **det_every** (480-556)
   - Input: Called with ng carrying optional `_agg_info`
   - Output: If `_agg_info` present: builds `AggregationApplication` in head
   - Output: If no `_agg_info`: builds `UniversalPredicate` (standard quantification)
   - Key: Extracts free variables sorted by name

## Examples from Codebase

### Example 1: Simple Probabilistic Fact
```python
# SQUALL: "every study activates with probability 0.7"
# IR:
Implication(
  ProbabilisticFact(Constant(0.7), activates(s)),
  study(s)
)
```

### Example 2: Probabilistic with Fresh Symbol
```python
# SQUALL: "define as probably published every study"
# IR:
Implication(
  ProbabilisticFact(Symbol('_fresh_1234'), published(s)),
  study(s)
)
```

### Example 3: Aggregation with Groupby
```python
# SQUALL: "define as result every max of the quantity per item"
# IR:
Implication(
  result(AggregationApplication(Constant(max), (Symbol('item'),))),
  Conjunction((quantity(...), ...))
)
```

### Example 4: Conditional Probability
```python
# SQUALL: "define as published with probability every voxel 
#          conditioned to every study activates"
# IR:
Implication(
  published(v, ProbabilisticQuery(PROB, (v,))),
  Condition(
    conditioned=voxel(v),
    conditioning=study(s) & activates(s)
  )
)
```

### Example 5: IR Builder Usage (plot_neurosynth_implementation.py:96)
```python
with nl.scope as e:
    e.ActivationGivenTerm[e.i, e.j, e.k, e.PROB[e.i, e.j, e.k]] = (
        e.Activation(e.i, e.j, e.k) // e.TermAssociation("auditory")
    )
```

The `e.PROB[e.i, e.j, e.k]` construct creates a `ProbabilisticQuery` in the head.

## Symbol Tracking Architecture

```python
class ProbabilisticPredicate(Definition):
    def __init__(self, probability, body):
        # Fuse symbols from both probability and body
        self._symbols = body._symbols | self.probability._symbols
```

**Why This Matters:**
- Solvers know all free variables that must be instantiated
- Type system correctly infers variable types
- Probabilistic rules with variables in both probability and body work correctly

**Example:**
```python
ProbabilisticFact(
  p_var,        # Free symbol
  pred(x, y)    # Free symbols: x, y
)
# Combined _symbols: {p_var, x, y}
```

## Document Structure

Three detailed reports have been generated:

1. **WEIGHTED_PROBABILISTIC_FACTS_REPORT.md** (10 sections)
   - Complete IR-level analysis
   - Handler code with full context
   - Grammar rules
   - Aggregation CPS chain execution
   - Complete end-to-end example

2. **PROBABILISTIC_FACTS_QUICK_REFERENCE.md** (7 sections)
   - Concise how-to guide
   - Quick handler lookup table
   - Aggregation determinism explanation
   - Key files reference
   - Summary bullet points

3. **PROBABILISTIC_FACTS_TEST_EXAMPLES.md** (8 test sections)
   - Concrete test cases from test_squall_parser.py
   - Expected IR output for each test
   - Test code snippets
   - How to run tests

## Key Insights for Implementation

1. **Probability Field is Flexible**
   - Not limited to `Constant` — can be `Symbol` or complex expressions
   - Enables symbolic probabilities that solver resolves later

2. **CPS Architecture Guarantees Correctness**
   - Variable binding through continuations ensures proper scoping
   - Works for simple predicates and complex aggregations

3. **Aggregation Determinism**
   - Free variables sorted by name in `det_every`
   - Guarantees reproducible IR regardless of parse order

4. **Fresh Symbols Allow Deferred Resolution**
   - `probably` keyword creates `Symbol.fresh()`
   - Solver must assign probability values later

5. **No Custom Weight Syntax Needed**
   - `with probability <expr>` handles all cases
   - Expression can be simple constant or complex formula
   - CPS extraction allows noun phrase probabilities

## Verification Checklist

- [x] `ProbabilisticFact` takes ANY expression as probability
- [x] `vpdo_explicit_prob_v1` handler produces correct IR
- [x] Aggregation `_agg_info` properly propagated
- [x] `det_every` handles `_agg_info` inspection correctly
- [x] Aggregation arguments sorted by name for determinism
- [x] `@` symbol confirmed as label marker only
- [x] Four main probabilistic expression paths identified
- [x] Examples verified from actual codebase
- [x] Test cases located and documented
- [x] CPS continuation chain verified

## Recommendations

For weighted probabilistic facts in NeuroLang:

1. **Use explicit syntax** when probability is known: `with probability 0.7`
2. **Use fresh symbols** when probability must be learned: `probably`
3. **For complex probabilities**, leverage CPS noun phrases: `with probability some complex expression`
4. **Aggregation works automatically** via `ng1_agg_npc + det_every`
5. **Don't use @ for weights** — it's a label marker only

---

**Report Generated:** 2026-04-17
**Analysis Scope:** NeuroLang codebase (squall-cnl branch)
**Files Examined:** 5 core files, 1 grammar file, 1 test file, 1 example file
