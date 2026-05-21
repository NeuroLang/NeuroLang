# Probabilistic Facts: Test Examples from NeuroLang

This document shows concrete test cases from `test_squall_parser.py` demonstrating how probabilistic facts work.

## Test 1: Probabilistic Conditioned Prior

**Grammar:** `define as probably verb1 rule_body1_cond_prior`

**SQUALL Input:**
```
define as probably Published every Voxel conditioned to every Study activates.
```

**Expected IR:**
```python
Implication(
  head=Symbol('published'),
  antecedent=Condition(
    conditioned=voxel(v),
    conditioning=study(s) & activates(s)
  )
)
```

**Test Code (test_squall_parser.py:639-668):**
```python
def test_squall_conditioned_prior_produces_condition_node():
    """rule_body1_cond_prior returns an Implication with Condition antecedent."""
    from neurolang.probabilistic.expressions import Condition

    result = parser(
        "define as probably Published every Voxel conditioned to every Study activates."
    )
    assert isinstance(result, Implication)
    assert isinstance(result.antecedent, Condition)
    
    # Check conditioned part contains voxel
    cond = result.antecedent
    cond_syms = {s.name for s in cond.conditioned._symbols}
    assert "voxel" in cond_syms
    
    # Check conditioning part contains study/activates
    ing_syms = {s.name for s in cond.conditioning._symbols}
    assert "study" in ing_syms or "activates" in ing_syms
```

---

## Test 2: Rule with Explicit Probability (MARG Query)

**Grammar:** `define as verb with probability rule_body1_cond`

**SQUALL Input:**
```
define as Published with probability every Voxel conditioned to every Study activates.
```

**Expected IR:**
```python
Implication(
  head=published(v, ProbabilisticQuery(PROB, (v,))),
  antecedent=Condition(
    conditioned=voxel(v),
    conditioning=study(s) & activates(s)
  )
)
```

**Note:** The `ProbabilisticQuery(PROB, (v,))` marks this as a marginal/conditional query.
Later transformation by `TranslateProbabilisticQueryMixin.rewrite_conditional_query` expands
this into the three-rule conditional probability form.

**Test Code (test_squall_parser.py:724-742):**
```python
def test_rule_op_marg_produces_prob_query_in_head():
    """'with probability … conditioned to …' produces ProbabilisticQuery(PROB,...) in head."""
    from neurolang.probabilistic.expressions import Condition, ProbabilisticQuery, PROB

    result = parser(
        "define as Published with probability every Voxel "
        "conditioned to every Study activates."
    )
    assert isinstance(result, Implication)
    assert isinstance(result.antecedent, Condition)
    
    # Head must contain a ProbabilisticQuery(PROB, ...) argument
    head_args = result.consequent.args
    prob_args = [a for a in head_args if isinstance(a, ProbabilisticQuery)
                 and a.functor == PROB]
    assert prob_args
```

---

## Test 3: Aggregation with Arbitrary Functor

**Grammar:** `define as result every Custom_func of the Relation`

**SQUALL Input:**
```
define as Result every Create_overlay of the Prob_map.
```

**Expected IR:**
```python
Implication(
  head=result(FunctionApplication(Symbol('create_overlay'), (args...))),
  antecedent=prob_map(...)
)
```

**Test Code (test_squall_parser.py:746-765):**
```python
def test_ng1_agg_npc_arbitrary_functor():
    """'every Custom_func of the Relation' uses Symbol as aggregation functor."""
    from neurolang.expressions import Symbol, FunctionApplication
    from neurolang.datalog.expressions import AggregationApplication

    # Non-builtin aggregation function: create_overlay is not in _AGG_FUNC_MAP
    result = parser(
        "define as Result every Create_overlay of the Prob_map."
    )
    assert isinstance(result, Implication)
    
    head_args = result.consequent.args
    assert len(head_args) >= 1
    
    # The arg should be a FunctionApplication with functor Symbol("create_overlay")
    agg_arg = head_args[0]
    assert isinstance(agg_arg, (FunctionApplication, AggregationApplication))
    assert agg_arg.functor == Symbol("create_overlay")
```

---

## Test 4: Global Aggregation with Free Variable Collection

**Grammar:** `every noun of the npc` (no explicit per clause)

**SQUALL Input:**
```
define as Result every Create_overlay of the Prob_map.
```

**Behavior:**
When no explicit `per` dimension is given, `det_every` extracts **all free variables**
from the npc body and uses them as aggregation arguments (sorted by name).

**Test Code (test_squall_parser.py:768-790):**
```python
def test_det_every_agg_free_var_fallback():
    """Global aggregation: agg args are all free vars in npc body, not just one var."""
    from neurolang.expressions import Symbol, FunctionApplication
    from neurolang.datalog.expressions import AggregationApplication

    # Prob_map is a binary relation (x, p) — both should be agg args
    result = parser(
        "define as Result every Create_overlay of the Prob_map."
    )
    assert isinstance(result, Implication)
    
    head_args = result.consequent.args
    assert len(head_args) >= 1
    
    agg_arg = head_args[0]
    assert isinstance(agg_arg, (FunctionApplication, AggregationApplication))
    
    # When Prob_map introduces variables, agg_arg.args should have >= 1 arg
    # (the free variable(s) from the npc body)
    assert len(agg_arg.args) >= 1
```

**Determinism Note:**
Variables are sorted by name:
```python
agg_args = tuple(sorted(free_vars, key=lambda s: s.name))
```

This ensures:
- Reproducible argument order
- Same aggregation regardless of parse order
- Deterministic IR generation

---

## Test 5: Tuple Variable Info in Conditioned Rules

**SQUALL Input:**
```
define as probably Published every Voxel (?x; ?y; ?z) 
  conditioned to every Study activates.
```

**Expected IR:**
```python
Implication(
  head=published(x, y, z),  # Three separate arguments, not a tuple
  antecedent=Condition(...)
)
```

**Test Code (test_squall_parser.py:701-722):**
```python
def test_rule_body1_cond_prior_tuple_var_info():
    """rule_body1_cond_prior unpacks tuple _var_info into separate head args."""
    from neurolang.probabilistic.expressions import Condition

    result = parser(
        "define as probably Published every Voxel (?x; ?y; ?z) "
        "conditioned to every Study activates."
    )
    assert isinstance(result, Implication)
    assert isinstance(result.antecedent, Condition)
    
    head_args = result.consequent.args
    # head should have 3 args (x, y, z), not 1 arg that is a tuple
    assert len(head_args) == 3
    
    from neurolang.expressions import Symbol
    assert all(isinstance(a, Symbol) for a in head_args)
```

**Key Point:** Tuple `_var_info` is unpacked into separate head arguments, not kept as a tuple.

---

## Test 6: VP-Level Probabilistic Operators

### vpdo_explicit_prob_v1
```
Subject activates with probability 0.7
↓
Returns CPS function:
lambda x: ProbabilisticFact(Constant(0.7), activates(x))
```

### vpdo_prob_v1
```
Subject probably activates
↓
Returns CPS function:
lambda x: ProbabilisticFact(Symbol('_fresh_1234'), activates(x))
```

**Verification:** These are tested indirectly through complete sentence parsing.

---

## Test 7: N-ary Probabilistic Operators

### vpdo_explicit_prob_vn
```
Subject activates Object with probability 0.5
↓
Returns CPS function:
lambda x: ProbabilisticFact(Constant(0.5), _apply_ops(ops, activates, x))
```

### vpdo_prob_vn
```
Subject probably activates Object
↓
Returns CPS function:
lambda x: ProbabilisticFact(Symbol('_fresh_5678'), _apply_ops(ops, activates, x))
```

---

## Test 8: Rule-Level Operators

### rule_op_prob
**Grammar:** `define as verb with probability np rule_body1`

```python
def rule_op_prob(self, args):
    verb = items[0]
    np_prob = items[1]      # CPS noun phrase for probability
    body_result = items[2]
    
    # Extract body variables and formula
    head_args, body_formula = extract_from(body_result)
    head = verb(*head_args) if head_args else verb()
    
    # Extract probability from CPS noun phrase
    if callable(np_prob) and not isinstance(np_prob, (Symbol, Constant)):
        prob_val = np_prob(lambda x: x)  # Apply identity continuation
    else:
        prob_val = np_prob
    
    return Implication(ProbabilisticFact(prob_val, head), body_formula)
```

**Key:** The `np_prob` CPS continuation allows complex probability expressions.

### rule_opnn_per
**Grammar:** `define as probably verbn rule_body1 ops`

```python
def rule_opnn_per(self, args):
    verb = items[0]
    body_result = items[1]
    ops = items[2] if len(items) > 2 else None
    
    # Extract body variables
    body_args, body_formula = extract_from(body_result)
    head_args = list(body_args)
    
    # Extract ops variables
    all_body_parts = [body_formula]
    if ops is not None:
        _extract_datalog_body(ops, head_args)
    
    # Create fresh probability symbol
    fresh_prob = Symbol.fresh()
    
    consequent = verb(*head_args) if head_args else verb()
    full_body = Conjunction(all_body_parts) if len(all_body_parts) > 1 else all_body_parts[0]
    
    return Implication(ProbabilisticFact(fresh_prob, consequent), full_body)
```

**Key:** Always creates a fresh probability symbol, allowing the solver to assign it.

---

## Summary of Test Insights

1. **Conditioned Rules:** `Condition(conditioned, conditioning)` wraps the antecedent
2. **Marginal Queries:** `ProbabilisticQuery(PROB, vars)` marks probability queries
3. **Aggregation:** `_agg_info` attribute on noun groups triggers special handling
4. **Tuple Variables:** Unpacked into separate arguments in rule heads
5. **Fresh Symbols:** Used for `probably` keyword to defer probability resolution
6. **Free Variable Collection:** Extracted from npc bodies, sorted for determinism
7. **CPS Continuations:** Allow complex probability expressions via noun phrases

---

## How to Run These Tests

```bash
cd /Users/dwasserm/sources/NeuroLang/.worktrees/squall-cnl
python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_squall_conditioned_prior_produces_condition_node -xvs
python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_rule_op_marg_produces_prob_query_in_head -xvs
python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_ng1_agg_npc_arbitrary_functor -xvs
python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_det_every_agg_free_var_fallback -xvs
```

Or run all probabilistic-related tests:
```bash
python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py -k "prob\|cond\|agg" -xvs
```
