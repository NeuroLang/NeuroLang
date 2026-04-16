# SQUALL: MARG Queries and Arbitrary Aggregation — Fix Design

**Date:** 2026-04-16
**Branch:** squall-cnl
**Scope:** Two transformer-level fixes enabling:
1. **Gap A** — MARG/SUCC conditioned queries with tuple subjects (`with probability … conditioned to …`)
2. **Gap B** — Arbitrary function as aggregation functor; free-variable fallback when no explicit variables are given

---

## Motivating examples

```
# Rule 3 — MARG query
ProbMap(x, y, z, PROB) :- FocusReported(x, y, z, s) // (SelectedStudy(s) & OpenWorldStudies(s))

# SQUALL equivalent:
define as prob_map with probability every focus_reported (?x; ?y; ?z; ?s)
    conditioned to every selected_study ?s that open_world_studies.

# Rule 4 — global aggregation
ProbabilityImage(create_region_overlay(x, y, z, p)) :- ProbMap(x, y, z, p)

# SQUALL equivalent:
define as probability_image every create_region_overlay of the prob_map.
```

---

## Background

### MARG query IR shape

`TranslateProbabilisticQueryMixin.rewrite_conditional_query` (in
`neurolang/frontend/datalog/sugar/__init__.py`) fires on:

```python
Implication(head_with_PROB_arg, Condition(conditioned, conditioning))
```

where `head_with_PROB_arg` contains a `ProbabilisticQuery(PROB, (x, y, z))` term as one
of its arguments.  It rewrites this into three rules (numerator, denominator, final
conditional) and injects the probability as the last column.

The SQUALL transformer must therefore produce:
```python
Implication(
    prob_map(x, y, z, ProbabilisticQuery(PROB, (x, y, z))),
    Condition(focus_reported(x, y, z, s),
              Conjunction((selected_study(s), open_world_studies(s))))
)
```

### Aggregation IR shape

`TranslateToLogicWithAggregation` (in `neurolang/datalog/aggregation.py`) promotes any
`FunctionApplication` in a rule head to `AggregationApplication` automatically:

```python
# Transformer emits:
Implication(
    probability_image(Symbol("create_region_overlay")(x, y, z, p)),
    prob_map(x, y, z, p)
)
# Walker promotes to:
Implication(
    probability_image(AggregationApplication(Symbol("create_region_overlay"), (x,y,z,p))),
    prob_map(x, y, z, p)
)
```

So the SQUALL transformer only needs to emit `Symbol("create_region_overlay")(free_vars)`
in the head; the aggregation machinery handles the rest.

---

## Gap A — `with probability … conditioned to …`

### Grammar change

**File:** `neurolang/frontend/datalog/neurolang_natural.lark`

Add one new `rule_op` alternative:

```lark
rule_op : _rule_start [ PROBABLY ] verb1 rule_body1
        | _rule_start PROBABLY verb1 rule_body1_cond
        | _rule_start verb1 _WITH _PROBABILITY np rule_body1 -> rule_op_prob
        | _rule_start verb1 _WITH _PROBABILITY rule_body1_cond -> rule_op_marg
```

The new alternative `rule_op_marg` differs from `rule_op_prob` in two ways:
- No `np` (no explicit probability value — the probability is computed, not asserted)
- Uses `rule_body1_cond` (conditioned form) instead of `rule_body1`

### `rule_body1_cond_prior` / `rule_body1_cond_posterior` / `rule_body2_cond` fix

**File:** `neurolang/frontend/datalog/squall_syntax_lark.py`

All three currently extract a single variable from `ng1._var_info`.  Fix to unpack
tuple `_var_info` (for multi-dimensional subjects like `(?x; ?y; ?z)`), mirroring the
logic already in `rule_body1`:

```python
var_info = getattr(ng1, '_var_info', None)
if isinstance(var_info, tuple):
    head_args = list(var_info)
    conditioned_body = ng1(var_info)
else:
    x = var_info if var_info is not None else Symbol.fresh()
    head_args = [x]
    conditioned_body = ng1(x)
```

Apply the same pattern to `ng1_left` in `rule_body2_cond`.

### New `rule_op_marg` transformer method

**File:** `neurolang/frontend/datalog/squall_syntax_lark.py`

```python
def rule_op_marg(self, args):
    """Build a MARG query from ``define as verb with probability rule_body1_cond``.

    Emits Implication(verb(..., ProbabilisticQuery(PROB, head_vars)), Condition(...))
    so that TranslateProbabilisticQueryMixin.rewrite_conditional_query rewrites it
    into the standard three-rule conditional probability form.
    """
    items = [a for a in args if a is not None]
    verb = items[0]
    body_result = items[1]

    if isinstance(body_result, tuple) and body_result[0] == '_rule_body':
        head_args, body_formula = body_result[1]
    else:
        head_args, body_formula = [], Constant(True)

    prob_query_arg = ProbabilisticQuery(PROB, tuple(head_args))
    head = verb(*(head_args + [prob_query_arg]))
    return Implication(head, body_formula)
```

### New imports in `squall_syntax_lark.py`

```python
from ...probabilistic.expressions import Condition, ProbabilisticFact, ProbabilisticQuery
from ...probabilistic.expressions import PROB
```

(`Condition` and `ProbabilisticFact` are already imported; add `ProbabilisticQuery` and `PROB`.)

### SQUALL syntax

```
# Prior form (conditioned NP first):
define as prob_map with probability every focus_reported (?x; ?y; ?z; ?s)
    conditioned to every selected_study ?s that open_world_studies.

# Posterior form (conditioning NP last):
define as prob_map with probability every selected_study ?s that open_world_studies
    conditioned to every focus_reported (?x; ?y; ?z; ?s).
```

---

## Gap B — Arbitrary aggregation functor and free-variable fallback

### `ng1_agg_npc` change

**File:** `neurolang/frontend/datalog/squall_syntax_lark.py`

Current code enters the aggregation path only when `noun_name in _AGG_FUNC_MAP`.
Change: enter whenever an `npc` (`OF the …`) is present. When the noun is not in
`_AGG_FUNC_MAP`, use `Symbol(noun_name)` as the functor:

```python
# was:
if agg_func_from_noun is not None and npc is not None:
    ...

# becomes:
if npc is not None:
    agg_func = agg_func_from_noun if agg_func_from_noun is not None \
               else (Symbol(noun_name) if noun_name else None)
    if agg_func is not None:
        ng_agg._agg_info = (agg_func, npc, list(per_vars))
        if app is not None:
            ng_agg._var_info = app
        return ng_agg
```

### `det_every` free-variable fallback

**File:** `neurolang/frontend/datalog/squall_syntax_lark.py`

In the `_agg_info` handling branch of `det_every`, when `per_vars` is empty and no
explicit `app` variable is set, compute the aggregation arguments as all free variables
in the npc body formula:

```python
if agg_info is not None:
    agg_func_const, npc_cps, per_vars = agg_info

    # Build body to discover free variables
    q_box = []
    def capturing_cont(v, _cap=q_box):
        _cap.append(v)
        return Constant(True)
    npc_formula = npc_cps(capturing_cont)

    if per_vars:
        # Explicit groupby variables take precedence
        agg_args = tuple(per_vars)
    else:
        # Fall back to all free variables in the npc body
        from ...logic.expression_processing import extract_logic_free_variables
        free_vars = extract_logic_free_variables(npc_formula)
        # Sort by name for deterministic argument order
        agg_args = tuple(sorted(free_vars, key=lambda s: s.name)) \
                   if free_vars else (Symbol.fresh(),)

    agg_expr = _AggApp(agg_func_const, agg_args)
    d(agg_expr)
    return npc_formula
```

### SQUALL syntax

```
# Global aggregation (no per): aggregates over all free variables of prob_map
define as probability_image every create_region_overlay of the prob_map.

# Grouped aggregation (with per): aggregates max count per item (existing behaviour)
define as max_items for every Item ?i ;
    where every Max of the Quantity where ?i item_count per ?i.
```

---

## Files changed

| File | Change |
|------|--------|
| `neurolang/frontend/datalog/neurolang_natural.lark` | Add `rule_op_marg` alternative |
| `neurolang/frontend/datalog/squall_syntax_lark.py` | Add `rule_op_marg`; fix `rule_body1_cond_prior/posterior/rule_body2_cond` for tuple `_var_info`; fix `ng1_agg_npc` for arbitrary functor; fix `det_every` for free-variable fallback; add `ProbabilisticQuery`/`PROB` imports |
| `neurolang/frontend/datalog/tests/test_squall_parser.py` | New parser tests for both gaps |
| `neurolang/frontend/tests/test_squall_pdl_integration.py` | New integration tests for both gaps |
| `doc/tutorial_squall.rst` | Add examples for both constructs |

---

## Acceptance criteria

1. `define as prob_map with probability every focus_reported (?x; ?y; ?z; ?s) conditioned to every selected_study ?s that open_world_studies.` parses to `Implication(prob_map(x,y,z,s, ProbabilisticQuery(PROB,(x,y,z,s))), Condition(focus_reported(x,y,z,s), ...))`
2. `define as probability_image every create_region_overlay of the prob_map.` parses and walks to a rule with `AggregationApplication(Symbol("create_region_overlay"), free_vars)` in the head
3. Existing aggregation tests (`test_execute_squall_aggregation`) still pass
4. Existing conditioned-rule tests (`test_squall_conditioned_prior_produces_condition_node`) still pass
5. `uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py -k "not slow"` — all pass
6. `uv run python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py` — all pass
7. Tutorial doctests pass
