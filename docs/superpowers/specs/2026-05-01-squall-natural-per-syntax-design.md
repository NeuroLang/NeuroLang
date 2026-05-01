# Design: More Natural SQUALL Syntax for Aggregation Rules

**Date:** 2026-05-01  
**Branch:** squall-cnl  
**Files affected:**
- `neurolang/frontend/datalog/neurolang_natural.lark`
- `neurolang/frontend/datalog/squall_syntax_lark.py`
- `neurolang/frontend/tests/test_squall_syntax_lark.py` (new tests)
- `neurolang/frontend/tests/test_squall_pdl_integration.py` (new tests)
- `examples/plot_squall_cbma_spatial_prior.py` (updated sentence)

---

## Goal

The existing SQUALL sentence for probabilistic voxel proximity is verbose and technical:

```
define as Voxel_reported with a probability of
    the Agg_max_proximity of the Focus_reported (?i2; ?j2; ?k2; ?s)
        per ?i1 and per ?j1 and per ?k1 and per ?s
        that voxel(?i1, ?j1, ?k1) holds
        and such that ?d is equal to EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2)
        and ?d is lower than 5.
```

The target sentence (Approach C) reads naturally:

```
define as Voxel_reported with a probability of
    the Agg_max_proximity of the Focus_reported (?i2; ?j2; ?k2; ?s)
        per ?i1, ?j1, ?k1 and ?s
        where (?i1; ?j1; ?k1) is a Voxel
        and such that EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 5.
```

Three independent grammar/transformer extensions implement this. B (inline comparison) already works through existing paths; A (compact `per`) and C (tuple membership) each require a small grammar rule and transformer method.

---

## Extension A — Compact `per` list

### Motivation

`per ?i1 and per ?j1 and per ?k1 and per ?s` repeats the keyword four times. English naturally uses `per ?i1, ?j1, ?k1 and ?s`.

### Grammar change

In `neurolang_natural.lark`, add `dim_npc_list` as a new alternative to the `dim` rule:

```lark
# Before
dim : _PER ng2              -> dim_ng2
    | _PER npc{THE}         -> dim_npc
    | agg_func _OF npc{THE} -> dim_agg

# After
dim : _PER ng2                              -> dim_ng2
    | _PER npc{THE}                         -> dim_npc
    | _PER npc{THE} ("," npc{THE})+         -> dim_npc_list
    | agg_func _OF npc{THE}                 -> dim_agg
```

The trailing `and` in `per ?i1, ?j1, ?k1 and ?s` is handled by the existing `dims_rec` rule which accepts `_CONJUNCTION` (already includes `_AND`) between consecutive `dim` nodes. So `per ?i1, ?j1, ?k1` becomes one `dim_npc_list` node and `and ?s` cannot parse unless `?s` is preceded by a new `per`. Two options:

**Option A1 (recommended):** `per ?i1, ?j1, ?k1, ?s` — all in one `dim_npc_list`, final `and ?s` becomes `and per ?s` (keep the last `per`):
```
per ?i1, ?j1, ?k1 and per ?s
```
This is unambiguous — one `dim_npc_list` node for `(?i1, ?j1, ?k1)` and one `dim_npc` node for `?s`, joined by `_CONJUNCTION`.

**Option A2:** Allow the final item to drop `per` by adding `("," | _AND) npc{THE}` to the list tail in the grammar. More natural but slightly more grammar complexity.

**Recommended: Option A1** — least grammar surface, still significantly shorter than four `per`s.

Target sentence fragment: `per ?i1, ?j1, ?k1 and per ?s`

### Transformer changes

```python
# squall_syntax_lark.py

def dim_npc_list(self, args):
    """Handle 'per ?i, ?j, ?k' — multiple per-variables under one per keyword."""
    per_syms = []
    for npc in args:
        if isinstance(npc, (Symbol, Constant)):
            per_syms.append(npc)
        elif callable(npc) and not isinstance(npc, (Symbol, Constant)):
            result = npc(lambda x: x)
            per_syms.append(result if isinstance(result, (Symbol, Constant)) else Symbol.fresh())
        else:
            per_syms.append(Symbol.fresh())
    return [('_per', s) for s in per_syms]  # list of tagged tuples

def dims_base(self, args):
    d = args[0]
    return ('_dims', d if isinstance(d, list) else [d])  # handle list-returning dim

def dims_rec(self, args):
    dim, rest = args[0], args[1]
    rest_list = rest[1] if (isinstance(rest, tuple) and rest[0] == '_dims') else [rest]
    dim_list = dim if isinstance(dim, list) else [dim]
    return ('_dims', dim_list + rest_list)
```

No changes needed in `ng1_agg_npc` or `rule_op_prob_agg` — they iterate `('_dims', [...])` and collect every `('_per', sym)` entry.

### Tests

- `dim_npc_list` produces the same per-vars as equivalent individual `dim_npc` nodes.
- `ng1_agg_npc` with compact list produces the correct head args in the emitted `Implication`.
- Integration test: parse-and-solve a rule with `per ?i, ?j, ?k and per ?s` produces the same result as the verbose form.

---

## Extension B — Inline expression comparison (no relay variable)

### Motivation

`?d is equal to EUCLIDEAN(…) and ?d is lower than 5` requires a relay variable. The form `EUCLIDEAN(…) is lower than 5` is more direct.

### Grammar change

**None required.** The path `such that s` → `rel_s` → `s_np_vp` with NP = `expr_atom_fun_upper` and VP = `vpbe_rel(rel_comp)` already handles this. `EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 5` parses today as:

1. `s` → `s_np_vp`: subject NP = function call expression, VP = `vp_aux(be, vpbe_rel(rel_comp(lower, 5)))`
2. `rel_comp` receives `x = FunctionApplication(EUCLIDEAN, …)` from the NP and constructs `lt(EUCLIDEAN(…), 5)`
3. `rel_s` wraps it: `('_rel', lambda _: lt(EUCLIDEAN(…), 5))`
4. Conjoined into the rule body normally.

### Transformer changes

None. Verify by running the existing test suite and adding a targeted test.

### Tests

- Unit test: `parser("… and such that EUCLIDEAN(?i1,?j1,?k1,?i2,?j2,?k2) is lower than 5.")` produces the same IR as the relay-variable form.
- Integration test: solve a minimal program using the inline form; result matches the relay-variable form.

---

## Extension C — Tuple membership clause

### Motivation

`that voxel(?i1, ?j1, ?k1) holds` uses the function-call-holds pattern. `where (?i1; ?j1; ?k1) is a Voxel` reads naturally as English.

### Grammar change

Add `rel_tuple_noun` to `rel_b` in `neurolang_natural.lark`:

```lark
# Existing relevant alternatives
rel_b : (_THAT | _WHICH | _WHERE | _WHO ) vp                -> rel_vp
      | ...
      | (_THAT | _WHICH | _WHERE ) identifier "(" label ("," label)* ")" _HOLDS -> rel_fun_call
      | ...

# New alternative (add after rel_fun_call)
      | _WHERE label (_IS | _ARE) (_A | _AN) noun1           -> rel_tuple_noun
```

The `label` rule already handles `"(" label_tuple_item (";" label_tuple_item)* ")"` (semicolon-separated, consistent with existing tuple label syntax). So the surface form is `where (?i1; ?j1; ?k1) is a Voxel`.

This alternative is unambiguous with existing `_WHERE` alternatives: none of the existing `rel_b` `_WHERE` forms expect a `label` (i.e., a token starting with `?` or `(`) as the immediate next token after `where`.

### Transformer method

```python
def rel_tuple_noun(self, args):
    """Handle 'where (?i; ?j; ?k) is a Noun' → Noun(i, j, k)."""
    label_cps = args[0]   # CPS lambda from `label` handler
    noun = args[1]        # Symbol from `noun1` (already lowercased)

    raw = label_cps(lambda x: x)
    tup = raw if isinstance(raw, tuple) else (raw,)

    body_syms = tuple(
        item.as_symbol() if isinstance(item, _AnonymousVar) else item
        for item in tup
    )
    return ('_rel', lambda x: noun(*body_syms))
```

The enclosing subject `x` is ignored (same pattern as `rel_s` and `rel_fun_call`).

### IR produced

`where (?i1; ?j1; ?k1) is a Voxel` → `FunctionApplication(Symbol('voxel'), (i1, j1, k1))` added to the body conjunction.

### Tests

- `rel_tuple_noun` single tuple: `where (?i; ?j; ?k) is a Voxel` → `voxel(i, j, k)`.
- `rel_tuple_noun` single-element: `where ?x is a Study` → `study(x)`.
- Lark does not confuse `rel_tuple_noun` with `rel_vp` (no ambiguity).
- Integration test: the target sentence parses and solves correctly.

---

## Final example sentence (Approach C)

After all three extensions:

```
define as Voxel_reported with a probability of
    the Agg_max_proximity of the Focus_reported (?i2; ?j2; ?k2; ?s)
        per ?i1, ?j1, ?k1 and per ?s
        where (?i1; ?j1; ?k1) is a Voxel
        and such that EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 5.
```

The old sentence that gets replaced:

```
define as Voxel_reported with a probability of
    the Agg_max_proximity of the Focus_reported (?i2; ?j2; ?k2; ?s)
        per ?i1 and per ?j1 and per ?k1 and per ?s
        that voxel(?i1, ?j1, ?k1) holds
        and such that ?d is equal to EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2)
        and ?d is lower than 5.
```

---

## Implementation order

1. **Extension B (verify)** — run tests; add inline comparison unit + integration test. No code changes expected.
2. **Extension A** — grammar `dim_npc_list`, patch `dims_base`/`dims_rec`/add `dim_npc_list` in transformer; tests.
3. **Extension C** — grammar `rel_tuple_noun`, add transformer method; tests.
4. **Update example** — change the sentence in `plot_squall_cbma_spatial_prior.py` to Approach C form.
5. **Update docstring** in same file.

Each extension is independently mergeable.

---

## Testing strategy

Each extension gets:
- A **unit parse test** in `test_squall_syntax_lark.py`: parse a minimal SQUALL snippet, assert the resulting `Implication` structure matches the expected IR.
- An **integration test** in `test_squall_pdl_integration.py`: load minimal EDB, run `execute_squall_program` + `solve_all`, assert the result matches the verbose-syntax equivalent.

Regression: the existing full test suite must continue to pass.
