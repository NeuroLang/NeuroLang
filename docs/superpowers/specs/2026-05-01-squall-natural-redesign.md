# Design: SQUALL Naturalness Extensions D–H + Renames

**Date:** 2026-05-01  
**Branch:** squall-cnl  
**Files affected:**
- `neurolang/frontend/datalog/neurolang_natural.lark`
- `neurolang/frontend/datalog/squall_syntax_lark.py`
- `neurolang/frontend/tests/test_squall_syntax_lark.py`
- `neurolang/frontend/tests/test_squall_pdl_integration.py`
- `examples/plot_squall_cbma_spatial_prior.py`

---

## Goal

Transform the working SQUALL program (post-Approach-C) into maximally natural English.

**Current script:**

```
define as Voxel_reported with a probability of
    the Agg_max_proximity of the Focus_reported (?i2; ?j2; ?k2; ?s)
        per ?i1, ?j1, ?k1 and per ?s
        where (?i1; ?j1; ?k1) is a Voxel
        and such that EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 5.

define as Term_association every Term_in_study_tfidf (?s; ?t; _)
    such that ?s is a Selected_study.

define as Activation every Voxel_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

define as Probmap with probability every Activation (?i; ?j; ?k; _)
    conditioned to every Term_association (_; ?t) such that ?t is 'emotion'.

define as Img every Agg_create_region_overlay of the Probmap (?i; ?j; ?k; ?p).
```

**Target script (Approach D+E+F+G+H + renames):**

```
define as Reported_voxel with a probability of
    the Kernelized_max_proximity of the Reported_focus (?i2; ?j2; ?k2; ?s)
        for each ?i1, ?j1, ?k1 and for each ?s
        where (?i1; ?j1; ?k1) is a Voxel
        and where EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 5.

define as Study_term every Term_in_study_with_tfidf (?s; ?t; _)
    where ?s is a Selected_study.

define as Active_voxel every Reported_voxel (?i; ?j; ?k; ?s)
    where ?s is a Selected_study.

define as Activation_map with inferred probability every Active_voxel (?i; ?j; ?k; _)
    given every Study_term (_; ?t) where ?t is 'emotion'.

obtain the Brain_image of the Activation_map (?i; ?j; ?k; ?p) as Image.
```

Five independent grammar/transformer extensions (D–H) plus a rename pass in the example file.

---

## Rename Table

| Old Python name | New Python name | New SQUALL name | Notes |
|---|---|---|---|
| `focus_reported` | `reported_focus` | `Reported_focus` | adj-noun order |
| `term_in_study_tfidf` | `term_in_study_with_tfidf` | `Term_in_study_with_tfidf` | spell out tfidf |
| `selected_study` | `selected_study` | `Selected_study` | unchanged |
| `voxel` | `voxel` | `Voxel` | unchanged |
| `agg_max_proximity` | `kernelized_max_proximity` | `Kernelized_max_proximity` | drop `agg_` prefix |
| `agg_create_region_overlay` | `brain_image` | `Brain_image` | what it produces |
| `EUCLIDEAN` | `EUCLIDEAN` | `EUCLIDEAN` | ⚠ spatial-sugar match; must not change |
| `Voxel_reported` (IDB) | — | `Reported_voxel` | adj-noun order |
| `Term_association` (IDB) | — | `Study_term` | |
| `Activation` (IDB) | — | `Active_voxel` | |
| `Probmap` (IDB) | — | `Activation_map` | |
| `Img` (IDB) | — | dropped | replaced by `obtain … as Image` |

Renames are confined to `examples/plot_squall_cbma_spatial_prior.py`. No other files reference these names.

---

## Extension D — `for each` alias for `per`

### Motivation

`per ?i1, ?j1, ?k1 and per ?s` is technically correct but not idiomatic English. `for each ?i1, ?j1, ?k1 and for each ?s` reads naturally.

### Grammar change

`_EACH` terminal does not yet exist. Add it, then add `_FOR _EACH` alternatives to the `dim` rule:

```lark
# Add terminal (alongside _FOR which already exists at line 278)
_EACH : "each"

# dim rule — add FOR EACH alternatives before existing PER alternatives
dim : (_PER | _FOR _EACH) ng2                         -> dim_ng2
    | (_PER | _FOR _EACH) npc{THE} ("," npc{THE})+    -> dim_npc_list
    | (_PER | _FOR _EACH) npc{THE}                    -> dim_npc
    | agg_func _OF npc{THE}                            -> dim_agg
```

`_EACH` must be added to the keyword-exclusion regex so it cannot appear as an identifier. In `neurolang_natural.lark`, the exclusion regex is built from listed terminal names at lines 319–321. Add `_EACH` there.

### Transformer changes

None. `dim_ng2`, `dim_npc_list`, `dim_npc` already exist and handle the args correctly. The alternative preamble tokens (`_PER` vs `_FOR _EACH`) are transparent terminals and do not appear as `args`.

### Tests

- Unit: `for each ?i, ?j, ?k and for each ?s` produces identical `('_dims', [...])` structure as `per ?i, ?j, ?k and per ?s`.
- Integration: rule using `for each` groupby produces same `Implication` head args as equivalent `per` form.

---

## Extension E — `where` as alias for `such that`

### Motivation

`such that ?s is a Selected_study` is a formal logic phrasing. `where ?s is a Selected_study` is how English speakers naturally restrict a set.

### Grammar change

`_WHERE` is already a terminal (used in `rel_vp`, `rel_vpn`, `rel_fun_call`, `rel_tuple_noun`). Add it as an additional alternative for `rel_s` in `rel_b`:

```lark
# Before (line 103):
| _SUCH _THAT s  -> rel_s

# After:
| _SUCH _THAT s  -> rel_s
| _WHERE s       -> rel_s
```

**Disambiguation:** The existing `_WHERE vp` alternative (`rel_vp`, line 99) begins with `_WHERE` followed by a VP (verb phrase, starting with a verb or auxiliary). The new `_WHERE s` begins with `_WHERE` followed by a full sentence `s`, which starts with an NP (variable `?x`, a function call `FUNC(…)`, a determiner `every/a/the`, or a proper noun). Lark's Earley parser handles the ambiguity correctly because `vp` and `s` (which starts with `np vp`) have non-overlapping first sets at the token level when `_WHERE` is followed by `?`-prefixed variables or uppercase identifiers. Verified to be unambiguous in the existing grammar for all test cases.

### Transformer changes

None. The grammar alias `rel_s` already maps to the existing `rel_s` transformer method:

```python
def rel_s(self, args):
    s = args[0]
    return ('_rel', lambda x: s)
```

### Tests

- Unit: `where ?s is a Selected_study` produces same `('_rel', ...)` tagged tuple as `such that ?s is a Selected_study`.
- Unit: `and where EUCLIDEAN(…) is lower than 5` produces same IR as `and such that EUCLIDEAN(…) is lower than 5`.
- Regression: existing `_WHERE vp` alternatives (`rel_vp`, `rel_tuple_noun`, `rel_fun_call`) still parse correctly.
- Integration: full target script with all `such that` replaced by `where` solves to same result.

---

## Extension F — `given` as alias for `conditioned to`

### Motivation

`conditioned to every Term_association` is probabilistic-logic jargon. `given every Study_term` is natural English for conditional probability.

### Grammar change

`_GIVEN` is already defined as a terminal (line 280) but unused in any rule. Wire it into the three `rule_body1_cond` / `rule_body2_cond` alternatives:

```lark
# Before (lines 36–39):
rule_body1_cond : det ng1 _CONDITIONED _TO s        -> rule_body1_cond_prior
                | s _CONDITIONED _TO det ng1        -> rule_body1_cond_posterior

rule_body2_cond : det ng1 _CONDITIONED _TO det ng1

# After:
rule_body1_cond : det ng1 (_CONDITIONED _TO | _GIVEN) s       -> rule_body1_cond_prior
                | s (_CONDITIONED _TO | _GIVEN) det ng1       -> rule_body1_cond_posterior

rule_body2_cond : det ng1 (_CONDITIONED _TO | _GIVEN) det ng1
```

`_GIVEN` must be added to the keyword-exclusion regex (same location as `_EACH` above) — it is already defined as a terminal so it likely already excludes itself from `LOWER_NAME`, but this must be verified.

Also applies to `rule_op_marg` which references `rule_body1_cond` / `rule_body2_cond`:

```lark
rule_op : ...
        | _rule_start verb1 _WITH _PROBABILITY rule_body1_cond -> rule_op_marg
        | _rule_start verb1 _WITH _PROBABILITY rule_body2_cond -> rule_op_marg
```

No change needed to `rule_op` itself — it already delegates to `rule_body1_cond`.

### Transformer changes

None. `rule_body1_cond_prior`, `rule_body1_cond_posterior`, `rule_body2_cond` transformer methods already exist and produce `Condition(conditioned, conditioning)`. The grammar terminals `_CONDITIONED _TO` vs `_GIVEN` are transparent and not passed as `args`.

### Tests

- Unit: `given every Study_term (_; ?t) where ?t is 'emotion'` parsed via `rule_body1_cond_prior` produces same `Condition(…)` as `conditioned to every …`.
- Regression: existing `conditioned to` form continues to parse correctly.

---

## Extension G — `obtain … as Name`

### Motivation

`define as Img every Agg_create_region_overlay of the Probmap (?i; ?j; ?k; ?p).` followed by a separate retrieval step is mechanical boilerplate. `obtain the Brain_image of the Activation_map (?i; ?j; ?k; ?p) as Image.` defines the relation and names it for retrieval in one sentence.

### Grammar change

Extend the `query` rule with a `query_as` alternative:

```lark
# Before (line 19):
query : _OBTAIN ops

# After:
query : _OBTAIN ops                    -> query_unnamed
      | _OBTAIN ops _AS identifier     -> query_as
```

`_AS` terminal does not yet exist. Add it:

```lark
_AS : "as"
```

`_AS` must also be added to the keyword-exclusion regex.

The existing `query : _OBTAIN ops` becomes `query_unnamed`. The existing `query` transformer method must be renamed to `query_unnamed`.

### Transformer changes

**Rename existing method:**

```python
# Before:
def query(self, args):
    ...

# After:
def query_unnamed(self, args):
    # unchanged body
    ops = args[0]
    if not callable(ops) or isinstance(ops, (Symbol, Constant)):
        x = Symbol.fresh()
        return ('_query', Query(x, ops if not isinstance(ops, Symbol) else ops(x)))
    formula = ops(lambda x: x)
    return ('_query', _cps_formula_to_query(formula))
```

**New method:**

```python
def query_as(self, args):
    """Handle 'obtain ops as Name' — define an IDB rule and expose result.

    Emits an Implication whose head is Name(*free_vars) and whose body is
    the materialised ops formula, then returns a ('_query', Query(...)) tagged
    tuple so the squall program builder adds it to the query set.
    """
    ops, name_sym = args[0], args[1]   # name_sym: Symbol from `identifier`

    # Materialise ops (CPS NP) to extract free variables and body formula.
    free_vars = []
    body_formulas = []

    def capturing_cont(v, *extra):
        if isinstance(v, Symbol):
            free_vars.append(v)
        if extra:
            body_formulas.extend(extra)
        return Constant(True)

    if callable(ops) and not isinstance(ops, (Symbol, Constant)):
        result = ops(capturing_cont)
    else:
        result = ops

    # Strip trivial True wrappers.
    body = result
    while isinstance(body, ExistentialPredicate):
        body = body.body
    if (
        isinstance(body, Conjunction)
        and len(body.formulas) == 2
        and body.formulas[1] == Constant(True)
    ):
        body = body.formulas[0]

    head = name_sym(*free_vars) if free_vars else name_sym()
    impl = Implication(head, body)

    # Also register as a query so solve_all returns it.
    query_expr = Query(Symbol.fresh(), head)
    return ('_query_as', (impl, query_expr, name_sym))
```

The program builder (wherever `('_query', ...)` is processed in the SQUALL-to-program translation layer) must also handle `('_query_as', (impl, query_expr, name_sym))` by:
1. Adding `impl` to the IDB.
2. Registering `query_expr` as a query.

The exact location is in `squall_syntax_lark.py`'s `squall` or `start` method (or wherever tagged tuples are consumed after parsing). This must be verified during implementation.

### Tests

- Unit: `obtain the Brain_image of the Activation_map (?i; ?j; ?k; ?p) as Image` produces a `('_query_as', ...)` tagged tuple where the `Implication` head functor is `Symbol('image')` and it has 4 args.
- Integration: the target script ending with the `obtain … as Image` sentence produces `solution["image"]` in `solve_all()` output.
- Regression: bare `obtain ops` (without `as`) continues to work via `query_unnamed`.

---

## Extension H — `with inferred probability`

### Motivation

`with probability` in `define as Activation_map with probability every Active_voxel conditioned to …` sounds like a directly-assigned probability. The rule actually computes a **MARG** conditional probability (inferred from the probabilistic program). `with inferred probability` makes the semantic distinction explicit.

### Grammar change

`_INFERRED` terminal does not yet exist. Add it, then make it optional in the `rule_op_marg` alternatives:

```lark
# Add terminal
_INFERRED : "inferred"

# Before (lines 28–29):
| _rule_start verb1 _WITH _PROBABILITY rule_body1_cond -> rule_op_marg
| _rule_start verb1 _WITH _PROBABILITY rule_body2_cond -> rule_op_marg

# After:
| _rule_start verb1 _WITH _INFERRED? _PROBABILITY rule_body1_cond -> rule_op_marg
| _rule_start verb1 _WITH _INFERRED? _PROBABILITY rule_body2_cond -> rule_op_marg
```

`_INFERRED` must be added to the keyword-exclusion regex.

### Transformer changes

None. `rule_op_marg` transformer method receives `rule_body1_cond`/`rule_body2_cond` result as args regardless of whether `_INFERRED` appeared. The optional terminal is transparent.

### Tests

- Unit: `with inferred probability every Active_voxel conditioned to …` parses via `rule_op_marg` and produces same `Implication` as `with probability every Active_voxel conditioned to …`.
- Regression: `with probability` (without `inferred`) continues to parse correctly.

---

## Final example script (Approach D+E+F+G+H + renames)

```
define as Reported_voxel with a probability of
    the Kernelized_max_proximity of the Reported_focus (?i2; ?j2; ?k2; ?s)
        for each ?i1, ?j1, ?k1 and for each ?s
        where (?i1; ?j1; ?k1) is a Voxel
        and where EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 5.

define as Study_term every Term_in_study_with_tfidf (?s; ?t; _)
    where ?s is a Selected_study.

define as Active_voxel every Reported_voxel (?i; ?j; ?k; ?s)
    where ?s is a Selected_study.

define as Activation_map with inferred probability every Active_voxel (?i; ?j; ?k; _)
    given every Study_term (_; ?t) where ?t is 'emotion'.

obtain the Brain_image of the Activation_map (?i; ?j; ?k; ?p) as Image.
```

---

## Implementation order

1. **Extension E** (`where` as `such that`) — pure grammar, no new terminals, lowest risk.
2. **Extension F** (`given`) — wire existing terminal; no transformer change.
3. **Extension H** (`with inferred probability`) — new terminal, optional token in existing rule.
4. **Extension D** (`for each`) — new terminal, grammar mirrors existing `per` alternatives.
5. **Extension G** (`obtain … as`) — new terminal, new transformer method, program-builder change.
6. **Renames** — update `plot_squall_cbma_spatial_prior.py` Python symbol names and SQUALL program strings.

Each extension is independently mergeable. Extensions E–D have no transformer changes; only G requires new transformer logic.

---

## New terminals summary

| Terminal | String | Used in |
|---|---|---|
| `_EACH` | `"each"` | `dim` rule (Extension D) |
| `_AS` | `"as"` | `query` rule (Extension G) |
| `_INFERRED` | `"inferred"` | `rule_op_marg` (Extension H) |

`_GIVEN` already defined (line 280) — no new terminal needed for Extension F.

All three new terminals must be added to the keyword-exclusion regex so they cannot appear as predicate or variable identifiers.

---

## Testing strategy

Each extension gets:
- A **unit parse test** in `test_squall_syntax_lark.py`: parse a minimal SQUALL snippet, assert the resulting `Implication` or tagged-tuple structure matches expected IR.
- An **integration test** in `test_squall_pdl_integration.py`: load minimal EDB, run `execute_squall_program` + `solve_all`, assert result matches the pre-rename, pre-extension form.

Regression: existing full test suite must continue to pass after each extension.
