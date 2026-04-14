# SQUALL Improvements + Tutorial Design

**Date:** 2026-04-14  
**Branch:** squall-cnl  
**Scope:** Quick-win code improvements + Sphinx RST tutorial with pytest-runnable examples

---

## Background

The SQUALL (Semantically controlled Query-Answerable Logical Language) frontend translates
controlled English sentences into NeuroLang logical IR. The implementation in
`squall_syntax_lark.py` / `squall.py` / `neurolang_natural.lark` is functionally correct
for the implemented subset, but contains dead code, duplicate constants, and zero user-facing
documentation.

A thorough analysis (see exploration notes) identified issues across four categories:
correctness, quality, robustness, and feature gaps. This spec covers **quick-win quality
improvements only** (zero behaviour change) plus a **Sphinx RST tutorial** with examples
testable via `pytest --doctest-glob`.

---

## Part 1 — Quick-Win Code Improvements

### 1.1 Module-level aggregation constant

**Problem:** Two near-identical dicts exist in `ng1_agg_npc` and `agg_func` — defined inside
method bodies, with one extra dead entry (`'mean'`, not a valid grammar token):

```python
# ng1_agg_npc (inside method):
_agg_name_map = { 'count': Constant(len), 'sum': Constant(sum),
                  'max': Constant(max), 'min': Constant(min),
                  'average': Constant(np.mean), 'mean': Constant(np.mean) }

# agg_func (inside method):
_agg_map = { 'count': Constant(len), 'sum': Constant(sum),
             'max': Constant(max), 'min': Constant(min),
             'average': Constant(np.mean) }
```

**Fix:** Extract a single module-level constant `_AGG_FUNC_MAP` (without `'mean'`, which is
unreachable from the grammar's `AGG_FUNC` terminal). Both methods reference this constant.
The `import numpy as np` call that was inside these methods is moved to the top of the module.
The dead `import builtins` line (also inside these methods, never used) is deleted entirely.

**Files:** `squall_syntax_lark.py`  
**Behaviour change:** None  
**Tests required:** None (existing 38 pass)

---

### 1.2 Delete `_flatten_npc_body` dead function

**Problem:** `_flatten_npc_body` is defined at module level (lines ~1176–1194) but has zero
call sites anywhere in the codebase. It was created during aggregation development and
superseded.

**Fix:** Delete the function and its docstring.

**Files:** `squall_syntax_lark.py`  
**Behaviour change:** None  
**Tests required:** None

---

### 1.3 Remove dead `ops_var` variable in `rule_opnn`

**Problem:** In `rule_opnn`, a fresh symbol is created and immediately abandoned:

```python
ops_var = Symbol.fresh()   # never used again
ops_body = ops(lambda x: x)
if isinstance(ops_body, (Symbol, Constant)):
    head_args.append(ops_body)
else:
    ops_extracted = _extract_datalog_body(ops, head_args)
    all_body_parts.extend(ops_extracted)
```

The `ops_var` line is dead. The `ops_body` branch also double-invokes the CPS NP (once here,
once inside `_extract_datalog_body`), but the `Symbol/Constant` fast path is redundant because
`_extract_datalog_body` handles it correctly via `collect_scope`. The entire
`ops_body = ... / if isinstance ...` block can be replaced with an unconditional
`_extract_datalog_body` call.

**Fix:** Remove `ops_var = Symbol.fresh()`. Replace the `ops_body` if/else block with a direct
call: `all_body_parts.extend(_extract_datalog_body(ops, head_args))`.

**Files:** `squall_syntax_lark.py`  
**Behaviour change:** None (the fast-path for Symbol/Constant was handled identically by
`_extract_datalog_body` via `collect_scope` which just appends x to head_args)  
**Tests required:** None

---

### 1.4 Move `Disjunction` import to module top level

**Problem:** `Disjunction` is imported inside the `bool_disjunction` method body — the only
place it is used. All other logic imports are at module top level.

**Fix:** Add `Disjunction` to the existing `from ...logic import ...` block at the top of the
file. Remove the inline `from ...logic import Disjunction` inside the method.

**Files:** `squall_syntax_lark.py`  
**Behaviour change:** None  
**Tests required:** None

---

### 1.5 Add docstrings to key transformer methods

**Problem:** `SquallTransformer` has ~50 methods; none have per-method docstrings. The most
complex ones are entirely opaque to new contributors.

**Fix:** Add brief (1–4 line) docstrings to these methods:
- `rule_opnn` — explain CPS body extraction and head construction
- `ng1_agg_npc` — explain noun-as-aggregation-function + `_agg_info` attribute contract
- `det_every` — explain `_agg_info` aggregation path vs. standard universal path
- `rel_ng2` — explain `whose NG2 VP` → `∃y. ng2(x,y) ∧ vp(y)`
- `_extract_datalog_body` — explain side-effect on `head_args` + return value
- `_flatten_to_datalog` — clarify what it strips (quantifiers) and what it preserves

Also add a **KNOWN STUBS** section to the module docstring noting:
- `rule_body1_cond` / `rule_body1_cond_prior/posterior` — conditioned rules not implemented
- `rule_body2_cond` — no transformer handler (falls to `_default`)
- `~` prefix (inverse transitive) — parsed but semantically inert (`_inverse` flag never read)

**Files:** `squall_syntax_lark.py`  
**Behaviour change:** None  
**Tests required:** None

---

## Part 2 — RST Tutorial

### 2.1 File location

`doc/tutorial_squall.rst`

Added to toctree in `doc/index.rst` after `tutorial`:
```rst
   tutorial
   tutorial_squall
   auto_examples/index
```

### 2.2 Doctest mechanism

`sphinx.ext.doctest` is not enabled in `doc/conf.py`. The existing tutorials use
`.. code-block:: python` (non-executable). We use **pytest `--doctest-glob`** instead:

- Examples are written as standard Python `>>>` doctests inside `.. code-block:: python`
  sections — readable as prose in Sphinx HTML, runnable by pytest
- `pyproject.toml` gains one line in `[tool.pytest.ini_options]`:
  ```toml
  addopts = "--doctest-glob=doc/*.rst"
  ```
- All `>>>` examples that produce non-deterministic repr (fresh symbol names like
  `S{fresh_00000002}`) use `# doctest: +ELLIPSIS` and match only the stable parts, or use
  `assert` statements instead of output comparison
- A `.. testsetup:: *` block at the top of the RST provides shared imports so each example
  section is self-contained

### 2.3 Tutorial structure

| Section | Title | Doctest examples |
|---------|-------|-----------------|
| 1 | What is SQUALL? | Prose only |
| 2 | Installing and importing | `from neurolang.frontend.datalog.squall_syntax_lark import parser` |
| 3 | Basic sentences: free variables | `parser("squall ?s reports")` |
| 4 | Quantifiers: every, a/an, no | `every person plays`, `a person plays`, `no person plays` |
| 5 | Relative clauses | `every person that plays runs` |
| 6 | Multi-word (tuple) subjects | `every voxel (?x; ?y; ?z) ...` |
| 7 | Defining rules with `define as` | `define as Active every person that plays.` |
| 8 | Multi-variable rules and joins | `define as merge for every Item ?i ; with every Quantity...` |
| 9 | Filtering with comparisons | `define as Large every Item that has an item_count greater equal than 2.` |
| 10 | Querying with `obtain` | `obtain every Item that has an item_count.` |
| 11 | Aggregations | `define as max_items for every Item ?i ; where every Max...` |
| 12 | Running programs end-to-end | Full `parser()` → `engine.walk()` → `Chase` → `solution` example |
| 13 | Reserved words and quoting | Backtick quoting, `?` labels, `'string literals'` |
| 14 | Known limitations | Stub features (conditioned rules, `~` inversion), Earley ambiguity note |

### 2.4 Non-determinism strategy

Fresh symbols in repr are avoided by:
1. Using `assert isinstance(result, SomeType)` checks
2. Using `assert "keyword" in repr(result)` where keyword is stable
3. Using `# doctest: +ELLIPSIS` with `...` for variable parts like fresh symbol names
4. Using `result.value` on chase solutions (plain Python sets, fully deterministic)

---

## Part 3 — What is NOT in scope

The following identified issues are **explicitly deferred** to a future correctness pass:

- `vpdo_v1` silently dropping the `cp` complement phrase argument
- `rule_opnn_per` discarding body predicates from `_extract_datalog_body`
- `_flatten_to_datalog` including consequent predicates in body extraction
- `~` inversion (`_inverse` attribute) semantic implementation
- `rule_body1_cond` / `rule_body2_cond` stub implementations
- User-friendly error messages wrapping Lark parse exceptions
- Engine rollback on partial failure in `V2SquallHandler`
- `bool_disjunction` mixed callable/non-callable handling

These are documented via the **KNOWN STUBS** module docstring note added in 1.5.

---

## Acceptance Criteria

1. `uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py -k "not slow"` — all 13 tests pass
2. `uv run python -m pytest neurolang/utils/server/tests/test_v2_api.py -k squall` — all 5 tests pass
3. `uv run python -m pytest doc/tutorial_squall.rst --doctest-glob="doc/*.rst"` — all tutorial doctests pass
4. No `import` statement appears inside a function/method body in `squall_syntax_lark.py`
5. `_flatten_npc_body` does not appear in the file
6. `_AGG_FUNC_MAP` exists at module level; `ng1_agg_npc` and `agg_func` both reference it
7. `ops_var = Symbol.fresh()` does not appear in `rule_opnn`
8. `Disjunction` appears in the top-level import block

---

## Files Changed

| File | Change |
|------|--------|
| `neurolang/frontend/datalog/squall_syntax_lark.py` | Quick wins 1.1–1.5 |
| `doc/tutorial_squall.rst` | New tutorial file |
| `doc/index.rst` | Add `tutorial_squall` to toctree |
| `pyproject.toml` | Add `addopts = "--doctest-glob=doc/*.rst"` |
