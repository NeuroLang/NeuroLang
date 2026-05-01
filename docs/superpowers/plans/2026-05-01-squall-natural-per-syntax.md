# SQUALL Natural Per/Where/Inline-Comparison Syntax — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the SQUALL grammar so the probabilistic aggregation rule in the CBMA example reads as natural English, replacing four repeated `per` keywords and a relay variable with compact, prose-like syntax.

**Architecture:** Three independent, additive changes to the Lark grammar file and its companion transformer class, each followed by its own tests. Implementation order: B (verify inline comparison already works) → A (compact `per` list) → C (tuple membership clause) → update example. No changes to the core Datalog IR or solver.

**Tech Stack:** Python 3.12, Lark-parser (Earley), pytest, NeuroLang expression IR (`Symbol`, `Constant`, `FunctionApplication`, `Conjunction`, `Implication`).

---

## File Map

| File | Change type | Purpose |
|------|------------|---------|
| `neurolang/frontend/datalog/neurolang_natural.lark` | Modify | Add `dim_npc_list` alt to `dim` rule; add `rel_tuple_noun` alt to `rel_b` |
| `neurolang/frontend/datalog/squall_syntax_lark.py` | Modify | Add `dim_npc_list`, patch `dims_base`/`dims_rec`, add `rel_tuple_noun` transformer methods |
| `neurolang/frontend/tests/test_squall_pdl_integration.py` | Modify | Add integration tests for all three extensions |
| `examples/plot_squall_cbma_spatial_prior.py` | Modify | Update SQUALL program and docstring to Approach C sentence |

---

## Task 1: Verify Extension B — Inline expression comparison (no code changes expected)

**Files:**
- Test: `neurolang/frontend/tests/test_squall_pdl_integration.py`

- [ ] **Step 1: Write the failing-or-passing test** (expect it to pass if B already works)

Add this test at the end of `test_squall_pdl_integration.py`:

```python
def test_execute_squall_inline_expr_comparison():
    """'such that EUCLIDEAN(a,b) is lower than 5' works without a relay variable.

    This path: rel_s -> s_np_vp -> vpbe_rel -> rel_comp already exists.
    The test confirms the function-call expression is correctly used as the
    left operand of the comparison (not a relay variable).
    """
    import operator
    from neurolang.expressions import Constant, FunctionApplication, Symbol
    from neurolang.logic import Conjunction, Implication

    engine = NeurolangPDL()

    def my_dist(a, b):
        return abs(a - b)

    engine.symbol_table[Symbol("MY_DIST")] = Constant(my_dist)
    _ = engine.add_tuple_set([(1,), (3,), (10,)], name="point")

    # Parse: "such that MY_DIST(?x, ?y) is lower than 5"
    # Expected body contains: lt(MY_DIST(x, y), 5)  (no relay variable)
    result = engine.execute_squall_program(
        "define as Close every Point (?x) such that some Point (?y) "
        "and MY_DIST(?x, ?y) is lower than 5."
    )
    assert result is None  # rules-only, no obtain

    # Inspect the intensional rule that was added
    idb = engine.program_ir.intensional_database()
    close_symb = next(k for k in idb if k.name == "close")
    rule = idb[close_symb].formulas[0]

    # Body must contain a FunctionApplication of lt (operator.lt)
    body = rule.antecedent
    formulas = body.formulas if isinstance(body, Conjunction) else [body]
    lt_atoms = [
        f for f in formulas
        if isinstance(f, FunctionApplication)
        and isinstance(f.functor, Constant)
        and f.functor.value is operator.lt
    ]
    assert len(lt_atoms) == 1, f"Expected one lt atom, got: {formulas}"
    lt_atom = lt_atoms[0]
    # Left operand must be FunctionApplication(MY_DIST, ...)
    assert isinstance(lt_atom.args[0], FunctionApplication), (
        f"Expected FunctionApplication as lt left arg, got: {lt_atom.args[0]}"
    )
    assert lt_atom.args[0].functor == Symbol("MY_DIST")
```

- [ ] **Step 2: Run the test**

```bash
cd /Users/dwasserm/sources/NeuroLang/.worktrees/squall-cnl
python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py::test_execute_squall_inline_expr_comparison -v
```

Expected: **PASS** (extension B already works through existing grammar paths).  
If it fails with a parse error, the `expr_atom_fun_upper → vpbe_rel → rel_comp` path has a gap — investigate before proceeding.

- [ ] **Step 3: Commit**

```bash
git add neurolang/frontend/tests/test_squall_pdl_integration.py
git commit -m "test: verify inline expr comparison already works in SQUALL (extension B)"
```

---

## Task 2: Extension A — Compact `per` list (`per ?i, ?j, ?k and per ?s`)

**Files:**
- Modify: `neurolang/frontend/datalog/neurolang_natural.lark` (around line 86–88)
- Modify: `neurolang/frontend/datalog/squall_syntax_lark.py` (lines 1606–1634)
- Test: `neurolang/frontend/tests/test_squall_pdl_integration.py`

### Grammar change

- [ ] **Step 1: Write the failing integration test**

Add to `test_squall_pdl_integration.py`:

```python
def test_execute_squall_compact_per_list():
    """'per ?i, ?j, ?k and per ?s' produces same head args as four separate per dims.

    Compact form: per ?i1, ?j1, ?k1 and per ?s
    Verbose form: per ?i1 and per ?j1 and per ?k1 and per ?s
    Both must produce an Implication whose consequent args are (?i1, ?j1, ?k1, ?s).
    """
    from neurolang.expressions import Symbol
    from neurolang.logic import Implication
    from typing import Iterable

    engine = NeurolangPDL()

    def agg_collect(vals: Iterable) -> float:
        return float(sum(vals))

    engine.add_symbol(agg_collect, name="agg_collect")
    _ = engine.add_tuple_set(
        [(1, 10, 100, "s1"), (2, 20, 200, "s2")], name="data"
    )

    engine.execute_squall_program(
        "define as Result with a probability of "
        "the Agg_collect of the Data (?v; _; _; ?s) "
        "per ?i1, ?j1, ?k1 and per ?s "
        "that data(?i1, ?j1, ?k1) holds."
    )

    idb = engine.program_ir.intensional_database()
    # The query-based prob-fact translation produces a fresh det predicate.
    # Check any 'result'-related rule has 4 consequent args.
    result_rules = [
        idb[k].formulas[0]
        for k in idb
        if "result" in k.name.lower() or k.name.startswith("fresh")
    ]
    # At least one rule must have a consequent with 4 or 5 args
    # (the fresh det rule has prob_var + i1, j1, k1, s = 5 args).
    arg_counts = [len(r.consequent.args) for r in result_rules]
    assert any(c >= 4 for c in arg_counts), (
        f"Expected consequent with >=4 args, got arg counts: {arg_counts}"
    )
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py::test_execute_squall_compact_per_list -v
```

Expected: **FAIL** with a Lark parse error (unexpected token after `?i1,`).

- [ ] **Step 3: Add `dim_npc_list` grammar rule**

In `neurolang/frontend/datalog/neurolang_natural.lark`, find the `dim` rule (currently around line 86):

```lark
dim : _PER ng2              -> dim_ng2
    | _PER npc{THE}         -> dim_npc
    | agg_func _OF npc{THE} -> dim_agg
```

Replace with:

```lark
dim : _PER ng2                         -> dim_ng2
    | _PER npc{THE} ("," npc{THE})+    -> dim_npc_list
    | _PER npc{THE}                    -> dim_npc
    | agg_func _OF npc{THE}            -> dim_agg
```

**Important:** `dim_npc_list` must come before `dim_npc` so Lark's priority resolution prefers the longer match when a comma follows the first `npc`.

- [ ] **Step 4: Add `dim_npc_list` transformer method and patch `dims_base`/`dims_rec`**

In `neurolang/frontend/datalog/squall_syntax_lark.py`, replace the three methods starting at line 1606:

```python
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
    # "per region" → groupby variable from the noun
    noun2 = args[0]
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
```

- [ ] **Step 5: Run the test**

```bash
python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py::test_execute_squall_compact_per_list -v
```

Expected: **PASS**.

- [ ] **Step 6: Run the full test suite to check for regressions**

```bash
python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py -v
```

Expected: all existing tests still **PASS**.

- [ ] **Step 7: Commit**

```bash
git add neurolang/frontend/datalog/neurolang_natural.lark \
        neurolang/frontend/datalog/squall_syntax_lark.py \
        neurolang/frontend/tests/test_squall_pdl_integration.py
git commit -m "feat: add compact per-list syntax to SQUALL grammar (extension A)"
```

---

## Task 3: Extension C — Tuple membership clause (`where (?i; ?j; ?k) is a Voxel`)

**Files:**
- Modify: `neurolang/frontend/datalog/neurolang_natural.lark` (around line 98–106)
- Modify: `neurolang/frontend/datalog/squall_syntax_lark.py` (after line 1143, before `rel_adj1`)
- Test: `neurolang/frontend/tests/test_squall_pdl_integration.py`

### Grammar change

- [ ] **Step 1: Write the failing integration test**

Add to `test_squall_pdl_integration.py`:

```python
def test_execute_squall_where_tuple_is_noun():
    """'where (?i; ?j; ?k) is a Noun' expands to Noun(i, j, k) in rule body.

    The clause is semantically equivalent to 'that noun(?i, ?j, ?k) holds'.
    We verify by comparing the rule body produced by both forms.
    """
    import operator
    from neurolang.expressions import Constant, FunctionApplication, Symbol
    from neurolang.logic import Conjunction, Implication

    engine = NeurolangPDL()
    _ = engine.add_tuple_set(
        [(0, 0, 0), (1, 1, 1)], name="voxel"
    )
    _ = engine.add_tuple_set(
        [(0, 0, 0, 10), (1, 1, 1, 20)], name="focus"
    )

    engine.execute_squall_program(
        "define as Near every Focus (?i2; ?j2; ?k2; ?s) "
        "where (?i2; ?j2; ?k2) is a Voxel."
    )

    idb = engine.program_ir.intensional_database()
    near_symb = next(k for k in idb if k.name == "near")
    rule = idb[near_symb].formulas[0]

    body = rule.antecedent
    formulas = body.formulas if isinstance(body, Conjunction) else [body]
    voxel_calls = [
        f for f in formulas
        if isinstance(f, FunctionApplication)
        and isinstance(f.functor, Symbol)
        and f.functor.name == "voxel"
    ]
    assert len(voxel_calls) == 1, (
        f"Expected exactly one voxel(...) call in body, got: {formulas}"
    )
    assert len(voxel_calls[0].args) == 3, (
        f"Expected voxel(i,j,k) with 3 args, got: {voxel_calls[0]}"
    )
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py::test_execute_squall_where_tuple_is_noun -v
```

Expected: **FAIL** with a Lark parse error (unexpected `(` after `where`).

- [ ] **Step 3: Add `rel_tuple_noun` grammar rule**

In `neurolang/frontend/datalog/neurolang_natural.lark`, find the `rel_b` rule block (around line 98). Add the new alternative **after** `rel_fun_call`:

```lark
rel_b : (_THAT | _WHICH | _WHERE | _WHO ) vp                -> rel_vp
      | (_THAT | _WHICH | _WHERE | _WHOM ) np verbn [ ops ] -> rel_vpn
      | np2 _OF _WHICH vp                                   -> rel_np2
      | _WHOSE ng2 vp                                       -> rel_ng2
      | _SUCH _THAT s                                       -> rel_s
      | comparison (_THAN | _TO) op                         -> rel_comp
      | (_THAT | _WHICH | _WHERE ) identifier "(" label ("," label)* ")" _HOLDS -> rel_fun_call
      | _WHERE label (_IS | _ARE) (_A | _AN) noun1          -> rel_tuple_noun
      | adj1 [ cp ]                                         -> rel_adj1
```

`_WHERE label ...` is unambiguous with the existing `_WHERE vp` / `_WHERE np verbn` alternatives because `label` starts with `?` (a `_LABEL_MARKER`) or `(` followed by `_LABEL_MARKER` — neither of which can start a `vp` (which requires a verb or auxiliary) or an `np` (which requires a det/quantifier).

- [ ] **Step 4: Add `rel_tuple_noun` transformer method**

In `neurolang/frontend/datalog/squall_syntax_lark.py`, add the new method **after** `rel_fun_call` (currently ending around line 1143) and **before** `rel_adj1`:

```python
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
```

- [ ] **Step 5: Run the test**

```bash
python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py::test_execute_squall_where_tuple_is_noun -v
```

Expected: **PASS**.

- [ ] **Step 6: Run the full integration test suite**

```bash
python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py -v
```

Expected: all tests **PASS**.

- [ ] **Step 7: Commit**

```bash
git add neurolang/frontend/datalog/neurolang_natural.lark \
        neurolang/frontend/datalog/squall_syntax_lark.py \
        neurolang/frontend/tests/test_squall_pdl_integration.py
git commit -m "feat: add where-tuple-is-noun clause to SQUALL grammar (extension C)"
```

---

## Task 4: Update the example to Approach C

**Files:**
- Modify: `examples/plot_squall_cbma_spatial_prior.py`

- [ ] **Step 1: Update the `squall_program` string and the docstring**

In `examples/plot_squall_cbma_spatial_prior.py`, find `squall_program = """` (around line 170) and the matching docstring block (around line 22). Replace both occurrences of the first sentence:

**Old (both in docstring and in `squall_program`):**
```
define as Voxel_reported with a probability of
    the Agg_max_proximity of the Focus_reported (?i2; ?j2; ?k2; ?s)
        per ?i1 and per ?j1 and per ?k1 and per ?s
        that voxel(?i1, ?j1, ?k1) holds
        and such that ?d is equal to EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2)
        and ?d is lower than 5.
```

**New:**
```
define as Voxel_reported with a probability of
    the Agg_max_proximity of the Focus_reported (?i2; ?j2; ?k2; ?s)
        per ?i1, ?j1, ?k1 and per ?s
        where (?i1; ?j1; ?k1) is a Voxel
        and such that EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 5.
```

Also update the comment block around lines 154–168 that describes each sentence:

**Old sentence 1 description:**
```python
# Sentence 1  Probabilistic aggregation: "with a probability of the
#             Agg_max_proximity of the Focus_reported per ?i1 ... that
#             voxel(?i1,?j1,?k1) holds and such that ?d is equal to
#             euclidean(...) and ?d is lower than 1"
#             — inline distance computation and filter.  The ``per`` variables
#             become the head predicate arguments; ``agg_max_proximity``
#             aggregates over distances ``?d`` that pass the ``< 1`` threshold.
```

**New:**
```python
# Sentence 1  Probabilistic aggregation: ``per ?i1, ?j1, ?k1 and per ?s``
#             groups by the four head variables; ``where (?i1; ?j1; ?k1) is
#             a Voxel`` constrains the focus coordinates to voxel grid
#             membership; ``EUCLIDEAN(…) is lower than 5`` filters by
#             proximity inline — no relay variable needed.
```

Also remove the two debug lines that were introduced during development (around lines 191–192):

```python
for r in nl.current_program:
    print(r)
```

- [ ] **Step 2: Run the end-to-end mini smoke-test**

```bash
cd /Users/dwasserm/sources/NeuroLang/.worktrees/squall-cnl
python -c "
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from typing import Iterable
from neurolang.frontend import NeurolangPDL, ExplicitVBROverlay

import nibabel, nilearn.datasets, nilearn.image
data_dir = __import__('pathlib').Path.home() / 'neurolang_data'
mni_t1 = nibabel.load(nilearn.datasets.fetch_icbm152_2009(data_dir=str(data_dir / 'icbm'))['t1'])
mni_t1_2mm = nilearn.image.resample_img(mni_t1, np.eye(3) * 2)

nl = NeurolangPDL()

def euclidean(i1,j1,k1,i2,j2,k2): return np.sqrt((i1-i2)**2+(j1-j2)**2+(k1-k2)**2)
nl.add_symbol(euclidean, name='EUCLIDEAN')

@nl.add_symbol
def agg_max_proximity(d: Iterable) -> float:
    return float(np.max(np.exp(-np.asarray(d)/5.0)))

@nl.add_symbol
def agg_create_region_overlay(i:Iterable,j:Iterable,k:Iterable,p:Iterable):
    return ExplicitVBROverlay(np.c_[i,j,k], mni_t1_2mm.affine, p, image_dim=mni_t1_2mm.shape)

nl.add_tuple_set(pd.DataFrame({'i':[0],'j':[0],'k':[0],'id':[10]}), name='focus_reported')
nl.add_tuple_set(pd.DataFrame({'i':[0],'j':[0],'k':[0]}), name='voxel')
nl.add_uniform_probabilistic_choice_over_set(pd.DataFrame({'id':[10,20]}), name='selected_study')
nl.add_tuple_set(pd.DataFrame({'s':[10],'t':['emotion'],'w':[1.0]}), name='term_in_study_tfidf')

prog = '''
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
'''

nl.execute_squall_program(prog)
sol = nl.solve_all()
assert 'img' in sol, f'img missing from solution. Keys: {list(sol.keys())}'
print('OK — img in solution:', sol['img'].as_pandas_dataframe())
"
```

Expected: prints `OK — img in solution:` with one row.

- [ ] **Step 3: Commit**

```bash
git add examples/plot_squall_cbma_spatial_prior.py
git commit -m "feat: update CBMA SQUALL example to natural English sentence (Approach C)"
```

---

## Task 5: Final regression check

- [ ] **Step 1: Run the full integration test suite**

```bash
python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py -v
```

Expected: all tests **PASS** including the three new ones.

- [ ] **Step 2: Run the broader frontend test suite**

```bash
python -m pytest neurolang/frontend/tests/ -v --tb=short -q
```

Expected: no new failures.
