# SQUALL Naturalness Extensions D–H Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five independent grammar extensions (for-each, where-as-such-that, given, obtain-as, with-inferred-probability) plus a rename pass to make the CBMA example SQUALL script read as natural English.

**Architecture:** Each extension adds one or more terminals/rule alternatives to `neurolang_natural.lark` and zero or one transformer methods to `squall_syntax_lark.py`. Extensions E/F/H require only grammar changes. Extension D adds one new terminal. Extension G adds a new terminal plus a new transformer method and a one-line change to `squall()`. The rename pass only touches `examples/plot_squall_cbma_spatial_prior.py`.

**Tech Stack:** Python 3, Lark Earley parser, NeuroLang IR (Implication, Conjunction, Condition, Query, ExistentialPredicate, Symbol, Constant)

---

## File Map

| File | Changes |
|---|---|
| `neurolang/frontend/datalog/neurolang_natural.lark` | All grammar changes (D, E, F, G, H) |
| `neurolang/frontend/datalog/squall_syntax_lark.py` | Rename `query→query_unnamed`, add `query_as`, extend `squall()` for `_query_as` |
| `neurolang/frontend/tests/test_squall_syntax_lark.py` | Unit tests for each extension |
| `neurolang/frontend/tests/test_squall_pdl_integration.py` | Integration tests for each extension |
| `examples/plot_squall_cbma_spatial_prior.py` | Rename pass + update SQUALL program string |

**Important context:**
- The grammar file `neurolang_natural.lark` is compiled into `COMPILED_GRAMMAR` at module import time. Any edit to the `.lark` file takes effect on the next Python import.
- All grammar rule aliases (e.g. `-> dim_npc`) map to transformer methods of the same name in `SquallTransformer`. Terminals prefixed `_` are transparent (not passed as args).
- The keyword-exclusion regex is on lines 319–320 of the lark file. New reserved words must be added to both the `LOWER_NAME` and `UPPER_NAME` regex alternation groups.
- `_AS` and `_GIVEN` are **already defined** in the grammar (lines 267, 280). `_EACH` and `_INFERRED` are not yet defined.
- `squall()` method (line 225 of `squall_syntax_lark.py`) processes `('_query', ...)` tagged tuples. Extension G adds handling for `('_query_as', ...)`.

---

## Task 1: Extension E — `where` as alias for `such that`

**Files:**
- Modify: `neurolang/frontend/datalog/neurolang_natural.lark` (line 103, `rel_b` rule)
- Test: `neurolang/frontend/tests/test_squall_syntax_lark.py`
- Test: `neurolang/frontend/tests/test_squall_pdl_integration.py`

- [ ] **Step 1: Write the failing unit test**

Add to `test_squall_syntax_lark.py`:

```python
def test_extension_e_where_as_such_that_variable(nl_setup):
    """'where ?x is a Noun' parses identically to 'such that ?x is a Noun'."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser
    from neurolang.expressions import Symbol

    prog_such = parser(
        "define as Foo every Bar (?x) such that ?x is a Selected_study."
    )
    prog_where = parser(
        "define as Foo every Bar (?x) where ?x is a Selected_study."
    )
    # Both should yield an Implication; the antecedents should be structurally equal
    from neurolang.logic import Implication
    assert isinstance(prog_such, Implication)
    assert isinstance(prog_where, Implication)
    assert prog_such.antecedent == prog_where.antecedent


def test_extension_e_where_inline_expr(nl_setup):
    """'and where FUNC(…) is lower than N' parses like 'and such that FUNC(…) is lower than N'."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser
    import operator
    from neurolang.expressions import Constant

    prog_such = parser(
        "define as Foo every Bar (?x) such that MY_DIST(?x, ?x) is lower than 5."
    )
    prog_where = parser(
        "define as Foo every Bar (?x) where MY_DIST(?x, ?x) is lower than 5."
    )
    from neurolang.logic import Implication, Conjunction
    for prog in (prog_such, prog_where):
        assert isinstance(prog, Implication)
    # Both antecedents must contain a lt(…) atom
    def has_lt(formula):
        from neurolang.expressions import FunctionApplication
        if isinstance(formula, FunctionApplication):
            return formula.functor == Constant(operator.lt)
        if isinstance(formula, Conjunction):
            return any(has_lt(f) for f in formula.formulas)
        return False
    assert has_lt(prog_such.antecedent)
    assert has_lt(prog_where.antecedent)
    assert prog_such.antecedent == prog_where.antecedent
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/dwasserm/sources/NeuroLang/.worktrees/squall-cnl
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py::test_extension_e_where_as_such_that_variable neurolang/frontend/tests/test_squall_syntax_lark.py::test_extension_e_where_inline_expr -v 2>&1 | tail -20
```

Expected: FAIL (parse error — `_WHERE s` not yet a valid alternative).

- [ ] **Step 3: Add `_WHERE s -> rel_s` alternative in the grammar**

In `neurolang/frontend/datalog/neurolang_natural.lark`, find line 103:
```lark
      | _SUCH _THAT s                                       -> rel_s
```
Change to:
```lark
      | _SUCH _THAT s                                       -> rel_s
      | _WHERE s                                            -> rel_s
```

**Important:** place this new line AFTER the `_WHERE label … -> rel_tuple_noun` line (line 106) to avoid shadowing it, or verify Earley handles it by longest match. The safest placement is immediately after `| _SUCH _THAT s -> rel_s` at line 103, since `rel_tuple_noun` (line 106) has a more specific first token sequence (`_WHERE label`).

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py::test_extension_e_where_as_such_that_variable neurolang/frontend/tests/test_squall_syntax_lark.py::test_extension_e_where_inline_expr -v 2>&1 | tail -20
```

Expected: PASS

- [ ] **Step 5: Write integration test**

Add to `neurolang/frontend/tests/test_squall_pdl_integration.py`:

```python
def test_extension_e_where_integration(tmp_path):
    """Full program using 'where' instead of 'such that' produces same solution."""
    from neurolang.frontend import NeurolangPDL
    import pandas as pd

    nl = NeurolangPDL()
    df = pd.DataFrame({"x": [1, 2, 3]})
    nl.add_tuple_set(df, name="item")
    df2 = pd.DataFrame({"x": [1, 2]})
    nl.add_tuple_set(df2, name="selected")

    # with 'such that'
    nl.execute_squall_program(
        "define as Result every Item (?x) such that ?x is a Selected."
    )
    sol_such = nl.solve_all()["result"].as_pandas_dataframe()

    nl2 = NeurolangPDL()
    nl2.add_tuple_set(df, name="item")
    nl2.add_tuple_set(df2, name="selected")

    # with 'where'
    nl2.execute_squall_program(
        "define as Result every Item (?x) where ?x is a Selected."
    )
    sol_where = nl2.solve_all()["result"].as_pandas_dataframe()

    assert set(sol_such["x"]) == set(sol_where["x"])
```

- [ ] **Step 6: Run integration test**

```bash
python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py::test_extension_e_where_integration -v 2>&1 | tail -20
```

Expected: PASS

- [ ] **Step 7: Run existing test suite (regression)**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py neurolang/frontend/tests/test_squall_pdl_integration.py -v 2>&1 | tail -30
```

Expected: All previously-passing tests still PASS.

- [ ] **Step 8: Commit**

```bash
git add neurolang/frontend/datalog/neurolang_natural.lark \
        neurolang/frontend/tests/test_squall_syntax_lark.py \
        neurolang/frontend/tests/test_squall_pdl_integration.py
git commit -m "feat(squall): Extension E — where as alias for such that in rel_b"
```

---

## Task 2: Extension F — `given` as alias for `conditioned to`

**Files:**
- Modify: `neurolang/frontend/datalog/neurolang_natural.lark` (lines 36–39, `rule_body1_cond` / `rule_body2_cond`)
- Test: `neurolang/frontend/tests/test_squall_syntax_lark.py`
- Test: `neurolang/frontend/tests/test_squall_pdl_integration.py`

**Background:** `_GIVEN : "given"` is already defined at line 280 and already excluded from `LOWER_NAME`/`UPPER_NAME` on lines 319–320. No terminal definition or keyword-exclusion change needed.

- [ ] **Step 1: Write the failing unit test**

Add to `test_squall_syntax_lark.py`:

```python
def test_extension_f_given_as_conditioned_to(nl_setup):
    """'given every X (…)' parses identically to 'conditioned to every X (…)'."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser
    from neurolang.probabilistic.expressions import Condition
    from neurolang.logic import Implication

    prog_cond = parser(
        "define as Probmap with probability every Activation (?i; ?j; ?k; _) "
        "conditioned to every Selected_study (_)."
    )
    prog_given = parser(
        "define as Probmap with probability every Activation (?i; ?j; ?k; _) "
        "given every Selected_study (_)."
    )
    assert isinstance(prog_cond, Implication)
    assert isinstance(prog_given, Implication)
    # Both antecedents must be Condition instances
    assert isinstance(prog_cond.antecedent, Condition)
    assert isinstance(prog_given.antecedent, Condition)
    # conditioned and conditioning bodies must match structurally
    assert type(prog_cond.antecedent.conditioned) == type(prog_given.antecedent.conditioned)
    assert type(prog_cond.antecedent.conditioning) == type(prog_given.antecedent.conditioning)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py::test_extension_f_given_as_conditioned_to -v 2>&1 | tail -20
```

Expected: FAIL (parse error).

- [ ] **Step 3: Update `rule_body1_cond` and `rule_body2_cond` in the grammar**

In `neurolang/frontend/datalog/neurolang_natural.lark`, change lines 36–39 from:

```lark
rule_body1_cond : det ng1 _CONDITIONED _TO s -> rule_body1_cond_prior
                | s _CONDITIONED _TO det ng1 -> rule_body1_cond_posterior

rule_body2_cond : det ng1 _CONDITIONED _TO det ng1
```

To:

```lark
rule_body1_cond : det ng1 (_CONDITIONED _TO | _GIVEN) s -> rule_body1_cond_prior
                | s (_CONDITIONED _TO | _GIVEN) det ng1 -> rule_body1_cond_posterior

rule_body2_cond : det ng1 (_CONDITIONED _TO | _GIVEN) det ng1
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py::test_extension_f_given_as_conditioned_to -v 2>&1 | tail -20
```

Expected: PASS

- [ ] **Step 5: Write integration test**

Add to `neurolang/frontend/tests/test_squall_pdl_integration.py`:

```python
def test_extension_f_given_integration(tmp_path):
    """'given every X' produces same MARG solution as 'conditioned to every X'."""
    from neurolang.frontend import NeurolangPDL
    import pandas as pd

    def _build_engine():
        nl = NeurolangPDL()
        df_act = pd.DataFrame({"i": [1, 2, 3], "j": [0, 0, 0], "k": [0, 0, 0]})
        nl.add_tuple_set(df_act, name="activation")
        study_ids = pd.DataFrame({"s": ["s1", "s2"]})
        nl.add_uniform_probabilistic_choice_over_set(study_ids, name="selected_study")
        return nl

    nl1 = _build_engine()
    nl1.execute_squall_program(
        "define as Probmap with probability every Activation (?i; ?j; ?k) "
        "conditioned to every Selected_study (_)."
    )
    sol1 = nl1.solve_all().get("probmap")

    nl2 = _build_engine()
    nl2.execute_squall_program(
        "define as Probmap with probability every Activation (?i; ?j; ?k) "
        "given every Selected_study (_)."
    )
    sol2 = nl2.solve_all().get("probmap")

    assert sol1 is not None
    assert sol2 is not None
    df1 = sol1.as_pandas_dataframe().sort_values(list(sol1.as_pandas_dataframe().columns)).reset_index(drop=True)
    df2 = sol2.as_pandas_dataframe().sort_values(list(sol2.as_pandas_dataframe().columns)).reset_index(drop=True)
    import pandas.testing as tm
    tm.assert_frame_equal(df1, df2)
```

- [ ] **Step 6: Run integration test**

```bash
python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py::test_extension_f_given_integration -v 2>&1 | tail -20
```

Expected: PASS

- [ ] **Step 7: Run regression tests**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py neurolang/frontend/tests/test_squall_pdl_integration.py -v 2>&1 | tail -30
```

Expected: All previously-passing tests still PASS.

- [ ] **Step 8: Commit**

```bash
git add neurolang/frontend/datalog/neurolang_natural.lark \
        neurolang/frontend/tests/test_squall_syntax_lark.py \
        neurolang/frontend/tests/test_squall_pdl_integration.py
git commit -m "feat(squall): Extension F — given as alias for conditioned to"
```

---

## Task 3: Extension H — `with inferred probability`

**Files:**
- Modify: `neurolang/frontend/datalog/neurolang_natural.lark` (lines 28–29, `rule_op_marg`; lines 319–320, keyword exclusion)
- Test: `neurolang/frontend/tests/test_squall_syntax_lark.py`

**Background:** `_INFERRED` does not yet exist. Must add terminal definition AND add `"inferred"` to the keyword-exclusion regex in `LOWER_NAME`/`UPPER_NAME`.

- [ ] **Step 1: Write the failing unit test**

Add to `test_squall_syntax_lark.py`:

```python
def test_extension_h_with_inferred_probability(nl_setup):
    """'with inferred probability … conditioned to …' parses same as 'with probability … conditioned to …'."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser
    from neurolang.probabilistic.expressions import Condition
    from neurolang.logic import Implication

    prog_plain = parser(
        "define as Probmap with probability every Activation (?i; ?j; ?k; _) "
        "conditioned to every Selected_study (_)."
    )
    prog_inferred = parser(
        "define as Probmap with inferred probability every Activation (?i; ?j; ?k; _) "
        "conditioned to every Selected_study (_)."
    )
    assert isinstance(prog_plain, Implication)
    assert isinstance(prog_inferred, Implication)
    assert isinstance(prog_plain.antecedent, Condition)
    assert isinstance(prog_inferred.antecedent, Condition)
    # Consequent functors should match
    assert prog_plain.consequent.functor == prog_inferred.consequent.functor
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py::test_extension_h_with_inferred_probability -v 2>&1 | tail -20
```

Expected: FAIL (parse error — `inferred` not a recognised token).

- [ ] **Step 3: Add `_INFERRED` terminal**

In `neurolang/frontend/datalog/neurolang_natural.lark`, add after the `_GIVEN : "given"` line (line 280):

```lark
_INFERRED : "inferred"
```

- [ ] **Step 4: Add `"inferred"` to the keyword-exclusion regex**

On lines 319–320, both `LOWER_NAME` and `UPPER_NAME` end with `…|whose)\b))`. Add `inferred` to the alternation list. The updated regex alternation should include `inferred|`:

```
…|given|inferred|holds|…
```

The exact edit: in both lines 319 and 320, change `given|holds` to `given|inferred|holds`.

- [ ] **Step 5: Make `_INFERRED` optional in `rule_op_marg`**

Change lines 28–29 from:

```lark
        | _rule_start verb1 _WITH _PROBABILITY rule_body1_cond -> rule_op_marg
        | _rule_start verb1 _WITH _PROBABILITY rule_body2_cond -> rule_op_marg
```

To:

```lark
        | _rule_start verb1 _WITH _INFERRED? _PROBABILITY rule_body1_cond -> rule_op_marg
        | _rule_start verb1 _WITH _INFERRED? _PROBABILITY rule_body2_cond -> rule_op_marg
```

- [ ] **Step 6: Run test to verify it passes**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py::test_extension_h_with_inferred_probability -v 2>&1 | tail -20
```

Expected: PASS

- [ ] **Step 7: Run regression tests**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py neurolang/frontend/tests/test_squall_pdl_integration.py -v 2>&1 | tail -30
```

Expected: All previously-passing tests still PASS.

- [ ] **Step 8: Commit**

```bash
git add neurolang/frontend/datalog/neurolang_natural.lark \
        neurolang/frontend/tests/test_squall_syntax_lark.py
git commit -m "feat(squall): Extension H — with inferred probability optional token"
```

---

## Task 4: Extension D — `for each` alias for `per`

**Files:**
- Modify: `neurolang/frontend/datalog/neurolang_natural.lark` (lines 86–89, `dim` rule; lines 319–320, keyword exclusion; terminal block)
- Test: `neurolang/frontend/tests/test_squall_syntax_lark.py`
- Test: `neurolang/frontend/tests/test_squall_pdl_integration.py`

**Background:** `_FOR` already exists (line 278). `_EACH` does not. The word `"each"` is not yet in the keyword-exclusion regex.

- [ ] **Step 1: Write the failing unit test**

Add to `test_squall_syntax_lark.py`:

```python
def test_extension_d_for_each_alias_per(nl_setup):
    """'for each ?i, ?j, ?k and for each ?s' produces same dims as 'per ?i, ?j, ?k and per ?s'."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser
    from neurolang.logic import Implication

    prog_per = parser(
        "define as Reported_voxel with a probability of "
        "the Kernelized_max_proximity of the Focus (?i2; ?j2; ?k2; ?s) "
        "per ?i1, ?j1, ?k1 and per ?s "
        "where (?i1; ?j1; ?k1) is a Voxel."
    )
    prog_foreach = parser(
        "define as Reported_voxel with a probability of "
        "the Kernelized_max_proximity of the Focus (?i2; ?j2; ?k2; ?s) "
        "for each ?i1, ?j1, ?k1 and for each ?s "
        "where (?i1; ?j1; ?k1) is a Voxel."
    )
    assert isinstance(prog_per, Implication)
    assert isinstance(prog_foreach, Implication)
    # Head args (per-vars) must be the same set
    per_args = {a.name for a in prog_per.consequent.body.args}
    foreach_args = {a.name for a in prog_foreach.consequent.body.args}
    assert per_args == foreach_args
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py::test_extension_d_for_each_alias_per -v 2>&1 | tail -20
```

Expected: FAIL (parse error — `each` not a recognised token in `dim`).

- [ ] **Step 3: Add `_EACH` terminal**

In `neurolang/frontend/datalog/neurolang_natural.lark`, add after `_FOR : "for"` (line 278):

```lark
_EACH : "each"
```

- [ ] **Step 4: Add `"each"` to the keyword-exclusion regex**

On lines 319–320, add `each` to the alternation list alongside other short keywords. Change `equal|from|for` to `each|equal|from|for` (or any alphabetically adjacent position — order doesn't matter as long as it's in the group):

```
…|did|each|every|equal|…
```

- [ ] **Step 5: Update the `dim` rule to accept `_FOR _EACH`**

Change lines 86–89 from:

```lark
dim : _PER ng2                         -> dim_ng2
    | _PER npc{THE} ("," npc{THE})+    -> dim_npc_list
    | _PER npc{THE}                    -> dim_npc
    | agg_func _OF npc{THE}            -> dim_agg
```

To:

```lark
dim : (_PER | _FOR _EACH) ng2                         -> dim_ng2
    | (_PER | _FOR _EACH) npc{THE} ("," npc{THE})+    -> dim_npc_list
    | (_PER | _FOR _EACH) npc{THE}                    -> dim_npc
    | agg_func _OF npc{THE}                            -> dim_agg
```

- [ ] **Step 6: Run test to verify it passes**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py::test_extension_d_for_each_alias_per -v 2>&1 | tail -20
```

Expected: PASS

- [ ] **Step 7: Write integration test**

Add to `neurolang/frontend/tests/test_squall_pdl_integration.py`:

```python
def test_extension_d_for_each_integration(tmp_path):
    """Rule using 'for each' groupby produces same result as 'per'."""
    from neurolang.frontend import NeurolangPDL
    import pandas as pd
    import numpy as np
    from typing import Iterable

    def _build(keyword):
        nl = NeurolangPDL()

        def my_max(vals: Iterable) -> float:
            return float(np.max(list(vals)))

        nl.add_symbol(my_max, name="My_max")
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        nl.add_tuple_set(df, name="pair")
        nl.execute_squall_program(
            f"define as Result with a probability of "
            f"the My_max of the Pair (?y; ?z) "
            f"{keyword} ?x "
            f"where ?z is equal to ?y."
        )
        return nl.solve_all().get("result")

    sol_per = _build("per ?x")
    sol_foreach = _build("for each ?x")
    assert sol_per is not None
    assert sol_foreach is not None
    df_per = sol_per.as_pandas_dataframe().sort_values("x_0").reset_index(drop=True)
    df_fe = sol_foreach.as_pandas_dataframe().sort_values("x_0").reset_index(drop=True)
    import pandas.testing as tm
    tm.assert_frame_equal(df_per, df_fe)
```

- [ ] **Step 8: Run integration test**

```bash
python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py::test_extension_d_for_each_integration -v 2>&1 | tail -20
```

Expected: PASS

- [ ] **Step 9: Run regression tests**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py neurolang/frontend/tests/test_squall_pdl_integration.py -v 2>&1 | tail -30
```

Expected: All previously-passing tests still PASS.

- [ ] **Step 10: Commit**

```bash
git add neurolang/frontend/datalog/neurolang_natural.lark \
        neurolang/frontend/tests/test_squall_syntax_lark.py \
        neurolang/frontend/tests/test_squall_pdl_integration.py
git commit -m "feat(squall): Extension D — for each as alias for per in dim rule"
```

---

## Task 5: Extension G — `obtain … as Name`

**Files:**
- Modify: `neurolang/frontend/datalog/neurolang_natural.lark` (line 19, `query` rule; terminal block; lines 319–320, keyword exclusion)
- Modify: `neurolang/frontend/datalog/squall_syntax_lark.py` (rename `query`→`query_unnamed`; add `query_as`; extend `squall()`)
- Test: `neurolang/frontend/tests/test_squall_syntax_lark.py`
- Test: `neurolang/frontend/tests/test_squall_pdl_integration.py`

**Background:** `_AS : "as"` is already defined at line 267 and `"as"` is already in the keyword-exclusion regex on lines 319–320. No terminal definition or keyword-exclusion change needed. The existing `query` transformer method must be renamed `query_unnamed`. The `squall()` method must be extended to handle `('_query_as', ...)` tagged tuples.

- [ ] **Step 1: Write the failing unit test**

Add to `test_squall_syntax_lark.py`:

```python
def test_extension_g_query_as_tagged_tuple(nl_setup):
    """'obtain ops as Name' returns a ('_query_as', ...) tagged tuple."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser, SquallProgram
    from neurolang.expressions import Symbol
    from neurolang.logic import Implication

    result = parser(
        "define as Activation_map with inferred probability every Active_voxel (?i; ?j; ?k; _) "
        "given every Study_term (_; ?t) where ?t is 'emotion'.\n"
        "obtain the Brain_image of the Activation_map (?i; ?j; ?k; ?p) as Image."
    )
    # Should be a SquallProgram with a query registered under 'image'
    assert isinstance(result, SquallProgram)
    assert result.queries  # at least one query
    # The IDB rules should include an Image(...) head
    rule_functors = {r.consequent.functor.name for r in result.rules if hasattr(r, 'consequent')}
    assert 'image' in rule_functors
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py::test_extension_g_query_as_tagged_tuple -v 2>&1 | tail -20
```

Expected: FAIL (parse error — `_OBTAIN ops _AS identifier` not yet valid).

- [ ] **Step 3: Update the grammar `query` rule**

In `neurolang/frontend/datalog/neurolang_natural.lark`, change line 19 from:

```lark
query : _OBTAIN ops
```

To:

```lark
query : _OBTAIN ops _AS identifier  -> query_as
      | _OBTAIN ops                 -> query_unnamed
```

- [ ] **Step 4: Rename `query` transformer method to `query_unnamed`**

In `neurolang/frontend/datalog/squall_syntax_lark.py`, rename the existing method:

```python
# Before (line 1605):
def query(self, args):

# After:
def query_unnamed(self, args):
```

The body stays completely unchanged.

- [ ] **Step 5: Add `query_as` transformer method**

In `squall_syntax_lark.py`, add the following method immediately after `query_unnamed`:

```python
def query_as(self, args):
    """Handle 'obtain ops as Name'.

    Builds an Implication(Name(*free_vars), body) that defines the named
    relation, then registers a Query so solve_all() returns it under
    the lowercased Name.

    Returns a ('_query_as', (impl, name_sym)) tagged tuple consumed by
    the squall() method.
    """
    from .squall import LogicSimplifier

    ops, name_sym = args[0], args[1]   # name_sym: Symbol from identifier

    # Materialise the CPS NP to recover free variables and body formula.
    # The NP is a CPS lambda: applying it with a capturing continuation
    # collects the bound variable and the restriction body.
    if callable(ops) and not isinstance(ops, (Symbol, Constant)):
        formula = ops(lambda x: x)
    else:
        formula = ops

    formula = LogicSimplifier().walk(formula)

    # Extract the query body and bound variable from the quantified formula.
    if isinstance(formula, (ExistentialPredicate, UniversalPredicate)):
        body = formula.body
        if isinstance(body, Implication):
            body = body.antecedent
    else:
        body = formula

    # Free variables in the body become the head args.
    free = sorted(extract_logic_free_variables(body), key=lambda s: s.name)
    head = name_sym(*free) if free else name_sym()
    impl = Implication(head, body)

    return ('_query_as', (impl, name_sym))
```

- [ ] **Step 6: Extend `squall()` to handle `('_query_as', ...)`**

In `squall_syntax_lark.py`, the `squall()` method starts at line 225. Update it:

```python
def squall(self, args):
    rules = []
    queries = []
    for a in args:
        if a is None:
            continue
        if isinstance(a, tuple) and len(a) == 2 and a[0] == '_query':
            queries.append(a[1])
        elif isinstance(a, tuple) and len(a) == 2 and a[0] == '_query_as':
            impl, name_sym = a[1]
            rules.append(impl)
            # Build a Query that asks for all tuples of the named relation.
            from ...logic.expression_processing import extract_logic_free_variables
            free = sorted(
                extract_logic_free_variables(impl.consequent),
                key=lambda s: s.name
            )
            queries.append(Query(name_sym(*free) if free else name_sym(), impl.consequent))
        else:
            rules.append(a)

    if queries:
        return SquallProgram(rules=rules, queries=queries)

    # Backward compat: no queries → return Union or single rule
    if len(rules) == 1:
        return rules[0]
    return Union(tuple(rules))
```

Note: `Query` is already imported at the top of the file (check with `grep -n "^from.*import\|^import" squall_syntax_lark.py | grep Query`). If not, add `from ...datalog import Query` or check exact import path with `grep -rn "class Query" neurolang/`.

- [ ] **Step 7: Run unit test to verify it passes**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py::test_extension_g_query_as_tagged_tuple -v 2>&1 | tail -20
```

Expected: PASS

- [ ] **Step 8: Write integration test**

Add to `neurolang/frontend/tests/test_squall_pdl_integration.py`:

```python
def test_extension_g_obtain_as_integration(tmp_path):
    """'obtain … as Image' defines a relation and exposes it in solve_all()."""
    from neurolang.frontend import NeurolangPDL
    import pandas as pd
    import numpy as np
    from typing import Iterable

    nl = NeurolangPDL()

    def identity_agg(vals: Iterable):
        return list(vals)[0]

    nl.add_symbol(identity_agg, name="Id_agg")
    df = pd.DataFrame({"x": [1, 2, 3]})
    nl.add_tuple_set(df, name="item")
    nl.execute_squall_program(
        "obtain the Id_agg of the Item (?x) as Result."
    )
    sol = nl.solve_all()
    assert "result" in sol
    result_df = sol["result"].as_pandas_dataframe()
    assert len(result_df) > 0
```

- [ ] **Step 9: Run integration test**

```bash
python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py::test_extension_g_obtain_as_integration -v 2>&1 | tail -20
```

Expected: PASS. If the `query_as` body extraction needs tuning for the engine's query dispatch, debug the `squall()` output by printing `result.queries` and `result.rules` before committing.

- [ ] **Step 10: Run regression tests**

```bash
python -m pytest neurolang/frontend/tests/test_squall_syntax_lark.py neurolang/frontend/tests/test_squall_pdl_integration.py -v 2>&1 | tail -30
```

Expected: All previously-passing tests still PASS.

- [ ] **Step 11: Commit**

```bash
git add neurolang/frontend/datalog/neurolang_natural.lark \
        neurolang/frontend/datalog/squall_syntax_lark.py \
        neurolang/frontend/tests/test_squall_syntax_lark.py \
        neurolang/frontend/tests/test_squall_pdl_integration.py
git commit -m "feat(squall): Extension G — obtain … as Name query form"
```

---

## Task 6: Rename pass + update example script

**Files:**
- Modify: `examples/plot_squall_cbma_spatial_prior.py`

**Rename table:**

| Old Python name | New Python name |
|---|---|
| `focus_reported` (add_tuple_set name) | `reported_focus` |
| `term_in_study_tfidf` (load name) | `term_in_study_with_tfidf` |
| `agg_max_proximity` (function + add_symbol) | `kernelized_max_proximity` |
| `agg_create_region_overlay` (function + add_symbol) | `brain_image` |
| SQUALL program string | see target below |

`EUCLIDEAN` must **not** be renamed (spatial-sugar pattern match).

- [ ] **Step 1: Update Python symbol registrations**

In `examples/plot_squall_cbma_spatial_prior.py`:

Change:
```python
@nl.add_symbol
def agg_max_proximity(d_values: Iterable) -> float:
    """Aggregate: max exp(−d/5) over a collection of distances."""
    return float(np.max(np.exp(-np.asarray(d_values) / 5.0)))
```

To:
```python
@nl.add_symbol
def kernelized_max_proximity(d_values: Iterable) -> float:
    """Aggregate: max exp(−d/5) over a collection of distances."""
    return float(np.max(np.exp(-np.asarray(d_values) / 5.0)))
```

Change:
```python
@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> ExplicitVBR:
    """Aggregate (i,j,k,probability) rows into a brain overlay image."""
```

To:
```python
@nl.add_symbol
def brain_image(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> ExplicitVBR:
    """Aggregate (i,j,k,probability) rows into a brain overlay image."""
```

- [ ] **Step 2: Update `add_tuple_set` and `load_neurosynth_*` names**

Change:
```python
nl.add_tuple_set(peak_data, name="focus_reported")
```
To:
```python
nl.add_tuple_set(peak_data, name="reported_focus")
```

Change (the neurosynth term associations load call):
```python
nl.load_neurosynth_term_study_associations(
    data_dir, "term_in_study_tfidf", tfidf_threshold=1e-3
)
```
To:
```python
nl.load_neurosynth_term_study_associations(
    data_dir, "term_in_study_with_tfidf", tfidf_threshold=1e-3
)
```

- [ ] **Step 3: Replace the SQUALL program string**

Change the `squall_program` variable from:

```python
squall_program = """
define as Voxel_reported with a probability of
    the Agg_max_proximity of the Focus_reported (?i2; ?j2; ?k2; ?s)
        per ?i1, ?j1, ?k1 and per ?s
        where (?i1; ?j1; ?k1) is a Voxel
        and such that EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 2.

define as Term_association every Term_in_study_tfidf (?s; ?t; _)
    such that ?s is a Selected_study.

define as Activation every Voxel_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

define as Probmap with probability every Activation (?i; ?j; ?k; _)
    conditioned to every Term_association (_; ?t) such that ?t is 'emotion'.

define as Img every Agg_create_region_overlay of the Probmap (?i; ?j; ?k; ?p).
"""
```

To:

```python
squall_program = """
define as Reported_voxel with a probability of
    the Kernelized_max_proximity of the Reported_focus (?i2; ?j2; ?k2; ?s)
        for each ?i1, ?j1, ?k1 and for each ?s
        where (?i1; ?j1; ?k1) is a Voxel
        and where EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 2.

define as Study_term every Term_in_study_with_tfidf (?s; ?t; _)
    where ?s is a Selected_study.

define as Active_voxel every Reported_voxel (?i; ?j; ?k; ?s)
    where ?s is a Selected_study.

define as Activation_map with inferred probability every Active_voxel (?i; ?j; ?k; _)
    given every Study_term (_; ?t) where ?t is 'emotion'.

obtain the Brain_image of the Activation_map (?i; ?j; ?k; ?p) as Image.
"""
```

- [ ] **Step 4: Update the result retrieval to use `"image"`**

Change:
```python
result_image = (
    solution["img"]
    .as_pandas_dataframe()
    .iloc[0, 0]
    .spatial_image()
)
```

To:
```python
result_image = (
    solution["image"]
    .as_pandas_dataframe()
    .iloc[0, 0]
    .spatial_image()
)
```

- [ ] **Step 5: Update the module docstring**

At the top of the file (the `r"""..."""` docstring), update the embedded SQUALL program in the `.. code-block:: text` section and the comment prose to reflect the new names and syntax. Replace the old program block with:

```
define as Reported_voxel with a probability of
    the Kernelized_max_proximity of the Reported_focus (?i2; ?j2; ?k2; ?s)
        for each ?i1, ?j1, ?k1 and for each ?s
        where (?i1; ?j1; ?k1) is a Voxel
        and where EUCLIDEAN(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) is lower than 2.

define as Study_term every Term_in_study_with_tfidf (?s; ?t; _)
    where ?s is a Selected_study.

define as Active_voxel every Reported_voxel (?i; ?j; ?k; ?s)
    where ?s is a Selected_study.

define as Activation_map with inferred probability every Active_voxel (?i; ?j; ?k; _)
    given every Study_term (_; ?t) where ?t is 'emotion'.

obtain the Brain_image of the Activation_map (?i; ?j; ?k; ?p) as Image.
```

Also update the inline comment block at lines ~151–165 to match new names (``Kernelized_max_proximity``, ``Reported_focus``, ``for each``, ``where``, ``given``, ``obtain … as Image``).

- [ ] **Step 6: Verify the file parses cleanly**

```bash
python -c "import ast; ast.parse(open('examples/plot_squall_cbma_spatial_prior.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Run the SQUALL program through the parser (smoke test)**

```bash
python -c "
from neurolang.frontend.datalog.squall_syntax_lark import parser
prog = open('examples/plot_squall_cbma_spatial_prior.py').read()
# Extract squall_program string manually for a quick parse check
import re
m = re.search(r'squall_program = \"\"\"(.*?)\"\"\"', prog, re.DOTALL)
result = parser(m.group(1).strip())
print(type(result).__name__)
"
```

Expected: `SquallProgram` (or `Implication`/`Union` if no `obtain` — should be `SquallProgram` after Extension G).

- [ ] **Step 8: Commit**

```bash
git add examples/plot_squall_cbma_spatial_prior.py
git commit -m "feat(example): rename predicates and adopt extensions D-H in CBMA SQUALL script"
```
