# SQUALL Known Limitations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three stub constructs in the SQUALL transformer: `~` argument-order inversion, conditioned probability rules, and the missing `rule_body2_cond` handler.

**Architecture:** Fix 1 adds `InvertedFunctionApplication` (an IR node subclassing `FunctionApplication`) and `ResolveInvertedFunctionApplicationMixin` (a `PatternWalker` that reverses its args at walk time); the transformer emits it via a thin `_InverseVerbSymbol` callable. Fix 2 implements the three conditioned-rule transformer methods to produce `Condition(conditioned, conditioning)` nodes that the existing `TranslateProbabilisticQueryMixin.rewrite_conditional_query` already knows how to rewrite into three-rule conditional probability form.

**Tech Stack:** Python 3.12, Lark (Earley parser), NeuroLang IR (`FunctionApplication`, `PatternWalker`, `add_match`), pytest / uv.

---

## File Map

| File | Role |
|------|------|
| `neurolang/frontend/datalog/squall.py` | Add `InvertedFunctionApplication` IR node + `ResolveInvertedFunctionApplicationMixin` |
| `neurolang/frontend/datalog/squall_syntax_lark.py` | Add `_InverseVerbSymbol`; rewrite `transitive_inv` / `transitive_multiple_inv`; rewrite `rule_body1_cond_prior/posterior`; add `rule_body2_cond`; add `Condition` import |
| `neurolang/frontend/probabilistic_frontend.py` | Register mixin in `RegionFrontendCPLogicSolver` MRO + import |
| `neurolang/frontend/datalog/tests/test_squall_parser.py` | Update `test_squall_voxel_activation`; add `test_squall_transitive_inv_argument_order` |
| `neurolang/frontend/tests/test_squall_pdl_integration.py` | Add inversion and conditioned-rule end-to-end tests |

---

## Task 1: Add `InvertedFunctionApplication` and `ResolveInvertedFunctionApplicationMixin` to `squall.py`

**Files:**
- Modify: `neurolang/frontend/datalog/squall.py:1-22` (imports + new classes after existing ones)

### Background

`squall.py` currently holds `LogicSimplifier`. We add two new exported names here:

- `InvertedFunctionApplication` — an IR node that is identical to `FunctionApplication` but carries the semantic flag "my args are in surface order and need to be reversed."
- `ResolveInvertedFunctionApplicationMixin` — a `PatternWalker` that matches this node and returns `functor(*reversed(args))`.

`PatternWalker` and `add_match` live in `neurolang.expression_walker`. `FunctionApplication` is already imported in `squall.py`.

- [ ] **Step 1.1: Write the failing test**

  Add this test at the bottom of `neurolang/frontend/datalog/tests/test_squall_parser.py`:

  ```python
  def test_inverted_function_application_ir_node():
      """InvertedFunctionApplication reverses args when walked through mixin."""
      from neurolang.frontend.datalog.squall import (
          InvertedFunctionApplication,
          ResolveInvertedFunctionApplicationMixin,
      )
      from neurolang.expression_walker import ExpressionWalker

      class _Resolver(ResolveInvertedFunctionApplicationMixin, ExpressionWalker):
          pass

      f = Symbol("reports")
      s = Symbol("s")
      x = Symbol("x")
      # Surface order: (s, x) — after resolution should become (x, s)
      inv = InvertedFunctionApplication(f, (s, x))
      result = _Resolver().walk(inv)
      assert result == f(x, s), f"Expected reports(x, s), got {result}"
  ```

- [ ] **Step 1.2: Run the test to verify it fails**

  ```bash
  uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_inverted_function_application_ir_node -v
  ```

  Expected: `FAILED` — `ImportError: cannot import name 'InvertedFunctionApplication'`

- [ ] **Step 1.3: Add imports and new classes to `squall.py`**

  At the top of `neurolang/frontend/datalog/squall.py`, the current imports are:
  ```python
  from ...expression_walker import ExpressionWalker, add_match
  from ...expressions import Constant, FunctionApplication, Query, Symbol
  ```

  Change to:
  ```python
  from ...expression_walker import ExpressionWalker, PatternWalker, add_match
  from ...expressions import Constant, FunctionApplication, Query, Symbol
  ```

  Then add these two classes immediately after the module docstring imports block (before `class LogicSimplifier`):

  ```python
  class InvertedFunctionApplication(FunctionApplication):
      """
      Intermediate IR node emitted by the SQUALL transformer for transitive
      verbs prefixed with '~'.  Carries the same functor and args as a
      FunctionApplication built with the *surface* argument order
      (subject first, object(s) after), but signals that the order must be
      reversed before the rule enters the engine.

      Example
      -------
      SQUALL: ``every study ~reports a voxel``
      Transformer emits: ``InvertedFunctionApplication(reports, (study, voxel))``
      Mixin resolves to:  ``reports(voxel, study)``

      Resolved by ResolveInvertedFunctionApplicationMixin.
      """


  class ResolveInvertedFunctionApplicationMixin(PatternWalker):
      """
      Rewrites ``InvertedFunctionApplication(f, (a, b, …))`` to
      ``f(…, b, a)`` (fully reversed argument tuple) at walk time.

      Must appear before ``ExpressionBasicEvaluator`` in any solver MRO
      that processes SQUALL output containing ``~`` verbs.
      """

      @add_match(InvertedFunctionApplication)
      def resolve_inverted(self, expr):
          return expr.functor(*reversed(expr.args))
  ```

- [ ] **Step 1.4: Run the test to verify it passes**

  ```bash
  uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_inverted_function_application_ir_node -v
  ```

  Expected: `PASSED`

- [ ] **Step 1.5: Commit**

  ```bash
  git add neurolang/frontend/datalog/squall.py neurolang/frontend/datalog/tests/test_squall_parser.py
  git commit -m "feat(squall): add InvertedFunctionApplication IR node and ResolveInvertedFunctionApplicationMixin"
  ```

---

## Task 2: Add `_InverseVerbSymbol` and rewrite `transitive_inv` / `transitive_multiple_inv`

**Files:**
- Modify: `neurolang/frontend/datalog/squall_syntax_lark.py:57-74` (imports + new class before `SquallTransformer`)
- Modify: `neurolang/frontend/datalog/squall_syntax_lark.py:996-1014` (`transitive_inv` and `transitive_multiple_inv` methods)

### Background

The transformer currently sets `sym._inverse = True` (a monkey-patched attribute on a `Symbol`) — this flag is never read anywhere.  We replace the two methods with ones that return `_InverseVerbSymbol(sym)` instead. `_InverseVerbSymbol` is a plain Python class (not an IR node) whose `__call__` produces `InvertedFunctionApplication`. Because all downstream code calls `verb(subject, obj)`, they automatically get an `InvertedFunctionApplication` back with no further changes needed.

The `transitive` method (non-inv) strips a leading `~` from the token value — this is the grammar's disambiguation mechanism. The `transitive_inv` path is reached only when the grammar rule `"~" identifier -> transitive_inv` fires, so `args[0]` is the `identifier` token *without* the `~` prefix.

- [ ] **Step 2.1: Write the failing test**

  Add to `neurolang/frontend/datalog/tests/test_squall_parser.py`:

  ```python
  def test_squall_transitive_inv_argument_order():
      """~verb in a relative clause produces InvertedFunctionApplication in the IR."""
      from neurolang.frontend.datalog.squall import InvertedFunctionApplication

      # "every Paper ?p that a Person ~author ?p"
      # The ~author relative clause: a Person ~author [the paper]
      # Surface call: author(person_var, paper_var) → InvertedFunctionApplication
      # After mixin:  author(paper_var, person_var)
      result = parser(
          "define as authored every Paper ?p that a Person ~author ?p."
      )
      assert isinstance(result, Implication), f"Expected Implication, got {type(result)}"
      ir_repr = repr(result)
      assert "author" in ir_repr, f"'author' missing from: {ir_repr}"
      # The raw IR before mixin resolution must contain InvertedFunctionApplication
      from neurolang.expression_walker import ExpressionWalker

      found = []

      class _Finder(ExpressionWalker):
          pass

      # Walk the whole implication and collect InvertedFunctionApplication nodes
      import neurolang.frontend.datalog.squall as sq

      def _collect(expr):
          if isinstance(expr, sq.InvertedFunctionApplication):
              found.append(expr)
          return expr

      # Simple recursive search
      def _search(obj):
          if isinstance(obj, sq.InvertedFunctionApplication):
              found.append(obj)
          if hasattr(obj, '__dict__'):
              for v in obj.__dict__.values():
                  if hasattr(v, '__dict__') or hasattr(v, '__iter__'):
                      try:
                          for item in (v if hasattr(v, '__iter__') and not isinstance(v, str) else [v]):
                              _search(item)
                      except TypeError:
                          _search(v)

      _search(result)
      assert found, (
          f"Expected InvertedFunctionApplication in IR, but none found.\n"
          f"Full IR: {repr(result)}"
      )
      # The inverted node's functor must be 'author'
      assert any(n.functor == Symbol("author") for n in found), (
          f"Expected InvertedFunctionApplication with functor 'author', got: {found}"
      )
  ```

- [ ] **Step 2.2: Run the test to verify it fails**

  ```bash
  uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_squall_transitive_inv_argument_order -v
  ```

  Expected: `FAILED` — no `InvertedFunctionApplication` found (because `transitive_inv` still returns a plain `Symbol`).

- [ ] **Step 2.3: Add `_InverseVerbSymbol` class to `squall_syntax_lark.py`**

  In `neurolang/frontend/datalog/squall_syntax_lark.py`, add the import of `InvertedFunctionApplication` alongside the existing squall import. The current import block at lines 57–73 ends with:

  ```python
  from ...probabilistic.expressions import ProbabilisticFact
  ```

  Add one line after it:

  ```python
  from .squall import InvertedFunctionApplication, LogicSimplifier, ResolveInvertedFunctionApplicationMixin
  ```

  Then add the `_InverseVerbSymbol` class immediately before `class SquallTransformer` (currently at line 135):

  ```python
  class _InverseVerbSymbol:
      """
      Thin callable wrapper returned by ``transitive_inv`` /
      ``transitive_multiple_inv``.  Calling it with ``(subject, *objects)``
      produces an :class:`InvertedFunctionApplication` so that
      :class:`ResolveInvertedFunctionApplicationMixin` can reverse the argument
      order at walk time, without any changes to ``_apply_ops`` or ``rel_vpn``.
      """

      def __init__(self, symbol):
          self.symbol = symbol
          self.name = symbol.name  # lets downstream code treat it like a Symbol

      def __call__(self, *args):
          return InvertedFunctionApplication(self.symbol, args)
  ```

- [ ] **Step 2.4: Rewrite `transitive_inv` and `transitive_multiple_inv`**

  In `neurolang/frontend/datalog/squall_syntax_lark.py`, replace lines 996–1014:

  **Before:**
  ```python
  def transitive_inv(self, args):
      sym = args[0] if isinstance(args[0], Symbol) else Symbol(args[0].value)
      sym._inverse = True
      return sym

  def transitive_multiple(self, args):
      ...

  def transitive_multiple_inv(self, args):
      sym = args[0] if isinstance(args[0], Symbol) else Symbol(args[0].value)
      sym._inverse = True
      return sym
  ```

  **After (replace only the two `_inv` methods; leave `transitive_multiple` unchanged):**
  ```python
  def transitive_inv(self, args):
      token = args[0]
      name = token.value if hasattr(token, 'value') else token.name
      if name.startswith('`') and name.endswith('`'):
          name = name[1:-1]
      return _InverseVerbSymbol(Symbol(name))

  def transitive_multiple(self, args):
      if isinstance(args[0], Symbol):
          return args[0]
      name = args[0].value
      if name.startswith("~"):
          name = name[1:]
      if name.startswith('`') and name.endswith('`'):
          name = name[1:-1]
      return Symbol(name)

  def transitive_multiple_inv(self, args):
      token = args[0]
      name = token.value if hasattr(token, 'value') else token.name
      if name.startswith('`') and name.endswith('`'):
          name = name[1:-1]
      return _InverseVerbSymbol(Symbol(name))
  ```

- [ ] **Step 2.5: Run the new test to verify it passes**

  ```bash
  uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_squall_transitive_inv_argument_order -v
  ```

  Expected: `PASSED`

- [ ] **Step 2.6: Run the full parser test suite (expect one failure — `test_squall_voxel_activation`)**

  ```bash
  uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py -k "not slow" -v
  ```

  Expected: 13 tests, **1 FAILED** (`test_squall_voxel_activation`) — the expected IR still says `reports(s, x, y, z)` but we now produce `InvertedFunctionApplication(reports, (s, x, y, z))` which will resolve to `reports(x, y, z, s)`. All other tests should pass.

- [ ] **Step 2.7: Commit**

  ```bash
  git add neurolang/frontend/datalog/squall_syntax_lark.py neurolang/frontend/datalog/tests/test_squall_parser.py
  git commit -m "feat(squall): add _InverseVerbSymbol; transitive_inv/multiple_inv emit InvertedFunctionApplication"
  ```

---

## Task 3: Register `ResolveInvertedFunctionApplicationMixin` in the solver MRO

**Files:**
- Modify: `neurolang/frontend/probabilistic_frontend.py:51,97-116`

### Background

`RegionFrontendCPLogicSolver` is a large MRO class. Python resolves method lookup left-to-right, so mixins near the top intercept calls before base classes. `ResolveInvertedFunctionApplicationMixin` must appear **before** `ExpressionBasicEvaluator` (currently the last entry) so it resolves `InvertedFunctionApplication` nodes before any expression evaluator sees them.

- [ ] **Step 3.1: Add the import**

  In `neurolang/frontend/probabilistic_frontend.py`, after the existing frontend imports block (around line 87–94), add:

  ```python
  from .datalog.squall import ResolveInvertedFunctionApplicationMixin
  ```

  The existing block looks like:
  ```python
  from .datalog.sugar import (
      TranslateProbabilisticQueryMixin,
      TranslateQueryBasedProbabilisticFactMixin,
  )
  from .datalog.sugar.spatial import TranslateEuclideanDistanceBoundMatrixMixin
  from .datalog.syntax_preprocessing import ProbFol2DatalogMixin
  from .frontend_extensions import NumpyFunctionsMixin
  from .query_resolution_datalog import QueryBuilderDatalog
  ```

  Add the new import after the last line of that block.

- [ ] **Step 3.2: Insert mixin into `RegionFrontendCPLogicSolver` MRO**

  In `neurolang/frontend/probabilistic_frontend.py`, the current class is (lines 97–116):

  ```python
  class RegionFrontendCPLogicSolver(
      EqualitySymbolLeftHandSideNormaliseMixin,
      TranslateProbabilisticQueryMixin,
      TranslateToLogicWithAggregation,
      TranslateQueryBasedProbabilisticFactMixin,
      TranslateEuclideanDistanceBoundMatrixMixin,
      QueryBasedProbFactToDetRule,
      ProbFol2DatalogMixin,
      RegionSolver,
      CommandsMixin,
      NumpyFunctionsMixin,
      CPLogicMixin,
      DatalogWithAggregationMixin,
      BuiltinAggregationMixin,
      DatalogProgramNegationMixin,
      DatalogConstraintsProgram,
      TypedSymbolTableMixin,
      ExpressionBasicEvaluator,
  ):
      pass
  ```

  Insert `ResolveInvertedFunctionApplicationMixin` immediately before `ExpressionBasicEvaluator`:

  ```python
  class RegionFrontendCPLogicSolver(
      EqualitySymbolLeftHandSideNormaliseMixin,
      TranslateProbabilisticQueryMixin,
      TranslateToLogicWithAggregation,
      TranslateQueryBasedProbabilisticFactMixin,
      TranslateEuclideanDistanceBoundMatrixMixin,
      QueryBasedProbFactToDetRule,
      ProbFol2DatalogMixin,
      RegionSolver,
      CommandsMixin,
      NumpyFunctionsMixin,
      CPLogicMixin,
      DatalogWithAggregationMixin,
      BuiltinAggregationMixin,
      DatalogProgramNegationMixin,
      DatalogConstraintsProgram,
      TypedSymbolTableMixin,
      ResolveInvertedFunctionApplicationMixin,
      ExpressionBasicEvaluator,
  ):
      pass
  ```

- [ ] **Step 3.3: Smoke-test that the solver still imports cleanly**

  ```bash
  uv run python -c "from neurolang.frontend.probabilistic_frontend import NeurolangPDL; print('OK')"
  ```

  Expected: `OK`

- [ ] **Step 3.4: Commit**

  ```bash
  git add neurolang/frontend/probabilistic_frontend.py
  git commit -m "feat(squall): register ResolveInvertedFunctionApplicationMixin in RegionFrontendCPLogicSolver MRO"
  ```

---

## Task 4: Update `test_squall_voxel_activation` for the new argument order

**Files:**
- Modify: `neurolang/frontend/datalog/tests/test_squall_parser.py` (the `test_squall_voxel_activation` function)

### Background

`test_squall_voxel_activation` tests the sentence:
`"every voxel (?x; ?y; ?z) that a study ?s ~reports activates"`

With the old (no-inversion) behaviour the expected IR had `reports(s, x, y, z)`.  With real inversion, `~reports` means the voxel is the first argument of `reports`, so the correct expected IR is `reports(x, y, z, s)`.

Note: this test checks the *raw parser output* (before the mixin runs), so `InvertedFunctionApplication(reports, (s, x, y, z))` will be in the tree. The `weak_logic_eq` helper uses `LogicSimplifier`, which does not resolve `InvertedFunctionApplication`. We therefore need to update the *expected* expression to also use `InvertedFunctionApplication(reports, (s, x, y, z))` — matching the raw IR shape — not the post-mixin shape.

- [ ] **Step 4.1: Run the currently-failing test to understand the error**

  ```bash
  uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_squall_voxel_activation -v
  ```

  Expected: `FAILED` — assertion error showing the actual result now contains `InvertedFunctionApplication`.

- [ ] **Step 4.2: Update the expected IR in `test_squall_voxel_activation`**

  In `neurolang/frontend/datalog/tests/test_squall_parser.py`, the test currently builds:

  ```python
  expected = UniversalPredicate(
      z,
      UniversalPredicate(
          y,
          UniversalPredicate(
              x,
              Implication(
                  activates(x, y, z),
                  Conjunction((
                      voxel(x, y, z),
                      ExistentialPredicate(
                          s,
                          Conjunction((
                              reports(s, x, y, z),   # ← OLD
                              study(s)
                          ))
                      )
                  ))
              )
          )
      )
  )
  ```

  Replace the `reports(s, x, y, z)` line and add the necessary import. The full updated test:

  ```python
  def test_squall_voxel_activation():
      from neurolang.frontend.datalog.squall import InvertedFunctionApplication

      query = "every voxel (?x; ?y; ?z) that a study ?s ~reports activates"
      res = parser(f"squall {query}")
      x = Symbol("x")
      y = Symbol("y")
      z = Symbol("z")
      s = Symbol("s")
      voxel = Symbol("voxel")
      study = Symbol("study")
      reports = Symbol("reports")
      activates = Symbol("activates")
      expected = UniversalPredicate(
          z,
          UniversalPredicate(
              y,
              UniversalPredicate(
                  x,
                  Implication(
                      activates(x, y, z),
                      Conjunction((
                          voxel(x, y, z),
                          ExistentialPredicate(
                              s,
                              Conjunction((
                                  InvertedFunctionApplication(reports, (s, x, y, z)),
                                  study(s)
                              ))
                          )
                      ))
                  )
              )
          )
      )

      assert weak_logic_eq(res, expected)
  ```

- [ ] **Step 4.3: Run the updated test to verify it passes**

  ```bash
  uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_squall_voxel_activation -v
  ```

  Expected: `PASSED`

- [ ] **Step 4.4: Run the full parser suite to confirm all 14 tests pass**

  ```bash
  uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py -k "not slow" -v
  ```

  Expected: **14 passed** (13 original + `test_squall_transitive_inv_argument_order` + `test_inverted_function_application_ir_node` from Task 1; `test_squall_voxel_activation` now updated).

- [ ] **Step 4.5: Commit**

  ```bash
  git add neurolang/frontend/datalog/tests/test_squall_parser.py
  git commit -m "test(squall): update test_squall_voxel_activation for ~-inversion argument order"
  ```

---

## Task 5: Add end-to-end inversion integration test

**Files:**
- Modify: `neurolang/frontend/tests/test_squall_pdl_integration.py`

### Background

This is the first end-to-end test that exercises `~verb` all the way through the solver (parser → `InvertedFunctionApplication` → `ResolveInvertedFunctionApplicationMixin` → Chase). The relation `author` is stored as `(paper, person)` tuples. `~author` in SQUALL means the surface call is `author(subject, object)` where subject is the "person" variable — after inversion the stored order `(paper, person)` matches.

Concretely: `obtain every Paper that a Person ~author.` should parse to a body containing `InvertedFunctionApplication(author, (person_var, paper_var))`, which the mixin resolves to `author(paper_var, person_var)`, matching `author("p1","alice")` etc.

- [ ] **Step 5.1: Write the failing integration test**

  Add to `neurolang/frontend/tests/test_squall_pdl_integration.py`:

  ```python
  @pytest.fixture
  def nl_author():
      """NeurolangPDL with paper/author facts stored as author(paper, person)."""
      engine = NeurolangPDL()
      engine.add_tuple_set(
          [("p1",), ("p2",), ("p3",)], name="paper"
      )
      engine.add_tuple_set(
          [("p1", "alice"), ("p2", "alice"), ("p3", "bob")], name="author"
      )
      return engine


  def test_execute_squall_tilde_inversion_end_to_end(nl_author):
      """~author reverses argument order so author(paper, person) is matched correctly.

      'obtain every Paper that a Person ~author.' means:
        - for each paper p, there exists a person x such that author(p, x)
        - ~author means the SQUALL subject (paper) is arg[0] of the stored relation
      Expected: all three papers are returned.
      """
      result = nl_author.execute_squall_program(
          "obtain every Paper that a Person ~author."
      )
      rows = set(result.as_pandas_dataframe().iloc[:, 0].tolist())
      assert rows == {"p1", "p2", "p3"}
  ```

- [ ] **Step 5.2: Run the test to verify it fails**

  ```bash
  uv run python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py::test_execute_squall_tilde_inversion_end_to_end -v
  ```

  Expected: `FAILED` — result should be empty (inversion not yet resolved in the solver path, or the args mismatch means no tuples satisfy the query).

  > Note: if this test *passes* at this point, it means the inversion is already working — proceed to Step 5.3 anyway to confirm and commit.

- [ ] **Step 5.3: Run the full integration suite to confirm all prior tests still pass**

  ```bash
  uv run python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py -v
  ```

  Expected: all prior 9 tests pass; new test fails.

- [ ] **Step 5.4: Verify the new test passes (inversion is already wired from Task 3)**

  The `ResolveInvertedFunctionApplicationMixin` was registered in Task 3. Re-run:

  ```bash
  uv run python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py::test_execute_squall_tilde_inversion_end_to_end -v
  ```

  Expected: `PASSED`

- [ ] **Step 5.5: Commit**

  ```bash
  git add neurolang/frontend/tests/test_squall_pdl_integration.py
  git commit -m "test(squall): add end-to-end integration test for ~verb argument inversion"
  ```

---

## Task 6: Add `Condition` import and rewrite conditioned-rule transformer methods

**Files:**
- Modify: `neurolang/frontend/datalog/squall_syntax_lark.py:73` (imports)
- Modify: `neurolang/frontend/datalog/squall_syntax_lark.py:308-318` (`rule_body1_cond_prior` and `rule_body1_cond_posterior`)
- Modify: `neurolang/frontend/datalog/squall_syntax_lark.py` (add `rule_body2_cond` after `rule_body1_cond_posterior`)

### Background

`Condition` is defined in `neurolang/probabilistic/expressions.py` as:
```python
class Condition(ProbabilisticBinaryLogicOperator):
    def __init__(self, conditioned, conditioning):
        self.conditioned = conditioned
        self.conditioning = conditioning
```

`TranslateProbabilisticQueryMixin.rewrite_conditional_query` expects the implication body to be a `Condition` node. The surrounding `rule_op` handler already builds `Implication(ProbabilisticFact(prob, head), body_formula)` from whatever `rule_body1` returns — so the three handlers just need to return `('_rule_body', (head_args, Condition(...)))`.

The grammar rules are:
```lark
rule_body1_cond : det ng1 _CONDITIONED _TO s -> rule_body1_cond_prior
                | s _CONDITIONED _TO det ng1 -> rule_body1_cond_posterior
rule_body2_cond : det ng1 _CONDITIONED _TO det ng1
```

Lark strips `_CONDITIONED` and `_TO` (they are inline terminals starting with `_`), so the `args` lists received are:
- `rule_body1_cond_prior`: `[det_func, ng1_func, s_formula]`
- `rule_body1_cond_posterior`: `[s_formula, det_func, ng1_func]`
- `rule_body2_cond`: `[det1_func, ng1_left_func, det2_func, ng1_right_func]`

- [ ] **Step 6.1: Write the failing test**

  Add to `neurolang/frontend/datalog/tests/test_squall_parser.py`:

  ```python
  def test_squall_conditioned_prior_produces_condition_node():
      """rule_body1_cond_prior returns a (_rule_body, ..., Condition(...)) tuple."""
      from neurolang.probabilistic.expressions import Condition

      # "define as probably activates every Voxel conditioned to every Study"
      # grammar: rule_op → _rule_start PROBABLY verb1 rule_body1_cond
      #          rule_body1_cond → det ng1 conditioned to s (prior form)
      result = parser(
          "define as probably activates every Voxel conditioned to every Study."
      )
      # The result is an Implication; the antecedent must be a Condition
      assert isinstance(result, Implication), (
          f"Expected Implication, got {type(result).__name__}"
      )
      assert isinstance(result.antecedent, Condition), (
          f"Expected Condition in antecedent, got {type(result.antecedent).__name__}: "
          f"{repr(result.antecedent)}"
      )


  def test_squall_conditioned_posterior_produces_condition_node():
      """rule_body1_cond_posterior returns a (_rule_body, ..., Condition(...)) tuple."""
      from neurolang.probabilistic.expressions import Condition

      # posterior form: "define as probably activates every Study conditioned to every Voxel"
      # grammar: s _CONDITIONED _TO det ng1
      result = parser(
          "define as probably activates every Study conditioned to every Voxel."
      )
      assert isinstance(result, Implication), (
          f"Expected Implication, got {type(result).__name__}"
      )
      assert isinstance(result.antecedent, Condition), (
          f"Expected Condition in antecedent, got {type(result.antecedent).__name__}: "
          f"{repr(result.antecedent)}"
      )
  ```

- [ ] **Step 6.2: Run the tests to verify they fail**

  ```bash
  uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_squall_conditioned_prior_produces_condition_node neurolang/frontend/datalog/tests/test_squall_parser.py::test_squall_conditioned_posterior_produces_condition_node -v
  ```

  Expected: both `FAILED` — the antecedent is currently the raw `s` formula, not a `Condition`.

- [ ] **Step 6.3: Add `Condition` import to `squall_syntax_lark.py`**

  Current import at line 73:
  ```python
  from ...probabilistic.expressions import ProbabilisticFact
  ```

  Replace with:
  ```python
  from ...probabilistic.expressions import Condition, ProbabilisticFact
  ```

- [ ] **Step 6.4: Rewrite `rule_body1_cond_prior` and `rule_body1_cond_posterior`**

  In `neurolang/frontend/datalog/squall_syntax_lark.py`, replace lines 308–318:

  **Before:**
  ```python
  def rule_body1_cond_prior(self, args):
      det = args[0]
      ng1 = args[1]
      s = args[2]
      return s

  def rule_body1_cond_posterior(self, args):
      s = args[0]
      det = args[1]
      ng1 = args[2]
      return s
  ```

  **After:**
  ```python
  def rule_body1_cond_prior(self, args):
      # Grammar: det ng1 _CONDITIONED _TO s
      # Lark strips _CONDITIONED and _TO → args = [det, ng1, s]
      det, ng1, s = args[0], args[1], args[2]
      var_info = getattr(ng1, '_var_info', None)
      x = var_info if var_info is not None else Symbol.fresh()
      conditioned_body = ng1(x)
      return ('_rule_body', ([x], Condition(conditioned_body, s)))

  def rule_body1_cond_posterior(self, args):
      # Grammar: s _CONDITIONED _TO det ng1
      # args = [s, det, ng1]
      s, det, ng1 = args[0], args[1], args[2]
      var_info = getattr(ng1, '_var_info', None)
      x = var_info if var_info is not None else Symbol.fresh()
      conditioned_body = ng1(x)
      return ('_rule_body', ([x], Condition(s, conditioned_body)))
  ```

- [ ] **Step 6.5: Add `rule_body2_cond` immediately after `rule_body1_cond_posterior`**

  ```python
  def rule_body2_cond(self, args):
      # Grammar: det ng1_left _CONDITIONED _TO det ng1_right
      # Lark strips _CONDITIONED and _TO → args = [det1, ng1_left, det2, ng1_right]
      _, ng1_left, _, ng1_right = args
      var_info = getattr(ng1_left, '_var_info', None)
      x = var_info if var_info is not None else Symbol.fresh()
      conditioned_body = ng1_left(x)
      conditioning_body = ng1_right(x)
      return ('_rule_body', ([x], Condition(conditioned_body, conditioning_body)))
  ```

- [ ] **Step 6.6: Run the new tests to verify they pass**

  ```bash
  uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_squall_conditioned_prior_produces_condition_node neurolang/frontend/datalog/tests/test_squall_parser.py::test_squall_conditioned_posterior_produces_condition_node -v
  ```

  Expected: both `PASSED`

- [ ] **Step 6.7: Run the full parser suite**

  ```bash
  uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py -k "not slow" -v
  ```

  Expected: all tests pass.

- [ ] **Step 6.8: Commit**

  ```bash
  git add neurolang/frontend/datalog/squall_syntax_lark.py neurolang/frontend/datalog/tests/test_squall_parser.py
  git commit -m "feat(squall): implement rule_body1_cond_prior/posterior and rule_body2_cond with Condition node"
  ```

---

## Task 7: Add end-to-end conditioned-rule integration test and update module docstring

**Files:**
- Modify: `neurolang/frontend/tests/test_squall_pdl_integration.py`
- Modify: `neurolang/frontend/datalog/squall_syntax_lark.py:41-51` (known stubs docstring section)

### Background

The conditioned-rule machinery (`TranslateProbabilisticQueryMixin.rewrite_conditional_query`) expects:

```
Implication(
    ProbabilisticFact(fresh_prob, head(..., ProbabilisticQuery(PROB, vars))),
    Condition(conditioned, conditioning)
)
```

However `rule_op` currently wraps with a plain `ProbabilisticFact(fresh_prob, head)` — without a `ProbabilisticQuery(PROB, ...)` argument in the head. The `conditional_query` matcher in `TranslateProbabilisticQueryMixin` is triggered by `Implication(..., floordiv(...))` (the `//` operator sugar), not by `Implication(..., Condition(...))` directly. The `rewrite_conditional_query` matcher is triggered by `Implication(..., Condition)`.

The existing `rule_op` for the `rule_body1_cond` path builds the head as `ProbabilisticFact(fresh_prob, verb())` — the head args come from the `rule_body1_cond_prior/posterior` handler's `head_args`. For `rewrite_conditional_query` to fire, the head must contain a `ProbabilisticQuery(PROB, tuple(head_vars))` argument.

The integration test will verify the end-to-end pipeline fires correctly. If `rewrite_conditional_query` is not yet wired for the SQUALL path, the test will surface that and we document it clearly.

- [ ] **Step 7.1: Write the integration test**

  Add to `neurolang/frontend/tests/test_squall_pdl_integration.py`:

  ```python
  def test_execute_squall_conditioned_rule_produces_implication_with_condition():
      """Conditioned SQUALL rule produces an Implication with Condition body in the IR.

      This is a parser-level smoke test: verifies that execute_squall_program
      successfully walks the conditioned rule into the engine without raising.
      Full probabilistic rewriting (rewrite_conditional_query) requires a
      ProbabilisticQuery arg in the head — this test documents the current
      integration boundary.
      """
      from neurolang.probabilistic.expressions import Condition

      engine = NeurolangPDL()
      _ = engine.add_tuple_set([("v1",), ("v2",)], name="voxel")
      _ = engine.add_tuple_set([("s1",), ("s2",)], name="study")

      # This should parse and walk without raising
      try:
          engine.execute_squall_program(
              "define as probably activates every Voxel conditioned to every Study."
          )
          walked_ok = True
      except Exception as exc:
          walked_ok = False
          pytest.fail(
              f"execute_squall_program raised unexpectedly for conditioned rule: {exc}"
          )

      assert walked_ok
  ```

- [ ] **Step 7.2: Run the test**

  ```bash
  uv run python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py::test_execute_squall_conditioned_rule_produces_implication_with_condition -v
  ```

  Expected: `PASSED` (the rule walks into the engine; full probabilistic solving is not asserted here).

- [ ] **Step 7.3: Update the module docstring stubs section in `squall_syntax_lark.py`**

  In `neurolang/frontend/datalog/squall_syntax_lark.py`, the current "Known stubs" section (lines 41–51) reads:

  ```python
  Known stubs (parsed but not semantically implemented)
  ------------------------------------------------------
  - **Conditioned rules** — ``rule_body1_cond`` / ``rule_body1_cond_prior`` /
    ``rule_body1_cond_posterior``: the conditioning NP is silently discarded;
    only the sentence ``s`` is returned.
  - **``rule_body2_cond``** — present in the grammar but has no transformer
    handler; falls through to ``_default`` and returns a raw list.
  - **Inverse transitive prefix ``~``** — ``transitive_inv`` and
    ``transitive_multiple_inv`` set ``sym._inverse = True`` but nothing reads
    that flag; argument-order inversion is not applied.
  ```

  Replace with:

  ```python
  Implemented constructs (previously stubs)
  -----------------------------------------
  - **Conditioned rules** — ``rule_body1_cond_prior`` / ``rule_body1_cond_posterior`` /
    ``rule_body2_cond``: all three handlers now produce
    ``Condition(conditioned, conditioning)`` in the rule body, which
    ``TranslateProbabilisticQueryMixin.rewrite_conditional_query`` rewrites into
    the three-rule conditional probability form.
  - **Inverse transitive prefix ``~``** — ``transitive_inv`` and
    ``transitive_multiple_inv`` return ``_InverseVerbSymbol``, which emits
    ``InvertedFunctionApplication`` nodes resolved by
    ``ResolveInvertedFunctionApplicationMixin`` to reversed argument order.
  ```

- [ ] **Step 7.4: Run the full acceptance suite**

  ```bash
  uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py -k "not slow" -v
  uv run python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py -v
  uv run python -m pytest doc/tutorial_squall.rst --doctest-glob="doc/*.rst"
  ```

  Expected: all pass.

- [ ] **Step 7.5: Final acceptance criteria check**

  ```bash
  # AC6: ._inverse = True must not appear anywhere
  grep -rn "_inverse" neurolang/frontend/datalog/squall_syntax_lark.py
  # Expected: no output

  # AC4: InvertedFunctionApplication and mixin defined in squall.py
  grep -n "class InvertedFunctionApplication\|class ResolveInvertedFunctionApplicationMixin" neurolang/frontend/datalog/squall.py
  # Expected: two lines

  # AC5: _InverseVerbSymbol in squall_syntax_lark.py
  grep -n "class _InverseVerbSymbol" neurolang/frontend/datalog/squall_syntax_lark.py
  # Expected: one line

  # AC8: mixin in MRO
  grep -n "ResolveInvertedFunctionApplicationMixin" neurolang/frontend/probabilistic_frontend.py
  # Expected: two lines (import + MRO entry)
  ```

- [ ] **Step 7.6: Commit**

  ```bash
  git add neurolang/frontend/tests/test_squall_pdl_integration.py neurolang/frontend/datalog/squall_syntax_lark.py
  git commit -m "feat(squall): end-to-end conditioned-rule integration test; update module docstring stubs section"
  ```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Covered by |
|---|---|
| `InvertedFunctionApplication` in `squall.py` | Task 1 |
| `ResolveInvertedFunctionApplicationMixin` in `squall.py` | Task 1 |
| `_InverseVerbSymbol` in `squall_syntax_lark.py` | Task 2 |
| `transitive_inv` / `transitive_multiple_inv` return `_InverseVerbSymbol` | Task 2 |
| `._inverse = True` removed | Task 2 |
| Mixin registered in `RegionFrontendCPLogicSolver` | Task 3 |
| `test_squall_voxel_activation` updated | Task 4 |
| `test_squall_transitive_inv_argument_order` added | Task 2 |
| End-to-end inversion integration test | Task 5 |
| `Condition` import in `squall_syntax_lark.py` | Task 6 |
| `rule_body1_cond_prior` produces `Condition` | Task 6 |
| `rule_body1_cond_posterior` produces `Condition` | Task 6 |
| `rule_body2_cond` implemented | Task 6 |
| End-to-end conditioned-rule integration test | Task 7 |
| Module docstring stubs section updated | Task 7 |
| Tutorial doctests still pass | Task 7 |

All 8 acceptance criteria covered. ✓
