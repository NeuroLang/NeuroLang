# SQUALL Known Limitations — Fix Design

**Date:** 2026-04-15
**Branch:** squall-cnl
**Scope:** Implement the three known-stub constructs in the SQUALL transformer:
  1. `~` inverse transitive prefix (argument-order reversal)
  2. Conditioned rules (`define as probably … conditioned to …`)
  3. `rule_body2_cond` transformer handler (two-sided conditioned NPs)

---

## Background

Three constructs are parsed by `neurolang_natural.lark` but produce incorrect or missing IR:

| Stub | Grammar rule(s) | Current behaviour |
|------|----------------|-------------------|
| `~verb` inversion | `transitive_inv`, `transitive_multiple_inv` | Sets `._inverse = True` on Symbol; flag is never read. Argument order is NOT reversed. |
| Conditioned rules (prior/posterior) | `rule_body1_cond_prior`, `rule_body1_cond_posterior` | Returns raw sentence `s`, silently discarding the conditioning NP. |
| Two-sided conditioned NP | `rule_body2_cond` | No transformer method; `_default` returns a raw list. |

---

## Fix 1 — `~` Inversion via `InvertedFunctionApplication` + Mixin

### Design principle

Rather than patching argument order at parse time (fragile, spreads logic across
`_apply_ops` / `rel_vpn` / other call sites), the transformer emits a typed IR node
`InvertedFunctionApplication` wherever `~verb` appears in subject position.  A
dedicated `PatternWalker` mixin resolves it to a plain `FunctionApplication` with
reversed argument order when the expression is walked into the engine.  This keeps
the transformer simple, makes the inversion explicit in the IR, and follows the
established NeuroLang pattern (cf. `TranslateProbabilisticQueryMixin`).

### New IR node

**File:** `neurolang/frontend/datalog/squall.py`

```python
class InvertedFunctionApplication(FunctionApplication):
    """
    Intermediate IR node emitted by the SQUALL transformer for transitive
    verbs prefixed with '~'.  Carries the same functor and args as a
    FunctionApplication built with the *surface* argument order
    (subject first, object second), but signals that the order must be
    reversed before the rule enters the engine.

    Example
    -------
    SQUALL: ``every study ~reports a voxel``
    Transformer emits: ``InvertedFunctionApplication(reports, (study, voxel))``
    Mixin resolves to: ``reports(voxel, study)``

    Resolved by ResolveInvertedFunctionApplicationMixin.
    """
```

`InvertedFunctionApplication` subclasses `FunctionApplication` verbatim — no new
`__init__`, no new fields.  All existing walkers that pattern-match on
`FunctionApplication` will also match it, which is safe because the mixin runs
**before** any solver or chase logic.

### New mixin

**File:** `neurolang/frontend/datalog/squall.py`

```python
class ResolveInvertedFunctionApplicationMixin(PatternWalker):
    """
    Rewrites InvertedFunctionApplication(f, (a, b, …)) to f(…, b, a)
    (fully reversed argument tuple) at walk time.
    Must appear early in the solver MRO, before ExpressionBasicEvaluator.
    """
    @add_match(InvertedFunctionApplication)
    def resolve_inverted(self, expr):
        return expr.functor(*reversed(expr.args))
```

### Transformer changes

**File:** `neurolang/frontend/datalog/squall_syntax_lark.py`

`transitive_inv` and `transitive_multiple_inv` no longer set `._inverse`.
Instead they return a `Symbol` wrapped in a thin callable class
`_InverseVerbSymbol` defined in `squall_syntax_lark.py` (private, not exported):

```python
class _InverseVerbSymbol:
    """
    Thin wrapper returned by transitive_inv / transitive_multiple_inv.
    Calling it with (subject, *objects) produces an
    InvertedFunctionApplication so that ResolveInvertedFunctionApplicationMixin
    can reverse the argument order at walk time.
    """
    def __init__(self, symbol: Symbol):
        self.symbol = symbol
        self.name = symbol.name   # lets caller code treat it like a Symbol

    def __call__(self, *args):
        return InvertedFunctionApplication(self.symbol, args)
```

`transitive_inv`:
```python
def transitive_inv(self, args):
    name = args[0].value if hasattr(args[0], 'value') else args[0].name
    if name.startswith('`') and name.endswith('`'):
        name = name[1:-1]
    return _InverseVerbSymbol(Symbol(name))
```

`transitive_multiple_inv`: identical.

No changes to `_apply_ops`, `vpdo_vn`, or `rel_vpn` — they call `verb(subject, obj)`
and get an `InvertedFunctionApplication` automatically because `_InverseVerbSymbol.__call__`
produces one.

### Registration

**File:** `neurolang/frontend/probabilistic_frontend.py`

Add `ResolveInvertedFunctionApplicationMixin` to `RegionFrontendCPLogicSolver`'s MRO,
immediately before `ExpressionBasicEvaluator` (last in the list):

```python
class RegionFrontendCPLogicSolver(
    ...
    ResolveInvertedFunctionApplicationMixin,   # ← new
    ExpressionBasicEvaluator,
):
```

Import added at top of `probabilistic_frontend.py`:
```python
from .datalog.squall import ResolveInvertedFunctionApplicationMixin
```

### Test changes

**File:** `neurolang/frontend/datalog/tests/test_squall_parser.py`

`test_squall_voxel_activation`:  the expected IR for
`every voxel (?x; ?y; ?z) that a study ?s ~reports activates` changes from
`reports(s, x, y, z)` to `reports(x, y, z, s)` (voxel coordinates first,
study variable last — the reversed order).

New parser-level test `test_squall_transitive_inv_argument_order`:
```
SQUALL: "define as authored every Paper ?p that a Person ~author ?p."
Expected: Implication body contains InvertedFunctionApplication(author, (person_var, p))
After mixin resolves: author(p, person_var)  ← reversed: paper is arg[0]
```

New end-to-end integration test in `test_squall_pdl_integration.py`:
```
relations: author(paper, person) tuples:
    [("p1", "alice"), ("p2", "alice"), ("p3", "bob")]
SQUALL: "obtain every Paper that a Person ~author."
Semantics: "papers where some person is the author-of"
           → author(paper, person) after inversion
Expected result rows: {("p1",), ("p2",), ("p3",)}
```

---

## Fix 2 — Conditioned Rules

### Background

`TranslateProbabilisticQueryMixin.rewrite_conditional_query` (in
`neurolang/frontend/datalog/sugar/__init__.py`) already handles:

```
Implication(head_with_PROB_arg, Condition(conditioned_body, conditioning_body))
```

and rewrites it into three rules (numerator, denominator, final conditional).
The SQUALL transformer just needs to produce this shape.

### Semantics

The conditioning clause goes in the **body**, not the head.

| SQUALL pattern | `Condition` structure |
|---|---|
| `define as probably verb every A conditioned to s` | `Condition(A_body, s_body)` |
| `define as probably verb s conditioned to every A` | `Condition(s_body, A_body)` |
| `rule_body2_cond: det ng1 conditioned to det ng1` | `Condition(ng1_left_body, ng1_right_body)` |

The surrounding `rule_op` already builds
`Implication(ProbabilisticFact(prob, head), body)` from whatever body the
`rule_body1` handler returns — so returning `Condition(...)` as the body
is all that is needed.

### Transformer changes

**File:** `neurolang/frontend/datalog/squall_syntax_lark.py`

Replace the current stub handlers:

```python
def rule_body1_cond_prior(self, args):
    # Grammar: det ng1 _CONDITIONED _TO s
    # args order produced by Lark: [det, ng1, s]
    det, ng1, s = args[0], args[1], args[2]
    var_info = getattr(ng1, '_var_info', None)
    x = var_info if var_info is not None else Symbol.fresh()
    conditioned_body = ng1(x)
    return ('_rule_body', ([x], Condition(conditioned_body, s)))

def rule_body1_cond_posterior(self, args):
    # Grammar: s _CONDITIONED _TO det ng1
    # args order: [s, det, ng1]
    s, det, ng1 = args[0], args[1], args[2]
    var_info = getattr(ng1, '_var_info', None)
    x = var_info if var_info is not None else Symbol.fresh()
    conditioned_body = ng1(x)
    return ('_rule_body', ([x], Condition(s, conditioned_body)))
```

New handler (currently falling to `_default`):

```python
def rule_body2_cond(self, args):
    # Grammar: det ng1_left _CONDITIONED _TO det ng1_right
    # args order: [det1, ng1_left, det2, ng1_right]
    _, ng1_left, _, ng1_right = args
    var_info = getattr(ng1_left, '_var_info', None)
    x = var_info if var_info is not None else Symbol.fresh()
    conditioned_body = ng1_left(x)
    conditioning_body = ng1_right(x)
    return ('_rule_body', ([x], Condition(conditioned_body, conditioning_body)))
```

**New import** in `squall_syntax_lark.py`:
```python
from ...probabilistic.expressions import Condition
```
(alongside the existing `from ...probabilistic.expressions import ProbabilisticFact`)

### No changes needed to `rule_op`

`rule_op` already unpacks `('_rule_body', (head_args, body_formula))` and builds
`Implication(ProbabilisticFact(prob, head), body_formula)`.  Passing
`Condition(...)` as `body_formula` is sufficient — `rewrite_conditional_query`
handles the rest.

---

## Files Changed Summary

| File | Change |
|------|--------|
| `neurolang/frontend/datalog/squall.py` | Add `InvertedFunctionApplication`, `ResolveInvertedFunctionApplicationMixin` |
| `neurolang/frontend/datalog/squall_syntax_lark.py` | Add `_InverseVerbSymbol`; rewrite `transitive_inv` / `transitive_multiple_inv`; rewrite `rule_body1_cond_prior` / `rule_body1_cond_posterior`; add `rule_body2_cond`; add `Condition` import |
| `neurolang/frontend/probabilistic_frontend.py` | Register `ResolveInvertedFunctionApplicationMixin` in `RegionFrontendCPLogicSolver` MRO; add import |
| `neurolang/frontend/datalog/tests/test_squall_parser.py` | Update `test_squall_voxel_activation`; add `test_squall_transitive_inv_argument_order` |
| `neurolang/frontend/tests/test_squall_pdl_integration.py` | Add conditioned-rule and inversion end-to-end tests |

---

## Acceptance Criteria

1. `uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py -k "not slow"` — all tests pass (including updated `test_squall_voxel_activation`)
2. `uv run python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py` — all tests pass (including new conditioned-rule and inversion tests)
3. `uv run python -m pytest doc/tutorial_squall.rst --doctest-glob="doc/*.rst"` — tutorial doctests still pass
4. `InvertedFunctionApplication` and `ResolveInvertedFunctionApplicationMixin` are defined in `squall.py`
5. `_InverseVerbSymbol` is defined in `squall_syntax_lark.py`; `transitive_inv` and `transitive_multiple_inv` return instances of it
6. `._inverse = True` does not appear anywhere in the codebase
7. `rule_body1_cond_prior`, `rule_body1_cond_posterior`, and `rule_body2_cond` all return `('_rule_body', ...)` tuples containing a `Condition` node
8. `ResolveInvertedFunctionApplicationMixin` appears in `RegionFrontendCPLogicSolver`'s MRO
