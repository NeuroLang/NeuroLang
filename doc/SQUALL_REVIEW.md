# SQUALL Semantic Error Review

## Summary

The SQUALL parser (`squall_syntax_lark.py`) now raises **explicit, descriptive errors** instead of silently producing syntactically valid but semantically incorrect IR. This review documents the changes, the error categories, and what users and maintainers should expect.

## Changes

### 1. New Error Class: `SquallSemanticError`

- **Location**: `neurolang/exceptions.py`
- **Fields**: `message`, `line`, `column`, `source_line`
- **Display**: `__str__` renders a caret-pointer format showing the offending line and position

### 2. Position Tracking (`SquallTransformer`)

- **`__init__`**: Accepts `source_lines` parameter (list of source code lines)
- **`_capture_pos(token)`**: Records `line`/`column` from Lark Token objects
- **`_make_error(message)`**: Creates `SquallSemanticError` with captured position and source line
- **`_default(data, children, meta)`**: Captures position from rule-level metadata (fallback)
- **Terminal handlers** with `_capture_pos` calls:
  - `intransitive`, `transitive`, `transitive_inv`
  - `transitive_multiple`, `transitive_multiple_inv`
  - `upper_identifier`, `identifier`, `label_identifier`

### 3. Error Raising (Silent Fallback → Error)

Silent fallbacks that produced `Constant(True)` or truncated IR now raise `SquallSemanticError`:

| Handler | Condition | Message |
|---------|-----------|---------|
| `rule_op` | `body_result` not `_rule_body` | "expected a noun phrase after the verb" |
| `rule_op_prob` | `body_result` not `_rule_body` | same |
| `rule_op_marg` | `body_result` not `_rule_body` | same |
| `rule_opnn` / `rule_opnn_per` | `body_result` not `_rule_body2` | "expected 'for every X and for every Y where ...'" |
| `rule_opnn_compound` | Same | Same |
| `rule_opnn_prob` | Same | Same |
| `rule_opnn_marg` | Same | Same |
| `rule_opnn_per_compound` | Same | Same |
| `rule_body1` | `det` or `ng1` is `None` | "missing determiner or noun group" / "incomplete rule body" |
| `comparison()` | Unrecognised comparison key | Lists valid comparison operators |
| `agg_func()` | Unrecognised aggregation name | Lists valid: count, sum, max, min, average |
| `npc_det()` | Empty npc (no determiners) | "expected a noun or determiner + noun" |
| `expr_np()` | Uninterpretable value | "could not be interpreted as a value expression" |
| `dim_agg()` | Missing/bad relation reference | "Expected a relation name" |
| `_apply_to_vars` | `except TypeError` narrowed | Prevents hiding genuine errors |
| `det_the` anaphora | `noun_name` not in `_symbol_scope` when scope is non-empty | "'the X' was not introduced by any preceding 'for every X' clause" |

### 4. Parse-level Error Wrapping

The `parser()` function wraps Lark exceptions:

| Lark Error | Wraps to |
|-----------|----------|
| `UnexpectedToken` | `UnexpectedTokenError` |
| `UnexpectedCharacters` | `UnexpectedCharactersError` |
| `VisitError` (inner `SquallSemanticError`) | Re-raises inner error |
| `VisitError` (other) | `SquallSemanticError` with rule name |

### 5. CPS Chain Fix for `rel_vpn`

**Before**: `rel_vpn` ignored the `ops` (object NP) argument — transitive verbs in WHERE clauses always called `verb(y, x)` directly, bypassing the CPS chain. This meant `det_the` for object-position NPs (e.g., "a Study reports **the Term**") never invoked its `apply_d`, so anaphora checks never fired.

**After**: When `ops` is present, `rel_vpn` routes through `ops(lambda z: verb(y, z))`, ensuring the object NP's CPS is invoked and anaphora errors are detected.

### 6. Scope Registration Fix for `rule_body1`

**Before**: `ng1(body_args)` was called before the noun name was registered in `_symbol_scope`, so `det_the` always saw an empty scope in single-quantifier rules.

**After**: The noun name is registered before `ng1(body_args)` is evaluated, just as `quant_clause_ng1` already does for compound quantifiers.

## Error Semantics

### "the X" Anaphora

- **When `_symbol_scope` is non-empty** (inside a `for ... where` rule): If "the X" does not match any noun introduced by a preceding `for every X`, the parser raises `SquallSemanticError`.
- **When `_symbol_scope` is empty** (outside any `for every` clause, e.g., in `obtain` queries or standalone noun phrases): "the X" falls through to existential-creation semantics — a fresh variable `∃x. X(x) ∧ ...` is created. This is the pre-existing behaviour and has not changed.

### Aggregation Functions

- Previously, unrecognized aggregation function names silently defaulted to `count` (via `_AGG_FUNC_MAP.get(name, Constant(len))`).
- Now they raise `SquallSemanticError` with a list of valid functions.

### Comparison Operators

- Previously, unrecognized comparison operators produced a `StopIteration` or `None` leading to downstream errors.
- Now they raise `SquallSemanticError` with a list of valid operators.

## Tests

### New: `test_squall_errors.py`

- 8 tests covering anaphora errors (in subject/object position, single/compound rules, legitimate resolution) and parse errors.

### Existing: `test_squall_parser.py`

- All 34 tests pass (1 skipped, unrelated to these changes).

## Limitations

1. **`propagate_positions=True` on Earley** causes ambiguity resolution changes that collapse compound quantifier variants into wrong handlers. This is permanently disabled.
2. **Position accuracy**: Not all rule-level positions are captured — `_capture_pos` is called only in terminal handlers and via `_default`. Error messages may point to the nearest captured token rather than the exact error site.
3. **Earley parser can't convert to LALR** due to template-expanded `bool{x}` rules and 12 ambiguous `rel_b` alternatives. Performance is adequate for the expected input sizes.
