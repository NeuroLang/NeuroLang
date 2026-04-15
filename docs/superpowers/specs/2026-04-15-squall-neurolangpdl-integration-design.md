# SQUALL + NeurolangPDL Integration

**Date:** 2026-04-15
**Branch:** squall-cnl
**Scope:** Add `execute_squall_program` to `QueryBuilderDatalog`; rewrite RST tutorial using `NeurolangPDL` API

---

## Background

The first tutorial (`doc/tutorial_squall.rst`) exposed the low-level `parser()` / `Chase` /
`RegionFrontendCPLogicSolver` machinery directly. The user-facing API is `NeurolangPDL` (and its
parent `QueryBuilderDatalog`), which already has `execute_datalog_program`. The goal is to:

1. Add an analogous `execute_squall_program(code: str)` method.
2. Rewrite the tutorial entirely in terms of `NeurolangPDL().execute_squall_program(...)`.

---

## Part 1 — `execute_squall_program` method

### Location

`neurolang/frontend/query_resolution_datalog.py`, class `QueryBuilderDatalog`.
Placed immediately after `execute_datalog_program` (~line 261).

### Top-level import addition

```python
from .datalog.squall_syntax_lark import parser as squall_parser, SquallProgram
```

Added alongside the existing `from .datalog.standard_syntax import parser as datalog_parser`.

### Signature

```python
def execute_squall_program(
    self, code: str
) -> Union[None, NamedRelationalAlgebraFrozenSet, Dict[str, NamedRelationalAlgebraFrozenSet]]:
```

### Return semantics (mirrors execute_datalog_program)

| Program contains | Return value |
|-----------------|-------------|
| Rules only (`define as …`) | `None` |
| Exactly one `obtain` clause | `NamedRelationalAlgebraFrozenSet` |
| Multiple `obtain` clauses | `dict[str, NamedRelationalAlgebraFrozenSet]` keyed `"obtain_0"`, `"obtain_1"`, … |

### Algorithm

```
parsed = squall_parser(code)
if not isinstance(parsed, SquallProgram):
    self.program_ir.walk(parsed)
    return None

for rule in parsed.rules:
    self.program_ir.walk(rule)

query_entries = []
for i, q in enumerate(parsed.queries):
    h = Symbol.fresh()
    self.program_ir.walk(Implication(h(q.head), q.body))
    query_entries.append((f"obtain_{i}", h))

if not query_entries:
    return None

solution = self.chase_class(self.program_ir).build_chase_solution()

results = {}
for key, h in query_entries:
    col_names = self.predicate_parameter_names(h.name)
    ra = NamedRelationalAlgebraFrozenSet(
        col_names, solution[h].value.unwrap()
    )
    ra.row_type = solution[h].value.row_type
    results[key] = ra

if len(results) == 1:
    return results["obtain_0"]
return results
```

### Files changed

| File | Change |
|------|--------|
| `neurolang/frontend/query_resolution_datalog.py` | Add import + method |
| `neurolang/frontend/tests/test_squall_pdl_integration.py` | New test file |

---

## Part 2 — Tutorial rewrite

### Goal

Replace all `parser()` / `Chase` / `Symbol` / `RegionFrontendCPLogicSolver` references with
clean `NeurolangPDL` calls.

### Canonical pattern

```python
>>> from neurolang.frontend import NeurolangPDL
>>> nl = NeurolangPDL()
>>> nl.add_tuple_set([("alice",), ("bob",)], name="person")
>>> nl.add_tuple_set([("alice",)], name="plays")
>>> nl.execute_squall_program("define as Active every person that plays.")
>>> solution = nl.solve_all()
>>> sorted(solution["active"].as_pandas_dataframe()["0"].tolist())
['alice']
```

### `obtain` query pattern

```python
>>> result = nl.execute_squall_program("obtain every Person that plays.")
>>> sorted(result.as_pandas_dataframe()["0"].tolist())
['alice']
```

### Structure

Same 14 sections as the previous tutorial. All end-to-end examples share a single `nl`
instance set up in the Setup section and reset between sections using `nl = NeurolangPDL()`
at the start of each section that needs fresh state.

### Doctest stability

- `add_tuple_set(...)` prints a repr line → suppress with `# doctest: +ELLIPSIS` or assign to `_`
- `solve_all()` / `execute_squall_program()` return values are deterministic (sorted sets)
- Column names in `NamedRelationalAlgebraFrozenSet` come from `predicate_parameter_names`; for
  SQUALL results with unnamed variables these are `"0"`, `"0"`, `"1"` etc.

### Files changed

| File | Change |
|------|--------|
| `doc/tutorial_squall.rst` | Complete rewrite |

---

## Acceptance Criteria

1. `uv run python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py` — all tests pass
2. `uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py -k "not slow"` — all 13 still pass
3. `uv run python -m pytest doc/tutorial_squall.rst --doctest-glob="doc/tutorial_squall.rst"` — all tutorial doctests pass
4. `execute_squall_program` appears in `QueryBuilderDatalog`
5. `squall_parser` imported at module top level in `query_resolution_datalog.py`
6. Tutorial contains zero references to `parser`, `Chase`, `RegionFrontendCPLogicSolver`, `Symbol.fresh`
