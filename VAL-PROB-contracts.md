# Validation Contracts: Probabilistic Constructs for SQUALL CNL Parser

**Area:** Probabilistic Constructs
**IDs:** VAL-PROB-001 through VAL-PROB-028
**Scope:** Grammar parsing, transformer IR generation, and end-to-end solver execution for ProbabilisticChoice, SUCC queries, MARG (Condition) queries, and query-based probabilistic facts.

**Test command:**
```bash
cd /Users/dwasserm/sources/NeuroLang/.worktrees/squall-cnl
../../.venv/bin/python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py -v
```

**Examples command:**
```bash
cd /Users/dwasserm/sources/NeuroLang/.worktrees/squall-cnl
../../.venv/bin/python examples/squall_examples.py
```

**Key IR types (from `neurolang.probabilistic.expressions`):**
- `ProbabilisticFact(probability, body)` — independent probabilistic fact
- `ProbabilisticChoice(probability, body)` — mutually exclusive alternatives
- `Condition(conditioned, conditioning)` — P(A|B)
- `ProbabilisticQuery(PROB, args)` — marker in head for probability queries
- `PROB = Symbol("PROB")` — sentinel symbol
- `Implication(consequent, antecedent)` — standard rule

**Key grammar rules (from `neurolang_natural.lark`):**
- `rule_op_prob`: `define as VERB with probability NP rule_body1`
- `rule_opnn_per`: `define as probably VERBN rule_body1 [conditioned] ops`
- `rule_body1_cond_prior/posterior`: conditioned-to clauses
- `query`: `obtain ops`

---

## 1. ProbabilisticChoice — Mutually Exclusive Alternatives

### VAL-PROB-001: Grammar parses ProbabilisticChoice basic syntax
**Description:** The grammar must accept a sentence expressing mutually exclusive alternatives using a "choice" keyword/construct. Candidate syntaxes: `"define as a choice selected with probability 1/3 every color ?c"` or similar. The Lark parser must produce a valid parse tree without errors.
**Pass/fail:** `parser(code)` returns a non-None expression without raising any exception.
**Tool:** pytest
**Evidence:** `assert parser("define as a choice ...") is not None`

### VAL-PROB-002: Transformer produces ProbabilisticChoice IR with constant probability
**Description:** For a SQUALL sentence expressing a probabilistic choice with a constant probability (e.g., `0.3`), the transformer must produce `Implication(ProbabilisticChoice(Constant(0.3), head), body)`.
**Pass/fail:** `isinstance(result.consequent, ProbabilisticChoice)` and `result.consequent.probability == Constant(0.3)`.
**Tool:** pytest
**Evidence:**
```python
from neurolang.probabilistic.expressions import ProbabilisticChoice
result = parser("define as a choice selected with probability 0.3 every color ?c that ...")
assert isinstance(result, Implication)
assert isinstance(result.consequent, ProbabilisticChoice)
assert result.consequent.probability == Constant(0.3)
```

### VAL-PROB-003: Transformer produces ProbabilisticChoice IR with fraction probability
**Description:** The probability expression `1/3` must be parsed as an arithmetic expression yielding `Constant(truediv)(Constant(1), Constant(3))`, not a float literal.
**Pass/fail:** `result.consequent.probability` is a `FunctionApplication` with `Constant(truediv)` as functor and `(Constant(1), Constant(3))` as args.
**Tool:** pytest
**Evidence:**
```python
result = parser("define as a choice selected with probability 1/3 every color ?c that ...")
prob = result.consequent.probability
assert isinstance(prob, FunctionApplication)
```

### VAL-PROB-004: ProbabilisticChoice head predicate has correct functor and args
**Description:** The body of the `ProbabilisticChoice` wrapper (i.e., `result.consequent.body`) must be a `FunctionApplication` with the correct predicate symbol and variable arguments extracted from the SQUALL noun phrase.
**Pass/fail:** `result.consequent.body.functor == Symbol("color")` and `result.consequent.body.args` contains the expected variable(s).
**Tool:** pytest
**Evidence:**
```python
assert result.consequent.body.functor == Symbol("color")
```

### VAL-PROB-005: ProbabilisticChoice antecedent (body) is correct
**Description:** The antecedent of the Implication must contain the body conditions from the SQUALL relative clause or quantifier.
**Pass/fail:** `result.antecedent` is a `Conjunction` or `FunctionApplication` containing the expected predicates.
**Tool:** pytest
**Evidence:** Check `result.antecedent` structure matches expected body.

### VAL-PROB-006: ProbabilisticChoice end-to-end through CPLogicProgram
**Description:** Walking the produced IR through `CPLogicProgram` (or `RegionFrontendCPLogicSolver`) must register the predicate as a probabilistic choice (`pchoice_pred_symbs`).
**Pass/fail:** The predicate symbol appears in `solver.pchoice_pred_symbs`.
**Tool:** pytest
**Evidence:**
```python
solver = RegionFrontendCPLogicSolver()
# ... add base data ...
result = parser("define as a choice ...")
solver.walk(result)
assert Symbol("color") in solver.pchoice_pred_symbs
```

### VAL-PROB-007: ProbabilisticChoice with computed probability expression
**Description:** A choice where probability is a function of body variables, e.g., `"define as a choice with probability exp(-d/5) every voxel ?x that ..."`. The probability field must be a `FunctionApplication` expression.
**Pass/fail:** `result.consequent.probability` is a `FunctionApplication` (not just a `Constant`).
**Tool:** pytest
**Evidence:** Check probability is an expression tree involving `exp(...)`.

---

## 2. ProbabilisticQuery (SUCC) — "the probability that..."

### VAL-PROB-008: Grammar parses SUCC query with "obtain the probability of"
**Description:** The grammar must accept `"obtain the probability of likes for every person ?x."` The `_OBTAIN` + `_PROBABILITY` keywords must route through the `query` rule.
**Pass/fail:** `parser(code)` returns without error.
**Tool:** pytest
**Evidence:** `assert parser("obtain the probability of likes for every person ?x.") is not None`

### VAL-PROB-009: Transformer produces Implication with ProbabilisticQuery(PROB, ...) in head
**Description:** The SUCC query must produce `Implication(head_pred(x, ProbabilisticQuery(PROB, (x,))), body)` where the head contains a `ProbabilisticQuery` marker as one of its arguments.
**Pass/fail:** At least one arg of `result.consequent` is an instance of `ProbabilisticQuery` with `functor == PROB`.
**Tool:** pytest
**Evidence:**
```python
from neurolang.probabilistic.expressions import ProbabilisticQuery, PROB
result = parser("obtain the probability of likes for every person ?x.")
assert any(
    isinstance(arg, ProbabilisticQuery) and arg.functor == PROB
    for arg in result.consequent.args
)
```

### VAL-PROB-010: SUCC query head variables match body variables
**Description:** The variables in `ProbabilisticQuery(PROB, (x,))` must be exactly the free variables from the quantified noun phrases in the SQUALL sentence. This mirrors the validation in `CPLogicProgram._validate_within_language_succ_query`.
**Pass/fail:** The set of variable symbols in the ProbabilisticQuery args equals the set of non-PROB arguments in the head predicate.
**Tool:** pytest
**Evidence:**
```python
prob_arg = [a for a in result.consequent.args if isinstance(a, ProbabilisticQuery)][0]
head_vars = {a for a in result.consequent.args if isinstance(a, Symbol) and a != result.consequent.functor}
assert set(prob_arg.args) == head_vars
```

### VAL-PROB-011: SUCC query body contains expected predicate conjunction
**Description:** The antecedent of the Implication must contain the body predicates. E.g., for `"obtain the probability of likes for every person ?x"`, the body should contain `likes(x)` and `person(x)`.
**Pass/fail:** Body predicates are present in `result.antecedent`.
**Tool:** pytest
**Evidence:** Inspect `result.antecedent` formulas.

### VAL-PROB-012: SUCC query registers as within-language prob query in CPLogicProgram
**Description:** Walking the parsed SUCC query through `CPLogicProgram` must register it via `within_language_succ_query`, making it appear in `solver.within_language_prob_queries()`.
**Pass/fail:** The head predicate symbol is in `solver.within_language_prob_queries()`.
**Tool:** pytest
**Evidence:**
```python
solver = RegionFrontendCPLogicSolver()
# ... add base facts ...
result = parser("obtain the probability of likes for every person ?x.")
solver.walk(result)
assert Symbol("likes_prob") in solver.within_language_prob_queries()  # or whatever head symbol
```

### VAL-PROB-013: SUCC query with multi-variable head
**Description:** `"obtain the probability of ~connects for every region ?x, with every region ?y"` must produce `ProbabilisticQuery(PROB, (x, y))` with both variables.
**Pass/fail:** `prob_arg.args == (Symbol('x'), Symbol('y'))` (up to fresh variable renaming — use `weak_logic_eq`).
**Tool:** pytest
**Evidence:** Parse and check arity of ProbabilisticQuery args.

### VAL-PROB-014: SUCC query end-to-end execution produces probability values
**Description:** Given a database with probabilistic facts, the SUCC query must resolve to a result set containing tuples with probability values (floats between 0 and 1).
**Pass/fail:** The chase solution for the query predicate contains tuples where the last column is a float in [0, 1].
**Tool:** script
**Evidence:**
```python
solver = RegionFrontendCPLogicSolver()
solver.add_probabilistic_facts_from_tuples(Symbol("likes"), [(0.8, 'alice'), (0.6, 'bob')])
result = parser("obtain the probability of likes for every person ?x.")
solver.walk(result)
chase = Chase(solver)
solution = chase.build_chase_solution()
# Verify result tuples have probability column
```

---

## 3. Condition (MARG) — Conditional Probability P(A|B)

### VAL-PROB-015: Grammar parses MARG query with "conditioned to" syntax (prior form)
**Description:** The grammar must accept `"define as probably likes every person ?x conditioned to happy ?x."` using the `rule_body1_cond_prior` rule (det ng1 CONDITIONED TO s).
**Pass/fail:** `parser(code)` returns without error.
**Tool:** pytest
**Evidence:** `assert parser("define as probably likes every person ?x conditioned to happy ?x.") is not None`

### VAL-PROB-016: Grammar parses MARG query with "conditioned to" syntax (posterior form)
**Description:** The grammar must accept `"define as probably happy ?x conditioned to every person ?x likes."` using the `rule_body1_cond_posterior` rule (s CONDITIONED TO det ng1).
**Pass/fail:** `parser(code)` returns without error.
**Tool:** pytest
**Evidence:** `assert parser(...) is not None`

### VAL-PROB-017: Grammar parses "given that" syntax for conditional probability
**Description:** If a `"given that"` alternative syntax is supported (using `_GIVEN`), the grammar must accept `"obtain the probability of likes given that happy for every person ?x."`.
**Pass/fail:** `parser(code)` returns without error, or the grammar is extended to support `_GIVEN _THAT` as an alias for `_CONDITIONED _TO`.
**Tool:** pytest
**Evidence:** Parse succeeds.

### VAL-PROB-018: Transformer produces Condition(conditioned, conditioning) in body
**Description:** The MARG query must produce `Implication(head_with_PROB, Condition(conditioned_formula, conditioning_formula))` where `Condition` wraps the conditioned predicate and the conditioning predicate.
**Pass/fail:** `isinstance(result.antecedent, Condition)` and `result.antecedent.conditioned` and `result.antecedent.conditioning` are both valid formulas.
**Tool:** pytest
**Evidence:**
```python
from neurolang.probabilistic.expressions import Condition
result = parser("define as probably likes every person ?x conditioned to happy ?x.")
assert isinstance(result.antecedent, Condition)
assert result.antecedent.conditioned is not None
assert result.antecedent.conditioning is not None
```

### VAL-PROB-019: MARG query head contains ProbabilisticQuery(PROB, ...)
**Description:** The head of the MARG query Implication must contain a `ProbabilisticQuery(PROB, vars)` argument, same as SUCC queries. This is required for `CPLogicProgram` to recognize it as a within-language probabilistic query.
**Pass/fail:** Head has `ProbabilisticQuery` argument.
**Tool:** pytest
**Evidence:** Same check as VAL-PROB-009 but for the MARG case.

### VAL-PROB-020: MARG query registers as within-language MARG query in CPLogicProgram
**Description:** Walking the parsed MARG query through `CPLogicProgram` must be matched by the `within_language_marg_query` handler (pattern: `Implication(..., Condition), is_within_language_prob_query`).
**Pass/fail:** The predicate appears in `solver.within_language_prob_queries()`.
**Tool:** pytest
**Evidence:**
```python
solver = RegionFrontendCPLogicSolver()
result = parser("define as probably likes every person ?x conditioned to happy ?x.")
solver.walk(result)
assert Symbol("likes") in solver.within_language_prob_queries()  # head pred
```

### VAL-PROB-021: MARG query shared variable between conditioned and conditioning
**Description:** When the conditioned and conditioning predicates share a variable (e.g., `?x`), the `Condition` node must use the same `Symbol` instance for both, so the CPLogic solver can correctly join them.
**Pass/fail:** The shared variable in `condition.conditioned` and `condition.conditioning` is the same Symbol (or equivalent under `weak_logic_eq`).
**Tool:** pytest
**Evidence:** Extract variable symbols from both sides and verify overlap.

### VAL-PROB-022: MARG query with conjunctive conditioning
**Description:** `"obtain the probability of likes given that happy and active for every person ?x"` must produce `Condition(likes(x), Conjunction((happy(x), active(x))))`.
**Pass/fail:** `result.antecedent.conditioning` is a `Conjunction` with two formulas.
**Tool:** pytest
**Evidence:** Check `isinstance(result.antecedent.conditioning, Conjunction)`.

### VAL-PROB-023: MARG query end-to-end execution produces conditional probabilities
**Description:** Given appropriate probabilistic data, the MARG query must resolve and produce correct conditional probability values. This requires the full chase with MARG resolution.
**Pass/fail:** Solution contains tuples with probability values matching expected conditional probabilities.
**Tool:** script
**Evidence:** Set up a small probabilistic model, run MARG query, verify numeric results.

---

## 4. Query-Based Probabilistic Facts — Computed Probability from Body

### VAL-PROB-024: Grammar parses "with probability EXPR" syntax (already partially implemented)
**Description:** The existing `rule_op_prob` grammar rule `"define as VERB with probability NP rule_body1"` must parse. E.g., `"define as reports with probability exp(-d/5) every voxel ?x that ..."`. This grammar rule already exists in the Lark file.
**Pass/fail:** `parser(code)` returns without error.
**Tool:** pytest
**Evidence:** `assert parser("define as reports with probability 0.8 every voxel ?x.") is not None`

### VAL-PROB-025: Transformer produces ProbabilisticFact IR with expression probability
**Description:** The `rule_op_prob` transformer must produce `Implication(ProbabilisticFact(prob_expr, head), body)` where `prob_expr` is the arithmetic expression from the SQUALL sentence.
**Pass/fail:** `isinstance(result.consequent, ProbabilisticFact)` and `result.consequent.probability` is a valid expression (Constant or FunctionApplication).
**Tool:** pytest
**Evidence:**
```python
from neurolang.probabilistic.expressions import ProbabilisticFact
result = parser("define as reports with probability 0.8 every voxel ?x.")
assert isinstance(result, Implication)
assert isinstance(result.consequent, ProbabilisticFact)
```

### VAL-PROB-026: Probability expression references body variables
**Description:** For `"define as reports with probability exp(-?d / 5) every voxel ?x that has a ~distance ?d"`, the probability expression must reference the variable `?d` from the body, producing `FunctionApplication(Symbol('exp'), (Constant(truediv)(Constant(sub)(Constant(0), Symbol('d')), Constant(5)),))`.
**Pass/fail:** The probability expression contains `Symbol('d')` (or equivalent fresh variable) that also appears in the body.
**Tool:** pytest
**Evidence:** Check that variable symbols in `result.consequent.probability` intersect with variables in `result.antecedent`.

### VAL-PROB-027: Query-based probabilistic fact registers in CPLogicProgram
**Description:** Walking the result through `CPLogicProgram` must trigger `query_based_probabilistic_fact` handler, registering the predicate in `pfact_pred_symbs`.
**Pass/fail:** The predicate symbol appears in `solver.pfact_pred_symbs`.
**Tool:** pytest
**Evidence:**
```python
solver = RegionFrontendCPLogicSolver()
result = parser("define as reports with probability 0.8 every voxel ?x.")
solver.walk(result)
assert Symbol("reports") in solver.pfact_pred_symbs
```

### VAL-PROB-028: Query-based probabilistic fact end-to-end execution
**Description:** Given extensional data, a query-based probabilistic fact with a computed probability must resolve correctly through the chase.
**Pass/fail:** The chase solution for the predicate contains tuples associated with the correct computed probability values.
**Tool:** script
**Evidence:**
```python
solver = RegionFrontendCPLogicSolver()
solver.add_extensional_predicate_from_tuples(Symbol("voxel"), [('v1',), ('v2',)])
result = parser("define as reports with probability 0.5 every voxel ?x.")
solver.walk(result)
chase = Chase(solver)
solution = chase.build_chase_solution()
# Verify reports has probability-weighted entries
```

---

## 5. Edge Cases and Cross-Cutting Concerns

### VAL-PROB-029: Nested quantifiers inside probabilistic choice
**Description:** A probabilistic choice that contains nested quantifiers in its body, e.g., `"define as a choice with probability 0.5 every color ?c that a person ~likes"`, must correctly handle the existential quantifier for "a person" inside the rule body.
**Pass/fail:** The antecedent contains an `ExistentialPredicate` for the inner quantifier, or the body is correctly flattened for Datalog.
**Tool:** pytest
**Evidence:** Inspect `result.antecedent` for correct quantifier structure.

### VAL-PROB-030: Probability value boundary — zero
**Description:** `"define as reports with probability 0 every voxel ?x."` must parse and produce `ProbabilisticFact(Constant(0), ...)`.
**Pass/fail:** `result.consequent.probability == Constant(0)`.
**Tool:** pytest
**Evidence:** Parse and check.

### VAL-PROB-031: Probability value boundary — one
**Description:** `"define as reports with probability 1 every voxel ?x."` must parse and produce `ProbabilisticFact(Constant(1), ...)`.
**Pass/fail:** `result.consequent.probability == Constant(1)`.
**Tool:** pytest
**Evidence:** Parse and check.

### VAL-PROB-032: Reject MARG query without PROB in head
**Description:** If a Condition body is produced but the head lacks a `ProbabilisticQuery(PROB, ...)` argument, the CPLogicProgram must raise `ForbiddenConditionalQueryNoProb`. The transformer must always include PROB in the head for conditioned queries.
**Pass/fail:** Either the transformer always includes PROB (preventing this case), or `CPLogicProgram.walk()` raises `ForbiddenConditionalQueryNoProb`.
**Tool:** pytest
**Evidence:**
```python
with pytest.raises(ForbiddenConditionalQueryNoProb):
    solver.walk(malformed_marg_implication)
```

### VAL-PROB-033: SUCC query rejects disjunctive definitions
**Description:** If a SUCC query predicate is defined twice (producing a disjunction), the CPLogicProgram must raise `ForbiddenDisjunctionError`. Verify that parsing two SUCC sentences for the same predicate and walking both through the solver fails appropriately.
**Pass/fail:** Second `solver.walk()` raises `ForbiddenDisjunctionError`.
**Tool:** pytest
**Evidence:**
```python
solver.walk(parser("obtain the probability of likes for every person ?x."))
with pytest.raises(ForbiddenDisjunctionError):
    solver.walk(parser("obtain the probability of likes for every person ?x."))
```

### VAL-PROB-034: Probabilistic constructs compose with boolean connectives
**Description:** Probability queries whose body uses AND/OR connectives (e.g., `"obtain the probability of likes and runs for every person ?x"`) must correctly produce a body with `Conjunction` or `Disjunction`.
**Pass/fail:** Body formula type matches the connective used.
**Tool:** pytest
**Evidence:** Check `isinstance(result.antecedent, Conjunction)`.

### VAL-PROB-035: ProbabilisticChoice with tuple labels
**Description:** `"define as a choice with probability 0.5 every voxel (?x; ?y; ?z)"` must handle multi-dimensional individuals correctly, producing a head with tuple-labeled variables.
**Pass/fail:** Head predicate args are the tuple `(Symbol('x'), Symbol('y'), Symbol('z'))`.
**Tool:** pytest
**Evidence:** Verify `result.consequent.body.args`.

### VAL-PROB-036: LogicSimplifier preserves probabilistic IR types
**Description:** Running `LogicSimplifier().walk(result)` on any probabilistic IR expression must preserve the `ProbabilisticFact`, `ProbabilisticChoice`, `Condition`, and `ProbabilisticQuery` wrapper types — it must not strip or flatten them.
**Pass/fail:** After simplification, the type of the consequent/antecedent is unchanged.
**Tool:** pytest
**Evidence:**
```python
simplified = LogicSimplifier().walk(result)
assert type(simplified.consequent) == type(result.consequent)
```

---

## Summary Table

| ID | Construct | Layer | Description |
|----|-----------|-------|-------------|
| VAL-PROB-001 | ProbabilisticChoice | Grammar | Basic syntax parses |
| VAL-PROB-002 | ProbabilisticChoice | Transformer | Constant probability IR |
| VAL-PROB-003 | ProbabilisticChoice | Transformer | Fraction probability expression |
| VAL-PROB-004 | ProbabilisticChoice | Transformer | Head predicate correctness |
| VAL-PROB-005 | ProbabilisticChoice | Transformer | Body/antecedent correctness |
| VAL-PROB-006 | ProbabilisticChoice | End-to-end | CPLogicProgram registration |
| VAL-PROB-007 | ProbabilisticChoice | Transformer | Computed probability expression |
| VAL-PROB-008 | SUCC Query | Grammar | "obtain the probability of" parses |
| VAL-PROB-009 | SUCC Query | Transformer | ProbabilisticQuery(PROB) in head |
| VAL-PROB-010 | SUCC Query | Transformer | Head vars match PROB vars |
| VAL-PROB-011 | SUCC Query | Transformer | Body predicates correct |
| VAL-PROB-012 | SUCC Query | End-to-end | Registers as WLQ |
| VAL-PROB-013 | SUCC Query | Transformer | Multi-variable head |
| VAL-PROB-014 | SUCC Query | End-to-end | Produces probability values |
| VAL-PROB-015 | MARG (Condition) | Grammar | Prior form parses |
| VAL-PROB-016 | MARG (Condition) | Grammar | Posterior form parses |
| VAL-PROB-017 | MARG (Condition) | Grammar | "given that" syntax |
| VAL-PROB-018 | MARG (Condition) | Transformer | Condition IR produced |
| VAL-PROB-019 | MARG (Condition) | Transformer | Head has ProbabilisticQuery |
| VAL-PROB-020 | MARG (Condition) | End-to-end | Registers as MARG WLQ |
| VAL-PROB-021 | MARG (Condition) | Transformer | Shared variable correctness |
| VAL-PROB-022 | MARG (Condition) | Transformer | Conjunctive conditioning |
| VAL-PROB-023 | MARG (Condition) | End-to-end | Conditional probability values |
| VAL-PROB-024 | Query-based PFact | Grammar | "with probability" parses |
| VAL-PROB-025 | Query-based PFact | Transformer | ProbabilisticFact IR |
| VAL-PROB-026 | Query-based PFact | Transformer | Probability references body vars |
| VAL-PROB-027 | Query-based PFact | End-to-end | CPLogicProgram registration |
| VAL-PROB-028 | Query-based PFact | End-to-end | Chase execution |
| VAL-PROB-029 | Edge case | Transformer | Nested quantifiers in choice |
| VAL-PROB-030 | Edge case | Transformer | Probability = 0 boundary |
| VAL-PROB-031 | Edge case | Transformer | Probability = 1 boundary |
| VAL-PROB-032 | Edge case | End-to-end | Reject MARG without PROB |
| VAL-PROB-033 | Edge case | End-to-end | Reject disjunctive SUCC |
| VAL-PROB-034 | Edge case | Transformer | Boolean connectives in prob body |
| VAL-PROB-035 | Edge case | Transformer | Tuple labels in choice |
| VAL-PROB-036 | Edge case | Transformer | Simplifier preserves prob types |
