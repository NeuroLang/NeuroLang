# SQUALL Natural-Language Improvements: Compound Quantifiers, Anaphora, and Probabilistic N-ary Predicates

## Date: 2026-05-04

## 1. Background

SQUALL (Semantically controlled Query-Answerable Logical Language) is NeuroLang's controlled-English interface. It currently supports:

- **Unary rules**: `define as Active_region every Region that a Selected_study activates.`
- **Binary rules**: `define as Has_author every Paper that a Person authors.`
- **N-ary rules (with explicit variables)**: `define as Cooccurrence for every Region ?r ; for every Term ?t where ...`
- **Probabilistic unary/binary**: `with inferred probability`, `with probability 0.5`

The current syntax forces users to drop into explicit variables (`?r`, `?t`) when they need ternary relationships, which breaks the natural-language flow. Additionally, probabilistic rules (`with probability`, `with inferred probability`) only work on unary heads (`rule_op`).

## 2. Problem Statement

Two related problems:

1. **Ternary joins are not expressible in natural English**. The user must write `for every Region ?r ; for every Term ?t where a Selected_study ?s ~activates ?r and ~mentions ?t` instead of something like `for every Region and for every Term where a Selected_study activates the Region and mentions the Term`.

2. **Probabilistic n-ary predicates are unsupported**. `define as Verb with inferred probability for every Region ...` is parsed as `rule_op` (unary head) and fails when the body contains multiple quantifiers.

## 3. Design

### 3.1 Compound Quantifiers with `where`

**New syntax**: n-ary rules can use `and` to chain multiple `for every` clauses, followed by a `where` sentence.

```text
define as Cooccurrence with inferred probability
    for every Region ?r and for every Term ?t
    where a Selected_study ?s activates ?r and mentions ?t.
```

**Grammar changes**:

- `rulen` gains new alternatives for compound quantification and probabilistic compound rules:
  ```lark
  ?rulen : _rule_start verbn rule_body1 _BREAK? ops                    -> rule_opnn
         | _rule_start verbn rule_body2                               -> rule_opnn_compound
         | _rule_start PROBABLY verbn rule_body1 _CONDITIONED? _BREAK? ops -> rule_opnn_per
         | _rule_start PROBABLY verbn rule_body2                      -> rule_opnn_per_compound
         | _rule_start verbn _WITH _PROBABILITY np rule_body2        -> rule_opnn_prob
         | _rule_start verbn _WITH _INFERRED? _PROBABILITY rule_body2 -> rule_opnn_marg
  ```
- `rule_body2` is a new grammar non-terminal:
  ```lark
  rule_body2 : quant_list _WHERE s                    -> rule_body2_where

  quant_list : quant_clause                           -> quant_list_single
             | quant_list _AND quant_clause            -> quant_list_rec

  quant_clause : _FOR _EVERY ng1 [ app ]               -> quant_clause_ng1
               | _FOR _EVERY npc{THE}                  -> quant_clause_npc
  ```

**Transformer changes**:

- `quant_clause_ng1` extracts the variable and type predicate from `ng1`, registers the noun name in scope for anaphora, and returns `('_quant_clause', (var, type_pred))`.
- `quant_list_rec` accumulates a flat list of quantifier clauses.
- `rule_body2_where` extracts all variables, conjoins all type predicates with the `where` sentence, and returns `('_rule_body2', (head_vars, body_formula))`.
- `rule_opnn_compound` receives `verb + rule_body2` and produces `Implication(verb(*head_vars), body_formula)`.
- `rule_opnn_prob`, `rule_opnn_marg`, `rule_opnn_per_compound` mirror their unary counterparts but use `rule_body2` for the body.

### 3.2 Anaphoric Definite References (`the Noun`)

**New syntax**: within the body of a rule, `the Region` refers back to the variable introduced by `for every Region`.

```text
define as Cooccurrence with inferred probability
    for every Region and for every Term
    where a Selected_study activates the Region and mentions the Term.
```

**Grammar changes**: None. The existing grammar already supports `the Region` in object/subject positions via `npc{THE}` → `npc_det` → `det_the`. The change is purely in the transformer.

**Transformer changes**:

1. **Add `_symbol_scope` to `SquallTransformer`** (a `dict[str, Symbol]` mapping noun names to bound variables). Cleared at the start of each rule.

2. **Register quantifier variables in scope**. When `det_every` processes `every Region` (or `every Region ?r`), it stores `region -> bound_var` in `_symbol_scope`.

3. **Resolve definite references from scope**. When `det_the` processes `the Region`, it checks `_symbol_scope`:
   - If `region` is in scope: apply the continuation directly to the bound variable (`return d(x)`), bypassing the existential creation. This produces `verb(subject, r)` instead of `∃r. region(r) ∧ verb(subject, r)`.
   - If not in scope: fall back to the existing existential behavior (for queries or unbound nouns).

4. **Clear scope at rule boundaries**. All `rule_op*`, `rule_opnn*`, and `query_as` handlers call `_clear_scope()` at the start.

5. **Track noun names in `ng1_noun`**. `ng1` functions carry a `_noun_name` attribute (e.g., `"region"`) so that scope registration and anaphora resolution know which noun is being bound.

### 3.3 Probabilistic N-ary Predicates

**New syntax**: `with [inferred] probability` and `probably` can appear in n-ary `define as` rules with compound bodies.

```text
define as Weighted_cooccurrence with probability 0.3
    for every Region and for every Term
    where a Selected_study activates the Region and mentions the Term.
```

**Grammar changes**: See section 3.1 — the `rulen` alternatives already cover this.

**Transformer changes**:

- `rule_opnn_prob`: produces `Implication(ProbabilisticFact(prob, verb(*head_vars)), body)`.
- `rule_opnn_marg`: produces `Implication(verb(*head_vars, ProbabilisticQuery(PROB, tuple(head_vars))), body)`. The existing `CPLogicMixin.within_language_marg_query` and `_validate_within_language_marg_query` already support n-ary heads because they compare `csqt_vars` (all Symbol args in the head) with `prob_vars` (all vars inside `PROB(...)`).
- `rule_opnn_per_compound`: produces `Implication(ProbabilisticFact(fresh, verb(*head_vars)), body)`.

## 4. Why No Engine Changes Are Needed

The probabilistic engine (`CPLogicMixin`) validates n-ary MARG/succ queries by comparing the head variables with the `PROB(...)` variables. For a head like `cooccurrence(r, t, PROB(r, t))`:
- `csqt_vars = {r, t}` (all Symbol args)
- `prob_vars = {r, t}` (all vars in `PROB(r, t)`)
- Validation passes because they match.

The query resolution pipeline (`_solve_within_language_prob_query`, `_solve_for_probabilistic_rule`) handles the head arity transparently. So no changes are needed in `neurolang/probabilistic/`.

## 5. Detailed Grammar

### 5.1 New non-terminals

```lark
rule_body2 : quant_list _WHERE s                    -> rule_body2_where

quant_list : quant_clause                           -> quant_list_single
           | quant_list _AND quant_clause            -> quant_list_rec

quant_clause : _FOR _EVERY ng1 [ app ]               -> quant_clause_ng1
             | _FOR _EVERY npc{THE}                  -> quant_clause_npc

?rulen : _rule_start verbn rule_body1 _BREAK? ops   -> rule_opnn
       | _rule_start verbn rule_body2               -> rule_opnn_compound
       | _rule_start PROBABLY verbn rule_body1 _CONDITIONED? _BREAK? ops -> rule_opnn_per
       | _rule_start PROBABLY verbn rule_body2      -> rule_opnn_per_compound
       | _rule_start verbn _WITH _PROBABILITY np rule_body2 -> rule_opnn_prob
       | _rule_start verbn _WITH _INFERRED? _PROBABILITY rule_body2 -> rule_opnn_marg
```

### 5.2 Terminal reuse

- `_AND` is the existing terminal for the word "and" (already used in `_CONJUNCTION`). In `quant_list_rec` it serves as the list separator. There is no ambiguity with boolean conjunction because `for every Region and for every Term` cannot be parsed as a boolean expression (the RHS is not a valid boolean atom).
- `_WHERE` is the existing terminal for "where" (already used in `rel_s`). In `rule_body2` it introduces the body sentence. There is no ambiguity because `rule_body2` only appears inside `rulen`.
- `_FOR` and `_EVERY` are existing terminals already used in `det_every` and `s_for`.

## 6. Transformer Implementation

### 6.1 Scope tracking

```python
class SquallTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self._symbol_scope = {}  # noun_name -> Symbol

    def _clear_scope(self):
        self._symbol_scope.clear()
```

### 6.2 Noun-name tagging in `ng1_noun`

```python
def ng1_noun(self, args):
    # ... existing code ...
    ng._noun_name = noun1.name if isinstance(noun1, Symbol) else None
    return ng
```

### 6.3 Quantifier registration in `det_every`

```python
def det_every(self, args):
    def every(ng):
        def apply_d(d):
            # ... existing aggregation handling ...
            var_info = getattr(ng, '_var_info', None)
            if var_info is not None:
                x = var_info
                # Register in scope for anaphora
                noun_name = getattr(ng, '_noun_name', None)
                if noun_name:
                    self._symbol_scope[noun_name] = x
                # ... rest of existing code ...
```

### 6.4 Anaphora resolution in `det_the`

```python
def det_the(self, args):
    def the(ng):
        def apply_d(d):
            noun_name = getattr(ng, '_noun_name', None)
            if noun_name and noun_name in self._symbol_scope:
                x = self._symbol_scope[noun_name]
                return d(x)  # bypass existential; apply continuation to bound var
            # ... existing existential fallback ...
```

### 6.5 Compound rule body handler

```python
def rule_body2_where(self, args):
    quant_list = args[0]   # list of ('_quant_clause', (var, type_pred))
    where_sentence = args[1]
    head_vars = []
    type_preds = []
    for _, (var, type_pred) in quant_list:
        head_vars.append(var)
        type_preds.append(type_pred)
    body_parts = type_preds + [where_sentence]
    body_formula = Conjunction(tuple(body_parts))
    return ('_rule_body2', (head_vars, body_formula))
```

### 6.6 Compound rule head handlers

```python
def rule_opnn_compound(self, args):
    items = [a for a in args if a is not None]
    verb = items[0]
    body_result = items[1]
    if isinstance(body_result, tuple) and body_result[0] == '_rule_body2':
        head_vars, body_formula = body_result[1]
    else:
        head_vars, body_formula = [], Constant(True)
    consequent = verb(*head_vars) if head_vars else verb()
    return Implication(consequent, body_formula)

# rule_opnn_prob, rule_opnn_marg, rule_opnn_per_compound
# mirror their unary counterparts using rule_body2
```

### 6.7 Scope clearing in rule handlers

Every `rule_op*`, `rule_opnn*`, and `query_as` handler starts with:
```python
self._clear_scope()
```

## 7. Testing Strategy

1. **Parser tests**:
   - `define as Cooccurrence for every Region ?r and for every Term ?t where a Study ?s activates ?r and mentions ?t` → head `cooccurrence(r, t)`, body contains `region(r), term(t), selected_study(s), activates(s, r), mentions(s, t)`.
   - `define as Cooccurrence with inferred probability for every Region and for every Term where a Study activates the Region and mentions the Term` → head `cooccurrence(r, t, PROB(r, t))`, body uses scoped symbols (no fresh vars for `the Region`/`the Term`).
   - `define as Weighted with probability 0.5 for every Region and for every Term where a Study activates the Region and mentions the Term` → `ProbabilisticFact(0.5, weighted(r, t))`.

2. **Anaphora tests**:
   - `define as Test for every Region where a Study activates the Region` → body contains `activates(s, r)` where `r` is the same symbol as in `region(r)` (not a fresh existential).
   - Query fallback: `obtain every Region that activates the Term` → body contains `∃t. term(t) ∧ activates(r, t)` because `Term` is not in scope.

3. **Integration tests**:
   - End-to-end with `NeurolangPDL` for all three probabilistic compound rule forms.

## 8. Migration Path

- Existing SQUALL programs remain unchanged — all changes are additive.
- The old `;` separator syntax still works: `for every Region ?r ; for every Term ?t where ...`.
- The old explicit variable syntax still works: `?r`, `?t`.
- Users can migrate incrementally from explicit variables to anaphora.

## 9. Scope Boundaries

**What this spec does NOT cover** (future work):
- Anaphora across multiple sentences (inter-sentence scope).
- Number agreement (e.g., `the Regions` referring to multiple variables).
- Relative pronoun anaphora (e.g., `which`/`that` in the head referring back to body variables).
- `conditioned to`/`given` variants for compound probabilistic rules.
