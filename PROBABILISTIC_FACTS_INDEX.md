# NeuroLang Weighted Probabilistic Facts: Complete Analysis

## 📋 Documentation Index

This directory contains comprehensive analysis of weighted probabilistic facts in NeuroLang:

### 1. **ANALYSIS_SUMMARY.md** — Start Here! ⭐
   - **Length:** ~2 pages
   - **Purpose:** Executive summary and key findings
   - **Contains:**
     - Investigation scope overview
     - Key findings (6 main insights)
     - Handler catalog with line numbers
     - Examples from codebase
     - Verification checklist
     - Implementation recommendations

   **Read this first to understand the big picture.**

---

### 2. **WEIGHTED_PROBABILISTIC_FACTS_REPORT.md** — Complete Reference
   - **Length:** ~8 pages
   - **Purpose:** Detailed technical analysis
   - **Contains:**
     - Section 1: ProbabilisticFact IR node definition & properties
     - Section 2: SQUALL syntax handlers (vpdo_*)
     - Section 3: Rule-level handlers (rule_op_*)
     - Section 4: Grammar rules (neurolang_natural.lark)
     - Section 5: @ symbol analysis (NOT a weight operator)
     - Section 6: Aggregation chain (ng1_agg_npc → det_every)
     - Section 7: Complete end-to-end example
     - Section 8: Type system integration
     - Section 9: Key insights
     - Section 10: Integration points

   **Read this for deep technical understanding and exact code.**

---

### 3. **PROBABILISTIC_FACTS_QUICK_REFERENCE.md** — How-To Guide
   - **Length:** ~4 pages
   - **Purpose:** Practical reference for creating probabilistic facts
   - **Contains:**
     - What is ProbabilisticFact? (definition + examples)
     - 4 ways to create probabilistic facts in SQUALL
     - Aggregation chain overview
     - Probability forms table
     - @ symbol clarification
     - Aggregation determinism explanation
     - End-to-end flow diagram
     - Symbol tracking architecture
     - Key files reference
     - Summary bullet points

   **Read this when you need to quickly understand how to use features.**

---

### 4. **PROBABILISTIC_FACTS_TEST_EXAMPLES.md** — Concrete Test Cases
   - **Length:** ~6 pages
   - **Purpose:** Real test cases showing expected behavior
   - **Contains:**
     - Test 1: Probabilistic conditioned prior
     - Test 2: Rule with explicit probability (MARG query)
     - Test 3: Aggregation with arbitrary functor
     - Test 4: Global aggregation with free variable collection
     - Test 5: Tuple variable info in conditioned rules
     - Test 6: VP-level probabilistic operators
     - Test 7: N-ary probabilistic operators
     - Test 8: Rule-level operators
     - Summary of test insights
     - How to run tests

   **Read this to see concrete examples and understand test structure.**

---

## 🎯 Quick Navigation

**By Use Case:**

- **"I want to understand the big picture"**
  → Start with ANALYSIS_SUMMARY.md

- **"I need to implement a feature using probabilistic facts"**
  → Use PROBABILISTIC_FACTS_QUICK_REFERENCE.md

- **"I need exact handler code and implementation details"**
  → Consult WEIGHTED_PROBABILISTIC_FACTS_REPORT.md (Section 2-3)

- **"I want to see test cases and expected output"**
  → Check PROBABILISTIC_FACTS_TEST_EXAMPLES.md

- **"I need to understand aggregation determinism"**
  → See WEIGHTED_PROBABILISTIC_FACTS_REPORT.md (Section 6) or QUICK_REFERENCE.md

---

## 🔑 Key Concepts at a Glance

### ProbabilisticFact IR Node
```python
class ProbabilisticFact(ProbabilisticPredicate):
    def __init__(self, probability, body):
        self.probability = probability  # Any IR expression!
        self.body = body                # Wrapped predicate
        self._symbols = ...             # Combined symbol tracking
```

### Four Paths to Create ProbabilisticFact

| Path | Syntax | Handler | Probability |
|------|--------|---------|-------------|
| VP Explicit | `subject verb with probability 0.7` | `vpdo_explicit_prob_v1` | `Constant(0.7)` |
| VP Fresh | `subject probably verb` | `vpdo_prob_v1` | `Symbol(fresh)` |
| Rule Explicit | `define as verb with probability ... every ...` | `rule_op_prob` | `Constant` or expr |
| Rule Fresh | `define as probably verb every ...` | `rule_opnn_per` | `Symbol(fresh)` |

### Aggregation Chain
```
ng1_agg_npc (marks with _agg_info) → det_every (executes, sorts by name)
```

### @ Symbol
- **Is:** An alternate label marker (like `?`)
- **Is NOT:** A probability/weight operator
- **Probability expressed via:** `with probability <expr>` or `probably` keyword

---

## 📚 File Structure

All analysis files are located in:
```
/Users/dwasserm/sources/NeuroLang/.worktrees/squall-cnl/
├── ANALYSIS_SUMMARY.md                          (2 pages)
├── WEIGHTED_PROBABILISTIC_FACTS_REPORT.md       (8 pages)
├── PROBABILISTIC_FACTS_QUICK_REFERENCE.md       (4 pages)
├── PROBABILISTIC_FACTS_TEST_EXAMPLES.md         (6 pages)
├── PROBABILISTIC_FACTS_INDEX.md                 (this file)
└── neurolang/
    ├── probabilistic/expressions.py             (ProbabilisticFact definition)
    └── frontend/datalog/
        ├── squall_syntax_lark.py                (Handlers)
        ├── neurolang_natural.lark               (Grammar)
        ├── sugar/__init__.py                    (Sugar processing)
        └── tests/test_squall_parser.py          (Test cases)
```

---

## 🔗 Code References

### Key Handlers (with line numbers)

**VP-Level (CPS functions):**
- `vpdo_explicit_prob_v1` (861-865) — `verb with probability number`
- `vpdo_explicit_prob_vn` (867-876) — `verbn opn with probability number`
- `vpdo_prob_v1` (848-852) — `probably verb`
- `vpdo_prob_vn` (854-859) — `probably verbn opn`

**Rule-Level (Implications):**
- `rule_op_prob` (212-229) — `define as verb with probability np body`
- `rule_opnn_per` (297-319) — `define as probably verbn body ops`
- `rule_op_marg` (231-255) — `define as verb with probability body_cond`

**Aggregation:**
- `ng1_agg_npc` (663-758) — Creates `_agg_info` attribute
- `det_every` (480-556) — Inspects `_agg_info`, builds aggregation

---

## ✅ Verification Completed

- [x] ProbabilisticFact takes ANY expression as probability
- [x] Four main IR paths identified and documented
- [x] All handlers cataloged with exact line numbers
- [x] Aggregation determinism mechanism explained (sorted by name)
- [x] @ symbol confirmed as label marker only
- [x] Examples verified from actual codebase
- [x] Test cases located and documented
- [x] CPS continuation architecture verified
- [x] Symbol tracking mechanism explained
- [x] Conditional probability handling documented

---

## 📖 Reading Guide

### For Different Audiences

**Developers implementing features:**
1. Read ANALYSIS_SUMMARY.md (5 min)
2. Reference PROBABILISTIC_FACTS_QUICK_REFERENCE.md as needed
3. Consult test examples for validation

**Researchers understanding the system:**
1. Start with ANALYSIS_SUMMARY.md
2. Deep dive into WEIGHTED_PROBABILISTIC_FACTS_REPORT.md
3. Study test cases in PROBABILISTIC_FACTS_TEST_EXAMPLES.md

**Maintainers/Code reviewers:**
1. ANALYSIS_SUMMARY.md for overview
2. Test examples for regression checking
3. Report for detailed line-by-line analysis

**Users of NeuroLang:**
1. PROBABILISTIC_FACTS_QUICK_REFERENCE.md for how-to
2. Test examples for concrete usage patterns
3. Report for troubleshooting

---

## 📞 Questions Answered

This analysis addresses:

1. **What is ProbabilisticFact?**
   → See ANALYSIS_SUMMARY (Section 1) or REPORT (Section 1)

2. **Does it take scalar probability only?**
   → No! See QUICK_REFERENCE or REPORT (Section 1) — any IR expression

3. **How does vpdo_explicit_prob_v1 work?**
   → REPORT Section 2 (exact handler code) or QUICK_REFERENCE

4. **How does ng1_agg_npc + det_every work?**
   → REPORT Section 6 (complete flow) or QUICK_REFERENCE

5. **Is @ a probability operator?**
   → No! See REPORT Section 5 or QUICK_REFERENCE

6. **What IR does "activates with probability 0.7" produce?**
   → QUICK_REFERENCE (Example 1) or TEST_EXAMPLES (Test 1)

7. **How does aggregation determinism work?**
   → QUICK_REFERENCE (Aggregation Determinism section) with code example

8. **Where are examples?**
   → ANALYSIS_SUMMARY (Section: Examples from Codebase) and TEST_EXAMPLES

---

## 🔄 How to Use These Documents

1. **First Time:**
   - Open ANALYSIS_SUMMARY.md
   - Spend 5-10 minutes getting oriented
   - Note which sections interest you most

2. **Deep Dive:**
   - Open WEIGHTED_PROBABILISTIC_FACTS_REPORT.md
   - Read Section 1 for fundamentals
   - Jump to relevant sections (2-6) based on interests
   - Use Section 9 (Key Insights) for synthesis

3. **Quick Lookup:**
   - QUICK_REFERENCE.md is organized by use case
   - Table at top summarizes all probability forms
   - Inline code examples for each pattern

4. **Testing/Validation:**
   - TEST_EXAMPLES.md has 8 concrete test cases
   - Each shows input, expected output, and test code
   - "How to run tests" section at bottom

---

## 📝 Document Metadata

| Document | Pages | Words | Focus | Level |
|----------|-------|-------|-------|-------|
| ANALYSIS_SUMMARY | 2 | ~1,500 | Overview | Beginner-Intermediate |
| WEIGHTED_PROBABILISTIC_FACTS_REPORT | 8 | ~5,000 | Complete | Intermediate-Advanced |
| PROBABILISTIC_FACTS_QUICK_REFERENCE | 4 | ~2,500 | Practical | Beginner-Intermediate |
| PROBABILISTIC_FACTS_TEST_EXAMPLES | 6 | ~3,500 | Concrete | Intermediate |

**Total:** 20 pages, ~12,500 words of analysis

---

## 🎓 Key Learning Outcomes

After reading these documents, you will understand:

1. ✅ The structure and purpose of `ProbabilisticFact`
2. ✅ How probability can be any IR expression
3. ✅ All four paths to create probabilistic facts
4. ✅ The CPS (Continuation-Passing Style) architecture
5. ✅ How aggregation determinism works (sorting by name)
6. ✅ Why @ is NOT a weight operator
7. ✅ How to read and verify test cases
8. ✅ Integration points with the rest of NeuroLang

---

**Last Updated:** 2026-04-17
**Analysis Scope:** NeuroLang squall-cnl branch
**Status:** Complete & Verified ✅
