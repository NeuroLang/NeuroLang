# Does the engine gain on an LLM as questions get logically harder? (written before any run, 2026-11-25)

On the one-atom question "will held-out dataset h find gene g significant for disease d?", the engine equals Sonnet given the same evidence and
trails Sonnet given the full table by 0.016 Brier (`PROTOCOL_SAME.md`). Hypothesis: as the minimal first-order formula of the question grows
(more atoms, quantifiers, negation, counting, joins), the engine, which composes probabilities exactly, gains on an LLM, which has to compose them
in its head.

## Items and gold answers (real outcomes, never the engine's model)
- For a (gene, disease) pair tested by at least 4 QTL datasets, **three datasets H = {h1, h2, h3} are held out**. The others are the evidence.
- The gold answer is what H actually shows: significant when p_SMR_multi < 2.95e-6, min over probes.
- Items are drawn from all pairs with seed 3; one H per pair; one pair per gene and disease. Their predictions have been seen in earlier analyses.
- Intervals are 95% and resample loci (1 Mb single-linkage, as in `leak_checks.py`), 2,000 draws.

## The ladder: questions, minimal formulas and complexity
`s(q)` abbreviates `sig(q, g, d)`; `H`, `brain`, `blood` are given sets. Complexity is counted on the minimal formula:
- atoms;
- quantifiers;
- quantifier depth;
- negations;
- counting quantifiers;
- distinct relations joined.

| Rung | Question | Minimal formula | Atoms | Quant. | Neg. | Count | Joins |
|---|---|---|---|---|---|---|---|
| Q1 | Will h1 find it? | s(h1) | 1 | 0 | 0 | 0 | 1 |
| Q2 | Will at least one of H find it? | ∃q ∈ H. s(q) | 2 | 1 | 0 | 0 | 2 |
| Q3 | Will all of H find it? | ∀q ∈ H. s(q) | 2 | 1 | 0 | 0 | 2 |
| Q4 | Will at least two of H find it? | ∃≥2 q ∈ H. s(q) | 2 | 1 | 0 | 1 | 2 |
| Q5 | Will a brain dataset of H find it while no blood dataset of H does? | ∃q (H(q) ∧ brain(q) ∧ s(q)) ∧ ¬∃r (H(r) ∧ blood(r) ∧ s(r)) | 6 | 2 | 1 | 0 | 3 |
| Q6 | Will some dataset of H find it for both PD and AD? | ∃q ∈ H. sig(q, g, PD) ∧ sig(q, g, AD) | 3 | 1 | 0 | 0 | 2 |

- Q5 items need H to hold at least one brain and one blood dataset (H is drawn under that constraint).
- Q6 items need g tested for both PD and AD by every dataset of H; the evidence is the other datasets' tests for both diseases.
- Planned sizes:
  - Q1 to Q4: the same 300 items, each question asked separately;
  - Q5: 150 items;
  - Q6: 150 items.

## Systems
- **Engine:** the held-out datasets' outcomes are independent probabilistic facts `held_sig(q, g, d)`, each with the probability `support_prior` gives
  from the evidence. For Q5 this is computed per organ from that organ's evidence (a, b as shipped). The rung's formula is then a NeuroLang query
  over these facts, solved by lifted inference only. The engine's values must equal the closed forms (1 − Π(1 − p), Πp, the at-least-two
  polynomial, …) to 1e-9 on every item.
- **Sonnet-flags:** the engine's evidence. Each other dataset is listed by number, its organ, and significant yes/no; the gene is masked; the held-out
  datasets are named only by their organ.
- **Sonnet-full:** the full table, as L2 (dataset, tissue, kind, smallest p-value, significant, gene named), and the held-out datasets named.
- Sonnet 5.5 through OpenRouter, `max_tokens` 4,000, one question per call, its own cache, JSON `{"probability": x}`.

## Measures and readings, fixed now
1. **Per rung:** Brier (clipped to [0.02, 0.98]) of each system, and the paired differences engine − Sonnet-full and engine − Sonnet-flags.
2. **Primary hypothesis (H1).** The engine's deficit against Sonnet-full shrinks with complexity. Test: the paired difference
   (engine − Sonnet-full) on the pooled harder rungs Q4 to Q6, minus the same on Q1, with an interval resampling loci.
   - Below 0: H1 supported.
   - Containing 0: not shown.
   - Above 0: the engine falls further behind on harder questions.
3. **H2, same evidence.** On the pooled Q4 to Q6, engine minus Sonnet-flags:
   - below 0: the engine beats Sonnet given the same evidence on complex questions;
   - containing 0: equal.
4. **H3, coherence.** On the 300 items of Q1 to Q4, each system's answers must satisfy P(Q3) ≤ P(Q4) ≤ P(Q2) and P(Q3) ≤ P(Q1) ≤ P(Q2). Reported for
   each system: the share of items violating at least one, with a tolerance of 0.01. The engine is coherent by construction; a Sonnet rate above
   it is the finding.
5. Base rates differ by rung (Q3 and Q6 are rare). Brier skill against each rung's base rate is reported beside the Brier.
6. Every rung and every system is reported. The ladder's complexity measures are fixed above, not tuned after the results.

## Amendment, before any LLM call (2026-11-25)
- **Draw restricted to pairs with signal.** Drawn as written, the items have almost no positives: only 333 of 80,549 pairs tested by at least 4
  datasets have any significant dataset, and the base rates were 0 to 0.7%. Items are therefore drawn from pairs whose evidence holds at least one
  significant dataset (`ladder_items.py SMR 1 ladder_items_signal.json`), the support > 0 condition of every earlier experiment here.
- The resulting base rates:

  | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 |
  |---|---|---|---|---|---|
  | 0.31 | 0.55 | 0.18 | 0.33 | 0.06 | 0.05 |

  The draw covers 58 loci (38 for Q5, 41 for Q6). Q5 and Q6 have about 8 positives each, so their intervals will be wide.
- **Seen before this amendment:** the engine-only scores on this draw (`results/ladder_analysis_signal.txt`; for example, Q5 skill -1.13 against its
  base rate). No LLM answer and no comparison existed.
- **Engine queries.** Q3, written as three `held_sig` atoms with distinct variables, is a self-join the lifted solvers refuse. Q1, Q3 and Q4 use three
  positional relations instead (`held1`, `held2`, `held3`), and Q4 is a union of the three pairs through a helper rule. The engine equals the closed
  forms to 4.4e-16 on every rung, with no weighted model counting.
- **Q6 details.** The evidence is a joint table: dataset, organ, PD, AD, with "not tested" where missing. Each disease must be tested in at least 4
  datasets.
