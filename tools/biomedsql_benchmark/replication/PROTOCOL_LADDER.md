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
