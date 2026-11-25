# Does the probability help? Replication prediction on the SMR evidence

Protocol: `PROTOCOL.md` (written before the run; `PROTOCOL.at` is its time). Raw answers: `results/`. Everything below was
computed by `analyse.py` (pre-registered metrics) except the last section, which is **post hoc** (`smoothing.py`).

240 items (gene, disease, held-out QTL dataset), 222 genes, base rate 29%: will the held-out dataset find the gene significant,
given the other datasets' tests? Every system outputs a probability (D1 and D2 a 0/1). 95% intervals resample genes.

| System | What it has | Brier [95% CI] | Brier skill | Log-loss | ECE | AUC |
|---|---|---|---|---|---|---|
| D1 BiomedSQL's yes/no ("any signal") | the other datasets | 0.680 [0.620, 0.739] | -2.29 | 2.777 | 0.688 | 0.500 |
| D2 majority vote | the other datasets | 0.304 [0.243, 0.363] | -0.47 | 1.253 | 0.297 | 0.692 |
| L1 LLM, no data | its own knowledge | 0.216 [0.201, 0.232] | -0.05 | 0.626 | 0.196 | 0.699 |
| L2 LLM + the evidence table | p-values, tissues, kinds | **0.138** [0.116, 0.159] | +0.33 | 0.431 | 0.117 | 0.858 |
| **P NeuroLang** (a ReAct agent writes the conditional query) | the QTL tokens | 0.221 [0.180, 0.261] | -0.07 | 0.803 | 0.194 | 0.757 |

The agent returned the engine's exact share on 240 of 240 items (valid query every time, median 3 model calls).

Paired, P minus the other (Brier; negative means P is better):
- vs D2 majority: -0.083 [-0.117, -0.050]. **The probability beats the deterministic decision made from the same evidence**, and
  BiomedSQL's yes/no answer is the worst system (it says "yes" for every item, and 29% replicate).
- vs L1 LLM without data: +0.005 [-0.035, +0.047] on Brier (log-loss worse for P, +0.177 [+0.031, +0.338]).
- vs L2 LLM with the evidence: **+0.083 [+0.053, +0.114]. The LLM that reads the table is clearly better than the share.**

## Where the share loses

The loss is entirely in genes tested in few other datasets (Brier of the share, and of L2):

| Datasets that tested the gene (other than the held-out one) | Items | Replicated | Share | L2 |
|---|---|---|---|---|
| 1 | 27 | 19% | 0.783 | 0.250 |
| 2-3 | 43 | 28% | 0.271 | 0.194 |
| 4-9 | 79 | 23% | 0.133 | 0.109 |
| 10 or more | 91 | 38% | 0.107 | 0.103 |

With one other dataset the share says 1.0 (one of one found it significant) while 19% replicate: the share has no notion of how
little evidence it has. The LLM shrinks towards a base rate. With 10 or more datasets the two are equal. The calibration doc
already listed this as a limit ("hard 0 and 1 supports, no pseudo-count"); this experiment measures its cost.

## Post hoc: a prior on the units of evidence (`smoothing.py`)

Decided after seeing the table above, so exploratory. A Beta prior (a imaginary datasets that found the gene significant, b that
did not) makes the probability (k + a) / (n + a + b): still an exact conditional probability over a uniform choice of unit of
evidence, and expressible as extra tokens in the same Datalog program. a and b were fitted by log-loss on 1,197 predictions of
131 genes **that are not among the 240 items' genes** (a = 0.05, b = 1; a is at the edge of the grid), then applied unchanged.

| | Brier | skill | Log-loss | ECE | AUC |
|---|---|---|---|---|---|
| share, unsmoothed | 0.221 | -0.07 | 0.803 | 0.194 | 0.756 |
| share + prior (0.05, 1) | 0.144 | +0.30 | 0.451 | 0.088 | 0.817 |
| L2 LLM + evidence | 0.138 | +0.33 | 0.431 | 0.117 | 0.858 |

Brier, share + prior minus L2: +0.006 [-0.006, +0.019]: **no detectable difference**. The smoothed share is better calibrated
(ECE 0.088 against 0.117, not tested for significance) and ranks worse (AUC 0.817 against 0.858: it ignores how strong each
p-value is, which the LLM sees). Per bin of n the two agree within 0.01 to 0.02 except n = 1 to 3.

## What this shows, and what it does not

Shown:
1. A probability carries information that the deterministic answer of the benchmark throws away: 0.680 and 0.304 for the yes/no and
   the majority against 0.221 (0.144 smoothed). On BiomedSQL's own question the deterministic answer is not merely less
   accurate, it is uninformative.
2. NeuroLang computes that probability exactly; an agent writes the query correctly 240 of 240 times from the engine's guidance.
3. With the prior, the engine's number matches in accuracy a strong LLM that reads the same evidence, at about $0.0046 per answer
   for the LLM (list price, about $3,800 for the 825,346 predictions of the calibration experiment) against minutes of local
   computation for the engine, and with a stated, auditable model (a share of the units of evidence).

Not shown:
- That the probabilistic engine is **more accurate** than an LLM reading the evidence: unsmoothed it is worse (0.221 against
  0.138); smoothed, post hoc, it is equal. The LLM ranks better (AUC 0.858 against 0.817 or 0.757).
- Any advantage on uncertain questions in general: one question template, one disease family, one model.

## Limits
240 items from 430 pairs (222 genes); one model for the LLM systems and the agent; the QTL datasets share GWAS and donors, so the
outcome is not an independent replication; the prior was chosen after the failure was seen (fitted on other genes, but chosen
as a remedy post hoc); the LLM was asked once per item (no repeats); the item set and thresholds are BiomedSQL's.

# The prior in the engine, on held-out genes (`PROTOCOL_PRIOR.md`)

Pre-registered after the post hoc result above. `support_prior` is now an engine predicate: two imaginary units of evidence (one always
significant, one never) weigh in a probabilistic choice over 30 units, so that the probability is (k + a) / (n + a + b). The engine's value equals
the closed form to 2.2e-16 on all 1,197 evaluation predictions, solved by lifted inference only (0.2 s each).

Fit: a = 0.01, b = 0.5 by log-loss on the 2,763 predictions of the 222 genes of the first item set. **a is on the edge of the pre-registered grid** (its
smallest value); the surface is flat there (log-loss 0.413 for a = 0.01 and 0.05, 0.414 at 0.1), so the grid being too small changes nothing visible.

Amendments before the run (see the protocol): only 145 fresh pairs exist, and the 131 genes behind them are those on which the earlier post hoc
prior was fitted, so the evaluation uses every held-out prediction with support > 0 of those 131 genes (1,197 predictions, base rate 31.7%). Their data
informed the earlier post hoc fit, not this one; the form of the prior was chosen after seeing the failure on the first 222 genes.

| System | Brier [95% CI over genes] | Skill | Log-loss | ECE | AUC |
|---|---|---|---|---|---|
| share (unsmoothed) | 0.157 [0.136, 0.181] | +0.27 | 0.512 | 0.057 | 0.810 |
| `support_prior` (engine) | 0.147 [0.127, 0.169] | +0.32 | 0.458 | 0.033 | 0.820 |
| L2, Sonnet + the evidence table | **0.129** [0.114, 0.146] | +0.40 | 0.403 | 0.041 | 0.875 |

Paired Brier: `support_prior` - share **-0.010 [-0.016, -0.006]**; `support_prior` - L2 **+0.018 [+0.009, +0.028]**.

- The prior generalises as a modest improvement on the share (and on log-loss and AUC); it is the better calibrated of the three (ECE 0.033).
- **The tie of the earlier post hoc result did not hold**: on these genes the LLM that reads the evidence is better than the smoothed share on Brier, log-loss and AUC,
  with intervals excluding 0, in every group of the number of datasets (n = 1: 0.339 against 0.256; n >= 10: 0.125 against 0.109).
- The unsmoothed share is much better here (0.157) than on the first item set (0.221): these genes were tested in more datasets (65% of the predictions have 10 or more),
  where the prior matters little. The first set's n = 1 weakness (27 of 240 items) is 25 of 1,197 here.
- Reading: the engine gives an exact, calibrated, auditable probability; it does not beat an LLM that sees the p-values and tissues of the other datasets, which the share ignores.
  A weighted choice of units (by the strength of each test) is the obvious next step and is not tested.

## Post hoc: what does L2 see that the share does not? (`post_hoc_logit.py`)

Decided after the result above, so exploratory. A logistic regression on eight features of the other datasets' tests (the smoothed share, n, the
strongest and the mean -log10 p, same-organ and same-source signals), fitted on the 2,763 predictions of the 222 genes of the first item set and
evaluated on the same 1,197 predictions of the 131 other genes (`results/post_hoc_logit.txt`):

| System | Brier | AUC |
|---|---|---|
| `support_prior` (engine) | 0.147 | 0.820 |
| logistic on the table | 0.131 | 0.858 |
| L2, Sonnet + the evidence table | 0.129 | 0.875 |

Paired Brier: logistic - `support_prior` **-0.016 [-0.022, -0.011]**; logistic - L2 **+0.002 [-0.006, +0.010]**.

- A simple model fitted on the table matches L2 within the interval: the LLM's edge over the share is the information the share ignores, mostly how
  strong the other tests are (largest coefficient: the mean -log10 p, +2.78) and whether datasets of the held-out dataset's organ found the gene
  (same-organ smoothed share, +1.19). It is not evidence of knowledge the table does not hold.
- These 131 genes were already the evaluation set of `PROTOCOL_PRIOR.md`; this is their second use, and the features were chosen after seeing that result.

# Graded units of evidence in the engine (`PROTOCOL_WEIGHTED.md`)

Pre-registered (protocol committed before the evaluation). The form was chosen by gene-grouped cross-validation on the 222 fitting genes
(`prototype_weighted.py`): **graded evidence**. Each dataset's test is evidence with a probability π set by the strength of its p-value, through a
probabilistic fact `unit_evidence(u, g, d)`. The choice `selected_unit` of `support_prior` is kept unchanged, and

    support_graded(g, d, PROB) :- (unit_evidence(u, g, d) & selected_unit(u)) // (unit_tested(u, g, d) & selected_unit(u))

equals (Σ π + a) / (n + a + b). The engine matches it to 3.3e-16 on all 6,482 evaluation predictions, with lifted solvers only. One batched query, with
the held-out dataset as an answer variable, solves all of them in 1.2 s (`analyse_weighted.py`, `results/weighted_analysis.txt`).

Primary: 1,197 predictions with support > 0 of the 131 evaluation genes. **These genes are not fresh**: this is their third use, and the logistic of
the previous section pointed to the strength of the tests.

| System | Brier [95% CI over genes] | Log-loss | ECE | AUC |
|---|---|---|---|---|
| `support_prior` (engine) | 0.147 [0.127, 0.169] | 0.458 | 0.033 | 0.820 |
| **`support_graded` (engine)** | 0.143 [0.124, 0.164] | 0.449 | 0.047 | 0.825 |
| logistic on the table (`post_hoc_logit.py`) | 0.131 [0.112, 0.151] | 0.413 | 0.010 | 0.858 |
| L2, Sonnet + the evidence table | **0.129** [0.114, 0.146] | 0.403 | 0.041 | 0.875 |

Paired, `support_graded` minus:
- `support_prior`: Brier **-0.0036 [-0.0058, -0.0016]**, log-loss -0.0087 [-0.0153, -0.0027];
- L2: Brier **+0.014 [+0.005, +0.024]**, log-loss +0.046 [+0.023, +0.070];
- logistic: Brier **+0.013 [+0.007, +0.018]**.

By the number n of other datasets (Brier: graded / prior / L2): n = 1 (25 predictions) 0.255 / 0.339 / 0.256; n = 2-3 (81) 0.206 / 0.229 / 0.205;
n = 4-9 (317) 0.161 / 0.163 / 0.149; n >= 10 (774) 0.126 / 0.125 / 0.109.

Secondary (fresh for this comparison): the 5,285 predictions with support = 0 of the same genes (base rate 1.3%). Here `support_prior` is
0.01 / (n + 0.51), at most 0.02, so it ranks only by how many datasets tested the pair. Graded minus prior: Brier -0.0005 [-0.0007, -0.0003],
log-loss -0.0067 [-0.0098, -0.0039]. AUC is 0.925 against 0.787 on the raw probabilities. The protocol's metrics clip to [0.02, 0.98], which
flattens `support_prior` here and gives 0.791 against 0.500; the clipped AUC of `support_prior` is an artefact of the clip.

False alarms: on the 806,883 predictions of genes never significant anywhere, the mean probability is 0.0044 (max 0.086), against 0.0019 for
`support_prior`.

Reading, by the rules fixed beforehand:
- **The graded evidence generalises**: it is better than `support_prior` on both sets, with intervals excluding 0. On the primary set the gain is small
  (a fifth of the gap to L2: 0.0036 of 0.018). Among pairs where no other dataset reached the threshold, it ranks the ones that replicate much
  better (AUC 0.925 against 0.787, unclipped).
- **L2 and the logistic remain better**, with intervals excluding 0. Grading closes the gap where few datasets tested the gene (n = 1 to 3: equal to L2).
  It leaves the gap where many did (n >= 4). There the logistic's other features (organ, source, mean strength across datasets) carry what a weighted
  share over units cannot. Neither a soft organ weight nor per-dataset weights helped in cross-validation (`results/prototype_weighted.txt`).
- The graded predicate is less well calibrated than `support_prior` (ECE 0.047 against 0.033) and better on Brier and log-loss.

Notes:
- The engine's bin edge at the threshold is 5.530178 (-log10 2.95e-6). It was first written 5.53; no test has a p-value between the two, and the
  rerun with the exact edge gives the same values.
- After the protocol commit, two variables of `prototype_weighted.py` were renamed (a spelling check); its output is identical.

`freeze.py check` now reports the engine files as changed since the freeze. That is expected: the frozen agent runs are reproducible at commit
`1f3ea39ce`. `support` and `support_prior` are unchanged (checked against the closed forms and against `results/eval_answers.jsonl`).

# Leak checks, after the evaluation (`leak_checks.py`, `results/leak_checks.txt`)

Two ways information could leak from the fitting genes into the evaluation, checked after the results above (so not pre-registered):
- **Linked genes.** Genes in one genomic region share a GWAS signal, so their SMR results are correlated, and the split was by gene. Loci were
  built from the probes' positions (`ProbeChr`, `Probe_bp`), joining genes within 1 Mb: 65 loci over the 353 genes. **111 of the 131 evaluation
  genes share a locus with a fitting gene** (992 of 1,197 predictions); only 205 predictions in 13 loci are clean.
- **The held-out dataset.** The fit had seen how each held-out dataset behaves on other genes. Check 2 refits π, a and b without the held-out
  dataset's outcomes.

| Set | `support_prior` | `support_graded` | logistic | L2 | graded - prior | graded - L2 |
|---|---|---|---|---|---|---|
| As published, 1,197 | 0.147 | 0.143 | 0.131 | 0.129 | -0.0036 [-0.0059, -0.0016] | +0.014 [+0.006, +0.023] |
| Clean loci only, 205 (13 loci) | 0.152 | 0.149 | 0.137 | 0.138 | -0.0033 [-0.0100, +0.0002] | +0.011 [-0.005, +0.039] |
| Locus-grouped 5-fold refit, out of fold, 3,960 (65 loci) | 0.137 | 0.135 | 0.127 | - | -0.0026 [-0.0049, +0.0011] | - |
| Dataset-nested refit, 1,197 | 0.147 | 0.145 | - | 0.129 | -0.0026 [-0.0048, -0.0007] | +0.016 [+0.007, +0.025] |

Intervals resample loci in the first three rows below the published one, and genes in the last. With a locus-grouped split, the logistic stays
better than `support_graded` (-0.0078 [-0.0120, -0.0050]).

Corrected reading:
- **The evaluation split leaked**: most evaluation genes are linked to fitting genes. The point estimate of the gain of `support_graded` over
  `support_prior` barely moves (-0.0026 to -0.0036 everywhere), but **once the split respects loci its interval reaches 0**. "The graded evidence
  generalises" is therefore not established: it holds with a gene split and with a dataset-nested fit, and is borderline with a locus split.
- Leaving the held-out dataset out of the fit changes little (graded still better than prior, the LLM still better than graded).
- The ordering logistic < graded <= prior holds under every split. The gap of the engine's predicates to the LLM holds except on the 13 clean loci,
  where the interval is too wide to say.
- For future evaluations of this kind, split by locus, not by gene.

# Can the engine beat an LLM given the same evidence? (`PROTOCOL_SAME.md`)

Pre-registered. Models were scored out of fold under 5 locus-grouped folds (`same_evidence_models.py`, `results/same_models.txt`). The LLM
arms are Sonnet 5.5 through OpenRouter, 2,874 answers, none cut short, $11.31 (`run_same_llm.py`, `results/same_llm.jsonl`). Readings are on the
1,437 items of `items.json` and `items_eval.json`, 65 loci, base rate 31.2% (`analyse_same.py`, `results/same_analysis.txt`). All these
predictions had been seen in earlier analyses.

| System | Evidence | Brier [95% CI over loci] | Log-loss | AUC |
|---|---|---|---|---|
| L2-full (Sonnet) | p-values, tissues, kinds, gene | **0.128** [0.112, 0.148] | 0.402 | 0.874 |
| F2 logistic (fitted) | full table, 8 features | 0.131 [0.110, 0.156] | 0.418 | 0.850 |
| L2-flags (Sonnet) | which datasets found it, gene masked | 0.143 [0.123, 0.166] | 0.450 | 0.824 |
| E* engine (`support_graded` in every fold) | flags and binned p-values | 0.145 [0.123, 0.169] | 0.453 | 0.821 |
| E1 `support_prior` | flags | 0.145 [0.124, 0.168] | 0.456 | 0.819 |
| F1 best function of (k, n) | flags | 0.147 [0.125, 0.170] | 0.465 | 0.806 |
| E3 noisy-OR | flags and binned p-values | 0.150 [0.128, 0.173] | 0.472 | 0.786 |
| F3 boosted trees (fitted) | full table | 0.159 [0.124, 0.193] | 0.532 | 0.846 |

Readings, by the rules fixed beforehand:
1. **Ceiling: not the case.** No model fitted on the full table beats L2-full: logistic +0.003 [-0.005, +0.013], trees +0.031 [-0.004, +0.070],
   F1 +0.019 [+0.009, +0.028]. On this data, "the engine beats an LLM reading the full evidence" is not the case.
2. **Same evidence: equal.** E* - L2-flags +0.0016 [-0.0011, +0.0044]; `support_prior` - L2-flags +0.0020 [-0.0013, +0.0053]. Given only the
   engine's evidence, Sonnet and the engine are indistinguishable: the engine does not beat it, and it does not beat the engine.
3. **Parity on the full table: not reached.** E* - L2-full +0.016 [+0.009, +0.025], above the 0.010 margin. Exact lifted forms recover none of the
   logistic's advantage on these items (E* - logistic +0.013 [+0.010, +0.017]).

What this means:
- **The LLM's whole lead is the extra evidence.** L2-flags - L2-full is +0.015 [+0.008, +0.023], the same size as the engine's gap. Given the same
  evidence, Sonnet's reasoning adds nothing over the engine's exact share.
- `support_prior` sits at the ceiling of what (k, n) allows: F1, the best function of k and n, scores 0.1367 against 0.1374 on all 3,960
  predictions.
- The rerun through OpenRouter matches the earlier direct answers: L2-full - L2 direct is -0.002 [-0.005, +0.001].
- The engine's case rests on what it is, not on accuracy:
  - exact, by lifted inference;
  - calibrated (ECE 0.033 for `support_prior`);
  - auditable, as a stated model over units of evidence;
  - fast: milliseconds per answer, against $11.31 for these 2,874 answers.

Deviations:
- F1 was first run shrinking each (k, n) cell towards the base rate, which my brief to the agent said but the protocol did not. It was rerun
  shrinking towards the fitted prior, as the protocol says; the first version scored 0.165, the corrected one 0.137 on all predictions.
- In F3, one dataset never appears among the "other" datasets in training. Its column is all missing and is dropped before fitting.
- One outer fold holds 1,729 of the 3,960 predictions: one locus is very large.

# A ladder of questions by first-order complexity (`PROTOCOL_LADDER.md`)

Pre-registered, with one amendment before any LLM call: the draw is restricted to pairs with signal (see the protocol).
- Three datasets are held out per item.
- The engine composes `support_prior` exactly, with lifted solvers only (equal to the closed forms to 4.4e-16), and treats the held-out datasets'
  outcomes as independent.
- Sonnet 5.5 through OpenRouter, 3,000 answers, 3 without a probability, $16.94.
- Brier is clipped to [0.02, 0.98]; intervals resample loci (`analyse_ladder.py`, `results/ladder_analysis_signal.txt`).

| Rung (formula) | n | Base rate | Engine | Sonnet, same evidence | Sonnet, full table | Engine - full [95% CI] |
|---|---|---|---|---|---|---|
| Q1 s(h1) | 300 | 0.31 | 0.140 | 0.120 | **0.108** | +0.033 [+0.009, +0.066] |
| Q2 ∃q∈H. s(q) | 300 | 0.55 | 0.226 | 0.195 | **0.168** | +0.059 [+0.027, +0.106] |
| Q3 ∀q∈H. s(q) | 300 | 0.18 | **0.063** | 0.063 | 0.071 | -0.008 [-0.029, +0.017] |
| Q4 ∃≥2 q∈H. s(q) | 300 | 0.33 | 0.148 | 0.133 | **0.115** | +0.033 [+0.009, +0.064] |
| Q5 brain ∧ ¬blood | 150 | 0.06 | 0.120 | **0.077** | 0.083 | +0.037 [-0.008, +0.072] |
| Q6 PD ∧ AD in one dataset | 150 | 0.05 | 0.036 | 0.039 | **0.035** | +0.001 [-0.008, +0.009] |

Readings, by the rules fixed beforehand:
1. **H1, not shown.** The engine's deficit against the full table on the pooled harder rungs (Q4 to Q6, +0.026) minus that on Q1 (+0.033) is
   -0.007 [-0.038, +0.018]. Logical complexity does not close the gap.
2. **H2, the opposite.** On the pooled Q4 to Q6, the engine minus Sonnet given the same evidence is **+0.017 [+0.005, +0.026]**: Sonnet with the
   engine's own evidence is better on the harder questions.
3. **H3, coherence.**
   - Sonnet with the full table violates the ordering P(all) ≤ P(at least 2) ≤ P(at least 1) ≥ P(h1) ≥ P(all) on 5 of 299 items (1.7%
     [0.5, 3.3]).
   - Sonnet with the same evidence violates it on none of 298.
   - The engine never violates it, by construction.

Why, from the per-rung pattern (exploratory):
- The engine is worst where independence matters most. On "at least one of three" (Q2) its skill is 0.09 against 0.32 for Sonnet, and on Q5 its
  skill is -1.13. It is best on "all three" (Q3).
- The held-out datasets share the GWAS and some donors, so their outcomes are strongly correlated. The engine composes exactly, but under an
  independence assumption that is wrong for these data, and composition compounds the error. Sonnet's answers behave as if it allowed for the
  dependence.
- The lesson is the one of the HPO posterior: exact inference over a misspecified model loses to an LLM's looser judgement. Exactness pays only
  when the model's structure is right.
- A model that represents the dependence would test this directly. One option is a latent "real effect" fact per gene and disease, with each
  dataset detecting it with some probability and false positives otherwise; it is still a probabilistic Datalog program.
