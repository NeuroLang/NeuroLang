# Does a model of the dependence make exact composition pay? (written before any fit or run, 2026-11-25)

On the ladder (`PROTOCOL_LADDER.md`) the engine composed `support_prior` exactly but treated the three held-out datasets as independent. It lost to
Sonnet given the same evidence on the harder rungs (+0.017 [+0.005, +0.026]), and lost most on "at least one of three" (Q2). The held-out
outcomes are correlated: the datasets share the GWAS and some donors. This protocol gives the engine a model of that dependence and re-scores the
same items against the same Sonnet answers (no new LLM call).

## Model: a latent real effect
For each gene and disease, a latent fact `real` holds with probability ρ. Given `real`, each dataset that tested the pair finds it significant
independently, with probability α_o set by the dataset's organ o (brain, blood, other). Given not `real`, it does so with probability β (false
positives). The datasets' outcomes are therefore independent given the latent and correlated without it. For Q6, PD and AD have independent latents.

Every question is answered exactly by conditioning on the evidence (which other datasets found it significant, and which did not):

    P(Q | e) = P(real | e) · P(Q | real) + P(¬real | e) · P(Q | ¬real),
    P(real | e) ∝ ρ · Π_e α_o^s (1 − α_o)^(1−s),   P(¬real | e) ∝ (1 − ρ) · Π_e β^s (1 − β)^(1−s).

Here P(Q | real) is the rung's formula over independent held-out facts with probabilities α_o, and P(Q | ¬real) the same with β.

- **Engine.** The model is written as a probabilistic Datalog program. The latent is a probabilistic choice between `real` and `null`; the held-out
  facts are probabilistic, conditioned on the latent through rules; the evidence enters a conditional query `//`. It is solved with the lifted
  solvers only.
- If the lifted solvers refuse a form, the refusal is reported with the program. The closed form above is then reported as "the model in closed
  form", not as the engine.
- The engine must equal the closed form to 1e-9 wherever it solves.

## Fitting (no ladder outcome used)
- The five parameters (ρ, α_brain, α_blood, α_other, β) are fitted by log-loss on the 3,960 single-held-out predictions with support > 0
  (`predictions.parquet`): P(held-out significant | evidence) is the Q1 form of the model.
- The fit is out of fold, over the same 5 locus-grouped folds as `PROTOCOL_SAME.md`. Each ladder item is predicted with the parameters fitted on the
  folds that do not contain its locus.
- Bounds: 0 < β < α_o < 1 and 0 < ρ < 1, which keeps the latent identifiable as the "real" state.

## Readings, fixed now (same Brier, clipping and locus resampling as the ladder)
1. **H4, the dependence helps.** Latent engine minus independent engine, on the pooled dependence-sensitive rungs Q2, Q4 and Q5, and on all rungs
   pooled:
   - below 0: modelling the dependence helps;
   - containing 0: not shown.
2. **H5, same evidence.** Latent engine minus Sonnet-flags, on the pooled Q4 to Q6 (as the ladder's H2) and on all rungs pooled:
   - below 0: the engine beats Sonnet given the same evidence;
   - containing 0: equal;
   - above 0: Sonnet is better.
3. **H6, full table.** Latent engine minus Sonnet-full, on the same pools, read the same way.
4. **Single question.** Out-of-fold Brier of the latent model on all 3,960 single-held-out predictions, against `support_prior` (E1) and F1, the best
   function of (k, n), from `same_models_oof.json`, with paired intervals over loci.
5. Coherence: the latent engine is coherent by construction; this is checked on Q1 to Q4 with the ladder's tolerance.
6. Everything is reported, whichever way it comes out. The model's form and its five parameters are fixed above; no other variant is tried before
   this is reported.
