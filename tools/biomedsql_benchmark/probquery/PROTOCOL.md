# Carrying probabilities through queries: Sonnet alone, Sonnet with SQL, Sonnet with probabilistic Datalog (written before any build or run, 2026-11-25)

Earlier tests asked Sonnet one-atom questions over evidence that had already been fetched and joined for it. There, an LLM reading a short table is
as good as the engine. BiomedSQL's own 40 templates have no join, no subquery and 2 aggregations. This benchmark asks the question where the
engine should matter: when carrying probabilities through a query becomes complex, does an agent writing probabilistic Datalog get the right
answer more often than an agent writing SQL?

## Database (identical for both agent arms)
The five BiomedSQL tables, as in the `biomedsql` engine, plus four derived **uncertain tables**. Every row of an uncertain table is an independent
event with probability `prob`. Questions state this, and both agents are told it.

| Uncertain table | Row | `prob` |
|---|---|---|
| `test_evidence(dataset, gene, disease, prob)` | a QTL dataset's SMR test is evidence that the gene mediates the disease | π of the test's −log10 p_SMR_multi (min over probes), the `evidence_strength` bins of `engines.yaml`; rows with π below 0.01 are dropped |
| `dataset_reliable(dataset, prob)` | the QTL dataset is reliable | brain 0.9, blood 0.8, other 0.7 (stated constants) |
| `indication_success(drug, mesh_heading, prob)` | the drug succeeds for the indication | by `max_phase_for_ind`: 4 → 1.0, 3 → 0.58, 2 → 0.29, 1 → 0.13, else 0.05 (stated constants) |
| `snp_causal(snp, gene, disease, prob)` | a GWAS SNP's nearest gene is causal | p_GWAS < 5e-8 → 0.5, < 1e-5 → 0.1 (PD and AD GWAS; others dropped) |

Deterministic joins:
- `drug_target(drug, gene)`, from `drug_indication.approvedSymbol`;
- `drug_approved(drug)`;
- `dataset_organ(dataset, organ)`.

The SQL arm sees every table in DuckDB. The NeuroLang arm sees the deterministic ones as relations and the uncertain ones as probabilistic facts.

## Levels of probabilistic structure (3 templates each, 6 instances per template: 126 questions)

| Level | Structure | Example template |
|---|---|---|
| P1 | one uncertain row | probability that dataset D's test of gene G is evidence for PD |
| P2 | ∃ over independent rows | probability that G is evidence for PD in at least one brain dataset |
| P3 | ∃ through a join | probability that drug X targets at least one gene with evidence for PD |
| P4 | conjunction of subqueries with a shared variable (hierarchical) | probability that some target of X has both evidence for PD and a causal PD SNP |
| P5 | events sharing rows (inclusion-exclusion) | probability that G has evidence for PD in a brain dataset, or in at least two datasets |
| P6 | conditioning and negation | probability that X targets a PD-evidence gene, given that target G has evidence in some blood dataset; or targets PD evidence but no AD evidence |
| P7 | aggregation of probabilities | expected number of approved drugs, among those for indication M, that target a gene with evidence for PD |

- **Templates** are written as natural-language questions that name the uncertainty model in one sentence ("each row of `test_evidence` is an
  independent event with probability `prob`").
- **Instances** (genes, drugs, indications) are drawn with seed 5 among those whose lineage holds 2 to 20 uncertain rows, so that exact
  enumeration is possible. For P7, the expectation is a sum of per-drug probabilities, each with lineage of at most 20 rows.
- **No instance has a #P-hard query:** every template is tractable, hierarchical or a union of hierarchical ones. The NeuroLang arm is solved with
  the lifted solvers only; a refusal is a failure of that question.

## Gold (independent of NeuroLang)
- For every question, a Python reference enumerates all possible worlds of its lineage rows (at most 2^20), evaluates the question as a Python
  predicate on each world, and sums the world weights.
- It shares no code with NeuroLang.
- 10% of questions are also checked by Monte Carlo (10^6 samples, within 3 standard errors).

## Arms (Sonnet 5.5 through OpenRouter; same step cap 25, same `max_tokens`, one run per question)
- **S0, Sonnet alone:** the question and the schema description, no database.
- **S1, Sonnet + SQL:** a ReAct agent with one tool, `run_sql` (DuckDB, read-only, 200 result rows shown), the schema and the uncertainty paragraph;
  no Python and no calculator; the final answer is a number.
- **S2, Sonnet + probabilistic Datalog:** the existing NeuroLang ReAct agent and engine guidance, plus the same schema and the same uncertainty
  paragraph, with the uncertain tables named as probabilistic relations.

No worked examples for either agent arm. Development uses 2 extra instances per template, not scored, only to make the harnesses run.

## Scoring and readings, fixed now
- **Correct:** within 1e-3 absolute of gold (P7: within 1% relative).
- Also reported: absolute error, the share of questions with no answer, steps and cost.
- Intervals are 95% and resample templates (21 clusters), 10,000 draws.

1. **Per level:** the accuracy of S0, S1 and S2.
2. **Primary (H7):** S2 − S1 accuracy on the pooled P4 to P7, against the same difference on the pooled P1 and P2 (difference of differences),
   resampling templates.
   - Above 0: the Datalog arm's advantage grows as the probabilistic structure gets harder. That is the claim this benchmark exists to test.
   - Containing 0: not shown.
3. **Secondary:** S2 − S1 on the pooled P4 to P7 alone, and on all levels.
4. **S0 is a floor.** Without the database it cannot know the rows, so near-zero accuracy is expected and shows that the questions need the data.
5. Every arm and level is reported, whichever way it comes out.

## Amendment, before any scored run (2026-11-25): the cost of each agent's effort
Requested before the build finished; no scored answer exists. For every question and arm, the harness logs:
- input, output and reasoning tokens, and the dollar cost at list price ($2 / $10 per million);
- steps (model calls) and tool calls, and of those, failed tool calls (SQL errors, NeuroLang parse or solver errors);
- wall-clock time, split into model time and tool time (SQL or engine execution);
- the length of the final query the agent relied on (characters and lines; for SQL also the number of `JOIN`, sub-`SELECT` and aggregate
  clauses).

Reported per level and pooled, S2 against S1, with intervals resampling templates:
- mean cost, steps, failed tool calls and time per question;
- **cost per correct answer** (total dollars over number correct), and the ratio S2 / S1;
- the final query's length.

Reading: no threshold is fixed. The effort comparison is descriptive and reported beside H7: whether the Datalog arm is cheaper, dearer, or
equal per question and per correct answer, and how the gap moves with the level.

## Amendment 2, before any scored run (2026-11-25): the build
What the build changed, and why, before any scored answer existed (development answers only: 7 questions per arm, S1 and S2 7/7, S0 0/7):
- **Templates.** 21 templates were written, 3 per level, as listed in `questions.json`. Four example shapes of the level table were first replaced
  because NeuroLang's lifted solvers refused them:
  - P4, evidence in both a brain and a blood dataset;
  - P4, both PD and AD evidence (too few instances rather than refused);
  - P5, "or in at least two datasets" (needs `!=`);
  - P6, PD but not AD evidence (the same table used positively and negated).

  Choosing templates by what the engine can solve would favour S2. **These four shapes are therefore restored as additional scored templates**
  (P4d, P4e, P5d, P6d, `engine_refused: true`), worded as in the protocol. Where S2 cannot produce an answer, it is scored as wrong, as the protocol
  says. Every reading is reported twice: on all templates, and without these four.
- **Other build choices:**
  - Instances whose gold is exactly 0 or 1 are not drawn.
  - Evidence-only templates draw from all six diseases (PD alone gives too few drug instances).
  - P6a conditions on evidence in a reliable dataset; the protocol's example, "given G has blood evidence", makes the answer always 1.
  - P1's lineage is one row, by the level's definition.
- **The S2 engine** is a NeurolangPDL with only the two lifted solvers. It loads the five BiomedSQL tables, the deterministic tables, and the
  uncertain tables as probabilistic facts, and holds no other relation of the `biomedsql` engine (no `qtl_*`, `unit_*`), so S1 and S2 see the same
  data.
  - S2's guidance is the uncertainty paragraph plus the Datalog language notes of the `biomedsql` guidance. The rest of that guidance (BiomedSQL's
    answer conventions and the `qtl_*` relations) is left out.
  - No worked examples for either arm.
- **Time.** Agent time (model plus tool) is reported separately from wall time, because S2's harness adds about 45 s per question that is neither.
- **Runs** happen on the Margaret cluster: DuckDB 2.0.1, one run per question and arm, `--max-usd 40`.

## Amendment 3, before any run with the second model (2026-11-25): a second model, Qwen3.5-122B-A10B
- **Why:** to test whether the comparison between the arms depends on the model.
- **Model:** Qwen3.5-122B-A10B (open weights; 10 B parameters active per token), official GPTQ-Int4 release. It is served with vLLM 0.33 on
  2 hopper GPUs of the Margaret cluster, with native tool calls (`qwen3_coder` parser) and thinking on (`qwen3` reasoning parser), as Sonnet
  reasons through OpenRouter.
- **What stays the same:** context 65,536 tokens; `max_tokens` 8,000; temperature 0; step cap 25. The same 143 scored questions, gold, prompts,
  schema and guidance. All three arms (S0, S1, S2), one run each; results in `results/qwen122/`.
- **Readings:** every reading of the protocol, including H7 and the effort comparison, is repeated for Qwen. Cost is reported as tokens and GPU time
  (no list price applies to a self-hosted model).
- **Comparison between the models:** descriptive only; no threshold is fixed. Is the S2 - S1 difference of the same sign and size?
- **Disclosed:** while setting this up, a few lines of the Sonnet S1 run's log were seen (single answers against gold). No Sonnet reading had been
  computed.

## Amendment 4, before any run with the second model (2026-11-25): Qwen3.6-27B instead of Qwen3.5-122B-A10B
At the user's choice, the second model of amendment 3 is **Qwen3.6-27B** instead: the newest Qwen generation, dense, 27 B parameters, official FP8
release, served with vLLM on 1 hopper GPU. Qwen 3.6 has no larger open-weight model. Everything else in amendment 3 is unchanged (parsers,
thinking on, context, `max_tokens`, temperature, step cap, questions, readings, descriptive comparison between the models). Results go to
`results/qwen36-27b/`. No answer from either Qwen model existed when this was written.
