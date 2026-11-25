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
