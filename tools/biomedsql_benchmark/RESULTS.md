# Results of the BiomedSQL benchmark on NeuroLang, and comparison with the published ones

Test split of `PROTOCOL.md`: 20 templates, 80 items (13 with an empty gold answer, 7 truncated), 3 runs of each condition
(720 answers), agent = ReAct, Claude Sonnet 5.5, NeuroLang Datalog on the `biomedsql` engine (5 tables). Frozen files unchanged
(`freeze.py check`). 95% intervals resample templates.

| Condition | What the agent has | Judge accuracy | Strict | Projection-EX (post hoc) | Cost, 240 answers |
|---|---|---|---|---|---|
| A | schema only | 87.5% [75.8, 96.2] | 10.8% | 53.8% [39.2, 67.1] | $3.26 |
| B | A + guidance | 87.5% [76.2, 97.1] | 11.7% | 43.3% [29.2, 57.5] | $6.12 |
| C | B + 19 worked examples | 87.9% [77.9, 96.7] | 27.1% | 52.5% [35.8, 68.8] | $8.08 |
| Sonnet, no database | its own knowledge | 25.0% [8.8, 43.8] | 0% | - | - |
| always empty | - | - | 16.2% | 16.2% | - |
| oracle (gold SQL translated) | - | - | 100% | - | - |

Paired judge differences: B - A +0.0 [-14.2, +15.0]; C - B +0.4 [-13.3, +15.0]. Runs apart (judge): A 90.0 / 88.8 / 83.8,
B 85.0 / 86.2 / 91.2, C 86.2 / 88.8 / 88.8.

## Against the published results

BiomedSQL paper (Koretsky et al., arXiv:2505.20321v6, COLM 2026), 546 questions over the same 40 templates, BigQuery SQL,
ten tables. EX = row identities (UUIDs) returned equal the gold's; RQR = answer quality judged by GPT-4o (BioScore).

| System (published) | EX | RQR |
|---|---|---|
| Expert, 20 questions | 90.0 (95% CI about 68 to 99) | 95.0 |
| Gemini-3-Pro, baseline prompt | 58.1 | 81.8 |
| Claude-4.5-Opus, baseline prompt | 54.8 | 80.6 |
| GPT-o3-mini, baseline / 10-shot | 53.5 / 61.3 | 73.3 |
| ReAct-GPT-o3-mini | 56.2 | 73.6 |
| DAIL-SQL-GPT-o3-mini | 61.2 | 81.4 |
| BMSQL-GPT-o3-mini (best) | 62.6 | 83.2 |
| **This benchmark, agent A / B / C** | projection-EX 53.8 / 43.3 / 52.5 (post hoc) | judge 87.5 / 87.5 / 87.9 |

Reading it:
- The numbers are in the same range as the published systems (EX 47 to 63, RQR 70 to 85): **there is no evidence here that the
  NeuroLang agent is better or worse**. Our intervals are 20 to 30 points wide.
- The metrics differ. Our judge is the nearest to RQR (a model judging the answer against the gold; ours Sonnet, the agent's own model;
  theirs GPT-4o) and is lenient on columns: 372 of the 548 answers it accepted (non-empty gold) have fewer columns than the gold table.
  Our strict scorer is stricter than EX (all columns), so its 11 to 27% is not comparable. Projection-EX is the nearest to EX but was
  defined after seeing the results, and it requires the same number of distinct rows as the gold (it rejects an answer that lists
  distinct drug names where the gold has several rows per drug).
- The tasks differ. Ours has 5 tables (the ones the templates read), with descriptions; the paper's commonest error, the wrong table,
  is easier for us. Ours is in Datalog, a language the model knows less than SQL. Ours is 20 templates, theirs 40.
- The paper's few-shot examples are the 40 template queries, test templates included. Our condition C has examples of development
  templates only, yet 18 of the 20 test templates have a development twin.

## What the benchmark shows beyond the comparison

1. **The guidance did not help on the test templates** (A 87.5, B 87.5, C 87.9; development: 48 to 77 to 78, but that was tuned on).
   It helped where a convention is shared (Q52 SMR threshold: A 0%, B and C 100%; Q22 67 to 100; Q9 83 to 100; Q63 83 to 100) and hurt where the
   test template differs from the development ones: Q10 100 to 58 to 33, Q67 92 to 25 to 42, Q45 67 to 42 to 58, Q40 100 to 75 to 75.
   Cause, read in the failures: Q10 and Q45 need the genome-wide threshold `p < 5e-8` (not in the guidance, since only test templates show it);
   the round-2 sentence "list all variants, not only the best one", written for development templates Q35 and Q50, made the agent drop it.
   Q67 and Q40 are hurt by "report only significant SMR rows", which their gold does not apply. Conventions inferred from some templates
   are over-applied to others: the paper's own point on implicit conventions, here as an effect of the guidance.
2. **Without a database the model gets 25%**, mostly on drug lookups it knows; the engine adds 62 points (paired [+41, +82]).
3. **Empty gold answers (13 of 80 items) are the main error**: agents were right on 83 of 117 such answers (71%), and fail by returning
   non-significant rows without saying none is significant, the paper's "missing threshold" error again.
4. **Cost** is $0.014 to $0.034 per answer, median 3 to 4 model calls, median 11 to 17 s; the paper's ReAct and BMSQL use 13,000 to 39,000 tokens per question.

## Limits

20 templates (the unit of independence), 3 runs, one model; the judge is the agent's model; the guidance author had seen the gold SQL of
one item of every template before the split; the schema, the guidance and the benchmark were written by the same people; projection-EX is
post hoc; BiomedSQL's questions are templated and its gold queries are the authors' own, so scores measure agreement with them.
