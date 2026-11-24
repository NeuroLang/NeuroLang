# BiomedSQL on NeuroLang: protocol of the agent benchmark

Written before any run on the test items. Deviations are listed at the end, with the date and the reason.

## Question

Can a ReAct agent (Claude Sonnet 5.5) answer the questions of BiomedSQL, a text-to-SQL benchmark of the genetics of
neurodegenerative diseases and of drugs, by writing NeuroLang (Datalog) queries over the `biomedsql` engine, and how
much of that depends on what the engine tells it?

This benchmark is **deterministic**: BiomedSQL has no uncertain questions. The probabilistic side of the engine (the
QTL datasets as units of evidence) is evaluated by its calibration against held-out data, not here
(`doc/biomedsql_calibration.rst`).

## What makes it independent of the people who built the engine

1. **Questions and gold answers are BiomedSQL's**, written by its authors (NIH-CARD): the question text and the
   stored execution results of its gold SQL. No question and no gold query is written for this benchmark.
2. **All 40 templates** of BiomedSQL are used, 4 items each, drawn uniformly at random (seed 0) **whatever their
   answer**: empty answers (20% of items) and answers cut by their LIMIT are kept.
3. **Development and test templates** (`items.json`): the 40 templates are split by table, at random (seed 0), into
   20 development and 20 test templates. The engine's guidance and the few-shot bank are written from development
   templates only. Every headline number is on the 80 test items (20 templates).
4. **The scorers are fixed before the test run** and validated on controls built from development items only.
5. **Chance is measured**: an always-empty answer, another item's gold table, and Sonnet with no database.
6. **Repeats**: every condition is run 3 times; intervals resample templates.

Limits that remain: the engine, its schema text and the guidance were written by the authors of this benchmark; the
author of the guidance had seen the gold SQL of one item of every template (including the test templates) before the
split was drawn; BiomedSQL's templates are few (40), so test items are 20 templates, not 80 independent tests.

## Systems and conditions

The agent is the repository's `ReactAgent` (Sonnet 5.5 through the Inria key; Datalog; at most 16 steps; no
verification step; a final query is run up to 300 s; the engine holds all five BiomedSQL tables with their columns).

| Condition | What the agent is given |
|---|---|
| A | the engine's schema: its relations, their columns and descriptions (written once, from column names and types) |
| B | A and `guidance.txt`, written from development templates |
| C | B and `bank.json`: worked examples (question, query) of development templates, retrieved by similarity (6 per question) |

Baselines: `always empty`; `other item` (the gold table of another item of the same template); `no database`
(Sonnet answers from its own knowledge, as a text table); `oracle` (BiomedSQL's gold SQL translated mechanically
to Datalog, run on the engine).

## Scoring (frozen)

Gold = BiomedSQL's stored result of the item, without its UUID column, as a set of rows with canonical values
(`common.canon`: 6 significant digits, NaN and None empty, booleans true/false). A result cut by its LIMIT
(LIMIT > 1 and as many rows as the LIMIT) is a *truncated* list; LIMIT 1 is an exact answer.

* **Judge (primary)**: Sonnet, blind to the query and the condition, compares the system's answer table with the gold
  table under the rubric of `scorer.py` (extra columns, other names or order are fine; a column the question does
  not need may be missing; missing or extra rows are wrong; an empty gold needs an empty answer; a truncated gold
  needs all its rows). On controls of the 80 development items (`validate_judge.py`): 400 of 402 correct (99.5%).
  Its limit: the controls that change the answer exist for the 28 development items with an answer-bearing column
  (a value that BiomedSQL's own prose answer states and the question does not).
* **Strict (secondary, a lower bound)**: the answer reproduces all the gold columns, exactly, as a set of rows
  (names and order do not matter; extra columns do not pass; truncated gold: its rows must be in the answer).

## Comparison with published results (fixed before the test run)

The reference is the BiomedSQL paper (Koretsky et al., arXiv:2505.20321v6, COLM 2026), Tables 3 and 4: 546 questions
sampled from the 68,227 over the same 40 templates; BigQuery SQL; ten tables; execution accuracy (EX: the set of
UUIDs, or the numbers, returned equals the gold result), Jaccard (JAC), and an LLM-judged answer quality (RQR, GPT-4o).
Best published: Gemini-3-Pro 58.1% EX / 81.8% RQR (isolated prompt), BMSQL-o3-mini 62.6% EX / 83.2% RQR, DAIL-SQL-o3-mini
61.2% EX; experts 90.0% EX on 20 questions (95% CI about 68 to 99).

Our metrics are not theirs, so no number of ours is set against one of theirs as "better":
* our **judge** accuracy is the nearest to their RQR (a model judging the answer against the gold result), with a different judge
  (Sonnet 5.5, the same model as the agent, so a self-preference bias is possible; theirs is GPT-4o);
* our **strict** accuracy is stricter than their EX (all columns must match, not only the row identities);
* our task is easier in the schema (5 tables the templates read, not ten, with descriptions: table choice is the
  commonest published error) and harder in the language (Datalog, with a few queries per question and the
  engine's quirks); the agent runs queries and sees their results, as their ReAct and BMSQL systems do;
* our items are 20 templates (80 items), theirs the 40 templates (546); their intervals are over items, ours over templates;
* their few-shot examples are the 40 template queries, the test templates included: published few-shot gains are
  in-template, and our condition C is in-template for 18 of the 20 test templates (twins) and out-of-template for 2.

What is reported: the table of published and our numbers with these differences next to it, and the conditions that
match best (A or B with the closest prompt content; C against DAIL-SQL and 10-shot). No claim of superiority is made
from numbers on different metrics, items, schemas and languages.

## Statistics (frozen)

For a condition: the accuracy of each template is the mean over its 4 items and 3 runs; the accuracy of the
condition is the mean over the 20 test templates. 95% intervals: bootstrap over templates (10,000 resamples). Paired
differences between conditions use the same resamples. Reported: judge and strict accuracy; the same on the items
with a non-empty gold answer (where an empty answer cannot pass); per template; the three runs apart; the baselines
and the oracle; cost and time.

No hypothesis is tested against a threshold. The comparisons asked are A with B and B with C, and each condition with
the baselines.

## Development

The guidance and the bank are written by running A on the development items, reading the failures, adding to the
guidance only what the engine can state without reference to a template's answer, and rerunning B and C on the
development items. Development results are reported but are not evidence (they were used to tune). `guidance.txt`,
`bank.json`, `items.json`, `scorer.py`, `run_agent.py` and `llm.yaml` are hashed in `FROZEN.json` before the test
run, and `run_agent.py` refuses to start a test run whose hashes differ.

## Development log (before the freeze)

Development items only (80 items, 20 templates, one run). Judge accuracy.

| Step | A | B | C | Change made |
|---|---|---|---|---|
| Schema fixed (full BiomedSQL tables, original column names; booleans stored as the text 'True'/'False' because Datalog has no True/False constants) | 48 | | | |
| Round 1 guidance and bank (19 examples) | 48 | 66 | 70 | from A's failures: bare booleans, unstated SMR criteria, approval, mechanism |
| Round 2 guidance (two clarifications of sentences already present: report only significant SMR rows; list all GWAS variants, not the best one) | 48 | 77 | 78 | none after |

Development scores are not evidence: the guidance and the bank were written from these items. Two rounds were the limit.

Changes made on development data before any test run: (1) a bare true/false is not an answer to a yes/no question
(the first rubric accepted it when it agreed with whether the gold table had rows, which rewards guessing);
(2) LIMIT 1 results are exact answers, not truncated lists; (3) the rubric asks the judge to compare row by row
and list the differences before its verdict; (4) controls of the judge were redesigned three times because the
first controls changed values the question does not ask for (the judge was right and the controls were wrong); the
final controls change only columns whose values BiomedSQL's own prose answer states and the question does not.
Final judge validation on the development items: 481 of 482 controls (99.8%); the strict scorer 482 of 482.

### Provenance of the guidance

Each sentence of `guidance.txt` states a convention visible in the gold SQL of development templates (checked by
pattern, `provenance` below). A convention visible only in test templates is not stated: in particular the
genome-wide threshold of GWAS variants (p < 5e-8; test templates Q26 and Q45) and HEIDI in questions that do not
ask for a top SNP.

| Guidance | Development templates where the gold SQL shows it | Also in test templates |
|---|---|---|
| answer with supporting rows, never a bare boolean | Q1, Q5, Q23, Q35, Q50 (the gold is the table) | many |
| drug names by substring over name, trade names, synonyms | Q5, Q12, Q14, Q15, Q17, Q23, Q58, Q59, Q80 | Q6, Q7, Q9, Q22, Q34, Q38, Q61, Q63, Q65 |
| genes of a drug: newLinkedTargets_string, 'SYMBOL ' | Q1 | Q4 |
| approved for a disease: max_phase_for_ind = 4.0 | Q16 | none |
| mechanism of action = drugMechanismOfAction | Q14 | Q22, Q38 |
| SMR significant: p_SMR_multi < 2.95e-6 | Q11, Q19, Q49, Q55 | Q40, Q52 |
| top SNP: topRSID, smallest p, p_HEIDI > 0.01, AGGREGATE min | Q19, Q55 (smallest p: Q11, Q49) | Q67 (smallest p) |
| tissue/type by lcase(Omic_tissue) / lcase(Omic_type) | Q19, Q55 | Q67 |
| GWAS: all variants with p and b, smallest p first; nearestGene substring | Q35, Q50 | Q45 |
| language notes (one ans; lcase(x) == c needs a binding) | the engine, not a template | |

Not guaranteed: the author of the guidance had seen the gold SQL of one item per template, test templates
included, before the split was drawn. The test templates are 18 of 20 with a development twin (`families.py`:
same table and WHERE columns sharing half or more); only Q26 and Q71 (aggregates) have none. The test therefore
measures mostly generalization to a new constant and a sibling template, not to a new kind of question.

## Deviations
- 2026-11-24 (after the test run): `post_hoc_ex.py` defines a projection-EX metric, an analogue of the paper's EX,
  after seeing that the strict scorer is not comparable to it. It is exploratory and reported only next to the judge.
- `oracle.jsonl` was copied into the results directory for `analyse.py`, which looks for it there (no code change).
- The test run was one launch of `run_test.sh` (3 repeats of each condition); no item was rerun and none dropped.
  `freeze.py check` reported no change before and after the run.
- The judge is the same model as the agent (Sonnet 5.5): a possible self-preference bias, not controlled here.
- 2026-11-24 (after the test run), engine v2: (1) `engines.yaml` gives `biomedsql` a `datalog_init`, the rule
  `support(g, d, PROB)` (the share of the QTL datasets that tested a gene for a disease and found it significant), and
  declares it as a predicate; a test of the repository (`test_datalog_init_in_config`) requires one for every engine with a
  `python_init`. The agent now sees `support` in the schema. (2) The `label` of the engine was removed: it would have put an
  8 GB engine on the browser page (`test_repl_bundle`). Neither changes a deterministic question's answer. The test results above
  are v1 (the hashes of `FROZEN.v1.json`, taken before the run). To check that the new rule does not change the agent's results,
  conditions A and C were run once more on v2 (`v2check`, below).
- 2026-11-24, formatting: the code of `tools/biomedsql_benchmark` was reformatted with black (line length 79) after the run;
  the syntax trees are identical except for the whitespace of two docstrings (checked). `FROZEN.json` is the v2 freeze (new hashes);
  `FROZEN.v1.json` has those of the test run.

- v2check result (2026-11-24): conditions A and C rerun once on the v2 engine (test split, judge): A 70/80 (72/80 on v1), C 69/80 (69/80 on v1):
  no change beyond run-to-run noise.
- 2026-11-24, engine v3: `engines.yaml` gives `biomedsql` the predicates `support_prior`, `evidence_unit`, `unit_tested`, `unit_significant`, `selected_unit`
  (two imaginary units of evidence; `replication/PROTOCOL_PRIOR.md`) and its guidance one sentence about them. The test results above are v1
  (`FROZEN.v1.json`); `FROZEN.v2.json` is the freeze after the formatting; `FROZEN.json` is v3. No run under this protocol follows; the runs of
  `strategies/` and the replication experiments are under their own protocols.
