# Carrying probabilities through queries: results (protocol: `PROTOCOL.md`, amendments 1 to 5 written before the runs)

## Sonnet 5.5 (through OpenRouter)
- 143 scored questions, three arms, one run each, on the Margaret cluster: job 987206, 4 h 05 min.
- Cost: S0 $1.03, S1 $1.63, S2 $2.01, as billed.
- Answers: `results/S0.jsonl`, `S1.jsonl`, `S2.jsonl`. Scoring: `results/score_sonnet.txt`.
- **Correct** = within 1e-3 of gold (P7: within 1%). Intervals resample templates.

| Level | S0 Sonnet alone | S1 Sonnet + SQL | S2 Sonnet + probabilistic Datalog |
|---|---|---|---|
| P1 one uncertain row | 0.17 | 1.00 | 1.00 |
| P2 ∃ over rows | 0.00 | 0.83 | 1.00 |
| P3 ∃ through a join | 0.00 | 1.00 | 1.00 |
| P4 shared variable | 0.00 | 0.96 | 1.00 |
| P5 events sharing rows | 0.00 | 0.96 | 0.96 |
| P6 conditioning, negation | 0.14 | 0.95 | 1.00 |
| P7 expected counts | 0.00 | 1.00 | 1.00 |
| **All (143)** | **0.04** [0.00, 0.11] | **0.958** [0.909, 0.993] | **0.993** [0.978, 1.000] |
| Without the 4 restored templates (126) | 0.05 | 0.968 [0.913, 1.000] | 1.000 [1.000, 1.000] |

Readings, by the rules fixed beforehand:
1. **H7, not shown.** (S2 - S1 on P4 to P7) - (S2 - S1 on P1 and P2) is -0.061 [-0.250, +0.045]; without the restored templates, -0.069 [-0.250,
   +0.036]. Both agents are near the ceiling at every level, so the advantage cannot grow with the level.
2. **S2 against S1:** +0.022 [0.000, +0.054] on P4 to P7, and +0.035 [0.000, +0.084] on all levels. The Datalog arm is better by a few points;
   the intervals touch 0.
3. **S0, the floor:** 4%. The questions need the data.

**Effort, S2 against S1** (means per question, all levels; intervals over templates):

| | S1 SQL | S2 Datalog |
|---|---|---|
| Cost | $0.030 | $0.053 |
| Cost per correct answer | $0.031 | $0.053 (ratio 1.72 [1.49, 1.97]) |
| Model calls | 2.1 | 2.8 |
| Failed tool calls | 0.02 | 0.09 |
| Agent time (model + tool) | 7.7 s | 12.7 s |
| Wall time | 7.7 s | 77.6 s (the NeuroLang harness adds about 65 s per question) |
| Final query | 149 characters, 1.2 lines | 162 characters, 2.6 lines |

**Where the arms erred** (exploratory, from the answers):
- **S1, 6 errors.**
  - In 5 of them (P2a three times, P5d-5, P6b-2) the final SQL only retrieved the rows, and Sonnet composed the probabilities in its own reasoning,
    getting the arithmetic wrong.
  - In the sixth (P4d-0) it wrote SQL enumerating possible worlds and got the sum wrong.
  - So S1's short queries partly reflect arithmetic moved out of SQL, which the query-length measure does not see.
- **S2, 1 error.**
  - On the three shapes the lifted solvers refuse as written (P4d, P5d, P6d), the Datalog agent found formulations the engine accepts, or wrote
    the probability formula by hand as Datalog arithmetic.
  - Its one error (P5d-1) is such a hand-derived formula.

What this shows for Sonnet 5.5:
- Composing probabilities through queries of this size is not hard enough for Sonnet to separate the arms by much. With SQL it is right 96% of the
  time, with probabilistic Datalog 99%.
- The Datalog arm's few-point edge comes from not doing arithmetic by hand. It costs about 1.7 times more per correct answer and takes longer.
- The claim that SQL "becomes hard to follow" is not borne out at these sizes: Sonnet keeps its SQL short by doing the probability in its head,
  which is where its errors are.

## Qwen3.6-27B (vLLM on Margaret, amendment 4)
- 143 scored questions, three arms, one run each: jobs 988204–988206 (S2 took about 14 h on one A40). Answers in `results/qwen36-27b/`.

| Level | S0 alone | S1 + SQL | S2 + probabilistic Datalog |
|---|---|---|---|
| P1 | 0.11 | 1.00 | 1.00 |
| P2 | 0.00 | 0.72 | 0.94 |
| P3 | 0.00 | 1.00 | 0.89 |
| P4 | 0.00 | 0.89 | 0.88 |
| P5 | 0.00 | 0.83 | 0.83 |
| P6 | 0.14 | 0.81 | 0.62 |
| P7 | 0.00 | 0.83 | 0.94 |
| **All (143)** | **0.035** [0.000, 0.089] | **0.867** [0.771, 0.953] | **0.867** [0.776, 0.938] |

Readings, by the rules fixed beforehand:
1. **H7, not shown.** (S2 - S1 on P4 to P7) - (S2 - S1 on P1 and P2): -0.134 [-0.341, +0.028].
2. **S2 against S1:** +0.000 [-0.073, +0.075] on all levels, -0.022 [-0.120, +0.068] on P4 to P7. The arms tie.
3. **S0:** 3.5%.

Exploratory:
- **Query length:** SQL grows with the level and Datalog does not. P5: SQL 13.3 lines (433 characters), Datalog 2.0 lines (184). P6: 9.3 against
  2.1 lines.
- **Datalog misses are translation, not refusals:** zero lifting refusals. P6 (0.62) is the weak level: Qwen drops the "given" of the P6a
  questions (5 of 6 wrong), and writes quantifiers in the wrong scope.
- **Form of the final program** (`classify_queries.py`): declarative 90 (63%), hybrid 38 (27%), hand-grounded 8 (6%), hand-coded 1, no answer 6.
  Unlike Sonnet, Qwen composes probabilities by hand at every level from P2 on, often as noisy-or written with `log` and `exp` over an
  `AGGREGATE` sum.

## Amendment 5, run A: Sonnet 5.5 S2 on master with the lifting fixes
- NeuroLang master b89d83164 (#1414–#1418 merged), prompts and guidance unchanged, job 989049. $1.46 (the original S2 run: $2.01). Answers in
  `results/sonnet-master/S2.jsonl`.

| | Original S2 | Run A |
|---|---|---|
| Correct | 142 / 143 | **143 / 143** |
| Declarative (primary) | 112 (78%) | **126 (88%)**, paired difference +0.098 [+0.007, +0.225] |
| Declarative, on the 14 questions that hit a refusal | 1 / 14 | **14 / 14** |
| Hand-grounded | 8 | 0 |
| Hybrid | 22 | 17 |
| Hand-coded | 1 | 0 |

- **Primary, as registered:** the declarative share rises by 9.8 points; the interval resamples templates and excludes 0. Every question that hit
  a lifting refusal is now answered by one `PROB[...]` query; the hand-grounded genes, the hand-derived formula of P5d-1 (the only error of the
  original run) and the arithmetic over several `PROB` results are gone.
- **Transitions:** 14 answers became declarative (8 hand-grounded, 5 hybrid, 1 hand-coded); none moved the other way.
- **The 17 remaining hybrid answers are all P7** (expected counts): a sum of `PROB` results by `AGGREGATE`, the only way to write an expectation
  before `EXPECT`. Run B tests that.
- **Secondary:** accuracy 143/143 against 142/143.
- **The classifier against a hand reading** of 20 answers (10 of run A, 10 of Qwen S2, drawn with seed 0): 20 of 20 agree.
