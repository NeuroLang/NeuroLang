# Carrying probabilities through queries: results (protocol: `PROTOCOL.md`, amendments 1 to 4 written before the runs)

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
