# HPO union hybrid: does Sonnet do better with its own list and the posterior's candidates? (written before the run, 2026-11-25)

Context (`RESULTS_HYBRID.md`):
- Restricted to the posterior's 30 candidates, Sonnet does worse than answering directly, because recall@30 is only 54%.
- The two retrievals are complementary: 73 of 100 true diseases are among the candidates or in Sonnet's direct top 10.
- Sonnet's advantage is where the HPO annotation covers little of the patient's terms (post hoc).

Hypothesis: given both its own direct answer and the posterior's candidates, Sonnet ranks the true disease better than directly.

## Patients
All 184 test patients (`results/hpo_split.json`) not used in phase 2 or in the first hybrid, whose disease is one of the 12,867. They cover
66 diseases: one disease may have several patients, and only 6 diseases are new, though every patient is new. The order is shuffled with seed 12,
and the 4 noise cells (0 or 2 spurious terms × 0% or 50% dropped) are assigned in turn, with the perturbation seeds of phase 1
(`union_hybrid.py plan`, `results/union_plan.json`). Intervals resample **diseases** (10,000 draws), because patients of one disease are not
independent.

## Arms (Sonnet 5.5, Inria key, one answer per patient and arm; answers are ranked lists of at most 10)
| Arm | What it is |
|---|---|
| posterior | the exact posterior with leak (`results/hpo_chosen.json`), its own ranking |
| H0 | Sonnet, the patient's terms only: rerun on these patients with the prompt of the first hybrid |
| **U1** | Sonnet, the terms plus the posterior's top 30 (most probable first, with probabilities) and its own H0 list; it may name any disease |
| U2 | reciprocal-rank fusion of H0 and the posterior's top 100 (constant 60, not fitted); no extra call |

The candidates are ranked by posterior with a stable sort, so that ties are ordered the same on every machine. The first hybrid's unstable sort
ordered ties differently across machines.

Dropped before the run: **U3, a gate** (posterior when a truth-free confidence proxy is high, H0 otherwise). On the 160 used patients
(`union_hybrid.py dev`, `results/union_dev.json`), neither proxy tracked the posterior-minus-Sonnet gap. The Spearman correlation was
+0.08 [-0.07, +0.23] for the top posterior probability and -0.02 [-0.18, +0.14] for the share of the patient's terms annotated to the top candidate,
and the rule fixed in the script before it ran (an interval excluding 0 and an in-sample gain of at least 0.02 MRR) kept neither. The oracle coverage
of `post_hoc_coverage.py` needs the true disease, so it cannot gate.

Development (not evidence): on the 100 patients of the first hybrid, U2 scored MRR 0.428, the same as H0 (0.427 after the amendment below).

## Metrics
- The rank of the true disease (a miss if absent), top-1, top-10, MRR, with strict matching by OMIM id or normalised name as before.
- **Primary: U1 - H0, paired MRR difference.**
- Secondary: U2 - H0, U1 - posterior, U2 - posterior, U1 - U2.
- Also recall@30, unparsed answers, and cost.

## Reading
- U1 better than H0, with an interval excluding 0: the posterior adds to Sonnet when Sonnet keeps its own list; the union is a useful hybrid.
- U1 equal to H0 (interval contains 0): no detectable gain; the earlier conclusion (Sonnet does not need the posterior) stands for this design.
- U1 worse than H0: the candidates mislead.
- No arm is rerun, and all are reported. The earlier 160 patients are not mixed in.

## Amendment, before any Sonnet call of this protocol (2026-11-25)
After a review of the protocol, before the run:
- **Naming sensitivity.** U1 sees exact OMIM ids and names of 30 candidates, while H0 must recall them, and scoring is strict. A U1 gain could
  therefore be a gain in naming, not in diagnosis. Also reported, as a sensitivity analysis: H0 and U1 scored with phase 2's lenient match (the
  OMIM id, or a name with difflib ratio >= 0.85 to the true disease's), and U1 - H0 on it. If U1 - H0 holds strictly but not leniently, the
  reading is "naming", not "diagnosis".
- **Split** of every arm's MRR by whether the true disease is among the posterior's 30 candidates (a property of the posterior only).
- **Fusion matching** (U2) now credits an answer item to every disease it names, by OMIM id or by name, as strict scoring does. Before, the OMIM
  field took precedence. The development fusion becomes MRR 0.427 (H0 0.428).
- `analyse` was exercised end to end on mocked answers; no real answer has been requested.
