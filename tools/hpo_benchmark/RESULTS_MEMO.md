# Memorisation probe: results (protocol: `PROTOCOL_MEMO.md`, written before the run)

Sonnet 5.5 through OpenRouter (reasoning on, as in the union run). Code `memo_probe.py`; per-item results `results/memo_results.json`;
output `results/memo_analysis.txt`.

## A. Synthetic patients (drawn from the annotation; in no paper)
127 patients, 7 terms each (2 H0 answers unparsed).

| | Posterior MRR | Sonnet H0 MRR | Sonnet - posterior [95% CI over diseases] |
|---|---|---|---|
| synthetic | 0.979 | 0.466 (lenient 0.584) | **-0.513** [-0.592, -0.433] |
| real (union run, 184) | 0.290 | 0.427 | **+0.137** [+0.057, +0.215] |

Real gap minus synthetic gap: **+0.650 [+0.551, +0.747]**.

Reading, by the protocol's rule:
- **Sonnet's edge on real patients comes from information outside the HPO annotation.** On patients drawn from the annotation, the posterior is
  nearly perfect (its own model generates them) and Sonnet is far behind. Sonnet is not better at reasoning over annotation-like evidence.
- On real patients the order flips: their terms are not samples from the annotation, which is the posterior's misspecification. Sonnet brings
  knowledge of what such patients look like.
- Control: Sonnet claimed a source paper for 24 of the 127 synthetic patients (19%), although none exists. A claimed source is weak evidence.

## B. Recognition probe (the union run's 184 real patients)
- Sonnet claimed a source for 21 patients and named the right paper for **1** (0.5%; right PMID, or first author with the year within one).
  Its rate of claiming a source on real patients (11%) is below its rate of claiming one for synthetic patients that have no paper (19%).
- On the 183 unrecognised patients, Sonnet still beats the posterior: MRR 0.424 against 0.291, **+0.132 [+0.051, +0.211]**, the same as on all
  patients.

Reading: Sonnet does not explicitly recall the case reports, and its edge does not need recall. This rules out explicit memory as the explanation.
It cannot rule out implicit memory (a paper read in training helping without being recalled), as the protocol says.

## C. Post-cutoff cases (descriptive)
23 cases from 7 papers published after June 2026: MRR posterior 0.635, Sonnet 0.620; Sonnet - posterior -0.016 [-0.557, +0.287] (resampling
papers). No edge here, but these cases are not comparable. Their annotation coverage is higher (mean 0.60 against 0.54 for the older test patients;
48% in the union run's high-coverage tertile), and that is where Sonnet and the posterior were already equal on older cases (-0.035,
`RESULTS_HYBRID.md`). Seven papers and a coverage confound: this part settles nothing.

## Overall
- The evidence favours **clinical knowledge outside the annotation** over memorised cases: the edge flips on synthetic patients, explicit recall is
  near zero, and the edge is unchanged on unrecognised patients.
- Implicit memory of the papers is not excluded. The test that would settle it, cases published after the cutoff, cannot be run yet: there are too
  few, and they are confounded by coverage.
- Caveats:
  - Synthetic patients are noise-free and drawn from the annotation, unlike the real patients, which differ in more than their source.
  - One model, one prompt per part, and reasoning on through OpenRouter.
