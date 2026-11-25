# HPO union hybrid: results (protocol: `PROTOCOL_UNION.md`, written and amended before the run)

184 fresh real patients (66 diseases; intervals resample diseases, 10,000 draws), Sonnet 5.5 through OpenRouter (amendment 2). Recall@30 of the
posterior: 58.2%. Unparsed: 4 H0 answers (misses), 0 U1. Cost $3.10 at list price. Code: `union_hybrid.py`; per-patient results
`results/union_results.json`; output `results/union_analysis.txt`.

| Arm | Top-1 | Top-10 | MRR [95% CI over diseases] |
|---|---|---|---|
| posterior alone | 19.6% | 42.9% | 0.290 [0.205, 0.382] |
| H0 Sonnet direct | 35.9% | 57.6% | 0.427 [0.326, 0.530] |
| **U1 Sonnet with its own list and the 30 candidates** | **37.5%** | **64.1%** | **0.455** [0.359, 0.555] |
| U2 reciprocal-rank fusion of H0 and the posterior | 30.4% | 61.4% | 0.406 [0.314, 0.505] |

Paired MRR differences:

| Difference | Estimate [95% CI] |
|---|---|
| **U1 - H0 (primary)** | **+0.028 [-0.015, +0.081]** |
| U2 - H0 | -0.021 [-0.057, +0.017] |
| U1 - posterior | +0.165 [+0.115, +0.220] |
| U2 - posterior | +0.116 [+0.063, +0.170] |
| U1 - U2 | +0.049 [+0.013, +0.088] |

Lenient name matching (amendment 1): H0 0.526, U1 0.558; U1 - H0 +0.033 [-0.016, +0.089].

Split by whether the true disease is among the 30 candidates:

| | Posterior | H0 | U1 | U2 |
|---|---|---|---|---|
| among them (n = 107) | 0.494 | 0.594 | 0.669 | 0.633 |
| not among them (n = 77) | 0.007 | 0.195 | 0.158 | 0.091 |

## Reading, by the rules fixed beforehand
- **U1 is not detectably better than H0.** The interval of the primary difference contains 0, strictly and leniently, so the earlier conclusion
  stands for this design: Sonnet does not detectably gain from the posterior, even when it keeps its own list. Unlike the first hybrid, the
  candidates no longer cost anything (U1 is not worse; H1 was, -0.135).
- Both hybrids beat the posterior alone by a wide margin. Sonnet with the candidates (U1) beats the mechanical fusion (U2).
- Exploratory: when the true disease is among the candidates, U1 is the best arm (0.669 against 0.594 for H0). When it is not, U1 loses a little
  against H0 (0.158 against 0.195). The candidates pull Sonnet towards them, and that helps only when they contain the answer. These two subsets
  depend on the posterior only, but they are not tested for significance.

## Deviation found after the run: OpenRouter turns on reasoning
Through OpenRouter, Sonnet 5.5 reasons by default. A test call on one of this run's prompts used 737 reasoning tokens of 1,101 output tokens. The
direct Anthropic API of the earlier runs does not reason unless asked. The protocol did not anticipate this. Consequences:
- Both arms ran the same way, so the paired U1 - H0 comparison is internally consistent. H0 here is "Sonnet with reasoning", though, and its MRR is
  not directly comparable with the earlier runs' H0 (0.428 in the first hybrid, which had no reasoning; the values happen to be close).
- 3 of the 4 unparsed H0 answers were cut off at the 1,200-token limit, because reasoning used most of it. Counting them as misses favours U1.
  Without the 4 patients, U1 - H0 is +0.023 [-0.020, +0.076] (lenient +0.026 [-0.023, +0.082]), and the reading does not change.
- Answers by host: 365 served by Anthropic through OpenRouter. 3 H0 prompts were identical to prompts of the first hybrid (same terms) and were
  replayed from its cache, without reasoning.
- Switching reasoning off through OpenRouter (`"reasoning": {"enabled": false}`) was refused (HTTP 400). A replication without reasoning (direct
  key, or the right OpenRouter setting) would remove this caveat. It would be a new run, not a rerun of this one.

## Limits
- One model, one prompt per arm, one answer per patient.
- 184 patients over only 66 diseases, of which 60 were seen in earlier runs with other patients.
- Sonnet may have seen the published cases.
