# HPO rare-disease diagnosis: symbolic posterior, Sonnet, and hybrids

Rank the disease of real patients (phenopacket-store) from their HPO terms with:
- an exact naive-Bayes posterior over the 12,867 diseases of the HPO annotations, computed by a lifted NeuroLang query and by a numpy reference
  that agrees with it to 3.4e-13;
- Sonnet answering directly;
- hybrids of the two.

`ARTICLE.md` is the report of phases 1 and 2, `DESIGN.md` the design, and `PROTOCOL_*.md` / `RESULTS_*.md` the pre-registered follow-ups
(`.at` files give the time each protocol was written).

## Data (not in the repository)
Put these under `neurolang_data/probe/sci/` at the repository root:
- `phenotype.hpoa`: HPO annotations, version 2026-09-02;
- `hp.json`: the HPO ontology, release 2026-09-01;
- `pps/`: the phenopacket-store release 0.1.27 (`all_phenopackets.zip`, unzipped; BSD-3-Clause).

**The HPO licence has not been verified** (its page could not be read by our tools). The HPO files are not redistributed here, and the licence
must be checked before any publication.

## Running
Run every script from this directory: `uv run --package neurolang-apps --extra llm python <script>.py`. Scripts that ask Sonnet read
`ANTHROPIC_API_INRIA_KEY` from the environment (never from a file), and cache answers in `results/hybrid_cache/` (git-ignored).
Committed results are in `results/`. `hpo_phase1.jsonl` and the calibration dumps are not committed, because `hpo_phase1.py` and
`hpo_calibration.py` regenerate them deterministically.

| Step | Script | Output |
|---|---|---|
| Split and configuration (development diseases) | `hpo_dev.py` | `results/hpo_split.json`, `results/hpo_chosen.json` |
| Phase 1, held-out posterior | `hpo_phase1.py`, `hpo_summary.py` | `results/hpo_phase1.jsonl` |
| Engine against numpy | `hpo_lifted_check.py` | `results/hpo_lifted_check.jsonl` |
| Phase 2, agent and direct Sonnet (60 patients) | `step9_hpo_phase2.py`, `hpo_phase2_summary.py`, `hpo_phase2_family.py` | `results/hpo_phase2.jsonl` |
| Hybrid, candidates to Sonnet (100 patients) | `step14_hpo_hybrid.py` | `results/hybrid_results.json`, `results/hybrid_analysis.txt` |
| Post hoc: Sonnet's advantage by annotation coverage | `post_hoc_coverage.py` | `results/post_hoc_coverage.txt` |
| Union hybrid: development, plan, Sonnet arms | `union_hybrid.py dev`, `plan`; `run_union.sh` | `results/union_dev.json`, `results/union_plan.json`, `results/union_results.json` |

`step14_hpo_hybrid.py` ranks the candidates with numpy's default (unstable) sort. Tied posteriors may come out in a different order on another
machine, so its H1 to H3 prompts are not replayed from a cache made elsewhere (87 of 100 were not, on the machine of the union run). Its H0 prompts
are, and the committed results stand. `union_hybrid.py` uses a stable sort.
