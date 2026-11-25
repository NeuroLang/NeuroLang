# Does Sonnet's edge on real patients come from memory of the case reports? (written before the run, 2026-11-25)

Context: answering directly, Sonnet ranks the true disease of real patients far better than the exact posterior (union run: MRR 0.427 against
0.290). Its edge is where the HPO annotation covers little of the patient's terms (`post_hoc_coverage.py`). Every real patient comes from a
published case report, which Sonnet may have read. Two explanations remain open:
- clinical knowledge beyond the annotation;
- memory of the paper.

**Planned test that cannot be run.** Sonnet 5.5's training data cutoff is June 2026 (Anthropic's models overview). phenopacket-store 0.1.27 (June
2026) has no paper published after it. Even its main branch (2026-11-25) has only 23 usable cases from 7 such papers, too few clusters for a test.

## A. Synthetic patients (no paper exists)
- 127 patients, one per test disease with at least 6 annotated terms (seed 21; `memo_probe.py plan`, `results/memo_plan.json`).
- Each patient has the median number of terms of the real union patients (7). Terms are drawn from the disease's own annotation with their
  frequencies, as `hpo_bench.simulate` does.
- The posterior's model generates these patients exactly, so this is the posterior's best case. Sonnet answers with the H0 prompt.
- Compared: Sonnet minus posterior MRR on these patients, against the same gap on the union run's 184 real patients. The intervals resample
  diseases; the difference between the two gaps uses an independent bootstrap of each set.
- Reading:
  - The real gap clearly larger than the synthetic gap: Sonnet's real-case edge comes from information outside the annotation (knowledge or memory;
    part B separates the two).
  - Sonnet ahead on synthetic patients too: it reasons over annotation-like evidence better than the naive-Bayes posterior.
- Control: how often Sonnet claims a source paper for a synthetic patient, where none exists.

## B. Recognition probe (the union run's 184 real patients, with the same terms as there)
- Sonnet is asked which publication describes the patient (PMID, first author, year, title), with "unknown" allowed.
- **Recognised** = the right PMID, or the source paper's first author with the year within one (from PubMed).
- Reported: the share recognised (and the share claiming any source), Sonnet's and the posterior's MRR on recognised and unrecognised patients, and
  Sonnet minus posterior on the unrecognised ones (intervals resample diseases).
- Reading:
  - Sonnet keeps its edge on unrecognised patients: memory of the paper is not needed for the edge.
  - The edge is concentrated on recognised patients: memory explains part of it.
  - Not recalling a paper does not prove it was not seen (memory can help without explicit recall), so B can show memory but cannot rule it out.

## C. Post-cutoff cases (descriptive)
The 23 cases from 7 papers published after June 2026 (main branch). Sonnet minus posterior MRR, with intervals resampling papers. Descriptive
only: 7 papers.

## Setup
- Sonnet 5.5 through OpenRouter (`HPO_LLM_ROUTE=openrouter`), as in the union run, so reasoning is on, as there.
- `max_tokens` 1,200, one answer per prompt, answers cached by prompt.
- The union run's H0 answers and posterior ranks are reused, not rerun.
