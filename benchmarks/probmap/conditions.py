"""
Two side conditions of the counting (sum-product) rewrite, on small
in-memory programs, compared with NeuroLang's answers:

1. a non-uniform choice: the sum is weighted by the choice's
   probabilities, not a count;
2. a variable occurring in deterministic atoms only (dorsal(s) :-
   label(l), act(l, s), sel(s)): it is an existential (Boolean OR),
   to project out (deduplicate) before summing on the choice; a plain
   sum over it counts a study once per label.

    python conditions.py
"""
import numpy as np
import pandas as pd

from neurolang.frontend import NeurolangPDL

from probtrace import SOLVER_LOG, instrument

rng = np.random.default_rng(0)
n_studies, n_labels = 40, 6
w = rng.exponential(size=n_studies)
w /= w.sum()
act = [(l, s) for l in range(n_labels) for s in range(n_studies)
       if rng.random() < 0.3]
term = [(s, "t") for s in range(n_studies) if rng.random() < 0.5]

nl = NeurolangPDL()
nl.add_probabilistic_choice_from_tuples(
    [(float(p), s) for s, p in enumerate(w)], name="sel")
nl.add_tuple_set(act, name="act")
nl.add_tuple_set(term, name="term")
nl.add_tuple_set([(0,), (1,), (2,)], name="label")
instrument(nl)

A = np.zeros((n_labels, n_studies))
for l, s in act:
    A[l, s] = 1
T = np.zeros(n_studies)
for s, _ in term:
    T[s] = 1

# 1. P[act(l, S) | term(S)] = sum_s w_s A[l,s] T[s] / sum_s w_s T[s]
with nl.scope:
    res = nl.execute_datalog_program(
        "ans(l, p) :- PROB[act(l, s) & sel(s) // (term(s, 't') & sel(s))] = p"
    ).as_pandas_dataframe()
weighted = (A @ (w * T)) / (w @ T)
counted = (A @ T) / T.sum()
res = res.set_index(res.columns[0]).iloc[:, 0].sort_index()
mask = weighted > 0
print("1. non-uniform choice: max |neurolang - weighted| =",
      np.abs(res.to_numpy() - weighted[mask]).max(),
      "| max |neurolang - unweighted count| =",
      np.abs(res.to_numpy() - counted[mask]).max())

# 2. dorsal(s) :- label(l), act(l, s), sel(s): P = sum_s w_s [exists l]
SOLVER_LOG.clear()
with nl.scope:
    res2 = nl.execute_datalog_program("""
dorsal(s) :- label(l), act(l, s), sel(s)
ans(p) :- PROB[dorsal(s)] = p
""").as_pandas_dataframe()
hit = A[:3].max(axis=0)             # exists l in label: act(l, s)
times = A[:3].sum(axis=0)           # number of labels: sum over l
print("2. deterministic-only variable: neurolang", float(res2.iloc[0, 0]),
      "| exists-then-sum", float(w @ hit),
      "| plain sum over l", float(w @ times))
print("   solvers:", [(e["solver"], (e["raised"] or "")[:60]) for e in SOLVER_LOG])
