"""
Viterbi Decoding with Max-Product Semiring
===========================================

This example shows how to perform Viterbi-style decoding — finding the most
likely sequence of hidden states in an HMM — using NeuroLang's
:class:`neurolang.probabilistic.MaxProductSemiring`.

The core idea: a **semiring** generalises the operations used in probabilistic
relational algebra. The standard **probability semiring** uses (+, ×) and
computes marginal probabilities over existential variables as
:math:`1 - \prod(1 - p_i)`. The **max-product semiring** replaces addition
with :math:`\max`, so existential quantification picks the path with the
highest probability — exactly what Viterbi decoding needs.

We'll walk through a 2-state, 2-step HMM example step by step.
"""

# %%
import math

from neurolang.expressions import Constant, Symbol
from neurolang.logic import Conjunction, Implication, Union
from neurolang.probabilistic import MaxProductSemiring, dalvi_suciu_lift
from neurolang.probabilistic.cplogic.program import CPLogicProgram
from neurolang.probabilistic.viterbi import (
    compute_backpointers,
    decode_viterbi,
    trace_path,
)

# %%
# Step 1: Set up the HMM
# -----------------------
#
# We have two hidden states (``s1``, ``s2``) and two time steps. The model
# is defined by four probability distributions:
#
# - **Initial probabilities** ``init(s)`` — :math:`P(s_1)`
# - **Emission probabilities** ``emit_t(s)`` — :math:`P(o_t | s)`
# - **Transition probabilities** ``trans(s', s)`` — :math:`P(s_t | s_{t-1})`

cpl = CPLogicProgram()

init = Symbol("init")
trans = Symbol("trans")
emit1 = Symbol("emit1")
emit2 = Symbol("emit2")

cpl.add_probabilistic_facts_from_tuples(
    init, {(0.6, "s1"), (0.4, "s2")}
)
cpl.add_probabilistic_facts_from_tuples(
    emit1, {(0.7, "s1"), (0.2, "s2")}
)
cpl.add_probabilistic_facts_from_tuples(
    emit2, {(0.1, "s1"), (0.8, "s2")}
)
cpl.add_probabilistic_facts_from_tuples(
    trans, {
        (0.8, "s1", "s1"),
        (0.2, "s1", "s2"),
        (0.3, "s2", "s1"),
        (0.7, "s2", "s2"),
    }
)

# %%
# Step 2: Define the inference rules
# -----------------------------------
#
# The Viterbi forward pass is expressed as two CP-Logic rules:
#
# **Step 1** — initial belief:
#    :math:`\alpha_1(s) \;:=\; \texttt{init}(s) \wedge \texttt{emit1}(s)`
#
# **Step 2** — recursive update:
#    :math:`\alpha_2(z) \;:=\; \texttt{viterbi\_1}(y) \wedge \texttt{trans}(y, z) \wedge \texttt{emit2}(z)`
#
# The existential quantifier over :math:`y` in Step 2 is where the
# max-product semiring makes the difference — it uses **max** instead of
# the probability semiring's **1 - ∏(1-p)**.

v1 = Symbol("viterbi_1")
ans2_sym = Symbol("ans_2")
x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

# Step 1: v1(s) :- init(s), emit1(s)
cpl.walk(Union((
    Implication(v1(x), Conjunction((init(x), emit1(x)))),
)))

# Step 2: ans2(z) :- v1(y), trans(y, z), emit2(z)
cpl.walk(Union((
    Implication(ans2_sym(z), Conjunction((
        v1(y), trans(y, z), emit2(z)
    ))),
)))

# %%
# Step 3: Solve — two approaches
# -------------------------------
#
# **Approach A:** Solve each step individually, then extract backpointers
# using :func:`~neurolang.probabilistic.viterbi.compute_backpointers`.
#
# **Approach B:** Use the convenience :func:`~neurolang.probabilistic.viterbi.decode_viterbi`
# helper which does it all in one call.
#
# Let's do both to show how they relate.

# --- Approach A: Manual step-by-step ---

ans = Symbol("ans")

query1 = Implication(ans(x), v1(x))
result1 = dalvi_suciu_lift.solve_succ_query(
    query1, cpl, semiring=MaxProductSemiring()
)

query2 = Implication(ans(z), ans2_sym(z))
result2 = dalvi_suciu_lift.solve_succ_query(
    query2, cpl, semiring=MaxProductSemiring()
)

print("Step 1 — initial beliefs:")
for row in result1.relation.value:
    print(f"  α₁({row[1]}) = {float(row[0]):.4f}")

print("\nStep 2 — max-product beliefs:")
for row in result2.relation.value:
    print(f"  α₂({row[1]}) = {float(row[0]):.4f}")

# Backpointer recovery: which previous state gave the max?
bp_result = compute_backpointers(
    cpl, prev_symbol=v1, transition_symbol=trans
)

print("\nBackpointers (which previous state contributed the max):")
bp_map = {}
for row in bp_result.relation.value:
    prob = float(row[0])
    state = str(row[1])
    prev_state = str(row[2]) if len(row) > 2 else None
    if state not in bp_map or prob > bp_map[state][0]:
        bp_map[state] = (prob, prev_state)
    print(f"  pair ({state} ← {prev_state}): product = {prob:.4f}")

print("\nBest predecessor for each state:")
for state, (score, prev) in sorted(bp_map.items()):
    print(f"  {state} ← {prev}  (score = {score:.4f})")

# %%
# **Approach B:** Using ``decode_viterbi``
# ----------------------------------------
#
# The helper function runs the same computation but packages the
# results and backpointers together.

# We need to re-create the program since it has accumulated intermediate
# rules from the manual steps above.
cpl2 = CPLogicProgram()
cpl2.add_probabilistic_facts_from_tuples(init, {(0.6, "s1"), (0.4, "s2")})
cpl2.add_probabilistic_facts_from_tuples(emit1, {(0.7, "s1"), (0.2, "s2")})
cpl2.add_probabilistic_facts_from_tuples(emit2, {(0.1, "s1"), (0.8, "s2")})
cpl2.add_probabilistic_facts_from_tuples(trans, {
    (0.8, "s1", "s1"), (0.2, "s1", "s2"),
    (0.3, "s2", "s1"), (0.7, "s2", "s2"),
})

results, traces, backpointers = decode_viterbi(
    cpl2,
    variables={"step": [1, 2]},
    rules={
        1: Implication(v1(x), Conjunction((init(x), emit1(x)))),
        2: Implication(ans2_sym(z), Conjunction((
            v1(y), trans(y, z), emit2(z)
        ))),
    },
    query_symbols={1: v1, 2: ans2_sym},
    transition_symbols={
        2: {"prev": v1, "transition": trans},
    },
)

print("\nApproach B — decode_viterbi results:")
for step in sorted(results.keys()):
    print(f"  Step {step}:")
    for row in results[step].relation.value:
        print(f"    score = {float(row[0]):.4f}  state = {row[1]}")

# %%
# Step 4: Reconstruct the most likely path
# -----------------------------------------
#
# Using the backpointers from Approach B, we can recover the full
# state sequence. The final state with the highest probability
# is ``s2`` (0.0672).

final_state = "s2"
path = trace_path(backpointers, final_state, [1, 2])
print(f"Most likely state sequence: {path}")

# %%
# Step 5: Verification
# --------------------
#
# Let's verify the computation manually:
#
# **Step 1:**
#   :math:`\alpha_1(s_1) = 0.6 \times 0.7 = 0.42`
#   :math:`\alpha_1(s_2) = 0.4 \times 0.2 = 0.08`
#
# **Step 2 for s₁:**
#   inner product over y: :math:`\max(0.42 \times 0.8,\, 0.08 \times 0.3)`
#   = :math:`\max(0.336, 0.024) = 0.336` → max came from :math:`s_1`
#   :math:`\alpha_2(s_1) = 0.336 \times 0.1 = 0.0336`
#
# **Step 2 for s₂:**
#   inner product over y: :math:`\max(0.42 \times 0.2,\, 0.08 \times 0.7)`
#   = :math:`\max(0.084, 0.056) = 0.084` → max came from :math:`s_1`
#   :math:`\alpha_2(s_2) = 0.084 \times 0.8 = 0.0672`
#
# The backpointer query ``bp(z, y) :- v1(y), trans(y, z)`` keeps
# both the result state and the existential variable in the output.
# The MaxProductSemiring's **max** aggregation over y picks the
# argmax automatically.

print(f"\nα₁(s1) = 0.6 × 0.7 = {0.6*0.7:.4f}")
print(f"α₁(s2) = 0.4 × 0.2 = {0.4*0.2:.4f}")
print(f"α₂(s1) = max({0.42*0.8:.4f}, {0.08*0.3:.4f}) × 0.1 = {max(0.42*0.8, 0.08*0.3)*0.1:.4f}")
print(f"α₂(s2) = max({0.42*0.2:.4f}, {0.08*0.7:.4f}) × 0.8 = {max(0.42*0.2, 0.08*0.7)*0.8:.4f}")
print(f"\nMost likely path: {path}")
print("✅  Viterbi decoding complete using MaxProductSemiring")

# %%
# What's happening under the hood
# --------------------------------
#
# The key mechanism is in
# :class:`~neurolang.relational_algebra_provenance.IndependentDisjointProjectionsAndUnionMixin`.
# When processing ``independent_projection`` (the existential quantifier),
# the solver checks which semiring is active:
#
# - **ProbabilitySemiring** → ``GROUP BY state, AGG=sum``
# - **MaxProductSemiring** → ``GROUP BY state, AGG=max``
#
# The backpointer query ``bp(z, y) :- v1(y), trans(y, z)`` keeps both
# the state and the existential in the output. Even though
# MaxProductSemiring is used, the query has no existential quantifier
# (both :math:`z` and :math:`y` appear in the head), so the solver
# computes the raw product and we select the argmax manually as a
# post-processing step. This gives us complete visibility into which
# previous state contributed the maximum for each result.
