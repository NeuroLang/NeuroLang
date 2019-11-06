import itertools

from neurolang.expressions import ExpressionBlock, Constant, Symbol
from neurolang.datalog.expressions import Implication, Conjunction, Fact
from neurolang.probabilistic.expressions import ProbabilisticPredicate
from neurolang.probabilistic.probdatalog_gm import marg_query, succ_query

s1 = Constant[str]("s1")
s2 = Constant[str]("s2")
v1 = Constant[str]("v1")
v2 = Constant[str]("v2")
v3 = Constant[str]("v3")
t1 = Constant[str]("t1")
t2 = Constant[str]("t2")

neurosynth_entries = [
    (s1, v1, t1),
    (s1, v1, t2),
    (s1, v2, t2),
    (s1, v3, t1),
    (s2, v2, t1),
    (s2, v2, t2),
    (s2, v3, t1),
]

voxel_probs = {
    voxel: Constant[float](
        len(set((v, s) for s, v, _ in neurosynth_entries if v == voxel))
        / len(set(s for s, _, _ in neurosynth_entries))
    )
    for voxel in (v1, v2, v3)
}
term_probs = {
    term: Constant[float](
        len(set((t, s) for s, _, t in neurosynth_entries if t == term))
        / len(set(s for s, _, _ in neurosynth_entries))
    )
    for term in (t1, t2)
}

joint_probs = {
    (voxel, term): Constant[float](
        len(
            set(
                (s, v, t)
                for s, v, t in neurosynth_entries
                if v == voxel and t == term
            )
        )
        / len(set(s for s, _, _ in neurosynth_entries))
    )
    for voxel, term in itertools.product((v1, v2, v3), (t1, t2))
}
selected_voxels = set(voxel_probs)
selected_terms = set(term_probs)

Voxel = Symbol("Voxel")
Term = Symbol("Term")
Activation = Symbol("Activation")
VoxelActivated = Symbol("VoxelActivated")
TermObserved = Symbol("TermObserved")
Target = Symbol("Target")
v = Symbol("v")
t = Symbol("t")

facts = [Fact(Voxel(voxel)) for voxel in selected_voxels] + [
    Fact(Term(term)) for term in selected_terms
]

activation_probfacts = [
    Implication(
        ProbabilisticPredicate(
            joint_probs[(voxel, term)], Activation(voxel, term)
        ),
        Constant[bool](True),
    )
    for voxel, term in itertools.product(selected_voxels, selected_terms)
]
voxel_activation_probfacts = [
    Implication(
        ProbabilisticPredicate(voxel_probs[voxel], VoxelActivated(voxel)),
        Constant[bool](True),
    )
    for voxel in selected_voxels
]
term_observed_probfacts = [
    Implication(
        ProbabilisticPredicate(term_probs[term], TermObserved(term)),
        Constant[bool](True),
    )
    for term in selected_terms
]

rules = [
    Implication(
        Symbol("Target")(v, t),
        Conjunction([VoxelActivated(v), TermObserved(t), Voxel(v), Term(t)]),
    )
]

code = ExpressionBlock(
    facts
    # + activation_probfacts
    + voxel_activation_probfacts
    + term_observed_probfacts
    + rules
)

for pair, prob in joint_probs.items():
    print(pair, prob)

result = succ_query(code, Target(v, t))
print(result)

# print("Activation(v, t)")
# result = succ_query(code, Activation(v, t))
# print(result)
# print("TermObserved(t)")
# result = succ_query(code, TermObserved(t))
# print(result)
