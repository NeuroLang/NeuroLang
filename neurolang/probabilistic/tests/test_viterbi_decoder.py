"""Tests for the Viterbi decoder (backpointer tracking)."""
import operator

from ...datalog import Fact
from ...expressions import Constant, Symbol
from ...logic import Conjunction, Implication, Union
from .. import dalvi_suciu_lift
from ..cplogic import testing
from ..cplogic.program import CPLogicProgram
from ..semiring import MaxProductSemiring
from ..viterbi import compute_backpointers, decode_viterbi, trace_path

ans = Symbol("ans")
P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
S = Symbol("S")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")


def _make_hmm():
    """Create a 2-state, 2-step HMM for testing."""
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
        trans,
        {
            (0.8, "s1", "s1"),
            (0.2, "s1", "s2"),
            (0.3, "s2", "s1"),
            (0.7, "s2", "s2"),
        },
    )
    return cpl, init, trans, emit1, emit2


# ---------------------------------------------------------------------------
# Backpointer recovery on a small HMM chain
# ---------------------------------------------------------------------------


def test_compute_backpointers_simple():
    """Recover argmax for a one-step transition."""
    cpl, init, trans, emit1, emit2 = _make_hmm()

    v1 = Symbol("viterbi_1")

    # Step 1 rule: v1(s) :- init(s), emit1(s)
    cpl.walk(Union((
        Implication(v1(x), Conjunction((init(x), emit1(x)))),
    )))

    # Solve step 1 so v1 is available as a derived relation
    query1 = Implication(ans(x), v1(x))
    dalvi_suciu_lift.solve_succ_query(
        query1, cpl, semiring=MaxProductSemiring()
    )

    # Compute backpointers: bp(z, y) :- v1(y), trans(y, z)
    bp_result = compute_backpointers(
        cpl, prev_symbol=v1, transition_symbol=trans
    )

    # The result has one row per (next_state, prev_state) pair.
    # Expected products for each (y, z):
    #    v1(s1)*trans(s1,s1) = 0.42 * 0.8 = 0.336
    #    v1(s2)*trans(s2,s1) = 0.08 * 0.3 = 0.024
    #    v1(s1)*trans(s1,s2) = 0.42 * 0.2 = 0.084
    #    v1(s2)*trans(s2,s2) = 0.08 * 0.7 = 0.056
    bp_map = {}
    for row in bp_result.relation.value:
        prob = float(row[0])
        state = str(row[1])
        prev_state = str(row[2]) if len(row) > 2 else None
        if state not in bp_map or prob > bp_map[state][0]:
            bp_map[state] = (prob, prev_state)

    assert bp_map.get("s1") is not None, "Missing backpointer for s1"
    assert bp_map["s1"][1] == "s1", (
        f"Expected s1 backpointer to s1, got {bp_map['s1'][1]}"
    )

    assert bp_map.get("s2") is not None, "Missing backpointer for s2"
    assert bp_map["s2"][1] == "s1", (
        f"Expected s2 backpointer to s1, got {bp_map['s2'][1]}"
    )

    assert round(bp_map["s1"][0], 10) == round(0.42 * 0.8, 10), (
        f"Expected backpointer score 0.336 for s1, got {bp_map['s1'][0]}"
    )


# ---------------------------------------------------------------------------
# End-to-end Viterbi decoding
# ---------------------------------------------------------------------------


def test_decode_viterbi_2step():
    """Full Viterbi decode including backpointer recovery."""
    cpl, init, trans, emit1, emit2 = _make_hmm()

    v1 = Symbol("viterbi_1")
    ans2_sym = Symbol("ans_2")

    results, traces, backpointers = decode_viterbi(
        cpl,
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

    # Step 2 should have results
    assert 2 in results, "Missing results for step 2"
    assert 2 in traces, "Missing trace for step 2"
    step2_results = results[2]
    assert len(step2_results.relation.value) > 0, (
        f"Empty results for step 2: {step2_results.relation.value}"
    )

    # Path reconstruction
    final_state = "s2"  # highest probability
    path = trace_path(backpointers, final_state, [1, 2])
    assert len(path) == 2, f"Expected 2 states in path, got {path}"
    assert path[1] == final_state, f"Expected {final_state} at end, got {path}"


# ---------------------------------------------------------------------------
# Trace path reconstruction
# ---------------------------------------------------------------------------


def test_trace_path():
    """Reconstruct state sequence from backpointer maps."""
    backpointers = {
        2: {"s1": "s1", "s2": "s1"},
    }
    path = trace_path(backpointers, "s1", [1, 2])
    assert path == ["s1", "s1"], f"Expected ['s1', 's1'], got {path}"

    path_s2 = trace_path(backpointers, "s2", [1, 2])
    assert path_s2 == ["s1", "s2"], f"Expected ['s1', 's2'], got {path_s2}"
