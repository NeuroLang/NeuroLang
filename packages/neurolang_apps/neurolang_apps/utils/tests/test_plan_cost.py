"""
The cost of the plans of representative queries, kept within budgets.

The engine ``neurosynth_disconnectome7t`` is built on synthetic data (300
studies, the toy atlas and tractogram of the engine tracts' tests), and the
queries of its bank's kinds run on it: forward inference, decoding by Bayes
factor, co-activation, a ``similar_study`` bound by a rule, negation with
aggregation, resampling and tract_querier's bundles. Each query's cost is
measured in numbers that depend neither on the machine nor on its load:

- ``max_rows``: the rows of the largest relation the solvers computed,
  intermediate ones included: each result of the relational algebra
  solvers goes through ``RelationalAlgebraSolver._build_relation_constant``,
  each tensor of the sums of products through ``ScipyTensorSolver.walk``;
- ``magic_rows``: the rows of the magic predicates of the chase's solution;
- ``similarity_calls``: the calls to ``query_similarity``, a counting stub.

A query fails when one of them is more than ``factor`` times its baseline
(plus ``slack`` rows) in ``plan_cost.yaml``. When a cost grows on purpose,
the baselines are rewritten by running this module with
``NEUROLANG_UPDATE_PLAN_COST=1``, and the change of the YAML file is
reviewed with the pull request (doc/contributing.rst).
"""

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

from neurolang.atlases import label_name
from neurolang.frontend.neurosynth_utils import StudyID
from neurolang.frontend.show_ra import relation_rows
from neurolang_apps.utils import engine_registry
from nl_core.relational_algebra.linear_algebra.scipy_backend import (
    ScipyTensorSolver,
)
from nl_core.relational_algebra.relational_algebra import (
    RelationalAlgebraSolver,
)

from .test_engines import _COGNITIVE_ATLAS_TASKS, _fake_relations
from .test_tracts_engine import FAKE_LABELS

BASELINES = Path(__file__).with_name("plan_cost.yaml")
UPDATE = os.environ.get("NEUROLANG_UPDATE_PLAN_COST") == "1"
NS_INIT = "neurolang_apps.utils.engines.neurosynth.init"
N_STUDIES = 300
TERMS = ["memory", "language", "pain", "fear", "attention", "emotion",
         "reward", "vision"]
REGION = label_name(FAKE_LABELS["destrieux"][1])


class OverBudget(AssertionError):
    """A cost of a query is above its budget."""


# The fix of the bundles' magic sets, broken by #1391 (f5b4820b)
TRACT_FIX = pytest.mark.xfail(
    raises=OverBudget, strict=True,
    reason="#1391 solves the bundles without magic sets: "
    "fix/magic-sets-tract-regression",
)


def synthetic_studies(rng):
    """Peaks, terms and metadata of ``N_STUDIES`` studies."""
    ids = np.arange(1, N_STUDIES + 1)
    n_peaks = rng.integers(1, 5, size=N_STUDIES)
    study = np.repeat(ids, n_peaks)
    xyz = rng.uniform((-58, -98, -38), (58, 58, 48), size=(len(study), 3))
    peaks = pd.DataFrame({
        "id": study, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "space": "MNI",
    })
    mentioned = [
        (StudyID(s), t, round(float(w), 3))
        for s in ids
        for t, w in zip(
            rng.choice(TERMS, size=2, replace=False),
            rng.uniform(.05, .5, size=2),
        )
    ]
    terms = pd.DataFrame(mentioned, columns=["id", "term", "tfidf"])
    metadata = pd.DataFrame({
        "id": ids, "title": [f"Study {s}" for s in ids],
        "year": 2000 + ids % 20, "journal": "NeuroImage",
        "doi": [f"10.1/{s}" for s in ids],
    })
    return peaks, terms, metadata


def atlas_volume(uncinate_files):
    """Four regions, the quadrants of the toy atlas's space: left and
    right, anterior and posterior."""
    x, y = np.meshgrid(
        *(np.arange(n) for n in uncinate_files.SHAPE[:2]), indexing="ij")
    volume = 1 + (x > uncinate_files.SHAPE[0] // 2) + 2 * (
        y < uncinate_files.SHAPE[1] // 2)
    volume = np.repeat(volume[..., None], uncinate_files.SHAPE[2], axis=2)
    return volume.astype(np.int16)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """``neurosynth_disconnectome7t`` on synthetic studies, and on the
    toy atlas and tractogram of the engine tracts' tests, in place of the
    files it downloads; a row of each connectivity relation."""
    pytest.importorskip("neurolang_apps.utils.engines.tracts")
    from neurolang_apps.utils.engines.tracts.tests import uncinate_files

    tmp = tmp_path_factory.mktemp("plan_cost")
    uncinate_files.write_files(tmp, n_random=300, seed=3)
    data = tmp / "data"
    tracts = data / "tracts" / "disconnectome7t"
    tracts.mkdir(parents=True)
    (data / "tracts" / "fs").mkdir()
    nib.streamlines.save(
        nib.streamlines.load(str(tmp / "tracts.trk")).tractogram,
        str(tracts / "disconnectome7t.tck"))
    shutil.copy(tmp / "labels.nii.gz",
                tracts / "disconnectome7t_wmparc.nii.gz")
    shutil.copy(tmp / "lut.txt",
                data / "tracts" / "fs" / "FreeSurferColorLUT.txt")
    atlas = tmp / "quadrants.nii.gz"
    nib.save(nib.Nifti1Image(atlas_volume(uncinate_files),
                             uncinate_files.AFFINE), atlas)
    peaks, terms, metadata = synthetic_studies(np.random.default_rng(0))
    mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4))
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name, labels in FAKE_LABELS.items():
            monkeypatch.setitem(
                engine_registry._ATLAS_REGISTRY[name], "fetch",
                lambda labels=labels, **kwargs: {
                    "maps": str(atlas), "labels": list(labels)},
            )
        monkeypatch.chdir(data)
        with patch(f"{NS_INIT}.fetch_neurosynth_peak_data",
                   return_value=peaks), \
                patch(f"{NS_INIT}.get_ns_term_study_associations",
                      return_value=terms), \
                patch(f"{NS_INIT}.fetch_cortical_atlas",
                      return_value=nib.Nifti1Image(
                          np.zeros((4, 4, 4), dtype=np.int16), np.eye(4))), \
                patch(f"{NS_INIT}.fetch_cognitive_atlas_tasks",
                      return_value=_COGNITIVE_ATLAS_TASKS), \
                patch(f"{NS_INIT}.fetch_study_metadata",
                      return_value=metadata), \
                patch.object(engine_registry, "_resolve_mask",
                             return_value=mask), \
                patch.object(engine_registry, "_phase_relations",
                             _fake_relations):
            nl = engine_registry.build_engine(
                "neurosynth_disconnectome7t", data)
        # 60 tasks with a definition each, 7 of them of the kind "ri"
        nl.add_tuple_set(
            [(f"t{i}", "ri" if i % 9 == 0 else "other") for i in range(60)],
            name="task_kind")
        nl.add_tuple_set(
            [(f"t{i}", f"definition of task {i}") for i in range(60)],
            name="task_definition")
        yield nl


def similarity(calls):
    """A deterministic ``query_similarity`` counting its calls."""
    def query_similarity(q: str, s: int) -> float:
        calls.append((q, s))
        return 1. if (len(q) + int(s)) % 7 == 0 else 0.
    return query_similarity


@contextmanager
def measured(engine):
    """The costs of the queries run within the block."""
    cost = {"max_rows": 0, "magic_rows": 0, "similarity_calls": 0}
    calls = []
    build = RelationalAlgebraSolver._build_relation_constant
    tensor = ScipyTensorSolver.walk

    def build_and_count(self, relation, *args, **kwargs):
        try:
            rows = int(relation_rows(relation))
        except TypeError:  # not a set of rows
            rows = 0
        cost["max_rows"] = max(cost["max_rows"], rows)
        return build(self, relation, *args, **kwargs)

    def tensor_and_count(self, expression):
        # the entries of a tensor of the sums of products are its rows
        result = tensor(self, expression)
        cost["max_rows"] = max(cost["max_rows"], len(result.values))
        return result

    class MagicRows(engine.chase_class):
        def build_chase_solution(self):
            solution = super().build_chase_solution()
            cost["magic_rows"] += sum(
                len(relation.value) for symbol, relation in solution.items()
                if symbol.name.startswith("magic_")
            )
            return solution

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            RelationalAlgebraSolver, "_build_relation_constant",
            build_and_count)
        monkeypatch.setattr(ScipyTensorSolver, "walk", tensor_and_count)
        monkeypatch.setattr(engine, "chase_class", MagicRows)
        with engine.scope:
            engine.add_symbol(similarity(calls), name="query_similarity")
            engine.add_symbol(lambda q: .5, name="query_similarity_cutoff")
            yield cost
    cost["similarity_calls"] = len(calls)


ACTIVE = (
    "active(l, s) :- destrieux_activated(l, s), selected_study(s)\n"
)
CASES = {
    "forward_inference": ACTIVE + (
        "about(s) :- mentions(s, 'memory'), selected_study(s)\n"
        "ans(l, p) :- PROB[active(l, s) // about(s)] = p"
    ),
    "bayes_factor": (
        "active_region(l) :- destrieux_activated(l, s), selected_study(s)\n"
        "mentioned_term(t) :- mentions(s, t), selected_study(s)\n"
        "cooccurrence(l, t) :- destrieux_activated(l, s), mentions(s, t), "
        "selected_study(s)\n"
        "region_probability(l, p_r) :- PROB[active_region(l)] = p_r\n"
        "term_probability(t, p_t) :- PROB[mentioned_term(t)] = p_t\n"
        "joint_probability(l, t, p_rt) :- PROB[cooccurrence(l, t)] = p_rt\n"
        "bayes_factor(l, t, bf) :- joint_probability(l, t, p_rt), "
        "region_probability(l, p_r), term_probability(t, p_t), "
        "(bf == (p_rt / p_r) / ((p_t - p_rt) / (1.0 - p_r)))\n"
        f"ans(t, bf) :- bayes_factor('{REGION}', t, bf)"
    ),
    "coactivation": ACTIVE + (
        f"seed(s) :- active('{REGION}', s)\n"
        "ans(l, p) :- PROB[active(l, s) // seed(s)] = p"
    ),
    # #1379: the text of similar_study bound by every definition, not by
    # those of the tasks of the rule restricting them (fixed by #1386)
    "similar_study_bound": (
        "ri_task(t) :- task_kind(t, 'ri')\n"
        "uses(s) :- ri_task(t), task_definition(t, d), "
        "similar_study(d, s)\n"
        "ans(count(s)) :- uses(s)"
    ),
    "negation_aggregation": (
        "ans(t, count(s)) :- mentions(s, t), ~mentions(s, 'pain')"
    ),
    "resampling": ACTIVE + (
        "about(s) :- mentions(s, 'memory'), selected_study(s)\n"
        "p_active(l, PROB) :- active(l, s) // about(s)\n"
        "ans(b, l, p) :- RESAMPLE[boot_studies](p_active(l, p)) = b"
    ),
}
BUNDLES = ["uf", "ilf", "af", "ec", "slf_i", "cst"]
CASES.update(
    {bundle: f'ans(x) :- {bundle}(x, "left")' for bundle in BUNDLES})
MARKS = {bundle: TRACT_FIX for bundle in ["uf", "ilf", "af", "ec"]}


def read_budgets():
    return yaml.safe_load(BASELINES.read_text())


def write_baseline(name, cost):
    """Rewrite the baselines of ``name``, keeping the file's comments."""
    text = BASELINES.read_text()
    header = text[:text.index("cases:")]
    budgets = read_budgets()
    budgets["cases"][name] = dict(sorted(cost.items()))
    BASELINES.write_text(header + yaml.safe_dump(
        {"cases": dict(sorted(budgets["cases"].items()))},
        sort_keys=False))


@pytest.mark.parametrize("name", [
    pytest.param(name, marks=MARKS.get(name, ())) for name in CASES
])
def test_the_plan_cost_is_within_its_budget(engine, name, request):
    with measured(engine) as cost:
        result = engine.execute_datalog_program(CASES[name])
    print(f"{name}: {cost}, {len(result)} rows")
    if UPDATE:
        if request.node.get_closest_marker("xfail"):
            pytest.skip("the baseline of a known regression, set by hand")
        write_baseline(name, cost)
        return
    budgets = read_budgets()
    baseline = budgets["cases"].get(name)
    assert baseline is not None, (
        f"{name} has no baseline: run with NEUROLANG_UPDATE_PLAN_COST=1")
    over = [
        f"{measure}: {cost[measure]} > {budgets['factor']} x "
        f"{baseline[measure]} + {budgets['slack']} "
        f"({cost[measure] / max(baseline[measure], 1):.1f} times)"
        for measure in sorted(baseline)
        if cost[measure]
        > budgets["factor"] * baseline[measure] + budgets["slack"]
    ]
    if over:
        raise OverBudget(
            f"the plan of {name} costs more than its budget, "
            f"{BASELINES.name}: " + "; ".join(over))
