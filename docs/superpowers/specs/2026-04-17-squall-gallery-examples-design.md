# SQUALL Gallery Examples — Design Spec

**Date:** 2026-04-17
**Branch:** squall-cnl
**Scope:** Two new Sphinx Gallery example files that mirror existing examples using SQUALL controlled English, plus one targeted transformer fix to support the needed syntax.

---

## Goal

Add two `examples/plot_squall_*.py` Sphinx Gallery files that reproduce the logic of the two NeuroSynth gallery examples using `nl.execute_squall_program(...)` with maximally natural English sentences. No IR builder DSL (`with nl.scope as e:`) is used for the logic rules — only SQUALL.

---

## Files Changed

| File | Change |
|------|--------|
| `examples/plot_squall_neurosynth.py` | New — SQUALL version of `plot_neurosynth_implementation.py` |
| `examples/plot_squall_cbma_spatial_prior.py` | New — SQUALL version of `plot_cbma_spatial_prior.py` |
| `neurolang/frontend/datalog/squall_syntax_lark.py` | Fix `rel_fun_call` tuple-subject semantics for the CBMA proximity rule |
| `neurolang/frontend/datalog/tests/test_squall_parser.py` | New parser test for the tuple-subject rel_fun_call fix |

---

## Part 1 — Transformer Fix: `rel_fun_call` with tuple subject

### Problem

The CBMA example needs to express:

> for each voxel `(?i1;?j1;?k1)` and each focus `(?i2;?j2;?k2;?s)`, if `is_near(i1,j1,k1,i2,j2,k2)` holds …

The current `rel_fun_call` always prepends the outer noun's subject as the **first argument**:

```python
# N-ary path (current):
def rel(x, _vars=label_vars):
    return func_sym(x, *_vars)
```

When the outer noun is a **tuple** subject (e.g. `Focus_reported (?i2;?j2;?k2;?s)`), `x` is the whole 4-tuple `(i2, j2, k2, s)` — not a single variable — so `func_sym(x, *_vars)` passes a tuple as the first argument, which doesn't match the arity of `is_near(i1,j1,k1,i2,j2,k2)`.

### Fix

When all argument positions are provided explicitly via labels (i.e. the labels cover every parameter), **do not prepend** the subject `x`. Only prepend when fewer labels are given than the function needs.

The heuristic: if the label count equals the expected arity (i.e. no "missing" subject slot), skip prepend. In practice we detect this by checking: if the subject `x` is a **tuple** (which happens when `_var_info` is a tuple from `ng1` with a tuple `app`), the labels are meant to cover all positions and `x` should **not** be prepended.

```python
def rel_fun_call(self, args):
    """Handle ``identifier(label, label, ...)`` as a body predicate atom.

    When the outer noun has a scalar subject (single variable), the subject
    is prepended as the first argument: ``func(subject, *labels)``.

    When the outer noun has a tuple subject (multiple variables from a tuple
    app like ``(?i;?j;?k)``), the labels cover all argument positions
    explicitly and the subject is NOT prepended.
    """
    func_sym = args[0]
    label_cps_list = args[1:]

    label_vars = []
    for lbl_cps in label_cps_list:
        if callable(lbl_cps) and not isinstance(lbl_cps, (Symbol, Constant)):
            label_vars.append(lbl_cps(lambda v: v))
        elif isinstance(lbl_cps, Symbol):
            label_vars.append(lbl_cps)
        else:
            label_vars.append(Symbol.fresh())

    if len(label_vars) == 1:
        # Binary predicate: subject + one explicit arg (unchanged)
        y = label_vars[0]
        return ('_rel', lambda x: func_sym(x, y))
    else:
        # N-ary: prepend subject only when it is a scalar (single Symbol).
        # When the subject is a tuple (multi-var noun like (?i;?j;?k)),
        # the labels cover all positions and the subject is not prepended.
        def rel(x, _vars=label_vars):
            if isinstance(x, tuple):
                return func_sym(*_vars)   # labels cover all positions
            return func_sym(x, *_vars)   # scalar subject prepended
        return ('_rel', rel)
```

### New parser test

```python
def test_rel_fun_call_tuple_subject_no_prepend():
    """rel_fun_call with a tuple-subject noun does NOT prepend the tuple as first arg.

    'every Focus_reported (?i2;?j2;?k2;?s) that is_near(?i1,?j1,?k1,?i2,?j2,?k2) holds'
    should emit is_near(i1,j1,k1,i2,j2,k2) — 6 args, not 7.
    """
    from neurolang.expressions import Symbol, FunctionApplication

    tree = parser().parse(
        "define as Near every Focus_reported (?i2; ?j2; ?k2; ?s) "
        "that is_near(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) holds."
    )
    result = SquallTransformer().transform(tree)
    rules = result if isinstance(result, list) else [result]
    implications = [r for r in rules if isinstance(r, Implication)]
    assert len(implications) == 1
    body_str = str(implications[0].antecedent)
    assert "is_near" in body_str.lower()
    # The function call should have 6 args (not 7 with a prepended tuple)
    from neurolang.logic.expression_processing import extract_logic_free_variables
    free_vars = extract_logic_free_variables(implications[0].antecedent)
    var_names = {v.name for v in free_vars}
    assert "i1" in var_names and "i2" in var_names, \
        f"Expected i1 and i2 in free vars, got {var_names}"
```

---

## Part 2 — `plot_squall_neurosynth.py`

Mirrors `plot_neurosynth_implementation.py`. All data loading is identical Python. The logic section replaces the `with nl.scope as e:` block with a single `nl.execute_squall_program(...)` call.

### SQUALL program

```text
define as Activation every Peak_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

define as Term_association every Term_in_study_tfidf (?s; ?t; _)
    such that ?s is a Selected_study.

define as Activation_given_term with probability
    every Activation (?i; ?j; ?k)
    conditioned to every Term_association ?t that is 'auditory'.

define as Activation_given_term_image
    every Agg_create_region_overlay of the Activation_given_term.
```

### Sentence-by-sentence English explanation

| Sentence | What it means |
|---|---|
| `every Peak_reported (…) such that ?s is a Selected_study` | "A peak is an Activation when it belongs to a selected study." |
| `every Term_in_study_tfidf (…) such that ?s is a Selected_study` | "A term association holds when the study is selected." |
| `with probability … conditioned to … that is 'auditory'` | "Compute P(Activation \| Term = 'auditory')." |
| `every Agg_create_region_overlay of the Activation_given_term` | "Aggregate all (i,j,k,p) tuples into a brain overlay image." |

### Grammar constructs used

- `such that ?s is a Selected_study` → `rel_s(s_np_vp(?s, vpbe_npc(det1_some, ng1(selected_study))))` — produces `selected_study(s)`
- `?t that is 'auditory'` → `np_quantified(det_every, ng1(term_association, rel_vp(vpbe_npc(Constant('auditory')))))` — produces `eq(t, 'auditory')`
- `with probability … conditioned to …` → `rule_op_marg` → `Implication(head(…, ProbabilisticQuery(PROB, vars)), Condition(…))`
- `every Agg_create_region_overlay of the Activation_given_term` → `ng1_agg_npc` with arbitrary functor (Gap B fix) → `AggregationApplication(Symbol('agg_create_region_overlay'), free_vars)`

### Result retrieval

```python
nl.execute_squall_program(squall_program)
solution = nl.solve_all()
result_image = solution["activation_given_term_image"] \
    .as_pandas_dataframe().iloc[0, 0].spatial_image()
```

### Full file skeleton

```python
# -*- coding: utf-8 -*-
r"""
NeuroSynth Query in SQUALL Controlled English
==============================================

Reproduces the NeuroSynth forward model — P(Activation | Term) — using
SQUALL controlled natural language instead of the IR builder DSL.

The four logic rules are expressed as plain English sentences passed to
:func:`~neurolang.frontend.NeurolangPDL.execute_squall_program`.
"""
# %%
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Iterable

import nibabel
import nilearn.datasets
import nilearn.image
import nilearn.plotting
import numpy as np
import pandas as pd
from neurolang import ExplicitVBROverlay, NeurolangPDL
from neurolang.frontend.neurosynth_utils import get_ns_mni_peaks_reported

###############################################################################
# Data preparation
# ----------------
# Identical to plot_neurosynth_implementation.py — load MNI atlas,
# NeuroSynth peaks, study IDs, and term associations.

data_dir = Path.home() / "neurolang_data"
mni_t1 = nibabel.load(
    nilearn.datasets.fetch_icbm152_2009(data_dir=str(data_dir / "icbm"))["t1"]
)
mni_t1_4mm = nilearn.image.resample_img(mni_t1, np.eye(3) * 4)

nl = NeurolangPDL()

@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> ExplicitVBROverlay:
    mni_coords = np.c_[i, j, k]
    return ExplicitVBROverlay(
        mni_coords, mni_t1_4mm.affine, p, image_dim=mni_t1_4mm.shape
    )

peak_data = get_ns_mni_peaks_reported(data_dir)
ijk_positions = np.round(
    nibabel.affines.apply_affine(
        np.linalg.inv(mni_t1_4mm.affine),
        peak_data[["x", "y", "z"]].values.astype(float),
    )
).astype(int)
peak_data["i"] = ijk_positions[:, 0]
peak_data["j"] = ijk_positions[:, 1]
peak_data["k"] = ijk_positions[:, 2]
peak_data = peak_data[["i", "j", "k", "id"]]

nl.add_tuple_set(peak_data, name="peak_reported")
study_ids = nl.load_neurosynth_study_ids(data_dir, "selected_study")
nl.add_uniform_probabilistic_choice_over_set(
    study_ids.value, name="selected_study"
)
nl.load_neurosynth_term_study_associations(
    data_dir, "term_in_study_tfidf", tfidf_threshold=1e-3
)

###############################################################################
# SQUALL program
# --------------
# The four logic rules in plain English.

squall_program = """
define as Activation every Peak_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

define as Term_association every Term_in_study_tfidf (?s; ?t; _)
    such that ?s is a Selected_study.

define as Activation_given_term with probability
    every Activation (?i; ?j; ?k)
    conditioned to every Term_association ?t that is 'auditory'.

define as Activation_given_term_image
    every Agg_create_region_overlay of the Activation_given_term.
"""

nl.execute_squall_program(squall_program)

###############################################################################
# Solve and retrieve result
# -------------------------

solution = nl.solve_all()
result_image = (
    solution["activation_given_term_image"]
    .as_pandas_dataframe().iloc[0, 0]
    .spatial_image()
)

###############################################################################
# Plot
# ----

img = result_image.get_fdata()
nilearn.plotting.plot_stat_map(
    result_image, threshold=np.percentile(img[img > 0], 95)
)
nilearn.plotting.show()
```

---

## Part 3 — `plot_squall_cbma_spatial_prior.py`

Mirrors `plot_cbma_spatial_prior.py`. Adds two Python helper symbols for the spatial decay model, then expresses all logic rules in SQUALL.

### Python helpers to register

```python
@nl.add_symbol
def is_near(i1: int, j1: int, k1: int, i2: int, j2: int, k2: int) -> bool:
    """True when euclidean distance between two voxels is strictly less than 1."""
    return float(np.linalg.norm([i1 - i2, j1 - j2, k1 - k2])) < 1.0

@nl.add_symbol
def max_weighted_proximity(d_values: Iterable) -> float:
    """max(exp(-d/5)) across all focus distances matching this voxel+study."""
    return max(float(np.exp(-d / 5.0)) for d in d_values)
```

### SQUALL program

```text
%% Step 1: for each (voxel, focus, study) triple where the voxel is within
%% distance 1 of the focus, record the proximity.
define as Near_focus for every Voxel (?i1; ?j1; ?k1) ;
    where every Focus_reported (?i2; ?j2; ?k2; ?s)
    that is_near(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) holds.

%% Step 2: for each (voxel, study), compute the max weighted proximity.
define as Voxel_reported every Max_weighted_proximity of the Near_focus
    per ?i1 per ?j1 per ?k1 per ?s.

%% Step 3: a voxel is active in a study if it is reported and the study is selected.
define as Term_association every Term_in_study_tfidf (?s; ?t; _)
    such that ?s is a Selected_study.

define as Activation every Voxel_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

%% Step 4: compute P(Activation | Term = 'emotion').
define as Probmap with probability every Activation (?i; ?j; ?k)
    conditioned to every Term_association ?t that is 'emotion'.

%% Step 5: aggregate into a brain overlay image.
define as Img every Agg_create_region_overlay of the Probmap.
```

### Sentence-by-sentence English explanation

| Sentence | What it means |
|---|---|
| `every Voxel (…) where every Focus_reported (…) that is_near(…) holds` | "A near-focus triple exists when voxel and focus are within distance 1." |
| `every Max_weighted_proximity of the Near_focus per ?i1 per ?j1 per ?k1 per ?s` | "For each voxel and study, aggregate proximity weights with max(exp(-d/5))." |
| `such that ?s is a Selected_study` | "Restrict to selected studies." |
| `with probability … conditioned to … that is 'emotion'` | "Compute P(Activation \| Term = 'emotion')." |
| `every Agg_create_region_overlay of the Probmap` | "Collapse (i,j,k,p) rows into a single brain overlay." |

### Grammar constructs used

- `for every Voxel … ; where every Focus_reported … that … holds` → `rule_opnn` outer structure binding both noun groups; inner `rel_fun_call` with tuple subject (uses the Part 1 fix)
- `every Max_weighted_proximity of the Near_focus per ?i1 per ?j1 per ?k1 per ?s` → `ng1_agg_npc` with arbitrary functor + multi-`dim_npc` per-vars
- All others identical to Part 2

### Full file skeleton

```python
# -*- coding: utf-8 -*-
r"""
CBMA Spatial Prior in SQUALL Controlled English
===============================================

Reproduces the coordinate-based meta-analysis spatial decay prior example
using SQUALL controlled natural language. Each voxel's probability of being
reported by a study is modelled via a distance-decay function
max(exp(−d/5)) over nearby foci, then marginalised over the term 'emotion'.
"""
# %%
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Iterable

import nibabel
import nilearn.datasets
import nilearn.image
import nilearn.plotting
import numpy as np
from neurolang.frontend import ExplicitVBR, ExplicitVBROverlay, NeurolangPDL
from neurolang.frontend.neurosynth_utils import get_ns_mni_peaks_reported

###############################################################################
# Data preparation
# ----------------

data_dir = Path.home() / "neurolang_data"
mni_t1 = nibabel.load(
    nilearn.datasets.fetch_icbm152_2009(data_dir=str(data_dir / "icbm"))["t1"]
)
mni_t1_2mm = nilearn.image.resample_img(mni_t1, np.eye(3) * 2)

nl = NeurolangPDL()

###############################################################################
# Register Python helper symbols
# --------------------------------
# These are called from within SQUALL sentences.

@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> ExplicitVBR:
    voxels = np.c_[i, j, k]
    return ExplicitVBROverlay(
        voxels, mni_t1_2mm.affine, p, image_dim=mni_t1_2mm.shape
    )

@nl.add_symbol
def is_near(i1: int, j1: int, k1: int, i2: int, j2: int, k2: int) -> bool:
    """True when euclidean distance between two voxels is strictly less than 1."""
    return float(np.linalg.norm([i1 - i2, j1 - j2, k1 - k2])) < 1.0

@nl.add_symbol
def max_weighted_proximity(d_values: Iterable) -> float:
    """max(exp(-d/5)) aggregation over all proximity values."""
    return max(float(np.exp(-d / 5.0)) for d in d_values)

###############################################################################
# Load data
# ---------

peak_data = get_ns_mni_peaks_reported(data_dir)
ijk_positions = np.round(
    nibabel.affines.apply_affine(
        np.linalg.inv(mni_t1_2mm.affine),
        peak_data[["x", "y", "z"]].values.astype(float),
    )
).astype(int)
peak_data["i"] = ijk_positions[:, 0]
peak_data["j"] = ijk_positions[:, 1]
peak_data["k"] = ijk_positions[:, 2]
peak_data = peak_data[["i", "j", "k", "id"]]

nl.add_tuple_set(peak_data, name="focus_reported")
study_ids = nl.load_neurosynth_study_ids(data_dir, "study")
nl.add_uniform_probabilistic_choice_over_set(
    study_ids.value, name="selected_study"
)
nl.load_neurosynth_term_study_associations(
    data_dir, "term_in_study_tfidf", tfidf_threshold=1e-3
)
nl.add_tuple_set(
    np.hstack(
        np.meshgrid(
            *(np.arange(0, dim) for dim in mni_t1_2mm.get_fdata().shape)
        )
    ).swapaxes(0, 1).reshape(3, -1).T,
    name="voxel",
)

###############################################################################
# SQUALL program
# --------------

squall_program = """
%% Step 1: voxel-focus proximity triples within distance threshold
define as Near_focus for every Voxel (?i1; ?j1; ?k1) ;
    where every Focus_reported (?i2; ?j2; ?k2; ?s)
    that is_near(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) holds.

%% Step 2: max weighted proximity per (voxel, study)
define as Voxel_reported every Max_weighted_proximity of the Near_focus
    per ?i1 per ?j1 per ?k1 per ?s.

%% Step 3: filter to selected studies
define as Term_association every Term_in_study_tfidf (?s; ?t; _)
    such that ?s is a Selected_study.

define as Activation every Voxel_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

%% Step 4: conditional probability map
define as Probmap with probability every Activation (?i; ?j; ?k)
    conditioned to every Term_association ?t that is 'emotion'.

%% Step 5: build brain overlay image
define as Img every Agg_create_region_overlay of the Probmap.
"""

nl.execute_squall_program(squall_program)

###############################################################################
# Solve and retrieve result
# -------------------------

solution = nl.solve_all()
result_image = (
    solution["img"]
    .as_pandas_dataframe().iloc[0, 0]
    .spatial_image()
)

###############################################################################
# Plot
# ----

img = result_image.get_fdata()
nilearn.plotting.plot_stat_map(
    result_image, threshold=np.percentile(img[img > 0], 95)
)
nilearn.plotting.show()
```

---

## Spec Self-Review

**Placeholder scan:** None found. All code blocks are complete.

**Internal consistency:**
- `rel_fun_call` fix (Part 1) is required by the CBMA example's `is_near` call with tuple subject (Part 3) — dependency is explicit.
- Both gallery files use `nl.execute_squall_program` → `nl.solve_all()` which is confirmed to exist in `query_resolution_datalog.py`.
- SQUALL relation names in the program must be lowercase (SQUALL lowercases `upper_identifier`); EDB tuple-sets are registered with lowercase names accordingly.
- `such that ?s is a Selected_study` uses `rel_s(s_np_vp)` which is confirmed to work (Symbol identity shared between `app_label` and sub-sentence).
- `?t that is 'auditory'` uses `vpbe_npc` → `eq(t, Constant('auditory'))` — confirmed working.
- `with probability … conditioned to …` uses `rule_op_marg` — implemented in the prior gap-fix sprint.
- `every Agg_create_region_overlay of the Activation_given_term` uses arbitrary-functor `ng1_agg_npc` — implemented in Gap B fix.

**Scope check:** Two gallery files + one transformer fix + one test. Single implementation plan scope. ✅

**Ambiguity check:**
- `Near_focus` rule uses `rule_opnn` (`for every … ; where every …`). The `for … ; where …` structure binds both noun groups at the same level — this is the confirmed path for multi-noun body rules.
- `per ?i1 per ?j1 per ?k1 per ?s` — four `dim_npc` items. The `det_every` handler uses only `(captured[0],)` for `agg_args` but the per-vars appear as free variables in the `npc_formula` body, so `TranslateToLogicWithAggregation` infers groupby from free variables. This is the same mechanism used by the existing `test_execute_squall_aggregation` test.
