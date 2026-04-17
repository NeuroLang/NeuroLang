# SQUALL Gallery Examples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two Sphinx Gallery example files that reproduce the NeuroSynth and CBMA spatial-prior examples using SQUALL controlled-English sentences instead of the IR builder DSL, plus a one-line transformer fix that makes `rel_fun_call` work correctly when the outer noun has a tuple subject.

**Architecture:** Task 1 fixes `rel_fun_call` in `squall_syntax_lark.py` so that when the enclosing noun phrase binds a tuple of variables (e.g. `Focus_reported (?i2;?j2;?k2;?s)`), the explicit labels in `func(l1,l2,…)` are used as-is rather than having the tuple prepended as a spurious first argument. Tasks 2 and 3 are new `examples/plot_squall_*.py` gallery files that use `nl.execute_squall_program()` for all logic rules. Task 4 runs the parser and integration test suites to confirm no regressions.

**Tech Stack:** Python, Lark LALR grammar, NeuroLang IR (`Implication`, `Conjunction`, `ProbabilisticQuery`, `Condition`, `AggregationApplication`), `NeurolangPDL`, nibabel, nilearn, numpy, Sphinx Gallery.

**Working directory:** `/Users/dwasserm/sources/NeuroLang/.worktrees/squall-cnl`

**Run tests with:** `uv run python -m pytest`

---

## File Map

| File | Change |
|------|--------|
| `neurolang/frontend/datalog/squall_syntax_lark.py` | Modify `rel_fun_call` lines 837–841: skip subject-prepend when `x` is a tuple |
| `neurolang/frontend/datalog/tests/test_squall_parser.py` | Add `test_rel_fun_call_tuple_subject_no_prepend` |
| `examples/plot_squall_neurosynth.py` | New gallery file — NeuroSynth in SQUALL |
| `examples/plot_squall_cbma_spatial_prior.py` | New gallery file — CBMA spatial prior in SQUALL |

---

## Task 1: Fix `rel_fun_call` tuple-subject prepend

When a SQUALL sentence binds a multi-variable noun like `Focus_reported (?i2;?j2;?k2;?s)`, the transformer sets the subject `x` to a **tuple** `(Symbol('i2'), Symbol('j2'), Symbol('k2'), Symbol('s'))`. The current multi-label path in `rel_fun_call` always prepends `x` as the first argument: `func_sym(x, *_vars)`. When `x` is a tuple and `_vars` already covers all parameter positions, this produces `func_sym((i2,j2,k2,s), i1, j1, k1, i2, j2, k2)` — wrong arity and wrong type for the first arg.

The fix: when `x` is a tuple, the labels cover all positions and `x` should not be prepended.

**Files:**
- Modify: `neurolang/frontend/datalog/squall_syntax_lark.py` lines 837–841
- Test: `neurolang/frontend/datalog/tests/test_squall_parser.py`

- [ ] **Step 1: Write the failing parser test**

Add at the end of `neurolang/frontend/datalog/tests/test_squall_parser.py`:

```python
def test_rel_fun_call_tuple_subject_no_prepend():
    """rel_fun_call with a tuple-subject noun must NOT prepend the tuple as first arg.

    'every Focus_reported (?i2;?j2;?k2;?s) that is_near(?i1,?j1,?k1,?i2,?j2,?k2) holds'
    should emit is_near(i1,j1,k1,i2,j2,k2) — the six explicit labels, nothing extra.
    Specifically, both i1 and i2 must appear as free variables in the rule body.
    """
    from ..squall_syntax_lark import parser
    from ....datalog import Implication
    from ....logic.expression_processing import extract_logic_free_variables

    result = parser(
        "squall define as Near every Focus_reported (?i2; ?j2; ?k2; ?s) "
        "that is_near(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) holds."
    )
    rules = result if isinstance(result, list) else [result]
    implications = [r for r in rules if isinstance(r, Implication)]
    assert len(implications) == 1
    body_str = str(implications[0].antecedent)
    assert "is_near" in body_str.lower(), f"is_near not in body: {body_str}"
    free_vars = extract_logic_free_variables(implications[0].antecedent)
    var_names = {v.name for v in free_vars}
    assert "i1" in var_names, f"i1 missing from free vars {var_names}"
    assert "i2" in var_names, f"i2 missing from free vars {var_names}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/dwasserm/sources/NeuroLang/.worktrees/squall-cnl
uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_rel_fun_call_tuple_subject_no_prepend -v
```

Expected: FAIL — either the is_near atom contains a tuple as its first argument, or `i1` is absent from free vars because the tuple `(i2,j2,k2,s)` is prepended instead.

- [ ] **Step 3: Apply the one-line fix**

In `neurolang/frontend/datalog/squall_syntax_lark.py`, the current `rel_fun_call` `else` branch is at lines 837–841:

```python
        else:
            # N-ary: subject is prepended so noun head is constrained
            def rel(x, _vars=label_vars):
                return func_sym(x, *_vars)
            return ('_rel', rel)
```

Replace it with:

```python
        else:
            # N-ary: prepend subject only when it is a scalar Symbol.
            # When the enclosing noun binds a tuple (e.g. Foo (?a;?b;?c)),
            # the labels already cover every argument position and the
            # tuple must NOT be prepended — it would produce wrong arity.
            def rel(x, _vars=label_vars):
                if isinstance(x, tuple):
                    return func_sym(*_vars)
                return func_sym(x, *_vars)
            return ('_rel', rel)
```

- [ ] **Step 4: Run the new test — expect PASS**

```bash
uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_rel_fun_call_tuple_subject_no_prepend -v
```

Expected: PASS

- [ ] **Step 5: Confirm the existing `rel_fun_call` scalar test still passes**

```bash
uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py::test_rel_body_function_call_parses -v
```

Expected: PASS (scalar path `func_sym(x, *_vars)` unchanged)

- [ ] **Step 6: Run full parser suite**

```bash
uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py -v
```

Expected: all pass (≥27 passed, 1 skipped).

- [ ] **Step 7: Commit**

```bash
git add neurolang/frontend/datalog/squall_syntax_lark.py \
        neurolang/frontend/datalog/tests/test_squall_parser.py
git commit -m "fix(squall): rel_fun_call tuple-subject must not prepend tuple as first arg"
```

---

## Task 2: `plot_squall_neurosynth.py`

New Sphinx Gallery file that mirrors `examples/plot_neurosynth_implementation.py`. Data loading is identical Python; the four logic rules are expressed as SQUALL sentences passed to `nl.execute_squall_program()`.

**SQUALL grammar constructs used:**
- `such that ?s is a Selected_study` — `rel_s(s_np_vp)` → `selected_study(s)`; confirmed working because `Symbol('s')` identity is shared between the tuple `app_label` and the sub-sentence.
- `?t that is 'auditory'` — `rel_vp(vp_aux(be, vpbe_npc(Constant('auditory'))))` → `eq(t, 'auditory')`.
- `with probability … conditioned to …` — `rule_op_marg` → `Implication(head(…, ProbabilisticQuery(PROB, vars)), Condition(…))`.
- `every Agg_create_region_overlay of the Activation_given_term` — `ng1_agg_npc` with arbitrary functor (Gap B) → `AggregationApplication(Symbol('agg_create_region_overlay'), free_vars)`.

**Important naming note:** `nl.execute_squall_program` lowercases all relation names. EDB tuple-sets must therefore be registered with lowercase names (`peak_reported`, `selected_study`, `term_in_study_tfidf`). Results are retrieved from `nl.solve_all()` by lowercase key.

**Files:**
- Create: `examples/plot_squall_neurosynth.py`

- [ ] **Step 1: Create the file**

Create `examples/plot_squall_neurosynth.py` with exactly this content:

```python
# -*- coding: utf-8 -*-
r"""
NeuroSynth Query in SQUALL Controlled English
=============================================

Reproduces the NeuroSynth forward model — P(Activation | Term = 'auditory') —
using `SQUALL controlled natural language
<https://doi.org/10.18653/v1/2020.acl-main.235>`_ instead of the IR builder
DSL.

The four logic rules are expressed as plain English sentences and executed via
:func:`~neurolang.frontend.NeurolangPDL.execute_squall_program`. Compare with
:ref:`sphx_glr_auto_examples_plot_neurosynth_implementation.py` which writes
identical rules using ``with nl.scope as e:``.

.. rubric:: The SQUALL program

.. code-block:: text

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
from neurolang import ExplicitVBROverlay, NeurolangPDL
from neurolang.frontend.neurosynth_utils import get_ns_mni_peaks_reported

# %%
###############################################################################
# Data preparation
# ----------------
# Load the MNI T1 atlas resampled to 4 mm isotropic voxels.

data_dir = Path.home() / "neurolang_data"

mni_t1 = nibabel.load(
    nilearn.datasets.fetch_icbm152_2009(data_dir=str(data_dir / "icbm"))["t1"]
)
mni_t1_4mm = nilearn.image.resample_img(mni_t1, np.eye(3) * 4)

# %%
###############################################################################
# Set up the probabilistic engine and register helper symbols
# -----------------------------------------------------------

nl = NeurolangPDL()


@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> ExplicitVBROverlay:
    """Aggregate (i,j,k,probability) rows into a brain overlay image."""
    mni_coords = np.c_[i, j, k]
    return ExplicitVBROverlay(
        mni_coords, mni_t1_4mm.affine, p, image_dim=mni_t1_4mm.shape
    )


# %%
###############################################################################
# Load the NeuroSynth database
# ----------------------------
# Register peaks, study IDs, and term–study associations as extensional facts.
# Relation names are lowercase so they match SQUALL's case-folding convention.

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
study_ids = nl.load_neurosynth_study_ids(data_dir, "study")
nl.add_uniform_probabilistic_choice_over_set(
    study_ids.value, name="selected_study"
)
nl.load_neurosynth_term_study_associations(
    data_dir, "term_in_study_tfidf", tfidf_threshold=1e-3
)

# %%
###############################################################################
# SQUALL controlled-English program
# ----------------------------------
# Four sentences replace the entire ``with nl.scope as e:`` block.
#
# * ``such that ?s is a Selected_study``  — existential filter: study is selected
# * ``?t that is 'auditory'``             — string equality filter on the term
# * ``with probability … conditioned to`` — MARG query (conditional probability)
# * ``every Agg_create_region_overlay of the …`` — aggregation into brain image

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

# %%
###############################################################################
# Solve and retrieve the probability map
# ---------------------------------------

solution = nl.solve_all()
result_image = (
    solution["activation_given_term_image"]
    .as_pandas_dataframe()
    .iloc[0, 0]
    .spatial_image()
)

# %%
###############################################################################
# Plot
# ----

img = result_image.get_fdata()
plot = nilearn.plotting.plot_stat_map(
    result_image, threshold=np.percentile(img[img > 0], 95)
)
nilearn.plotting.show()
```

- [ ] **Step 2: Verify the file is picked up by Sphinx Gallery config**

```bash
cd /Users/dwasserm/sources/NeuroLang/.worktrees/squall-cnl
python -c "
from pathlib import Path
files = sorted(Path('examples').glob('plot_squall_*.py'))
print('Gallery files found:', [f.name for f in files])
assert Path('examples/plot_squall_neurosynth.py') in files
print('OK')
"
```

Expected:
```
Gallery files found: ['plot_squall_neurosynth.py']
OK
```

- [ ] **Step 3: Verify the SQUALL program parses without error**

```bash
uv run python -c "
from neurolang.frontend.datalog.squall_syntax_lark import parser

prog = '''
define as Activation every Peak_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

define as Term_association every Term_in_study_tfidf (?s; ?t; _)
    such that ?s is a Selected_study.

define as Activation_given_term with probability
    every Activation (?i; ?j; ?k)
    conditioned to every Term_association ?t that is 'auditory'.

define as Activation_given_term_image
    every Agg_create_region_overlay of the Activation_given_term.
'''
result = parser(prog)
print('Parsed OK:', type(result).__name__)
"
```

Expected output (no exception):
```
Parsed OK: SquallProgram
```

- [ ] **Step 4: Commit**

```bash
git add examples/plot_squall_neurosynth.py
git commit -m "feat(gallery): add plot_squall_neurosynth.py — NeuroSynth query in SQUALL controlled English"
```

---

## Task 3: `plot_squall_cbma_spatial_prior.py`

New Sphinx Gallery file mirroring `examples/plot_cbma_spatial_prior.py`. Registers two additional Python helper symbols (`is_near`, `max_weighted_proximity`) then expresses all six logic rules as SQUALL sentences.

**Key grammar constructs:**
- `for every Voxel (?i1;?j1;?k1) ; where every Focus_reported (?i2;?j2;?k2;?s) that is_near(…) holds` — `rule_opnn` outer structure; inner `rel_fun_call` with tuple subject (requires Task 1 fix).
- `every Max_weighted_proximity of the Near_focus per ?i1 per ?j1 per ?k1 per ?s` — `ng1_agg_npc` with arbitrary functor + four `dim_npc` per-vars.
- All others identical to Task 2.

**Files:**
- Create: `examples/plot_squall_cbma_spatial_prior.py`

- [ ] **Step 1: Verify the Near_focus rule parses correctly (depends on Task 1 fix)**

```bash
cd /Users/dwasserm/sources/NeuroLang/.worktrees/squall-cnl
uv run python -c "
from neurolang.frontend.datalog.squall_syntax_lark import parser

prog = '''
define as Near_focus for every Voxel (?i1; ?j1; ?k1) ;
    where every Focus_reported (?i2; ?j2; ?k2; ?s)
    that is_near(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) holds.
'''
result = parser(prog)
print('Parsed OK:', type(result).__name__)
"
```

Expected (no exception):
```
Parsed OK: SquallProgram
```

- [ ] **Step 2: Verify the full SQUALL program parses**

```bash
uv run python -c "
from neurolang.frontend.datalog.squall_syntax_lark import parser

prog = '''
define as Near_focus for every Voxel (?i1; ?j1; ?k1) ;
    where every Focus_reported (?i2; ?j2; ?k2; ?s)
    that is_near(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) holds.

define as Voxel_reported every Max_weighted_proximity of the Near_focus
    per ?i1 per ?j1 per ?k1 per ?s.

define as Term_association every Term_in_study_tfidf (?s; ?t; _)
    such that ?s is a Selected_study.

define as Activation every Voxel_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

define as Probmap with probability every Activation (?i; ?j; ?k)
    conditioned to every Term_association ?t that is 'emotion'.

define as Img every Agg_create_region_overlay of the Probmap.
'''
result = parser(prog)
print('Parsed OK:', type(result).__name__, '— rules:', len(result.rules))
"
```

Expected (no exception):
```
Parsed OK: SquallProgram — rules: 6
```

- [ ] **Step 3: Create the file**

Create `examples/plot_squall_cbma_spatial_prior.py` with exactly this content:

```python
# -*- coding: utf-8 -*-
r"""
CBMA Spatial Prior in SQUALL Controlled English
================================================

Reproduces the coordinate-based meta-analysis spatial decay prior — each
voxel's probability of being reported is weighted by ``max(exp(−d/5))``
over nearby foci — using `SQUALL controlled natural language
<https://doi.org/10.18653/v1/2020.acl-main.235>`_ instead of the IR
builder DSL.

Two small Python helper functions are registered as SQUALL-callable symbols:

* ``is_near(i1,j1,k1,i2,j2,k2)`` — True when euclidean distance < 1 voxel
* ``max_weighted_proximity(d_values)`` — max(exp(−d/5)) aggregation

All six logic rules are then expressed as plain English sentences passed to
:func:`~neurolang.frontend.NeurolangPDL.execute_squall_program`. Compare with
:ref:`sphx_glr_auto_examples_plot_cbma_spatial_prior.py` which writes the same
rules using ``with nl.environment as e:``.

.. rubric:: The SQUALL program

.. code-block:: text

    define as Near_focus for every Voxel (?i1; ?j1; ?k1) ;
        where every Focus_reported (?i2; ?j2; ?k2; ?s)
        that is_near(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) holds.

    define as Voxel_reported every Max_weighted_proximity of the Near_focus
        per ?i1 per ?j1 per ?k1 per ?s.

    define as Term_association every Term_in_study_tfidf (?s; ?t; _)
        such that ?s is a Selected_study.

    define as Activation every Voxel_reported (?i; ?j; ?k; ?s)
        such that ?s is a Selected_study.

    define as Probmap with probability every Activation (?i; ?j; ?k)
        conditioned to every Term_association ?t that is 'emotion'.

    define as Img every Agg_create_region_overlay of the Probmap.
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

# %%
###############################################################################
# Data preparation
# ----------------
# Load MNI T1 atlas resampled to 2 mm isotropic voxels.

data_dir = Path.home() / "neurolang_data"

mni_t1 = nibabel.load(
    nilearn.datasets.fetch_icbm152_2009(data_dir=str(data_dir / "icbm"))["t1"]
)
mni_t1_2mm = nilearn.image.resample_img(mni_t1, np.eye(3) * 2)

# %%
###############################################################################
# Set up engine and register helper symbols
# -----------------------------------------
# ``is_near`` and ``max_weighted_proximity`` are called from within SQUALL
# sentences.  ``agg_create_region_overlay`` aggregates the final result into
# a brain overlay image.

nl = NeurolangPDL()


@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> ExplicitVBR:
    """Aggregate (i,j,k,probability) rows into a brain overlay image."""
    voxels = np.c_[i, j, k]
    return ExplicitVBROverlay(
        voxels, mni_t1_2mm.affine, p, image_dim=mni_t1_2mm.shape
    )


@nl.add_symbol
def is_near(i1: int, j1: int, k1: int, i2: int, j2: int, k2: int) -> bool:
    """True when the euclidean distance between two voxels is strictly less than 1."""
    return float(np.linalg.norm([i1 - i2, j1 - j2, k1 - k2])) < 1.0


@nl.add_symbol
def max_weighted_proximity(d_values: Iterable) -> float:
    """Aggregation: max(exp(−d/5)) over all proximity values for a (voxel, study) pair."""
    return max(float(np.exp(-d / 5.0)) for d in d_values)


# %%
###############################################################################
# Load the NeuroSynth database
# ----------------------------

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
    )
    .swapaxes(0, 1)
    .reshape(3, -1)
    .T,
    name="voxel",
)

# %%
###############################################################################
# SQUALL controlled-English program
# ----------------------------------
# Six sentences replace the entire ``with nl.environment as e:`` block.
#
# Sentence 1  ``for every Voxel … where every Focus_reported … that is_near(…) holds``
#             — ``rel_fun_call`` with tuple subject; subject NOT prepended (Task 1 fix).
# Sentence 2  ``every Max_weighted_proximity of the Near_focus per ?i1 …``
#             — arbitrary-functor aggregation with four per-vars.
# Sentences 3–4  ``such that ?s is a Selected_study`` — existential filter.
# Sentence 5  ``with probability … conditioned to … that is 'emotion'`` — MARG query.
# Sentence 6  ``every Agg_create_region_overlay of the Probmap`` — brain image aggregation.

squall_program = """
define as Near_focus for every Voxel (?i1; ?j1; ?k1) ;
    where every Focus_reported (?i2; ?j2; ?k2; ?s)
    that is_near(?i1, ?j1, ?k1, ?i2, ?j2, ?k2) holds.

define as Voxel_reported every Max_weighted_proximity of the Near_focus
    per ?i1 per ?j1 per ?k1 per ?s.

define as Term_association every Term_in_study_tfidf (?s; ?t; _)
    such that ?s is a Selected_study.

define as Activation every Voxel_reported (?i; ?j; ?k; ?s)
    such that ?s is a Selected_study.

define as Probmap with probability every Activation (?i; ?j; ?k)
    conditioned to every Term_association ?t that is 'emotion'.

define as Img every Agg_create_region_overlay of the Probmap.
"""

nl.execute_squall_program(squall_program)

# %%
###############################################################################
# Solve and retrieve the probability map
# ---------------------------------------

solution = nl.solve_all()
result_image = (
    solution["img"]
    .as_pandas_dataframe()
    .iloc[0, 0]
    .spatial_image()
)

# %%
###############################################################################
# Plot
# ----

img = result_image.get_fdata()
plot = nilearn.plotting.plot_stat_map(
    result_image, threshold=np.percentile(img[img > 0], 95)
)
nilearn.plotting.show()
```

- [ ] **Step 4: Verify both gallery files are present**

```bash
python -c "
from pathlib import Path
files = sorted(Path('examples').glob('plot_squall_*.py'))
names = [f.name for f in files]
print('Gallery files:', names)
assert 'plot_squall_neurosynth.py' in names
assert 'plot_squall_cbma_spatial_prior.py' in names
print('OK')
"
```

Expected:
```
Gallery files: ['plot_squall_cbma_spatial_prior.py', 'plot_squall_neurosynth.py']
OK
```

- [ ] **Step 5: Commit**

```bash
git add examples/plot_squall_cbma_spatial_prior.py
git commit -m "feat(gallery): add plot_squall_cbma_spatial_prior.py — CBMA spatial prior in SQUALL"
```

---

## Task 4: Regression sweep

Run the full parser and integration test suites to confirm the Task 1 transformer fix introduced no regressions.

**Files:** none changed — verification only.

- [ ] **Step 1: Run all SQUALL parser tests**

```bash
cd /Users/dwasserm/sources/NeuroLang/.worktrees/squall-cnl
uv run python -m pytest neurolang/frontend/datalog/tests/test_squall_parser.py -v 2>&1 | tail -15
```

Expected: all pass (≥28 passed, 1 skipped).

- [ ] **Step 2: Run SQUALL integration tests**

```bash
uv run python -m pytest neurolang/frontend/tests/test_squall_pdl_integration.py -v 2>&1 | tail -10
```

Expected: all pass (≥15 passed).

- [ ] **Step 3: Run broader frontend suite**

```bash
uv run python -m pytest neurolang/frontend/ -q --tb=short 2>&1 | tail -15
```

Expected: 0 new failures (pre-existing probabilistic solver failures are acceptable).

- [ ] **Step 4: Commit any fixups if needed**

If any test fails due to our changes, fix the issue and commit:

```bash
git add <files>
git commit -m "fix(squall): <description of fixup>"
```

If all pass with no changes needed, no commit is necessary.

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `rel_fun_call` tuple-subject fix | Task 1 ✅ |
| `test_rel_fun_call_tuple_subject_no_prepend` parser test | Task 1 ✅ |
| `plot_squall_neurosynth.py` gallery file | Task 2 ✅ |
| SQUALL sentences: Activation, Term_association, Activation_given_term, Activation_given_term_image | Task 2 ✅ |
| `plot_squall_cbma_spatial_prior.py` gallery file | Task 3 ✅ |
| `is_near`, `max_weighted_proximity` Python helpers | Task 3 ✅ |
| SQUALL sentences: Near_focus, Voxel_reported, Term_association, Activation, Probmap, Img | Task 3 ✅ |
| Regression sweep | Task 4 ✅ |

**Placeholder scan:** None found.

**Type consistency:**
- `nl.execute_squall_program(squall_program)` — confirmed method in `query_resolution_datalog.py`.
- `nl.solve_all()` returns `Dict[str, NamedRelationalAlgebraFrozenSet]` — confirmed.
- `solution["activation_given_term_image"]` — lowercase key matches SQUALL case-folding of `Activation_given_term_image`.
- `solution["img"]` — lowercase key matches `define as Img`.
- `.as_pandas_dataframe().iloc[0, 0].spatial_image()` — identical retrieval pattern to the original `plot_neurosynth_implementation.py` and `plot_cbma_spatial_prior.py`.
- `agg_create_region_overlay` registered with `@nl.add_symbol` before any SQUALL sentence references it — order is correct.
- Task 1 fix line numbers (837–841) verified against current file state.
