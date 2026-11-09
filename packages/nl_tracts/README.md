# nl_tracts

Tractography for NeuroLang: tractograms and atlases enter an engine as
extensional relations (`streamline_label`, `endpoint_label`, `label`,
`label_term`, `streamline_extent`, ...) computed in bulk with numpy, a
streamline being one integer id `x` that `streamline(x, t, s)` decodes to its
tractogram and its position in the file, and queries are written
in Datalog, with the vocabulary of tractography.

## The vocabulary of tractography

Rules over the relations, in which a bundle is a rule or two
([`nl_tracts/vocabulary.dl`](nl_tracts/vocabulary.dl),):
`traverses(x, term, h)`, `endpoint_in(x, e, term, h)`, `ends_in(x, term, h)`,
`in_hemisphere(x, h)`, `only_in(x, term, h)`, `anterior_of(x, term, h)` and
the other directions (anatomical; `medial_of` and `lateral_of` relative to
the hemisphere `h`), and `endpoint_anterior_of(x, e, term, h)` and the like
for an end. A term is one of the term mapping (`"temporal"`, `"insula"`...),
`h` is `"left"`, `"right"` or `"none"`.

The uncinate fasciculus, tract_querier's `UF.side`, is
[`examples/uncinate.dl`](examples/uncinate.dl):

```
uncinate(x, h) :-
    (traverses(x, "orbitofrontalgyrus", h) | traverses(x, "middle_frontal_gyrus", h)
     | traverses(x, "inferior_frontal_gyrus", h)),
    endpoint_in(x, e, "temporal", h), endpoint_anterior_of(x, e, "amygdala", h),
    traverses(x, "insula", h),
    ~traverses(x, "occipital", h), ~traverses(x, "parietal", h),
    ~traverses(x, "cingular", h), ~posterior_of(x, "putamen", h),
    opposite_hemisphere(h, o), ~in_hemisphere(x, o)
ans(x) :- uncinate(x, "left")
```

after three commands that load a FreeSurfer `wmparc` with its colour LUT and
a tractogram in its space, and index the one by the other. Edit their paths
in the file (they are read relative to where it runs), then run it from
Python with the engine of tractography, `nl_tracts.frontend.tracts_engine()`:

```python
from nl_tracts.frontend import tracts_engine

result = tracts_engine().execute_datalog_program(open("uncinate.dl").read())
```

An atlas of nilearn is loaded by the name of its fetcher, its other
options going to it, in MNI152 (nilearn's Destrieux has the terms of
`freesurfer_destrieux`):

```
.load_tract_atlas("d", kind="nilearn", fetcher="destrieux_2009")
.load_tract_atlas("ho", kind="nilearn", fetcher="harvard_oxford",
                  atlas_name="cort-maxprob-thr25-1mm")
```

The package reads atlases (`nl_tracts.atlas`, `nl_tracts.spaces`, the term
mappings of `nl_tracts.terms` and `nl_tracts/mappings`) and tractograms
(`nl_tracts.tractogram`), indexes the one by the other into the relations
(`nl_tracts.indexing`, `nl_tracts.ids`, cached by `nl_tracts.cache`) with
numpy, scipy, nibabel, pandas and pyarrow, without NeuroLang;
`nl_tracts.frontend` hands the relations to a NeuroLang engine.
