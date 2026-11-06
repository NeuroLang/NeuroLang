# nl_tracts

Tractography for NeuroLang: tractograms and atlases enter an engine as
extensional relations (`streamline_label`, `endpoint_label`, `label`,
`label_term`, `streamline_extent`, ...) computed in bulk with numpy, a
streamline being one integer id `x` that `streamline(x, t, s)` decodes to its
tractogram and its position in the file, and queries are written
in Datalog, with the vocabulary of tractography. See [DESIGN.md](DESIGN.md).

## The vocabulary of tractography

Rules over the relations, in which a bundle is a rule or two
([`nl_tracts/vocabulary.dl`](nl_tracts/vocabulary.dl), DESIGN.md section 16):
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
in the file (they are read relative to where it runs), then, with the extra
`tracts` of `neurolang-apps`:

```sh
neurolang-query --engine tracts --file uncinate.dl     # the ids x
neurolang-query --engine tracts -i --file uncinate.dl  # the REPL, then:
nl(tracts:DL)> .view
```

`.view` draws the streamlines of the answer, coloured by direction, on the
label image of the atlas, in the browser. At the REPL's prompt the commands
and the rules can also be typed or pasted one by one, and other questions
asked of the same relations (`ans(x) :- uncinate(x, "right")`). From
Python, `nl_tracts.frontend.tracts_engine()` is the same engine.

## Bundles and the studies of Neurosynth

The engine `neurosynth_tracts` is `neurosynth` and `tracts` together (the
extras `tracts` and `embeddings`): the ends of a bundle's streamlines,
`endpoint_point(x, e, px, py, pz)` in mm, join the voxels of the parcels of
its atlases, `schaefer_points(x, y, z, n)`, and the parcels the studies
that activate them, `schaefer_activated(n, s)`. The parcels of the uncinate
are those near (3 mm) at least 5 % of its ends, an end being the integer
`2 x + e`: a parcel near a few ends only would bring thousands of studies.
[`examples/uncinate_cogat.dl`](examples/uncinate_cogat.dl) asks which
Cognitive Atlas concepts the studies of the parcels the uncinate connects
are about:

```
uf_end_near(i, n) :- uncinate(x, "left"), endpoint_point(x, e, ex, ey, ez),
    schaefer_points(mx, my, mz, n),
    (EUCLIDEAN(ex, ey, ez, mx, my, mz) <= 3), (i == 2 * x + e)
uf_parcel_ends(n, count(i)) :- uf_end_near(i, n)
uf_ends(count(i)) :- uncinate(x, "left"), endpoint_point(x, e, ex, ey, ez),
    (i == 2 * x + e)
uf_parcel(n) :- uf_parcel_ends(n, c), uf_ends(t), (c >= 0.05 * t)
uf_study(s) :- uf_parcel(n), schaefer_activated(n, s), selected_study(s)
measuring(c) :- cogat_measures(c, x)
cogat_concept(c, t) :- cogat_term(c, t),
    cogat_subclass_of(c, <http://www.cognitiveatlas.org/ontology/cogat.owl#CAO_00001>), ~measuring(c)
concept_definition(t, d) :- cogat_concept(c, t), skos:definition(c, d)
about(t, s) :- concept_definition(t, d), similar_study(d, s), selected_study(s)
p_concept_given_uf(t, PROB) :- about(t, s) // uf_study(s)
ans(t, p) :- p_concept_given_uf(t, p)
```

after the commands and the rules of the uncinate of `uncinate.dl`, the
tractogram being also indexed by the Schaefer atlas of the engine
(`.index_tracts("hcp842", "schaefer")`), which has no terms and leaves the
uncinate unchanged. `similar_study(d, s)` keeps the 5 % of the studies
closest to the definition `d` (SPECTER2), so a concept is about 5 % of the
studies whatever it is, and ranking the concepts by `p` ranks them by the
lift `p / 0.05`. The concepts are the classes below the root concept of the
Cognitive Atlas that measure nothing: the ontology also puts a few
contrasts there.

```sh
neurolang-query --engine neurosynth_tracts --file uncinate_cogat.dl
```

## The atlases of the engine

The engine `tracts` has the atlases of the engine `neurosynth`, Destrieux,
Schaefer and Tian, in MNI152: a tractogram in MNI152 is indexed by them
without loading an atlas, and their labels are named as their relations
name them, so that `label("destrieux", l, n, h)` joins
`destrieux(n, region)`:

```
.load_tractogram("hcp842", "whole_brain_MNI.trk", space="MNI152")
.index_tracts("hcp842", "destrieux")
.index_tracts("hcp842", "tian")
```

Destrieux has the terms of `freesurfer_destrieux` (`frontal`, `insula`,
`temporal`...), Tian those of the aseg structures (`amygdala`, `putamen`,
`striatum`...), so the rules of the uncinate run unchanged over the two.
Both are atlases of the grey matter; the uncinate, defined on the cortex
and the white matter under it, is found far better with a `wmparc`
(DESIGN.md, 16.4).

An atlas of nilearn is loaded by the name of its fetcher, its other
options going to it, in MNI152, its names as the engine's (nilearn's
Destrieux has the terms of `freesurfer_destrieux`):

```
.load_tract_atlas("d", kind="nilearn", fetcher="destrieux_2009")
.load_tract_atlas("ho", kind="nilearn", fetcher="harvard_oxford",
                  atlas_name="cort-maxprob-thr25-1mm")
```

