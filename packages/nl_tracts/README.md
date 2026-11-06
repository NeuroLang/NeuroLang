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

## tract_querier's queries in the vocabulary

`nl-tracts wmql` translates a WMQL program of tract_querier (a `.qry`) into
rules of the vocabulary, one predicate per query, with the WMQL statement
over each (DESIGN.md section 17): `UF.side` becomes `uf(x, h)`, the rule
above; a name of labels is a term of the mapping; what the vocabulary
cannot say is left out and listed in the header.

```sh
nl-tracts wmql my_queries.qry -I path/to/tract_querier/data \
    --terms freesurfer_aseg_aparc > my_queries.dl
```

`--terms` checks the names of labels against a mapping (a name that is not
one of its terms stands for those it is made of). The translations of
tract_querier's own files are shipped (`freesurfer_queries`,
`freesurfer_queries_bsf2016`, `freesurfer_queries_bsf2016_toad`,
`mori_queries`): rules without commands, to put after the commands of a
program, or from Python

```python
from nl_tracts.frontend import tracts_engine
from nl_tracts.wmql_vocab import bundle

nl = tracts_engine()
nl.execute_datalog_program(commands)   # .load_tract_atlas, ...
nl.execute_datalog_program(bundle("freesurfer_queries"))
nl.execute_datalog_program('ans(x) :- af(x, "left")')
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

