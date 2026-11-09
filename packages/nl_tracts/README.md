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

after three commands that load a FreeSurfer `wmparc` with its colour LUT
(`.load_atlas("fs", "wmparc.mgz", "FreeSurferColorLUT.txt",
kind="freesurfer", space="MNI152")`, the command of every engine of
NeuroLang, see below) and a tractogram in its space, and index the one by
the other. Edit their paths
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

## The atlases

An atlas is NeuroLang's (`neurolang.atlases.Atlas`): whatever loaded it,
`.load_atlas` (of kind `freesurfer`, `bids`, `tsv`, `destrieux`,
`schaefer`, `tian`, `nilearn`, or the CSV at a URL of old),
`add_atlas_set` from Python, or the `atlases:` of an engine of the apps,
the engine records it by name and `.index_tracts(t, name)` indexes by it.
nl_tracts adds the term mapping of its labels, which `.index_tracts`
attaches: `term_mapping=...` (the key of a shipped mapping or the path of
one), else the default of the atlas's kind, `freesurfer_aseg_aparc` for a
FreeSurfer colour LUT, `freesurfer_destrieux` for Destrieux (also nilearn's
`destrieux_2009`), `tian_subcortex_s2` for Tian; Schaefer and the other
kinds have none.

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
options going to it, in MNI152, its names normalised as the engine's
(nilearn's Destrieux has the terms of `freesurfer_destrieux`):

```
.load_atlas("d", kind="nilearn", fetcher="destrieux_2009")
.load_atlas("ho", kind="nilearn", fetcher="harvard_oxford",
            atlas_name="cort-maxprob-thr25-1mm")
.index_tracts("hcp842", "d")
```

