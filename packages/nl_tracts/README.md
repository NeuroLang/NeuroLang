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

after three commands that load a FreeSurfer `wmparc` with its colour LUT,
the whole-brain tractogram of the HCP842 atlas (downloaded, below), and
index the one by the other. Edit the paths of the atlas in the file (they
are read relative to where it runs), then, with the extra `tracts` of
`neurolang-apps`:

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

[`examples/uncinate_hcp842.dl`](examples/uncinate_hcp842.dl) counts the
streamlines of the atlas's own uncinates, `UF_L` and `UF_R`, that the rule
selects: 363 of 370 and 324 of 328 with the `wmparc` of FreeSurfer's
`cvs_avg35_inMNI152`.

## Getting the data: the atlases of tractography of Yeh

Two public atlases of streamlines in MNI152, whose bundles were labelled by
experts, are loaded by their kind, downloaded the first time:

```
.load_tractogram("hcp842", kind="hcp842")                 % the whole brain
.load_tractogram("uf_l", kind="hcp842", bundle="UF_L")    % a bundle
.load_tractogram("af_l", kind="hcp1065", bundle="AF_L")
```

The archive is kept, and the files asked for extracted next to it, in
`<data_dir>/tracts/<atlas>/` with `neurolang-query` (`--data-dir`,
`neurolang_data` by default), else in `~/.cache/nl_tracts/<atlas>/`;
`data_dir="..."` chooses another, `url="..."` (and its `md5="..."`)
another source. From Python, `nl_tracts.fetch_tractogram("hcp842",
bundle="UF_L")` is the path of the `.trk`. The streamlines of one point
(37 of the whole brain of HCP842) are in no relation, with a warning; the
others keep their positions in the file as ids `s`.

* **`hcp842`**: the HCP842 atlas, Yeh et al., "Population-averaged atlas
  of the macroscale human structural connectome and its network
  topology", *NeuroImage* 178:57-68 (2018), as curated by E. Garyfallidis
  for DIPY's RecoBundles, "Advanced Atlas of 80 Bundles in MNI space"
  ([figshare](https://doi.org/10.6084/m9.figshare.7375883), the
  `Atlas_80_Bundles.zip` that DIPY's `fetch_bundle_atlas_hcp842`
  downloads, 320 MB, md5 `78331d527a10ec000d4f33bac472e099`): the
  whole-brain tractogram (144,678 streamlines) and 78 bundles, `.trk`,
  ICBM 2009a. Licence **CC BY 4.0**; cite Garyfallidis et al.,
  "Recognition of white matter bundles using local and global
  streamline-based registration and clustering", *NeuroImage*
  170:283-297 (2017), and Yeh et al. 2018.
* **`hcp1065`**: the HCP1065 atlas, Yeh, "Population-based tract-to-region
  connectome of the human brain and its hierarchical topology", *Nat
  Commun* 13:4933 (2022), from its
  [release](https://github.com/data-others/atlas/releases/tag/hcp1065)
  (`hcp1065_avg_tracts_trk.zip`, 588 MB): 87 bundles, `.trk.gz`, ICBM
  2009a, 1 mm; no whole brain. Licence **CC BY-SA 4.0**
  ([brain.labsolver.org](https://brain.labsolver.org/hcp_trk_atlas.html));
  cite Yeh 2022 and acknowledge the Human Connectome Project, whose data
  they are.

The bundles are named by the stem of their file (any case), `_L` and `_R`
for the sides; `nl_tracts.fetch.ATLASES[atlas]["bundles"]` has them with
their full names.

HCP842: `AC` anterior commissure; `AF` arcuate fasciculus; `AR` acoustic
radiation; `AST` frontal aslant tract; `C` cingulum; `CB` cerebellum; `CC`
corpus callosum; `CC_ForcepsMajor`, `CC_ForcepsMinor`, `CCMid` its forceps
major, forceps minor and body; `CNII`, `CNIII`, `CNIV`, `CNV`, `CNVII`,
`CNVIII` the cranial nerves II (optic), III (oculomotor), IV (trochlear), V
(trigeminal), VII (facial), VIII (vestibulocochlear); `CS` corticostriatal
pathway; `CST` corticospinal tract; `CT` corticothalamic pathway; `CTT`
central tegmental tract; `DLF` dorsal longitudinal fasciculus; `EMC`
extreme capsule; `F_L_R` fornix, both sides; `FPT` frontopontine tract;
`ICP` inferior cerebellar peduncle; `IFOF` inferior fronto-occipital
fasciculus; `ILF` inferior longitudinal fasciculus; `LL` lateral lemniscus;
`MCP` middle cerebellar peduncle; `MdLF` middle longitudinal fasciculus;
`ML` medial lemniscus; `MLF` medial longitudinal fasciculus; `OPT`
occipitopontine tract; `OR` optic radiation; `PC` posterior commissure;
`PPT` parietopontine tract; `RST` rubrospinal tract; `SCP` superior
cerebellar peduncle; `SLF` superior longitudinal fasciculus; `STT`
spinothalamic tract; `TPT` temporopontine tract; `UF` uncinate fasciculus;
`V` vermis; `VOF` vertical occipital fasciculus. (The archive's
`IF0F_R.trk` is a copy of `IFOF_R.trk`.)

HCP1065: `AF`, `EMC`, `IFOF`, `ILF`, `MdLF`, `UF`, `VOF`, `AR`, `CST`,
`ML`, `OR`, `CNII` to `CNVIII` (no `CNIV`), `CB`, `ICP`, `MCP`, `SCP`, `V`,
`AC`, `CC` as above; `C_FPH`, `C_FP`, `C_PHP`, `C_PH`, `C_PO` the frontal
parahippocampal, frontal parietal, parahippocampal parietal,
parahippocampal and parolfactory cingulum; `FAT` frontal and `PAT`
parietal aslant tract; `SLF1`, `SLF2`, `SLF3` superior longitudinal
fasciculus I, II, III; `CBT` corticobulbar tract; `CPT_F`, `CPT_O`,
`CPT_P` frontal, occipital and parietal corticopontine tract; `CS_A`,
`CS_P`, `CS_S` anterior, posterior and superior corticostriatal tract;
`DRTT` dentatorubrothalamic tract; `F` fornix (`F_L`, `F_R`); `RST`
reticulospinal tract (rubrospinal in HCP842); `TR_A`, `TR_P`, `TR_S`
anterior, posterior and superior thalamic radiation.

## The atlases of the engine

The engine `tracts` has the atlases of the engine `neurosynth`, Destrieux,
Schaefer and Tian, in MNI152: a tractogram in MNI152 is indexed by them
without loading an atlas, and their labels are named as their relations
name them, so that `label("destrieux", l, n, h)` joins
`destrieux(n, region)`:

```
.load_tractogram("hcp842", kind="hcp842")
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

