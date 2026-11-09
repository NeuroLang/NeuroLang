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
`h` is `"left"`, `"right"` or `"none"`. Each predicate is documented by the
`%%` block before its rule, its signature with the sorts of its arguments
(`traverses(x: streamline, term: term, h: hemisphere)`) and what it means,
which the query assistant and the REPL's `.predicates` show; the engine
`tracts` describes the sorts.

The uncinate fasciculus, tract_querier's `UF.side`, is
[`examples/uncinate.dl`](examples/uncinate.dl):

```
%% uncinate(x: streamline, h: hemisphere)
%% The uncinate fasciculus of hemisphere h: ...
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
NeuroLang, see below), the whole-brain tractogram of the HCP842 atlas
(downloaded, below), and index the one by the other. Edit the paths of the
atlas in the file (they are read relative to where it runs), then, with the
extra `tracts` of `neurolang-apps`:

```sh
neurolang-query --engine tracts --file uncinate.dl     # the ids x
neurolang-query --engine tracts -i --file uncinate.dl  # the REPL, then:
nl(tracts:DL)> .view
```

`.view` draws the streamlines of the answer, coloured by direction, on the
label image of the atlas, in the browser. At the REPL's prompt the commands
and the rules can also be typed or pasted one by one, and other questions
asked of the same relations (`ans(x) :- uncinate(x, "right")`).

nl_tracts is an engine, not an application: `nl_tracts.frontend` has the
solver's mixin of the commands, `TractsCommandsMixin`, and the
`python_init` that imports the vocabulary, `vocabulary.dl`, and adds the
functions of the ids and the hooks of `.view`. The engine `tracts` of the
apps (`engines.yaml`) puts them together, and another engine of the apps has
them by extending it, as `neurosynth_tracts` does (`extends: [neurosynth,
tracts]`).

[`examples/uncinate_hcp842.dl`](examples/uncinate_hcp842.dl) counts the
streamlines of the atlas's own uncinates, `UF_L` and `UF_R`, that the rule
selects: 363 of 370 and 324 of 328 with the `wmparc` of FreeSurfer's
`cvs_avg35_inMNI152`.

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

## tract_querier's queries in the vocabulary

`tools/wmql_to_datalog.py` translates a WMQL program of tract_querier (a `.qry`) into
rules of the vocabulary, one predicate per query, each documented by a `%%`
block, its signature, the full name of the bundle for tract_querier's own
files (`uf` of `freesurfer_queries` is the uncinate fasciculus) and the WMQL
statement it is translated from (DESIGN.md section 17): `UF.side` becomes
`uf(x, h)`, the rule above; a name of labels is a term of the mapping; what the vocabulary
cannot say is left out and listed in the header.

```sh
python tools/wmql_to_datalog.py my_queries.qry \
    -I path/to/tract_querier/data --terms freesurfer_aseg_aparc > my_queries.dl
```

`--terms` checks the names of labels against a mapping (a name that is not
one of its terms stands for those it is made of). The translations of
tract_querier's own files are shipped (`freesurfer_queries`,
`freesurfer_queries_bsf2016`, `freesurfer_queries_bsf2016_toad`,
`mori_queries`) as modules that a program imports after its commands:

```
.import("nl_tracts:bundles/freesurfer_queries.dl")
ans(x) :- af(x, "left")
```

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
options going to it, in MNI152, its names normalised as the engine's
(nilearn's Destrieux has the terms of `freesurfer_destrieux`):

```
.load_atlas("d", kind="nilearn", fetcher="destrieux_2009")
.load_atlas("ho", kind="nilearn", fetcher="harvard_oxford",
            atlas_name="cort-maxprob-thr25-1mm")
.index_tracts("hcp842", "d")
```

