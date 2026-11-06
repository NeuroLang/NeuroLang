# nl_tracts: tractography for NeuroLang

Status: **implemented, from files to the answer**: the atlas layer (`spaces`,
`terms`, `atlas`: sections 3.2, 4, 5, and `label_extent` of 6.1), the tractogram
and its relations (`Tractogram`, `streamline_extent`, `endpoint_coord`,
`streamline_length`), the **voxel indexing** (`streamline_label`,
`endpoint_label`: 6.2), the parquet cache and the `RelationStore` (7.3), the
streamlines identified by **one integer id `x`** (3.1), the commands that
build the relations from files in a Datalog program (8.2), or by the atlases
of an engine of the apps, those of neurosynth (4.2), the **vocabulary
of tractography**,
Datalog rules over this schema in which a bundle is a rule or two
(`traverses(x, term, h)`, `anterior_of(x, term, h)`...: section 16), with the
uncinate as an example, and the engine `tracts` of `neurolang-query` and of
its interactive REPL, whose `.view` draws the streamlines (16.3). Not done:
the relations as a Python call for many atlases at once beyond what
`RelationStore` does, the AAL and Schaefer mappings, the probabilistic
variants (4.4, 12). Numbers below marked
*measured* come from probes run on this repository (NeuroLang `master` at the
time of the split, pandas 3.0.6, pandas backend, an Intel Mac); the rest are
estimates.

Contents: 1 Goals · 2 Decisions taken · 3 Relations · 4 Atlases ·
5 Spaces · 6 Indexing semantics · 7 Data formats and cache ·
8 Handing the relations to the engine · 9 Packaging and layering ·
10 Public API · 11 WMQL (a later PR) · 12 Directional terms ·
13 Test plan · 14 Phases · 15 Open questions ·
16 The vocabulary of tractography

## 1. Goals

`tract_querier` answers questions such as "streamlines that traverse the
precentral gyrus and end in the temporal pole, in either hemisphere" with
its own query language over two indices: label -> streamlines and
streamline -> labels. NeuroLang already has a Datalog with negation,
aggregation and (in `NeurolangPDL`) probabilities. nl_tracts makes
tractography a set of **extensional relations** so that such questions are
Datalog rules, composed with everything else in an engine (Neurosynth terms,
regions, probabilistic facts), and so that the same rule runs on several
atlases.

Non-goals: registration or resampling between spaces; tractography
algorithms; visualisation; probabilistic atlases (4.4); the variants of
the directional terms that are not tract_querier's (12); WMQL, tract_querier's
language (a transpiler to Datalog is a later PR, 11).

## 2. Decisions taken

These are fixed; the sections they affect expand them.

1. **Streamlines are not `Region` objects.** A streamline is an integer id
   inside a tractogram. Tractography enters NeuroLang as extensional
   relations computed in bulk with numpy. Pairwise Python relation
   evaluation (`Region` x `Region`, as `CD_relations.cardinal_relation`
   does) is forbidden on streamline-sized data.
2. **Multi-atlas is first-class.** Every label-dependent relation carries an
   atlas column `a`; several atlases coexist in one engine and a query
   switches atlas by changing a constant. A shared vocabulary,
   `label_term`, makes rules portable across atlases (3, 4.5).
3. **Atlas abstraction**: a deterministic atlas is a label volume + affine +
   label table, with four loaders and configurable hemisphere inference
   (4). Probabilistic (4D) atlases get an explicit extension point, not an
   implementation (4.4).
4. **Spaces**: no registration. Atlases and tractograms carry a space tag;
   combining mismatched tags raises unless explicitly overridden (5).
5. **Indexing semantics match tract_querier**, the equivalence oracle (6).
6. **Directional relations are future work** (12). *Overridden on
   2026-11-03: the directional terms of tract_querier are required, and
   implemented as it has them, in the vocabulary (12, 16); only the variants
   that are not its are future work.*
7. **Layering**: a core that depends only on numpy/scipy/nibabel/pandas
   (and pyarrow for parquet, 15.1), and a thin `nl_tracts.frontend`. The
   package sits between `neurolang_apps` and `neurolang` (9).
8. **Relations can have 10^7-10^8 rows.** DataFrames in memory, parquet on
   disk keyed by a content hash, handed to the engine without building
   Python tuples (7, 8).

## 3. Relations

### 3.1 Schema

Types are the *physical* dtypes of the DataFrame columns. Narrow integers
are safe in the engine (*measured*: `int32`/`int16`/`int8` columns join with
`int64` relations and are selected on Python int constants correctly, and
the engine keeps the dtypes).

| Relation | Columns | Dtypes | Atlas-dependent | Rows |
|---|---|---|---|---|
| `tractogram(t)` | t | str | no | #tractograms |
| `streamline(x, t, s)` | x, t, s | int64, str, int32 | no | S per tractogram |
| `atlas(a)` | a | str | - | #atlases |
| `label(a, l, name, hemi)` | a, l, name, hemi | str, int32, str, str | yes | ~10^2 per atlas |
| `label_term(a, l, term)` | a, l, term | str, int32, str | yes | ~10^2 per atlas |
| `opposite_hemisphere(h1, h2)` | h1, h2 | str, str | no | 3 (left/right, right/left, none/none) |
| `streamline_label(a, x, l)` | a, x, l | str, int64, int32 | yes | ~5-15 per streamline |
| `endpoint_label(a, x, e, l)` | a, x, e, l | str, int64, int8, int32 | yes | 2 per streamline |
| `streamline_extent(x, axis, lo, hi)` | x, axis, lo, hi | int64, int8, float32, float32 | no | 3 per streamline |
| `endpoint_coord(x, e, axis, v)` | x, e, axis, v | int64, int8, int8, float32 | no | 6 per streamline |
| `endpoint_point(x, e, px, py, pz)` | x, e, px, py, pz | int64, int8, float32 x 3 | no | 2 per streamline |
| `label_extent(a, l, axis, lo, hi)` | a, l, axis, lo, hi | str, int32, int8, float32, float32 | yes | 3 per label |
| `streamline_length(x, mm)` | x, mm | int64, float32 | no | 1 per streamline |

`endpoint_point` has the points of `endpoint_coord`, one row per end: a
rule that compares an end with the points of another stored relation by
distance, `EUCLIDEAN(px, py, pz, mx, my, mz) <= r` (the voxels of an atlas
in mm, `schaefer_points(x, y, z, label)` of the engine `tracts`), has both
sides ranged by stored relations, which the engine needs to find the pairs
with a k-d tree instead of comparing every pair. `endpoint_coord` stays, one
coordinate a row, for the directions of the vocabulary.

The points of the atlases are the option `points: {predicate_name: ...}` of
an atlas of an engine of the apps (`engines.yaml`, `doc/
engine_configuration.rst`): the centre of each labelled voxel through the
image's affine, as floats, with the engine's name of its region. The
engines extending `mni_atlases` have `schaefer_points`, `destrieux_points`
and `tian_points` (132,032 + 95,328 + 8,562 rows, 13 MB as DataFrames; the
neurosynth build, 12-13 s and 4.5-4.7 GB at its peak, does not change
measurably, *measured*). The parcels of Schaefer near the ends:

```
near(x, e, n) :- endpoint_point(x, e, ex, ey, ez),
    schaefer_points(mx, my, mz, n), (EUCLIDEAN(ex, ey, ez, mx, my, mz) <= 3)
```

*Measured* on the HCP842 whole-brain tractogram (144,641 streamlines) in
MNI152: 1.8 s for all the ends, 314,733 rows, no comparison of every pair.
Of the ends of the uncinate of `examples/uncinate.dl` (indexed by the
wmparc of `cvs_avg35_inMNI152`), 1,444 of 1,448 on the left and 1,575 of
1,620 on the right are within 3 mm of a parcel (9 and 7 parcels: TempPole
1, OFC 1, PFCv 1, Temp 1 ... on the left; TempPole 1, OFC 1, TempPar 1,
PFCv 1 ... on the right), where `endpoint_label("schaefer", x, e, l)`
gives a parcel to 1,319 and 1,323 of them (5 and 4 parcels): the distance
also finds the ends that stop just outside a parcel, and the parcels
next to the one an end is in.

**A streamline is one id, `x`.** `x = (index << 32) | s`, an int64, where
`s` is the position of the streamline in its file and `index` the load order
of its tractogram in the `RelationStore` (0 for the first, so that with one
tractogram `x == s`); `nl_tracts.ids` has `encode` and `decode`, vectorised.
`streamline(x, t, s)` is the **decoding relation**: a rule that needs the
tractogram or the position joins it (`ans(t, s) :- q_uf_left(x),
streamline(x, t, s)`); a rule that does not never carries them. The engine of
the commands (`tracts_engine()`, 8.2) also has the functions
`streamline_id(t, s)`, `subject_of(x)` (the `t` of `x`) and `index_of(x)`
(its `s`), for a comparison in a rule body: `ans(t) :- streamline_label("fs",
x, 1), (t == subject_of(x))`. Why one id rather than the pair `(t, s)`: a
join on a streamline is on one integer column instead of a string and an
integer, the string `t` is not repeated on every row of the large relations
(`streamline_label` is 10^7 rows, 7.2), and the rules over streamlines are
unary in them, which is what a vocabulary of rules over this schema
(`traverses(x, term, h)`, section 16) is written in. The subject is decoded
when asked. `index` is the load order, not a hash of the id, so the ids of a
tractogram depend on what was loaded before it in the same store: `x` is a
key *within* one store (one engine), not across them; the parquet cache
(7.3) is per tractogram and keeps the positions `s`, and the store makes `x`
when it appends. (The relations were keyed by `(t, s)`, and the two
streamline relations called `traverses` and `endpoint`, until this change.)

Changes with respect to the brief, each of which is a refinement, not a
deviation from a decision:

* **`axis` is an int** (0, 1, 2 = x, y, z of the space's world axes, RAS+)
  rather than a string: 3 to 6 rows per streamline times 10^6-10^7
  streamlines makes the column the largest after `streamline_label`. Constants
  are not a usability problem: the frontend does not need to define them;
  rules write `axis = 1` and the vocabulary is in the documentation.
* **`streamline_length(x, mm)` is added.** tract_querier's
  `length_threshold` is applied while indexing; for a query that is made of
  rules (`long(x) :- streamline_length(x, d), (d > 40)`) the length has
  to be a relation. The *index* parameter `length_threshold` remains, to
  keep out of every relation the streamlines nobody will query (6.1).
* **`label_extent` and `streamline_extent` are in world mm.** `label_extent`
  is by voxel *edges*, from the box of each label, see 6.1 for why and how
  it relates to the reference; `streamline_extent` is the box of the points.
* `opposite_hemisphere` is `{(left, right), (right, left), (none, none)}`:
  the brief leaves it open; `none` is its own opposite so that rules over
  midline structures keep working.

`tractogram`, `atlas`, `label` and `label_term` are the *catalogue*
relations: they say what exists, and are what makes the engine
introspectable from a prompt (a text-to-query assistant reads the schema
from them).

### 3.2 The term vocabulary

`label_term(a, l, term)` maps each atlas's raw labels to a shared
vocabulary, **many-to-many**: a label may have several terms
(`precentral` and `frontal`) and a term covers labels of several atlases.
The hemisphere is *not* in the term: it lives in `label(a, l, name, hemi)`,
so that `precentral` is the same term for the left and the right label, and
rules choose hemispheres explicitly. A test checks that no shipped term has
a hemisphere token (`lh`, `rh`, `left`, `right` as a word, `_L`/`_R` suffix).

**The vocabulary is tract_querier's**, from its
`tract_querier/data/freesurfer_queries.qry`: the Desikan-Killiany names in
lower case (`precentral`, `temporalpole`, `lateralorbitofrontal`, ...), and
the groups that file defines as unions of them: `frontal`, `temporal`,
`parietal`, `occipital`, `cingular`, `orbitofrontalgyrus`,
`middle_frontal_gyrus`, `inferior_frontal_gyrus`,
`superior_parietal_lobule`, `inferior_parietal_lobule`, `subcortical`,
`cerebellum`, `ventricles`, `corpus_callosum`, and `striatum`,
`temporal_frontal`, `cingular_cortex`, `frontal_frontal_section`. Reusing it
makes the standard bundle definitions of the phase-3 tutorial a
transcription of tract_querier's queries (`UF.side` ...). Each Desikan name
is made as the file makes it: **the cortex and the white matter under it**
(`precentral` is `ctx-lh-precentral` and `wm-lh-precentral`), with
`ctx_precentral` and `wm_precentral` for either alone. The groups are the
file's, with **one deliberate difference**: `subcortical` is completed with
the caudate, hippocampus and accumbens, which the file lacks (it has them
in `striatum`, in the first case). The file's `.side` becomes `label.hemi`.
Rules over the terms, for instance:

```
% uncinate fasciculus, any atlas, either hemisphere
uf(a, x, h) :-
    streamline_label(a, x, l1), label_term(a, l1, "temporalpole"),
        label(a, l1, _n1, h),
    streamline_label(a, x, l2), label_term(a, l2, "orbitofrontalgyrus"),
        label(a, l2, _n2, h)
% the two ends in opposite hemispheres
callosal(a, x) :-
    endpoint_label(a, x, 0, l1), label(a, l1, _n1, h1),
    endpoint_label(a, x, 1, l2), label(a, l2, _n2, h2),
    opposite_hemisphere(h1, h2)
```

(The surface syntax of these rules, including `_`-prefixed anonymous
variables, negation `~` and `AGGREGATE`, is that of
`neurolang/frontend/datalog/standard_syntax.py`; the test plan checks
every rule of this document parses, 13.)

**Mapping file format** (`nl_tracts.terms`): a TSV with the columns
`atlas_label_name`, `term` and optionally `hemi`; `#` lines and blank lines
are ignored. `atlas_label_name` is the **raw** name as the atlas has it
(`ctx-lh-precentral`, `L G_precentral`); a name has one row per term;
`hemi` is `left`, `right`, `none` or empty. A mapping is rejected on a
missing column, an empty name or term, a bad `hemi`, a duplicate
(name, term), or two rows of one name with different `hemi`. There is no
default term for an atlas without a mapping: `label_term` is then empty and
every label is reported by `Atlas.unmapped_labels()`, which returns the
atlas's labels (background excluded) that have no term, to see what a
mapping leaves out.

A mapping is passed to `Atlas(..., term_mapping=...)` as a DataFrame, a
path or the key of a shipped one. Shipped, in `nl_tracts/mappings`, all
generated by `tools/make_freesurfer_mappings.py` (which holds the compact
tables; edit that, not the TSVs):

| Key | Labels |
|---|---|
| `freesurfer_aseg_aparc` | aseg (both `Left-Thalamus` and `Left-Thalamus-Proper`, the LUT's two spellings) and aparc, as `ctx-lh-*`/`ctx-rh-*`; vocabulary and groups from the `.qry` |
| `freesurfer_destrieux` | a2009s, in **both** naming schemes: FreeSurfer's LUT (`ctx_lh_G_precentral`) and nilearn's lateralised atlas (`L G_precentral`), because the mapping is keyed by raw name |
| `tian_subcortex_s2` | Tian's subcortex at scale 2, the 32 names of its label file (`aHIP-rh`, `THA-DA-lh`, `NAc-shell-lh`, `pGP-rh`, ...): each region has the terms of the aseg structure it is part of, exactly those of `freesurfer_aseg_aparc` (`hippocampus`, `amygdala`, `thalamus_proper`, `accumbens_area`, `pallidum`, `putamen`, `caudate`, with `subcortical`, and `striatum` for the last three) |

The Destrieux assignments are a **draft to be reviewed by an anatomist**:
the parcellation has no one-to-one Desikan equivalent, so a label gets a
Desikan name only where it is the obvious one, and a lobe otherwise. The
boundary structures (lateral fissure, central sulcus, parieto-occipital
sulcus) have the terms of **both** regions they separate (`S_central`:
`precentral`, `postcentral`, `frontal`, `parietal`); only the medial wall
and `Unknown` have none, on purpose, and are what `unmapped_labels` reports
(15.13). Against a real FreeSurfer LUT (skipped if there
is none) a test checks that every shipped FreeSurfer name exists and that
the standard aseg, aparc and a2009s labels are all mapped, except those
left out on purpose; against nilearn's Destrieux, that the unmapped labels
are exactly those.

An atlas of an engine of the apps (4.2) names its labels as the engine
does, its raw names with `-` and `_` as spaces (`L G and S frontomargin`,
`aHIP rh`); its shipped mapping, keyed by the raw names, is matched on the
names normalised the same way (`frontend.engine_label_name`, a copy of the
apps' `_label_name`, which nl_tracts cannot import; a test of the apps
checks that they agree on every name of the mappings).

## 4. Atlases

### 4.1 Model

`Atlas(name, labels, affine, table, space, background=0, hemisphere_rules,
term_mapping)`: a 3D integer volume, the 4x4 affine from voxel indices to
the space's world mm (RAS+), a label table (`id`, `name`, optional `hemi`),
the space tag, the id that means "no region", the hemisphere rules and the
term mapping. It produces the catalogue relations (`label`, `label_term`,
`label_extent`) and a content hash. It never touches a tractogram: indexing
is in `indexing.py` (6).

An atlas `name` is unique in an engine and is the value of the column `a`.
Ids that are in the table but not in the volume, and ids in the volume but
not in the table, are reported with a warning that names them: the former
are dropped from `label` (nilearn's Destrieux has two, the medial walls),
the latter are kept with `name = "label_<id>"` and `hemi = "none"`, since
`streamline_label` would otherwise refer to an unknown label. The background has a
row in `label` only if the table has one, and none in `label_extent`. Labels
must be non-negative integers (a float volume with integral values is
accepted, read through `dataobj`, never `get_fdata`); a 4D image is a
probabilistic atlas (4.4) and raises `NotImplementedError`. Attributes:
`name`, `space`, `labels`, `affine`, `background`, and `table`, the
resolved label table (`id` int32, `name`, `hemi`).

### 4.2 Loaders

| Loader | Input | Notes |
|---|---|---|
| `Atlas.from_freesurfer` | label image + FreeSurfer LUT (`FreeSurferColorLUT.txt`) | `load_label_table_freesurfer_lut` skips comments; ids and names only, colours are ignored |
| `Atlas.from_bids_dseg` | label image + `*_dseg.tsv` | columns `index`, `name` |
| `Atlas.from_nilearn` | the object a nilearn fetcher returns | **duck-typed**: `.maps` (path or image) and `.lut` (a DataFrame with `index`, `name`) if present, else `.labels`, the id of a name being its position; bytes are decoded; nilearn is not imported (decision 7 forbids it in the core). Only the tests fetch atlases with nilearn, behind the `network` marker: a fetch that fails skips the test |
| `Atlas.from_tsv` | label image + plain TSV | columns `id`, `name`, optional `hemi` |

All accept a path or an already loaded nibabel image, and forward keyword
arguments (`background`, `hemisphere_rules`, `term_mapping`) to `Atlas`.
The nilearn convention that `labels[i]` is the name of id `i` (with a
leading `Background`) is the loader's, not the core's.

**The atlases of an engine.** An engine of the apps that extends
`mni_atlases` (`engines.yaml`: `neurosynth`, `neurostore`, `tracts`) has
loaded Destrieux, Schaefer and Tian, each as a relation `destrieux(name,
region)`, the label image in `engine.grids[name]` and the values of each
region in `engine.atlas_labels[name]` (a name -> its values).
`frontend.engine_atlas(engine, name)` makes an `Atlas` of it, by
`from_nilearn`, in the space `MNI152`, its names exactly the engine's (so
that `label("destrieux", l, n, h)` joins `destrieux(n, r)` and, in
neurosynth, `destrieux_activated(n, s)`), with the shipped mapping of
`ATLAS_MAPPINGS` (`destrieux`: `freesurfer_destrieux`, `tian`:
`tian_subcortex_s2`; Schaefer has none, its labels are then without a term,
as an atlas without a mapping). `.index_tracts(t, "destrieux")` asks for it
when the program loaded no atlas of that name (8.2).

**The atlases of nilearn, by name.** `frontend.nilearn_atlas(name,
fetcher, space="MNI152", term_mapping=None, background=0, **options)`
calls `nilearn.datasets.fetch_atlas_<fetcher>(**options)` (nilearn is
imported there, lazily; a missing nilearn is an error that says so) and
makes the atlas the same way, `frontend.mni_atlas`: the label names
normalised as the engine's, so that nilearn's Destrieux and the engine's
`destrieux` have the same names (and the first a row `Unknown`, its
background, which the engine drops), the term mapping the one given, else
that of the fetcher in `ATLAS_MAPPINGS` (`destrieux_2009`:
`freesurfer_destrieux`), the hemispheres by the rules (4.3). It is the
command `.load_tract_atlas(name, kind="nilearn", fetcher=...)`.

### 4.3 Hemisphere inference

A `HemisphereRule(pattern, hemi)` has a regular expression that must match
the **whole** raw name, and `hemi` `left` or `right`; the rules are ordered
and the first match wins; `infer_hemispheres(names, rules)` returns a Series
of `left`, `right` or `none`. Defaults, `HEMISPHERE_RULES` (the first five
are the brief's; the last three are what real atlases needed):

| Convention | Left | Right | Seen in |
|---|---|---|---|
| `Left-`/`Right-` (or a space) | `Left[- ].+` | `Right[- ].+` | FreeSurfer aseg; Harvard-Oxford (`Left Thalamus`) |
| `ctx-lh-` / `ctx_lh_` | `ctx[-_]lh[-_].+` | `ctx[-_]rh[-_].+` | FreeSurfer aparc; a2009s |
| Suffix `_L`/`_R` | `.+_L` | `.+_R` | AAL |
| Prefix `lh.`/`rh.` | `lh\..+` | `rh\..+` | |
| Suffix `.left`/`.right` | `.+\.left` | `.+\.right` | |
| `L `/`R ` | `L .+` | `R .+` | nilearn's Destrieux |
| Suffix `-lh`/`-rh` (or a space) | `.+[- ]lh` | `.+[- ]rh` | Tian's subcortex (`aHIP-lh`); as an engine of the apps names it (`aHIP lh`) |
| `<n>Networks_LH_` (or spaces) | `\d+Networks[_ ]LH[_ ].+` | `\d+Networks[_ ]RH[_ ].+` | Schaefer 2018 (`17Networks_LH_VisCent_ExStr_1`; `17Networks LH ...` in an engine of the apps) |

No match: `none`. `hemisphere_rules=()` infers nothing, and rules given
*replace* the defaults. **Precedence** for the `hemi` of a label: the
`hemi` of the label table if not empty (an explicit `none` is a decision);
else the `hemi` of the term mapping for that raw name if not empty; else the
inference. A warning names each label for which two of these sources
disagree (the inference counting only when it is not `none`). Known hazards,
each covered by a test: the `_L`/`_R` suffixes match any name that ends that
way; matching is case-sensitive; `Leftover`, `Rightmost_area` and
`Lateral-Ventricle` are *not* hemispheric.

### 4.4 Extension point: probabilistic atlases

A `ProbabilisticAtlas` (4D volume: one probability map per label) is a
class in `atlas.py` whose constructor raises `NotImplementedError`. It will
share the label table, hemisphere rules, term mapping and space tag with
`Atlas`, and produce a *probabilistic* relation `streamline_label_p(a, x, l)`
with the probability for `NeurolangPDL.add_probabilistic_facts_from_tuples`
(first column = probability). Open design questions, deliberately not
answered now: whether the probability of traversal is a function of the
fraction of points weighted by the map, or of the maximum along the
streamline; independence assumptions between the labels of one streamline
(a probabilistic *choice* per point, not a fact per label, is the sound
model for a partition of unity).

**Engine finding** (*measured*): `NeurolangPDL.add_probabilistic_facts_from_tuples`
**does not accept a DataFrame**: the frontend renames the columns to
positions and `DatalogProgram.infer_iterable_type` then takes the first
*element of the iterable*, which for a DataFrame is a column label, giving
`AttributeError: type object 'int' has no attribute '__origin__'`. Lists of
tuples work. This is an engine bug, outside this task and not worked around
here; it must be fixed in `neurolang`/`nl_core` before probabilistic atlases
(and noted in 15.7).

## 5. Spaces

nl_tracts does no registration. `Space` is a string tag
(`"MNI152NLin2009cAsym"`, `"subject:100307"`); two tags are the same space
iff equal. Atlas and tractogram each carry one; `RelationStore.add_index`
calls `check_same_space`, which raises `SpaceMismatchError` unless
`allow_space_mismatch=True`, in which case it warns. The override is for
"I know they are aligned"; nl_tracts never infers it. The relations that do
not depend on an atlas (`streamline_extent`, `endpoint_coord`,
`streamline_length`) carry no space and are not checked.

Coordinates: a tractogram's points are in mm, RAS+, in the space's world
coordinates, the frame in which the atlas affine maps voxels. Loading
(`Tractogram.from_file`) converts to RAS+ mm with nibabel (`to_rasmm`);
tract_querier takes the points as they are in the file, which is the same
for files already in RAS+ and **differs** for LPS files (VTK from some
tools): the equivalence tests use RAS+ inputs (13.2). The space tag is
recorded in the cache key (7.3) and in the `tractogram` relation's
metadata only; a rule cannot select by space, by design: switching the
space is switching the atlas *and* the tractogram, which are different
constants.

## 6. Indexing semantics

### 6.1 The equivalence oracle

Reference: `tract_querier/tract_label_indices.py`
(`compute_tract_label_indices`, `compute_label_crossings`,
`compute_label_endings_start_end`). Parity points, **each of which must
have a test** (13.2):

| # | Reference behaviour | nl_tracts |
|---|---|---|
| 1 | Points mapped by `affine_ras_2_ijk = inv(affine)`, `np.round(...).astype(int)` | same, with `np.round`, which rounds **half to even** (not `floor(x + .5)`) |
| 2 | Points outside the volume: one `warnings.warn`, then each coordinate `clip(0, shape - 1)` | same; the warning states how many points were clipped |
| 3 | `point_labels = img[ijk]`, labels used as indices of `np.bincount` | integer labels; **negative labels** are not supported by the reference (bincount fails); we raise a clear error |
| 4 | Streamline `i` traverses `l` iff `count_l / n_points >= threshold / 100.`, `threshold` = **2.0** by default (the CLI's), and the denominator is **all** the points, label 0 included | `crossing_threshold`, default 2.0; the same float expression `count / n >= threshold / 100.` so that the comparison is bit-identical |
| 5 | The result includes label **0** when >= threshold of the points are in the background | see 6.3 |
| 6 | Endpoint labels are the labels of the first and of the last point, as `ending_tracts_labels[0|1]`, label 0 included | `endpoint_label(a, x, e, l)`, `e` = 0 first, 1 last; **rows with the background label are never emitted** (6.3), so the equivalence tests drop label 0 from the reference's endpoints before comparing |
| 7 | With `length_threshold > 0` the tract list is **filtered before enumeration**, so the ids of the survivors are *renumbered* | **ids are never renumbered**: `s` is the position in the file (and `x` encodes it, 3.1); filtered streamlines are absent from every relation. The equivalence test maps the reference's ids back through the kept-index list. **The reference's `tract_bounding_boxes` and `tract_endpoints_pos` are computed on all the tracts, unfiltered, so its directional terms are misaligned with the renumbered ids when `length_threshold > 0`**: not comparable there |
| 8 | Length = sum of segment norms of the polyline, `>= length_threshold` keeps | same, in float64, then stored as float32 in `streamline_length` |
| 9 | Streamlines with fewer than 2 points make the bounding-box step raise | `Tractogram` rejects them at construction, with the id of the first offender; with `drop_short=True`, those of one point are kept in its arrays and are in no relation, as the streamlines under `length_threshold` (parity 7: the ids are not renumbered), and none still raises (15.37) |
| 10 | `compute_label_bounding_boxes`: `ndimage.find_objects` (labels start at 1, `stop` exclusive), then **two** corners `[start, stop]` mapped by the linear part and the translation, `BoundingBox` takes their min and max. That is the centre of the first voxel to the centre of the voxel *after* the last (half a voxel off at both ends), and a box only if each world axis depends on one voxel axis (wrong for oblique affines) | `label_extent` (implemented, 6.1.1): the same `find_objects` and the same "box of a box" idea, by **voxel edges** `[start - .5, stop - .5]`, **8 corners** through the full affine, min and max per world axis. Tests transcribe the reference and check the relation exactly (below) |
| 11 | `tract_endpoints_pos` = first and last point | `endpoint_coord`, float32 (the reference holds float64; tests compare with `atol` 1e-4 mm) |
| 12 | At a `crossing_threshold` of 0 every label up to the largest the streamline has is kept, whether or not a point is in it (`bincount` has a zero for it, and `0 >= 0`) | **not reproduced**: a label is traversed only if at least one point is in it; the tests of parity use thresholds above 0 |

#### 6.1.1 `label_extent`: voxel edges, not centres

*Choice.* A label whose first and last voxel are `start` and `stop - 1`
along a voxel axis spans `[start - 0.5, stop - 0.5]` in voxel units, the
physical extent of its voxels (nibabel's affine maps an index to the
**centre** of its voxel). A one-voxel label has the extent of one voxel,
not zero; the extent is what a directional relation (12) wants to compare a
streamline against. The alternative, voxel centres `[start, stop - 1]`, is
the extent of the voxel *centres*: shorter by one voxel, zero for a single
voxel.

*Consistency with the reference*, tested by transcribing
`compute_label_bounding_boxes`: the reference maps `[start, stop]`, which is
`[start - 0.5, stop - 0.5]` shifted by `+0.5` voxel at both ends, so for
every affine in which each world axis depends on one voxel axis (aligned,
flipped by negative scales, or permuted) `label_extent` equals the
reference's box shifted by `affine[:3, :3] @ (-.5, -.5, -.5)`, exactly, and
equals a brute force over every corner of every voxel. Flipped axes: both
implementations take min and max, so `lo < hi` always.

*Oblique affines.* The reference maps two corners, which is not a box of
anything. nl_tracts maps the 8 corners of the label's box, which gives the
world box of the *box*: it **contains** the extent of the label's voxels and
is larger than it when the label is not box-shaped (the tests have a
diagonal line to show it, and check both the containment and the equality
with the 8-corner brute force). Tightening it needs the voxels, not the
box: a pass over all the voxels of the atlas (the earlier design) instead of
one `find_objects`; not done, because atlases are in practice aligned or
nearly, and the loose box is safe for the uses planned (a conservative
bound). Open question 15.13.

### 6.2 Algorithm (all vectorised)

Given `P` points with a streamline id each (`np.repeat(arange(S), lengths)`):

1. `ijk = points_to_voxels(points, affine, shape)`: one matrix product on the
   whole `(P, 3)` array, `np.round`, clip, count clipped points. Chunked over
   blocks of streamlines of at most `2^24` points to bound the temporary
   `(P, 3)` float64 and int64 arrays.
2. `lab = labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]`.
3. `streamline_label`: build `key = sid * (L + 1) + dense_label`, where
   `dense_label` ranks the atlas's labels (the table order, background
   first) so that `L` is the number of labels, not the largest id;
   `np.unique(key, return_counts=True)` per chunk; divide the counts by
   `lengths[sid]`; keep `count / n >= threshold / 100.`; decode `sid` and
   label. If `S * (L + 1) < 2^28` a `np.bincount` over the same key is used
   instead of the sort. Complexity O(P log P), O(P) memory per chunk.
4. `endpoint_label`: `lab[offsets[:-1]]` and `lab[offsets[1:] - 1]`: two fancy
   indexings.
5. In the store, `x = (index << 32) | s` replaces the columns `t` and `s` of
   each relation of a tractogram: one shift and one or over the column.
6. Streamline extents: `np.minimum.reduceat` / `np.maximum.reduceat` over
   the points at `offsets[:-1]`, per axis; lengths: `np.add.reduceat` of the
   segment norms, with the last segment of each streamline masked out.
7. `label_extent`: one `ndimage.find_objects` for all the labels, the
   `(L, 3)` starts and stops as arrays, the 8 corners of every box mapped by
   the affine in one product, min and max (6.1.1). The only Python loop is
   the unpacking of the slices `find_objects` returns, one per label.

No Python loop over streamlines or over labels; the only loops are over
chunks of points and over the three axes.

### 6.3 Label 0 and the background

The reference treats 0 as a label like any other: it counts in the
denominators, can be traversed, and can be an endpoint label. nl_tracts:

* Points in the background **always count in the denominator** (parity 4).
* **`endpoint_label` never carries the background label** (decision of the
  maintainer, not a flag): an endpoint whose point is in the background has
  *no row* for that end. "This end is in no region" is the safe negation
  `no_region(a, x, e) :- atlas(a), streamline(x, t, s), end_index(e), ~has_end(a, x, e)`
  with `has_end(a, x, e) :- endpoint_label(a, x, e, l)`; `end_index(e)`
  is `{0, 1}` (a two-row relation in the rule, or `e = 0`/`e = 1` written
  out).
* `streamline_label` has rows for the background label **only if
  `IndexingParams.emit_background=True`**. Default is **False**: with 0
  emitted, almost every streamline has a `(a, x, 0)` row and
  `streamline_label` grows by a factor of 1.1-1.3 for facts no rule should
  use. `emit_background=True` reproduces the reference's
  `crossing_tracts_labels` and is what the equivalence tests use; it has no
  effect on `endpoint_label`.
* `label` has a row for the background only if the label table has one (the
  FreeSurfer LUT does: `0 Unknown`).
* tract_querier's `not`/`-` operate on the set of labels that appear in
  the data, which includes 0 in the reference; a rule with negation states
  which label set it ranges over (`only_in`, 16.2, is over every label the
  streamline goes through).

## 7. Data formats and cache

### 7.1 In memory: DataFrames

The relation is a `pandas.DataFrame` with the columns and dtypes of 3.1. The
reasons it is not a CSR matrix:

* the engine's relational algebra sets are DataFrames (*measured*: 8.1) and
  would convert a CSR into one anyway, with the same memory;
* a CSR (streamline x label) matrix is the natural *internal* structure of
  step 3 of 6.2 (`scipy.sparse.csr_array` from the unique keys) and may be
  used inside `traverses_relation`; it is not part of the API;
* the three atlas-independent relations and `endpoint_label` are dense per
  streamline, and flat arrays are what they are.

The core never builds tuples, sets or dicts per streamline.

### 7.2 Sizes

For 10^6 streamlines of 100 points, a 100-label atlas, 10 traversed labels
per streamline on average:

| Relation | Rows | Bytes/row | Memory |
|---|---|---|---|
| `streamline_label` | 10^7 | 10 (a) + 8 (x) + 4 = 22 | 220 MB |
| `endpoint_label` | 2x10^6 | 10 + 8 + 1 + 4 = 23 | 46 MB |
| `streamline_extent` | 3x10^6 | 8 + 1 + 4 + 4 = 17 | 51 MB |
| `endpoint_coord` | 6x10^6 | 8 + 1 + 1 + 4 = 14 | 84 MB |
| `streamline` | 10^6 | 8 + 10 (t) + 4 = 22 | 22 MB |

(Computed from the dtypes; with `(t, s)` for a streamline, before the id `x`
of 3.1, the four were 280, 58, 69 and 120 MB.)

**String columns.** The `a` and `t` columns are pandas' default string
dtype. *Measured* (pandas 3.0.6): a constant string column of 3x10^6 rows is
`str` dtype, Arrow-backed, **30 MB** (10 bytes a row); the same column
forced to `object` is 153 MB (a pointer, a string object, and the
bookkeeping); before pandas 3, `object` is the default and is what the
pointer arithmetic of an earlier version of this section described.
**A correction to the earlier measurement**: the `object` column of the
probe (3x10^6 rows, join of `traverses` with `label_term`: `int` atlas
column 719 MB peak RSS / 0.95 s, 841 MB / 1.12 s for the "object" column,
751 MB / 1.43 s for an explicit `string[pyarrow]`) was, under pandas 3, the
Arrow `str` dtype too, since a DataFrame built from an object array of
strings converts it. So the 841 MB figure is for `str`, and the engine runs
joins and constant selections on it without trouble. **Decision: keep
strings** for `a` and `t`: queries switch atlas by editing a literal
(decision 2), and surrogate integer keys would need a lookup relation in
every rule. Since streamlines are the id `x` (3.1), `t` is only in
`tractogram` and `streamline`, one row per streamline, and not in the large
relations; `a` is. At 10^8 rows the `a` column is about 1 GB under either
dtype and an integer would save most of it: revisit then (15.3).
`Categorical` is not used because the engine's behaviour on categorical
columns is untested. The code does not choose the dtype: it builds
DataFrames from Python strings and lets pandas decide, so that it behaves
the same on pandas 1.5 (`object`) and 3 (`str`); the tests assert a string
dtype, not the identity of the objects.

### 7.3 On disk: parquet, keyed by content

Only the **atlas-dependent relations that are expensive** are cached:
`streamline_label` and `endpoint_label`. The others are `O(S)` numpy
reductions that cost less than reading a file.

```
<cache_dir>/<key>/traverses.parquet       # streamline_label: s, l
<cache_dir>/<key>/endpoint.parquet        # endpoint_label: s, e, l
<cache_dir>/<key>/meta.json     # params, nl_tracts version, INDEXING_VERSION, row counts
```

An entry is of one tractogram and one atlas, so it holds the position `s`,
not `x`, which depends on the load order of the store (3.1): the store adds
`a` and makes `x` when it reads it. (The files keep the names of the
relations as they were when the cache was written; renaming them would only
invalidate the caches.)

`key = sha256(tractogram.content_hash() ‖ atlas.content_hash() ‖
repr(params) ‖ INDEXING_VERSION)`, hex, first 32 characters.
`Tractogram.content_hash` hashes the points (as float32 bytes), the offsets
and the space tag; the `id` is **not** in it, so two runs of the same file
under different ids share a cache entry, and `x` is made from the load order
when the entry is read. `Atlas.content_hash` hashes the volume, the affine,
the label table, the background id and the space tag, not the atlas name, for
the same reason. `INDEXING_VERSION` (in `cache.py`) is bumped whenever the semantics
of 6 change. Files are written to a temporary name and renamed, so an
interrupted run never leaves a half file that is later trusted; zstd,
row groups of 10^6, dictionary encoding on the string columns (a column of
one repeated string is a few bytes on disk).

Reading is `pandas.read_parquet` with the dtypes of 3.1 restored (parquet
stores `int32`/`int8`/`float32` natively); the `a` column is added back in
pandas' default string dtype (7.2) and `x` made from `s`, so the memory of
7.2 holds after a cache hit.

### 7.4 Chunking

`traverses_relation` and `endpoint_relation` process the tractogram in
chunks of streamlines (about 2^24 points) and concatenate, so the peak memory
is the result plus a chunk, not the result plus several copies of the
points. Chunking is internal; the result is one DataFrame.

### 7.5 Downloaded atlases of tractography

`nl_tracts.fetch` (standard library only, a module of the core) downloads
two public atlases of F.-C. Yeh, whose bundles were labelled by experts:

| | `hcp842` | `hcp1065` |
|---|---|---|
| Paper | Yeh et al. 2018, NeuroImage 178:57-68 | Yeh 2022, Nat Commun 13:4933 |
| Archive | `Atlas_80_Bundles.zip`, 319,525,089 bytes, md5 `78331d527a10ec000d4f33bac472e099` | `hcp1065_avg_tracts_trk.zip`, 587,869,457 bytes, md5 `0b4530ea43ea22394f3fecd301e0b7c4` (measured: none is published) |
| URL | `https://ndownloader.figshare.com/files/13638644`, then DIPY's mirror `https://workshop.dipy.org/services/data/files/13638644` | `https://github.com/data-others/atlas/releases/download/hcp1065/hcp1065_avg_tracts_trk.zip` (release `hcp1065` of 2024-09-25) |
| Bundles | 78, `bundles/<stem>.trk` (79 files: `IF0F_R.trk` is a copy of `IFOF_R.trk`; `F_L_R` is both fornices) | 87, `<group>/<stem>.trk.gz` |
| Whole brain | `whole_brain/whole_brain_MNI.trk`, 144,678 streamlines, 37 of one point | none |
| Space | MNI152, ICBM 2009a (the zip's README) | MNI152, ICBM 2009a nonlinear asymmetric, 1 mm |
| Licence | CC BY 4.0, "(c) Eleftherios Garyfallidis", the curation for DIPY's RecoBundles (figshare 10.6084/m9.figshare.7375883); cite Garyfallidis et al. 2017 (NeuroImage 170:283-297) and Yeh et al. 2018. Yeh's own distribution of HCP842 (brain.labsolver.org) is CC BY-SA 4.0 | CC BY-SA 4.0 (brain.labsolver.org/hcp_trk_atlas.html); cite Yeh 2022; data of the Human Connectome Project, whose terms ask for an acknowledgement |

`fetch_tractogram(atlas="hcp842", bundle=None, data_dir=None, url=None,
md5=None) -> Path` keeps the archive in `<data_dir>/<atlas>/`, by default
`~/.cache/nl_tracts`, and extracts next to it only what is asked:
`<stem>.trk` of a bundle (a `.trk.gz` gunzipped, as nibabel does not read
it by its name) or the whole brain. The download is streamed to a temporary
name, its md5 checked, then renamed; a URL that fails or gives another md5
is followed by the next (figshare's redirection is to a signed URL valid
for 10 seconds, so the one `urlopen` follows it). An extracted file present
is returned without the archive; a missing one is extracted from the
archive kept. `url=` replaces the URLs, its checksum then `md5=` (not
checked if None); a `file://` URL is how the tests serve small archives.
Bundles are named by the stem of their file, in any case; the full names
are in `ATLASES[atlas]["bundles"]` (README). HCP1065's DSI Studio `.tt.gz`
are not read by nibabel and are not used; its archive of `.trk.gz` is.
Neither DIPY nor nilearn is imported.

On this Mac (measured): the download of HCP842 from figshare and its md5,
10 s; the extraction of the whole brain, 0.7 s, of a bundle, < 0.1 s; the
loading of the whole brain, 1.7 s.

## 8. Handing the relations to the engine

### 8.1 Finding: `add_tuple_set` accepts a DataFrame, efficiently

*Measured*, `NeurolangDL`, pandas backend:

* `QueryBuilderDatalog.add_tuple_set` (`frontend/query_resolution_datalog.py`)
  special-cases `pd.DataFrame`: it renames the columns to `0..n-1` and
  passes the frame to `DatalogProgram.add_extensional_predicate_from_tuples`
  -> `new_set` -> `WrappedRelationalAlgebraSet`; `NeurolangPDL` inherits it
  (the **probabilistic** variants do not, 4.4).
* The base class's `QueryBuilderBase.add_tuple_set`
  (`query_resolution.py`) is the generic iterable path
  (`_add_tuple_set_elements`); it is **not** what the Datalog engines use.
* `RelationalAlgebraFrozenSet.__init__` takes the frame with **one
  `DataFrame.copy()`** and no iteration over rows; `verify_row_type` is off.
  No Python tuple is created. 10^4 rows: 4 ms; **2x10^6 rows, 4 columns:
  26 ms** (`NeurolangPDL`, 10^6 rows with a string column: 94 ms).
* The dtypes are preserved (`int32` stays `int32`), the row type is
  inferred from the first row (`Tuple[int, int, int, int]`).
* A rule over a 2x10^6-row `traverses(a, t, s, l)` (now `streamline_label`)
  joined with `label_term`, with negation, ran in **2.1 s, 436 MB** peak RSS; at 3x10^6 rows a join
  0.95-1.4 s.
* The one `copy()` doubles the footprint momentarily. If it matters,
  the fix is in `RelationalAlgebraFrozenSet.__init__` (adopt the frame
  without copying when the caller says it relinquishes it); a candidate
  engine change, not part of this task (15.4).

So **the frontend passes DataFrames**, and the core never needs to know the
engine.

### 8.2 The frontend module

`nl_tracts.frontend.register(engine, store, only=None)` calls
`engine.add_tuple_set(frame, name=relation_name)` for each relation of a
`RelationStore`. It is a function, not an engine subclass: a subclass would
be a parallel class hierarchy and would not compose with `NeurolangPDL`
and the other engines. The column names of 3.1 are carried by
`RELATION_NAMES` for documentation and for rule authors; `add_tuple_set`
renames columns to positions, so the engine itself does not hold them
(rules bind by position).

Because `add_tuple_set` **replaces** the symbol, several atlases or
tractograms must be concatenated *before* registration: that is the
`RelationStore`'s job (`add_atlas`, `add_tractogram`, `add_index` accumulate
into one frame per relation), and `register` is called once. Registering
again replaces everything: there is no incremental add in phase 1 (15.5).
`add_tractogram` gives the tractogram its index in the ids `x` (the number of
tractograms added before it, kept in `RelationStore.tractograms`), so it comes
before `add_index`, which raises otherwise.

```python
store = RelationStore()
store.add_atlas(Atlas.from_freesurfer("dk", img, lut, space="subject:100307"))
store.add_atlas(Atlas.from_bids_dseg("aal", img2, tsv, space="subject:100307"))
tg = Tractogram.from_file("100307", "tracts.trk", space="subject:100307")
store.add_tractogram(tg, params)
for atlas in atlases:
    store.add_index(tg, atlas, params, cache_dir="~/.cache/nl_tracts")
nl = NeurolangDL()
register(nl, store)
nl.execute_datalog_program(
    'ans(s) :- streamline_label("dk", x, l), streamline(x, "100307", s), ...')
```

#### The commands: the second hand-off path

A Datalog program can also build the relations itself, from files, with three
commands that `TractsCommandsMixin` handles (`nl_tracts.frontend`):

```
.load_tract_atlas("fs", "aparc+aseg.nii.gz", "FreeSurferColorLUT.txt",
                  kind="freesurfer", space="subject:1",
                  term_mapping="freesurfer_aseg_aparc")
.load_tractogram("100307", "tracts.trk", space="subject:1")
.index_tracts("100307", "fs", crossing_threshold=2.0, length_threshold=0.0,
              emit_background=1, cache="~/.cache/nl_tracts")
```

A tractogram of the atlases of 7.5 is loaded by its kind, downloaded the
first time, in `MNI152` unless `space` is given; `bundle` is the stem of a
bundle, else it is the whole brain; `data_dir`, `url` and `md5` are those
of `fetch_tractogram`, the data directory by default that of the engine,
`<data_dir>/tracts` (the `--data-dir` of `neurolang-query`, which
`init_engine` is given), or `~/.cache/nl_tracts` for an engine without one:

```
.load_tractogram("hcp842", kind="hcp842")
.load_tractogram("uf_l", kind="hcp842", bundle="UF_L")
.load_tractogram("af_l", kind="hcp1065", bundle="AF_L", data_dir="data")
```

Their streamlines of one point are in no relation (`drop_short=1`, the
default of a kind; 15.37), with a warning; `drop_short=1` does the same for
a file.

`kind` is `freesurfer` (a colour LUT), `bids` (a `dseg.tsv`) or `tsv`. The
paths are read as they are written (relative to where the program runs).
An atlas of nilearn is loaded by the name of its fetcher, without files
(4.2); the options of the command other than its own go to the fetcher,
and the space is `MNI152` unless given:

```
.load_tract_atlas("d", kind="nilearn", fetcher="destrieux_2009")
.load_tract_atlas("ho", kind="nilearn", fetcher="harvard_oxford",
                  atlas_name="cort-maxprob-thr25-1mm")
```

The atlas of `.index_tracts` is one the program loaded or, if it loaded
none of that name, one of the engine (`.index_tracts("100307",
"destrieux")`, 4.2), which the engine `tracts` has: the program then loads
only its tractograms, in `MNI152`. Nothing else is implicit: an atlas
neither loaded nor of the engine is an error that lists those that are.
`.index_tracts` is the command that publishes the relations: it adds the
atlas and the tractogram to a `RelationStore` that the solver keeps, indexes
the pair (reading or writing the cache), and gives the program *all* the
relations built so far (`add_extensional_predicate_from_tuples`, with
positions for the column names, as `.load_csv` does), so the commands can be
repeated for more atlases and tractograms. They fail with an
`InvalidCommandExpression` that names the file or the rule broken: a missing
file, an unknown `kind` or nilearn fetcher, a command before the one it needs,
a space mismatch (`allow_space_mismatch=1` overrides), a pair indexed twice, a
`length_threshold` that differs from that of the tractogram's first index (its
relations have those streamlines).

The relations are those of 3.1: `streamline(x, t, s)`, `streamline_label(a,
x, l)`, `endpoint_label(a, x, e, l)`, the geometry by `x`, and the catalogue.
A tractogram enters the store at its first `.index_tracts`, so its index in
`x` is the order of the first `.index_tracts` of each tractogram. The
engine `tracts_engine()` also has three functions, typed by their annotations,
over the tractograms its commands have indexed: `streamline_id(t, s)` (the
`x`), `subject_of(x)` (the `t`) and `index_of(x)` (the `s`). They are for
comparisons in a rule body, as other functions of the frontend are:

```
ans(t) :- streamline_label("fs", x, 1), (t == subject_of(x))
```

The join with `streamline(x, t, s)` gives the same.

Where the handler sits matters. The solver of `NeurolangPDL` is
`RegionFrontendCPLogicSolver`, whose `CommandsMixin` has a catch-all that
refuses every command it does not know; `TractsSolver` puts
`TractsCommandsMixin` *before* it, and `tracts_engine()` is
`NeurolangPDL(solver_class=TractsSolver)`, the hook `NeurolangPDL` has for
that. The matches are exact (no catch-all of its own): any other command
still reaches the refusal, which a test checks. The commands and the rules of
a program are run by one `execute_datalog_program`. The engine `tracts` of
`neurolang-query` (the apps' `engines.yaml`) is the same engine, built from
the YAML: its `solver_mixins` is `TractsCommandsMixin` and its `python_init`
is `nl_tracts.frontend`, whose `init_engine` gives it the vocabulary of
section 16, the three functions and the hooks of `.view` (16.3).

## 9. Packaging and layering

### 9.1 Layout

```
packages/nl_tracts/
  pyproject.toml        dist "nl-tracts"; deps: neurolang, numpy, scipy,
                        nibabel, pandas, pyarrow; extra dev (pytest, nilearn)
  README.md
  DESIGN.md             this file
  tools/make_freesurfer_mappings.py   writes the shipped mappings
  nl_tracts/
    __init__.py         re-exports the public classes (core only)
    spaces.py           Space, SpaceMismatchError, check_same_space
    terms.py            term-mapping format, validation, shipped keys
    atlas.py            Atlas, ProbabilisticAtlas, HemisphereRule, loaders
    tractogram.py       Tractogram
    ids.py              the id x of a streamline: encode, decode
    indexing.py         IndexingParams, *_relation, RelationStore, build_relations
    cache.py            parquet cache: cache_key, read_cached, write_cached
    fetch.py            the atlases of Yeh, HCP842 and HCP1065: download (7.5)
    vocabulary.dl       the vocabulary of tractography, Datalog rules (16)
    vocabulary.py       prelude(), predicates(): reads it; no NeuroLang
    frontend.py         register(), the commands, init_engine(),
                        tracts_engine() -- the only module that imports
                        NeuroLang
    mappings/           shipped term mappings (TSV): freesurfer_aseg_aparc,
                        freesurfer_destrieux, tian_subcortex_s2
    tests/              conftest.py: synthetic volumes, `network` marker;
                        reference_tq.py: tract_querier's indexing, loaded
                        by path, as the oracle of 6.1; uncinate_files.py:
                        an atlas and streamlines for the uncinate
  examples/uncinate.dl  the uncinate in the vocabulary (16.4)
  examples/uncinate_hcp842.dl  its recall on the expert bundles UF_L, UF_R
```

### 9.2 `.importlinter`

* `root_packages` gains `nl_tracts`.
* The layers contract becomes `neurolang_apps > nl_tracts > neurolang >
  nl_core > nl_expr`.
* A new `forbidden` contract: `nl_tracts.{spaces, terms, atlas, tractogram,
  indexing, cache, ids, vocabulary, fetch}` may import none of `neurolang`,
  `nl_core`, `nl_expr`, `lark`, `nilearn`. `frontend` is exempt, and is
  the only module that may import NeuroLang.

Justification of the position. `nl_tracts` needs the engine to exist (it
builds on `neurolang`, so it is **above** it); nothing in `neurolang` may
know about tractography (the language stays neuroimaging-generic; the
existing `regions` support is about volumes, not streamlines), so it is
**not below** it; and the applications (`neurolang_apps`: CLI, engines, the
text-to-query assistant) are the natural users of tractography relations,
so it is **below** them. The layers contract is therefore *the* mechanical
guarantee for the first two; the `forbidden` contract adds the third,
that the core can be used (and tested) without importing NeuroLang. It is
the same pattern as `nl_core` not importing `nibabel`. A layers
contract alone would not say that: nl_tracts's own core is *allowed* to
import `neurolang`, and the forbidden contract is what forbids it. Only
`frontend.py` imports NeuroLang: `register` uses the engine duck-typed, and
the commands extend the solver of `NeurolangPDL` (8.2); the directional
terms did not need it (they are comparisons over relations, 12).

`lint-imports` was run on the scaffold with the contracts above: 4 kept, 0
broken; with `import neurolang` planted in `nl_tracts/atlas.py` and
`import nl_tracts` planted in `neurolang/regions.py`, both are reported
broken.

### 9.3 CI and release touch points

Changed with the scaffold, because the `.importlinter` change would
otherwise break CI: `.github/workflows/layers.yml` (`PYTHONPATH` gains
`packages/nl_tracts`). Changed with the atlas layer, because it is what makes
the package's tests run: `.github/plan_tests.py` (`PACKAGES`, order is
semantic: `nl_tracts` between `neurolang` and `neurolang_apps`, extras
`--extra dev`) and `.github/workflows/tests.yml` (paths-filter `nl_tracts`);
`plan()` gives, for a change to `nl_tracts`, its tests on 3.12-3.14 and those
of `neurolang_apps` on 3.12, and a change to `neurolang` or below now also
runs `nl_tracts`. **Not changed** (15.8): `.github/workflows/publish.yml`
(build loop: do not publish before the package does something),
`doc/conf.py` (sys.path), `Makefile` (`--cov=nl_tracts`), and the
documentation page. The `network`-marked tests fetch nilearn atlases, which
`tests.yml` already caches (`~/nilearn_data`).

## 10. Public API

Complete signatures and docstrings are in the modules; this is the surface.

```python
# spaces.py
Space = NewType("Space", str)
class SpaceMismatchError(ValueError): ...
def check_same_space(atlas_space, tractogram_space, allow_mismatch=False) -> None

# atlas.py
@dataclass(frozen=True) class HemisphereRule: pattern: str; hemi: str
HEMISPHERE_RULES: Sequence[HemisphereRule]
def infer_hemispheres(names, rules=HEMISPHERE_RULES) -> Series        # left|right|none
def load_label_table_freesurfer_lut(path) -> DataFrame
def load_label_table_bids_dseg(path) -> DataFrame
def load_label_table_tsv(path) -> DataFrame
class Atlas:
    def __init__(self, name, labels, affine, table, space, background=0,
                 hemisphere_rules=HEMISPHERE_RULES, term_mapping=None)
    from_freesurfer(name, image, lut, space, **kw) / from_bids_dseg(...) /
    from_tsv(...) / from_nilearn(name, bunch, space, **kw)
    label_relation() / label_term_relation() / label_extent_relation()
    unmapped_labels() -> DataFrame                 # id, name, hemi
    check_same_space(other, allow_mismatch=False) -> None
    content_hash() -> str
    attributes: name, space, labels, affine, background, table
class ProbabilisticAtlas:      # extension point, constructor raises

# terms.py
COLUMNS = ["atlas_label_name", "term", "hemi"]
SHIPPED_MAPPINGS = ("freesurfer_aseg_aparc", "freesurfer_destrieux",
                    "tian_subcortex_s2")
def load_term_mapping(source) -> DataFrame   # DataFrame, path or shipped key
def shipped_mapping_path(key) -> Traversable

# tractogram.py
class Tractogram:
    def __init__(self, id, points, offsets, space, drop_short=False)
    from_file(id, path, space, drop_short=False) /
    from_streamlines(id, streamlines, space, drop_short=False)
    n_streamlines / lengths_in_points() / lengths_in_mm() / content_hash()

# ids.py
SHIFT = 32
def encode(index, s) -> int64 | ndarray       # (index << 32) | s, vectorised
def decode(x) -> (index, s)

# indexing.py
@dataclass(frozen=True) class IndexingParams:
    crossing_threshold: float = 2.0
    length_threshold: float = 0.0
    emit_background: bool = False
def points_to_voxels(points, affine, shape) -> ndarray
# the rows of one tractogram, by (t, s); the store makes x
def traverses_relation(tractogram, atlas, params) -> DataFrame   # a, t, s, l
def endpoint_relation(tractogram, atlas, params) -> DataFrame    # a, t, s, e, l
def streamline_extent_relation(tractogram, params) -> DataFrame
def endpoint_coord_relation(tractogram, params) -> DataFrame
def endpoint_point_relation(tractogram, params) -> DataFrame     # one row per end
def streamline_length_relation(tractogram, params) -> DataFrame
class RelationStore:    # relations (by x, 3.1), tractograms (ids, load order)
    add_atlas(atlas) / add_tractogram(tractogram, params) /
    add_index(tractogram, atlas, params, cache_dir=None, allow_space_mismatch=False)
def build_relations(tractogram, atlas, params=IndexingParams(), cache_dir=None,
                    allow_space_mismatch=False) -> RelationStore

# cache.py
def cache_key(tractogram, atlas, params) -> str
def read_cached(cache_dir, key, relation) -> DataFrame | None
def write_cached(cache_dir, key, relation, frame) -> None

# fetch.py
ATLASES: dict   # hcp842, hcp1065: urls, md5, archive, whole_brain, bundles
DEFAULT_DATA_DIR = "~/.cache/nl_tracts"
def fetch_tractogram(atlas="hcp842", bundle=None, data_dir=None, url=None,
                     md5=None) -> Path               # a .trk, downloaded once

# frontend.py
RELATION_NAMES: dict[str, tuple[str, ...]]
def register(engine, store, only=None) -> None
def relation_frames(store) -> dict[str, DataFrame]       # + opposite_hemisphere
class TractsCommandsMixin   # .load_tract_atlas, .load_tractogram, .index_tracts
class TractsSolver(TractsCommandsMixin, RegionFrontendCPLogicSolver)
ATLAS_MAPPINGS: dict[str, str]   # an engine's atlas or a nilearn fetcher
                                  # -> shipped mapping
def engine_label_name(name) -> str       # "-" and "_" as spaces, as the apps
def mni_atlas(name, bunch, key, space="MNI152", term_mapping=None, **kw) -> Atlas
def engine_atlas(engine, name) -> Atlas | None   # from grids and atlas_labels
def engine_atlases(engine) -> dict               # name -> its regions
def nilearn_atlas(name, fetcher, space="MNI152", term_mapping=None,
                  background=0, **options) -> Atlas   # fetch_atlas_<fetcher>
def tracts_engine(**kw) -> NeurolangPDL                  # solver_class=TractsSolver,
                    # then init_engine
def init_engine(engine, mask=None, data_dir=None) -> None
                    # the vocabulary (16), the functions streamline_id(t, s),
                    # subject_of(x), index_of(x), the hooks of .view
                    # streamline_points(ids), streamline_background()

# vocabulary.py
def prelude() -> str                    # the rules of vocabulary.dl
def predicates() -> tuple[str, ...]     # the predicates they define
```

## 11. WMQL (a later PR)

A transpiler from WMQL, tract_querier's language, to Datalog over these
relations, with tract_querier's semantics exactly (its query evaluator as
the oracle), is a separate, later PR. It will also be the oracle of the
vocabulary of section 16 (16.5).

## 12. Directional terms

**Implemented** as tract_querier has them, in the vocabulary (`anterior_of`
and the like, 16.2): a streamline is selected by comparing a face of its
bounding box with a face of the union of the boxes of the labels of the term,
and its ends by comparing their coordinates.
That needs no per-pair evaluation and no new relation beyond `label_extent`,
`streamline_extent` and `endpoint_coord` of 3.1, which `nl_tracts` computes in
bulk (6.2): the cardinal-direction machinery of `CD_relations` (interval
algebra on the boxes of two `Region`s, a per-pair Python evaluation) is not
used, in keeping with decision 1.

In terms of the variants this section first proposed, the reference's terms
are: **partial** on the streamlines (a face of the box, not the points) and
**endpoint** on the ends. The other two variants are *not* tract_querier's
and remain future work:

* **Strict**: *all* the points of the streamline are beyond the face of the
  box; **fractional** (probabilistic): the fraction of the points that are,
  as the probability of a probabilistic fact for `NeurolangPDL` (after the
  fix of 15.7). Both need the points, not the box of the streamline, so
  they need the tile counts below.
* **Vectorised cardinal-direction tile counts.** For a reference region (a
  label, a union of labels, or a box from `label_extent`) with box `[lo, hi]`
  per axis, the 3-vector of classes (before / within / after) of every point
  on each axis: 27 tiles, one pass over the `(P, 3)` array; `np.add.reduceat`
  over the streamline offsets gives an `(S, 27)` count matrix (108 MB per
  10^6 streamlines, per reference region). Each direction is then a 0/1 mask
  over the 27 columns (the anatomical variants of `region_solver` -- the
  direction holds, its inverse and the overlap do not -- masks over the same
  columns), and *strict* is `count == n_points`, *fractional* is
  `count / n_points`.
* **Schema** for them: `direction(a, t, s, l, dir, variant)`, with `dir`
  and `variant` small integers, the fractional one a probabilistic relation
  with the probability first. To decide when the phase starts.

## 13. Test plan

Tests live in `nl_tracts/tests`, with synthetic data built in the tests
(no downloads): small label volumes (e.g. 20x20x20 with 6 labels and
oblique affines) and tractograms of a few hundred streamlines made by
random walks plus hand-built edge cases.

### 13.1 Unit tests, per module

* `spaces`: equal tags pass; unequal raise `SpaceMismatchError`;
  `allow_mismatch=True` warns.
* `atlas`: each loader on a tiny fixture (a three-line LUT, a dseg.tsv, a
  plain TSV, a fake nilearn `Bunch`); hemisphere rules, each convention of
  4.3, first-match-wins, no match, the half-paired warning, custom rules;
  `label` / `label_term` / `label_extent` contents; ids in the volume but
  not in the table; `content_hash` changes with each ingredient and does
  **not** with the name; the `ProbabilisticAtlas` constructor raises.
* `tractogram`: construction validation (offsets, < 2 points), `from_file`
  round trip for `.trk` and `.tck` (written with nibabel in the test),
  lengths, hash independent of the id.
* `indexing`: `points_to_voxels` including **half-integer coordinates**
  (round half to even) and clipping with the warning; `streamline_label` at the
  threshold boundary (exactly 2.00 %, just below, just above) and
  denominators that include the background; `endpoint_label`; extents against a
  loop; chunking: the result for a tiny chunk size equals that for one
  chunk.
* `cache`: key changes with each ingredient; round trip preserves dtypes;
  interrupted write leaves nothing trusted; two ids of the same file share
  the entry.
* `frontend`: `register` adds each relation with `add_tuple_set` as a
  DataFrame (a recording stub engine: no tuple built); a real `NeurolangDL`
  answers the example rules of 3.2.

### 13.2 Equivalence against tract_querier

`tract_querier` is a **test-only**, optional reference, never a dependency.
Its `tract_label_indices.py` is loaded **by path** (`tests/reference_tq.py`),
with its relative `.aabb` import registered beside it, whereas the package's
`__init__` imports vtk and the tractography readers. Found through
`TRACT_QUERIER_DIR` (the directory of the `tract_querier` package) or an
installed copy, else the tests skip. (Installed from
`github.com/demianw/tract_querier` into a throwaway environment.) The oracle
of its query evaluator is the transpiler's, a later PR (11).

**The geometry of a tractogram (done)**: `Tractogram`, the three
atlas-independent relations and the store, against loops over the raw
points (`tests/test_tractogram.py`, `test_relations.py`); and a directional
query run end to end, from a `Tractogram` and an `Atlas` to NeurolangDL,
against numpy over the points and the voxels of the label
(`test_frontend.py`).

**The indexing (done)**: `tests/test_indexing.py`. The same volume, affine and
streamlines go through `TractographySpatialIndexing` (loaded by path, with its
relative `.aabb` import registered beside it) and through nl_tracts with
`emit_background=True`, and the sets are compared exactly:
`traverses_relation` against `crossing_tracts_labels`, `endpoint_relation`
against `ending_tracts_labels` without its label-0 entries, on the four affines (aligned, flipped, permuted, oblique), at
thresholds of 0.5, 2, 25, 50 and 100, with points outside the volume. The
length threshold is compared with the reference's ids mapped back through the
kept-index list. Dedicated tests for what random data does not reach: the
rounding of half-voxel coordinates to even (a mutation to round half up is
caught only there), the clip and its one warning that counts the points, also
when the indexing is in chunks, the background in the denominator and not in
the rows, endpoints in the background never having a row, chunks giving the
same relations, the dtypes. Mutations checked: `>` for `>=`, round half up, a
point left out of the denominator. The cache: key, round trip, an atomic
write, and a second store reading what the first wrote under another id and
another atlas name.

**From files to the answer (done)**: `tests/test_commands.py` (8.2) and
`test_uncinate.py` (16.4).

### 13.3 Scale and engine tests

* A synthetic tractogram of 10^6 streamlines (marked `slow`, not in CI by
  default) checks the time and memory of 6.2 and 7.2.
* Join, negation and `AGGREGATE` (count of labels per streamline) rules on a
  10^6-row `streamline_label` through `NeurolangDL`, as in the probes of 8.1.
* A **no-per-row-Python** check: the build of every relation for 10^5
  streamlines under a profile that fails if `Tractogram`/`indexing` call a
  Python function per streamline (counts calls of the module's functions).
* Multi-atlas: two atlases, one tractogram; the same rule with `"dk"` and
  `"aal"` in the constant, and a `label_term` rule giving the same
  streamlines on two atlases that are the same volume under two name tables.
* Every rule in this document is parsed with `standard_syntax`.
* `test_scaffold.py` (present) checks that the core imports without
  `neurolang`, `nl_core`, `nl_expr`, `lark` or `nilearn` in `sys.modules`;
  `lint-imports` enforces the same statically.

## 14. Phases

**Method: test-driven development.** In every phase the tests of 13 for that
phase are written first and seen to fail (for a function that is only a
signature the first red is the scaffold's `NotImplementedError` or a missing
module; after that, reds are assertions on values); then the smallest
implementation that turns them green; then refactor with the tests green.
A test that cannot fail is not a test: key tests are checked by mutating the
implementation (the atlas layer's were, for the extent convention and the
hemisphere precedence: 12 and 4 tests fail). Oracle cases (13.2) and
boundary cases (half-integer coordinates, the exact threshold, the
background) are tests before they are code. The CI wiring of 9.3 is done in the PRs of phase 1
(`plan_tests.py` and `tests.yml` with the atlas layer; the repository has
**no** tests of `plan_tests.py`, so the change was checked by calling
`plan()` for the changed-package cases, not by a test: adding a test for it
is an open item, 15.8). Each phase is one PR, merged on its
own; the scaffold's smoke test stays.

1. **Core**: `spaces`, `terms`, `atlas` (loaders, hemispheres, `label`,
   `label_term`, `label_extent`), `tractogram` with the geometry relations
   (`streamline_extent`, `endpoint_coord`, `streamline_length`), the store and
   `register`, the voxel indexing (`streamline_label`, `endpoint_label`), the
   streamline ids `x`, the cache and the commands -- **done**. Still to
   do: re-measure the sizes of 7.2 on a real tractogram.
2. **Relations, cache, frontend**: **done** but for the mapping files for AAL
   and Schaefer (FreeSurfer aseg+aparc and Destrieux are shipped).
3. **Examples and docs**: the vocabulary of rules over the schema of 3.1
   (`traverses(x, term, h)` and the like) and the uncinate in it, with the
   engine `tracts` -- **done** (16); still to do: a tutorial of
   tract_querier's other standard bundles (arcuate, corticospinal...) in the
   vocabulary; docs page; CI wiring (9.3).
4. **The variants of the directional terms that are not tract_querier's**
   (12): strict and fractional.
5. **Probabilistic atlases** (4.4), after the engine fix in 15.7.

## 15. Open questions

Recommendations in bold; none changes a decision of section 2.

1. **`pyarrow` as a hard dependency.** The brief says the core depends only
   on numpy/scipy/nibabel/pandas; parquet needs pyarrow (or fastparquet).
   **Hard dependency** (it is already one of `neurolang_apps`, and every
   engine call benefits from reading the cache); the alternative is a
   `cache` extra and an import inside `cache.py`.
2. ~~Should `endpoint_label` carry label 0?~~ **Decided: never.**
   `emit_background` governs `streamline_label` only; "no region at this end" is the negation of
   6.3.
3. **String `a` and `t` columns at 10^8 rows.** `t` is no longer in the
   large relations (the id `x`, 3.1); `a` is, about 1 GB at 10^8 rows (10
   bytes a row, pandas 3's `str` dtype, 7.2). **Keep strings until
   measured**; the fallback is a surrogate integer key with a lookup
   relation, as `streamline(x, t, s)` is for the tractogram.
4. **The extra copy in `RelationalAlgebraFrozenSet.__init__`** for
   DataFrames. A change in `nl_core` (adopt without copying); **separate PR
   if the footprint matters**, not part of this package.
5. **Incremental registration**: `add_tuple_set` replaces. Phase 1 builds the
   `RelationStore` first and registers once. **Keep**, and revisit if
   interactive loading of tractograms one by one is needed.
6. ~~Is `nl_tracts` above `neurolang` for anything yet?~~ (9.2.)
   **Closed**: the commands extend the solver of `NeurolangPDL` (8.2), so
   `frontend` imports it.
7. **Engine bug**: `add_probabilistic_facts_from_tuples` rejects
   DataFrames (4.4). **Fix in neurolang/nl_core** (infer the row type from
   the first row of a DataFrame, as `add_tuple_set` does), with a test,
   before phase 5; outside this task.
8. **CI and release wiring** (9.3): `layers.yml`, `plan_tests.py` and
   `tests.yml` are done; `publish.yml`, docs and the Makefile are not, and
   `plan_tests.py` has no test of its own. **Do the rest when the package
   does something worth publishing; add a test of `plan()`** (it is a pure
   function, five lines of table).
9. **tract_querier for the oracle tests** (13.2): resolved. It is not a
   dependency; its indexing is loaded by path from a copy found through
   `TRACT_QUERIER_DIR`, and the tests that need it skip without. **Open**:
   CI has no copy, so the oracle runs on a developer machine; a dedicated job
   that installs it from git and runs `--runslow` would put it in CI.
10. **Term vocabulary governance.** The vocabulary is now tract_querier's
    (3.2), which fixes the names but not who adds a term. **A single
    curated vocabulary file listing the allowed terms, and a test that each
    mapping file uses only listed terms**; today the tests only check the
    form (`[a-z][a-z0-9_]*`) and the absence of hemisphere tokens.
11. **Axis and variant encoding by int.** Rule authors must remember
    `axis = 1` is y. **Document it; if it hurts, add a tiny `axis(n, name)`
    relation rather than a string column.**
12. **LPS/RAS+ files.** `from_file` converts to RAS+ with nibabel and the
    oracle compares in RAS+. **State it and test only RAS+ against the
    reference**; the conversion itself is nibabel's.
13. **Destrieux mapping** (3.2) is a draft by a non-anatomist. The
    instruction "the medial wall should be excluded, all other labels too" was
    read as: the medial wall (and `Unknown`) have no term and **every other
    label has**, the boundary structures included, with the terms of both
    regions they separate. The other reading, excluding more, was not
    followed. **Veto it, or review the assignments** (`tools/
    make_freesurfer_mappings.py`, `DESTRIEUX`).
14. **`label_extent` under oblique affines** (6.1.1) is the box of the
    box, a superset. **Keep** (conservative; aligned atlases are exact), and
    tighten with a pass over the voxels if a use needs the exact extent of
    a label under an oblique affine.
15. **Combining atlases in different spaces.** Decision 4 says combining
    mismatched spaces raises; its subject is an atlas and a tractogram. Two
    atlases in different spaces *do* coexist in one engine (decision 2), so
    no cross-atlas rule is imposed; `Atlas.check_same_space(other)` and
    `check_same_space` implement the check for when two things must agree
    (an atlas and a tractogram, or two atlases that are to be compared
    voxel by voxel), and the tests exercise it on two atlases. **Keep**;
    `RelationStore.add_index` (phase 1, indexing) is where an atlas meets a
    tractogram.
16. **`subcortical`** departs from tract_querier's on purpose (3.2). **Say
    so in the tutorial**; the bundles that exclude `subcortical` (`not in
    subcortical.side`) now also exclude
    the caudate, hippocampus and accumbens, a change of what they select on
    a volume that has them. **Check the standard bundles before the
    tutorial**, on tractography.
17. **`X.left` of a group with midline members** (3.2): the file's
    `subcortical.left` includes the brain stem. A rule transcribed from it
    must select `hemi` left or none: a rule over `label_term` and
    `label.hemi` with `h` left is not the file's set. **Keep; consider a
    helper relation** `side(a, l, h)` that gives each midline label to both
    sides, once transcribed bundles are tried.
18. **`Left-Thalamus` is two things** (3.2, 4): the legacy label 9 in
    FreeSurfer 6 and the thalamus proper (10) in 7. The mapping is keyed by
    raw name and maps it to `thalamus_proper`. **Keep**, and give the mapping
    a LUT-version key if a FreeSurfer 7 LUT is ever found to need otherwise.
20. **`superior_of` and `inferior_of` are inverted in the reference**: they
    select the streamlines whose top is *below* the top of the box, and
    whose bottom is *above* its bottom, where the other four terms select
    those that extend beyond a face; no shipped query and no test of
    tract_querier uses them. The vocabulary's are anatomical (16.2); the
    transpiler (11) will have to reproduce the reference or not. It is the
    user's own code, and the fix belongs in tract_querier too.
24. **The geometry relations are float32** (3.1), the reference's float64: a
    tie in the reference can be a different answer here. Real coordinates in
    mm are not on a lattice, so ties are measure-zero. **Keep.**
25. ~~The bundles of the vocabulary file cannot yet be run on a tractogram~~:
    **closed**. The voxel indexing is done (6.2) and a program loads its
    files with the commands (8.2).
26. ~~`neurolang-query --engine tracts`~~: **closed**, the engine `tracts`
    of the apps (8.2, 16.3).
27. **Frontend bug found, not fixed**: a constant in the head of `ans`
    (`ans("one", x) :- r(x, y)`) fails with `AttributeError: 'str' object has
    no attribute 'expression'`, and a second `ans` rule is refused. A program
    therefore has one answer. **Fix in `neurolang`** so that one file can
    name several queries (`ans("uf.left", x)`).
28. **Paths in the commands are read as written**, relative to where the
    program runs. A file moved from the directory it was made in loses its
    files. **Keep; write absolute paths** in the commands of a file that
    moves, or run it from its directory.
29. **`.index_tracts` is the command that publishes the relations**, and a
    tractogram's relations have the streamlines that its first index kept
    (`length_threshold`): a second index of it with another threshold is an
    error. **Keep**; the alternative is to put the threshold on
    `.load_tractogram`, which the relations of the atlas do not need.
30. **Magic Sets and aggregation**. The check refuses a cycle through
    an aggregation or a negation in the rewritten program, and the query is
    solved as written, so for such a program the rewrite is not applied and
    its goal-directed pruning is lost: for the vocabulary, the cost of 15.38.
    A rewrite that keeps the program stratified (the aggregate evaluated before
    the magic predicates that depend on a literal that reads it) would keep
    the pruning. **Keep.**
31. **The vocabulary quantifies the atlas** (16.2): with several atlases a
    predicate is the union over them, and the box of a term is over all of
    them, which mixes atlases that should not be (two parcellations of one
    subject: fine; of two spaces: not). **Keep** for one atlas, the usual
    case; if several are common, add variants with the atlas as first
    argument (`traverses_in(a, x, term, h)`), not a constant in the rules.
32. **A set of terms is a disjunction** (16.4): `(traverses(x, "a", h) |
    traverses(x, "b", h) | ...)`, `|` being "or" in a rule body (the
    frontend translates it to rules). **Done.**
33. **The term names are tract_querier's** (`orbitofrontalgyrus`,
    `middle_frontal_gyrus`, `cingular`...), so that a bundle of
    tract_querier is a transcription. They are not those of an anatomist,
    nor of the other atlases' mappings to come. **Keep**, and add aliases to
    the mappings, not to the rules, if others are wanted.
34. **Names shared with the region relations of `NeurolangPDL`**
    (`anterior_of`, `left_of`...: 16.3): the tracts solver leaves those
    functions out, so an engine cannot have both the `Region` relations and
    the vocabulary under these names. **Keep** (the tracts engine has no
    regions); rename the vocabulary's if one engine needs both.
35. **`only_in` and the background** (16.2): with `emit_background=1` the
    background is a label of no term that almost every streamline
    traverses. **Keep**; `only_in` could leave out the labels of no term,
    which would also leave out the labels a mapping misses.
36. **aparc+aseg against wmparc** (16.4): the terms are the cortex and the
    white matter under it, so on an atlas without the white matter of each
    gyrus the bundles lose the streamlines that end in the white matter
    (recall 2-21% against 98-99% on HCP842). **Document**, as the example
    does; a check that warns when the terms of a rule have no white matter
    labels in the atlas is possible.
37. **Streamlines of one point** are refused (parity 9 of 6.1, as the
    reference); DIPY's HCP842 whole brain has 37. **Decided**: refused by
    default; with `Tractogram(..., drop_short=True)` (`from_file`, the
    option `drop_short=1` of `.load_tractogram`, the default of its kinds
    `hcp842` and `hcp1065`) they stay in the arrays and are in no relation
    (`_kept_streamlines`, as `length_threshold`), with a warning that says
    how many: `s` is still the position in the file (144,641 ids of the
    144,678 streamlines of `whole_brain_MNI.trk`, with gaps), so the ids
    agree with another reader of the file, the hooks of `.view` and the
    parquet cache need no change (`content_hash` is that of the file's
    streamlines), and the cache key differs from no other. Pruning them
    instead would have renumbered every streamline after the first one
    dropped. A streamline of no point is still refused.
38. **The cost of a predicate over all terms** (16.6): 17-26 s for the
    uncinate on 144k streamlines, 4.6 GB. **Open**: carry the constants of a
    rule into the vocabulary (Magic Sets through negation and aggregation,
    15.30).
39. **The vocabulary is not declared in `engines.yaml`** (`predicates`,
    `sorts`): the query assistant and `--list-predicates` see the
    predicates the engine has, not their documentation, which is in the
    `%` comments of `vocabulary.dl`. **Open**: give `engines.yaml` the
    sort `streamline` and the predicates' arguments, or let the apps read
    the comments of an engine's rules.

## 16. The vocabulary of tractography

**Implemented**: `nl_tracts/vocabulary.dl`, read by `nl_tracts.vocabulary`
(`prelude()`, `predicates()`), and `nl_tracts.frontend.init_engine`, which
gives it to an engine. It needs nothing but the relations of 3.1.

### 16.1 Why

The stored relations of 3.1 say which labels a streamline traverses and
where its ends and its box are, by label id and by axis number. A bundle
written on them directly is long, and a bundle transcribed from
tract_querier's queries with their semantics exactly, label set by label
set, is longer: `uf.left` of `freesurfer_queries.qry` is some 2,300 rules
(the transpiler of 11, a later PR). Neither
is what a person, or the query assistant, which writes Datalog, can read or
write. The vocabulary is a dozen predicates over those relations, named as
WMQL names its operations, in which a bundle is a rule or two, as natural
to write as in WMQL. Its semantics is the natural reading of the names
(16.2); where that differs from tract_querier's is stated in 16.5.

### 16.2 The predicates

Arguments: `x` a streamline (its id, 3.1); `e` an end of it, 0 or 1, which
only identifies the end (the order of the points of a streamline is
arbitrary, so "the first end" means nothing anatomically, but "the same end"
does); `term` a term of the atlas's term mapping (`label_term`, 3.2), a
string such as `"temporal"`; `h` a hemisphere, `"left"`, `"right"` or
`"none"` (`label`'s `hemi`). Axes are 0 x (left to right), 1 y (posterior to
anterior), 2 z (inferior to superior), mm, RAS+.

| Predicate | Holds when | From |
|---|---|---|
| `term_label(a, l, term, h)` | label `l` of atlas `a` has the term and hemisphere `h` | `label_term`, `label` |
| `traverses(x, term, h)` | `x` traverses a label of the term of hemisphere `h` | `streamline_label` |
| `endpoint_in(x, e, term, h)` | the end `e` of `x` is in such a label | `endpoint_label` |
| `ends_in(x, term, h)` | an end of `x` is | `endpoint_in` |
| `in_hemisphere(x, h)` | `x` traverses a label of hemisphere `h` | `streamline_label`, `label` |
| `only_in(x, term, h)` | every label `x` traverses is of the term and of `h` | `traverses`, a negation |
| `term_box_lo(term, h, axis, lo)`, `term_box_hi(term, h, axis, hi)` | the box of the term of hemisphere `h`: `min(lo)`, `max(hi)` of the boxes of **all** its labels in the atlas | `label_extent` |
| `anterior_of(x, term, h)` | the box of `x` reaches strictly beyond the anterior face of the term's box: `hi > box hi` on y | `streamline_extent` |
| `posterior_of`, `superior_of`, `inferior_of`, `left_of`, `right_of` | the same for the posterior (`lo <` on y), superior (`hi >` on z), inferior (`lo <` on z), left (`lo <` on x) and right (`hi >` on x) faces | |
| `medial_of(x, term, h)`, `lateral_of(x, term, h)` | relative to the hemisphere of the term: medial of a left term is `right_of`, of a right term `left_of`; lateral the reverse; a term of `"none"` has neither | |
| `endpoint_anterior_of(x, e, term, h)` ... `endpoint_lateral_of` | the end `e` of `x` is beyond that face (its coordinate, `endpoint_coord`) | `endpoint_coord` |

Kept as they are, from the store and the engine: `opposite_hemisphere(h1,
h2)`, `streamline(x, t, s)`, and the functions `subject_of(x)`, `index_of(x)`,
`streamline_id(t, s)` (8.2).

Every direction is **anatomical**: `superior_of` is above the box, which
tract_querier's is not (15.20). `medial_of` and `lateral_of` keep
tract_querier's meaning, relative to the side of their argument, and need
a side, as there (a term of hemisphere `"none"` has no rows).
`only_in` counts every label `x` traverses, the background included when
`.index_tracts(..., emit_background=1)` has its rows (6.3): it has no term,
so with them hardly any streamline is `only_in` a term; leave the option at
its default.

The atlas is not an argument: each predicate holds over every atlas loaded
(`a` is quantified in its rules). With one atlas, as usual, it is that one;
with several, a predicate is the union over them, and `term_box_*` is the
box of the term over all of them (15.31).

The file is documented by `%` comments, one per predicate with its
arguments (a test checks it), so that it reads as the reference of the
vocabulary, for a person or for the query assistant.

### 16.3 How it enters a program

`init_engine(engine, mask=None, data_dir=None)` gives an engine whose
solver has `TractsCommandsMixin` the rules of the vocabulary (it runs
`prelude()`), the three functions of the ids, and two hooks, the methods
that the REPL's `.view` calls on an engine to draw a result of streamline
ids: `streamline_points(ids)`, the points of each `x` (decoded by
`ids.decode` to the tractogram, by the store's load order, and the position
in it), in mm, RAS+; and `streamline_background()`, the label image of the
first atlas loaded (the tractograms are in its space), or None before
`.index_tracts`. `tracts_engine()` calls it, and so does the engine `tracts`
of the apps, as its `python_init` (its signature is that of a `python_init`).
The hooks are closures over the solver's state, so that the solver mixin
needs no access to the frontend and the core of nl_tracts no NeuroLang.

So **a program is pure Datalog and the three commands**: the rules of the
vocabulary are the engine's before the program runs. NeuroLang accepts
rules over predicates that do not exist yet (those `.index_tracts`
publishes); it warns that their types cannot be inferred, which
`init_engine` silences, and infers them when the query runs. The other way
considered, the commands adding the rules when they first publish the
relations, puts them in the scope of the submission that ran the command in
the REPL, where `.clear`, or the end of a query's scope, would drop them
while the commands' state stays. On an engine without the commands
(`register()` on a `NeurolangDL`), `engine.execute_datalog_program(prelude())`
adds the rules.

**A clash of names, solved in the solver.** The solver of `NeurolangPDL`
has the functions `anterior_of(r1, r2)`, `posterior_of`, `superior_of`,
`inferior_of`, `left_of` and `right_of` over `Region` values (the cardinal
relations of `region_solver.py`), and a rule cannot have a function's name
as its head. `TractsCommandsMixin` leaves out, of the solver's functions,
those named as a predicate of the vocabulary: in the tractography engine
the names are the rules over streamlines (it has no `Region` values;
`anatomical_anterior_of` and the others stay). Found while doing it, an
engine bug: such a rule failed with `AttributeError: 'Constant' object has
no attribute 'name'`; it is now refused with a message that names the
function (`nl_core`, with tests).

**In the REPL** the commands of a program typed or piped at the prompt run
as statements of their own, and its rules too; `neurolang-query -i --file`
runs the whole file as one submission, a query, whose rules are local to
it. The relations the commands of a query load used to be discarded with
them, while the commands' state (the store) stayed, so that the session
could neither query them nor load them again; they are now kept (a fix of
the REPL's submissions, with a test on the toy command and `.load_csv`).

**The atlases of the engine.** The engine `tracts` extends `mni_atlases`:
it has the atlases of neurosynth, Destrieux, Schaefer and Tian, loaded as
neurosynth loads them, without the studies that activate their regions
(which need its peaks) and without the MNI template, which nothing of it
uses. The registry loads them after `python_init`, which runs first so
that an engine's own data (neurosynth's peaks) is there for the atlases'
`activated`; `init_engine` keeps the engine in the solver's state, and
`.index_tracts` finds an atlas in it when it is asked for one (4.2). Its
cost: about 270 KB downloaded on the first build (Destrieux, Schaefer and
Tian, into the engine's own data directory) and 2-5 s to build (measured:
4.5 s the first time, 2.1 s after).

### 16.4 The uncinate: a worked example

tract_querier's `UF.side` of `freesurfer_queries.qry`,

```
UF.side = (((orbitofrontalgyrus.side or middle_frontal_gyrus.side or
             inferior_frontal_gyrus.side)
            and endpoints_in(temporal.side and anterior_of(amygdala.side))
            and insula.side)
           not in occipital.side not in parietal.side not in cingular.side
           not in posterior_of(putamen.side) not in HEMISPHERE.opposite)
```

is, in `examples/uncinate.dl`,

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

after the three commands. `.side` is the variable `h`, `.opposite` the
join with `opposite_hemisphere`; `endpoints_in(A and anterior_of(B))`,
whose ends are those in `A` *and* beyond `B`, is the same `e` in both
atoms; `not in` is a negation, safe because `x` and `h` are bound first.
The term names are the file's, and exist in the shipped mapping.

*Tested* (`tests/test_uncinate.py`): a FreeSurfer-like atlas of the labels
the rule names (their FreeSurfer ids and names, boxes in mm, the right
ones mirrored) and streamlines made for it, one uncinate, the same
reversed, and one that misses each condition (the temporal end not in front
of the amygdala, no insula, the cingulate, the other hemisphere, behind the
putamen, the occipital lobe), each also mirrored: the example, its paths
replaced, selects exactly the uncinates of each side. The hooks give the
points of ids of two tractograms. The engine `tracts` of the apps runs it,
through `neurolang-query`'s engine and through the REPL, statement by
statement and as one submission, then `.view`
(`neurolang_apps/utils/tests/test_tracts_engine.py`).

*On real data* (measured, not a test; this Mac, pandas backend). Atlas
`cvs_avg35_inMNI152` of FreeSurfer, tractogram the whole brain of DIPY's
HCP842 atlas (`whole_brain_MNI.trk`, MNI, 144,678 streamlines, of which 37
have one point and are in no relation, 15.37: 144,641 left; it is
`.load_tractogram("hcp842", kind="hcp842")`, 7.5), reference the expert bundles `UF_L.trk` (370) and
`UF_R.trk` (328) of the same atlas. Every reference streamline is, within
0.004 mm (the mean distance of 20 points along both, either orientation),
a streamline of the whole brain, so the comparison is by streamline:

| Atlas | Side | Selected | Precision | Recall | Dice of visited voxels |
|---|---|---|---|---|---|
| `aparc+aseg.mgz` | left | 333 | 0.23 | 0.21 | 0.39 |
| `aparc+aseg.mgz` | right | 158 | 0.04 | 0.02 | 0.14 |
| `wmparc.mgz` | left | 724 | 0.50 | 0.98 | 0.69 |
| `wmparc.mgz` | right | 810 | 0.40 | 0.99 | 0.67 |
| the engine's Destrieux + Tian | left | 304 | 0.46 | 0.38 | 0.54 |
| the engine's Destrieux + Tian | right | 25 | 0.28 | 0.02 | 0.19 |

Why `aparc+aseg` misses them, found by counting how many reference
streamlines pass each condition: the exclusions remove none; an end in a
temporal label keeps 123 of 370 (left) and 50 of 328 (right), the insula
156 and 36, the frontal terms 220 and 212. The terms are, as in
tract_querier, the cortex of a gyrus *and* the white matter under it
(3.2); aparc+aseg has the cortex only, its white matter is one label,
`Cerebral-White-Matter`, of no term, and the streamlines of the HCP842
atlas end in the white matter, short of the cortex. With `wmparc`, which
has the white matter under each gyrus, recall is 98-99%; the precision of
40-50% is against a curated bundle, from which an expert removed
streamlines the definition keeps. The recall is a program,
`examples/uncinate_hcp842.dl`: the expert bundles loaded as tractograms of
their own (`kind="hcp842", bundle="UF_L"`), indexed by the same atlas, and
the rule counted over them: 363 of 370 (left), 324 of 328 (right), 7 s.
The precision is not: it needs each streamline of a bundle matched to one
of the whole brain, within 0.004 mm but not equal point for point, which
is no join of the relations. Nothing was tuned: the default
`crossing_threshold` (2%), no length threshold. The example therefore
loads a `wmparc`. The hemispheres of the labels of both atlases are those
of the `.left` and `.right` names of `FreeSurfer.qry`, label by label, so
`in_hemisphere` is `HEMISPHERE.side` there (16.5).

With the atlases of the engine `tracts` (4.2), the commands are
`.load_tractogram` and `.index_tracts` by `"destrieux"` and `"tian"`, and
the rules are the same: every term of the uncinate exists, the cortical
ones in Destrieux, `amygdala` and `putamen` in Tian, and the vocabulary
holds over both atlases at once (16.2). Destrieux is the cortex only, as
aparc+aseg (not on the same template: nilearn's volume of it is not
FreeSurfer's `cvs_avg35_inMNI152`), and the reference streamlines that
pass each condition say why it does poorly: 348 of 370 (left) and 328 of
328 (right) go through a frontal term, 273 and 264 have an end in the
temporal lobe, the exclusions remove none, but only 165 and 7 go through
the `insula` of Destrieux, the cortex of the insula, which the uncinate
passes under, in the white matter of the extreme capsule; on the right
that condition loses nearly all of them. Not tuned: an atlas of the
cortex alone answers a definition written for the cortex and the white
matter under it.

Time per step (one run each): loading the atlas 0.5 s, the tractogram
1.6 s, `.index_tracts` 3.2 s (aparc+aseg) and 4.4 s (wmparc, writing the
cache); the query 17-19 s a side with aparc+aseg, 24-26 s with wmparc; the
whole file through `neurolang-query --engine tracts --file` 30 s, 4.6 GB
peak memory. 16.6 says where the time goes.

### 16.5 Natural and exact semantics

The vocabulary has the natural semantics; tract_querier's exact one is
that of the transpiler of section 11, a later PR, which will be the oracle
of the vocabulary. They differ in four places, by design:

* **Label sets.** A term is a set of labels of the mapping (`label_term`)
  and a hemisphere one of `label.hemi`; a WMQL name is a set of ids of the
  `.qry`. For the sided names of the shipped mapping they are the same sets,
  but for its stated differences (3.2; `subcortical.left` of the file has
  the brain stem, 15.17, ...).
* **Boxes.** `term_box_*` is the box of all the labels of the term in the
  atlas, whether a streamline traverses them or not: label sets are data.
  tract_querier's box is that of the label set of its argument, per
  tractogram; for a name made of label ids (`amygdala.left`), the same; for
  a label set computed from the streamlines (the labels of a directional
  result, of a complement), not a term at all, which the vocabulary has no
  way to name.
* **`HEMISPHERE.opposite`** is the union of every `.right` (or `.left`) name
  defined by `FreeSurfer.qry`, by id; `in_hemisphere` reads `label.hemi`.
  On `aparc+aseg` and `wmparc` of `cvs_avg35_inMNI152` they agree on every
  label in the volume; an atlas whose names the hemisphere rules (4.3) read
  otherwise than the file would differ.
* **Vertical directions** are anatomical here, inverted in tract_querier
  (15.20).

And the atlas is quantified rather than a constant of the program (16.2).

The comparison of the two, the uncinate of the vocabulary against
tract_querier's `UF.side` transpiled, on synthetic data and on the real data
of 16.4, belongs to the PR of the transpiler (11).

### 16.6 Performance

*Measured* on the data of 16.4 (wmparc, cache warm). A predicate asked
with constants alone is fast, as NeuroLang's Magic Sets rewrite carries the
constants into the vocabulary's rules: `traverses(x, "insula", "left")`
0.4 s (17,477 streamlines), `posterior_of(x, "putamen", "left")` 0.4 s;
`traverses(x, t, h)` for every term is 2.35 million rows, 1.0 s. The
uncinate is 23-25 s a side: the rewrite is refused for it (it would put
a negation in a cycle, 15.30), so each predicate its rule
reads is evaluated for every term and hemisphere, and the rule then selects
its constants. A profile of one side: 14 s translating the rules into
relational algebra, 10 s of joins, 9 s of selections. Levers, not taken: a
Magic Sets rewrite that keeps such programs stratified (15.30), so that the
constants of the rule reach the vocabulary; term-specialised rules would
make the vocabulary as long as the label-set programs of 16.1 (15.38).
