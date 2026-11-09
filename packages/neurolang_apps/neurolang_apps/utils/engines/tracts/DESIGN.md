# the engine tracts: tractography for NeuroLang

Status: **implemented, from files to the answer**: the atlas layer (`terms`,
`atlas`, over NeuroLang's `neurolang.atlases`: sections 3.2, 4, 5, and
`label_extent` of 6.1), the tractogram
and its relations (`Tractogram`, `streamline_extent`, `endpoint_coord`,
`streamline_length`), the **voxel indexing** (`streamline_label`,
`endpoint_label`: 6.2), the parquet cache and the `RelationStore` (7.3), the
streamlines identified by **one integer id `x`** (3.1), the commands that
build the relations in a Datalog program (8.2) by any atlas of the engine
(`.load_atlas`, or those of an engine of the apps, neurosynth's: 4.2), the
**vocabulary
of tractography**,
Datalog rules over this schema in which a bundle is a rule or two
(`traverses(x, term, h)`, `anterior_of(x, term, h)`...: section 16), with the
uncinate as an example, and the engine `tracts` of `neurolang-query` and of
its interactive REPL, whose `.view` draws the streamlines (16.3), and
tract_querier's queries translated into the vocabulary (17). Not done:
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
16 The vocabulary of tractography · 17 WMQL in the vocabulary

## 1. Goals

`tract_querier` answers questions such as "streamlines that traverse the
precentral gyrus and end in the temporal pole, in either hemisphere" with
its own query language over two indices: label -> streamlines and
streamline -> labels. NeuroLang already has a Datalog with negation,
aggregation and (in `NeurolangPDL`) probabilities. the engine tracts makes
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
   label table, with its loaders and configurable hemisphere inference
   (4). *Since 2026-11-07 it is NeuroLang's one class of atlases,
   `neurolang.atlases.Atlas`, the one `.load_atlas` and the engines load;
   the engine tracts adds the term mappings and the catalogue relations (4.1).*
   Probabilistic (4D) atlases get an explicit extension point, not an
   implementation (4.4).
4. **Spaces**: no registration. Atlases and tractograms carry a space tag;
   combining mismatched tags raises unless explicitly overridden (5).
5. **Indexing semantics match tract_querier**, the equivalence oracle (6).
6. **Directional relations are future work** (12). *Overridden on
   2026-11-03: the directional terms of tract_querier are required, and
   implemented as it has them, in the vocabulary (12, 16); only the variants
   that are not its are future work.*
7. **Layering**: a core that depends only on numpy/scipy/nibabel/pandas
   (and pyarrow for parquet, 15.1), and `neurolang.atlases`, which needs
   nothing else of NeuroLang (9.2), and a thin `neurolang_apps.utils.engines.tracts.frontend`. The
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
tractogram `x == s`); `neurolang_apps.utils.engines.tracts.ids` has `encode` and `decode`, vectorised.
`streamline(x, t, s)` is the **decoding relation**: a rule that needs the
tractogram or the position joins it (`ans(t, s) :- q_uf_left(x),
streamline(x, t, s)`); a rule that does not never carries them. The engine of
the commands (8.2) also has the functions
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

**Mapping file format** (`neurolang_apps.utils.engines.tracts.terms`): a TSV with the columns
`atlas_label_name`, `term` and optionally `hemi`; `#` lines and blank lines
are ignored. `atlas_label_name` is the **raw** name as the atlas has it
(`ctx-lh-precentral`, `L G_precentral`); a name has one row per term;
`hemi` is `left`, `right`, `none` or empty. A mapping is rejected on a
missing column, an empty name or term, a bad `hemi`, a duplicate
(name, term), or two rows of one name with different `hemi`. There is no
default term for an atlas without a mapping: `label_term` is then empty and
every label is reported by `atlas.unmapped_labels(atlas, mapping)`, which
returns the atlas's labels (background excluded) that have no term, to see
what a mapping leaves out.

The atlas does not hold its mapping (it is NeuroLang's, 4.1): a mapping is
given, as a DataFrame, a path or the key of a shipped one, to
`RelationStore.add_atlas(atlas, term_mapping)` (or `build_relations(...,
term_mapping=...)`), and in a program to `.index_tracts(t, atlas,
term_mapping=...)`, whose default is that of the atlas's kind
(`atlas.DEFAULT_TERM_MAPPINGS`: a FreeSurfer colour LUT
`freesurfer_aseg_aparc`, Destrieux, of the engines or nilearn's
`destrieux_2009`, `freesurfer_destrieux`, Tian `tian_subcortex_s2`;
Schaefer and the other kinds none). Shipped, in `mappings`, all
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

A fetched atlas (4.2) names its labels as the engines do, its raw names
with `-` and `_` as spaces (`L G and S frontomargin`, `aHIP rh`): the names
of a mapping match those of an atlas as `neurolang.atlases.label_name`
writes both, so that the shipped mappings, keyed by the raw names, serve
the raw names of a FreeSurfer LUT and the normalised names of the engines
alike (no two names of a shipped mapping are one once normalised).

## 4. Atlases

### 4.1 Model

The atlas is NeuroLang's, `neurolang.atlases.Atlas(name, labels, affine,
table, space="unspecified", background=0, hemisphere_rules, kind, image)`:
a 3D integer volume, the 4x4 affine from voxel indices to the space's world
mm (RAS+), a label table (`id`, `name`, optional `hemi`), the space tag, the
id that means "no region", the hemisphere rules, how it was loaded (`kind`,
that of `.load_atlas`) and the nibabel image it was read from. It is the
one class of atlases of NeuroLang: `.load_atlas`, `add_atlas_set` and the
`atlases:` of an engine of the apps load it, the engine adds the relation
`name(label, region)` of its regions and records it by name (the solver's
`atlases`), and the engine tracts indexes by it. It owns the loaders (4.2), the
normalisation of label names (`label_name`), the hemisphere rules (4.3),
the space check (5), the content hash, the regions and the centres of their
voxels. `neurolang.atlases` imports nothing else of NeuroLang (but its lazy
access to nilearn), so that the core of the engine tracts may import it (9.2).
The engine tracts adds, in `neurolang_apps.utils.engines.tracts.atlas`, what is specific to tracts: the term
mapping of its labels (3.2) and the catalogue relations `label`,
`label_term`, `label_extent`, functions of the atlas and of a mapping. It
never touches a tractogram: indexing is in `indexing.py` (6).

An atlas `name` is unique in an engine and is the value of the column `a`.
Ids that are in the table but not in the volume, and ids in the volume but
not in the table, are reported with a warning that names them: the former
are dropped from the table (nilearn's Destrieux has two, the medial walls);
the latter are no region of the engine, `Atlas.unlisted_labels`, and
The engine tracts keeps them in `label` with `name = "label_<id>"` and `hemi =
"none"`, since `streamline_label` would otherwise refer to an unknown
label. The background is no region; it has a row in `label` only if the
table has one, and none in `label_extent`. Labels must be non-negative
integers (a float volume with integral values is accepted, read through
`dataobj`, never `get_fdata`); a 4D image is a probabilistic atlas (4.4)
and raises `NotImplementedError`. Attributes: `name`, `space`, `kind`,
`labels`, `affine`, `background`, `image`, and `table`, the resolved label
table (`id` int32, `name`, `hemi`, by id).

### 4.2 Loaders

| Loader | Input | Notes |
|---|---|---|
| `Atlas.from_freesurfer` | label image + FreeSurfer LUT (`FreeSurferColorLUT.txt`) | `load_label_table_freesurfer_lut` skips comments; ids and names only, colours are ignored; kind `freesurfer` |
| `Atlas.from_bids_dseg` | label image + `*_dseg.tsv` | columns `index`, `name`; kind `bids` |
| `Atlas.from_tsv` | label image + plain TSV | columns `id`, `name`, optional `hemi`; kind `tsv` |
| `Atlas.from_csv` | label image + a CSV | the first two columns are the id and the name, read by `pandas.read_csv` with the options given: what `.load_atlas` reads at a URL; kind `csv` |
| `Atlas.from_labels` | label image + labels as nilearn's fetchers give them | a DataFrame with `index`, `name`, a dict, a list whose position is the id, pairs; bytes are decoded; the background's row dropped; `normalise=True` gives the names of `label_name` |
| `Atlas.from_nilearn` | the name of a nilearn fetcher and its options | `nilearn.datasets.fetch_atlas_<fetcher>(**options)`, imported there, lazily (a missing nilearn is an error that says so); names normalised, space `MNI152`; kind `nilearn:<fetcher>` |
| `Atlas.fetch` | `destrieux`, `schaefer` or `tian`, a data directory, options | the fetchers of the atlases of the engines (`neurolang.atlases.FETCHERS`); names normalised, space `MNI152`; the kind is the fetcher's |

All the file loaders accept a path or an already loaded nibabel image, and
forward keyword arguments (`background`, `hemisphere_rules`) to `Atlas`.
The nilearn convention that `labels[i]` is the name of id `i` (with a
leading `Background`) is the loader's, not the core's. Only the tests of
`neurolang` fetch atlases with nilearn, behind the `network` marker: a
fetch that fails skips the test.

**The atlases of an engine.** An engine of the apps that extends
`mni_atlases` (`engines.yaml`: `neurosynth`, `neurostore`, `tracts`) loads
Destrieux, Schaefer and Tian as `Atlas.fetch` does (the registry keeps a
cache of what it fetched), each as a relation `destrieux(name, region)`
whose names are the atlas's, its label image the grid `engine.grids[name]`,
and records it by name. `.index_tracts(t, "destrieux")` indexes by it as by
any other: `label("destrieux", l, n, h)` joins `destrieux(n, r)` and, in
neurosynth, `destrieux_activated(n, s)`; its terms are those of its kind's
mapping (`destrieux`: `freesurfer_destrieux`, `tian`: `tian_subcortex_s2`;
Schaefer has none, its labels are then without a term, as an atlas without
a mapping).

**The atlases of nilearn, by name.** `.load_atlas(name, kind="nilearn",
fetcher=..., ...)` is `Atlas.from_nilearn`: the label names normalised as
the engine's, so that nilearn's Destrieux and the engine's `destrieux` have
the same names and labels (both without the background), the hemispheres
by the rules (4.3), the terms of `freesurfer_destrieux` for
`destrieux_2009`.

### 4.3 Hemisphere inference

They are NeuroLang's (`neurolang.atlases`), with the atlas. A
`HemisphereRule(pattern, hemi)` has a regular expression that must match
the **whole** raw name, and `hemi` `left` or `right`; the rules are ordered
and the first match wins; `infer_hemispheres(names, rules)` returns a Series
of `left`, `right` or `none`. Defaults, `HEMISPHERE_RULES` (the first five
are the brief's; the last three are what real atlases needed):

| Convention | Left | Right | Seen in |
|---|---|---|---|
| `Left-`/`Right-` (or a space) | `Left[- ].+` | `Right[- ].+` | FreeSurfer aseg; Harvard-Oxford (`Left Thalamus`) |
| `ctx-lh-` / `ctx_lh_` (or spaces) | `ctx[-_ ]lh[-_ ].+` | `ctx[-_ ]rh[-_ ].+` | FreeSurfer aparc; a2009s; as `label_name` writes them |
| Suffix `_L`/`_R` | `.+_L` | `.+_R` | AAL |
| Prefix `lh.`/`rh.` | `lh\..+` | `rh\..+` | |
| Suffix `.left`/`.right` | `.+\.left` | `.+\.right` | |
| `L `/`R ` | `L .+` | `R .+` | nilearn's Destrieux |
| Suffix `-lh`/`-rh` (or a space) | `.+[- ]lh` | `.+[- ]rh` | Tian's subcortex (`aHIP-lh`); as an engine of the apps names it (`aHIP lh`) |
| `<n>Networks_LH_` (or spaces) | `\d+Networks[_ ]LH[_ ].+` | `\d+Networks[_ ]RH[_ ].+` | Schaefer 2018 (`17Networks_LH_VisCent_ExStr_1`; `17Networks LH ...` in an engine of the apps) |

No match: `none`. `hemisphere_rules=()` infers nothing, and rules given
*replace* the defaults. **Precedence** for the `hemi` of a label: the
`hemi` of the label table if not empty (an explicit `none` is a decision);
else a hint for its name, if not empty; else the inference
(`Atlas.hemispheres(hints)`; `Atlas.table` has no hint). The hints of
neurolang_apps.utils.engines.tracts' `label` are the `hemi` of the term mapping (FreeSurfer's
`wm-lh-*` of a wmparc, which no rule names, have theirs there). A warning
names each label for which two of these sources disagree (the inference
counting only when it is not `none`). Known hazards,
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

The engine tracts does no registration. `Space` is a string tag
(`"MNI152NLin2009cAsym"`, `"subject:100307"`); two tags are the same space
iff equal. Atlas and tractogram each carry one; `RelationStore.add_index`
calls `check_same_space`, which raises `SpaceMismatchError` unless
`allow_space_mismatch=True`, in which case it warns. The override is for
"I know they are aligned"; the engine tracts never infers it. The relations that do
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

| # | Reference behaviour | the engine tracts |
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
| 10 | `compute_label_bounding_boxes`: `ndimage.find_objects` (labels start at 1, `stop` exclusive), then **two** corners `[start, stop]` mapped by the linear part and the translation, `BoundingBox` takes their min and max. That is the centre of the first voxel to the centre of the voxel *after* the last (half a voxel off at both ends), and a box only if each world axis depends on one voxel axis (wrong for oblique affines) | `label_extent` (implemented, 6.1.1): the same `find_objects` and the same "box of a box" idea, by **voxel edges** `[start - .5, stop - .5]`, **8 corners** through the full affine, min and max per world axis. Tests transcribe the reference and check the relation exactly (below). **Where scipy is not installed** (it is not a dependency of tract_querier) the reference takes the minimum and maximum of the **voxel centres** instead, `[start, stop - 1]`; `.index_tracts(..., extent="centre")` reproduces that one (6.1.1) |
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
anything. the engine tracts maps the 8 corners of the label's box, which gives the
world box of the *box*: it **contains** the extent of the label's voxels and
is larger than it when the label is not box-shaped (the tests have a
diagonal line to show it, and check both the containment and the equality
with the 8-corner brute force). Tightening it needs the voxels, not the
box: a pass over all the voxels of the atlas (the earlier design) instead of
one `find_objects`; not done, because atlases are in practice aligned or
nearly, and the loose box is safe for the uses planned (a conservative
bound). Open question 15.13.

*Voxel centres, as tract_querier without scipy* (`extent="centre"`).
`compute_label_bounding_boxes` has two paths. With scipy it maps
`[start, stop]` (above). Without it (scipy is not among tract_querier's
dependencies: a `pip install` has none, so this is what most installations
run) it takes the world coordinates of **every voxel of the label** and the
`BoundingBox` is their minimum and maximum: the box of the voxel *centres*,
`[start, stop - 1]` in voxel units, a single voxel a point, a face of the box
at the centre of the outermost voxel. The directional rules compare a
streamline against the faces of the box, so the same streamline is "beyond
the anterior face" of a label with the edge extent and not with the centre
one when it ends within half a voxel of the last voxel's centre. The
extent is a property of the atlas relation `label_extent(a, l, axis, lo,
hi)`, whose rows are built when the atlas enters the program, so the option
sits where the atlas's other choice is, the term mapping: `.index_tracts(t,
atlas, extent="edge"|"centre")`, `"edge"` unless the first indexing by the
atlas says otherwise, then fixed (a later `.index_tracts` of the atlas with
another extent is an error, as with `term_mapping`); in Python
`label_extent_relation(atlas, extent)` and `RelationStore.add_atlas(atlas,
term_mapping, extent)`. Not an `IndexingParams` field, which determines the
cached relations of a tractogram: `label_extent` is not one of them. The
default is unchanged. `"centre"` is exact, against the voxels, under the same
affines as the edge (aligned, flipped or permuted: the tests loop over the
voxels), and under an oblique affine the box of the box of centres, which
contains the centres (tract_querier's, by the voxels, is the tight box of
the centres). The path with scipy, `[start, stop]`, has no option: it is
neither the edge nor the centre but half a voxel beyond the centre at the
`stop` side, which no atlas property justifies. 17.5 measures the three.

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
denominators, can be traversed, and can be an endpoint label. the engine tracts:

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
  effect on `endpoint_label`. It is what makes `only` (`only_in`, 16.2) the
  reference's: the background is a label of no term, so it counts as a label
  outside the term (17.3, tested against the reference on synthetic data
  with background labels; 17.5 on real data).
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
<cache_dir>/<key>/meta.json     # params, the engine tracts version, INDEXING_VERSION, row counts
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

`neurolang_apps.utils.engines.tracts.fetch` (standard library only, a module of the core) downloads
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
`<stem>.trk` of a bundle (a `.trk.gz` gunzipped, which nibabel would also
read, so that every file extracted is a `.trk`) or the whole brain. The download is streamed to a temporary
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

The engine tracts is an engine, not an application: `neurolang_apps.utils.engines.tracts.frontend` has the
mixin of the solver with the commands (below) and the `python_init` of the
engine, `init_engine` (16.3); the engine `tracts` of the apps'
`engines.yaml` puts them together, and an engine of the apps has them by
extending it (`extends: [neurosynth, tracts]`). There is no engine class nor
factory of its own: a subclass would be a parallel class hierarchy, and the
engines of the apps are built from their configuration. (The tests build
the same engine without the atlases of the apps, `neurolang_apps.utils.engines.tracts.tests.engine`.)

The relations are DataFrames, with the column names of 3.1; an engine
renames the columns to positions, so the engine itself does not hold them
(rules bind by position). Because adding a relation **replaces** the
symbol, several atlases or tractograms are concatenated *before* it is
given: that is the `RelationStore`'s job (`add_atlas`, `add_tractogram`,
`add_index` accumulate into one frame per relation). `add_tractogram` gives
the tractogram its index in the ids `x` (the number of tractograms added
before it, kept in `RelationStore.tractograms`), so it comes before
`add_index`, which raises otherwise. From Python, the frames of a store are
given to any engine with `add_tuple_set`:

```python
store = RelationStore()
store.add_atlas(Atlas.from_freesurfer("dk", img, lut, space="subject:100307"),
                "freesurfer_aseg_aparc")
tg = Tractogram.from_file("100307", "tracts.trk", space="subject:100307")
store.add_tractogram(tg, params)
store.add_index(tg, atlas, params, cache_dir="~/.cache/nl_tracts")
nl = NeurolangDL()
for name, frame in store.relations.items():
    nl.add_tuple_set(frame, name=name)
```

#### The commands: the second hand-off path

A Datalog program can also build the relations itself, from files, with the
command of every engine of NeuroLang that loads an atlas, `.load_atlas`
(`neurolang.commands`), and the two that `TractsCommandsMixin` handles
(`neurolang_apps.utils.engines.tracts.frontend`):

```
.load_atlas("fs", "aparc+aseg.nii.gz", "FreeSurferColorLUT.txt",
            kind="freesurfer", space="subject:1")
.load_tractogram("100307", "tracts.trk", space="subject:1")
.index_tracts("100307", "fs", crossing_threshold=2.0, length_threshold=0.0,
              emit_background=1, cache="~/.cache/nl_tracts",
              term_mapping="freesurfer_aseg_aparc")
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

`.load_atlas(name, ...)` takes `kind="freesurfer"` (a colour LUT),
`"bids"` (a `dseg.tsv`) or `"tsv"`, with an image and a table, read as
they are written (relative to where the program runs); `"destrieux"`,
`"schaefer"` or `"tian"`, the atlases of the engines, fetched; `"nilearn"`
with `fetcher=...`, the other options going to the fetcher; and, without a
kind, the form it always had, an image and a CSV at URLs. It adds the
relation `fs(label, region)` of the regions of the atlas and records the
`Atlas` in the engine by name (4.1):

```
.load_atlas("d", kind="nilearn", fetcher="destrieux_2009")
.load_atlas("ho", kind="nilearn", fetcher="harvard_oxford",
            atlas_name="cort-maxprob-thr25-1mm")
```

The atlas of `.index_tracts` is one the engine records by that name:
loaded by the program, by `add_atlas_set` from Python, or one of the engine
(`.index_tracts("100307", "destrieux")`, 4.2), which the engine `tracts`
has: the program then loads only its tractograms, in `MNI152`. An atlas the
engine does not have is an error that lists those it has. The first
`.index_tracts` of an atlas attaches its term mapping (3.2): its option
`term_mapping`, else the default of the atlas's kind; a later one cannot
give another, the relations have those terms; so with its option `extent`
(`"edge"`, or `"centre"`, tract_querier's boxes: 6.1.1). An atlas loaded again under
its name after its first index does not change the relations, which keep
the first. `.index_tracts` is the command that publishes the relations: it
adds the atlas and the tractogram to a `RelationStore` that the solver
keeps, indexes the pair (reading or writing the cache), and gives the
program *all* the relations built so far
(`add_extensional_predicate_from_tuples`, with positions for the column
names, as `.load_csv` does), so the commands can be repeated for more
atlases and tractograms. They fail with an `InvalidCommandExpression` that
names the file or the rule broken: a missing file, an unknown `kind` or
nilearn fetcher, a term mapping that is not one, a command before the one
it needs, a space mismatch (`allow_space_mismatch=1` overrides), a pair
indexed twice, a `length_threshold` that differs from that of the
tractogram's first index (its relations have those streamlines).

The relations are those of 3.1: `streamline(x, t, s)`, `streamline_label(a,
x, l)`, `endpoint_label(a, x, e, l)`, the geometry by `x`, and the catalogue.
A tractogram enters the store at its first `.index_tracts`, so its index in
`x` is the order of the first `.index_tracts` of each tractogram. The
engine also has three functions, typed by their annotations,
over the tractograms its commands have indexed: `streamline_id(t, s)` (the
`x`), `subject_of(x)` (the `t`) and `index_of(x)` (the `s`). They are for
comparisons in a rule body, as other functions of the frontend are:

```
ans(t) :- streamline_label("fs", x, 1), (t == subject_of(x))
```

The join with `streamline(x, t, s)` gives the same.

Where the handler sits matters. The solver of `NeurolangPDL` is
`RegionFrontendCPLogicSolver`, whose `CommandsMixin` has a catch-all that
refuses every command it does not know; the engine's `solver_mixins` puts
`TractsCommandsMixin` *before* it. The matches are exact (no catch-all of its own): any other command
still reaches the refusal, which a test checks. The commands and the rules of
a program are run by one `execute_datalog_program`. The engine `tracts` of
`neurolang-query` (the apps' `engines.yaml`) is the same engine, built from
the YAML: its `solver_mixins` is `TractsCommandsMixin` and its `python_init`
is `neurolang_apps.utils.engines.tracts.frontend`, whose `init_engine` gives it the vocabulary of
section 16, the three functions and the hooks of `.view` (16.3).

## 9. Packaging and layering

### 9.1 Layout

The engine `tracts` is a module of the applications,
`neurolang_apps.utils.engines.tracts`, as the engines `neurosynth` and
`neurostore` are; it was first a workspace package of its own, `nl_tracts`,
and needs nothing more than the dependencies of `neurolang_apps` (numpy,
scipy, nibabel, pandas, pyarrow). The extra `tracts` is kept, empty, for the
commands that name it.

```
packages/neurolang_apps/
  neurolang_apps/utils/engines/tracts/
    README.md, DESIGN.md (this file)
    __init__.py         re-exports the public classes (core only);
                        vocabulary_predicates(), the heads of vocabulary.dl
    terms.py            term-mapping format, validation, shipped keys
    atlas.py            over neurolang.atlases.Atlas: the term mapping of
                        its kind, label, label_term, label_extent,
                        unmapped_labels; ProbabilisticAtlas
    tractogram.py       Tractogram
    ids.py              the id x of a streamline: encode, decode
    indexing.py         IndexingParams, *_relation, RelationStore, build_relations
    cache.py            parquet cache: cache_key, read_cached, write_cached
    fetch.py            the atlases of Yeh, HCP842 and HCP1065: download (7.5)
    vocabulary.dl       the vocabulary of tractography, Datalog rules (16)
    wmql_vocab.py       WMQL to the vocabulary, text to text (17)
    bundles/            the .qry of tract_querier in the vocabulary (17)
    frontend.py         TractsCommandsMixin, init_engine() -- the only
                        module that imports NeuroLang beyond its atlases
    mappings/           shipped term mappings (TSV): freesurfer_aseg_aparc,
                        freesurfer_destrieux, tian_subcortex_s2
    tests/              conftest.py: synthetic volumes, `network` marker;
                        engine.py: the engine without the atlases of the
                        apps; reference_tq.py: tract_querier's indexing,
                        loaded by path, as the oracle of 6.1;
                        uncinate_files.py: an atlas and streamlines for the
                        uncinate
    examples/uncinate.dl  the uncinate in the vocabulary (16.4)
    examples/uncinate_hcp842.dl  its recall on the expert bundles UF_L, UF_R
  tools/tracts/
    make_freesurfer_mappings.py   writes the shipped mappings
    make_bundle_modules.py        writes the shipped bundle modules
    wmql_real_data.py             tract_querier against them (17.5)
    wmql_to_datalog.py            a .qry in the vocabulary (17)
```

### 9.2 `.importlinter`

A `forbidden` contract keeps the core of the engine (`terms`, `atlas`,
`tractogram`, `indexing`, `cache`, `wmql_vocab`, `ids`, `fetch`) free of
`neurolang`, `nl_core`, `nl_expr`, `lark` and `nilearn`, but
`neurolang.atlases`, the atlases of NeuroLang (4.1), whose own `forbidden`
contract forbids them the rest of NeuroLang. `frontend` is exempt: it
extends the solver of `NeurolangPDL` (8.2). At run time, importing the core
runs the `__init__` of `neurolang` (its configuration) and of the packages
of the apps above it, and nothing of the core, the parsers or the frontends
(`test_scaffold`). Nothing in `neurolang` knows about tractography: the
layers contract, with `neurolang_apps` above `neurolang`, says so.

### 9.3 CI

The tests of the engine are those of `neurolang_apps`
(`.github/plan_tests.py`, `.github/workflows/tests.yml`). The `network`
tests fetch nilearn atlases, which `tests.yml` caches (`~/nilearn_data`).

## 10. Public API

Complete signatures and docstrings are in the modules; this is the surface.

```python
# neurolang.atlases (NeuroLang's, 4.1)
Space = NewType("Space", str)
class SpaceMismatchError(ValueError): ...
def check_same_space(atlas_space, other_space, allow_mismatch=False) -> None
@dataclass(frozen=True) class HemisphereRule: pattern: str; hemi: str
HEMISPHERE_RULES: Sequence[HemisphereRule]
def infer_hemispheres(names, rules=HEMISPHERE_RULES) -> Series        # left|right|none
def label_name(value) -> str                  # "-" and "_" as spaces
def load_label_table_freesurfer_lut(path) / _bids_dseg(path) / _tsv(path) /
    _csv(path, **read_csv) -> DataFrame
FETCHERS: dict                                 # destrieux, schaefer, tian
class Atlas:
    def __init__(self, name, labels, affine, table, space="unspecified",
                 background=0, hemisphere_rules=HEMISPHERE_RULES, kind=None,
                 image=None)
    from_freesurfer(name, image, lut, space, **kw) / from_bids_dseg(...) /
    from_tsv(...) / from_csv(..., read_csv=None) /
    from_labels(name, image, labels, space, normalise=False, **kw) /
    from_nilearn(name, fetcher, space="MNI152", **options) /
    fetch(name, kind, data_dir, space="MNI152", **options)
    hemispheres(hints=None) -> Series
    region_names() / label_values() / label_voxels() / points()
    check_same_space(other, allow_mismatch=False) -> None
    content_hash() -> str
    attributes: name, space, kind, labels, affine, background, image,
                table, present_labels, unlisted_labels

# atlas.py
DEFAULT_TERM_MAPPINGS: dict[str, str]      # the kind of an atlas -> key
def atlas_term_mapping(atlas, term_mapping=None) -> DataFrame
def label_relation(atlas, term_mapping=None) -> DataFrame
def label_term_relation(atlas, term_mapping=None) -> DataFrame
def label_extent_relation(atlas, extent="edge") -> DataFrame   # or "centre"
def unmapped_labels(atlas, term_mapping=None) -> DataFrame      # id, name, hemi
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
    add_atlas(atlas, term_mapping=None, extent="edge") / add_tractogram(tractogram, params) /
    add_index(tractogram, atlas, params, cache_dir=None, allow_space_mismatch=False)
def build_relations(tractogram, atlas, params=IndexingParams(), cache_dir=None,
                    allow_space_mismatch=False, term_mapping=None,
                    extent="edge") -> RelationStore

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
class TractsCommandsMixin   # .load_tractogram, .index_tracts (term_mapping=, extent=)
def init_engine(engine, mask=None, data_dir=None, **options) -> None
                    # .import('neurolang_apps.utils.engines.tracts:vocabulary.dl') (16), the functions
                    # streamline_id(t, s), subject_of(x), index_of(x), the
                    # hooks of .view streamline_points(ids),
                    # streamline_background()

# __init__.py
def vocabulary_predicates() -> tuple[str, ...]   # those vocabulary.dl defines
# wmql_vocab.py
def translate(source, include_folders=(), terms=None,
              skip_unsupported=False, title=...) -> Translation
                                        # .text .queries .skipped .terms
def to_vocabulary(source, include_folders=(), **kw) -> str
def bundle(name) -> str                 # a shipped module, BUNDLES
```

## 11. WMQL (a later PR)

A transpiler from WMQL, tract_querier's language, to Datalog over these
relations, with tract_querier's semantics exactly (its query evaluator as
the oracle), is a separate, later PR. It will also be the oracle of the
vocabulary of section 16 (16.5). The translation of WMQL into the
vocabulary of section 16, a predicate per query, is section 17.

## 12. Directional terms

**Implemented** as tract_querier has them, in the vocabulary (`anterior_of`
and the like, 16.2): a streamline is selected by comparing a face of its
bounding box with a face of the union of the boxes of the labels of the term,
and its ends by comparing their coordinates.
That needs no per-pair evaluation and no new relation beyond `label_extent`,
`streamline_extent` and `endpoint_coord` of 3.1, which the engine computes in
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

Tests live in `tests/`, with synthetic data built in the tests
(no downloads): small label volumes (e.g. 20x20x20 with 6 labels and
oblique affines) and tractograms of a few hundred streamlines made by
random walks plus hand-built edge cases.

### 13.1 Unit tests, per module

* `neurolang.atlases` (tests of `neurolang`: `test_atlases*.py`): spaces,
  equal tags pass, unequal raise `SpaceMismatchError`,
  `allow_mismatch=True` warns; each loader on a tiny fixture (a three-line
  LUT, a dseg.tsv, a plain TSV, a CSV, the labels as nilearn gives them, a
  fetcher of the engines); hemisphere rules, each convention of 4.3,
  first-match-wins, no match, the half-paired warning, custom rules, hints;
  ids in the volume but not in the table; regions and points;
  `content_hash` changes with each ingredient and does **not** with the
  name; the kinds of `.load_atlas` (`test_commands.py`).
* `atlas`: `label` / `label_term` / `label_extent` contents, the term
  mapping of a kind, a mapping of raw names over normalised names, the
  hemisphere of the mapping, `label_<id>`; the `ProbabilisticAtlas`
  constructor raises.
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
* `frontend`: the relations of a store have the columns of 3.1, and a
  `NeurolangDL` given them answers the example rules of 3.2.

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
relative `.aabb` import registered beside it) and through the engine tracts with
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

1. **Core**: `terms`, `atlas` (loaders, hemispheres, `label`,
   `label_term`, `label_extent`), `tractogram` with the geometry relations
   (`streamline_extent`, `endpoint_coord`, `streamline_length`), the store, the voxel indexing (`streamline_label`, `endpoint_label`), the
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
6. ~~Is the engine above `neurolang` for anything yet?~~ (9.2.)
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
    background is a label of no term that many streamlines traverse, so
    they are not `only_in` a term: tract_querier's `only`, which is what
    the option is for (17.3, 17.5). **Keep**; `only_in` could leave out the
    labels of no term, which would also leave out the labels a mapping
    misses.
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
    `sorts`): the query assistant and `--list-predicates` saw the
    predicates the engine has, not their documentation, which was in the
    `%` comments of `vocabulary.dl`. **Done**: the apps read the `%%`
    documentation of an engine's rules (16.2), the engine `tracts` declares
    the sorts. `neurolang-query --list-predicates` reads `engines.yaml`
    without building the engine, so it lists the documented predicates of
    the program given with `-f` only, not the vocabulary's.

## 16. The vocabulary of tractography

**Implemented**: `vocabulary.dl`, a Datalog module that
`neurolang_apps.utils.engines.tracts.frontend.init_engine` imports into an engine. It needs nothing but the relations of 3.1.

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
so it is a label outside any term, and a streamline with `crossing_threshold`
% of its points in the background is not `only_in` a term, as in
tract_querier's `only` (17.3). By default, without those rows, it is.

The atlas is not an argument: each predicate holds over every atlas loaded
(`a` is quantified in its rules). With one atlas, as usual, it is that one;
with several, a predicate is the union over them, and `term_box_*` is the
box of the term over all of them (15.31).

Each predicate is documented by the `%%` block before its first rule (the
doc comments of NeuroLang's Datalog): its signature, with the sort of each
argument, `traverses(x: streamline, term: term, h: hemisphere)`, then what it
means; the file reads as the reference of the vocabulary, and the engine
keeps the documentation, which the query assistant is told and the REPL's
`.predicates` shows. The sorts, `streamline`, `end`, `term`, `hemisphere`,
`axis`, `atlas`, `label_id` and `coordinate`, are declared by the engine
`tracts` with their descriptions (a term is one of the atlas's term
mapping, `"left"`, `"right"` or `"none"` a hemisphere...); `streamline` is
the sort `.view` draws as streamlines. A test checks that every predicate is
documented with the sorts of its arguments.

### 16.3 How it enters a program

`init_engine(engine, mask=None, data_dir=None, **options)` gives an engine whose
solver has `TractsCommandsMixin` the rules of the vocabulary (it runs
`.import('neurolang_apps.utils.engines.tracts:vocabulary.dl')`), the three functions of the ids,
and two hooks, the methods that the REPL's `.view` calls on an engine to draw a result of streamline
ids: `streamline_points(ids)`, the points of each `x` (decoded by
`ids.decode` to the tractogram, by the store's load order, and the position
in it), in mm, RAS+; and `streamline_background()`, the label image of the
first atlas indexed (the tractograms are in its space), or None before
`.index_tracts`. It is the `python_init` of the engine `tracts` of the apps (its signature is that of a `python_init`).
The hooks are closures over the solver's state, so that the solver mixin
needs no access to the frontend and the core of the engine tracts no NeuroLang.

So **a program is pure Datalog and the three commands**: the rules of the
vocabulary are the engine's before the program runs. NeuroLang accepts
rules over predicates that do not exist yet (those `.index_tracts`
publishes); it warns that their types cannot be inferred, which
`init_engine` silences, and infers them when the query runs. The other way
considered, the commands adding the rules when they first publish the
relations, puts them in the scope of the submission that ran the command in
the REPL, where `.clear`, or the end of a query's scope, would drop them
while the commands' state stays. On an engine without the commands, the
relations of a store given with `add_tuple_set`,
`.import('neurolang_apps.utils.engines.tracts:vocabulary.dl')` adds the rules.

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
`activated`; it records each in the solver's `atlases`, where
`.index_tracts` finds it, as it finds those of `.load_atlas` (4.2). Its
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
of 16.4, belongs to the PR of the transpiler (11). WMQL translated into the
vocabulary, and compared with tract_querier itself, is section 17.

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

## 17. WMQL in the vocabulary

**Implemented**: `wmql_vocab.py` (`translate`, `to_vocabulary`,
`bundle`), a core module (text in, text out; no NeuroLang, checked by
`.importlinter`), the script `tools/wmql_to_datalog.py FILE.qry`, and the shipped
modules `bundles/*.dl`, the `.qry` files of tract_querier
translated by `tools/make_bundle_modules.py`.

It writes a WMQL program as a person writes it in the vocabulary of
section 16: **one predicate per query**, each a rule or two, documented by
a `%%` block, and a header that says what the module needs (its terms) and
what it leaves out. The block (17.6) gives the predicate's signature with
its sorts, `uf(x: streamline, h: hemisphere)`, the full name of the bundle
and a sentence on what it is, then the WMQL statement it is translated
from. It is not the exact transpiler of
section 11, whose label-set programs reproduce tract_querier tractogram by
tractogram and are thousands of rules; its semantics is the vocabulary's,
natural (16.5), and 17.3 says where that differs from tract_querier's. The
parsing is tract_querier's, as the exact transpiler had it (Python syntax,
lower case, `import` inlined from include folders, `not in` binding looser
than `and` and `or`, `for` over a list of names expanded).

### 17.1 The mapping

| WMQL | Datalog |
|---|---|
| `t.side \|= ...` of labels: ids, a wildcard, an `or` of names of labels | the **term** `"t"`; no predicate. With `terms` (a mapping's), a name that is not a term stands for the terms it is made of |
| `q.side = ...` (and `\|=` of a query) | `q(x, h) :- ...` |
| `q.left = ...` | the rule `q(x, "left") :- ...` |
| `q = ...` | `q(x) :- ...` |
| `t.side`, `t.opposite`, `t.left`, `t` (a term) | `traverses(x, "t", h)`, `traverses(x, "t", o)` with `opposite_hemisphere(h, o)`, `traverses(x, "t", "left")`, `traverses(x, "t", _)` (any hemisphere) |
| `HEMISPHERE.side` (`\|= '*.left'`) | `in_hemisphere(x, h)` |
| `q.side` (a query) | `q(x, h)` |
| `a or b`, `a + b` | `(A \| B)` |
| `a and b`, `a * b` | `A, B` |
| `a not in b`, `a - b`; `not a`, `-a` | `A, ~B`; `~A` (pushed to the atoms: `~(B \| C)` is `~B, ~C`, `~(B, C)` is `(~B \| ~C)`); the positive conditions first, so that `x` and `h` are bound |
| `endpoints_in(t1 or t2)` | `(ends_in(x, "t1", h) \| ends_in(x, "t2", h))` |
| `endpoints_in(E)`, E with `and`, `not in`, directions | the conditions of E on **one** end `e`: `endpoint_in(x, e, "temporal", h), endpoint_anterior_of(x, e, "amygdala", h)`; a new `e` (`e2`...) per `endpoints_in`; under a negation or in a disjunct, `exists(e; ...)` |
| `both_endpoints_in(E)` | E on the end `0` and on the end `1` |
| `only(t)`, `~t` | `only_in(x, "t", h)` |
| `only(E)`, E terms with `or` and `and` | `E, ~q_leaves(x, h)` and `q_leaves(x, h) :- in_hemisphere(x, h), streamline_label(a, x, l), ~term_label(a, l, "t1", h), ...` (x goes through a label of none of E's terms; the union of the labels, as tract_querier's) |
| `anterior_of(t.side)`, ..., `medial_of`, `lateral_of` | `anterior_of(x, "t", h)` ...; inside `endpoints_in`, `endpoint_anterior_of(x, e, "t", h)` ... **`superior_of` and `inferior_of` are anatomical**, the reverse of tract_querier's (15.20): the header of every module says so |

A predicate is named as the query, lower case (`UF.side` is `uf`); a name
of the vocabulary or of a stored relation (`traverses`, `label`, `ans`...)
is prefixed `wmql_`. A query defined again is a new predicate; the earlier
one, which what came before uses, is `q_1`. `Translation.queries` maps each
WMQL name to its atom (`uf.left` to `uf(x, "left")`).

The uncinate of `freesurfer_queries.qry`, generated:

```
uf(x, h) :-
    (traverses(x, "orbitofrontalgyrus", h)
     | traverses(x, "middle_frontal_gyrus", h)
     | traverses(x, "inferior_frontal_gyrus", h)),
    endpoint_in(x, e, "temporal", h),
    endpoint_anterior_of(x, e, "amygdala", h), traverses(x, "insula", h),
    ~traverses(x, "occipital", h), ~traverses(x, "parietal", h),
    ~traverses(x, "cingular", h), ~posterior_of(x, "putamen", h),
    opposite_hemisphere(h, o), ~in_hemisphere(x, o)
```

the rule of `examples/uncinate.dl`, laid out by the width of the line.

### 17.2 What is not translated

What the vocabulary cannot say is left out, with the reason
(`WMQLUnsupported`, or with `skip_unsupported` `Translation.skipped` and the
header), and so is every query that uses it: a directional term of a union
of terms (the box of a union is not a predicate) or of a name without a
side (the box of a term over both hemispheres); `endpoints_in`, `only` or a
direction of a query that is not made of terms; `only` of a query with
`not in`; a label id in a query, or a name of label ids that is not a term
of the mapping given; a `.side` name whose `.left` and `.right` are two
definitions and not a term (`thalamus.side |= thalamus.side or
thalamus_proper.side` of `freesurfer_queries_bsf2016.qry`: `thalamus` is not
a term of `freesurfer_aseg_aparc`, whose `thalamus_proper` has both
spellings of the LUT); a query that only negates (`not a`) or does not bind
its hemisphere (`only(p.side or u)`); `save` and unknown functions.

The shipped modules, with the terms of `freesurfer_aseg_aparc` for the
FreeSurfer files (the JHU files define labels only):

| `.qry` | Predicates | Left out |
|---|---|---|
| `freesurfer_queries` | 33 (and 4 `_leaves`) | none |
| `freesurfer_queries_bsf2016` | 25 (and 7 `_leaves`) | `thalamus` and the 6 `thalamo_*` that use it |
| `freesurfer_queries_bsf2016_toad` | 27 (and 7 `_leaves`; with the queries of the one above, which it imports, its `uf` and `cortico_spinal` redefining theirs, which are `uf_1` and `cortico_spinal_1`) | the same |
| `mori_queries` (JHU's names as terms, no mapping) | 10 | none |

### 17.3 Natural and exact semantics

Where they coincide, tested: each name of WMQL is a term of the mapping,
with the same labels and hemispheres (`HEMISPHERE.left` is then the labels
of hemisphere `left`); each label has a box, so that the box of a term is
its name's; the background is not traversed (`.index_tracts`' default,
6.3). Where they differ, by design:

* **Label sets** are the mapping's (16.5): `subcortical.side` of the file
  has the brain stem (`Brain_Stem`, of no side), the mapping's
  `subcortical` of a side does not, and it adds the caudate, hippocampus
  and accumbens (3.2). `not in subcortical.side` (ioff, soff, ec, slf,
  internal capsule) excludes the streamlines of the brain stem in
  tract_querier, of the caudate, hippocampus and accumbens here.
* **`not a`** is a negation; tract_querier's is a complement, the
  streamlines that go through a label outside `a`. No shipped query uses it.
* **`only`**: tract_querier counts the background (label 0, at least 2 % of
  the points) among a streamline's labels, so that a streamline partly in
  it is never `only` anything; the engine tracts emits no rows for it by default,
  so `only` is a superset. With `.index_tracts(..., emit_background=1)` it
  is the same (the background has no term: a label outside it), tested
  against the reference with background labels in the synthetic data.
* **The boxes of the directions** are those of `label_extent`: voxel edges
  by default, tract_querier's (without scipy) with `.index_tracts(...,
  extent="centre")`, 6.1.1.
* **Vertical directions**, anatomical here (15.20).
* A term without a side is of any hemisphere (`_`); `HEMISPHERE` is
  `label.hemi` (16.5).

### 17.4 Tests

`tests/test_wmql_vocab.py`: the text of each construct, the uncinate, the
layout, redefinitions, prefixes, what is left out and why (and what uses
it), the terms of a mapping, `import` and `for`, the command, the shipped
modules have no command and no `ans`. `tests/test_wmql_vocab_oracle.py`:
29 queries (55 names, every construct of 17.1 but the vertical directions,
checked against numpy) on synthetic data, against tract_querier's own
evaluator (`reference_tq.evaluate`), equal on two seeds.
`tests/test_wmql_vocab_bundles.py`: the shipped modules are the
translation (with `TRACT_QUERIER_DIR`), and `uf(x, h)` of
`freesurfer_queries` is `uncinate(x, h)` of the example on the atlas and
streamlines of `uncinate_files` with 300 random ones, the whole module
running.

Found doing it, two engine bugs, fixed with tests: `_` in a negated atom or
a disjunct was a variable of the rule (`~traverses(x, "cerebellum", _)`
was refused as unsafe); and a rule with k negated atoms compiled in time
2^k (each negation was joined with the difference of the previous ones,
which doubled the relational algebra): `cb`, `ec` and `af`, with 12 to 15
negations, did not finish in an hour; each now takes a few seconds.

### 17.5 On real data

*Measured* (not a test; `tools/wmql_real_data.py` for the two runs and
`tools/wmql_parity_causes.py` for the causes, this Mac, pandas backend,
the whole of one run is in `/tmp/parity`, outside git). Tractogram the
whole brain of DIPY's HCP842 atlas, MNI, without its streamlines of one
point (`whole_brain_MNI_2pts.trk`, 144,641); atlas `wmparc.mgz` of
FreeSurfer's `cvs_avg35_inMNI152` with its colour LUT. tract_querier
itself (0.10.1.dev18, its Python API, as its command line calls it, in its
own environment **without scipy**, which is not among its dependencies and
changes its boxes, 6.1.1): the `.trk` read by nibabel, in RAS+ mm, which is
what the engine tracts reads too, the label image with its affine, crossing
threshold 2 %, no length threshold, every query of the `.qry` kept; a
streamline is its position in the file in both. The vocabulary: the
translation with the terms of `freesurfer_aseg_aparc` (the shipped module)
after the three commands, every predicate at once (`solve_all`), by default
and with the two options that reproduce tract_querier (`.index_tracts(...,
emit_background=1, extent="centre")`, 6.1.1, 6.3).

`freesurfer_queries.qry`, 58 bundle-sides (the 59 of an earlier version of
this table was a miscount): n_tq, n_vocab by default, n_vocab with both
options. **50 of 58 are the same streamlines by default, 58 of 58 with both
options.**

| Tract | Side | n_tq | default | both options |
|---|---|---|---|---|
| cst | left, right | 2636, 3378 | the same | the same |
| cst_pre | left, right | 2464, 3218 | the same | the same |
| cst_post | left, right | 455, 491 | the same | the same |
| internal_capsule_new | left, right | 25, 85 | the same | the same |
| ioff | left, right | 70, 93 | the same | the same |
| soff | left, right | 2, 253 | the same | the same |
| uf | left, right | 724, 810 | the same | the same |
| ec | left, right | 68, 2 | the same | the same |
| slf_i, slf_ii, slf_iii | left | 0, 301, 0 | the same | the same |
| slf_i, slf_ii, slf_iii | right | 0, 407, 104 | the same | the same |
| extreme_capsule | left, right | 55, 0 | the same | the same |
| af | left, right | 5, 0 | the same | the same |
| internal_capsule, cc, cc_5, cc_6 | | 0 | the same | the same |
| thalamo_frontal, _parietal, _occipital | left | 1557, 1137, 232 | the same | the same |
| thalamo_frontal, _parietal, _occipital | right | 1210, 856, 185 | the same | the same |
| striato_frontal, _parietal, _occipital | left | 1471, 376, 85 | the same | the same |
| striato_frontal, _parietal, _occipital | right | 1160, 386, 27 | the same | the same |
| cc_1, cc_2, cc_3, cc_4, cc_7 | | 854, 5224, 276, 697, 4533 | the same | the same |
| **precentral_postcentral** | left, right | 47, 136 | 185, 284 | 47, 136 |
| **cb** | left, right | 1153, 829 | 1576, 1653 | 1153, 829 |
| **mdlf** | left, right | 89, 133 | 102, 191 | 89, 133 |
| **mdlf_probable_sections** | left, right | 76, 0 | 214, 0 | 76, 0 |
| **ilf** | left | 1167 | 1126 | 1167 |
| ilf | right | 10 | 10 | 10 |

The eight that differ by default, each streamline of the symmetric
difference checked against tract_querier's own index (the labels it says
the streamline crosses) and against a run with each option alone:

* the seven of **`only`** (`precentral_postcentral`, `cb`, `mdlf`,
  `mdlf_probable_sections`, both sides where they have streamlines) are
  supersets of tract_querier's, and **every** streamline in excess (138,
  148, 423, 824, 13, 58, 138) has the background among its labels in
  tract_querier's index (at least 2 % of its points in label 0), which
  tract_querier counts in `only`. `emit_background=1` alone makes the seven
  identical (57 of 58); no rule had to change, the background being a label
  of no term (6.3, 16.2, 17.3).
* **`ilf.left`** misses 41 streamlines of tract_querier's, all with a
  point between y = -5.9 and -5.5 mm at the temporal end, in front of the
  hippocampus by tract_querier's box and not by `label_extent`'s: the
  anterior face is at the centre of the last voxel in tract_querier and
  half a voxel further, at its edge, in `label_extent` (6.1.1).
  `extent="centre"` alone makes it identical (51 of 58 with it alone, the
  seven of `only` remaining). The boxes of the 179 labels of the atlas:
  `label_extent` with the centres equals tract_querier's boxes for 179 of
  179 (largest difference 0 mm), with the edges for 0 of 179 (0.5 mm).
* The label sets of `subcortical` (17.3) change nothing here: no
  streamline of `ioff`, `soff`, `ec`, `slf_*` goes through the brain stem
  and not through the caudate, hippocampus or accumbens, or the reverse.

`freesurfer_queries_bsf2016.qry` and its `_toad` variant: 55 queries each,
**12 skipped by the translator** (`thalamo_prefrontal`, `_premotor`,
`_precentral`, `_postcentral`, `_parietal`, `_occipital`, both sides:
"thalamus is not a term, and thalamus.left and thalamus.right are two
definitions", 17.2) and 43 translated: **33 of 43 the same by default (34
for `_toad`), 43 of 43 with both options** for both. The ten that differ by
default (nine in `_toad`, which has not `uf.left`), by the runs with each
option alone: `cb`, `slf_i.right`, `slf_ii`, `dlf` (both sides where
present) are `only` with the background in tract_querier's labels, for all
their streamlines (423, 824, 20, 36, 38, 410, 122), resolved by
`emit_background=1` (40 of 43 with it alone); `af` has both causes: 64 and
124 streamlines in excess, 37 and 68 of them with the background (resolved by
`emit_background=1`) and 27 and 56 without it, resolved by `extent="centre"`,
for its `not in medial_of(supramarginal.side)`; `uf.left` (not `_toad`)
misses two streamlines, resolved by the extent (34 of 43 with it alone).

*With scipy* tract_querier's boxes are `[start, stop]` (6.1.1): its
`uf.left` has 722 streamlines (724 without) and its `ilf.left` 1101 (1167).
Neither option reproduces it: with both, 56 of 58 of `freesurfer_queries`
(`uf.left` 724 against 722, `ilf.left` 1167 against 1101), 49 of 58 by
default. Its index takes 11 s, against 51-57 s for the fallback of a voxel
loop per label.

Time (the tractogram of 144,641 streamlines, each .qry alone):
tract_querier 1.8-2.0 s to load the files, 51-57 s to index (without
scipy; 10-11 s with), 1.2-2.6 s for all the queries, 3.1 GB at peak; the
vocabulary 6.8-7.3 s to load and index (the cache of 7.3 warm), 245-266 s
for all the queries of both sides at once (58 bundles of
`freesurfer_queries`; the same, 43 translated, for the others; with the
options the same, 256-266 s), 32-37 GB at peak (`ru_maxrss`; the 13 GB
of the earlier measurement was on an older branch: not investigated). The
queries are compile-bound, as in 16.6: the Magic Sets rewrite is refused
for these programs (negations), so every predicate of the vocabulary they
read is computed for every term and hemisphere, and the 37 rules, many with
ten negations or more, are translated into relational algebra; a run per
bundle was not measured, it would pay that compilation each time.

### 17.6 The documentation of the bundles

`translate(..., qry_file=...)` documents each predicate with a `%%` block
before its rule: its signature, `x: streamline` and, for a query with a
side, `h: hemisphere`; for the first rule of the predicate, the full name of
the bundle and a sentence on what it selects, read from
`tract_names.tsv` for the `.qry` files of tract_querier (by file,
as one name means different queries in two files: `uf` of
`freesurfer_queries_bsf2016_toad` adds the exclusions that of
`freesurfer_queries_bsf2016` lacks), "The query q of FILE" for another
program; then `Translated from tract_querier's FILE.qry:` and the WMQL
statement, the file being that of the statement (an imported file's
definitions, `uf_1` of the TOAD module, cite
`freesurfer_queries_bsf2016.qry`). An auxiliary `q_leaves` says it is that
of `q`'s `only`.

The names are grounded in the files: the comments of
`freesurfer_queries_bsf2016.qry` name each of its queries ("Cingulum
bundle", "Extreme capsule" for `EmC`, "Middle longitudinal fascicle" for
`dLF`, "Rostrum" ... "Splenium" for `CC_1` ... `CC_7`, "Thalamo-parietal"...),
which name the queries of `freesurfer_queries.qry` of the same
abbreviation and regions (its `cc_k` join much the same regions as `CC_k`);
the labels of the JHU atlas the queries of `mori_queries.qry` use
(`Inferior_fronto_occipital_fasciculus` for `IFO`, `Cingulum_*` for `CG`);
or the name of the query (`extreme_capsule`, `internal_capsule`,
`thalamic_radiations`). The column `source` of the table says which. What
nothing grounds is named `tract_querier's <name>`: `ec` of
`freesurfer_queries.qry` (the file defines the extreme capsule as
`extreme_capsule`, and says nothing of `ec`), `precentral_postcentral` and
`mdlf_probable_sections`; `internal_capsule_new` is called a variant of the
internal capsule, `soff` the superior occipito-frontal fasciculus by analogy
with `ioff` (its definition but for the temporal lobe), and the parts of
Mori's `SLF_t`, `SLF_fp` and `SLF_pt` (temporal, fronto-parietal,
parieto-temporal) are read from their definitions. A test checks that every
predicate of the shipped modules is named and has the sorts of its
arguments.

With the engine `tracts`, the modules loaded (`.import("neurolang_apps.utils.engines.tracts:bundles/
freesurfer_queries.dl")`, or `nl.execute_datalog_program(bundle(...))`),
the query assistant's catalog lists `uf(x: streamline, h: hemisphere) —
Uncinate fasciculus: ...`, and the descriptions of the sorts.
