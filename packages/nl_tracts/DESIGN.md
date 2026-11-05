# nl_tracts: tractography for NeuroLang

Status: **implemented, from files to the answer**: the atlas layer (`spaces`,
`terms`, `atlas`: sections 3.2, 4, 5, and `label_extent` of 6.1), the tractogram
and its relations (`Tractogram`, `streamline_extent`, `endpoint_coord`,
`streamline_length`), the **voxel indexing** (`traverses`, `endpoint`: 6.2),
the parquet cache and the `RelationStore` (7.3), the WMQL transpiler with the
directional terms (section 11), the **Datalog file** that a WMQL program is
turned into (11.7), the commands that build the relations from files when it
runs (8.2) and the `nl-tracts` command. Not done: the relations as a Python
call for many atlases at once beyond what `RelationStore` does, the AAL and
Schaefer mappings, the probabilistic variants (4.4, 12). Numbers below marked
*measured* come from probes run on this repository (NeuroLang `master` at the
time of the split, pandas 3.0.6, pandas backend, an Intel Mac); the rest are
estimates.

Contents: 1 Goals · 2 Decisions taken · 3 Relations · 4 Atlases ·
5 Spaces · 6 Indexing semantics · 7 Data formats and cache ·
8 Handing the relations to the engine · 9 Packaging and layering ·
10 Public API · 11 From tract_querier to Datalog · 12 Directional terms · 13 Test plan · 14 Phases · 15 Open questions

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
algorithms; visualisation; probabilistic atlases (4.4); the variants of the directional terms that are
not tract_querier's (12). (WMQL is a transpiler, section 11.)

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
   2026-11-03: the directional terms of WMQL are required, and implemented as
   tract_querier has them (11, 12); only the variants that are not its are
   future work.*
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
| `streamline(t, s)` | t, s | str, int32 | no | S per tractogram |
| `atlas(a)` | a | str | - | #atlases |
| `label(a, l, name, hemi)` | a, l, name, hemi | str, int32, str, str | yes | ~10^2 per atlas |
| `label_term(a, l, term)` | a, l, term | str, int32, str | yes | ~10^2 per atlas |
| `opposite_hemisphere(h1, h2)` | h1, h2 | str, str | no | 3 (left/right, right/left, none/none) |
| `traverses(a, t, s, l)` | a, t, s, l | str, str, int32, int32 | yes | ~5-15 per streamline |
| `endpoint(a, t, s, e, l)` | a, t, s, e, l | str, str, int32, int8, int32 | yes | 2 per streamline |
| `streamline_extent(t, s, axis, lo, hi)` | t, s, axis, lo, hi | str, int32, int8, float32, float32 | no | 3 per streamline |
| `endpoint_coord(t, s, e, axis, v)` | t, s, e, axis, v | str, int32, int8, int8, float32 | no | 6 per streamline |
| `label_extent(a, l, axis, lo, hi)` | a, l, axis, lo, hi | str, int32, int8, float32, float32 | yes | 3 per label |
| `streamline_length(t, s, mm)` | t, s, mm | str, int32, float32 | no | 1 per streamline |

Changes with respect to the brief, each of which is a refinement, not a
deviation from a decision:

* **`axis` is an int** (0, 1, 2 = x, y, z of the space's world axes, RAS+)
  rather than a string: 3 to 6 rows per streamline times 10^6-10^7
  streamlines makes the column the largest after `traverses`. Constants
  are not a usability problem: the frontend does not need to define them;
  rules write `axis = 1` and the vocabulary is in the documentation.
* **`streamline_length(t, s, mm)` is added.** tract_querier's
  `length_threshold` is applied while indexing; for a query that is made of
  rules (`long(t, s) :- streamline_length(t, s, d), d > 40`) the length has
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
`tests/test_wmql_vocabulary.py` transpiles the file (11.3) and states every
difference between its label sets and the mapping's, exactly:

```
% uncinate fasciculus, any atlas, either hemisphere
uf(a, t, h, s) :-
    traverses(a, t, s, l1), label_term(a, l1, "temporalpole"),
        label(a, l1, _n1, h),
    traverses(a, t, s, l2), label_term(a, l2, "orbitofrontalgyrus"),
        label(a, l2, _n2, h)
% the same, with the two ends in opposite hemispheres
callosal(a, t, s) :-
    endpoint(a, t, s, 0, l1), label(a, l1, _n1, h1),
    endpoint(a, t, s, 1, l2), label(a, l2, _n2, h2),
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
path or the key of a shipped one. Shipped, in `nl_tracts/mappings`, both
generated by `tools/make_freesurfer_mappings.py` (which holds the compact
tables; edit that, not the TSVs):

| Key | Labels |
|---|---|
| `freesurfer_aseg_aparc` | aseg (both `Left-Thalamus` and `Left-Thalamus-Proper`, the LUT's two spellings) and aparc, as `ctx-lh-*`/`ctx-rh-*`; vocabulary and groups from the `.qry` |
| `freesurfer_destrieux` | a2009s, in **both** naming schemes: FreeSurfer's LUT (`ctx_lh_G_precentral`) and nilearn's lateralised atlas (`L G_precentral`), because the mapping is keyed by raw name |

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
`traverses` would otherwise refer to an unknown label. The background has a
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

### 4.3 Hemisphere inference

A `HemisphereRule(pattern, hemi)` has a regular expression that must match
the **whole** raw name, and `hemi` `left` or `right`; the rules are ordered
and the first match wins; `infer_hemispheres(names, rules)` returns a Series
of `left`, `right` or `none`. Defaults, `HEMISPHERE_RULES` (the first five
are the brief's; the sixth is what real atlases needed):

| Convention | Left | Right | Seen in |
|---|---|---|---|
| `Left-`/`Right-` (or a space) | `Left[- ].+` | `Right[- ].+` | FreeSurfer aseg; Harvard-Oxford (`Left Thalamus`) |
| `ctx-lh-` / `ctx_lh_` | `ctx[-_]lh[-_].+` | `ctx[-_]rh[-_].+` | FreeSurfer aparc; a2009s |
| Suffix `_L`/`_R` | `.+_L` | `.+_R` | AAL |
| Prefix `lh.`/`rh.` | `lh\..+` | `rh\..+` | |
| Suffix `.left`/`.right` | `.+\.left` | `.+\.right` | |
| `L `/`R ` | `L .+` | `R .+` | nilearn's Destrieux |

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
`Atlas`, and produce a *probabilistic* relation `traverses_p(a, t, s, l)`
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
| 6 | Endpoint labels are the labels of the first and of the last point, as `ending_tracts_labels[0|1]`, label 0 included | `endpoint(a, t, s, e, l)`, `e` = 0 first, 1 last; **rows with the background label are never emitted** (6.3), so the equivalence tests drop label 0 from the reference's endpoints before comparing |
| 7 | With `length_threshold > 0` the tract list is **filtered before enumeration**, so the ids of the survivors are *renumbered* | **ids are never renumbered**: `s` is the position in the file; filtered streamlines are absent from every relation. The equivalence test maps the reference's ids back through the kept-index list. **The reference's `tract_bounding_boxes` and `tract_endpoints_pos` are computed on all the tracts, unfiltered, so its directional terms are misaligned with the renumbered ids when `length_threshold > 0`** (11.5): not comparable there |
| 8 | Length = sum of segment norms of the polyline, `>= length_threshold` keeps | same, in float64, then stored as float32 in `streamline_length` |
| 9 | Streamlines with fewer than 2 points make the bounding-box step raise | `Tractogram` rejects them at construction, with the id of the first offender |
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
3. `traverses`: build `key = sid * (L + 1) + dense_label`, where
   `dense_label` ranks the atlas's labels (the table order, background
   first) so that `L` is the number of labels, not the largest id;
   `np.unique(key, return_counts=True)` per chunk; divide the counts by
   `lengths[sid]`; keep `count / n >= threshold / 100.`; decode `sid` and
   label. If `S * (L + 1) < 2^28` a `np.bincount` over the same key is used
   instead of the sort. Complexity O(P log P), O(P) memory per chunk.
4. `endpoint`: `lab[offsets[:-1]]` and `lab[offsets[1:] - 1]`: two fancy
   indexings.
5. Streamline extents: `np.minimum.reduceat` / `np.maximum.reduceat` over
   the points at `offsets[:-1]`, per axis; lengths: `np.add.reduceat` of the
   segment norms, with the last segment of each streamline masked out.
6. `label_extent`: one `ndimage.find_objects` for all the labels, the
   `(L, 3)` starts and stops as arrays, the 8 corners of every box mapped by
   the affine in one product, min and max (6.1.1). The only Python loop is
   the unpacking of the slices `find_objects` returns, one per label.

No Python loop over streamlines or over labels; the only loops are over
chunks of points and over the three axes.

### 6.3 Label 0 and the background

The reference treats 0 as a label like any other: it counts in the
denominators, can be traversed, and can be an endpoint label. nl_tracts:

* Points in the background **always count in the denominator** (parity 4).
* **`endpoint` never carries the background label** (decision of the
  maintainer, not a flag): an endpoint whose point is in the background has
  *no row* for that end. "This end is in no region" is the safe negation
  `no_region(a, t, s, e) :- streamline(t, s), end_index(e), ~has_end(a, t, s, e)`
  with `has_end(a, t, s, e) :- endpoint(a, t, s, e, l)`; `end_index(e)`
  is `{0, 1}` (a two-row relation in the rule, or `e = 0`/`e = 1` written
  out).
* `traverses` has rows for the background label **only if
  `IndexingParams.emit_background=True`**. Default is **False**: with 0
  emitted, almost every streamline has a `(a, t, s, 0)` row and `traverses`
  grows by a factor of 1.1-1.3 for facts no rule should use.
  `emit_background=True` reproduces the reference's `traverses` and is what
  the equivalence tests use; it has no effect on `endpoint`.
* `label` has a row for the background only if the label table has one (the
  FreeSurfer LUT does: `0 Unknown`).
* WMQL's `not`/`-` operate on the set of labels that appear in the data,
  which includes 0 in the reference; the translations of 11 state exactly
  which label set they use.

## 7. Data formats and cache

### 7.1 In memory: DataFrames

The relation is a `pandas.DataFrame` with the columns and dtypes of 3.1. The
reasons it is not a CSR matrix:

* the engine's relational algebra sets are DataFrames (*measured*: 8.1) and
  would convert a CSR into one anyway, with the same memory;
* a CSR (streamline x label) matrix is the natural *internal* structure of
  step 3 of 6.2 (`scipy.sparse.csr_array` from the unique keys) and may be
  used inside `traverses_relation`; it is not part of the API;
* the three atlas-independent relations and `endpoint` are dense per
  streamline, and flat arrays are what they are.

The core never builds tuples, sets or dicts per streamline.

### 7.2 Sizes

For 10^6 streamlines of 100 points, a 100-label atlas, 10 traversed labels
per streamline on average:

| Relation | Rows | Bytes/row | Memory |
|---|---|---|---|
| `traverses` | 10^7 | 10 (a) + 10 (t) + 4 + 4 = 28 | 280 MB |
| `endpoint` | 2x10^6 | 10 + 10 + 4 + 1 + 4 = 29 | 58 MB |
| `streamline_extent` | 3x10^6 | 10 + 4 + 1 + 4 + 4 = 23 | 69 MB |
| `endpoint_coord` | 6x10^6 | 10 + 4 + 1 + 1 + 4 = 20 | 120 MB |

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
every rule. At 10^8 rows the two columns are about 2 GB under either dtype
and the integer alternative saves most of it: revisit then (15.3).
`Categorical` is not used because the engine's behaviour on categorical
columns is untested. The code does not choose the dtype: it builds
DataFrames from Python strings and lets pandas decide, so that it behaves
the same on pandas 1.5 (`object`) and 3 (`str`); the tests assert a string
dtype, not the identity of the objects.

### 7.3 On disk: parquet, keyed by content

Only the **atlas-dependent relations that are expensive** are cached:
`traverses` and `endpoint`. The others are `O(S)` numpy reductions that cost
less than reading a file.

```
<cache_dir>/<key>/traverses.parquet
<cache_dir>/<key>/endpoint.parquet
<cache_dir>/<key>/meta.json     # params, nl_tracts version, INDEXING_VERSION, row counts
```

`key = sha256(tractogram.content_hash() ‖ atlas.content_hash() ‖
repr(params) ‖ INDEXING_VERSION)`, hex, first 32 characters.
`Tractogram.content_hash` hashes the points (as float32 bytes), the offsets
and the space tag; the `id` is **not** in it, so two runs of the same file
under different ids share a cache entry and the `t` column is set when the
entry is read. `Atlas.content_hash` hashes the volume, the affine, the label
table, the background id and the space tag, not the atlas name, for the same
reason. `INDEXING_VERSION` (in `cache.py`) is bumped whenever the semantics
of 6 change. Files are written to a temporary name and renamed, so an
interrupted run never leaves a half file that is later trusted; zstd,
row groups of 10^6, dictionary encoding on the string columns (a column of
one repeated string is a few bytes on disk).

Reading is `pandas.read_parquet` with the dtypes of 3.1 restored (parquet
stores `int32`/`int8`/`float32` natively); the `a` and `t` columns are
read back in pandas' default string dtype (7.2), so the memory of 7.2 holds
after a cache hit.

### 7.4 Chunking

`traverses_relation` and `endpoint_relation` process the tractogram in
chunks of streamlines (about 2^24 points) and concatenate, so the peak memory
is the result plus a chunk, not the result plus several copies of the
points. Chunking is internal; the result is one DataFrame.

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
* A rule over 2x10^6-row `traverses` joined with `label_term`, with
  negation, ran in **2.1 s, 436 MB** peak RSS; at 3x10^6 rows a join
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
nl.execute_datalog_program('ans(s) :- traverses("dk", "100307", s, l), ...')
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

`kind` is `freesurfer` (a colour LUT), `bids` (a `dseg.tsv`) or `tsv`. The
paths are read as they are written (relative to where the program runs).
`.index_tracts` is the command that publishes the relations: it adds the
atlas and the tractogram to a `RelationStore` that the solver keeps, indexes
the pair (reading or writing the cache), and gives the program *all* the
relations built so far (`add_extensional_predicate_from_tuples`, with
positions for the column names, as `.load_csv` does), so the commands can be
repeated for more atlases and tractograms. They fail with an
`InvalidCommandExpression` that names the file or the rule broken: a missing
file, an unknown `kind`, a command before the one it needs, a space mismatch
(`allow_space_mismatch=1` overrides), a pair indexed twice, a
`length_threshold` that differs from that of the tractogram's first index
(its relations have those streamlines).

Where the handler sits matters. The solver of `NeurolangPDL` is
`RegionFrontendCPLogicSolver`, whose `CommandsMixin` has a catch-all that
refuses every command it does not know; `TractsSolver` puts
`TractsCommandsMixin` *before* it, and `tracts_engine()` is
`NeurolangPDL(solver_class=TractsSolver)`, the hook `NeurolangPDL` has for
that. The matches are exact (no catch-all of its own): any other command
still reaches the refusal, which a test checks. The commands and the rules of
a program are run by one `execute_datalog_program`. They are not in
`neurolang-query --engine ...`: that registry is YAML-driven with data
downloads, and a `tracts` engine in it is a follow-up (15.26); `nl-tracts run`
is the way for now (11.7).

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
    wmql.py             WMQL -> Datalog transpiler (11); no NeuroLang
    dl.py               the Datalog file of a WMQL program (11.7); no NeuroLang
    cli.py              `nl-tracts dl` and `nl-tracts run`
    atlas.py            Atlas, ProbabilisticAtlas, HemisphereRule, loaders
    tractogram.py       Tractogram
    indexing.py         IndexingParams, *_relation, RelationStore, build_relations
    cache.py            parquet cache: cache_key, read_cached, write_cached
    frontend.py         register(), the commands, tracts_engine() -- the only
                        module that imports NeuroLang (cli.py runs it)
    mappings/           shipped term mappings (TSV): freesurfer_aseg_aparc,
                        freesurfer_destrieux
    tests/              conftest.py: synthetic volumes, `network` marker;
                        reference_tq.py: tract_querier's evaluator, loaded
                        by path, as the oracle of wmql
```

### 9.2 `.importlinter`

* `root_packages` gains `nl_tracts`.
* The layers contract becomes `neurolang_apps > nl_tracts > neurolang >
  nl_core > nl_expr`.
* A new `forbidden` contract: `nl_tracts.{spaces, terms, wmql, dl, atlas,
  tractogram, indexing, cache}` may import none of `neurolang`, `nl_core`, `nl_expr`,
  `lark`, `nilearn`. `frontend` is exempt, and is the only module that may
  import NeuroLang.

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
import `neurolang`, and the forbidden contract is what forbids it. Today
neither `frontend.py` nor `wmql.py` imports `neurolang` (the first uses the
engine duck-typed, the second turns text into text), and the directional
terms, now implemented, did not need it either (they are comparisons over
relations, 12), so the declared dependency on `neurolang` is for what is
still to come (the text-to-query integration, the probabilistic variants of
the directional terms, which read the engine's types) and for the tests; if
that does not come, the layer and the dependency should be reconsidered
(15.6).

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
SHIPPED_MAPPINGS = ("freesurfer_aseg_aparc", "freesurfer_destrieux")
def load_term_mapping(source) -> DataFrame   # DataFrame, path or shipped key
def shipped_mapping_path(key) -> Traversable

# wmql.py
def transpile(source, atlas, include_folders=(), skip_unsupported=False,
              anatomical_vertical=False)
    -> Transpiled   # .text, .program(*names, components=...), .defined,
                    # .saved, .skipped
class QueryPredicates: tracts, labels, end0, end1     # predicate names | None
class WMQLError(ValueError); class WMQLUnsupported(WMQLError, NotImplementedError)

# tractogram.py
class Tractogram:
    def __init__(self, id, points, offsets, space)
    from_file(id, path, space) / from_streamlines(id, streamlines, space)
    n_streamlines / lengths_in_points() / lengths_in_mm() / content_hash()

# indexing.py
@dataclass(frozen=True) class IndexingParams:
    crossing_threshold: float = 2.0
    length_threshold: float = 0.0
    emit_background: bool = False
def points_to_voxels(points, affine, shape) -> ndarray
def traverses_relation(tractogram, atlas, params) -> DataFrame
def endpoint_relation(tractogram, atlas, params) -> DataFrame
def streamline_extent_relation(tractogram, params) -> DataFrame
def endpoint_coord_relation(tractogram, params) -> DataFrame
def streamline_length_relation(tractogram, params) -> DataFrame
class RelationStore:
    add_atlas(atlas) / add_tractogram(tractogram, params) /
    add_index(tractogram, atlas, params, cache_dir=None, allow_space_mismatch=False)
def build_relations(tractogram, atlas, params=IndexingParams(), cache_dir=None,
                    allow_space_mismatch=False) -> RelationStore

# cache.py
def cache_key(tractogram, atlas, params) -> str
def read_cached(cache_dir, key, relation) -> DataFrame | None
def write_cached(cache_dir, key, relation, frame) -> None

# frontend.py
RELATION_NAMES: dict[str, tuple[str, ...]]
def register(engine, store, only=None) -> None
def relation_frames(store) -> dict[str, DataFrame]       # + opposite_hemisphere
class TractsCommandsMixin   # .load_tract_atlas, .load_tractogram, .index_tracts
class TractsSolver(TractsCommandsMixin, RegionFrontendCPLogicSolver)
def tracts_engine(**kw) -> NeurolangPDL                  # solver_class=TractsSolver

# dl.py
class AtlasSource(name, image, table, kind="freesurfer", space, term_mapping, background)
class TractogramSource(id, path, space)
def datalog_file(source, atlas, tractograms, answer=None, include_folders=(),
                 crossing_threshold=2.0, length_threshold=0.0,
                 emit_background=True, cache=None, anatomical_vertical=False,
                 title=None) -> str

# cli.py     (console script `nl-tracts`)
def main(argv=None) -> int      # nl-tracts dl PROGRAM --atlas ... --tractogram ...
                                # nl-tracts run FILE.dl [--format csv] [--limit N]
```

## 11. From tract_querier to Datalog: `nl_tracts.wmql`

**Implemented** (a transpiler, which turns text into text and imports
nothing of NeuroLang; covered by the contract of 9.2). It is how the
vocabulary of 3.2 was checked against tract_querier's, and what the
standard bundles of the phase-3 tutorial will be transcribed with.

```python
transpile(source, atlas, include_folders=(), skip_unsupported=False)
    -> Transpiled   # .text, .program(*names, components=...),
                    # .defined, .saved, .skipped
```

### 11.1 Output

In the reference every query value is a **triple** (`FiberQueryInfo`): the
streamlines `tracts`, a set of `labels`, and two sets of streamlines
`tracts_endpoints[0|1]` (those whose first / last point is in one of the
labels). A WMQL name `x` becomes four predicates over `tractogram(t)`,
`traverses(a, t, s, l)` and `endpoint(a, t, s, e, l)` and, for the directional
terms, `label_extent(a, l, axis, lo, hi)`, `streamline_extent(t, s, axis,
lo, hi)` and `endpoint_coord(t, s, e, axis, v)`, with the atlas `a` a
constant of the program (WMQL labels are the ids of an atlas, as
`FreeSurfer.qry` has them). **The label set carries the tractogram**, since
the labels of a directional result are those its streamlines traverse, which
depends on the data of each tractogram, as in the reference, where each run
has one:

| Predicate | Component |
|---|---|
| `q_x(t, s)` | the streamlines |
| `q_x__label(t, l)` | the labels, for each tractogram |
| `q_x__end0(t, s)`, `q_x__end1(t, s)` | streamlines whose first / last point is in the labels |

`x.left` is `q_x_left`; the prefix keeps WMQL names (`label`, `endpoint`)
apart from the relations; two names that would mangle alike raise. A
component that is empty whatever the data (`not A` has no endpoints) is
`None` and has no predicate: it propagates (an intersection with it is
empty, a union is the other operand), so no rule is written for it.
`Transpiled.program(*names, components=...)` slices the program to what
the names need, e.g. only the label sets.

### 11.2 Semantics, as `tract_querier` has them

Read from `query_processor.py` and checked against it (13.2).

| WMQL | Reference semantics (componentwise on the triple) | Datalog |
|---|---|---|
| `7` | tracts = those traversing 7; labels = {7}; end_e = those whose end e is in 7 | `E(t,s) :- traverses(a,t,s,7)`; `E_label(t,7) :- tractogram(t)`; `E_end0(t,s) :- endpoint(a,t,s,0,7)`, same for 1 |
| `A or B`, `A + B` | union of tracts, labels, endpoints | two rules per predicate, same head |
| `A and B`, `A * B` | tracts ∩, **labels ∪**, endpoints ∩ | `E(t,s) :- A(t,s), B(t,s)`; `E_label(t,l)` from both; `E_end0(t,s) :- A_end0(t,s), B_end0(t,s)` |
| `A - B`, `A not in B` | set difference of the tracts, of the labels **and** of the endpoint sets; `a not in b not in c` is sequential | `E(t,s) :- A(t,s), ~B(t,s)`; the same for labels and ends |
| `not A`, `-A` (unary) | tracts traversing **some label outside A's labels**, over the labels *present in the data*; labels = that complement; **endpoints empty** | `U(t,l) :- traverses(a,t,s,l)`; `E_label(t,l) :- U(t,l), ~A_label(t,l)`; `E(t,s) :- traverses(a,t,s,l), E_label(t,l)`. **Not** `~A(t,s)` |
| `~A` (invert) | tracts of A whose **every** traversed label is in A's labels; labels as A; **endpoints empty** | `outside(t,s) :- A(t,s), traverses(a,t,s,l), ~A_label(t,l)`; `E(t,s) :- A(t,s), ~outside(t,s)` |
| `only(A)` | the same tracts and labels as `~A`; the endpoint sets **kept**, filtered to those whose end label is in A's labels | as `~A`, and `E_end0(t,s) :- A_end0(t,s), endpoint(a,t,s,0,l), A_label(t,l)` |
| `+A` | A | A |
| `endpoints_in(A)` | tracts = end0 ∪ end1 of A; labels, endpoints as A | `E(t,s) :- A_end0(t,s)`; `E(t,s) :- A_end1(t,s)` |
| `both_endpoints_in(A)` | tracts = end0 ∩ end1 of A | `E(t,s) :- A_end0(t,s), A_end1(t,s)` |
| `"pattern*"` | `fnmatch` over the names defined **so far**, union; no match: empty | resolved when transpiling, in order |
| `x.side = e`, `.opposite` | the statement is repeated for `x.left` and `x.right`, `side` and `opposite` swapped | the same, at transpile time |
| `for v in [a, b]` / `'p*'` | the body repeated with the **names** equal to `v` replaced (not attributes, not the names of targets) | expanded at transpile time |
| `import f.qry` | the file's statements, in place | inlined, with `include_folders` |
| `anterior_of(A)` etc. | six terms, 11.5: the streamlines (of **all** the tractogram, not of `A`) beyond a face of the union of the boxes of the labels of `A`; the ends likewise; labels = those the streamlines traverse | box aggregate and comparisons, 11.5 |
| `save(...)`, unknown functions | (`save` does not work in the reference) | `WMQLUnsupported` |

Rules of the language the transpiler reproduces: **everything is lower
case** (names, attributes, strings, but not `or`/`and`); evaluation is
**sequential** (a name must be defined before use); `|=` **overwrites**, it
does not accumulate; `=` also saves the query for output, as does a bare
name statement; `not in` binds *looser* than `and` and `or`
(`a and b not in c` is `(a and b) not in c`), by a port of the reference's
rewrite. A name defined again is **a new definition**: later statements read
it, earlier ones keep reading the old one (the old predicates are renamed
`q_x__v1`); the reference's evaluation is the same, and Datalog's union of
same-head rules would not be.

The traps, because they are the point of the oracle: (i) `not` and
`only`/`~` act on **label sets**, with the labels present in the data as
universe, not on streamline sets; plain Datalog `~` is the streamline-level
difference; (ii) the label set of `A and B` is a **union**, so
`only(A and B)` keeps streamlines whose labels lie in `L_A ∪ L_B`; (iii) the
data's label universe includes the background (6.3), so both traps need
`emit_background=True` to agree with the reference, and queries whose label
set contains the background *and* that read the endpoint predicates are out
of the equivalence scope (the endpoint relation has no background rows);
(iv) `not` and `~` give **empty endpoint sets**, `only` filters them: a
translation that treats them alike is wrong (an earlier version of this
section did, until the transpiler's tests were written against the
reference).

`skip_unsupported=True` leaves out the queries that use an unsupported
construct (`save`, an unknown function), and those that use them, and says
which and why (`Transpiled.skipped`). The whole of `freesurfer_queries.qry`,
the four directional bundles (`af`, `ec`, `ilf`, `uf`) included, transpiles
without it.

### 11.3 What the transpiler found: the vocabulary against the file

`tests/test_wmql_vocabulary.py` transpiles `freesurfer_queries.qry` (over
1500 names) and evaluates its label sets with NeuroLang, over an atlas made of
the colour LUT of FreeSurfer 6, against `label_term`. The first comparison,
before the mapping was changed, showed **168 differences**, a few of them
artifacts of comparing names without a side (settled below); the real ones
were these, and the mapping was changed:

* **White matter.** Each Desikan name of the file is the cortex *and* the
  white matter under it (`bankssts.left |= ctx_bankssts.left or
  wm_bankssts.left`: ids 1001 and 3001), with `ctx_` and `wm_` for either
  alone. The mapping had the cortex only; it now has `wm-lh-*`/`wm-rh-*`
  (the largest class: 90 of the 168 were only that), and the terms
  `ctx_<name>` and
  `wm_<name>`. Without it, a bundle transcribed from the file selects
  different streamlines on a wmparc atlas.
* **Terms the file's queries use and the mapping lacked**:
  `centrum_semiovale` and `unsegmentedwhitematter` (5001, 5002),
  `corpuscallosum` in `cingular`, `cingular_cortex`, `temporal_frontal`,
  `frontal_frontal_section`, the aliases `middlefrontalgyrus` and
  `inferiorfrontalgyrus`, `cerebellum_exterior`, `cerebral_exterior`, and
  the `Corpus_callosum` labels of aparc+aseg's extension (1101, 2101, 3101,
  4101).
* **`striatum`** (putamen, pallidum, caudate), and `subcortical` completed
  (3.2).

What still differs is **13 names**, all stated in the test, in four
classes: (1) the file's *unlateralised* names are single labels of an old LUT
(`Amygdala` is id 218, not the two sides), while a sided term of ours without
a side is the union of both: the sided names agree; (2) `corpus_callosum`:
the file has id 86, which this LUT has since given to another label, and we
add the five subdivisions of the modern aseg (251-255); (3) `subcortical`:
completed, and the legacy thalamus id 9; and **`subcortical.left` of the file
includes the brain stem**, which has no side, so a rule that transcribes
`X.left` from a group with midline members must select `hemi` left *or
none*; (4) `Left-Thalamus` is a legacy label in FreeSurfer 6 (9) and the
thalamus proper in FreeSurfer 7 (10): ours maps the name to
`thalamus_proper`, harmlessly for 6 (no volume has 9) and needed for 7.
The ids of these differences are those of the LUT of FreeSurfer 6; the test
skips on another.

### 11.4 The algorithm

`transpile(source, atlas, ...)` is a compiler with four stages: the program
is parsed and normalised as a syntax tree; its statements are interpreted in
order, which expands what can be expanded statically; each expression is
compiled to a `QueryPredicates` by set algebra on its four components, a rule
being emitted for each operation that is not trivial; and the predicates are
named. Nothing is evaluated: the output is text for NeuroLang.

**Stage 1: parse.** `ast.parse`: WMQL is Python syntax. A syntax error is a
`WMQLError` with the line.

**Stage 2: two normalising passes** (ports of tract_querier's
`RewritePreprocess` and `RewriteChangeNotInPrescedence`, so the tree is the
one the reference evaluates).

1. *Lower case and imports.* Every `Name.id`, `Attribute.attr` and string
   constant is lower-cased (so `Only(Alpha)` is `only(alpha)`; the operators
   `or`, `and`, `not` are Python keywords and stay case-sensitive). An
   `import f.qry` is replaced, in the statement list, by the statements of
   the file, found under `include_folders` (the current folder if none),
   which are parsed and normalised in turn (so their imports are expanded).
2. *`not in` binds looser than `and` and `or`.* Python reads `a and b not in
   c` as `a and (b not in c)`; WMQL reads `(a and b) not in c`. For a
   `BoolOp(op, [v0, ..., vk])`: find the first `vi`, `i >= 1`, that is a
   comparison whose first operator is `not in`; if there is none, leave the
   node. Otherwise build `Compare(left=BoolOp(op, [v0, ..., v(i-1),
   vi.left]), ops=vi.ops, comparators=vi.comparators)`; if values remain
   after `vi`, the new comparison becomes the first value of a `BoolOp(op,
   [new, v(i+1), ...])`, which is rewritten again (so a second `not in` is
   hoisted likewise). `a and b not in c and d` (Python: `And(a, b not in c,
   d)`) becomes `((a and b) not in c) and d`. A comparison chain `x not in y
   not in z` stays one node with two comparators, applied in order. **The
   rewrite does not descend into the values of a `BoolOp` that has no such
   comparison among its own**, as in the reference: in `a or b not in c and
   d` Python's grouping `a or ((b not in c) and d)` stands, since the
   comparison is a value of the inner `and`, not of the `or` (the oracle
   checks that this is what the reference does).

**Stage 3: the statement walk.** The state is a table `defined` (name to the
`QueryPredicates` of its *current* definition), the list of all the
definitions of each name (`versions`), the list of saved names, and, with
`skip_unsupported`, the table of skipped names. The statements are taken in
order, as in the reference (evaluation is sequential: a name must be defined
before it is used).

| Statement | Action |
|---|---|
| `x = e` | `define(x, e, saved=True)` |
| `x \|= e` | `define(x, e, saved=False)`: a plain definition, **not** a union with the earlier `x` |
| `x` (a bare name) | mark `x` saved (an error if unknown) |
| `x.attr = e` | the name is `"x.attr"` |
| `x.side = e` | the statement is copied twice, and in each copy every attribute `side` and `opposite` of the *whole statement* (target included) is replaced, by `left` and `right` in the first copy and by `right` and `left` in the second; each copy is executed as a statement, so the target becomes `x.left` and `x.right` |
| `for v in [a, b]:` / `for v in 'p*':` | the items are the listed names, or the names of `defined` that match the pattern (a snapshot at that point); for each item the body is copied and **the `Name` nodes whose id is `v`** (not attributes, not other names) are replaced by the item, then executed |

Two consequences of those rules, both the reference's: a loop whose body
defines `f_i` defines the same `f_i` once per item (only the *name* `i` is
substituted), and a name defined again replaces the earlier one for the
statements that follow (the transpiler keeps the earlier predicates for the
statements that already read them, below).

**Stage 4: expressions.** `expr(node)` returns a `QueryPredicates`, four
predicate names or `None`. *`None` means that the component is empty
whatever the data*; it is not a predicate and no rule is emitted for it. Each
set operation is done by one of three functions that know it:

| `a`, `b` | `union` | `intersection` | `difference` `a \ b` |
|---|---|---|---|
| `None`, `None` | `None` | `None` | `None` |
| `None`, `B` | `B` | `None` | `None` |
| `A`, `None` | `A` | `None` | `A` |
| `A`, `A` (the same predicate) | `A` | `A` | `None` |
| `A`, `B` | new `P`, rules `P :- A`; `P :- B` | new `P :- A, B` | new `P :- A, ~B` |

A new predicate is named `qi_N` with `N` increasing, and the arguments of a
component are `(t, s)` for streamlines and `(t, l)` for labels. The
operations, each on the four components (11.2 has the semantics):

* `Name`, `Attribute(Name, attr)`: look `name` or `name.attr` up in
  `defined`; an unknown name is an error (or, if it was skipped, a skipped
  query that depends on a skipped one).
* A string: the names of `defined` that `fnmatch` the pattern are united,
  in the order of `defined` (nothing matching: all four `None`).
* A number `n`: four predicates, `traverses(a, t, s, n)`;
  `tractogram(t)` gives `(t, n)` for the labels; `endpoint(a, t, s, 0, n)` and
  `endpoint(a, t, s, 1, n)`. Other numbers than integers are an error.
* `or`, `+`: `union` on the four components, folded left over the operands.
  `and`, `*`: `intersection` on the streamlines and the endpoint sets,
  **`union` on the labels**. `-` and `not in`: `difference` on all four.
* `+x`: `x`.
* `not x`, `-x` (**complement**): the *universe* `U(t, l) :- traverses(a, t,
  s, l)` is emitted once (the labels the streamlines of each tractogram
  traverse, the background included when it is a row of `traverses`); the
  labels are `difference(U, x.labels)`; the streamlines are `C(t, s) :-
  traverses(a, t, s, l), labels(t, l)`; the endpoint sets are `None`.
* `~x` (**invert**): `outside(t, s) :- x.tracts(t, s), traverses(a, t, s,
  l), ~x.labels(t, l)` (without the negation if `x.labels` is `None`: every
  traversed label is outside), which starts from the streamlines of `x`, so
  that the join and its result are of their size and not of all of
  `traverses`; the streamlines are `difference(x.tracts, outside)`; the labels
  are `x.labels`; the endpoint sets are `None`.
* `only(x)`: as `~x` for the streamlines and the labels; **each endpoint set
  is kept**, `E_end_i(t, s) :- x.end_i(t, s), endpoint(a, t, s, i, l),
  x.labels(t, l)` (the end label must be in the labels), `None` if `x.end_i`
  or `x.labels` is.
* `endpoints_in(x)`: streamlines `union(x.end0, x.end1)`, labels and endpoint
  sets of `x`. `both_endpoints_in(x)`: `intersection`.
* A directional term: 11.5.
* Anything else (`save`, an unknown function, a call with other than one
  argument, a comparison other than `not in`, another operator) is an error;
  `WMQLUnsupported` for `save` and the unknown functions.

**Definition.** `define(name, value, saved)` compiles `value`, binds `name`
to the resulting `QueryPredicates`, and appends them to the versions of
`name`. It emits **no rule of its own**: a name given by another name
(`x = a`) shares its predicates. That is safe because a rule is only ever
emitted for a *fresh* predicate (the numbering increases, and the two rules
of a union are emitted together), so no head gets a rule after the
predicates are bound, and a later redefinition of a name cannot extend what
an earlier use read. (An earlier version copied every component into a
predicate of its own; the copies were extra steps for the engine, each a
pass over a possibly large relation, and are now made only at the output.)

**Naming (`finish`).** The mangled form of a name replaces `.` by `_`
(`x.left` is `x_left`) and must be `[a-z0-9_]+`; two names with one mangled
form (`x.left`, `x_left`) are an error. The last definition of a name is
`q_<mangled>` (`q_<mangled>__label`, `__end0`, `__end1` for the other
components); the earlier definitions are `q_<mangled>__v1`, `__v2`, ... .
Each public predicate is an **alias rule** `q_x(t, s) :- qi_N(t, s)` for the
internal predicate of that component; the prefix `q_` keeps WMQL names
(`label`, `endpoint`, `ans`) apart from the relations and the keywords of
Datalog; the intermediate predicates are `qi_N`, which cannot collide with a
`q_` name. The rules are kept as structures, with the extensional atoms
(`traverses`, `endpoint`, `tractogram`, `label_extent`, `streamline_extent`,
`endpoint_coord`) marked.

**Output and slicing.** `Transpiled.text` is all the rules and the aliases of
every version of every name. `Transpiled.program(*names, components=...)`
takes the alias rules of the wanted components of the names and the reverse
dependency closure of their bodies (a rule depends on the non-extensional
atoms of its body); the rules of the closure are written in their original
order, then the aliases. So **a component is computed only if it is wanted or
another wanted component needs it**: the streamlines of a bundle do not pay
for the labels of its directional terms, a join of the streamlines with
`traverses` (the difference between 3.8 s and 0.5 s for `anterior_of(a)` at a
million streamlines, 11.6).

**Properties of the output.** (1) *Non-recursive*: a rule only reads
predicates created before it (the numbering increases), so the program is a
DAG and trivially stratified; the negations (`~`) are over predicates of a
lower stratum, as is the aggregate of the directional terms. (This relies on
the engine scheduling a rule that reads a predicate only after *all* the rules
of that predicate, which it did not do for predicates with rules at different
depths, such as a union of a union; fixed in `stratify`, 11.6.) (2) *Safe*:
every variable of a head or of a negated atom is bound by a positive atom
of the body. (3) *Size*: at most one rule per non-`None` component for each
operation (four for `or`, `and` and `not in`, about five for `only`, five
for a directional term: the box, the streamlines, two ends, the labels),
plus one alias per component at the output, so the number of rules is
proportional to the number of nodes of the tree. (4) *Cost*: NeuroLang
compiles a rule in about 55 ms and evaluates it in a pass over its inputs;
11.6.

**A worked example.** The program (an uncinate-like bundle; the labels are
made up):

```
amy.left = 18
amy.right = 54
tmp.left = 5
tmp.right = 6
uf.side = endpoints_in(tmp.side and anterior_of(amy.side)) not in only(tmp.opposite)
```

Stage 3 expands the last statement into `uf.left = endpoints_in(tmp.left and
anterior_of(amy.left)) not in only(tmp.right)` and the same with the sides
exchanged. `transpile(source, atlas="fs").program("uf.left")` is, for the
left bundle (31 rules; 16 with `components=("tracts",)`):

```
% the labels of amy.left (its streamlines are not read), then tmp.left
% (labels, ends) and tmp.right (streamlines, labels, ends)
qi_2(t, 18) :- tractogram(t)
qi_10(t, 5) :- tractogram(t)
qi_11(t, s) :- endpoint("fs", t, s, 0, 5)
qi_12(t, s) :- endpoint("fs", t, s, 1, 5)
qi_13(t, s) :- traverses("fs", t, s, 6)
qi_14(t, 6) :- tractogram(t)
qi_15(t, s) :- endpoint("fs", t, s, 0, 6)
qi_16(t, s) :- endpoint("fs", t, s, 1, 6)
% anterior_of(amy.left): the face of the box, per tractogram, then the tests
qi_17(t, max(hi)) :- qi_2(t, l), label_extent("fs", l, 1, lo, hi)
qi_18(t, s) :- streamline_extent(t, s, 1, lo, hi), qi_17(t, box), (hi > box)
qi_19(t, s) :- endpoint_coord(t, s, 0, 1, v), qi_17(t, box), (v > box)
qi_20(t, s) :- endpoint_coord(t, s, 1, 1, v), qi_17(t, box), (v > box)
qi_21(t, l) :- qi_18(t, s), traverses("fs", t, s, l)
% tmp.left and anterior_of(...): labels united, the ends intersected
qi_23(t, l) :- qi_10(t, l)
qi_23(t, l) :- qi_21(t, l)
qi_24(t, s) :- qi_11(t, s), qi_19(t, s)
qi_25(t, s) :- qi_12(t, s), qi_20(t, s)
% endpoints_in: the streamlines are the union of the two end sets
qi_26(t, s) :- qi_24(t, s)
qi_26(t, s) :- qi_25(t, s)
% only(tmp.right): from its streamlines only; its ends filtered
qi_27(t, s) :- qi_13(t, s), traverses("fs", t, s, l), ~qi_14(t, l)
qi_28(t, s) :- qi_13(t, s), ~qi_27(t, s)
qi_29(t, s) :- qi_15(t, s), endpoint("fs", t, s, 0, l), qi_14(t, l)
qi_30(t, s) :- qi_16(t, s), endpoint("fs", t, s, 1, l), qi_14(t, l)
% not in: the difference of the four components
qi_31(t, s) :- qi_26(t, s), ~qi_28(t, s)
qi_32(t, l) :- qi_23(t, l), ~qi_14(t, l)
qi_33(t, s) :- qi_24(t, s), ~qi_29(t, s)
qi_34(t, s) :- qi_25(t, s), ~qi_30(t, s)
% the public names of uf.left: one alias per component
q_uf_left(t, s) :- qi_31(t, s)
q_uf_left__label(t, l) :- qi_32(t, l)
q_uf_left__end0(t, s) :- qi_33(t, s)
q_uf_left__end1(t, s) :- qi_34(t, s)
```

(The `%` lines are added here to read it; the program has none. The numbers
`qi_N` skip because the slice leaves out the rules `uf.left` does not need,
such as the streamlines of `amy.left`.) `q_uf_left(t, s)` is the answer; a
query that asks for the streamlines of one subject is `ans(s) :-
q_uf_left("100307", s)`.

### 11.5 The directional terms

`anterior_of`, `posterior_of`, `superior_of`, `inferior_of`, `medial_of` and
`lateral_of`, as `process_relative_term` of the reference has them (the
transcription is `_DIRECTIONS` and `_SIDED_DIRECTIONS` in `wmql.py`, and the
oracle tests it on random geometry).

**What the reference does.**

1. The argument must be a **name**, or `name.left` / `name.right`; anything
   else (`anterior_of(a or b)`, `anterior_of(x.foo)`) is a syntax error. Only
   its **label set** is read, never its streamlines.
2. The box is the union of the boxes of those labels: the minimum of the
   lows and the maximum of the highs, per axis (`BoundingBox.union`).
3. The result streamlines are taken among **all** the streamlines of the
   tractography, whether or not they belong to the argument: a streamline is
   selected if one face of its own bounding box is beyond the same-named face
   of the box (below); an end, if its coordinate on that axis is.
4. The labels of the result are the union of the labels the selected
   streamlines traverse. The comparisons are strict.

| Term | Face of the box | Tested on | Operator |
|---|---|---|---|
| `anterior_of(A)` | `hi` on y (the maximum) | `hi` of the streamline | `>` |
| `posterior_of(A)` | `lo` on y | `lo` | `<` |
| `medial_of(A.left)` | `hi` on x | `hi` | `>` |
| `medial_of(A.right)` | `lo` on x | `lo` | `<` |
| `lateral_of(A.left)` | `lo` on x | `lo` | `<` |
| `lateral_of(A.right)` | `hi` on x | `hi` | `>` |
| `superior_of(A)` | `hi` on z | `hi` | **`<`** |
| `inferior_of(A)` | `lo` on z | `lo` | **`>`** |

(Axes 0, 1, 2 are x, y, z of the space, RAS+; `lo` and `hi` are the columns
of `label_extent` and `streamline_extent`.) The first six rows read as
"extends beyond that face of the box"; **the last two do not**:
`superior_of` selects the streamlines whose top is *below* the top of the
box, and `inferior_of` those whose bottom is *above* the bottom of the box,
the reverse of the other directions. They are the reference's, **reproduced
as they are**, because the reference is the oracle; none of the shipped
`.qry` files uses them (`anterior_of`, `posterior_of` and `medial_of` are the
ones used), and no test of tract_querier covers a directional term. The table
is one constant in `wmql.py`, and `transpile(..., anatomical_vertical=True)`
gives `superior_of` as `hi >` and `inferior_of` as `lo <`. Which is wanted is
15.20.

**Compilation.** For `T(A)` with the face `f`, the axis `k` and the operator
`op` of the table, and `A.labels` the label predicate of the argument:

```
box(t, max(hi)) :- A_label(t, l), label_extent(a, l, k, lo, hi)   % min(lo) for a lo face
T(t, s)         :- streamline_extent(t, s, k, lo, hi), box(t, v), (hi op v)     % lo op v for a lo face
T_end0(t, s)    :- endpoint_coord(t, s, 0, k, w), box(t, v), (w op v)
T_end1(t, s)    :- endpoint_coord(t, s, 1, k, w), box(t, v), (w op v)
T_label(t, l)   :- T(t, s), traverses(a, t, s, l)
```

The aggregate is NeuroLang's head aggregation (`max`, `min`) over a join,
grouped by `t`, so there is one box per tractogram; the comparisons are
parenthesised, as the syntax asks.

**Where it differs from the reference.**

* *A label without a box*: the background, or a label absent from the volume.
  The reference raises `TractQuerierLabelNotFound`; the join drops it, and
  the box is that of the other labels. The oracle tests avoid it.
* *No label, or no box at all*: the reference fails with `StopIteration` on
  an empty label set and returns an empty query when there are no boxes; here
  both give an empty result (`None` for a statically empty argument).
* *An un-sided `medial_of(x)` or `lateral_of(x)`*: the reference fails with an
  unbound variable; here a `WMQLError` that says what is needed.
* *The boxes come from other numbers.* The reference's label boxes are two
  corners `[start, stop]` of the voxels; ours are the voxel edges (6.1.1), so
  on a real atlas the faces differ by half a voxel. The oracle feeds the same
  numbers to both, to test the query and not the convention.
* *Float32*: `lo` and `hi` are float32 (3.1); a comparison that is a tie in
  float64 can differ. The oracle's coordinates are multiples of 0.25.
* *`length_threshold` in the reference*: its `tract_bounding_boxes` and
  `tract_endpoints_pos` are computed on **all** the tracts, while its other
  indices are renumbered after the length filter (6.1, parity 7), so its
  directional terms are misaligned when `length_threshold > 0`. Ours keep the
  ids (`streamline_extent` has the same streamlines as `traverses`), so the
  two cannot be compared there, and ours are the ones that are right.

### 11.6 Performance

*Measured* with `tools/bench_wmql.py`: the same synthetic indexing (100
labels, about 8 labels per streamline, one tractogram, one atlas) goes to
tract_querier's `EvaluateQueries` and, as the transpiled program, to
NeuroLang (pandas backend, one machine, one run each). The reference's
indexing (its dictionaries) is built beforehand and not timed, as is the
building of the relations, which is a vectorised `np.unique` once cached
(6.2, 7.3, not yet written); the table is the cost of *a query*. "All four"
computes the four components, as the reference always does; "streamlines
only" is what a user usually wants, and what slicing makes cheaper.

| Program | tract_querier | NeuroLang, all four | NeuroLang, streamlines only | Ratio, streamlines only |
|---|---|---|---|---|
| **1M streamlines (7.7M `traverses` rows)** | | | | |
| `a and b not in c` | 0.023 s | 1.75 s | 0.34 s | 15x |
| `only(a or b)` | 0.109 s | 5.93 s | 4.94 s | 45x |
| `not a` | 0.356 s | 2.82 s | 2.77 s | 8x |
| `endpoints_in(a) and endpoints_in(b)` | 0.010 s | 0.60 s | 0.40 s | 40x |
| `anterior_of(a)` | 0.533 s | 3.82 s | **0.51 s** | **1x** |
| uncinate-like bundle (11.4) | 1.021 s | 9.27 s | 5.67 s | 5.6x |
| **100k streamlines** | | | | |
| `a and b not in c` | 0.001 s | 1.05 s | 0.17 s | ~170x |
| `only(a or b)` | 0.009 s | 1.00 s | 0.58 s | 64x |
| `anterior_of(a)` | 0.062 s | 0.57 s | 0.16 s | 2.6x |
| uncinate-like bundle | 0.122 s | 1.54 s | 0.89 s | 7x |

What was changed after a first measurement (1M, all four components:
`only(a or b)` 8.12 s, bundle 11.54 s, `anterior_of(a)` 3.62 s):

* **No copy of a definition** (11.4): 8 rules fewer per program (32 to 24 for
  `only`, 39 to 31 for the bundle) and as many passes over relations that
  can be large.
* **`only` and `~` scan the streamlines of their argument**, not all of
  `traverses` (11.4): `only(a or b)` from 8.12 s to 5.93 s.
* **Slicing by component** (`components=("tracts",)`): the labels of a
  directional result are a join of its streamlines with `traverses`, which
  `anterior_of(a)` does not need to give its streamlines: 3.8 s to 0.5 s.
* **Two engine fixes found on the way** (`nl_core`): a predicate with rules at
  different depths could be read before it was complete, which made an
  aggregation over a union of a union fail (`NoValidChaseClassForStratum`;
  `stratify`, with a test); and the polars backend failed to union an `Int32`
  with an `Int64` column, which is what a literal and a column are (a
  numeric-type relaxation in the union, with tests).

Reading it:

* **For one query on one subject the Datalog route is slower**, from about
  the same (`anterior_of(a)`, 1x) to a few tens of times at a million
  streamlines, and by a hundred or more for the set algebra on a small
  tractogram, where the reference takes a millisecond. In absolute terms:
  under half a second to about six seconds at a million streamlines.
* **There is a fixed cost of about a second per program** with all four
  components (compiling and setting up the chase), which is why the small
  cases have the worst ratios and why the ratio falls with the size.
* **What costs at scale is each pass over `traverses`**: `only(a or b)`
  (4.9 s), `not a` (2.8 s). Each reference to the relation selects the atlas
  constant, drops that column and removes duplicates, which is of the order
  of a second on 7.7M rows; a profile of the earlier version at 300k put the anti-join at
  a third of the time, de-duplication at a fifth and the merges at a quarter.
  The set algebra on the four components is cheap.
* **What the table leaves out, in both directions.** tract_querier indexes
  at every invocation (a Python loop over the streamlines; 10 s for the 1M
  dictionaries of this benchmark, and more in the real one), has nothing to
  cache between runs and cannot run many subjects in one program; here the
  relations are built once by numpy and cached as parquet (7.3). A *session*
  of many queries on one tractogram pays the indexing once.
* **Tried, and not a lever**: the polars backend (`--backend polars`), which
  runs these programs since the union fix, is *slower*: 100k streamlines,
  `only(a or b)` 8.3 s against 1.0 s on pandas, and the next program aborts
  on polars' out-of-core budget (`POLARS_OOC_DISK_BUDGET_MB`). Pandas stays
  the backend.
* **Considered, not done**: folding the label constants and their unions into
  one rule each (the compile share is 0.1-0.5 s in these programs; it matters
  for the vocabulary file, 11.3, which is a minute of compilation); a
  per-streamline label count `streamline_label_count(t, s, n)` so that `only`
  is a join on a count (the scans are the cost, not the anti-join alone, after
  the restriction).
* **Not measured**: real tractograms and atlases (the data here is random);
  the memory (about 840 MB for a 3M-row join, 7.3); several runs, so no
  variance.

### 11.7 The Datalog file: `nl-tracts dl` and `nl-tracts run`

`nl-tracts dl PROGRAM.qry --atlas NAME KIND IMAGE TABLE --tractogram ID PATH
[--tractogram ...] [--ans NAME] [-o FILE.dl]` writes, from a WMQL program, a
file for the frontend that is meant to be **read and edited**
(`nl_tracts.dl.datalog_file`, text in and text out). For the program of the
worked example (11.4) it is:

```
% Datalog for the NeuroLang frontend, from the WMQL program uf.qry.
% Run it: nl-tracts run <this file>   (the answer is ans(t, s): the streamlines s of tractogram t, for uf.left).
% Atlas 'fs' (freesurfer), space 'subject:100307'; 1 tractogram(s): 100307.
% Other queries of the program: amy.left, amy.right, tmp.left, tmp.right, uf.right.

% The relations (traverses, endpoint, label, streamline_extent, ...) are built from the files by these commands.
.load_tract_atlas("fs", "aparc+aseg.nii.gz", "FreeSurferColorLUT.txt", kind="freesurfer", space="subject:100307", term_mapping="freesurfer_aseg_aparc")
.load_tractogram("100307", "tracts.trk", space="subject:100307")
.index_tracts("100307", "fs", crossing_threshold=2.0, length_threshold=0.0, emit_background=1, cache="~/.cache/nl_tracts")

% The query: the rules of each statement, cut to what the answer needs.
% amy.left = 18
qi_2(t, 18) :- tractogram(t)
% tmp.left = 5
qi_11(t, s) :- endpoint("fs", t, s, 0, 5)
qi_12(t, s) :- endpoint("fs", t, s, 1, 5)
% tmp.right = 6
qi_13(t, s) :- traverses("fs", t, s, 6)
qi_14(t, 6) :- tractogram(t)
% uf.left = endpoints_in(tmp.left and anterior_of(amy.left)) not in only(tmp.right)
qi_17(t, max(hi)) :- qi_2(t, l), label_extent("fs", l, 1, lo, hi)
...
% the names of the queries
q_uf_left(t, s) :- qi_31(t, s)

% The answer: the streamlines of uf.left.
ans(t, s) :- q_uf_left(t, s)
```

(`...` stands for the 9 rules left out here; the file has 16 rules for the
streamlines of this answer, `ans` excluded.)

**Four parts.** (1) A header: the source, how to run it, the atlas, the other
queries of the program. (2) The **commands** of 8.2, one atlas, one
`.load_tractogram` and one `.index_tracts` for each tractogram, with the
options the file was made with (`--crossing-threshold`, `--length-threshold`,
`--cache`, `--space`, `--term-mapping`, `--background`). (3) The **rules** of
the answer's streamlines only (`components=("tracts",)`, so no label set is
computed that the streamlines do not need), grouped under the WMQL statement
they come from (`.side` and `for` expanded), as `%` comments: `%` comments are
what the grammar of the frontend has (a `#` is a syntax error). (4) The
**answer**, `ans(t, s) :- q_<name>(t, s)`.

**One answer.** The frontend accepts one `ans(...)` rule per program (more is
`UnsupportedProgramError`), and a constant in its head fails with an
`AttributeError` (a frontend bug, 15.27), so the file has one query: `--ans
NAME`, by default the last query the program saves (`=`, or a name alone). To
get another, make another file, or change the name in the last line (every
query of the program is a predicate `q_<name>` of the whole program, and the
sliced rules are those of the answer only: for the others, run `dl` again).

**`emit_background`** is on in the file (`--no-background` turns it off):
WMQL's `not` has the background in its universe and `only` reads it (6.3,
11.2), so the file agrees with tract_querier only if `traverses` has the
background rows. The cost is about 1.1 to 1.3 times the rows of `traverses`.

`nl-tracts run FILE.dl [--format table|csv] [--limit N]` runs a file with
`tracts_engine()` and prints the streamlines (`t`, `s`) sorted, and the number
of rows on the standard error; errors are `nl-tracts: ...`, not tracebacks
(exit 1 for the frontend, 2 for the WMQL or the arguments). The same, by hand:

```python
from nl_tracts.frontend import tracts_engine
result = tracts_engine().execute_datalog_program(open("uf.dl").read())
result.as_pandas_dataframe()
```

**The engine fix this needed.** `execute_datalog_program` rewrites a program
with *Magic Sets* before it runs it, for the query `ans`. For a rule that has
two literals read one after the other, both reading an aggregate (the box of
`anterior_of`, in `both_endpoints_in(anterior_of(a)) ...`), the rewrite made
the aggregate depend on a literal that reads it: a cycle through an
aggregation, which no chase can solve (`NoValidChaseClassForStratum`, on
`NeurolangDL` and `NeurolangPDL`). `magic_rewrite` now checks the rewritten
program (`check_stratified`: no aggregation or negation in a cycle of its
dependencies) and raises an `InvalidMagicSetError` if it would not be
stratified, which the frontend answers by solving the query as written. Tests
in `nl_core` (the reduced program) and in the frontend (both engines).

**Tested end to end** (`tests/test_end_to_end.py`): an atlas image and a LUT,
two tractograms written by nibabel, a WMQL program of six queries
(complement, `only`, a directional `not in`, a nested bundle with
`both_endpoints_in(anterior_of(...))`, the two vertical terms); `datalog_file`
makes the file, a `NeurolangPDL` with the commands runs it, and
tract_querier's own `TractographySpatialIndexing` and `EvaluateQueries`,
loaded by path, give the same streamlines for each tractogram. One thing is
fed to the reference from our side: its label boxes, those of `label_extent`
(voxel edges) and not its own two corners, which differ by half a voxel
(6.1.1) and would make the directional queries differ for that reason alone.
`tests/test_cli.py` runs the command as a process, `dl` and then `run`.

## 12. Directional terms

**Implemented** as tract_querier has them (11.5): a streamline is selected by
comparing a face of its bounding box with a face of the union of the boxes of
the labels of the argument, and its ends by comparing their coordinates.
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
  (round half to even) and clipping with the warning; `traverses` at the
  threshold boundary (exactly 2.00 %, just below, just above) and
  denominators that include the background; `endpoint`; extents against a
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
Its `query_processor.py` is loaded **by path** with `code_util` stubbed
(`tests/reference_tq.py`): it needs only the standard library, whereas the
package's `__init__` imports vtk and the tractography readers. Found through
`TRACT_QUERIER_DIR` (the directory of the `tract_querier` package) or an
installed copy, else the tests skip. (Installed from
`github.com/demianw/tract_querier` into a throwaway environment; the
data files are those the vocabulary test reads.)

**The WMQL transpiler (done).** `EvaluateQueries` reads four dictionaries of
a spatial indexing, so no tractogram or voxels are needed: the same random
`crossing` and `endings` become the reference's dictionaries on one side and
the relations `traverses` and `endpoint` (DataFrames to `NeurolangDL`) on
the other. 74 tests: 41 programs of one expression each (every
operator, `not`/`-`/`~`/`only`, the endpoint functions, nesting, labels that
are absent from the data), 22 for the **directional terms** (each term;
`medial_of` and `lateral_of` with both sides; the box of a union of labels;
inside `endpoints_in`, `only`, `not in`, `and`; `.side` with `.opposite`;
two bundles shaped like the uncinate), the case of no label boxes, the
vertical terms, and 9 for the sequential rules, overwriting, wildcards over
the names so far, `.side`, `for`, redefinition, a statically empty query;
each on 3 seeds x 2 tractograms, with random label boxes, streamline boxes
and endpoint positions (multiples of 0.25, exact in float32); **every
defined query, not only the saved ones, is compared on its four components,
for each tractogram**. Data conventions: the
background is among the traversed labels, never the label of an endpoint
(6.3, 11.2), and every label is traversed in every tractogram, so that the
labels the reference takes from its one tractography (`not`, the labels of a
directional result) are comparable. The oracle found nothing wrong with the
transpiler's design; writing it against the reference source is what
corrected section 11 (`~A` has no endpoints). The tests were checked by
mutation (the operator of `anterior_of`, the face of `medial_of`: red). The tests are `slow` (about 5 minutes: each starts an engine).

**The vocabulary (done)**: 11.3.

**The geometry of a tractogram (done)**: `Tractogram`, the three
atlas-independent relations and the store, against loops over the raw
points (`tests/test_tractogram.py`, `test_relations.py`); and a directional
query run end to end, from a `Tractogram` and an `Atlas` to NeurolangDL,
against numpy over the points and the voxels of the label
(`test_frontend.py`).

**The indexing (done)**: `tests/test_indexing.py`. The same volume, affine and
streamlines go through `TractographySpatialIndexing` (loaded by path, with its
relative `.aabb` import registered beside it) and through nl_tracts with
`emit_background=True`, and the sets are compared exactly: `traverses` against
`crossing_tracts_labels`, `endpoint` against `ending_tracts_labels` without its
label-0 entries, on the four affines (aligned, flipped, permuted, oblique), at
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

**From files to the answer (done)**: `tests/test_end_to_end.py`, `test_cli.py`
and `test_commands.py`, 11.7 and 8.2.

### 13.3 Scale and engine tests

* A synthetic tractogram of 10^6 streamlines (marked `slow`, not in CI by
  default) checks the time and memory of 6.2 and 7.2.
* Join, negation and `AGGREGATE` (count of labels per streamline) rules on a
  10^6-row `traverses` through `NeurolangDL`, as in the probes of 8.1.
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
hemisphere precedence: 12 and 4 tests fail). Oracle cases (13.2) and boundary cases
(half-integer coordinates, the exact threshold, the background) are tests
before they are code. The CI wiring of 9.3 is done in the PRs of phase 1
(`plan_tests.py` and `tests.yml` with the atlas layer; the repository has
**no** tests of `plan_tests.py`, so the change was checked by calling
`plan()` for the changed-package cases, not by a test: adding a test for it
is an open item, 15.8). Each phase is one PR, merged on its
own; the scaffold's smoke test stays.

1. **Core**: `spaces`, `terms`, `atlas` (loaders, hemispheres, `label`,
   `label_term`, `label_extent`), `wmql`, the WMQL transpiler with the
   directional terms, and `tractogram` with the geometry relations
   (`streamline_extent`, `endpoint_coord`, `streamline_length`), the store and
   `register`, the voxel indexing (`traverses`, `endpoint`), the cache, the
   commands, the Datalog file and the `nl-tracts` command -- **done**. Still to
   do: re-measure the sizes of 7.2 on a real tractogram.
2. **Relations, cache, frontend**: **done** but for the mapping files for AAL
   and Schaefer (FreeSurfer aseg+aparc and Destrieux are shipped).
3. **Examples and docs**: a tutorial of tract_querier's standard bundles
   (uncinate, arcuate, corticospinal...) as Datalog rules over `label_term`;
   docs page; CI wiring (9.3).
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
2. ~~Should `endpoint` carry label 0?~~ **Decided: never.** `emit_background`
   governs `traverses` only; "no region at this end" is the negation of
   6.3.
3. **String `a` and `t` columns at 10^8 rows.** About 2 GB for the two
   (10 bytes a row each, pandas 3's `str` dtype, 7.2). **Keep strings until
   measured**; the fallback is surrogate integer keys with a lookup
   relation.
4. **The extra copy in `RelationalAlgebraFrozenSet.__init__`** for
   DataFrames. A change in `nl_core` (adopt without copying); **separate PR
   if the footprint matters**, not part of this package.
5. **Incremental registration**: `add_tuple_set` replaces. Phase 1 builds the
   `RelationStore` first and registers once. **Keep**, and revisit if
   interactive loading of tractograms one by one is needed.
6. **Is `nl_tracts` above `neurolang` for anything yet?** (9.2.) Neither the
   frontend (duck-typed) nor the transpiler imports it, and the directional
   terms did not need it. **Keep the layer and the dependency** as asked, and
   review when the probabilistic variants (12) or the text-to-query
   integration are started.
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
   dependency; its evaluator is loaded by path from a copy found through
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
16. **`subcortical`** departs from tract_querier's on purpose (3.2) and the
    transpiler states how (11.3). **Say so in the tutorial**; the bundles
    that exclude `subcortical` (`not in subcortical.side`) now also exclude
    the caudate, hippocampus and accumbens, a change of what they select on
    a volume that has them. **Check the standard bundles before the
    tutorial**, with the transpiler, on tractography.
17. **`X.left` of a group with midline members** (11.3): the file's
    `subcortical.left` includes the brain stem. A rule transcribed from it
    must select `hemi` left or none; the transpiler's output, which works on
    ids, is exact, but a hand-written rule over `label_term` and
    `label.hemi` is not. **Keep; consider a helper relation**
    `side(a, l, h)` that gives each midline label to both sides, once
    transcribed bundles are tried.
18. **`Left-Thalamus` is two things** (11.3, 4): the legacy label 9 in
    FreeSurfer 6 and the thalamus proper (10) in 7. The mapping is keyed by
    raw name and maps it to `thalamus_proper`. **Keep**, and give the mapping
    a LUT-version key if a FreeSurfer 7 LUT is ever found to need otherwise.
19. **Speed** (11.6). One query is a fixed second plus a second or so per
    pass over `traverses`: 0.5-6 s at a million streamlines for the
    streamlines, 1x to 45x the reference's wall-clock, the gap narrowing as the
    data grows. Evaluating the 945 label rules of the vocabulary file takes
    about a minute (a rule costs about 55 ms to compile), which is why the
    vocabulary and oracle tests are `slow`. **Keep for now**; the levers left
    are in 11.6 (folding constants; a label count per streamline). The polars
    backend is slower on these programs and was not adopted.
20. **`superior_of` and `inferior_of` are inverted in the reference** (11.5):
    they select the streamlines whose top is *below* the top of the box, and
    whose bottom is *above* its bottom, where the other four terms select
    those that extend beyond a face. Reproduced, because the reference is the
    oracle; no shipped query and no test of tract_querier uses them. **Decide**:
    keep (`anatomical_vertical=False`, the default), or make the anatomical
    reading the default (`True`), which is a one-line flip of `_DIRECTIONS`
    and a decision about the oracle. It is the user's own code, and the fix
    belongs in tract_querier too.
21. **`medial_of(x)` and `lateral_of(x)` need `.left` or `.right`** (11.5): the
    reference fails with an unbound variable on a bare name; here a
    `WMQLError` that says so. **Keep.**
22. **A label without a box** is ignored by the directional terms, where the
    reference raises `TractQuerierLabelNotFound` (11.5). A silent drop is
    unlike an error. **Keep, and warn at run time** in the frontend if a label
    of an argument has no `label_extent` row (not possible in Datalog itself).
23. **Label sets are per tractogram** (11.1): `q_x__label(t, l)`, so that a
    program for many subjects is the program of each one, as the reference is.
    It costs the `tractogram(t)` relation and one column. The alternative,
    atlas-level labels, is the same for everything but a directional term
    applied to a name whose label set depends on the data (`anterior_of(x)`
    where `x = anterior_of(...)`), where it would mix the subjects.
24. **The geometry relations are float32** (3.1), the reference's float64: a
    tie in the reference can be a different answer here. Real coordinates in
    mm are not on a lattice, so ties are measure-zero. **Keep.**
25. ~~The bundles of the vocabulary file cannot yet be run on a tractogram~~:
    **closed**. The voxel indexing is done (6.2) and a WMQL program is a
    Datalog file that runs (11.7). Not tried on the 1500 definitions of the
    file at once (a minute of compilation, 11.6); a bundle of it is a program
    of its own.
26. **`neurolang-query --engine tracts`.** The commands are in a solver for
    `NeurolangPDL(solver_class=...)` (8.2) and `nl-tracts run` uses it. The
    `neurolang-query` registry is YAML-driven (downloads, a data directory,
    `python_init`); an engine for it would be `apps > nl_tracts` code. **A
    follow-up**, if a file is to be run from the apps' command.
27. **Frontend bug found, not fixed**: a constant in the head of `ans`
    (`ans("one", x) :- r(x, y)`) fails with `AttributeError: 'str' object has
    no attribute 'expression'`, and a second `ans` rule is refused. The Datalog
    file therefore has one answer, `ans(t, s)`. **Fix in `neurolang`** so that
    one file can name several queries (`ans("uf.left", t, s)`).
28. **Paths in the commands are read as written**, relative to where the
    program runs. A file moved from the directory it was made in loses its
    files. **Keep; write absolute paths** (`--atlas ... /abs/path`) for a
    file that moves, or run it from its directory.
29. **`.index_tracts` is the command that publishes the relations**, and a
    tractogram's relations have the streamlines that its first index kept
    (`length_threshold`): a second index of it with another threshold is an
    error. **Keep**; the alternative is to put the threshold on
    `.load_tractogram`, which the relations of the atlas do not need.
30. **Magic Sets and aggregation** (11.7). The check refuses a cycle through
    an aggregation or a negation in the rewritten program, and the query is
    solved as written, so for such a program the rewrite is not applied and
    its goal-directed pruning is lost. For these programs, which are already
    cut to what the answer needs (11.4), that costs nothing measurable. A
    rewrite that keeps the program stratified (the aggregate evaluated before
    the magic predicates that depend on a literal that reads it) would keep
    the pruning. **Keep.**
