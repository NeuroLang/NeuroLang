# nl_tracts: tractography for NeuroLang

Status: **the atlas layer is implemented** (`spaces`, `terms`, `atlas`:
sections 3.2, 4, 5, and `label_extent` of 6.1); the rest (tractogram,
indexing, cache, frontend) is still a scaffold whose bodies raise
`NotImplementedError`. Numbers below marked *measured* come from probes run
on this repository (NeuroLang `master` at the time of the split, pandas
3.0.6, pandas backend, an Intel Mac); the rest are estimates to be checked
in phase 1.

Contents: 1 Goals · 2 Decisions taken · 3 Relations · 4 Atlases ·
5 Spaces · 6 Indexing semantics · 7 Data formats and cache ·
8 Handing the relations to the engine · 9 Packaging and layering ·
10 Public API · 11 From tract_querier to Datalog · 12 Future work:
directional relations · 13 Test plan · 14 Phases · 15 Open questions

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
algorithms; visualisation; directional relations in the first phases (12);
probabilistic atlases (4.4); a port of the WMQL language itself.

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
6. **Directional relations are future work** (12).
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
`cerebellum`, `ventricles`, `corpus_callosum`. Reusing it makes the
standard bundle definitions of the phase-3 tutorial a transcription of
tract_querier's queries (`UF.side` ...). The groups are copied as the file
has them, odd parts included (`subcortical` has no caudate, hippocampus or
accumbens), and the file's `.side` becomes `label.hemi`:

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
boundaries (lateral fissure, central sulcus, parieto-occipital sulcus), the
medial wall and `Unknown` have no term, on purpose, and are what
`unmapped_labels` reports. Against a real FreeSurfer LUT (skipped if there
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
| 7 | With `length_threshold > 0` the tract list is **filtered before enumeration**, so the ids of the survivors are *renumbered* | **ids are never renumbered**: `s` is the position in the file; filtered streamlines are absent from every relation. The equivalence test maps the reference's ids back through the kept-index list |
| 8 | Length = sum of segment norms of the polyline, `>= length_threshold` keeps | same, in float64, then stored as float32 in `streamline_length` |
| 9 | Streamlines with fewer than 2 points make the bounding-box step raise | `Tractogram` rejects them at construction, with the id of the first offender |
| 10 | `compute_label_bounding_boxes`: `ndimage.find_objects` (labels start at 1, `stop` exclusive), then **two** corners `[start, stop]` mapped by the linear part and the translation, `BoundingBox` takes their min and max. That is the centre of the first voxel to the centre of the voxel *after* the last (half a voxel off at both ends), and a box only if each world axis depends on one voxel axis (wrong for oblique affines) | `label_extent` (implemented, 6.1.1): the same `find_objects` and the same "box of a box" idea, by **voxel edges** `[start - .5, stop - .5]`, **8 corners** through the full affine, min and max per world axis. Tests transcribe the reference and check the relation exactly (below) |
| 11 | `tract_endpoints_pos` = first and last point | `endpoint_coord`, float32 (the reference holds float64; tests compare with `atol` 1e-4 mm) |

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
    indexing.py         IndexingParams, *_relation, RelationStore, build_relations
    cache.py            parquet cache: cache_key, read_cached, write_cached
    frontend.py         register(), RELATION_NAMES -- the only NeuroLang use
    mappings/           shipped term mappings (TSV): freesurfer_aseg_aparc,
                        freesurfer_destrieux
    tests/              conftest.py: synthetic volumes, `network` marker
```

### 9.2 `.importlinter`

* `root_packages` gains `nl_tracts`.
* The layers contract becomes `neurolang_apps > nl_tracts > neurolang >
  nl_core > nl_expr`.
* A new `forbidden` contract: `nl_tracts.{spaces, terms, atlas,
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
`frontend.py` needs no import of `neurolang` (it uses the engine duck-typed),
so the declared dependency on `neurolang` is for the directional relations
(12), which would read the region types and the thresholds in `config`; if
phase 4 turns out not to need them, the layer and the dependency should be
reconsidered (15.6).

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
```

## 11. From tract_querier to Datalog

In the reference every query value is a **triple** (`FiberQueryInfo`): the
streamlines `tracts`, a set of `labels`, and two sets of streamlines
`tracts_endpoints[0|1]` (those whose first / last point is in one of the
labels). Binary operators combine the three components *componentwise*,
with one twist: `and` / `*` **intersects** the streamlines and endpoint sets
but **unions** the label sets. So a WMQL expression `E` compiles to four
Datalog predicates, `E(s)`, `E_label(l)`, `E_end0(s)`, `E_end1(s)`, with
`a`, `t` fixed to constants; the equivalence tests (13.2) translate each
construct, run both, and compare.

| WMQL | Reference semantics (componentwise on the triple) | Datalog |
|---|---|---|
| `7` | tracts = those traversing 7; labels = {7}; end_e = those whose end e is in 7 | `E(s) :- traverses(a,t,s,7)`; `E_label(7)`; `E_end0(s) :- endpoint(a,t,s,0,7)`, same for 1 |
| `A or B`, `A + B` | union of tracts, labels, endpoints | two rules per predicate, same head |
| `A and B`, `A * B` | tracts ∩, **labels ∪**, endpoints ∩ | `E(s) :- A(s), B(s)`; `E_label(l) :- A_label(l)` and `E_label(l) :- B_label(l)`; `E_end0(s) :- A_end0(s), B_end0(s)` |
| `A - B`, `A not in B` | set difference of the tracts, of the labels **and** of the endpoint sets | `E(s) :- A(s), ~B(s)`; `E_label(l) :- A_label(l), ~B_label(l)`; same for ends |
| `not A`, `-A` (unary) | tracts traversing **some label outside A's label set** (labels of the data, `crossing_labels_tracts`, background included); labels = that complement; endpoints empty | `E_label(l) :- present(l), ~A_label(l)` with `present(l) :- traverses(a,t,s,l)`; `E(s) :- traverses(a,t,s,l), E_label(l)`. **Not** `~A(s)`: a streamline that traverses A *and* another label is in the result |
| `only(A)`, `~A` | tracts of A whose **every** traversed label is in A's label set; labels as A; endpoints of A whose label is in A's labels | `outside(s) :- traverses(a,t,s,l), ~A_label(l)`; `E(s) :- A(s), ~outside(s)`; `E_end0(s) :- A_end0(s), endpoint(a,t,s,0,l), A_label(l)` |
| `endpoints_in(A)` | tracts = end0 ∪ end1 of A; labels, endpoints as A | `E(s) :- A_end0(s)`; `E(s) :- A_end1(s)` |
| `both_endpoints_in(A)` | tracts = end0 ∩ end1 of A | `E(s) :- A_end0(s), A_end1(s)` |
| `"pattern*"` (a string) | `fnmatch` over the *names of queries already evaluated*, union | not a relation construct (query names are Datalog predicates); the rule author lists them. Out of scope |
| `X.side`, `X.opposite` | textual rewrite of the hemisphere of the label name | `label(a,l,n,h)` and `opposite_hemisphere(h,h2)` |
| `anterior_of(...)` etc. | cardinal relations to a reference region | future work (12) |
| `save(...)`, `for` loops, `import` | side effects and macros | out of scope |

The traps, because they are the point of the oracle: (i) `not` and
`only`/`~` act on **label sets**, with the labels present in the data as
universe, not on streamline sets; plain Datalog `~` is the streamline-level
difference; (ii) the label set of `A and B` is a **union**, so
`only(A and B)` keeps streamlines whose labels lie in `L_A ∪ L_B`; (iii) the
data's label universe includes the background (6.3), so both traps need
`emit_background=True` to agree with the reference, and queries whose label
set contains the background *and* that read the endpoint predicates are out
of the equivalence scope (the endpoint relation has no background rows); (iv) `not` ignores the
endpoint sets (they become empty) and `only` filters them: a translation
that treats them uniformly is wrong.

## 12. Future work: directional relations

**Not in phases 1-3**, and written down so nothing in the schema
forecloses it. The reference's `anterior_of`, `posterior_of`, `medial_of`,
`lateral_of`, `inferior_of`, `superior_of` are, in tract_querier, per-point
relations to a reference region's bounding box; NeuroLang's
`CD_relations.cardinal_relation` decides the cardinal-direction relation of
two `Region`s from interval algebra on their (tree) bounding boxes, which is
a per-pair Python evaluation and unusable for streamlines (decision 1).

Proposal:

1. **Vectorised cardinal-direction tile counts.** For a reference region
   (a label, a union of labels, or an explicit box from `label_extent`) with
   box `[lo, hi]` per axis, compute for every point the 3-vector of
   `sign` classes (before / within / after) on each axis: 27 tiles, in one
   pass over the `(P, 3)` array (`np.digitize` or two comparisons per
   axis). `np.add.reduceat` / `bincount` over the streamline ids gives a
   `(S, 27)` count matrix. Memory is `S x 27` int32, 108 MB per 10^6
   streamlines, and is computed **per reference region**, not per atlas.
2. **Column-wise directional maps.** Each directional relation is a linear
   map (a 0/1 matrix) from the 27 columns to a count: "anterior" is the sum
   of the tiles with the anterior class on the y axis, and so on, with the
   anatomical variants of `region_solver` (a direction holds and its inverse
   and the overlap do not) as masks over the same columns. All six
   directions are one matrix product on the count matrix.
3. **Four variants**, by how the counts become a relation:
   * *strict*: all the points of the streamline are in the direction;
   * *partial*: at least one point is;
   * *endpoint*: the first or the last point is (reading the same tiles at
     the offsets);
   * *fractional* (probabilistic): `count / n_points` as the probability of a
     probabilistic fact, for `NeurolangPDL` (and after 4.4's fix of
     probabilistic facts from DataFrames).
4. **Schema**: `anterior_of(a, t, s, l, variant)` (reference label `l`) and
   siblings, or one relation `direction(a, t, s, l, dir, variant)` with
   `dir` a small int; `variant` a small int; the fractional relation is the
   probabilistic one with the probability as first column. To decide when
   the phase starts.
5. These need `neurolang` (the region types and the thresholds in
   `config`), which is the reason `nl_tracts` is above `neurolang` and not a
   leaf next to `nl_core` (9.2, 15.6).

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

`tract_querier` is a **test-only** dependency, optional
(`pytest.importorskip`), not a runtime one and not in the `dev` extra
(whether it installs next to the repository's numpy and nibabel is to be
checked in phase 1, 15.9).

* The same synthetic atlas and tractogram go through
  `TractographySpatialIndexing` and through nl_tracts with
  `emit_background=True`, and the sets are compared exactly: `traverses`
  vs `crossing_tracts_labels`, `endpoint` vs `ending_tracts_labels` minus
  its label-0 entries (and, for a streamline whose end is in the
  background, the absence of the row), for
  several `crossing_threshold` values including 0, 2, 25, 50 and 100,
  oblique and flipped affines, points outside the volume and exactly on
  voxel boundaries.
* `length_threshold`: the reference's ids are renumbered, so the test maps
  them back through the kept-index list (6.1, parity 7).
* The WMQL constructs of section 11, each as a small query evaluated by
  `EvaluateQueries` and by the Datalog translation; the label-set `not` and
  `only` cases specifically, on data where a streamline traverses A and
  another label.
* Oracle-free brute force (Python loops, per streamline) on a tiny volume,
  to catch bugs shared by both implementations.
* Fixtures stay in RAS+ (6.1, 5).

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
   `label_term`, `label_extent`) -- **done**, the atlas layer --;
   `tractogram`, `indexing` (`traverses`, `endpoint`), and the oracle tests.
   Re-measure the sizes of 7.2.
2. **Relations, cache, frontend**: the atlas-independent relations,
   `RelationStore`, parquet cache, `register`, the mapping files for AAL and
   Schaefer (FreeSurfer aseg+aparc and Destrieux are shipped), multi-atlas
   tests.
3. **Examples and docs**: a tutorial of tract_querier's standard bundles
   (uncinate, arcuate, corticospinal...) as Datalog rules over `label_term`;
   docs page; CI wiring (9.3).
4. **Directional relations** (12).
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
6. **Is `nl_tracts` above `neurolang` only for the directional relations?**
   (9.2.) Today the frontend is duck-typed and needs no `neurolang` import.
   **Keep the layer and the dependency** as asked, and review at phase 4.
7. **Engine bug**: `add_probabilistic_facts_from_tuples` rejects
   DataFrames (4.4). **Fix in neurolang/nl_core** (infer the row type from
   the first row of a DataFrame, as `add_tuple_set` does), with a test,
   before phase 5; outside this task.
8. **CI and release wiring** (9.3): `layers.yml`, `plan_tests.py` and
   `tests.yml` are done; `publish.yml`, docs and the Makefile are not, and
   `plan_tests.py` has no test of its own. **Do the rest when the package
   does something worth publishing; add a test of `plan()`** (it is a pure
   function, five lines of table).
9. **Installing tract_querier for the oracle tests.** Not checked here
   (the reference was read, never run). **Vendor nothing**; test against an
   installed copy, skipped if absent, and if it does not install next to
   the repository's numpy, run the oracle in a dedicated CI job or compare
   against golden outputs generated once and committed (small fixtures).
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
13. **Destrieux mapping** (3.2) is a draft by a non-anatomist, with the
    boundary structures left unmapped on purpose. **Review it**, and decide
    whether the lateral fissure and the central sulcus should get terms
    (`central_sulcus` is not in tract_querier's vocabulary).
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
16. **The vocabulary's oddities** are copied from the `.qry` on purpose
    (3.2): `subcortical` without caudate, hippocampus and accumbens; the
    `.qry` also defines `middlefrontalgyrus` and `middle_frontal_gyrus`
    (likewise the inferior) with the same content, of which one is kept.
    **Keep, and say so in the tutorial**; a better `subcortical` is a
    vocabulary change, with the portability cost that implies.
