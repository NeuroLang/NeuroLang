# nl_tracts: tractography for NeuroLang

Status: **design only**. The package is a scaffold: modules hold signatures
and docstrings, every body raises `NotImplementedError`. Numbers below marked
*measured* come from probes run on this repository (NeuroLang `master`
at the time of the split, pandas backend, an Intel Mac); the rest are
estimates to be checked in phase 1.

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
* **`label_extent` and `streamline_extent` are in world mm**, computed from
  the data (voxel centres, and points), see 6.1 for why not the corner
  boxes of the reference.
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
(`("precentral", "frontal_lobe", "motor_cortex")`) and a term covers labels
of several atlases. The hemisphere is *not* in the term: it lives in
`label(a, l, name, hemi)`, so that `"precentral"` is the same term for the
left and the right label, and rules choose hemispheres explicitly:

```
% uncinate fasciculus, any atlas, either hemisphere
uf(a, t, h, s) :-
    traverses(a, t, s, l1), label_term(a, l1, "temporal_pole"),
        label(a, l1, _n1, h),
    traverses(a, t, s, l2), label_term(a, l2, "orbitofrontal"),
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

Mapping files ship in `nl_tracts/mappings/<atlas-key>.tsv`, columns
`base_name`, `term`; `base_name` is the raw label name **after** the
hemisphere token has been stripped (4.3), so `ctx-lh-precentral` and
`ctx-rh-precentral` both read the row `precentral<TAB>precentral`. Rows
are many-to-many. An atlas without a mapping file gets one term per label,
its base name lower-cased with non-alphanumerics replaced by `_`
(`default_term`), which keeps single-atlas use working. The vocabulary is
curated, small, and versioned with the package; the files are plain TSV so
users can pass their own (`Atlas(..., term_mapping=...)`).

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
not in the table, are reported: the former are dropped from `label`, the
latter are kept with `name = "label_<id>"` and `hemi = "none"` and a
warning, since `traverses` would otherwise refer to an unknown label.

### 4.2 Loaders

| Loader | Input | Notes |
|---|---|---|
| `Atlas.from_freesurfer` | label image + FreeSurfer LUT (`FreeSurferColorLUT.txt`) | `load_label_table_freesurfer_lut` skips comments; ids and names only, colours are ignored |
| `Atlas.from_bids_dseg` | label image + `*_dseg.tsv` | columns `index`, `name` |
| `Atlas.from_nilearn` | the object a nilearn fetcher returns | **duck-typed**: `.maps` (path or image), `.labels`, optional `.lut`; nilearn is not imported (decision 7 forbids it in the core). Only the tests build such objects with nilearn |
| `Atlas.from_tsv` | label image + plain TSV | columns `id`, `name`, optional `hemi` |

All accept a path or an already loaded nibabel image. The nilearn
convention that `labels[i]` is the name of id `i` (with a leading
`Background`) is the loader's, not the core's.

### 4.3 Hemisphere inference

A `HemisphereRule(pattern, hemi)` has a regular expression that must
**fully** match the raw name and capture a named group `base`; the rules
are ordered and the first match wins. Defaults, `HEMISPHERE_RULES`:

| Convention | Left | Right |
|---|---|---|
| FreeSurfer subcortical | `Left-(?P<base>.+)` | `Right-(?P<base>.+)` |
| FreeSurfer cortical | `ctx-lh-(?P<base>.+)` | `ctx-rh-(?P<base>.+)` |
| Suffix | `(?P<base>.+)_L` | `(?P<base>.+)_R` |
| Prefix `lh.`/`rh.` | `lh\.(?P<base>.+)` | `rh\.(?P<base>.+)` |
| Suffix `.left`/`.right` | `(?P<base>.+)\.left` | `(?P<base>.+)\.right` |

No match: `hemi = "none"`, `base_name = name`. The rules are a parameter of
the loaders; a table with an explicit `hemi` column bypasses them, but still
gets `base_name` from them (so mapping files work). Known hazards, each covered by a test: the `_L`/`_R` suffix rules match any
name that ends that way, so a name such as `Pallidum_R` is right but
`Foo_L_R` is read as right with base `Foo_L`; matching is case-sensitive;
and a name that is half of a pair (`Left-X` without `Right-X`) is
reported with a warning, not fixed.

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
| 6 | Endpoint labels are the labels of the first and of the last point, as `ending_tracts_labels[0|1]` | `endpoint(a, t, s, e, l)`, `e` = 0 first, 1 last; label 0 as in 6.3 |
| 7 | With `length_threshold > 0` the tract list is **filtered before enumeration**, so the ids of the survivors are *renumbered* | **ids are never renumbered**: `s` is the position in the file; filtered streamlines are absent from every relation. The equivalence test maps the reference's ids back through the kept-index list |
| 8 | Length = sum of segment norms of the polyline, `>= length_threshold` keeps | same, in float64, then stored as float32 in `streamline_length` |
| 9 | Streamlines with fewer than 2 points make the bounding-box step raise | `Tractogram` rejects them at construction, with the id of the first offender |
| 10 | `compute_label_bounding_boxes` uses `ndimage.find_objects` on the *ijk corners* `(start, stop)` mapped by the linear part: wrong for oblique affines, `stop` is exclusive, labels start at 1 | `label_extent` is from all the voxel centres of the label, mapped by the full affine, so it is right for any affine; **not** bit-identical to the reference, and the tests compare it to a brute force instead |
| 11 | `tract_endpoints_pos` = first and last point | `endpoint_coord`, float32 (the reference holds float64; tests compare with `atol` 1e-4 mm) |

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
6. `label_extent`: one pass over the nonzero voxels, `np.minimum.reduceat`
   on the voxels sorted by label.

No Python loop over streamlines or over labels; the only loops are over
chunks of points and over the three axes.

### 6.3 Label 0 and the background

The reference treats 0 as a label like any other: it counts in the
denominators, can be traversed, and can be an endpoint label. nl_tracts:

* Points in the background **always count in the denominator** (parity 4).
* `traverses` and `endpoint` have rows for the background label **only if
  `IndexingParams.emit_background=True`**. Default is **False**: with 0
  emitted, almost every streamline has a `(a, t, s, 0)` row and `traverses`
  grows by a factor of 1.1-1.3 for facts no rule should use.
  `emit_background=True` reproduces the reference exactly and is what the
  equivalence tests use.
* For `endpoint` the default hides a real fact, "this end is in no region"
  (rows with `l = background` are left out too). The factor is 1 (two rows
  per streamline either way), so open question 15.2 recommends that
  `endpoint` always carries the background label; as written, the flag
  governs both relations, as the brief's "document label 0 handling" asks
  for one rule.
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
| `traverses` | 10^7 | 8 (a) + 8 (t) + 4 + 4 = 24 | 240 MB |
| `endpoint` | 2x10^6 | 8 + 8 + 4 + 1 + 4 = 25 | 50 MB |
| `streamline_extent` | 3x10^6 | 8 + 4 + 1 + 4 + 4 = 21 | 63 MB |
| `endpoint_coord` | 6x10^6 | 8 + 4 + 1 + 1 + 4 = 18 | 108 MB |

The `a` and `t` columns are Python-object string columns whose entries are
all the *same* string object, so the cost is the 8-byte pointer, not the
string. *Measured* at 3x10^6 rows: an `int` atlas column 719 MB peak RSS /
0.95 s, an `object` string column 841 MB / 1.12 s, a `string[pyarrow]`
column 751 MB / 1.43 s for the join with `label_term` (the numbers include
the DataFrame and the engine's copy). **Decision: keep strings** for `a`
and `t`: queries switch atlas by editing a literal (decision 2), the
overhead is 17 % of a ~720 MB baseline, and surrogate integer keys would
need a lookup relation in every rule. Revisit if 10^8 rows (then the
pointer columns are 1.6 GB) is a real use; see 15.3. `t` and `a` are
low-cardinality: `Categorical` is not used because the engine's behaviour
on categorical columns is untested.

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
rebuilt as object columns pointing to **one** string object each, so the
memory of 7.2 holds after a cache hit.

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
  nl_tracts/
    __init__.py         re-exports the public classes (core only)
    spaces.py           Space, SpaceMismatchError, check_same_space
    atlas.py            Atlas, ProbabilisticAtlas, HemisphereRule, loaders
    tractogram.py       Tractogram
    indexing.py         IndexingParams, *_relation, RelationStore, build_relations
    cache.py            parquet cache: cache_key, read_cached, write_cached
    frontend.py         register(), RELATION_NAMES -- the only NeuroLang use
    mappings/           per-atlas label_term TSVs (shipped data)
    tests/
```

### 9.2 `.importlinter`

* `root_packages` gains `nl_tracts`.
* The layers contract becomes `neurolang_apps > nl_tracts > neurolang >
  nl_core > nl_expr`.
* A new `forbidden` contract: `nl_tracts.{spaces, atlas, tractogram,
  indexing, cache}` may import none of `neurolang`, `nl_core`, `nl_expr`,
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

### 9.3 CI and release touch points (not changed by the scaffold)

Changed because the `.importlinter` change would otherwise break CI:
`.github/workflows/layers.yml` (`PYTHONPATH` gains `packages/nl_tracts`).
**Not changed, to decide in review** (15.8): `.github/plan_tests.py`
(`PACKAGES`, order is semantic: insert `nl_tracts` between `neurolang` and
`neurolang_apps`, extras `--extra dev`), `.github/workflows/tests.yml`
(paths-filter `nl_tracts`), `.github/workflows/publish.yml` (build loop:
do not publish an empty scaffold), `doc/conf.py` (sys.path), `Makefile`
(`--cov=nl_tracts`), and the documentation page.

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
def infer_hemispheres(names, rules=HEMISPHERE_RULES) -> DataFrame     # name, base_name, hemi
def load_label_table_freesurfer_lut(path) -> DataFrame
def load_label_table_bids_dseg(path) -> DataFrame
def load_label_table_tsv(path) -> DataFrame
class Atlas:
    def __init__(self, name, labels, affine, table, space, background=0,
                 hemisphere_rules=HEMISPHERE_RULES, term_mapping=None)
    from_freesurfer(name, image, lut, space, **kw) / from_bids_dseg(...) /
    from_tsv(...) / from_nilearn(name, bunch, space, **kw)
    label_relation() / label_term_relation() / label_extent_relation()
    content_hash() -> str
class ProbabilisticAtlas:      # extension point, constructor raises

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
`emit_background=True` to agree with the reference; (iv) `not` ignores the
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
  vs `crossing_tracts_labels`, `endpoint` vs `ending_tracts_labels`, for
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

1. **Core**: `spaces`, `atlas` (loaders, hemispheres, `label` and
   `label_term`), `tractogram`, `indexing` (`traverses`, `endpoint`), and
   the oracle tests. Re-measure the sizes of 7.2.
2. **Relations, cache, frontend**: the atlas-independent relations,
   `label_extent`, `RelationStore`, parquet cache, `register`, the mapping
   files for FreeSurfer (aseg, DKT), AAL and Schaefer, multi-atlas tests.
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
2. **Should `endpoint` always carry label 0?** (6.3.) Two rows per
   streamline either way; omitting them loses "ends in no region".
   **Always emit the background label in `endpoint`**, and keep
   `emit_background` for `traverses` only. This changes the default of
   decision 5's "document label 0 handling" and so needs your agreement.
3. **String `a` and `t` columns at 10^8 rows.** 1.6 GB of pointers.
   **Keep strings until measured**; the fallback is `string[pyarrow]`
   (+27 % time in the probe), not surrogate ints.
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
8. **CI and release wiring** (9.3): `plan_tests.py`, `tests.yml`,
   `publish.yml`, docs, Makefile. **Do in the PR that implements phase 1**,
   not on the scaffold; `layers.yml` is already done.
9. **Installing tract_querier for the oracle tests.** Not checked here
   (the reference was read, never run). **Vendor nothing**; test against an
   installed copy, skipped if absent, and if it does not install next to
   the repository's numpy, run the oracle in a dedicated CI job or compare
   against golden outputs generated once and committed (small fixtures).
10. **Term vocabulary governance.** Who curates the shared terms and how a
    new atlas is added (a TSV and a test that every label has a term).
    **A single curated vocabulary file listing the allowed terms, and a
    test that each mapping file uses only listed terms.**
11. **Axis and variant encoding by int.** Rule authors must remember
    `axis = 1` is y. **Document it; if it hurts, add a tiny `axis(n, name)`
    relation rather than a string column.**
12. **LPS/RAS+ files.** `from_file` converts to RAS+ with nibabel and the
    oracle compares in RAS+. **State it and test only RAS+ against the
    reference**; the conversion itself is nibabel's.
