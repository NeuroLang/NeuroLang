# nl_tracts

Tractography for NeuroLang: tractograms and atlases enter an engine as
extensional relations (`streamline_label`, `endpoint_label`, `label`,
`label_term`, `streamline_extent`, ...) computed in bulk with numpy, a
streamline being one integer id `x` that `streamline(x, t, s)` decodes to its
tractogram and its position in the file, and queries are written
in WMQL (tract_querier's language) or in Datalog. See
[DESIGN.md](DESIGN.md).

## From a WMQL query to an answer

```sh
# 1. a Datalog file from a WMQL program (the atlas, the tractograms)
nl-tracts dl uf.qry \
    --atlas fs freesurfer aparc+aseg.nii.gz FreeSurferColorLUT.txt \
    --tractogram 100307 tracts.trk --space subject:100307 \
    --term-mapping freesurfer_aseg_aparc --cache ~/.cache/nl_tracts \
    --ans uf.left -o uf.dl

# 2. read it, change it; then run it
nl-tracts run uf.dl                 # or --format csv, --limit 20
```

The file has the commands that build the relations from the files
(`.load_tract_atlas`, `.load_tractogram`, `.index_tracts`), the rules of the
query with a `%` comment on each WMQL statement they come from, and
`ans(t, s)`. To run it from Python:

```python
from nl_tracts.frontend import tracts_engine
tracts_engine().execute_datalog_program(open("uf.dl").read())
```

The labels in a WMQL program are the ids of the atlas, as in tract_querier's
`FreeSurfer.qry`.
