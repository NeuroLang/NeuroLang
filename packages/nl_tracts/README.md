# nl_tracts

Tractography for NeuroLang: tractograms and atlases enter an engine as
extensional relations (`streamline_label`, `endpoint_label`, `label`,
`label_term`, `streamline_extent`, ...) computed in bulk with numpy, a
streamline being one integer id `x` that `streamline(x, t, s)` decodes to its
tractogram and its position in the file, and queries are written
in Datalog, with the vocabulary of tractography.

The package reads atlases: a label volume, its affine and a label table
(`nl_tracts.atlas`), tagged with the space they are in (`nl_tracts.spaces`),
and the term mappings that name their labels in a shared vocabulary, with a
hemisphere (`nl_tracts.terms`, the tables of `nl_tracts/mappings`, built by
`tools/make_freesurfer_mappings.py`). It reads tractograms
(`nl_tracts.tractogram`), indexes them by atlases into the relations
(`nl_tracts.indexing`, the ids `x` of `nl_tracts.ids`) and caches the
relations that depend on an atlas on disk (`nl_tracts.cache`). It needs
numpy, scipy, nibabel, pandas and pyarrow, not NeuroLang.
