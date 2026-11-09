# nl_tracts

Tractography for NeuroLang: tractograms and atlases enter an engine as
extensional relations computed in bulk with numpy, and queries are written
in Datalog, with the vocabulary of tractography.

This first part of the package reads atlases: a label volume, its affine
and a label table (`nl_tracts.atlas`), tagged with the space they are in
(`nl_tracts.spaces`), and the term mappings that name their labels in a
shared vocabulary, with a hemisphere (`nl_tracts.terms`, the tables of
`nl_tracts/mappings`, built by `tools/make_freesurfer_mappings.py`). It
needs numpy, scipy, nibabel and pandas, not NeuroLang.
