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
uf_frontal(x, h) :- traverses(x, "orbitofrontalgyrus", h)
uf_frontal(x, h) :- traverses(x, "middle_frontal_gyrus", h)
uf_frontal(x, h) :- traverses(x, "inferior_frontal_gyrus", h)
uncinate(x, h) :-
    uf_frontal(x, h),
    endpoint_in(x, e, "temporal", h), endpoint_anterior_of(x, e, "amygdala", h),
    traverses(x, "insula", h),
    ~traverses(x, "occipital", h), ~traverses(x, "parietal", h),
    ~traverses(x, "cingular", h), ~posterior_of(x, "putamen", h),
    opposite_hemisphere(h, o), ~in_hemisphere(x, o)
ans(x) :- uncinate(x, "left")
```

after three commands that load a FreeSurfer `wmparc` with its colour LUT and
a tractogram in its space, and index the one by the other. Edit their paths
in the file (they are read relative to where it runs), then, with the extra
`tracts` of `neurolang-apps`:

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
