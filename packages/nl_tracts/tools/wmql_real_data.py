"""The bundles of a ``.qry`` on real data: tract_querier against the
translation of the ``.qry`` to the vocabulary (``nl_tracts.wmql_vocab``).

Three steps, each writing or reading a JSON file of
``{"queries": {name: [streamline positions]}, "seconds": {...}}``:

* ``tq``: tract_querier itself, run with *its* Python (a separate
  environment: it needs vtk), on the tractogram and the label image as its
  command line reads them: the ``.trk`` by nibabel, in RAS+ mm, and the
  label image with its affine; crossing threshold 2 %, no length threshold.
  Every query defined (``=`` and ``|=``) is kept, ``x.left`` as such::

      /tmp/tq_venv/bin/python tools/wmql_real_data.py tq \\
          whole_brain_MNI_2pts.trk wmparc.mgz freesurfer_queries.qry tq.json

* ``vocab``: the ``.qry`` translated to the vocabulary (with the terms of
  ``freesurfer_aseg_aparc``, as ``nl_tracts/bundles``), on the same files,
  through the commands of nl_tracts, every bundle at once (``solve_all``;
  pandas backend)::

      .venv/bin/python tools/wmql_real_data.py vocab \\
          whole_brain_MNI_2pts.trk wmparc.mgz FreeSurferColorLUT.txt \\
          freesurfer_queries.qry vocab.json

* ``compare tq.json vocab.json``: a Markdown table, tract, side, the two
  counts and their Jaccard index.

The two options of the vocabulary that reproduce tract_querier where the
two differ by design (DESIGN.md, 6.1 and 6.1.1) are the words ``background``
(``.index_tracts(..., emit_background=1)``) and ``extent``
(``extent="centre"``) after the output of ``vocab``;
``wmql_parity_causes.py`` says why each difference is one.

Both read the streamlines in the order of the file, so that a streamline is
its position ``s`` in both (one tractogram: ``x`` is ``s``).
"""

import json
import resource
import sys
import time


def peak_gb():
    """Peak resident memory of this process, GB (macOS counts bytes,
    Linux kilobytes)."""
    scale = 1 << 30 if sys.platform == "darwin" else 1 << 20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def run_tq(trk, atlas, qry, out):
    import os

    import nibabel
    import tract_querier

    t0 = time.perf_counter()
    image = nibabel.load(atlas)
    tracts = tract_querier.tractography.tractography_from_file(trk).tracts()
    t1 = time.perf_counter()
    indexing = tract_querier.TractographySpatialIndexing(
        tracts, image.get_fdata(), image.affine, 0, 2.0
    )
    t2 = time.perf_counter()
    folders = [os.path.dirname(os.path.abspath(qry)),
               tract_querier.default_queries_folder]
    with open(qry) as f:
        body = tract_querier.queries_preprocess(
            f.read(), filename=qry, include_folders=folders
        )
    evaluator = tract_querier.query_processor.EvaluateQueries(indexing)
    evaluator.visit(tract_querier.query_processor.ast.Module(
        body, type_ignores=[]))
    t3 = time.perf_counter()
    queries = {
        name: sorted(int(t) for t in info.tracts)
        for name, info in evaluator.evaluated_queries_info.items()
    }
    seconds = {"load": t1 - t0, "index": t2 - t1, "queries": t3 - t2}
    with open(out, "w") as f:
        json.dump({
            "queries": queries, "seconds": seconds, "peak_gb": peak_gb(),
            # for tools/wmql_parity_causes.py: what the index says
            "crossing": {
                int(t): sorted(int(label) for label in labels)
                for t, labels in indexing.crossing_tracts_labels.items()
            },
            "label_boxes": {
                int(label): [float(v) for v in box]
                for label, box in indexing.label_bounding_boxes.items()
            },
        }, f)
    print(seconds, peak_gb())


def run_vocab(trk, atlas, lut, qry, out, *options):
    """``options``: ``background`` (``emit_background=1`` of
    ``.index_tracts``) and ``extent`` (``extent="centre"``)."""
    import os

    from nl_tracts.tests.engine import tracts_engine
    from nl_tracts.terms import load_term_mapping
    from nl_tracts.wmql_vocab import translate

    with open(qry) as f:
        translation = translate(
            f.read(), [os.path.dirname(os.path.abspath(qry))],
            terms=set(load_term_mapping("freesurfer_aseg_aparc")["term"]),
            skip_unsupported=True,
        )
    index = ""
    if "background" in options:
        index += ", emit_background=1"
    if "extent" in options:
        index += ', extent="centre"'
    t0 = time.perf_counter()
    nl = tracts_engine()
    nl.execute_datalog_program(
        f'.load_atlas("fs", "{atlas}", "{lut}", kind="freesurfer", '
        'space="MNI152")\n'
        f'.load_tractogram("t", "{trk}", space="MNI152")\n'
        '.index_tracts("t", "fs", term_mapping="freesurfer_aseg_aparc"'
        f'{index})\n'
    )
    t1 = time.perf_counter()
    nl.execute_datalog_program(translation.text)
    solution = nl.solve_all()  # every bundle, both sides, at once
    t2 = time.perf_counter()
    queries = {}
    for name, atom in translation.queries.items():
        pred, _, side = atom[:-1].partition("(x")
        rows = solution[pred].as_pandas_dataframe() if pred in solution \
            else None
        if rows is None or rows.empty:
            queries[name] = []
            continue
        if side:
            rows = rows[rows.iloc[:, 1] == side.strip(', "')]
        queries[name] = sorted(int(x) for x in rows.iloc[:, 0])
    seconds = {"load_and_index": t1 - t0, "queries": t2 - t1}
    with open(out, "w") as f:
        json.dump({"queries": queries, "seconds": seconds,
                   "skipped": translation.skipped, "options": options,
                   "peak_gb": peak_gb()}, f)
    print(seconds, peak_gb())


def compare(tq_json, vocab_json):
    with open(tq_json) as f:
        tq = json.load(f)
    with open(vocab_json) as f:
        vocab = json.load(f)
    print("| Tract | Side | n_tq | n_vocab | Jaccard |")
    print("|---|---|---|---|---|")
    identical = 0
    for key, xs in vocab["queries"].items():
        a, b = set(tq["queries"][key]), set(xs)
        union = a | b
        jaccard = len(a & b) / len(union) if union else 1.0
        identical += a == b
        tract, _, side = key.partition(".")
        print(f"| {tract} | {side or '-'} | {len(a)} | {len(b)} | "
              f"{jaccard:.3f} |")
    print(f"\n{identical} of {len(vocab['queries'])} the same streamlines, "
          f"{len(vocab['skipped'])} skipped")
    print(f"tract_querier: {tq['seconds']}, peak {tq['peak_gb']:.1f} GB")
    print(f"vocabulary: {vocab['seconds']}, peak {vocab['peak_gb']:.1f} GB, "
          f"options {vocab.get('options', [])}")
    for key, why in vocab["skipped"].items():
        print(f"skipped {key}: {why}")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    {"tq": run_tq, "vocab": run_vocab, "compare": compare}[command](*args)
