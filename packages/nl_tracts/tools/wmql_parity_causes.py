"""Why a bundle of the vocabulary is not tract_querier's, streamline by
streamline: the companion of ``wmql_real_data.py``.

::

    .venv/bin/python tools/wmql_parity_causes.py \\
        whole_brain_MNI_2pts.trk wmparc.mgz FreeSurferColorLUT.txt \\
        tq.json vocab.json [name=vocab_with_an_option.json ...]

``tq.json`` and ``vocab.json`` are the outputs of ``wmql_real_data.py``
(``tq`` keeps what tract_querier's index says: the labels each streamline
crosses and the boxes of the labels), ``vocab.json`` the default run and
the others runs with one option each (or both). For every query on which
the default run differs, each streamline of the symmetric difference gets:

* whether the background (label 0) is among the labels tract_querier says it
  crosses (at least 2 % of its points are in it): ``only`` counts it as a
  label outside the term (DESIGN.md, 6.3);
* by which of the other runs the streamline is no longer in the difference;
* if by none, its box and endpoints (the first five of them).

It also compares the boxes of tract_querier's index with ``label_extent``
of the atlas, by label, for both extents (``edge``, ``centre``).
"""

import json
import sys
from collections import Counter

import numpy as np


def box_check(atlas, tq):
    from nl_tracts.atlas import label_extent_relation

    boxes = {int(k): np.array(v) for k, v in tq["label_boxes"].items()}
    for extent in ("edge", "centre"):
        rel = label_extent_relation(atlas, extent)
        same = 0
        worst = 0.0
        for label, g in rel.groupby("l"):
            g = g.sort_values("axis")
            ours = np.r_[g["lo"].to_numpy(float), g["hi"].to_numpy(float)]
            delta = np.abs(ours - boxes[int(label)]).max()
            worst = max(worst, delta)
            same += delta < 1e-3
        print(f"label_extent({extent!r}) vs tract_querier's boxes: "
              f"{same} of {len(boxes)} labels the same, "
              f"largest difference {worst:.3f} mm")


def main(trk, atlas_path, lut, tq_json, vocab_json, *others):
    from neurolang.atlases import Atlas
    from nl_tracts.tractogram import Tractogram

    with open(tq_json) as f:
        tq = json.load(f)
    with open(vocab_json) as f:
        vocab = json.load(f)
    runs = {}
    for other in others:
        name, _, path = other.partition("=")
        with open(path) as f:
            runs[name] = json.load(f)["queries"]
    atlas = Atlas.from_freesurfer("fs", atlas_path, lut, "MNI152")
    box_check(atlas, tq)
    tractogram = Tractogram.from_file("t", trk, "MNI152")
    crossing = {int(k): set(v) for k, v in tq["crossing"].items()}
    for name, got in vocab["queries"].items():
        a, b = set(tq["queries"][name]), set(got)
        if a == b:
            continue
        for side, ids in (("tq only", sorted(a - b)),
                          ("vocabulary only", sorted(b - a))):
            causes = Counter()
            other = []
            for s in ids:
                fixed = tuple(
                    run for run, queries in runs.items()
                    if (s in set(queries[name])) == (s in a)
                )
                causes[(
                    "background in tq's labels" if 0 in crossing[s]
                    else "no background", "resolved by " + (
                        " and ".join(fixed) if fixed else "none")
                )] += 1
                if not fixed:
                    points = tractogram.points[
                        tractogram.offsets[s]:tractogram.offsets[s + 1]]
                    other.append((s, points.min(axis=0).tolist(),
                                  points.max(axis=0).tolist(),
                                  points[0].tolist(), points[-1].tolist()))
            if ids:
                print(f"{name}: {side} {len(ids)}: {dict(causes)}")
            for row in other[:5]:
                print("   unresolved:", row)


if __name__ == "__main__":
    main(*sys.argv[1:])
