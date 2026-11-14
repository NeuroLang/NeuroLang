"""Write the shipped bundle modules ``bundles/*.dl`` of the engine tracts.

Each is a ``.qry`` file of tract_querier, translated to the vocabulary of
tractography by the engine's ``wmql_vocab``: one predicate per query, with
the statement of WMQL over it. The files of FreeSurfer are translated with
the terms of the shipped mapping ``freesurfer_aseg_aparc`` (a name that is
not one of them stands for the terms it is made of, or is left out); Mori's
against no mapping, the names of its atlas (JHU) being its terms. The JHU
files define labels only, and give no module.

Usage: ``python tools/make_bundle_modules.py [tract_querier data directory
[output directory]]``; the data directory defaults to ``data`` of
``TRACT_QUERIER_DIR``, the output to the engine's ``bundles``.
"""

import os
import sys
from pathlib import Path

from neurolang_apps.utils.engines.tracts.terms import load_term_mapping
from neurolang_apps.utils.engines.tracts.wmql_vocab import BUNDLES, translate

#: module -> the shipped term mapping its names are checked against
MAPPINGS = {
    "freesurfer_queries": "freesurfer_aseg_aparc",
    "freesurfer_queries_bsf2016": "freesurfer_aseg_aparc",
    "freesurfer_queries_bsf2016_toad": "freesurfer_aseg_aparc",
    "mori_queries": None,
}
assert tuple(MAPPINGS) == BUNDLES


def module_text(name, data_dir):
    """The text of the module ``name``, from the ``.qry`` of ``data_dir``."""
    mapping = MAPPINGS[name]
    terms = None if mapping is None else set(
        load_term_mapping(mapping)["term"])
    path = Path(data_dir) / f"{name}.qry"
    return translate(
        path.read_text(), [str(data_dir)], terms=terms,
        skip_unsupported=True, qry_file=path.name,
        title=f"tract_querier's {path.name}"
        + ("" if mapping is None else f", with the terms of {mapping}"),
    ).text


def main(data_dir, out):
    out = Path(out)
    out.mkdir(exist_ok=True)
    for name in BUNDLES:
        (out / f"{name}.dl").write_text(module_text(name, data_dir))


if __name__ == "__main__":
    # the package of the engine, from packages/neurolang_apps/tools/tracts
    engine = (Path(__file__).resolve().parents[2]
              / "neurolang_apps" / "utils" / "engines" / "tracts")
    main(
        sys.argv[1] if len(sys.argv) > 1
        else Path(os.environ["TRACT_QUERIER_DIR"]) / "data",
        sys.argv[2] if len(sys.argv) > 2 else engine / "bundles",
    )
