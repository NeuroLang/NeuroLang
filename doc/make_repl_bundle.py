"""
Build what the in-browser REPL downloads, in OUTPUT_DIR:

- the wheels of NeuroLang's packages,
- the data the ``neurosynth`` engine needs, as a zip in the layout of its
  data directory, and
- ``bundle.json``, which names them for ``doc/_static/repl/worker.js``, and
- ``highlight.json``, the words the page's code box colours.

The engine downloads its data (about 100 MB) on first use. The browser
needs less: of the MNI152 and Harvard-Oxford images only the ones the
engine reads are kept, and the others are emptied rather than removed,
because the fetchers of nilearn look for every file of a dataset. The
tables that take the engine long to build are kept in the data directory by
its build (``neurolang_apps.utils.engine_cache``) and so are in the zip: the
page starts in about 30 s instead of 2 min. The engine is built again from the
result, to check that it suffices.

Usage::

    python doc/make_repl_bundle.py OUTPUT_DIR
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ENGINE = "neurosynth"
RESOLUTION = 2.0
DATA_ZIP = "neurolang-neurosynth-data.zip"
HIGHLIGHT_WORDS = "highlight.json"

# The images of the datasets the engine reads; the other ones are emptied.
KEPT_IMAGES = {
    "mni_icbm152_t1_tal_nlin_sym_09a.nii.gz",
    "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz",
}
TRIMMED_DATASETS = ("icbm", "fsl")

REPOSITORY = Path(__file__).resolve().parents[1]

# Where the system keeps its certificate authorities.
SYSTEM_CERTIFICATES = (
    Path("/etc/ssl/certs/ca-certificates.crt"),
    Path("/etc/pki/tls/certs/ca-bundle.crt"),
    Path("/etc/ssl/cert.pem"),
)


# The packages of NeuroLang, each depending on those before it: the order
# in which the REPL installs their wheels.
PACKAGES = ("nl_expr", "nl_core", "neurolang", "neurolang_apps")


def build_wheels(output: Path) -> list:
    """Build the wheels of NeuroLang's packages in *output* and return their
    file names, in the order of ``PACKAGES``."""
    uv = shutil.which("uv")
    subprocess.check_call(
        [uv, "build", "--all-packages", "--wheel", "--out-dir", str(output),
         str(REPOSITORY)]
    )
    # uv ignores everything in the directory it builds in, and the
    # documentation is deployed by committing it: the page would be lost.
    (output / ".gitignore").unlink(missing_ok=True)
    return [next(output.glob(f"{name}-*.whl")).name for name in PACKAGES]


def write_highlight_words(output: Path) -> None:
    """Write the words the page highlights in *output*, as the terminal does."""
    from neurolang_apps.utils import query_highlight

    (output / HIGHLIGHT_WORDS).write_text(
        json.dumps(query_highlight.word_sets()) + "\n"
    )


def build_engine(data_dir: Path) -> None:
    """
    Build the engine on *data_dir*, which downloads what it lacks.

    The certificate of a host, nitrc.org's for the Harvard-Oxford atlas,
    may be refused with the authorities of ``certifi`` and accepted with
    those of the system: the download is then tried again with them.
    """
    import requests
    from neurolang_apps.utils import engine_registry

    try:
        engine_registry.build_engine(ENGINE, data_dir, RESOLUTION)
    except requests.exceptions.SSLError:
        system = next((c for c in SYSTEM_CERTIFICATES if c.is_file()), None)
        if system is None:
            raise
        os.environ["REQUESTS_CA_BUNDLE"] = str(system)
        engine_registry.build_engine(ENGINE, data_dir, RESOLUTION)


def check_engine_cache(data_dir: Path) -> None:
    """
    Stop if building the engine kept no tables in *data_dir*.

    The tables the build takes long to find (the studies activating the
    regions of the atlases, see ``neurolang_apps.utils.engine_cache``) are in
    the data zip, so that the page reads them: without them it finds them
    again, which takes a minute and a half of the two it needs to start.
    """
    from neurolang_apps.utils import engine_cache

    files = engine_cache.cached_files(data_dir, ENGINE)
    if not files:
        sys.exit(
            "The engine kept no tables in its data directory: the page "
            "would find them again when it starts."
        )
    size = sum(path.stat().st_size for path in files) / 1048576
    print(f"The engine's tables kept for the page: {len(files)} files, "
          f"{size:.1f} MB")


def trim(data_dir: Path) -> None:
    """Empty the images of the datasets that the engine does not read."""
    for dataset in TRIMMED_DATASETS:
        for image in (data_dir / dataset).rglob("*.nii.gz"):
            if image.name not in KEPT_IMAGES:
                image.write_bytes(b"")


def zip_directory(directory: Path, archive: Path) -> None:
    """Write the files of *directory* in the zip *archive*."""
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(directory))


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    wheels = build_wheels(output)
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        build_engine(data_dir)
        check_engine_cache(data_dir)
        trim(data_dir)
        build_engine(data_dir)  # it must not need what was emptied
        zip_directory(data_dir, output / DATA_ZIP)
    (output / "bundle.json").write_text(
        json.dumps({"wheels": wheels, "data": DATA_ZIP}) + "\n"
    )
    write_highlight_words(output)
    print(f"Wrote {', '.join(wheels)}, {DATA_ZIP} and bundle.json in "
          f"{output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
