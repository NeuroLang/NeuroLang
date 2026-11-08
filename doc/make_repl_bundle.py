"""
Build what the in-browser REPL downloads, in OUTPUT_DIR:

- the wheels of NeuroLang's packages,
- for each engine of the page (``neurosynth``, the page's default, and
  ``neurostore``), the data it needs, as a zip in the layout of its data
  directory, and
- ``bundle.json``, which names them for ``doc/_static/repl/worker.js``,
- ``highlight.json``, the words the page's code box colours, and
- with torch installed (the ``embeddings`` extra), the files of the
  engines' text embeddings, which the page fetches when a query first
  needs them: the SPECTER2 query model for :mod:`specter2_numpy
  <neurolang_apps.utils.specter2_numpy>` and the embeddings of the
  studies, named in ``bundle.json`` by their path in the data directory.

The engine downloads its data (about 100 MB) on first use. The browser
needs less: of the MNI152 and Harvard-Oxford images only the ones the
engine reads are kept, and the others are emptied rather than removed,
because the fetchers of nilearn look for every file of a dataset; of the
NeuroStore release only the columns of three of its tables. The
tables that take the engine long to build are kept in the data directory by
its build (``neurolang_apps.utils.engine_cache``) and so are in the zip: the
page starts in about 30 s instead of 2 min. The engine is built again from the
result, with the tables and without them, to check that it suffices.

Usage::

    python doc/make_repl_bundle.py OUTPUT_DIR [EMBEDDINGS_DATA_DIR]
        [--downloads DIR] [--engine NAME ...]

with the engines of the page by default; the page downloads the data of the
one it is asked for. An engine other than the default that cannot be built is
left out, with a message.

Embedding the studies takes their abstracts from PubMed and a while:
EMBEDDINGS_DATA_DIR keeps them: a data directory where they were embedded
already (``python -m neurolang.utils.text_embeddings -d DIR``), or where
they are saved for the next time, as CI caches it.

The engine is built in a temporary data directory. ``--downloads`` keeps
what it downloads there between builds, as CI caches it: the files found in
DIR are not downloaded again, and those downloaded are added to DIR (the
engine's kept tables and embeddings are not), even when the build fails,
so that a data host that is down does not stop a build that had them.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# The engines the page offers, the first being its default.
ENGINES = ("neurosynth", "neurostore")
ENGINE = ENGINES[0]
LABELS = {"neurosynth": "Neurosynth", "neurostore": "NeuroStore"}
RESOLUTION = 2.0
HIGHLIGHT_WORDS = "highlight.json"

# The tables of a NeuroStore release that the engine reads, and their columns
# (all, or those that end with one of the suffixes, besides study_id).
NEUROSTORE_TABLES = {
    "coordinates.parquet": ("study_id", "x", "y", "z", "space"),
    "studies.parquet": ("study_id", "name", "description"),
    "annotations.parquet": None,
}


def data_zip(engine: str = ENGINE) -> str:
    """Return the name of the zip of the data of *engine*."""
    return f"neurolang-{engine}-data.zip"


# The images of the datasets the engine reads; the other ones are emptied.
KEPT_IMAGES = {
    "mni_icbm152_t1_tal_nlin_sym_09a.nii.gz",
    "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz",
}
TRIMMED_DATASETS = ("icbm", "fsl")

# GitHub refuses files of more than 100 MB, and the documentation is
# deployed by committing it: a larger file is written in parts of this
# size, which the page joins.
PART_SIZE = 95 * 2 ** 20

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
    file names, in the order of ``PACKAGES``.

    The wheels of an earlier build are removed first: their version is
    that of the commit they were built from, so they keep their names and
    would be taken for the new ones."""
    for name in PACKAGES:
        for wheel in output.glob(f"{name}-*.whl"):
            wheel.unlink()
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


def build_engine(
    data_dir: Path, use_cache: bool = True, engine: str = ENGINE
) -> None:
    """
    Build the engine on *data_dir*, which downloads what it lacks.

    The certificate of a host, nitrc.org's for the Harvard-Oxford atlas,
    may be refused with the authorities of ``certifi`` and accepted with
    those of the system: the download is then tried again with them.
    """
    import requests
    from neurolang_apps.utils import engine_registry

    try:
        engine_registry.build_engine(
            engine, data_dir, RESOLUTION, use_cache=use_cache
        )
    except requests.exceptions.SSLError:
        system = next((c for c in SYSTEM_CERTIFICATES if c.is_file()), None)
        if system is None:
            raise
        os.environ["REQUESTS_CA_BUNDLE"] = str(system)
        engine_registry.build_engine(
            engine, data_dir, RESOLUTION, use_cache=use_cache
        )


def keep_downloads(source: Path, target: Path) -> None:
    """Copy the data directory *source* into *target*, without
    the engine's kept tables nor its text embeddings."""
    from neurolang_apps.utils import engine_cache

    ignored = {engine_cache.DIRECTORY, "embeddings"}
    shutil.copytree(
        source, target, dirs_exist_ok=True,
        ignore=lambda directory, names: [n for n in names if n in ignored],
    )


def check_engine_cache(data_dir: Path, engine: str = ENGINE) -> None:
    """
    Stop if building the engine kept no tables in *data_dir*.

    The tables the build takes long to find (the studies activating the
    regions of the atlases, see ``neurolang_apps.utils.engine_cache``) are in
    the data zip, so that the page reads them: without them it finds them
    again, which takes a minute and a half of the two it needs to start.
    """
    from neurolang_apps.utils import engine_cache

    files = engine_cache.cached_files(data_dir, engine)
    if not files:
        sys.exit(
            "The engine kept no tables in its data directory: the page "
            "would find them again when it starts."
        )
    size = sum(path.stat().st_size for path in files) / 1048576
    print(f"The engine's tables kept for the page: {len(files)} files, "
          f"{size:.1f} MB")


def trim(data_dir: Path, engine: str = ENGINE) -> None:
    """Empty the images of the datasets that the engine does not read, and
    keep of a NeuroStore release only what it reads."""
    for dataset in TRIMMED_DATASETS:
        for image in (data_dir / dataset).rglob("*.nii.gz"):
            if image.name not in KEPT_IMAGES:
                image.write_bytes(b"")
    if engine == "neurostore":
        trim_neurostore(data_dir / "neurostore")


def trim_neurostore(directory: Path) -> None:
    """
    Keep in each release of *directory* the tables the engine reads, and of
    them the columns it reads: the annotations have almost a thousand, of
    which it takes those of the concepts and the domains of the tasks.
    """
    import pyarrow.parquet as parquet

    for release in directory.iterdir():
        if not release.is_dir():
            continue
        for path in release.iterdir():
            if path.name not in NEUROSTORE_TABLES:
                path.unlink()
        for name, columns in NEUROSTORE_TABLES.items():
            path = release / name
            if not path.exists():
                continue
            names = parquet.read_schema(path).names
            if columns is None:
                columns = [
                    n for n in names
                    if n == "study_id" or n.endswith((".Concepts", ".Domain"))
                ]
            table = parquet.read_table(path, columns=list(columns))
            parquet.write_table(table, path, compression="zstd")


def zip_directory(directory: Path, archive: Path) -> None:
    """Write the files of *directory* in the zip *archive*."""
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(directory))


def write_parts(path: Path, output: Path, name: str) -> list:
    """Copy *path* to *output* as *name*, in parts of ``PART_SIZE`` if it
    is larger; return the names of the files written."""
    names = []
    with open(path, "rb") as source:
        while True:
            chunk = source.read(PART_SIZE)
            if not chunk and names:
                break
            names.append(f"{name}.{len(names)}")
            (output / names[-1]).write_bytes(chunk)
            if len(chunk) < PART_SIZE:
                break
    if len(names) == 1:
        (output / names[0]).rename(output / name)
        names = [name]
    return names


def bundle_name(path: Path, engine: str = ENGINE) -> str:
    """Return the name in the bundle of the file *path* of the embeddings of
    *engine*: without ``+``, a space in the URLs of some servers."""
    return f"{engine}-{path.name.replace('+', '_')}"


def export_embeddings(
    data_dir: Path, output: Path, embedded=None, engine: str = ENGINE
) -> dict:
    """
    Write in *output* the files of the engine's text embeddings: for each
    index, the query model of the numpy encoder and the embeddings of the
    documents. Both are taken from the data directory *embedded* when it
    has them, and saved in it when made, which needs torch: without it,
    or if they cannot be made, the page goes without them.

    They are left out of *data_dir*, not to be zipped with the data.
    Return the names of the files of each, by its path in the data
    directory.
    """
    from neurolang.utils import engine_registry, text_embeddings
    from neurolang_apps.utils import specter2_numpy

    emb_dir = data_dir / engine / "embeddings"
    saved = None
    if embedded is not None:
        saved = Path(embedded) / engine / "embeddings"
    if saved is not None and saved.exists():
        shutil.copytree(saved, emb_dir, dirs_exist_ok=True)
    torch = text_embeddings._torch_available()
    files = {}
    try:
        nl = engine_registry.build_engine(engine, data_dir, RESOLUTION)
        for name, index in nl.text_embedding_indexes.items():
            model = index.encoder.numpy_path
            if not model.exists() and torch:
                base, _, query = index.encoder.download()
                specter2_numpy.export(base, query, model)
            if not model.exists() or not (torch or index.cache_path.exists()):
                print(f"No torch: the text embeddings {name} are left out.")
                continue
            index.build()
            for path in (model, index.cache_path):
                files[str(path.relative_to(data_dir))] = write_parts(
                    path, output, bundle_name(path, engine)
                )
    except Exception as error:  # the page works without them
        print(f"The text embeddings are left out: {error!r}")
        files = {}
    if saved is not None and emb_dir.exists():
        shutil.copytree(emb_dir, saved, dirs_exist_ok=True)
    shutil.rmtree(emb_dir, ignore_errors=True)
    return files


def bundle_engine(
    engine: str, output: Path, embedded=None, downloads=None
) -> dict:
    """
    Build the data of *engine* for the page and write its zip in *output*,
    in parts if it is larger than GitHub takes; return its entry of
    ``bundle.json``: its label, its data (the name of the zip, or those of its
    parts, which the page joins) and the files of its text embeddings.
    """
    # What an engine downloads is kept apart from that of the others, so that
    # the zip of an engine holds its data only.
    kept = None if downloads is None else Path(downloads) / engine
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        if kept is not None and kept.is_dir():
            keep_downloads(kept, data_dir)
        try:
            build_engine(data_dir, engine=engine)
        finally:
            if kept is not None:
                keep_downloads(data_dir, kept)
        check_engine_cache(data_dir, engine)
        trim(data_dir, engine)
        build_engine(data_dir, engine=engine)  # the page's way: from the kept tables
        build_engine(data_dir, use_cache=False, engine=engine)  # and without them
        files = export_embeddings(data_dir, output, embedded, engine)
        archive = Path(tmp) / data_zip(engine)
        zip_directory(data_dir, archive)
        names = write_parts(archive, output, data_zip(engine))
    return {
        "label": LABELS.get(engine, engine),
        "data": names[0] if len(names) == 1 else names,
        "files": files,
    }


def main(
    output: Path, embedded=None, downloads=None, engines=ENGINES
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    wheels = build_wheels(output)
    built = {}
    for engine in engines:
        try:
            built[engine] = bundle_engine(
                engine, output, embedded, downloads
            )
        except Exception as error:
            if engine == engines[0]:
                raise  # the page's default: there is no page without it
            print(f"The engine {engine} is left out of the bundle: {error!r}")
    default = engines[0]
    (output / "bundle.json").write_text(
        json.dumps({
            "wheels": wheels, "default": default, "engines": built,
            # What a page of before the choice of the engine reads.
            "data": built[default]["data"], "files": built[default]["files"],
        }) + "\n"
    )
    write_highlight_words(output)
    names = [
        name for engine in built for name in [built[engine]["data"]]
        for name in ([name] if isinstance(name, str) else name)
    ]
    parts = [
        name for engine in built for names_ in built[engine]["files"].values()
        for name in names_
    ]
    print(f"Wrote {', '.join(wheels)}, {', '.join(names)}, "
          f"{', '.join(parts) or 'no embeddings'}, bundle.json and "
          f"highlight.json in {output}")


def parse_args(argv=None):
    """Return the options of the command line *argv*."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("embedded", nargs="?")
    parser.add_argument("--downloads", type=Path)
    parser.add_argument(
        "--engine", action="append", choices=ENGINES, dest="engines",
        help="an engine of the page (repeat); by default all of them",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    # The default first, whatever the order of the options.
    chosen = [e for e in ENGINES if e in (args.engines or ENGINES)]
    main(args.output, args.embedded, args.downloads, tuple(chosen))
