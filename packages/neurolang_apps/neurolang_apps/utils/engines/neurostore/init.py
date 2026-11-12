"""
NeuroStore engine initialization.

NeuroStore (https://neurostore.org) is the database behind Neurosynth
Compose and the one NiMARE's ``fetch_neurostore`` downloads.  Releases are
``nimare-studyset-parquet`` archives; this module reads their parquet
tables directly, so NiMARE itself is not required.

Study ids are NeuroStore's alphanumeric strings, not PubMed integers.

Registers the predicates of the ``neurosynth`` engine (see
:func:`neurolang_apps.utils.engines.neurosynth.init.register_tables`) —
``term_in_study_tfidf(study_id, term, tfidf)`` is computed here from the
study abstracts with scikit-learn (words and sequences of two or three, so
that terms such as ``working memory`` exist; English stop words, terms found
in at least ``MIN_TERM_STUDIES`` studies, values above ``TFIDF_THRESHOLD``),
since NeuroStore ships no term table — plus:

* ``study_title(study_id, title)`` and ``study_abstract(study_id,
  abstract)`` — the texts of the study, which the engine's
  ``text_embeddings`` embed with SPECTER2.
* ``study_concept(study_id, concept)`` — cognitive concepts that the
  LLM-based task extraction attached to the study's analyses.
* ``study_domain(study_id, domain)`` — likewise for cognitive domains.

The release can be chosen with the ``NEUROLANG_NEUROSTORE_RELEASE``
environment variable (``latest``, ``nightly`` or a version such as
``2026-10``); the default is ``latest``, the newest monthly release.
"""

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import nibabel as nib
import numpy as np
import pandas as pd

from neurolang.frontend import NeurolangPDL
from neurolang_apps.utils import engine_cache
from neurolang_apps.utils.engines.neurosynth.init import (
    process_peaks,
    register_tables,
)

NEUROSTORE_URL = "https://neurostore.org"
RELEASES_PATH = "/api/neurostore-studyset-releases/"
RELEASE_ENV_VAR = "NEUROLANG_NEUROSTORE_RELEASE"
_TIMEOUT = 60
MIN_TERM_STUDIES = 30
NGRAM_RANGE = (1, 3)
TFIDF_THRESHOLD = 1e-3
_MNI_SPACES = {"MNI"}
_TALAIRACH_SPACES = {"TAL", "Talairach"}


def _fetch_releases() -> List[Dict[str, Any]]:
    with urllib.request.urlopen(
        NEUROSTORE_URL + RELEASES_PATH, timeout=_TIMEOUT
    ) as response:
        return json.load(response)["results"]


def _select_release(releases: List[Dict[str, Any]], version: str) -> Dict:
    if version == "latest":
        dated = [r for r in releases if r["release_type"] != "nightly"]
        return max(dated or releases, key=lambda r: r["built_at"])
    for release in releases:
        if release["version"] == version:
            return release
    available = ", ".join(r["version"] for r in releases)
    raise ValueError(
        f"Unknown NeuroStore release {version!r}. Available: {available}"
    )


def _download_release(release: Dict[str, Any], target: Path) -> None:
    """Download, verify and extract *release* into *target*."""
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive = Path(tmp) / "release.tar.gz"
        url = NEUROSTORE_URL + release["download_path"]
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as response:
            with open(archive, "wb") as f:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    digest.update(chunk)
                    f.write(chunk)
        expected = release.get("archive_checksum")
        if expected and digest.hexdigest() != expected:
            raise ValueError(
                "Checksum mismatch for NeuroStore release "
                f"{release['version']!r}"
            )
        extracted = Path(tmp) / "extracted"
        with tarfile.open(archive) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(extracted, filter="data")
            else:  # Python without extraction filters
                tar.extractall(extracted)
        (studyset_json,) = extracted.rglob("studyset.json")
        shutil.move(str(studyset_json.parent), str(target))


def _cached_release(root: Path, version: str):
    """The newest release of *root* matching *version*, or None."""
    cached = sorted(
        p for p in root.glob("*/coordinates.parquet")
        if version in ("latest", p.parent.name)
    )
    return cached[-1].parent if cached else None


def fetch_neurostore_dir(data_dir: Path) -> Path:
    """Return the directory of the selected release, downloading it once.

    In a browser (Pyodide) the release is the one that came with the page: the
    network is not asked.
    """
    version = os.environ.get(RELEASE_ENV_VAR, "latest")
    root = Path(data_dir) / "neurostore"
    if sys.platform == "emscripten":
        bundled = _cached_release(root, version)
        if bundled is not None:
            return bundled
    try:
        release = _select_release(_fetch_releases(), version)
    except OSError:
        # Offline: reuse the newest cached release matching the request.
        cached = _cached_release(root, version)
        if cached is None:
            raise
        return cached
    target = root / release["version"]
    # Nightly builds change under the same name.
    if release["release_type"] == "nightly" and target.exists():
        shutil.rmtree(target)
    if not (target / "coordinates.parquet").exists():
        _download_release(release, target)
    return target


def _peaks_from_coordinates(coordinates: pd.DataFrame) -> pd.DataFrame:
    """
    Shape NeuroStore coordinates like NeuroSynth's peak table.

    A space other than MNI or Talairach (or none) becomes ``UNKNOWN``,
    handled by :func:`process_peaks` as its *unknown_space* says.
    """
    space = coordinates["space"]
    peaks = coordinates[["x", "y", "z", "study_id"]].dropna()
    peaks = peaks.rename(columns={"study_id": "id"})
    peaks["space"] = space[peaks.index].map(
        lambda s: "MNI" if s in _MNI_SPACES
        else "TAL" if s in _TALAIRACH_SPACES else "UNKNOWN"
    )
    return peaks


def _extract_labels(annotations: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Collect the JSON-list valued ``*.<suffix>`` columns per study."""
    columns = [c for c in annotations.columns if c.endswith("." + suffix)]
    frames = []
    for column in columns:
        values = annotations[["study_id", column]].dropna()
        values[column] = values[column].map(_parse_list)
        frames.append(
            values.explode(column).rename(columns={column: suffix})
        )
    if not frames:
        return pd.DataFrame({"study_id": [], suffix: []}, dtype=str)
    labels = pd.concat(frames).dropna()
    labels[suffix] = labels[suffix].astype(str).str.strip()
    labels = labels[labels[suffix] != ""].drop_duplicates()
    return labels.reset_index(drop=True)


def _parse_list(raw: str) -> list:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def _tfidf_terms(abstracts: pd.Series) -> pd.DataFrame:
    """TF-IDF of the abstract terms as ``term, tfidf, study_id`` rows."""
    return _tfidf_table(_tfidf_arrays(abstracts))


def _tfidf_arrays(abstracts: pd.Series) -> dict:
    """
    Compute the TF-IDF of the terms of *abstracts* as arrays.

    They are the ``study_ids`` and the ``terms``, and for each of the
    studies, from ``indptr`` on, the ``col`` of the term of each ``value``
    above the threshold.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        stop_words="english",
        strip_accents="unicode",
        min_df=MIN_TERM_STUDIES,
        ngram_range=NGRAM_RANGE,
    )
    matrix = vectorizer.fit_transform(abstracts.values).tocoo()
    keep = matrix.data > TFIDF_THRESHOLD
    terms = np.array(vectorizer.get_feature_names_out(), dtype=str)
    # The rows are in order: they are kept as the offsets of each study's
    # terms, and the terms by 16 bits when the vocabulary allows, which makes
    # the file half as large.
    per_study = np.bincount(matrix.row[keep], minlength=len(abstracts))
    return {
        "study_ids": np.array(abstracts.index.values, dtype=str),
        "terms": terms,
        "indptr": np.concatenate([[0], np.cumsum(per_study)]).astype(np.int32),
        "col": matrix.col[keep].astype(
            np.uint16 if len(terms) < 2 ** 16 else np.int32
        ),
        "value": matrix.data[keep].astype(np.float32),
    }


def _tfidf_table(arrays: dict) -> pd.DataFrame:
    """The ``term, tfidf, study_id`` rows of the arrays of the TF-IDF."""
    # Object arrays share the strings of the vocabulary and of the ids: the
    # fixed-width strings of numpy would take hundreds of MB for the rows.
    terms = arrays["terms"].astype(object)
    study_ids = arrays["study_ids"].astype(object)
    row = np.repeat(np.arange(len(study_ids)), np.diff(arrays["indptr"]))
    return pd.DataFrame(
        {
            "term": terms[arrays["col"].astype(np.int64)],
            "tfidf": arrays["value"].astype(np.float64),
            "study_id": study_ids[row],
        }
    )


def _kept_tfidf_terms(abstracts, release, cache_dir) -> pd.DataFrame:
    """
    The TF-IDF of the terms of *abstracts*, kept in *cache_dir* (if any)
    for the same release and settings.

    It takes about 20 s, and more than a minute and a lot of memory in a
    browser; what is kept is arrays, which any Python reads (see
    :mod:`~neurolang_apps.utils.engine_cache`).
    """
    key = engine_cache.digest(
        "neurostore-tfidf", release, MIN_TERM_STUDIES, NGRAM_RANGE,
        TFIDF_THRESHOLD, np.array(abstracts.index.values, dtype=str),
    )
    arrays = engine_cache.keep_arrays(
        cache_dir, "tfidf", key, lambda: _tfidf_arrays(abstracts)
    )
    return _tfidf_table(arrays)


def _annotation_columns(path) -> List[str]:
    """The columns of the annotations the engine reads: the study and the
    concepts and domains of the tasks."""
    import pyarrow.parquet

    return [
        name for name in pyarrow.parquet.read_schema(path).names
        if name == "study_id" or name.endswith((".Concepts", ".Domain"))
    ]


def init_engine(
    nl: NeurolangPDL, mask: nib.Nifti1Image, data_dir: Path,
    unknown_space: str = "mni", cache_dir=None
) -> None:
    """Populate a fresh engine with NeuroStore data.

    Parameters
    ----------
    nl :
        Fresh :class:`~neurolang.frontend.NeurolangPDL` instance.
    mask :
        MNI template image for voxel-space coordinate mapping.
    data_dir :
        Directory under which downloaded data is cached.
    unknown_space :
        How the peaks of an unknown space are handled (``mni``, ``tal`` or
        ``drop``, see :func:`process_peaks`); the engine's
        ``python_init_options``.
    cache_dir :
        Directory where the TF-IDF table is kept, if any.

    """
    release_dir = fetch_neurostore_dir(data_dir)
    coordinates = pd.read_parquet(release_dir / "coordinates.parquet")
    peak_data = process_peaks(
        _peaks_from_coordinates(coordinates), mask, study_id_type=str,
        unknown_space=unknown_space,
    )
    study_ids = peak_data["study_id"].drop_duplicates()

    studies_table = pd.read_parquet(release_dir / "studies.parquet")
    texts = studies_table.set_index("study_id").reindex(study_ids.values)
    abstracts = texts["description"].fillna("")
    abstracts = abstracts[abstracts.str.strip() != ""]
    tfidf = _kept_tfidf_terms(abstracts, release_dir.name, cache_dir)
    register_tables(
        nl, mask, data_dir, peak_data,
        tfidf[["study_id", "term", "tfidf"]],
    )

    for name, column, values in (
        ("study_abstract", "abstract", abstracts),
        ("study_title", "title", texts["name"].fillna("")),
    ):
        table = values[values.str.strip() != ""].rename(column)
        table = table.rename_axis("study_id").reset_index()
        nl.add_tuple_set(table, name=name)

    annotations_path = release_dir / "annotations.parquet"
    annotations = pd.read_parquet(
        annotations_path, columns=_annotation_columns(annotations_path)
    )
    studies = set(study_ids)
    for label, suffix in (("concept", "Concepts"), ("domain", "Domain")):
        table = _extract_labels(annotations, suffix).rename(
            columns={suffix: label}
        )
        table = table[table["study_id"].isin(studies)]
        nl.add_tuple_set(table[["study_id", label]], name=f"study_{label}")
