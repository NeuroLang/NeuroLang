"""
NeuroStore engine initialization.

NeuroStore (https://neurostore.org) is the database behind Neurosynth
Compose and the one NiMARE's ``fetch_neurostore`` downloads.  Releases are
``nimare-studyset-parquet`` archives; this module reads their parquet
tables directly, so NiMARE itself is not required.

Study ids are NeuroStore's alphanumeric strings, not PubMed integers.

Registers the predicates of the ``neurosynth`` engine (see
:func:`neurolang.utils.engines.neurosynth.init.register_tables`) —
``term_in_study_tfidf(study_id, term, tfidf)`` is computed here from the
study abstracts with scikit-learn (unigrams, English stop words, terms found
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
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import nibabel as nib
import pandas as pd

from neurolang.frontend import NeurolangPDL
from neurolang.utils.engines.neurosynth.init import (
    process_peaks,
    register_tables,
)

NEUROSTORE_URL = "https://neurostore.org"
RELEASES_PATH = "/api/neurostore-studyset-releases/"
RELEASE_ENV_VAR = "NEUROLANG_NEUROSTORE_RELEASE"
_TIMEOUT = 60
MIN_TERM_STUDIES = 10
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


def fetch_neurostore_dir(data_dir: Path) -> Path:
    """Return the directory of the selected release, downloading it once."""
    version = os.environ.get(RELEASE_ENV_VAR, "latest")
    root = Path(data_dir) / "neurostore"
    try:
        release = _select_release(_fetch_releases(), version)
    except OSError:
        # Offline: reuse the newest cached release matching the request.
        cached = sorted(
            p for p in root.glob("*/coordinates.parquet")
            if version in ("latest", p.parent.name)
        )
        if not cached:
            raise
        return cached[-1].parent
    target = root / release["version"]
    # Nightly builds change under the same name.
    if release["release_type"] == "nightly" and target.exists():
        shutil.rmtree(target)
    if not (target / "coordinates.parquet").exists():
        _download_release(release, target)
    return target


def _peaks_from_coordinates(coordinates: pd.DataFrame) -> pd.DataFrame:
    """Shape NeuroStore coordinates like NeuroSynth's peak table."""
    space = coordinates["space"]
    keep = space.isin(_MNI_SPACES | _TALAIRACH_SPACES)
    peaks = coordinates.loc[keep, ["x", "y", "z", "study_id"]].dropna()
    peaks = peaks.rename(columns={"study_id": "id"})
    peaks["space"] = space[peaks.index].map(
        lambda s: "MNI" if s in _MNI_SPACES else "TAL"
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
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        stop_words="english",
        strip_accents="unicode",
        min_df=MIN_TERM_STUDIES,
    )
    matrix = vectorizer.fit_transform(abstracts.values).tocoo()
    terms = vectorizer.get_feature_names_out()
    table = pd.DataFrame(
        {
            "term": terms[matrix.col],
            "tfidf": matrix.data,
            "study_id": abstracts.index.values[matrix.row],
        }
    )
    return table[table["tfidf"] > TFIDF_THRESHOLD].reset_index(drop=True)


def init_engine(
    nl: NeurolangPDL, mask: nib.Nifti1Image, data_dir: Path
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

    """
    release_dir = fetch_neurostore_dir(data_dir)
    coordinates = pd.read_parquet(release_dir / "coordinates.parquet")
    peak_data = process_peaks(
        _peaks_from_coordinates(coordinates), mask, study_id_type=str
    )
    study_ids = peak_data["study_id"].drop_duplicates()

    studies_table = pd.read_parquet(release_dir / "studies.parquet")
    texts = studies_table.set_index("study_id").reindex(study_ids.values)
    abstracts = texts["description"].fillna("")
    abstracts = abstracts[abstracts.str.strip() != ""]
    tfidf = _tfidf_terms(abstracts)
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

    annotations = pd.read_parquet(release_dir / "annotations.parquet")
    studies = set(study_ids)
    for label, suffix in (("concept", "Concepts"), ("domain", "Domain")):
        table = _extract_labels(annotations, suffix).rename(
            columns={suffix: label}
        )
        table = table[table["study_id"].isin(studies)]
        nl.add_tuple_set(table[["study_id", label]], name=f"study_{label}")
