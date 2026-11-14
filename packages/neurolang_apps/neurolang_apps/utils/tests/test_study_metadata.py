"""Tests of the years, journals and abstracts of the studies of the
meta-analysis engines."""

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from neurolang_apps.utils.engines.neurostore import init as neurostore
from neurolang_apps.utils.engines.neurosynth import init as neurosynth


def _rows(table):
    return sorted(map(tuple, table.itertuples(index=False)))


class TestRegisterPublications:
    def test_the_known_years_and_journals(self):
        nl = MagicMock()
        neurosynth.register_publications(nl, pd.DataFrame({
            "study_id": ["a", "b", "c"],
            "year": [2001.0, None, 2003.0],
            "journal": ["Brain", " ", None],
            "pmid": ["101", "PMC7", None],
            "doi": ["https://doi.org/10.1/AB", "doi:10.2/x", "not a doi"],
        }))
        tables = {
            call.kwargs["name"]: call.args[0]
            for call in nl.add_tuple_set.call_args_list
        }
        assert _rows(tables["study_year"]) == [("a", 2001), ("c", 2003)]
        assert tables["study_year"]["year"].dtype == int
        assert _rows(tables["study_journal"]) == [("a", "Brain")]
        assert _rows(tables["study_pmid"]) == [("a", 101)]
        assert tables["study_pmid"]["pmid"].dtype == int
        assert _rows(tables["study_doi"]) == [("a", "10.1/ab"), ("b", "10.2/x")]


class TestAbstracts:
    def test_pubmed_is_asked_once(self, tmp_path, monkeypatch):
        asked = []

        def efetch(pmids, email, api_key):
            asked.extend(pmids)
            return {p: (f"Title {p}", f"Abstract {p}") for p in pmids
                    if p != "3"}

        monkeypatch.setattr(
            "neurolang_apps.utils.text_embeddings._efetch", efetch
        )
        ids = pd.Series([1, 2, 3])
        table = neurosynth.study_abstracts(ids, tmp_path)
        assert _rows(table) == [(1, "Abstract 1"), (2, "Abstract 2")]
        neurosynth.study_abstracts(ids, tmp_path)
        assert sorted(asked) == ["1", "2", "3"]

    def test_offline_the_abstracts_kept(self, tmp_path):
        path = tmp_path / "neurosynth" / neurosynth.ABSTRACTS_FILE
        path.parent.mkdir()
        path.write_text(json.dumps({"1": ["T", "Kept"]}))
        with pytest.warns(UserWarning, match="PubMed could not be asked"):
            table = neurosynth.study_abstracts(pd.Series([1, 2]), tmp_path)
        assert _rows(table) == [(1, "Kept")]

    def test_a_browser_does_not_ask(self, tmp_path, monkeypatch):
        monkeypatch.setattr(neurosynth.sys, "platform", "emscripten")
        table = neurosynth.study_abstracts(pd.Series([1]), tmp_path)
        assert table.empty


def _listing(studies, page_size):
    """NeuroStore's listing of *studies*, by pages of *page_size*."""
    urls = []

    def fetch(url):
        urls.append(url)
        page = int(url.split("page=")[-1].split("&")[0]) - 1
        rows = studies[page * page_size:(page + 1) * page_size]
        return {"metadata": {"total_count": len(studies)}, "results": rows}

    return fetch, urls


STUDIES = [
    {"id": f"s{i}", "pmid": str(100 + i), "doi": f"10.1/{i}",
     "year": 2000 + i, "name": "a study"}
    for i in range(5)
]


class TestNeuroStoreYears:
    def test_the_studies_of_the_release_are_kept(self, tmp_path):
        fetch, urls = _listing(STUDIES, 2)
        table = neurostore.study_identifiers(
            tmp_path, "2026-11", ["s1", "s4"], fetch
        )
        assert _rows(table) == [
            ("s1", "101", "10.1/1", 2001), ("s4", "104", "10.1/4", 2004),
        ]
        assert len(urls) == 3
        again = neurostore.study_identifiers(
            tmp_path, "2026-11", ["s1", "s4"], fetch
        )
        assert len(urls) == 3
        assert _rows(again) == _rows(table)

    def test_offline_nothing_is_kept(self, tmp_path):
        def offline(url):
            raise OSError("offline")

        with pytest.warns(UserWarning, match="NeuroStore could not list"):
            table = neurostore.study_identifiers(
                tmp_path, "2026-11", ["s1"], offline
            )
        assert table.empty
        assert not (tmp_path / "neurostore").exists()
