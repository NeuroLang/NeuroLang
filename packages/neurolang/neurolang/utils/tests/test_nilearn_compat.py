"""Tests of the access to nilearn's file fetcher."""
import sys
import types

import pytest

from neurolang.utils import nilearn_compat
from neurolang.utils.nilearn_compat import fetch_files


@pytest.fixture
def no_nilearn(monkeypatch):
    """A nilearn that cannot be imported: what needs it fails, whichever of
    its modules another test has imported already."""
    for name in list(sys.modules):
        if name == "nilearn" or name.startswith("nilearn."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "nilearn", None)


def test_files_already_there_are_returned_without_importing_nilearn(
    tmp_path, no_nilearn
):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    files = [
        ("a.txt", "http://example.test/a", {}),
        ("sub/b.txt", "http://example.test/b", {"move": "sub/b.txt"}),
    ]
    assert fetch_files(str(tmp_path), files, verbose=0) == [
        str(tmp_path / "a.txt"), str(tmp_path / "sub" / "b.txt"),
    ]
    assert fetch_files(tmp_path, files) == [
        str(tmp_path / "a.txt"), str(tmp_path / "sub" / "b.txt"),
    ]


@pytest.mark.parametrize("files", [
    [("missing.txt", "http://example.test/m", {})],
    [("a.txt", "http://example.test/a", {"overwrite": True})],
])
def test_a_missing_file_or_one_to_download_again_goes_to_nilearn(
    tmp_path, no_nilearn, files
):
    (tmp_path / "a.txt").write_text("a")
    with pytest.raises(ImportError):
        fetch_files(str(tmp_path), files, verbose=0)


def test_what_nilearn_would_not_return_is_not_returned_here(
    tmp_path, no_nilearn
):
    (tmp_path / "a.txt").write_text("a")
    with pytest.raises(ImportError):  # an option the fast path does not know
        fetch_files(
            str(tmp_path), [("a.txt", "http://example.test/a", {})],
            session=object(),
        )


@pytest.fixture
def fetcher(monkeypatch):
    """nilearn's fetcher, raising the errors in its list ``errors`` before
    it returns; the waits between the tries, recorded and not waited."""
    requests = pytest.importorskip("requests")
    calls, waits = [], []

    def fetch(data_dir, files, **kwargs):
        calls.append(files)
        if fetch.errors:
            raise fetch.errors.pop(0)
        return ["fetched"]

    fetch.errors, fetch.calls, fetch.waits = [], calls, waits
    utils = types.ModuleType("nilearn.datasets.utils")
    utils._fetch_files = fetch
    monkeypatch.setitem(sys.modules, "nilearn", types.ModuleType("nilearn"))
    monkeypatch.setitem(
        sys.modules, "nilearn.datasets", types.ModuleType("nilearn.datasets")
    )
    monkeypatch.setitem(sys.modules, "nilearn.datasets.utils", utils)
    monkeypatch.setattr(nilearn_compat.time, "sleep", waits.append)
    fetch.requests = requests
    return fetch


def http_error(requests, status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status}", response=response)


def test_a_download_failing_for_a_while_is_tried_again(tmp_path, fetcher):
    requests = fetcher.requests
    fetcher.errors = [
        http_error(requests, 504), requests.Timeout("slow"),
        requests.ConnectionError("lost"),
    ]
    files = [("a.txt", "http://example.test/a", {})]
    assert fetch_files(str(tmp_path), files, verbose=0) == ["fetched"]
    assert len(fetcher.calls) == 4
    assert fetcher.waits == list(nilearn_compat.RETRY_WAITS)


def test_a_download_failing_every_time_raises_its_error(tmp_path, fetcher):
    requests = fetcher.requests
    fetcher.errors = [http_error(requests, 503)] * 5
    with pytest.raises(requests.HTTPError):
        fetch_files(str(tmp_path), [("a.txt", "http://e.test/a", {})])
    assert len(fetcher.calls) == len(nilearn_compat.RETRY_WAITS) + 1


@pytest.mark.parametrize("error", [
    lambda requests: http_error(requests, 404),
    lambda requests: requests.exceptions.SSLError("certificate"),
    lambda requests: OSError("Fetching aborted: not provided"),
])
def test_a_download_that_cannot_succeed_is_not_tried_again(
    tmp_path, fetcher, error
):
    fetcher.errors = [error(fetcher.requests)]
    with pytest.raises(OSError):
        fetch_files(str(tmp_path), [("a.txt", "http://e.test/a", {})])
    assert len(fetcher.calls) == 1
    assert fetcher.waits == []
