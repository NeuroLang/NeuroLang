"""Tests of the access to nilearn's file fetcher."""
import sys

import pytest

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
