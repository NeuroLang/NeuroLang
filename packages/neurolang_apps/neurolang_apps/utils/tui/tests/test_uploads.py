"""Tests for the files a page uploads to a session (session.Uploads)."""

import pytest

pytest.importorskip("rich")

from neurolang_apps.utils.tui.session import Uploads  # noqa: E402


def test_uploads_are_kept_in_memory_by_default(tmp_path):
    uploads = Uploads()
    uploads.put("a.csv", b"x\n1\n")
    assert uploads.get("a.csv") == b"x\n1\n"
    assert uploads.get("b.csv") is None
    assert list(tmp_path.iterdir()) == []


def test_uploads_in_the_folder_of_a_session(tmp_path):
    """On a server, a session's uploads are files of its folder, which a
    new store (a restart of the session) empties."""
    folder = tmp_path / "uploads" / "7"
    uploads = Uploads(directory=folder)
    uploads.put("my scores.csv", b"a\n1\n")
    assert (folder / "my scores.csv").read_bytes() == b"a\n1\n"
    assert uploads.get("my scores.csv") == b"a\n1\n"
    uploads.put("my scores.csv", b"a\n2\n")  # the same name replaces it
    assert uploads.get("my scores.csv") == b"a\n2\n"
    again = Uploads(directory=folder)
    assert again.get("my scores.csv") is None
    assert not (folder / "my scores.csv").exists()


def test_the_sizes_of_the_uploads_are_bounded(tmp_path):
    uploads = Uploads(directory=tmp_path / "s", most_file=4, most_all=6)
    with pytest.raises(ValueError, match="over"):
        uploads.put("big.csv", b"12345")
    uploads.put("a.csv", b"1234")
    uploads.put("a.csv", b"1234")  # replacing is not adding
    with pytest.raises(ValueError, match="at most"):
        uploads.put("b.csv", b"123")
    assert uploads.get("b.csv") is None
    assert not (tmp_path / "s" / "b.csv").exists()
