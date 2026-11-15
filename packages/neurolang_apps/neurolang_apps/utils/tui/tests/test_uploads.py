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


def test_a_file_erased_by_an_admin_is_gone_for_the_session(tmp_path):
    """The admin console erases the files of a folder (Sessions.
    erase_uploads): the session's uploads are what the folder holds."""
    folder = tmp_path / "3"
    uploads = Uploads(directory=folder, most_file=4, most_all=6)
    uploads.put("a.csv", b"1234")
    (folder / "a.csv").unlink()
    assert uploads.get("a.csv") is None
    uploads.put("b.csv", b"1234")  # a.csv no longer counts


def test_the_uploads_tell_what_they_hold(tmp_path):
    """Each file: its size, when it was uploaded, and the relations loaded
    from it; in the memory of the session or in its folder."""
    for directory in (None, tmp_path / "9"):
        uploads = Uploads(directory=directory)
        uploads.put("pets.csv", b"pet\ncat\n")
        uploads.put("roi.nii.gz", b"x" * 10)
        uploads.loaded("pets.csv", "pets")
        uploads.loaded("pets.csv", "animals")
        uploads.loaded("pets.csv", "pets")
        listing = uploads.listing()
        assert [(f["name"], f["size"], f["loaded"]) for f in listing] == [
            ("pets.csv", 8, ["animals", "pets"]), ("roi.nii.gz", 10, [])]
        assert all(f["age"] >= 0 for f in listing)
        assert uploads.used() == 18


def test_the_session_lists_its_uploads(tmp_path):
    """.uploads lists the files the page uploaded in this session, what
    each was loaded as, and the room left."""
    import io

    from rich.console import Console

    from neurolang.frontend import NeurolangPDL
    from neurolang_apps.utils.tui import InteractiveTuiApp, TuiSettings
    from neurolang_apps.utils.tui.session import store_upload

    console = Console(file=io.StringIO(), width=200)
    app = InteractiveTuiApp(NeurolangPDL(), TuiSettings(), console=console,
                            err_console=console)
    app._handle_dot_command(".uploads")
    assert "No file was uploaded in this session" in console.file.getvalue()
    store_upload(app, "my scores.csv", b"study,score\ns1,0.5\n")
    app._handle_dot_command(".load 'my scores.csv' scores")
    app._handle_dot_command(".uploads")
    out = " ".join(console.file.getvalue().split())
    assert "my scores.csv" in out and "table" in out and "scores" in out
    assert "of 200 MB" in out


def test_a_file_is_removed_from_the_uploads(tmp_path):
    for directory in (None, tmp_path / "5"):
        uploads = Uploads(directory=directory)
        uploads.put("a.csv", b"x\n1\n")
        uploads.loaded("a.csv", "a")
        assert uploads.remove("a.csv") is True
        assert uploads.get("a.csv") is None and "a.csv" not in uploads
        assert uploads.listing() == [] and uploads.used() == 0
        assert uploads.remove("a.csv") is False
    assert [p for p in (tmp_path / "5").iterdir() if p.is_file()] == []


def test_the_user_removes_an_uploaded_file(tmp_path):
    """.uploads remove NAME deletes the file from the session (from its
    folder, on a server); the relations loaded from it stay, which .drop
    removes."""
    import io

    from rich.console import Console

    from neurolang.frontend import NeurolangPDL
    from neurolang_apps.utils.tui import InteractiveTuiApp, TuiSettings
    from neurolang_apps.utils.tui.session import store_upload

    console = Console(file=io.StringIO(), width=200)
    app = InteractiveTuiApp(NeurolangPDL(), TuiSettings(), console=console,
                            err_console=console)
    app.uploads = Uploads(directory=tmp_path / "s")
    store_upload(app, "my scores.csv", b"study,score\ns1,0.5\n")
    app._handle_dot_command(".load 'my scores.csv' scores")
    app._handle_dot_command(".uploads remove 'my scores.csv'")
    out = " ".join(console.file.getvalue().split())
    assert "removed my scores.csv" in out
    assert "scores, loaded from it, stays" in out and ".drop scores" in out
    assert app.uploads.listing() == []
    assert not (tmp_path / "s" / "my scores.csv").exists()
    assert "scores" in app.loaded_relations
    app._handle_dot_command(".uploads remove other.csv")
    assert "No file other.csv was uploaded" in console.file.getvalue()
    app._handle_dot_command(".uploads drop x")
    assert "Usage: .uploads [remove NAME | restore NAME]" in \
        console.file.getvalue()


def test_the_last_removed_file_is_restored(tmp_path):
    """The page's Undo after Remove: the session keeps the last file
    removed, which .uploads restore NAME brings back with what it was
    loaded as, until another file is removed or uploaded."""
    for directory in (None, tmp_path / "6"):
        uploads = Uploads(directory=directory)
        uploads.put("a.csv", b"x\n1\n")
        uploads.loaded("a.csv", "a")
        uploads.remove("a.csv")
        assert uploads.listing() == [] and uploads.used() == 0
        assert uploads.restore("b.csv") is False
        assert uploads.restore("a.csv") is True
        assert [(f["name"], f["loaded"]) for f in uploads.listing()] == [
            ("a.csv", ["a"])]
        assert uploads.get("a.csv") == b"x\n1\n"
        assert uploads.restore("a.csv") is False  # once
        uploads.remove("a.csv")
        uploads.put("c.csv", b"y\n")  # another change: no undo of a.csv
        assert uploads.restore("a.csv") is False
    assert sorted(p.name for p in (tmp_path / "6").iterdir()
                  if p.is_file()) == ["c.csv"]


def test_the_user_restores_a_removed_file(tmp_path):
    import io

    from rich.console import Console

    from neurolang.frontend import NeurolangPDL
    from neurolang_apps.utils.tui import InteractiveTuiApp, TuiSettings
    from neurolang_apps.utils.tui.session import store_upload

    console = Console(file=io.StringIO(), width=200)
    app = InteractiveTuiApp(NeurolangPDL(), TuiSettings(), console=console,
                            err_console=console)
    store_upload(app, "s.csv", b"study,score\ns1,0.5\n")
    app._handle_dot_command(".load s.csv scores")
    app._handle_dot_command(".uploads remove s.csv")
    app._handle_dot_command(".uploads restore s.csv")
    assert "restored s.csv" in console.file.getvalue()
    assert app.uploads.listing()[0]["loaded"] == ["scores"]
    app._handle_dot_command(".uploads restore s.csv")
    assert "s.csv cannot be restored" in console.file.getvalue()
