"""Tests of the results every session would find the same, kept for all."""

import os

import pandas as pd
import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from neurolang.frontend import NeurolangPDL  # noqa: E402
from neurolang_apps.utils import result_cache  # noqa: E402
from neurolang_apps.utils.tui import app as app_module  # noqa: E402
from neurolang_apps.utils.tui.local_backend import LocalBackend  # noqa: E402

KEPT = "kept from an earlier run"


def _engine(name, say):
    """An engine whose rule adult reads its relation person."""
    nl = NeurolangPDL()
    nl.add_tuple_set([("alice", 3), ("bob", 5)], name="person")
    nl.execute_datalog_program("adult(x) :- person(x, n) & (n > 4)")
    return nl


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "results"


@pytest.fixture
def user(tmp_path, cache_dir):
    """Start a session of a user, as a fork of the server's builder would:
    each its own engine, all the same, with the server's results."""
    def start(most_mb=1):
        backend = LocalBackend(
            tmp_path / "data", build=_engine,
            result_cache={"directory": str(cache_dir), "most_mb": most_mb},
        )
        backend.start("neurostore")  # no metadata fetched
        return backend
    return start


@pytest.fixture
def runs(monkeypatch):
    """Count the programs the sessions run."""
    ran = []
    execute = app_module._execute_program

    def counting(nl, program, **flags):
        ran.append(program)
        return execute(nl, program, **flags)

    monkeypatch.setattr(app_module, "_execute_program", counting)
    return ran


def _rows(reply):
    return [row for table in reply["tables"] for row in table["rows"]]


def _files(cache_dir):
    return sorted(cache_dir.glob("*.arrow")) if cache_dir.exists() else []


def test_a_query_reading_the_engine_runs_once_for_every_session(
        user, runs, cache_dir):
    ann, bob = user(), user()
    first = ann.submit("ans(x, n) :- person(x, n) & (n > 1)")
    assert KEPT not in first["html"] and len(_files(cache_dir)) == 1
    # Another user, spaces and comments apart: not run again.
    again = bob.submit("% people\nans(x, n) :-\n  person(x, n) & (n > 1)")
    assert KEPT in again["html"] and not again["failed"]
    assert again["tables"][0]["rows"] == first["tables"][0]["rows"]
    assert len(runs) == 1
    # Its rows are those of the page: downloads, viewers.
    assert b"alice" in bob.download(0, "csv").data


def test_the_kept_frame_is_the_one_computed(user, cache_dir):
    ann, bob = user(), user()
    ann.submit("ans(x, n) :- person(x, n)")
    bob.submit("ans(x, n) :- person(x, n)")
    computed, kept = (b.repl.app.results[0].frame for b in (ann, bob))
    pd.testing.assert_frame_equal(
        computed.reset_index(drop=True), kept, check_exact=True)


def test_session_rules_of_the_same_name_are_never_shared(
        user, runs, cache_dir):
    ann, bob = user(), user()
    ann.submit("big(x) :- person(x, n) & (n > 4)")
    bob.submit("big(x) :- person(x, n) & (n < 4)")
    assert _rows(ann.submit("ans(x) :- big(x)")) == [["bob"]]
    assert _rows(bob.submit("ans(x) :- big(x)")) == [["alice"]]
    assert _files(cache_dir) == [] and len(runs) == 4


def test_uploads_of_the_same_name_are_never_shared(user, cache_dir):
    ann, bob = user(), user()
    for backend, pet in ((ann, b"cat"), (bob, b"dog")):
        backend.upload("pets.csv", b"pet\n" + pet + b"\n")
        backend.submit(".load pets.csv")
    assert _rows(ann.submit("ans(x) :- pets(x)")) == [["cat"]]
    assert _rows(bob.submit("ans(x) :- pets(x)")) == [["dog"]]
    assert _files(cache_dir) == []


def test_a_fact_read_through_an_engine_rule_is_the_sessions(user, cache_dir):
    ann, bob = user(), user()
    ann.submit("person('carol', 9).")
    assert sorted(_rows(ann.submit("ans(x) :- adult(x)"))) == [
        ["bob"], ["carol"]]
    assert _files(cache_dir) == []
    reply = bob.submit("ans(x) :- adult(x)")
    assert _rows(reply) == [["bob"]] and KEPT not in reply["html"]
    # Bob's, which read the engine only, is kept: Ann does not read it.
    assert len(_files(cache_dir)) == 1
    assert KEPT not in ann.submit("ans(x) :- adult(x)")["html"]


def test_a_session_cleared_shares_again(user, cache_dir):
    ann = user()
    ann.submit("person('carol', 9).")
    ann.submit("ans(x) :- adult(x)")
    assert _files(cache_dir) == []
    ann.submit(".clear")
    assert _rows(ann.submit("ans(x) :- adult(x)")) == [["bob"]]
    assert len(_files(cache_dir)) == 1


def test_rules_the_session_defined_but_the_query_does_not_read(user, runs):
    ann, bob = user(), user()
    ann.submit("big(x) :- person(x, n) & (n > 4)")
    ann.submit("ans(x) :- adult(x)")
    assert KEPT in bob.submit("ans(x) :- adult(x)")["html"]
    assert len(runs) == 2


def test_after_a_command_the_engine_shares_nothing(user, cache_dir):
    ann, bob = user(), user()
    bob.submit("ans(x) :- adult(x)")
    ann.submit('.load_csv(t, "nowhere.csv")')  # fails, may have changed
    assert KEPT not in ann.submit("ans(x) :- adult(x)")["html"]
    ann.submit("ans(x) :- person(x, n)")
    assert len(_files(cache_dir)) == 1


def test_a_query_whose_rules_stay_runs(user, runs, cache_dir):
    ann, bob = user(), user()
    ann.submit("ans(x) :- adult(x)")
    bob.submit(".scope off")
    assert KEPT not in bob.submit("ans(x) :- adult(x)")["html"]
    assert len(runs) == 2


def test_the_notes_of_a_run_are_those_of_the_result_kept(user, monkeypatch):
    ann, bob = user(), user()
    monkeypatch.setattr(
        app_module, "constant_notes", lambda *args: ["a note on it"])
    assert "a note on it" in ann.submit("ans(x) :- adult(x)")["html"]
    monkeypatch.setattr(app_module, "constant_notes", lambda *args: [])
    reply = bob.submit("ans(x) :- adult(x)")
    assert KEPT in reply["html"] and "a note on it" in reply["html"]
    assert bob.repl.app.last_notes == ann.repl.app.last_notes


def test_constants_of_other_types_are_other_programs():
    nl = _engine("", None)
    cache = result_cache.ResultCache(nl, "e", "unused", 1)
    keys = {cache.key(nl, f"ans(x) :- person(x, {value})")
            for value in ("1", "'1'", "1.0")}
    assert len(keys) == 3


def test_the_fresh_variables_of_a_program_do_not_change_its_key():
    nl = _engine("", None)
    cache = result_cache.ResultCache(nl, "e", "unused", 1)
    program = "ans(x) :- person(x, _) & ~person(x, 3)"
    assert cache.key(nl, program) == cache.key(nl, program)
    assert cache.key(nl, program) != cache.key(nl, program, "polars")


def test_the_engine_digest_changes_with_its_data(tmp_path):
    (tmp_path / "studies.tsv").write_text("1\n")
    (tmp_path / "engine_cache").mkdir()
    before = result_cache.engine_digest("neurosynth", 2, tmp_path)
    (tmp_path / "engine_cache" / "atlas.npz").write_text("kept")
    assert result_cache.engine_digest("neurosynth", 2, tmp_path) == before
    assert result_cache.engine_digest("neurosynth", 3, tmp_path) != before
    assert result_cache.engine_digest("neurostore", 2, tmp_path) != before
    (tmp_path / "studies.tsv").write_text("1\n2\n")
    assert result_cache.engine_digest("neurosynth", 2, tmp_path) != before


def test_the_least_recently_used_go_first(tmp_path):
    nl = _engine("", None)
    cache = result_cache.ResultCache(nl, "e", tmp_path, 1)
    frame = pd.DataFrame({"x": range(40000)})  # 320 kB in Arrow
    for key, seconds in (("a", 1), ("b", 2), ("c", 3)):
        assert cache.put(key, frame)
        os.utime(cache._path(key), (seconds, seconds))
    assert cache.get("a") is not None  # now the most recently used
    assert cache.put("d", frame)
    assert {p.stem for p in tmp_path.glob("*.arrow")} == {"a", "c", "d"}
    assert not cache.put("e", pd.DataFrame({"x": range(200000)}))


def test_rows_arrow_does_not_hold_as_they_are_are_not_kept(tmp_path):
    nl = _engine("", None)
    cache = result_cache.ResultCache(nl, "e", tmp_path, 1)
    assert not cache.put("t", pd.DataFrame({"x": [(1, 2), (3, 4)]}))
    assert not cache.put("m", pd.DataFrame({"x": [1, "a"]}))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "fork"), reason="sessions fork")
def test_the_sessions_of_a_server_share_their_results(tmp_path):
    from neurolang_apps.utils.tui.sessions import Sessions
    from neurolang_apps.utils.tui.tests.test_sessions import build

    directory = tmp_path / "results"
    sessions = Sessions(tmp_path, build=build, result_cache={
        "directory": str(directory), "most_mb": 1})
    try:
        ann, bob = sessions.session("ann", "1"), sessions.session("bob", "1")
        for session in (ann, bob):
            session.start("neurostore")
        ann.submit("old(x) :- person(x, a), (a > 4)")
        assert _rows(ann.submit("ans(x) :- old(x)")) == [["bob"]]
        assert not directory.exists() or not any(directory.iterdir())
        assert _rows(bob.submit("ans(x) :- old(x)")) == []  # no rule old
        assert len(list(directory.iterdir())) == 1  # Bob's, of the engine
        again = ann.submit("ans(x) :- old(x)")
        assert _rows(again) == [["bob"]] and KEPT not in again["html"]
        ann.submit("ans(x) :- person(x, a)")
        assert KEPT in bob.submit("ans(x) :- person(x, a)")["html"]
    finally:
        sessions.close()


def _engine_with_constraint(name, say):
    """An engine whose constraint makes every vip noted."""
    nl = NeurolangPDL()
    nl.add_tuple_set([("alice",)], name="noted")
    nl.add_tuple_set([("zed",)], name="vip")
    with nl.environment as e:
        nl.add_constraint(e.vip[e.x], e.noted[e.x])
    return nl


def test_a_fact_read_through_an_engine_constraint_is_the_sessions(
        tmp_path, cache_dir):
    def start():
        backend = LocalBackend(
            tmp_path / "data", build=_engine_with_constraint,
            result_cache={"directory": str(cache_dir), "most_mb": 1},
        )
        backend.start("neurostore")
        return backend

    ann, bob = start(), start()
    ann.submit("vip('carol').")
    assert sorted(_rows(ann.submit("ans(x) :- noted(x)"))) == [
        ["alice"], ["carol"], ["zed"]]
    assert _files(cache_dir) == []
    reply = bob.submit("ans(x) :- noted(x)")
    assert sorted(_rows(reply)) == [["alice"], ["zed"]]
    assert KEPT not in reply["html"] and len(_files(cache_dir)) == 1
    assert KEPT not in ann.submit("ans(x) :- noted(x)")["html"]


def test_a_constraint_of_the_session_is_read_by_every_query():
    nl = _engine_with_constraint("", None)
    cache = result_cache.ResultCache(nl, "e", "unused", 1)
    program = "ans(x) :- person(x, n)"
    assert cache.key(nl, program)
    nl.program_ir.push_scope()  # the session's
    with nl.environment as e:
        nl.add_constraint(e.person[e.x, e.n], e.noted[e.x])
    assert cache.key(nl, program) is None


def _image(values) -> bytes:
    import gzip

    import nibabel as nib
    import numpy as np

    data = np.zeros((4, 4, 4), dtype=np.float32)
    for voxel, value in values.items():
        data[voxel] = value
    return gzip.compress(nib.Nifti1Image(data, np.eye(4)).to_bytes())


def _engine_with_template(name, say):
    import nibabel as nib
    import numpy as np

    nl = _engine(name, say)
    nl.template = nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), np.eye(4))
    return nl


def test_images_uploaded_of_the_same_name_are_never_shared(
        tmp_path, cache_dir):
    users = []
    for voxel in ((1, 2, 3), (0, 0, 0)):
        backend = LocalBackend(
            tmp_path / "data", build=_engine_with_template,
            result_cache={"directory": str(cache_dir), "most_mb": 1},
        )
        backend.start("neurostore")
        backend.upload("z.nii.gz", _image({voxel: 5.0}))
        assert not backend.submit(".image z.nii.gz --name img")["failed"]
        users.append(backend)
    ann, bob = users
    for relation in ("img(i, j, k, v)", "img_probability(i, j, k, v)"):
        query = f"ans(i, j, k) :- {relation}"
        assert _rows(ann.submit(query)) == [[1, 2, 3]]
        reply = bob.submit(query)
        assert _rows(reply) == [[0, 0, 0]] and KEPT not in reply["html"]
    assert _files(cache_dir) == []
