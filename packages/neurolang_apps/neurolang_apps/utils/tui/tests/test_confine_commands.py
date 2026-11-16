"""Every command of a confined session (neurolang.confine), given addresses
and paths outside its data, fetches no address and writes no file outside
its data."""

import io
import socket

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("rich")

from rich.console import Console  # noqa: E402

from nl_expr.expressions import Command, Symbol  # noqa: E402
from neurolang.confine import CONFINE_ENV  # noqa: E402
from neurolang.utils import nilearn_compat  # noqa: E402
from neurolang_apps.utils.engines.tracts.tests.engine import (  # noqa: E402
    TractsSolver,
    tracts_engine,
)
from neurolang_apps.utils.tui import InteractiveTuiApp, TuiSettings  # noqa
from neurolang_apps.utils.tui.commands import COMMANDS, QuitRepl  # noqa

URL = "http://127.0.0.1:9/probe"

# The statements of each Datalog command of the engines, ``{url}`` an
# address, ``{out}`` a directory outside the data and ``{data}`` the data.
DATALOG = {
    "load_csv": ['.load_csv(s, "{url}")', '.load_csv(s, "{out}/t.csv")'],
    "load_atlas": [
        '.load_atlas(a, "{url}/a.nii", "{url}/l.csv")',
        '.load_atlas(a, "{data}/a.nii", "{data}/l.csv", "{url}/a.tgz")',
        '.load_atlas(a, "{out}/a.nii", "{out}/l.csv", kind="tsv")',
        '.load_atlas(a, kind="nilearn", fetcher="craddock_2012", '
        'url="{url}")',
        '.load_atlas(a, kind="nilearn", fetcher="msdl")',
        '.load_atlas(a, kind="destrieux")',
        '.load_atlas(a, kind="schaefer", data_dir="{out}")',
        '.load_atlas(a, kind="tian")',
    ],
    "import": ['.import("{url}")', '.import("{out}/m.dl")'],
    "document": ['.document(p, "{url}")'],
    "load_tractogram": [
        '.load_tractogram(t, "{out}/t.trk")',
        '.load_tractogram(t, kind="hcp842", url="{url}")',
        '.load_tractogram(t, kind="hcp842", data_dir="{out}")',
        '.load_tractogram(t, kind="hcp842", bundle="UF_L")',
    ],
    "index_tracts": [
        '.index_tracts(t, a, cache="{out}/c")',
        '.index_tracts(t, a, term_mapping="{url}/m.csv")',
    ],
}

# The dot-commands not run: they leave the REPL, or ask the LLM of the
# server, at its own address.
NOT_RUN = {".quit", ".exit", ".ask", ".llm", ".explain", ".refine", ".rate"}


def datalog_commands():
    """The names of the Datalog commands of the engines."""
    return {
        p.pattern.name.name
        for cls in TractsSolver.mro() for p in getattr(cls, "__patterns__", ())
        if isinstance(p.pattern, Command)
        and isinstance(p.pattern.name, Symbol)
    }


def statements():
    for name in sorted(datalog_commands()):
        for statement in DATALOG.get(name, [f"no statement of .{name}"]):
            yield statement
    for name in sorted(set(COMMANDS) - NOT_RUN):
        yield f"{name} {{url}}"
        yield f"{name} {{out}}/x.csv"
        yield f"{name} {{out}}/x.nii.gz"


@pytest.fixture
def confined(tmp_path, monkeypatch):
    """The data of the session, every address refused (and recorded), its
    home, nilearn's directory and the working directory outside it."""
    data, out, home = (tmp_path / d for d in ("data", "out", "home"))
    for d in (data, out, home):
        d.mkdir()
    monkeypatch.setenv(CONFINE_ENV, str(data))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NILEARN_DATA", str(home / "nilearn_data"))
    monkeypatch.chdir(home)
    monkeypatch.setattr(nilearn_compat, "RETRY_WAITS", ())
    fetched = []

    def refuse(*args, **kwargs):
        fetched.append(args)
        raise OSError("no network in this test")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    return tmp_path, fetched


@pytest.mark.parametrize("statement", list(statements()))
def test_a_confined_command_fetches_and_writes_nothing(confined, statement):
    tmp_path, fetched = confined
    assert not statement.startswith("no statement"), (
        f"give DATALOG the statements of {statement.split('.')[-1]}")
    console = Console(file=io.StringIO(), width=200)
    engine = tracts_engine()
    engine.add_tuple_set([(1,)], name="one")
    app = InteractiveTuiApp(engine, TuiSettings(), console=console,
                            err_console=console)
    app._execute_and_display("ans(x) :- one(x)")  # a result to .save
    assert "error" not in console.file.getvalue().lower()
    before = sorted(p for p in tmp_path.rglob("*")
                    if not p.is_relative_to(tmp_path / "data"))
    try:
        app._handle_dot_command(
            statement.format(url=URL, out=tmp_path / "out",
                             data=tmp_path / "data"))
    except QuitRepl:
        pass
    after = sorted(p for p in tmp_path.rglob("*")
                   if not p.is_relative_to(tmp_path / "data"))
    assert "parse error" not in console.file.getvalue()
    assert fetched == [], console.file.getvalue()
    assert after == before, console.file.getvalue()
