"""Tests for the script that builds the bundle of the REPL in the browser."""

import importlib.util
import os
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")

SCRIPT = Path(__file__).resolve().parents[6] / "doc" / "make_repl_bundle.py"


@pytest.fixture
def bundle():
    spec = importlib.util.spec_from_file_location("make_repl_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_wheels_leave_no_gitignore_that_would_hide_the_bundle(
    bundle, monkeypatch, tmp_path
):
    """
    uv writes a .gitignore of ``*`` in the directory it builds in, and the
    documentation is deployed by committing that directory: the page and its
    bundle would not be committed. The wheels are those of all the packages,
    the engine first, as the REPL installs them in that order.
    """
    def uv_build(command):
        assert "--all-packages" in command
        out = Path(command[command.index("--out-dir") + 1])
        for name in ("neurolang_apps", "neurolang", "nl_core", "nl_expr"):
            (out / f"{name}-1-py3-none-any.whl").write_bytes(b"wheel")
        (out / ".gitignore").write_text("*")

    monkeypatch.setattr(bundle.subprocess, "check_call", uv_build)
    (tmp_path / "page.html").write_text("<html>")
    assert bundle.build_wheels(tmp_path) == [
        "nl_expr-1-py3-none-any.whl",
        "nl_core-1-py3-none-any.whl",
        "neurolang-1-py3-none-any.whl",
        "neurolang_apps-1-py3-none-any.whl",
    ]
    assert not (tmp_path / ".gitignore").exists()
    assert (tmp_path / "page.html").exists()


def test_a_certificate_error_is_retried_with_the_system_certificates(
    bundle, monkeypatch, tmp_path
):
    from neurolang_apps.utils import engine_registry

    system = tmp_path / "ca.pem"
    system.write_text("certificates")
    monkeypatch.setattr(bundle, "SYSTEM_CERTIFICATES", (system,))
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    calls = []

    def build(name, data_dir, resolution):
        calls.append(os.environ.get("REQUESTS_CA_BUNDLE"))
        if len(calls) == 1:
            raise requests.exceptions.SSLError("certificate has expired")

    monkeypatch.setattr(engine_registry, "build_engine", build)
    bundle.build_engine(tmp_path)
    assert calls == [None, str(system)]
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)


def test_without_system_certificates_the_error_is_raised(
    bundle, monkeypatch, tmp_path
):
    from neurolang_apps.utils import engine_registry

    monkeypatch.setattr(bundle, "SYSTEM_CERTIFICATES", (tmp_path / "none",))

    def build(name, data_dir, resolution):
        raise requests.exceptions.SSLError("certificate has expired")

    monkeypatch.setattr(engine_registry, "build_engine", build)
    with pytest.raises(requests.exceptions.SSLError):
        bundle.build_engine(tmp_path)
