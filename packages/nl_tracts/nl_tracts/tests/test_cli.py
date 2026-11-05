"""``nl-tracts dl`` writes the Datalog file, ``nl-tracts run`` runs it."""

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from nl_tracts.cli import main

from .conftest import AFFINES, synthetic_volume

LUT = "0 Unknown 0 0 0 0\n1 Left-Putamen 1 1 1 0\n2 Right-Thalamus 1 1 1 0\n"


@pytest.fixture
def dataset(tmp_path):
    nib.save(nib.Nifti1Image(synthetic_volume(), AFFINES["aligned"]),
             tmp_path / "labels.nii.gz")
    (tmp_path / "lut.txt").write_text(LUT)
    rng = np.random.default_rng(0)
    lines = [
        (rng.normal(size=(rng.integers(2, 20), 3)) * 6
         + [-4, -14, -26]).astype(np.float32)
        for _ in range(30)
    ]
    nib.streamlines.save(
        nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4)),
        str(tmp_path / "tracts.tck"),
    )
    (tmp_path / "bundle.qry").write_text(
        "a = 1\nb = 2\nx = a or b\ny = only(a)\n")
    return tmp_path


def dl_args(d, *extra):
    return ["dl", str(d / "bundle.qry"), "--atlas", "fs", "freesurfer",
            str(d / "labels.nii.gz"), str(d / "lut.txt"),
            "--tractogram", "t1", str(d / "tracts.tck"),
            "--space", "s", *extra]


def test_dl_writes_the_file(dataset, capsys):
    out = dataset / "bundle.dl"
    assert main(dl_args(dataset, "-o", str(out))) == 0
    text = out.read_text()
    assert ".load_tract_atlas(" in text and "ans(t, s) :- q_y(t, s)" in text
    assert "bundle.qry" in text  # the title
    assert capsys.readouterr().out == ""


def test_dl_writes_to_stdout_by_default(dataset, capsys):
    assert main(dl_args(dataset)) == 0
    assert capsys.readouterr().out.startswith("% Datalog for the NeuroLang")


@pytest.mark.parametrize(
    "extra,expected",
    [
        (["--ans", "x"], "ans(t, s) :- q_x(t, s)"),
        (["--crossing-threshold", "30"], "crossing_threshold=30.0"),
        (["--length-threshold", "12"], "length_threshold=12.0"),
        (["--no-background"], "emit_background=0"),
        (["--cache", "/tmp/c"], 'cache="/tmp/c"'),
        (["--term-mapping", "freesurfer_aseg_aparc"],
         'term_mapping="freesurfer_aseg_aparc"'),
        (["--space", "other"], 'space="other"'),
        (["--anatomical-vertical"], "ans(t, s)"),
    ],
)
def test_dl_options(dataset, capsys, extra, expected):
    assert main(dl_args(dataset, *extra)) == 0
    assert expected in capsys.readouterr().out


def test_dl_reads_the_program_from_stdin(dataset, capsys, monkeypatch):
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("a = 1\n"))
    args = dl_args(dataset)
    args[1] = "-"
    assert main(args) == 0
    assert "ans(t, s) :- q_a(t, s)" in capsys.readouterr().out


def test_dl_include_folders(dataset, capsys):
    (dataset / "lib").mkdir()
    (dataset / "lib" / "base.qry").write_text("a = 1\n")
    (dataset / "p.qry").write_text("import base.qry\nx = a or 2\n")
    args = dl_args(dataset, "--include", str(dataset / "lib"))
    args[1] = str(dataset / "p.qry")
    assert main(args) == 0


@pytest.mark.parametrize(
    "program,message",
    [("x = nope\n", "nope"), ("x = frobnicate(1)\n", "frobnicate"),
     ("x = (a\n", "syntax")],
)
def test_dl_errors_are_messages_not_tracebacks(dataset, capsys, program,
                                               message):
    (dataset / "bad.qry").write_text(program)
    args = dl_args(dataset)
    args[1] = str(dataset / "bad.qry")
    assert main(args) == 2
    err = capsys.readouterr().err
    assert err.startswith("nl-tracts: ") and message in err


def test_run_prints_the_answer(dataset, capsys):
    out = dataset / "bundle.dl"
    main(dl_args(dataset, "-o", str(out), "--ans", "x"))
    capsys.readouterr()
    assert main(["run", str(out)]) == 0
    captured = capsys.readouterr()
    assert "t1" in captured.out
    assert "rows" in captured.err


def test_run_csv_and_limit(dataset, capsys):
    out = dataset / "bundle.dl"
    main(dl_args(dataset, "-o", str(out), "--ans", "x"))
    capsys.readouterr()
    assert main(["run", str(out), "--format", "csv", "--limit", "3"]) == 0
    lines = capsys.readouterr().out.strip().split("\n")
    assert lines[0] == "t,s" and len(lines) == 4
    assert lines[1].startswith("t1,")


def test_run_reports_a_failure(dataset, capsys):
    (dataset / "bad.dl").write_text(
        '.load_tractogram("t", "/no/file", space="s")\n'
        "ans(t, s) :- streamline(t, s)\n"
    )
    assert main(["run", str(dataset / "bad.dl")]) == 1
    assert "nl-tracts: " in capsys.readouterr().err


def test_the_command_runs_as_a_process(dataset):
    """The whole path in two processes: ``python -m nl_tracts.cli``."""
    out = dataset / "bundle.dl"
    made = subprocess.run(
        [sys.executable, "-m", "nl_tracts.cli",
         *dl_args(dataset, "-o", str(out), "--ans", "x")],
        capture_output=True, text=True,
    )
    assert made.returncode == 0, made.stderr
    ran = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "nl_tracts.cli", "run",
         str(out), "--format", "csv"],
        capture_output=True, text=True,
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.startswith("t,s\nt1,")
