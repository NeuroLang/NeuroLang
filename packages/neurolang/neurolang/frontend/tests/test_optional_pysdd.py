"""Check NeuroLang runs without pysdd, which has no WebAssembly build."""

import subprocess
import sys
import textwrap

BLOCK_PYSDD = "import sys; sys.modules['pysdd'] = None; "


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", BLOCK_PYSDD + textwrap.dedent(code)],
        capture_output=True, text=True,
    )


def test_neurolang_imports_without_pysdd():
    result = _run("import neurolang.frontend; print('imported')")
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


def test_deterministic_and_lifted_queries_run_without_pysdd():
    result = _run(
        """
        from neurolang.frontend import NeurolangPDL
        nl = NeurolangPDL()
        nl.add_tuple_set([(1, 'a'), (2, 'b')], name='r')
        nl.add_uniform_probabilistic_choice_over_set(
            [(1,), (2,)], name='selected'
        )
        print(sorted(nl.execute_datalog_program('ans(x) :- r(x, "a")')))
        print(sorted(nl.execute_datalog_program(
            'p(t, PROB) :- r(s, t) & selected(s)\\nans(t, p) :- p(t, p)'
        )))
        """
    )
    assert result.returncode == 0, result.stderr
    assert "tuple(x='a')" not in result.stdout
    assert "p=0.5" in result.stdout


def test_weighted_model_counting_without_pysdd_says_what_is_missing():
    result = _run(
        """
        from neurolang.probabilistic import weighted_model_counting as wmc
        try:
            wmc.SemiRingRAPToSDD(3)
        except ImportError as error:
            print(error)
        """
    )
    assert result.returncode == 0, result.stderr
    assert "pysdd" in result.stdout
    assert "weighted model counting" in result.stdout.lower()
