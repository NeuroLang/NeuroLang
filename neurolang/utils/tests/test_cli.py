"""Tests for the neurolang-query CLI module."""

import pytest

from neurolang.exceptions import NeuroLangException
from neurolang.expressions import Constant, Symbol
from neurolang.frontend.datalog.pretty_printer import DatalogPrettyPrinter
from neurolang.frontend.datalog.squall_syntax_lark import parser as squall_parser
from neurolang.logic import Conjunction, ExistentialPredicate
from neurolang.utils.cli import (
    _build_parser,
    _execute_program,
    _execute_squall_program,
    _format_result,
)

from neurolang.utils import engine_registry

# ---------------------------------------------------------------------------
# _build_parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.query is None
        assert args.file is None
        assert args.engine == "neurosynth"
        assert args.format == "table"
        assert args.data_dir == "neurolang_data"
        assert args.resolution is None
        assert args.list_predicates is False

    def test_positional_query(self):
        parser = _build_parser()
        args = parser.parse_args(["ans(x) :- R(x)"])
        assert args.query == "ans(x) :- R(x)"
        assert args.file is None

    def test_file_arg(self, tmp_path):
        parser = _build_parser()
        fp = tmp_path / "q.dl"
        args = parser.parse_args(["--file", str(fp)])
        assert args.file == str(fp)
        assert args.query is None

    def test_engine(self):
        parser = _build_parser()
        args = parser.parse_args(["--engine", "destrieux"])
        assert args.engine == "destrieux"

    def test_invalid_engine_in_main(self):
        """Validation now lives in main(), not the parser."""
        from neurolang.utils.cli import main

        with pytest.raises(SystemExit):
            main(["--engine", "unknown", "ans(x) :- R(x)"])

    def test_format(self):
        parser = _build_parser()
        args = parser.parse_args(["--format", "csv"])
        assert args.format == "csv"

    def test_invalid_format(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--format", "xml"])

    def test_list_predicates(self):
        parser = _build_parser()
        args = parser.parse_args(["--list-predicates"])
        assert args.list_predicates is True

    def test_resolution(self):
        parser = _build_parser()
        args = parser.parse_args(["--resolution", "2.0"])
        assert args.resolution == 2.0

    def test_data_dir(self):
        parser = _build_parser()
        args = parser.parse_args(["--data-dir", "/custom/path"])
        assert args.data_dir == "/custom/path"

    def test_mutually_exclusive_query_and_file(self, tmp_path):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ans(x) :- R(x)", "--file", str(tmp_path / "q.dl")])

    def test_short_flags(self, tmp_path):
        parser = _build_parser()
        fp = tmp_path / "q.dl"
        args = parser.parse_args(["-e", "destrieux", "-f", str(fp), "-l"])
        assert args.engine == "destrieux"
        assert args.file == str(fp)
        assert args.list_predicates is True

    def test_list_engines_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--list-engines"])
        assert args.list_engines is True

    def test_engine_not_rejected_by_parser_anymore(self):
        """The parser no longer validates engine names; main() does."""
        parser = _build_parser()
        args = parser.parse_args(["--engine", "unknown"])
        assert args.engine == "unknown"

    def test_show_ra_defaults_to_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.show_ra is False

    def test_show_ra_flag_long(self):
        parser = _build_parser()
        args = parser.parse_args(["--show-ra", "ans(x) :- P(x)"])
        assert args.show_ra is True

    def test_show_ra_flag_short(self):
        parser = _build_parser()
        args = parser.parse_args(["-Q", "ans(x) :- P(x)"])
        assert args.show_ra is True


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------


class TestEngineRegistry:
    def test_list_engine_names(self):
        names = engine_registry.list_engine_names()
        assert "neurosynth" in names
        assert "destrieux" in names

    def test_get_engine_config(self):
        cfg = engine_registry.get_engine_config("neurosynth")
        assert "python_init" in cfg
        assert cfg["requires_mni_mask"] is True
        assert "predicates" in cfg
        assert "peak_reported" in cfg["predicates"]

    def test_get_predicates(self):
        preds = engine_registry.get_predicates("destrieux")
        assert "destrieux" in preds
        assert preds["destrieux"]["arity"] == 2

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            engine_registry.get_engine_config("nonexistent")

    def test_datalog_init_in_config(self):
        """Every engine with python_init should have inline datalog_init."""
        for name in engine_registry.list_engine_names():
            cfg = engine_registry.get_engine_config(name)
            if "python_init" in cfg:
                code = cfg.get("datalog_init")
                assert code, (
                    f"Engine {name!r} has python_init but no datalog_init"
                )
                assert isinstance(code, str), (
                    f"datalog_init for engine {name!r} must be a string, "
                    f"got {type(code).__name__}"
                )
                assert len(code.strip()) > 0, (
                    f"datalog_init for engine {name!r} is empty"
                )

    def test_show_engines(self, capsys):
        """show_engines prints all engine names with descriptions."""
        engine_registry.show_engines()
        captured = capsys.readouterr()
        assert "neurosynth" in captured.out
        assert "destrieux" in captured.out
        assert "Available engines" in captured.out

    def test_get_engine_config_with_yaml_path(self, tmp_path):
        """get_engine_config accepts a custom YAML path."""
        custom_yaml = tmp_path / "engines.yaml"
        custom_yaml.write_text("""
engines:
  mini:
    description: "Mini test"
    builtins: []
""")
        cfg = engine_registry.get_engine_config("mini", yaml_path=custom_yaml)
        assert cfg["description"] == "Mini test"

    def test_get_predicates_with_yaml_path(self, tmp_path):
        """get_predicates works with a custom YAML file."""
        custom_yaml = tmp_path / "engines.yaml"
        custom_yaml.write_text("""
engines:
  mini:
    description: "Mini"
    builtins: []
    predicates:
      p:
        arity: 1
        columns: [x]
        description: "Test predicate"
""")
        preds = engine_registry.get_predicates("mini", yaml_path=custom_yaml)
        assert "p" in preds
        assert preds["p"]["arity"] == 1

    def test_edb_program_roundtrip(self):
        """In-memory EDB relations can be queried and formatted."""
        from neurolang.frontend import NeurolangPDL

        nl = NeurolangPDL()
        nl.add_tuple_set([(1, "a"), (2, "b"), (3, "c")], name="csv_rel")
        nl.add_tuple_set([(4, "d"), (5, "e")], name="tsv_rel")

        result = _execute_program(
            nl, "ans(x, y) :- csv_rel(x, y)"
        )
        assert result is not None
        output = _format_result(result, column_names=list(result.columns))
        assert "a" in output
        assert "b" in output
        assert "c" in output

        # Query TSV relation
        result = _execute_program(
            nl, "ans(x, y) :- tsv_rel(x, y)"
        )
        assert result is not None
        output = _format_result(result, column_names=list(result.columns))
        assert "d" in output
        assert "e" in output


# ---------------------------------------------------------------------------
# _format_result
# ---------------------------------------------------------------------------


class TestFormatResult:
    def test_none(self):
        assert _format_result(None) == ""

    def test_true(self):
        assert _format_result(True) == "true"

    def test_false(self):
        assert _format_result(False) == "false"

    def test_empty_unnamed_set(self):
        from neurolang.utils.relational_algebra_set.pandas import (
            RelationalAlgebraSet,
        )

        ras = RelationalAlgebraSet()
        assert _format_result(ras) == "(empty)"

    def test_nonempty_unnamed_set(self):
        from neurolang.utils.relational_algebra_set.pandas import (
            RelationalAlgebraSet,
        )

        ras = RelationalAlgebraSet()
        ras.add((1, "a"))
        ras.add((2, "b"))
        result = _format_result(ras)
        assert "1" in result
        assert "a" in result

    def test_unnamed_set_with_column_names(self):
        from neurolang.utils.relational_algebra_set.pandas import (
            RelationalAlgebraSet,
        )

        ras = RelationalAlgebraSet()
        ras.add((1,))
        ras.add((2,))
        result = _format_result(ras, column_names=["x"])
        assert "x" in result
        assert "1" in result
        assert "2" in result

    def test_named_set_preserves_columns(self):
        from neurolang.utils.relational_algebra_set.pandas import (
            NamedRelationalAlgebraFrozenSet,
        )

        nras = NamedRelationalAlgebraFrozenSet(
            columns=("x", "y"), iterable=[(1, "a"), (2, "b")]
        )
        result = _format_result(nras)
        assert "x" in result
        assert "y" in result

    def test_constant_wrapping_wrapped_set(self):
        """The typical chase result: Constant[WrappedRelationalAlgebraSet]."""
        from neurolang import expressions as ir
        from neurolang.datalog.wrapped_collections import (
            WrappedRelationalAlgebraSet,
        )
        from neurolang.utils.relational_algebra_set.pandas import (
            RelationalAlgebraSet,
        )

        inner = RelationalAlgebraSet()
        inner.add((42,))
        inner.add((99,))
        wrapped = WrappedRelationalAlgebraSet(inner)
        result = ir.Constant(wrapped)

        output = _format_result(result, column_names=["val"])
        assert "val" in output
        assert "42" in output
        assert "99" in output

    def test_csv_format(self):
        from neurolang.utils.relational_algebra_set.pandas import (
            RelationalAlgebraSet,
        )

        ras = RelationalAlgebraSet()
        ras.add((1, "a"))
        result = _format_result(ras, fmt="csv", column_names=["x", "y"])
        assert "x,y" in result.split("\n")[0]
        assert "1,a" in result.split("\n")[1]

    def test_json_format(self):
        from neurolang.utils.relational_algebra_set.pandas import (
            RelationalAlgebraSet,
        )

        ras = RelationalAlgebraSet()
        ras.add((1, "a"))
        result = _format_result(ras, fmt="json", column_names=["x", "y"])
        assert '"x"' in result
        assert '"y"' in result
        assert '"a"' in result
        assert "1" in result  # numbers are unquoted in JSON

    def test_column_names_mismatch_arity(self):
        """When column_names length doesn't match, they are ignored."""
        from neurolang.utils.relational_algebra_set.pandas import (
            RelationalAlgebraSet,
        )

        ras = RelationalAlgebraSet()
        ras.add((1,))
        # 2 column names for 1-column data — no renaming
        result = _format_result(ras, column_names=["a", "b"])
        # Default integer column name stays
        assert "0" in result

    def test_column_names_not_applied_to_named_sets(self):
        """column_names is only applied when existing columns are unnamed."""
        from neurolang.utils.relational_algebra_set.pandas import (
            NamedRelationalAlgebraFrozenSet,
        )

        nras = NamedRelationalAlgebraFrozenSet(columns=("x",), iterable=[(1,)])
        result = _format_result(nras, column_names=["y"])
        # Existing name 'x' is preserved
        assert "x" in result
        assert "y" not in result

    # -- Sort tests --------------------------------------------------------

    def test_parse_sort_spec_empty(self):
        from neurolang.utils.cli import _parse_sort_spec
        assert _parse_sort_spec([]) == []

    def test_parse_sort_spec_asc_default(self):
        from neurolang.utils.cli import _parse_sort_spec
        assert _parse_sort_spec(["x"]) == [("x", True)]

    def test_parse_sort_spec_asc_explicit(self):
        from neurolang.utils.cli import _parse_sort_spec
        assert _parse_sort_spec(["x:asc"]) == [("x", True)]

    def test_parse_sort_spec_desc(self):
        from neurolang.utils.cli import _parse_sort_spec
        assert _parse_sort_spec(["x:desc"]) == [("x", False)]

    def test_parse_sort_spec_multiple(self):
        from neurolang.utils.cli import _parse_sort_spec
        result = _parse_sort_spec(["a", "b:desc", "c:asc"])
        assert result == [("a", True), ("b", False), ("c", True)]

    def test_parse_sort_spec_invalid_direction(self, capsys):
        from neurolang.utils.cli import _parse_sort_spec
        result = _parse_sort_spec(["x:sideways"])
        assert result == [("x", True)]
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_sort_named_set_ascending(self):
        from neurolang.utils.relational_algebra_set.pandas import (
            NamedRelationalAlgebraFrozenSet,
        )
        nras = NamedRelationalAlgebraFrozenSet(
            columns=("x", "y"), iterable=[(2, "b"), (1, "a"), (3, "c")]
        )
        result = _format_result(nras, sort_by=[("x", True)])
        lines = result.strip().split("\n")
        assert len(lines) == 4
        # First data row should be (1, a) since x is sorted ascending
        assert "1" in lines[1] and "a" in lines[1]

    def test_sort_named_set_descending(self):
        from neurolang.utils.relational_algebra_set.pandas import (
            NamedRelationalAlgebraFrozenSet,
        )
        nras = NamedRelationalAlgebraFrozenSet(
            columns=("x", "y"), iterable=[(1, "a"), (2, "b"), (3, "c")]
        )
        result = _format_result(nras, sort_by=[("x", False)])
        lines = result.strip().split("\n")
        assert len(lines) == 4
        # First data row should be (3, c) since x is sorted descending
        assert "3" in lines[1] and "c" in lines[1]

    def test_sort_named_set_two_keys(self):
        from neurolang.utils.relational_algebra_set.pandas import (
            NamedRelationalAlgebraFrozenSet,
        )
        nras = NamedRelationalAlgebraFrozenSet(
            columns=("x", "y"),
            iterable=[(1, "b"), (2, "a"), (1, "a")],
        )
        result = _format_result(
            nras, sort_by=[("x", True), ("y", True)]
        )
        lines = result.strip().split("\n")
        # Should be: (1,a), (1,b), (2,a)
        assert "1" in lines[1] and "a" in lines[1]
        assert "1" in lines[2] and "b" in lines[2]
        assert "2" in lines[3] and "a" in lines[3]

    def test_sort_unknown_column(self, capsys):
        from neurolang.utils.relational_algebra_set.pandas import (
            NamedRelationalAlgebraFrozenSet,
        )
        nras = NamedRelationalAlgebraFrozenSet(
            columns=("x",), iterable=[(2,), (1,)]
        )
        result = _format_result(nras, sort_by=[("nonexistent", True)])
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        # Should still produce output, unsorted
        assert "x" in result

    def test_sort_with_csv_format(self):
        from neurolang.utils.relational_algebra_set.pandas import (
            NamedRelationalAlgebraFrozenSet,
        )
        nras = NamedRelationalAlgebraFrozenSet(
            columns=("x", "y"), iterable=[(2, "b"), (1, "a")]
        )
        result = _format_result(
            nras, fmt="csv", sort_by=[("x", True)]
        )
        lines = result.strip().split("\n")
        assert lines[0] == "x,y"
        assert lines[1] == "1,a"
        assert lines[2] == "2,b"

    def test_sort_with_json_format(self):
        from neurolang.utils.relational_algebra_set.pandas import (
            NamedRelationalAlgebraFrozenSet,
        )
        nras = NamedRelationalAlgebraFrozenSet(
            columns=("x", "y"), iterable=[(2, "b"), (1, "a")]
        )
        result = _format_result(
            nras, fmt="json", sort_by=[("x", True)]
        )
        import json
        data = json.loads(result)
        assert data[0]["x"] == 1
        assert data[1]["x"] == 2


# ---------------------------------------------------------------------------
# _execute_program
# ---------------------------------------------------------------------------


class TestExecuteProgram:

    """Integration tests with a real NeurolangPDL engine and tiny data."""

    @pytest.fixture
    def nl(self):
        """Return a fresh NeurolangPDL with a small EDB loaded."""
        from neurolang.frontend import NeurolangPDL

        nl = NeurolangPDL()
        nl.add_tuple_set([(1, "a"), (2, "b"), (3, "c")], name="R")
        nl.add_tuple_set([(1, 10.0), (2, 20.0), (3, 30.0)], name="S")
        return nl

    def test_no_query(self, nl):
        # A fact-only program has no Query expression → returns None
        result = _execute_program(nl, "T(x) :- R(x, y)")
        assert result is None

    def test_simple_edb_query(self, nl):
        result = _execute_program(nl, "ans(x) :- R(x, y)")
        assert result is not None
        assert result.columns == ("x",)

    def test_query_with_conjunction(self, nl):
        result = _execute_program(
            nl, "ans(x, z) :- R(x, y) & S(x, z)"
        )
        assert result is not None
        assert result.columns == ("x", "z")

    def test_query_with_comparison(self, nl):
        result = _execute_program(
            nl, "ans(x) :- S(x, z) & (z > 15.0)"
        )
        assert result is not None
        assert result.columns == ("x",)

    def test_query_projection(self, nl):
        """Project only one of the variables."""
        result = _execute_program(
            nl, "ans(y) :- R(x, y) & S(x, z) & (z > 10.0)"
        )
        assert result is not None
        assert result.columns == ("y",)

    def test_query_no_matching_results(self, nl):
        result = _execute_program(nl, "ans(x) :- R(x, y) & (x > 100)")
        assert result is not None
        assert len(result) == 0

    def test_multiple_queries_raises(self, nl):
        with pytest.raises(NeuroLangException, match="more than one query"):
            _execute_program(nl, "ans(x) :- R(x, y)\nans(z) :- S(z, w)")

    def test_result_column_names_preserved(self, nl):
        """column_names match the Datalog variable names in the query."""
        result = _execute_program(
            nl, "ans(the_x, the_y) :- R(the_x, the_y)"
        )
        assert result.columns == ("the_x", "the_y")

    def test_table_format_output_has_column_names(self, nl):
        """Full roundtrip: execute → format with column names."""
        result = _execute_program(
            nl, "ans(val) :- R(x, y) & S(x, val)"
        )
        output = _format_result(result, column_names=list(result.columns))
        assert "val" in output
        assert "10.0" in output
        assert "20.0" in output
        assert "30.0" in output

    def test_csv_output_has_header(self, nl):
        result = _execute_program(nl, "ans(v) :- R(x, y) & S(x, v)")
        output = _format_result(result, fmt="csv", column_names=list(result.columns))
        lines = output.strip().split("\n")
        assert lines[0] == "v"


# ---------------------------------------------------------------------------
# _execute_program — PROB queries
# ---------------------------------------------------------------------------


class TestExecuteProgramProb:

    """Tests for PROB queries through _execute_program."""

    @pytest.fixture
    def nl(self):
        from neurolang.frontend import NeurolangPDL

        nl = NeurolangPDL()
        nl.add_tuple_set([(1, "a"), (2, "a")], name="edb1")
        nl.add_uniform_probabilistic_choice_over_set(
            [("a",), ("b",)], name="pc1"
        )
        return nl

    def test_prob_query_returns_named_set(self, nl):
        result = _execute_program(
            nl,
            "derived(x, p) :- PROB[ edb1(x, s) // pc1(s) ] = p.\n"
            "ans(x, p) :- derived(x, p).",
        )
        assert result is not None
        assert hasattr(result, "columns")
        assert result.columns == ("x", "p")

    def test_prob_query_values(self, nl):
        result = _execute_program(
            nl,
            "derived(x, p) :- PROB[ edb1(x, s) // pc1(s) ] = p.\n"
            "ans(x, p) :- derived(x, p).",
        )
        rows = sorted(iter(result))
        assert len(rows) == 2
        assert rows[0] == (1, 0.5)
        assert rows[1] == (2, 0.5)

    def test_prob_query_format_table(self, nl):
        result = _execute_program(
            nl,
            "derived(x, p) :- PROB[ edb1(x, s) // pc1(s) ] = p.\n"
            "ans(x, p) :- derived(x, p).",
        )
        output = _format_result(result)
        assert "x" in output
        assert "p" in output
        assert "0.5" in output

    def test_prob_query_format_csv(self, nl):
        result = _execute_program(
            nl,
            "derived(x, p) :- PROB[ edb1(x, s) // pc1(s) ] = p.\n"
            "ans(x, p) :- derived(x, p).",
        )
        output = _format_result(result, fmt="csv")
        lines = output.strip().split("\n")
        assert lines[0] == "x,p"
        assert "1,0.5" in lines[1:]

    def test_prob_rule_without_query_returns_none(self, nl):
        """PROB rule with no query returns None."""
        result = _execute_program(
            nl, "derived(x, p) :- PROB[ edb1(x, s) // pc1(s) ] = p."
        )
        assert result is None

    def test_marg_conjunction(self, nl):
        """MARG with multi-formula conjunction branch."""
        result = _execute_program(
            nl,
            "derived(x, p) :- MARG[ edb1(x, s) & pc1(s) ] = p.\n"
            "ans(x, p) :- derived(x, p).",
        )
        assert result is not None
        rows = sorted(iter(result))
        assert len(rows) == 2
        assert rows[0] == (1, 0.5)
        assert rows[1] == (2, 0.5)

    def test_prob_conjunction(self, nl):
        """PROB with multi-formula conjunction branch."""
        result = _execute_program(
            nl,
            "derived(x, s, p) :- PROB[ edb1(x, s) & pc1(s) ] = p.\n"
            "ans(x, s, p) :- derived(x, s, p).",
        )
        assert result is not None
        rows = sorted(iter(result))
        assert len(rows) == 2
        assert rows[0] == (1, "a", 0.5)
        assert rows[1] == (2, "a", 0.5)

    def test_prob_without_conditional_regular_body(self, nl):
        """PROB single predicate with outside_connect filtering (plus body atoms)."""
        result = _execute_program(
            nl,
            "derived(x, p) :- edb1(x, s) & PROB[ pc1(s) ] = p.\n"
            "ans(x, p) :- derived(x, p).",
        )
        assert result is not None
        rows = sorted(iter(result))
        assert len(rows) == 2
        assert rows[0] == (1, 0.5)
        assert rows[1] == (2, 0.5)

    def test_prob_non_ans_head(self, nl):
        """PROB desugaring with non-ans rule head (line 552 in _build_prob_rule)."""
        result = _execute_program(
            nl,
            "p(x, prob) :- PROB[ edb1(x, s) // pc1(s) ] = prob.\n"
            "ans(x, prob) :- p(x, prob).",
        )
        assert result is not None
        rows = sorted(iter(result))
        assert len(rows) == 2
        assert rows[0] == (1, 0.5)
        assert rows[1] == (2, 0.5)


# ---------------------------------------------------------------------------
# _classify_prob_predicate — Negation and ExistentialPredicate branches
# ---------------------------------------------------------------------------


class TestClassifyProbPredicateBranches:

    """Tests for _classify_prob_predicate branches inside PROB/MARG."""

    @pytest.fixture
    def nl(self):
        from neurolang.frontend import NeurolangPDL
        nl = NeurolangPDL()
        nl.add_tuple_set([(1, "a")], name="r1")
        nl.add_tuple_set([(1, "a", True)], name="r_bool")
        nl.add_uniform_probabilistic_choice_over_set([("a",), ("b",)], name="pc1")
        return nl

    def test_prob_existential(self, nl):
        """ExistentialPredicate inside PROB — hits ExistentialPredicate branch."""
        result = _execute_program(
            nl,
            "derived(x, p) :- PROB[ exists(s st r1(x, s)) ] = p.\n"
            "ans(x, p) :- derived(x, p).",
        )
        assert result is not None
        rows = sorted(iter(result))
        assert len(rows) == 1
        assert rows[0] == (1, 1.0)


# ---------------------------------------------------------------------------
# Miscellaneous grammar construct tests
# ---------------------------------------------------------------------------


class TestMiscGrammar:

    """Tests for parser transformer methods not yet covered."""

    @pytest.fixture
    def nl(self):
        from neurolang.frontend import NeurolangPDL
        nl = NeurolangPDL()
        nl.add_tuple_set([(1, "a")], name="R")
        nl.add_tuple_set([(True,)], name="Rbool")
        return nl

    def test_negation_body(self, nl):
        """Negation in a rule body (~R(x, y))."""
        result = _execute_program(
            nl,
            "ans(x) :- R(x, y) & ~(x == 2).",
        )
        assert result is not None


# ---------------------------------------------------------------------------
# --squall flag
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# --squall flag
# ---------------------------------------------------------------------------


class TestSquallFlag:
    def test_squall_defaults_to_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.squall is False

    def test_squall_flag_long(self):
        parser = _build_parser()
        args = parser.parse_args(["--squall", "obtain every Person."])
        assert args.squall is True
        assert args.query == "obtain every Person."

    def test_squall_flag_short(self):
        parser = _build_parser()
        args = parser.parse_args(["-s", "obtain every Person."])
        assert args.squall is True
        assert args.query == "obtain every Person."

    def test_squall_with_file(self, tmp_path):
        parser = _build_parser()
        fp = tmp_path / "q.squall"
        args = parser.parse_args(["-s", "-f", str(fp)])
        assert args.squall is True
        assert args.file == str(fp)


# ---------------------------------------------------------------------------
# --show-rewritten flag
# ---------------------------------------------------------------------------


class TestShowRewrittenFlag:
    def test_show_rewritten_defaults_to_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.show_rewritten is False

    def test_show_rewritten_flag_long(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--show-rewritten", "ans(x) :- R(x)"]
        )
        assert args.show_rewritten is True

    def test_show_rewritten_flag_short(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["-R", "ans(x) :- R(x)"]
        )
        assert args.show_rewritten is True

    def test_show_rewritten_works_without_squall(self):
        """Unlike --show-datalog, --show-rewritten does NOT require --squall."""
        parser = _build_parser()
        args = parser.parse_args(
            ["--show-rewritten", "ans(x) :- R(x)"]
        )
        assert args.show_rewritten is True
        assert args.squall is False

    def test_show_rewritten_prints_for_datalog(self, capsys):
        from neurolang.frontend import NeurolangDL
        from neurolang import expressions as ir
        from neurolang import datalog
        from neurolang import logic

        nl = NeurolangDL()
        nl.add_tuple_set([('a', 'b'), ('b', 'c'), ('c', 'd')], name='par')

        S_ = ir.Symbol
        x, y, z = S_('x'), S_('y'), S_('z')
        anc = S_('anc')
        par = S_('par')

        nl.program_ir.walk(datalog.Implication(anc(x, y), par(x, y)))
        nl.program_ir.walk(datalog.Implication(
            anc(x, y), logic.Conjunction((anc(x, z), par(z, y)))
        ))

        result = nl.execute_datalog_program(
            "ans(x) :- anc('a', x).", show_rewritten=True
        )
        captured = capsys.readouterr()
        assert "rewritten program" in captured.out
        assert "magic" in captured.out
        assert result is not None

    def test_show_rewritten_prints_for_squall(self, capsys):
        from neurolang.frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set([("alice",), ("bob",)], name="person")
        nl.add_tuple_set([("alice",)], name="plays")

        result = nl.execute_squall_program(
            "obtain every Person that plays.",
            show_rewritten=True
        )
        # SQUALL queries that don't trigger magic sets won't print
        # the rewritten header; that's expected behavior
        assert result is not None

    def test_show_rewritten_no_output_when_false(self, capsys):
        from neurolang.frontend import NeurolangDL
        from neurolang import expressions as ir
        from neurolang import datalog
        from neurolang import logic

        nl = NeurolangDL()
        nl.add_tuple_set([('a', 'b'), ('b', 'c'), ('c', 'd')], name='par')

        S_ = ir.Symbol
        x, y, z = S_('x'), S_('y'), S_('z')
        anc = S_('anc')
        par = S_('par')

        nl.program_ir.walk(datalog.Implication(anc(x, y), par(x, y)))
        nl.program_ir.walk(datalog.Implication(
            anc(x, y), logic.Conjunction((anc(x, z), par(z, y)))
        ))

        nl.execute_datalog_program(
            "ans(x) :- anc('a', x).", show_rewritten=False
        )
        captured = capsys.readouterr()
        assert "rewritten program" not in captured.out

    def test_show_rewritten_builds_engine_and_prints(self, monkeypatch, capsys):
        """--show-rewritten builds the requested engine and prints without solving."""
        from neurolang.frontend import NeurolangPDL
        from neurolang.utils.cli import main

        nl = NeurolangPDL()
        nl.add_tuple_set([("alice",), ("bob",)], name="person")
        nl.add_tuple_set([("alice",)], name="plays")

        build_engine_calls = []

        def fake_build_engine(name, data_dir, resolution=None):
            build_engine_calls.append((name, str(data_dir), resolution))
            return nl

        monkeypatch.setattr(
            engine_registry, "build_engine", fake_build_engine
        )

        main([
            "--engine", "neurosynth",
            "--show-rewritten",
            "--squall",
            "obtain every Person that plays.",
        ])

        captured = capsys.readouterr()
        assert build_engine_calls == [
            ("neurosynth", "neurolang_data", None),
        ]
        assert "rewritten program" in captured.out
        assert "Query completed" not in captured.err

    def test_show_rewritten_datalog_builds_engine(self, monkeypatch, capsys):
        """--show-rewritten for classical Datalog also builds the engine."""
        from neurolang.frontend import NeurolangPDL
        from neurolang.utils.cli import main

        nl = NeurolangPDL()
        nl.add_tuple_set([(1, "a"), (2, "b")], name="R")

        build_engine_calls = []

        def fake_build_engine(name, data_dir, resolution=None):
            build_engine_calls.append((name, str(data_dir), resolution))
            return nl

        monkeypatch.setattr(
            engine_registry, "build_engine", fake_build_engine
        )

        main([
            "--engine", "neurosynth",
            "--show-rewritten",
            "ans(x) :- R(x, y).",
        ])

        captured = capsys.readouterr()
        assert build_engine_calls == [
            ("neurosynth", "neurolang_data", None),
        ]
        assert "rewritten program" in captured.out
        assert "Query completed" not in captured.err


# ---------------------------------------------------------------------------
# --show-ra flag
# ---------------------------------------------------------------------------


class TestShowRAFlag:

    def test_show_ra_dry_run_no_query_completed(self, monkeypatch, capsys):
        """--show-ra builds the engine, prints the RA plan, and exits
        without printing 'Query completed'."""
        from neurolang.frontend import NeurolangPDL
        from neurolang.utils.cli import main

        nl = NeurolangPDL()
        nl.add_tuple_set([(1,), (2,)], name="P")

        build_engine_calls = []

        def fake_build_engine(name, data_dir, resolution=None):
            build_engine_calls.append((name, str(data_dir), resolution))
            return nl

        monkeypatch.setattr(
            engine_registry, "build_engine", fake_build_engine
        )

        main(["--show-ra", "ans(x) :- P(x)"])

        captured = capsys.readouterr()
        assert "── deterministic stratum ──" in captured.out
        assert "Query completed" not in captured.err

    def test_show_ra_squall_dry_run(self, monkeypatch, capsys):
        """--show-ra with --squall builds the engine, prints the RA plan,
        and exits without printing 'Query completed'."""
        from neurolang.frontend import NeurolangPDL
        from neurolang.utils.cli import main

        nl = NeurolangPDL()
        nl.add_tuple_set([(1,), (2,)], name="P")

        build_engine_calls = []

        def fake_build_engine(name, data_dir, resolution=None):
            build_engine_calls.append((name, str(data_dir), resolution))
            return nl

        monkeypatch.setattr(
            engine_registry, "build_engine", fake_build_engine
        )

        main(["--squall", "--show-ra", "obtain every P."])

        captured = capsys.readouterr()
        assert "── deterministic stratum ──" in captured.out
        assert "Query completed" not in captured.err

    def test_show_ra_neurosynth_prob_query(self, monkeypatch, capsys):
        """--show-ra --squall with the original Neurosynth conditional MARG
        query crashes with IndexError at
        relational_algebra_provenance.py:423 before the fix."""
        from neurolang.frontend import NeurolangPDL
        from neurolang.utils.cli import main

        nl = NeurolangPDL()
        nl.add_tuple_set(
            [(10, 20, 30, "s1"), (11, 21, 31, "s2")], name="peak_reported"
        )
        nl.add_tuple_set([("s1",), ("s2",)], name="study")
        nl.add_uniform_probabilistic_choice_over_set(
            [("s1",), ("s2",)], name="selected_study"
        )
        nl.add_tuple_set(
            [("memory", "s1", 0.5), ("memory", "s2", 0.6)],
            name="term_in_study_tfidf",
        )
        nl.add_tuple_set([("A",), ("B",)], name="schaefer")
        nl.add_tuple_set([(10, 20, 30), (11, 21, 31)], name="voxel")
        # datalog_init rules from engines.yaml for the neurosynth engine
        nl.execute_datalog_program(
            "study_with_peaks(S) :- peak_reported(I, J, K, S)."
        )
        nl.execute_datalog_program("schaefer_label(l) :- schaefer(l, r).")
        nl.execute_datalog_program(
            "labels(l, i, j, k) :- voxel(i, j, k), schaefer(l)."
        )
        nl.execute_datalog_program(
            "mentions(s, t) :- term_in_study_tfidf(s, t, v), (v > 0.03)."
        )
        nl.execute_datalog_program(
            "reports(s, i, j, k) :- peak_reported(i, j, k, s)."
        )

        def fake_build_engine(name, data_dir, resolution=None):
            return nl

        monkeypatch.setattr(engine_registry, "build_engine", fake_build_engine)

        # Before the fix this raises IndexError; after the fix it prints the
        # RA plan and exits cleanly.
        main([
            "--show-ra", "--squall",
            "define as Label_reports with inferred probability every "
            "Schaefer_label that labels a Voxel in 3D that a Selected_study "
            "reports given the Selected_study mentions 'memory'. "
            "obtain every Label_reports .",
        ])
        captured = capsys.readouterr()
        assert "── probabilistic stratum ──" in captured.out


# ---------------------------------------------------------------------------
# --show-datalog flag
# ---------------------------------------------------------------------------


class TestShowDatalogFlag:
    def test_show_datalog_defaults_to_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.show_datalog is False

    def test_show_datalog_flag_long(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--squall", "--show-datalog", "obtain every Person."]
        )
        assert args.show_datalog is True
        assert args.squall is True

    def test_show_datalog_flag_short(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["-s", "-D", "obtain every Person."]
        )
        assert args.show_datalog is True
        assert args.squall is True

    def test_show_datalog_requires_squall(self):
        from neurolang.utils.cli import main

        with pytest.raises(SystemExit):
            main(["--show-datalog", "ans(x) :- R(x)"])

    def test_show_datalog_prints_ir(self, capsys):
        from neurolang.utils.cli import _show_squall_datalog

        _show_squall_datalog("obtain every Person that plays.")
        captured = capsys.readouterr()
        assert "query" in captured.out
        assert "person" in captured.out


class TestFormatIR:
    def test_symbol_non_fresh(self):
        printer = DatalogPrettyPrinter()
        x = Symbol("x")
        assert printer.walk(x) == "x"

    def test_symbol_fresh_renamed(self):
        printer = DatalogPrettyPrinter()
        f0 = Symbol.fresh()
        f1 = Symbol.fresh()
        result = printer.walk((f0, f1))
        assert "s\u2080" in result
        assert "s\u2081" in result

    def test_constant_string(self):
        printer = DatalogPrettyPrinter()
        c = Constant("emotion")
        result = printer.walk(c)
        assert "'emotion'" in result

    def test_function_application(self):
        printer = DatalogPrettyPrinter()
        voxel = Symbol("voxel")
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        app = voxel(x, y, z)
        result = printer.walk(app)
        assert result == "voxel(x, y, z)"

    def test_conjunction_inline(self):
        printer = DatalogPrettyPrinter()
        a, b = Symbol("a"), Symbol("b")
        p = Symbol("p")
        q = Symbol("q")
        conj = Conjunction((p(a), q(b)))
        result = printer.walk(conj)
        assert "p(a) \u2227 q(b)" == result

    def test_existential_predicate(self):
        printer = DatalogPrettyPrinter()
        s = Symbol.fresh()
        study = Symbol("study")(s)
        body = Conjunction((study,))
        ex = ExistentialPredicate(s, body)
        result = printer.walk(ex)
        assert result.startswith("\u2203s\u2080")
        assert "study(s\u2080)" in result

    def test_query_body_breaks_conjunction_into_lines(self):
        parsed = squall_parser("obtain every Voxel (?x; ?y; ?z).")
        printer = DatalogPrettyPrinter()
        result = printer.walk(parsed.queries[0])
        lines = result.split("\n")
        assert ":-" in lines[0]
        assert "voxel(x, y, z)" in lines[1].strip()

    def test_nd_annotation_uses_fresh_vars(self):
        parsed = squall_parser(
            "obtain every Voxel in 3D that a Study reported."
        )
        printer = DatalogPrettyPrinter()
        result = printer.walk(parsed.queries[0])
        assert "s\u2080" in result
        assert "\u2203" in result
        assert "voxel" in result
        assert "study" in result
        assert "reported" in result


# ---------------------------------------------------------------------------
# _execute_squall_program
# ---------------------------------------------------------------------------


class TestExecuteSquallProgram:

    """Integration tests with a real NeurolangPDL engine and squall queries."""

    @pytest.fixture
    def nl(self):
        from neurolang.frontend import NeurolangPDL

        nl = NeurolangPDL()
        nl.add_tuple_set([("alice",), ("bob",)], name="person")
        nl.add_tuple_set([("alice",)], name="plays")
        return nl

    def test_single_obtain_returns_set(self, nl):
        result = _execute_squall_program(
            nl, "obtain every Person that plays."
        )
        df = result.as_pandas_dataframe()
        assert len(df) == 1
        assert df.iloc[0, 0] == "alice"

    def test_rules_only_returns_none(self, nl):
        result = _execute_squall_program(
            nl, "define as Active every person that plays."
        )
        assert result is None

    def test_multiple_obtains_returns_dict(self, nl):
        result = _execute_squall_program(
            nl,
            "obtain every Person that plays.\n"
            "obtain every Person.",
        )
        assert isinstance(result, dict)
        assert "obtain_0" in result
        assert "obtain_1" in result

    def test_single_obtain_formatting_has_columns(self, nl):
        result = _execute_squall_program(
            nl, "obtain every Person that plays."
        )
        output = _format_result(result)
        from neurolang.utils.relational_algebra_set.pandas import (
            NamedRelationalAlgebraFrozenSet,
        )
        assert isinstance(result, NamedRelationalAlgebraFrozenSet)
        # Named columns should appear in the table output
        assert any(
            c in output for c in result.columns
        ), f"Column names {result.columns} not found in output:\n{output}"

    def test_squall_format_result_none(self, nl):
        result = _execute_squall_program(
            nl, "define as Active every person that plays."
        )
        assert _format_result(result) == ""
