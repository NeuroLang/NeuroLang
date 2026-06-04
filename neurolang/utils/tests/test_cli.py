"""Tests for the neurolang-query CLI module."""

import pytest

from neurolang.utils.cli import (
    _build_parser,
    _execute_program,
    _execute_squall_program,
    _format_result,
    _show_squall_datalog,
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

        # Query CSV relation — _execute_program bypasses the probabilistic
        # frontend query path (which doesn't handle EDB-only queries).
        result, col_names = _execute_program(
            nl, "ans(x, y) :- csv_rel(x, y)"
        )
        assert col_names == ["x", "y"]
        assert result is not None
        output = _format_result(result, column_names=col_names)
        assert "a" in output
        assert "b" in output
        assert "c" in output

        # Query TSV relation
        result, col_names = _execute_program(
            nl, "ans(x, y) :- tsv_rel(x, y)"
        )
        assert col_names == ["x", "y"]
        assert result is not None
        output = _format_result(result, column_names=col_names)
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
        result, col_names = _execute_program(nl, "ans(x) :- R(x, y)")
        assert col_names == ["x"]
        assert result is not None

    def test_query_with_conjunction(self, nl):
        result, col_names = _execute_program(
            nl, "ans(x, z) :- R(x, y) & S(x, z)"
        )
        assert col_names == ["x", "z"]
        assert result is not None

    def test_query_with_comparison(self, nl):
        result, col_names = _execute_program(
            nl, "ans(x) :- S(x, z) & (z > 15.0)"
        )
        assert col_names == ["x"]
        assert result is not None

    def test_query_projection(self, nl):
        """Project only one of the variables."""
        result, col_names = _execute_program(
            nl, "ans(y) :- R(x, y) & S(x, z) & (z > 10.0)"
        )
        assert col_names == ["y"]
        assert result is not None

    def test_query_no_matching_results(self, nl):
        result = _execute_program(nl, "ans(x) :- R(x, y) & (x > 100)")
        # Chase may return None when a predicate yields no tuples
        assert result is None

    def test_multiple_queries_raises(self, nl):
        with pytest.raises(ValueError, match="single query"):
            _execute_program(nl, "ans(x) :- R(x, y)\nans(z) :- S(z, w)")

    def test_result_column_names_preserved(self, nl):
        """column_names match the Datalog variable names in the query."""
        _, col_names = _execute_program(
            nl, "ans(the_x, the_y) :- R(the_x, the_y)"
        )
        assert col_names == ["the_x", "the_y"]

    def test_table_format_output_has_column_names(self, nl):
        """Full roundtrip: execute → format with column names."""
        result, col_names = _execute_program(
            nl, "ans(val) :- R(x, y) & S(x, val)"
        )
        output = _format_result(result, column_names=col_names)
        assert "val" in output
        assert "10.0" in output
        assert "20.0" in output
        assert "30.0" in output

    def test_csv_output_has_header(self, nl):
        result, col_names = _execute_program(nl, "ans(v) :- R(x, y) & S(x, v)")
        output = _format_result(result, fmt="csv", column_names=col_names)
        lines = output.strip().split("\n")
        assert lines[0] == "v"


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
        from neurolang.expressions import Symbol
        from neurolang.utils.cli import DatalogPrettyPrinter

        printer = DatalogPrettyPrinter()
        x = Symbol("x")
        assert printer.walk(x) == "x"

    def test_symbol_fresh_renamed(self):
        from neurolang.expressions import Symbol
        from neurolang.utils.cli import DatalogPrettyPrinter

        printer = DatalogPrettyPrinter()
        f0 = Symbol.fresh()
        f1 = Symbol.fresh()
        result = printer.walk((f0, f1))
        assert "s\u2080" in result
        assert "s\u2081" in result

    def test_constant_string(self):
        from neurolang.expressions import Constant
        from neurolang.utils.cli import DatalogPrettyPrinter

        printer = DatalogPrettyPrinter()
        c = Constant("emotion")
        result = printer.walk(c)
        assert "'emotion'" in result

    def test_function_application(self):
        from neurolang.expressions import Symbol
        from neurolang.utils.cli import DatalogPrettyPrinter

        printer = DatalogPrettyPrinter()
        voxel = Symbol("voxel")
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        app = voxel(x, y, z)
        result = printer.walk(app)
        assert result == "voxel(x, y, z)"

    def test_conjunction_inline(self):
        from neurolang.expressions import Symbol
        from neurolang.logic import Conjunction
        from neurolang.utils.cli import DatalogPrettyPrinter

        printer = DatalogPrettyPrinter()
        a, b = Symbol("a"), Symbol("b")
        p = Symbol("p")
        q = Symbol("q")
        conj = Conjunction((p(a), q(b)))
        result = printer.walk(conj)
        assert "p(a) \u2227 q(b)" == result

    def test_existential_predicate(self):
        from neurolang.expressions import Symbol
        from neurolang.logic import Conjunction, ExistentialPredicate
        from neurolang.utils.cli import DatalogPrettyPrinter

        printer = DatalogPrettyPrinter()
        s = Symbol.fresh()
        study = Symbol("study")(s)
        body = Conjunction((study,))
        ex = ExistentialPredicate(s, body)
        result = printer.walk(ex)
        assert result.startswith("\u2203s\u2080")
        assert "study(s\u2080)" in result

    def test_query_body_breaks_conjunction_into_lines(self):
        from neurolang.frontend.datalog.squall_syntax_lark import (
            parser as squall_parser,
        )
        from neurolang.utils.cli import DatalogPrettyPrinter

        parsed = squall_parser("obtain every Voxel (?x; ?y; ?z).")
        printer = DatalogPrettyPrinter()
        result = printer.walk(parsed.queries[0])
        lines = result.split("\n")
        assert ":-" in lines[0]
        assert "voxel(x, y, z)" in lines[1].strip()

    def test_nd_annotation_uses_fresh_vars(self):
        from neurolang.frontend.datalog.squall_syntax_lark import (
            parser as squall_parser,
        )
        from neurolang.utils.cli import DatalogPrettyPrinter

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
