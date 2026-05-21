"""
Tests for the neurolang-query CLI module.
"""

import pytest

from neurolang.utils.cli import (
    _build_parser,
    _execute_program,
    _format_result,
)

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

    def test_file_arg(self):
        parser = _build_parser()
        args = parser.parse_args(["--file", "/tmp/q.dl"])
        assert args.file == "/tmp/q.dl"
        assert args.query is None

    def test_engine(self):
        parser = _build_parser()
        args = parser.parse_args(["--engine", "destrieux"])
        assert args.engine == "destrieux"

    def test_invalid_engine(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--engine", "unknown"])

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

    def test_mutually_exclusive_query_and_file(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ans(x) :- R(x)", "--file", "/tmp/q.dl"])

    def test_short_flags(self):
        parser = _build_parser()
        args = parser.parse_args(["-e", "destrieux", "-f", "/tmp/q.dl", "-l"])
        assert args.engine == "destrieux"
        assert args.file == "/tmp/q.dl"
        assert args.list_predicates is True


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
