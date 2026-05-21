"""
Command-line interface for NeuroLang query execution.

Provides a ``neurolang-query`` command that initializes a NeuroLang engine
with the NeuroSynth (or Destrieux) dataset and executes Datalog queries
from arguments, files, or standard input.

Usage
-----
::

    # Inline query
    neurolang-query "ans(term) :- TermInStudyTFIDF(term, tfidf, study_id)"

    # Query from file
    neurolang-query -f query.dl

    # Query from stdin
    echo "ans(term) :- TermInStudyTFIDF(term, tfidf, study_id)" | neurolang-query

    # Destrieux atlas engine
    neurolang-query --engine destrieux "ans(name) :- destrieux(name, region)"
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import nibabel as nib
import numpy as np
from nilearn import datasets, image

try:
    from nilearn.datasets.utils import _fetch_files
except ImportError:
    # nilearn >= 0.11: _fetch_files was moved and renamed
    from nilearn.datasets._utils import fetch_files as _fetch_files

from neurolang import expressions as ir
from neurolang.datalog.chase import Chase
from neurolang.datalog.expressions import Implication
from neurolang.frontend import NeurolangPDL
from neurolang.frontend.neurosynth_utils import StudyID
from neurolang.regions import ExplicitVBR, ExplicitVBROverlay, region_union


def _init_frontend(mni_mask: nib.Nifti1Image) -> NeurolangPDL:
    """Create a :class:`~neurolang.frontend.NeurolangPDL` instance and
    register the aggregation functions and symbols required for
    Neurosynth / Destrieux queries.

    Parameters
    ----------
    mni_mask : nib.Nifti1Image
        MNI template image used for voxel-space coordinate mapping.

    Returns
    -------
    NeurolangPDL
        Configured probabilistic Datalog engine.
    """
    nl = NeurolangPDL()

    nl.add_symbol(np.exp, name="exp", type_=Callable[[float], float])
    nl.add_symbol(np.log, name="log", type_=Callable[[float], float])

    @nl.add_symbol
    def agg_count(i: Iterable) -> np.int64:
        return np.int64(len(i))

    @nl.add_symbol
    def agg_create_region(
        i: Iterable, j: Iterable, k: Iterable
    ) -> ExplicitVBR:
        voxels = np.c_[i, j, k]
        return ExplicitVBR(voxels, mni_mask.affine, image_dim=mni_mask.shape)

    @nl.add_symbol
    def agg_create_region_overlay(
        i: Iterable, j: Iterable, k: Iterable, p: Iterable
    ) -> ExplicitVBROverlay:
        voxels = np.c_[i, j, k]
        return ExplicitVBROverlay(
            voxels, mni_mask.affine, p, image_dim=mni_mask.shape
        )

    @nl.add_symbol
    def startswith(prefix: str, s: str) -> bool:
        return s.startswith(prefix)

    @nl.add_symbol
    def principal_direction(
        s: ExplicitVBR, direction: str, eps: float = 1e-6
    ) -> bool:
        """Test the principal direction of a volumetric brain region."""
        c = ["LR", "AP", "SI"]
        s_xyz = s.to_xyz()
        cov = np.cov(s_xyz.T)
        evals, evecs = np.linalg.eig(cov)
        i = np.argmax(np.abs(evals))
        abs_max_evec = np.abs(evecs[:, i].squeeze())
        sort_dir = np.argsort(abs_max_evec)
        if (
            np.abs(abs_max_evec[sort_dir[-1]] - abs_max_evec[sort_dir[-2]])
            < eps
        ):
            return False
        else:
            main_dir = c[sort_dir[-1]]
        return (direction == main_dir) or (direction[::-1] == main_dir)

    return nl


def _load_neurosynth_data(
    data_dir: Path, nl: NeurolangPDL, mni_mask: nib.Nifti1Image
) -> None:
    """Download (if needed) and load the NeuroSynth dataset into the engine.

    Registers the following predicates:

    * ``PeakReported(study_id, i, j, k)`` — reported activation peaks
      converted to voxel coordinates.
    * ``Study(study_id)`` — all study identifiers.
    * ``TermInStudyTFIDF(term, tfidf, study_id)`` — term frequency–inverse
      document frequency values.
    * ``SelectedStudy(study_id)`` — uniform probabilistic choice over
      studies.
    * ``Voxel(i, j, k)`` — every voxel inside the MNI mask.
    """
    ns_database_fn, ns_features_fn = _fetch_files(
        str(data_dir / "neurosynth"),
        [
            (
                "database.txt",
                "https://github.com/neurosynth/neurosynth-data/raw/master"
                "/current_data.tar.gz",
                {"uncompress": True},
            ),
            (
                "features.txt",
                "https://github.com/neurosynth/neurosynth-data/raw/master"
                "/current_data.tar.gz",
                {"uncompress": True},
            ),
        ],
    )

    activations = _read_ns_database(ns_database_fn)
    peak_data = _process_peaks(activations, mni_mask)
    study_ids = peak_data[["study_id"]].drop_duplicates()

    features = _read_ns_features(ns_features_fn)
    term_data = features.melt(
        var_name="term", id_vars="study_id", value_name="tfidf"
    ).query("tfidf > 1e-3")[["term", "tfidf", "study_id"]]
    term_data["study_id"] = term_data["study_id"].apply(StudyID)

    nl.add_tuple_set(peak_data, name="PeakReported")
    nl.add_tuple_set(study_ids, name="Study")
    nl.add_tuple_set(term_data, name="TermInStudyTFIDF")
    nl.add_uniform_probabilistic_choice_over_set(
        study_ids, name="SelectedStudy"
    )
    nl.add_tuple_set(
        np.hstack(
            np.meshgrid(
                *(np.arange(0, dim) for dim in mni_mask.get_fdata().shape)
            )
        )
        .swapaxes(0, 1)
        .reshape(3, -1)
        .T,
        name="Voxel",
    )


def _read_ns_database(path: str) -> "pd.DataFrame":
    import pandas as pd

    activations = pd.read_csv(path, sep="\t")
    activations["id"] = activations["id"].apply(StudyID)
    return activations


def _process_peaks(
    activations: "pd.DataFrame", mni_mask: nib.Nifti1Image
) -> "pd.DataFrame":
    """Convert MNI / Talairach peak coordinates to voxel indices."""
    import pandas as pd

    mni_peaks = activations.loc[activations.space == "MNI"][
        ["x", "y", "z", "id"]
    ].rename(columns={"id": "study_id"})
    non_mni_peaks = activations.loc[activations.space != "MNI"][
        ["x", "y", "z", "id"]
    ].rename(columns={"id": "study_id"})

    proj_mat = np.linalg.pinv(
        np.array(
            [
                [0.9254, 0.0024, -0.0118, -1.0207],
                [-0.0048, 0.9316, -0.0871, -1.7667],
                [0.0152, 0.0883, 0.8924, 4.0926],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).T
    )
    projected = np.round(
        np.dot(
            np.hstack(
                (
                    non_mni_peaks[["x", "y", "z"]].values,
                    np.ones((len(non_mni_peaks), 1)),
                )
            ),
            proj_mat,
        )[:, 0:3]
    )
    projected_df = pd.DataFrame(
        np.hstack([projected, non_mni_peaks[["study_id"]].values]),
        columns=["x", "y", "z", "study_id"],
    )
    peak_data = pd.concat([projected_df, mni_peaks]).astype(
        {"x": int, "y": int, "z": int}
    )

    ijk_positions = np.round(
        nib.affines.apply_affine(
            np.linalg.inv(mni_mask.affine),
            peak_data[["x", "y", "z"]].values.astype(float),
        )
    ).astype(int)
    peak_data["i"] = ijk_positions[:, 0]
    peak_data["j"] = ijk_positions[:, 1]
    peak_data["k"] = ijk_positions[:, 2]
    peak_data = peak_data[["i", "j", "k", "study_id"]]
    return peak_data


def _read_ns_features(path: str) -> "pd.DataFrame":
    import pandas as pd

    features = pd.read_csv(path, sep="\t")
    features.rename(columns={"pmid": "study_id"}, inplace=True)
    return features


def _load_destrieux_atlas(data_dir: Path, nl: NeurolangPDL) -> None:
    """Load the Destrieux atlas into the engine."""
    destrieux_atlas = datasets.fetch_atlas_destrieux_2009(
        data_dir=str(data_dir / "destrieux")
    )
    nl.new_symbol(name="destrieux")
    destrieux_atlas_image = nib.load(destrieux_atlas["maps"])
    destrieux_labels = dict(destrieux_atlas["labels"])
    destrieux_set = set()
    for k, v in destrieux_labels.items():
        if k == 0:
            continue
        destrieux_set.add(
            (
                v.decode("utf8").replace("-", " ").replace("_", " "),
                ExplicitVBR.from_spatial_image_label(
                    destrieux_atlas_image, k
                ),
            )
        )
    nl.add_tuple_set(destrieux_set, name="destrieux")


def _get_mni_mask(data_dir: Path) -> nib.Nifti1Image:
    return nib.load(
        datasets.fetch_icbm152_2009(data_dir=str(data_dir / "icbm"))["t1"]
    )


def _build_engine(
    engine_name: str,
    data_dir: Path,
    resolution: Optional[float] = None,
) -> NeurolangPDL:
    """Create and populate a :class:`NeurolangPDL` engine.

    Parameters
    ----------
    engine_name :
        ``"neurosynth"`` or ``"destrieux"``.
    data_dir :
        Directory under which downloaded data is cached.
    resolution :
        If set, resample the MNI mask to the given isotropic resolution (mm).

    Returns
    -------
    NeurolangPDL
        A fully initialised engine ready for ``execute_datalog_program``.
    """
    data_dir = Path(data_dir)
    mask = _get_mni_mask(data_dir)
    if resolution is not None:
        mask = image.resample_img(mask, np.eye(4) * resolution)

    nl = _init_frontend(mask)

    if engine_name == "neurosynth":
        _load_neurosynth_data(data_dir, nl, mask)
    elif engine_name == "destrieux":
        nl.add_symbol(
            region_union,
            name="region_union",
            type_=Callable[[Iterable[ExplicitVBR]], ExplicitVBR],
        )
        _load_destrieux_atlas(data_dir, nl)
    else:
        raise ValueError(f"Unknown engine: {engine_name!r}")

    return nl


def _read_query(args) -> str:
    """Obtain the Datalog program from the CLI arguments."""
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.query:
        return args.query
    if not sys.stdin.isatty():
        return sys.stdin.read()
    parser = _build_parser()
    parser.print_help()
    sys.exit(1)


def _format_result(
    result, fmt: str = "table", column_names: Optional[list[str]] = None
) -> str:
    """Format a query result for display."""
    if result is None:
        return ""

    if isinstance(result, bool):
        return "true" if result else "false"

    import pandas as pd

    def _maybe_rename_columns(df):
        """Rename unnamed columns when variable names are available."""
        if column_names is not None and len(column_names) == len(df.columns):
            unnamed = all(
                isinstance(c, int) or (isinstance(c, str) and c.startswith("c"))
                for c in df.columns
            )
            if unnamed:
                df.columns = column_names
        return df

    try:
        if hasattr(result, "value") and hasattr(result.value, "unwrap"):
            inner = result.value.unwrap()
            if hasattr(inner, "as_pandas_dataframe"):
                df = _maybe_rename_columns(inner.as_pandas_dataframe())
                if df.empty:
                    return "(empty)"
                if fmt == "csv":
                    return df.to_csv(index=False)
                elif fmt == "json":
                    return df.to_json(orient="records", indent=2)
                return df.to_string(index=False)
            else:
                result = inner

        if hasattr(result, "columns") and result.columns:
            df = _maybe_rename_columns(
                pd.DataFrame(iter(result), columns=result.columns)
            )
        else:
            rows = list(iter(result))
            if not rows:
                return "(empty)"
            arity = result.arity if hasattr(result, "arity") else len(rows[0])
            df = _maybe_rename_columns(
                pd.DataFrame(rows, columns=[f"c{i}" for i in range(arity)])
            )

        if df.empty:
            return "(empty)"

        if fmt == "csv":
            return df.to_csv(index=False)
        elif fmt == "json":
            return df.to_json(orient="records", indent=2)
        return df.to_string(index=False)
    except Exception:
        return str(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurolang-query",
        description="Run a Datalog query against a NeuroLang dataset engine.",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Datalog program string (e.g. 'ans(x) :- P(x)'). "
        "If omitted and stdin is a pipe, read from stdin.",
    )
    source.add_argument(
        "--file",
        "-f",
        metavar="PATH",
        default=None,
        help="Read the Datalog program from a file.",
    )

    parser.add_argument(
        "--engine",
        "-e",
        choices=("neurosynth", "destrieux"),
        default="neurosynth",
        help="Dataset engine to use (default: neurosynth).",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        metavar="DIR",
        default="neurolang_data",
        help="Directory for cached dataset downloads (default: neurolang_data).",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=float,
        default=None,
        metavar="MM",
        help="Isotropic MNI resolution in mm (default: native).",
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--list-predicates",
        "-l",
        action="store_true",
        help="Instead of running a query, list the available predicates "
        "(extensional database symbols) for the engine and exit.",
    )

    return parser


def _execute_program(nl: NeurolangPDL, program_text: str):
    """Execute a Datalog program using direct chase evaluation.

    This bypasses :meth:`NeurolangPDL._execute_query` which does not support
    query bodies that are conjunctions, comparisons, or EDB-only predicates.
    Instead it converts every ``Query`` into an ``Implication`` rule, adds it
    to the program's IDB, and runs the deterministic chase — the same strategy
    used by the base :class:`~neurolang.frontend.QueryBuilderDatalog`.

    Parameters
    ----------
    nl :
        Initialised engine (``NeurolangPDL`` or ``NeurolangDL``).
    program_text :
        Datalog program, possibly containing one ``Query`` rule.

    Returns
    -------
    ``None``, ``bool``, or a relational algebra set.
    """
    from neurolang.frontend.datalog.standard_syntax import parser as datalog_parser

    ir_prog = datalog_parser(program_text)

    formulas = list(ir_prog.formulas)
    queries = [f for f in formulas if isinstance(f, ir.Query)]
    others = [f for f in formulas if not isinstance(f, ir.Query)]

    for f in others:
        nl.program_ir.walk(f)

    if len(queries) == 0:
        return None

    if len(queries) > 1:
        raise ValueError("Only a single query per program is supported.")

    q = queries[0]

    # Extract column names from the query head (the Datalog variable names).
    # The parsed head is FunctionApplication(functor, args…) where args are
    # Symbol nodes whose .name carries the variable name.
    if isinstance(q.head, ir.FunctionApplication):
        column_names = [a.name for a in q.head.args]
    else:
        column_names = None

    # Query → Implication so the DatalogProgram walker registers it as IDB
    rule = Implication(q.head, q.body)
    nl.program_ir.walk(rule)

    # Determine the predicate name from the query head.
    # The parsed head is FunctionApplication(Lambda, args).
    if isinstance(q.head, ir.FunctionApplication):
        if isinstance(q.head.functor, ir.Lambda):
            pred_symbol = q.head.functor.body
        else:
            pred_symbol = q.head.functor
    else:
        pred_symbol = q.head

    pred_name = (
        pred_symbol.name
        if isinstance(pred_symbol, ir.Symbol)
        else str(pred_symbol)
    )

    chase = Chase(nl.program_ir)
    solution = chase.build_chase_solution()

    for sym, val in solution.items():
        if sym.name == pred_name:
            return val, column_names

    return None


def main(argv: Optional[list] = None) -> None:
    """CLI entry point: parse arguments, build engine, run query."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    nl = _build_engine(args.engine, args.data_dir, args.resolution)

    if args.list_predicates:
        print(f"\nEngine: {args.engine}")
        print("Available predicates (EDB symbols):")
        print("─" * 40)
        for sym in nl.symbol_table:
            name = sym.name
            if name.startswith("_"):
                continue
            val = nl.symbol_table[sym]
            val_repr = repr(val)
            if len(val_repr) > 80:
                val_repr = val_repr[:77] + "..."
            print(f"  {name}: {val_repr}")
        return

    program = _read_query(args)
    if not program or not program.strip():
        print("Error: no query provided.", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    result = _execute_program(nl, program)
    elapsed = time.perf_counter() - t0

    if isinstance(result, tuple):
        result, column_names = result
    else:
        column_names = None

    output = _format_result(result, fmt=args.format, column_names=column_names)
    if output:
        print(output)

    print(f"Query completed in {elapsed:.2f} s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
