"""
The tables of an engine that are kept, to start it without finding them again.

Some of the tables of an engine take longer to find than the rest of its build
together: the studies activating the regions of its atlases take 48 s of the 56
s the ``neurosynth`` engine takes to build, and 99 s of the 112 s in a
browser. They depend only on the data (the atlas, the peaks and the radius), so
:func:`~neurolang_apps.utils.engine_registry.build_engine` keeps them in the
engine's data directory, in files named by what they depend on, and reads them
when it finds them there. The terminal REPL and the REPL in the browser use
the same files: they hold arrays and text only (``.npz``), so a file written by
CPython is read as it is by Pyodide, whatever the versions of numpy and pandas
on each side, and there is no copy to make for each Python implementation.
A cache of the engine's Python objects, which would be one per implementation
and version, is not possible: the expressions of the engine cannot be pickled.

``python -m neurolang_apps.utils.engine_cache ENGINE`` builds the engine so
that its tables are kept (the build of the browser's bundle does it, and its
data zip carries them), ``--list`` tells which are kept and ``--clear``
removes them.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

from neurolang_apps.utils import engine_registry


def cache_dir(data_dir, engine_name: str) -> Path:
    """Return the directory where the tables of the engine are kept."""
    return (
        Path(data_dir) / engine_name / engine_registry.ACTIVATION_CACHE_DIR
    )


def cached_files(data_dir, engine_name: str) -> List[Path]:
    """Return the files of the tables kept for the engine, sorted."""
    return sorted(cache_dir(data_dir, engine_name).glob("*.npz"))


def prebuild(
    engine_name: str, data_dir, resolution: Optional[float] = None,
    yaml_path=None,
) -> List[Path]:
    """
    Build the engine so that its tables are kept in the data directory, and
    return the files they are in.
    """
    engine_registry.build_engine(
        engine_name, Path(data_dir), resolution, yaml_path, use_cache=True
    )
    return cached_files(data_dir, engine_name)


def clear(data_dir, engine_name: str) -> int:
    """Remove the files of the tables kept for the engine, and count them."""
    files = cached_files(data_dir, engine_name)
    for path in files:
        path.unlink()
    return len(files)


def _describe(files: List[Path]) -> str:
    size = sum(path.stat().st_size for path in files) / 1048576
    return f"{len(files)} files, {size:.1f} MB"


def main(argv=None) -> int:
    """Run ``python -m neurolang_apps.utils.engine_cache``."""
    parser = argparse.ArgumentParser(
        prog="python -m neurolang_apps.utils.engine_cache",
        description="Keep the tables of an engine that take long to find.",
    )
    parser.add_argument("engine", help="The engine, e.g. neurosynth.")
    parser.add_argument(
        "--data-dir", "-d", default="neurolang_data", metavar="DIR",
        help="The data directory (default: neurolang_data).",
    )
    parser.add_argument(
        "--resolution", "-r", type=float, default=None, metavar="MM",
        help="Isotropic MNI resolution in mm, as of neurolang-query.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--list", action="store_true", help="List the tables kept.",
    )
    action.add_argument(
        "--clear", action="store_true", help="Remove the tables kept.",
    )
    args = parser.parse_args(argv)
    if args.list:
        files = cached_files(args.data_dir, args.engine)
    elif args.clear:
        print(f"Removed {clear(args.data_dir, args.engine)} files.")
        return 0
    else:
        files = prebuild(args.engine, args.data_dir, args.resolution)
    for path in files:
        print(f"{path.stat().st_size / 1024:8.0f} KB  {path}")
    print(_describe(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
