"""
The tables of an engine that are kept, to start it without finding them again.

Building the ``neurosynth`` engine takes 58 s natively and 112 s in a browser,
and most of it is finding what never changes for the same data: the studies
activating the regions of the atlases (48 s natively), and what nilearn is
asked for, the MNI mask resampled to the grid of the engine and the voxels of
the brain and of the cortex on it, which costs more than a second to import
nilearn and scikit-learn for, and several in a browser. So
:func:`~neurolang_apps.utils.engine_registry.build_engine` keeps them in the
engine's data directory (``DATA_DIR/ENGINE/engine_cache``), in files named by
their kind and by what they depend on, and reads them when it finds them
there: with them, nilearn is not imported at all, the build takes 6 s
natively and 8 s in a browser, and the page starts in 23 s instead of 127 s.

The kinds are ``atlas_activation`` (the pairs of studies and regions),
``atlas_coverage`` (how much the studies cover the regions),
``mask`` (the MNI mask), ``voxels`` (the brain and cortex voxels) and
``fetched`` (where nilearn put an atlas, and its labels). The files hold
arrays, an image and text only (``.npz``, ``.nii.gz``, ``.json``), so the
terminal REPL and the REPL in the browser use the same ones: a file written by
CPython is read as it is by Pyodide, whatever the versions of numpy and pandas
on each side, and there is no copy to make for each Python implementation. A
cache of the engine's Python objects, which would be one per implementation
and version, is not possible: the expressions of the engine cannot be pickled.

``python -m neurolang_apps.utils.engine_cache ENGINE -r MM`` builds the
engine so that its tables are kept, with the resolution it will be used with
(the build of the browser's bundle does it, and its data zip carries them),
``--list`` tells which are kept and ``--clear`` removes them.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import nibabel as nib
import numpy as np

# The directory of the tables of an engine, in its data directory, and the
# version of what they are made of: raise it when the way they are found
# changes.
DIRECTORY = "engine_cache"
VERSION = 2


def cache_dir(data_dir, engine_name: str) -> Path:
    """Return the directory where the tables of the engine are kept."""
    return Path(data_dir) / engine_name / DIRECTORY


def cached_files(data_dir, engine_name: str) -> List[Path]:
    """Return the files of the tables kept for the engine, sorted."""
    return sorted(
        path for path in cache_dir(data_dir, engine_name).glob("*")
        if path.is_file()
    )


def digest(*parts) -> str:
    """
    Return a name for *parts*: text, numbers and arrays written out the same
    way on any platform, so that the file one build kept is found by another
    (CPython and Pyodide, whatever the versions of numpy). Floats count to the
    micrometre.
    """
    hasher = hashlib.sha256()
    hasher.update(f"{VERSION}".encode())
    for part in parts:
        if isinstance(part, np.ndarray):
            if part.dtype.kind in "iu":
                part = part.astype("<i8")
            elif part.dtype.kind == "f":
                part = np.round(part * 1000).astype("<i8")
            else:
                # the distinct items, in order, as fixed-width UTF-32 little
                # endian, and the position of each: the same bytes anywhere,
                # without a Python string for each of millions of items
                import pandas as pd

                codes, items = pd.factorize(
                    part.ravel(), use_na_sentinel=False
                )
                items = np.asarray(items).astype(str)
                hasher.update(items.astype(
                    f"<U{max(items.dtype.itemsize // 4, 1)}"
                ).tobytes() + f"{part.shape}".encode())
                part = codes.astype("<i8")
            hasher.update(f"{part.shape}".encode() + part.tobytes())
        else:
            hasher.update(repr(part).encode())
    return hasher.hexdigest()[:24]


def grid_digest(img) -> str:
    """Return a name for the grid of an image: its shape and its affine."""
    return digest(img.shape[:3], np.ascontiguousarray(img.affine))


def image_digest(img) -> str:
    """Return a name for what an image holds: its voxels and its affine."""
    data = np.ascontiguousarray(np.asanyarray(img.dataobj))
    return digest(
        str(data.dtype), data.shape, np.ascontiguousarray(img.affine),
        hashlib.sha256(data.tobytes()).hexdigest(),
    )


def _write_directory(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Write the directory *path* of *arrays*, a ``.npy`` file each, in one
    piece or not at all."""
    import shutil

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.mkdir()
        for number, array in enumerate(arrays.values()):
            np.save(temporary / f"{number}.npy", array, allow_pickle=False)
        (temporary / "names.json").write_text(json.dumps(list(arrays)))
        os.replace(temporary, path)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _write(path: Path, write: Callable) -> None:
    """Write the file *path* with *write(file)*, in one piece or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as file:
            write(file)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def keep(directory, name: str, read: Callable, write: Callable, compute):
    """
    Return what the file *name* of *directory* holds, read by *read(path)*,
    or else *compute()*, written by *write(file, value)* for the next time.

    A directory that cannot be written to, or a file that cannot be read,
    only means that the value is computed again. *directory* None means
    there is no cache.
    """
    if directory is None:
        return compute()
    path = Path(directory) / name
    if path.exists():
        try:
            return read(path)
        except Exception:  # damaged, or written by another version
            pass
    value = compute()
    try:
        _write(path, lambda file: write(file, value))
    except (OSError, ValueError, TypeError):
        return value
    try:
        return read(path)  # what the next build reads, as this one sees it
    except Exception:
        return value


def keep_arrays(
    directory, kind: str, key: str, compute: Callable[[], Dict[str, Any]]
) -> Dict[str, np.ndarray]:
    """Return the arrays *compute()* gives, kept in ``kind-key.npz``."""
    def read(path):
        with np.load(path, allow_pickle=False) as stored:
            return {name: stored[name] for name in stored.files}

    def write(file, arrays):
        np.savez_compressed(file, **arrays)

    return keep(directory, f"{kind}-{key}.npz", read, write, compute)


def keep_image(directory, kind: str, key: str, compute: Callable):
    """Return the image *compute()* gives, kept in ``kind-key.nii.gz``."""
    def read(path):
        image = nib.Nifti1Image.from_bytes(gzip.decompress(path.read_bytes()))
        return nib.Nifti1Image(
            np.asanyarray(image.dataobj), image.affine, image.header
        )

    def write(file, image):
        file.write(gzip.compress(image.to_bytes()))

    return keep(directory, f"{kind}-{key}.nii.gz", read, write, compute)


def keep_json(directory, kind: str, key: str, compute: Callable):
    """Return the JSON value *compute()* gives, kept in ``kind-key.json``."""
    def read(path):
        return json.loads(path.read_text())

    def write(file, value):
        file.write(json.dumps(value).encode())

    return keep(directory, f"{kind}-{key}.json", read, write, compute)


#: The directory, in the data directory, of the tables of the builds of the
#: engines (:class:`DerivedTables`): one for all engines, as engines of a
#: family read and derive the same, and not in the browser's bundle.
DERIVED_DIRECTORY = "derived_tables"


def file_digest(path) -> str:
    """Return a name for what the file *path* holds: its bytes."""
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def frame_digest(frame) -> str:
    """Return a name for what the table *frame* holds: its column names,
    their types and every value, row by row."""
    import pandas as pd

    rows = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return digest(
        [str(name) for name in frame.columns],
        [str(dtype) for dtype in frame.dtypes],
        hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest(),
    )


def _arrays_identity(frame):
    """
    Return what identifies the rows of *frame* while the arrays of its
    columns live: the address, offset, size and type of each, in order,
    so that a frame of the same arrays under other column names (the
    relation of a rule renaming a stored one) is recognised; the frame's
    id if an array is not known.
    """
    import pandas as pd

    parts = [len(frame)]
    for name in frame.columns:
        array = frame[name].array
        arrow = getattr(array, "_pa_array", None)
        if arrow is not None:
            parts.append((str(arrow.type),) + tuple(
                (chunk.offset, len(chunk)) + tuple(
                    (buffer.address, buffer.size)
                    for buffer in chunk.buffers() if buffer is not None
                )
                for chunk in arrow.chunks
            ))
        elif isinstance(array, pd.arrays.NumpyExtensionArray):
            values = array.to_numpy()
            parts.append((
                values.dtype.str, values.shape, values.strides,
                values.__array_interface__["data"][0],
            ))
        else:
            return id(frame)
    return tuple(parts)


class DerivedTables:
    """
    The tables of an engine's build, kept in *directory* as Arrow files named
    by a checksum of what each is found from (:func:`frame_digest`,
    :func:`file_digest`, :func:`digest`): those its program derives from its
    source tables (the pairs of points of a Euclidean bound: ``activated``,
    the voxels within reach of each peak) and its large source tables, read
    from files (the peaks, the TF-IDF of the terms, the relation files) or
    found from them (the studies activating the regions of the atlases, the
    bootstrap replicates). A change of the data makes another one, and
    engines finding the same table from the same data (``neurostore``,
    ``neurostore_iit``...) find the same file.

    A table kept is read memory-mapped: in under a second rather than
    computed again (35 s for the ``activated`` of ``neurostore``), its
    columns views of the file, whose pages the processes reading it (the
    engines of the builder of the REPL server and the sessions forked from
    it) share, and which the system may drop and read again when memory is
    short. Assign one to ``program_ir.derived_tables`` (see
    :class:`~neurolang.frontend.datalog.sugar.spatial.TranslateEuclideanDistanceBoundMatrixMixin`
    and :func:`keep_table`).
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        # The checksums of the source tables, by their id, with the table,
        # whose id is then no other's: a table is read once per build (the
        # peaks, which the regions of each atlas are found from).
        self._digests = {}

    def _digest(self, frame) -> str:
        identity = _arrays_identity(frame)
        if identity not in self._digests:
            self._digests[identity] = frame, frame_digest(frame)
        return self._digests[identity][1]

    def __call__(self, kind: str, frames, parameters, compute):
        """Return the table derived as *kind* from the tables *frames* with
        *parameters*: the one kept, or *compute()*, kept for the next time.
        A directory that cannot be written to, or a file that cannot be
        read, only means that it is computed again."""
        key = digest(kind, parameters, *map(self._digest, frames))
        path = self.directory / f"{kind}-{key}.arrow"
        if path.exists():
            try:
                return self._known(self._read(path), key)
            except Exception:  # damaged, or written by another version
                pass
        frame = compute()
        try:
            _write(path, lambda file: self._write(file, frame))
            # mapped: its pages are shared
            return self._known(self._read(path), key)
        except Exception:
            return frame

    def _known(self, frame, key):
        """Return *frame*, the table named *key*: tables found from it
        are named from *key*, without reading its rows again."""
        self._digests[_arrays_identity(frame)] = frame, key
        return frame

    def arrays(self, kind: str, frames, parameters, compute):
        """
        Return the arrays, by name, found as *kind* from the tables
        *frames* with *parameters*: those kept, memory-mapped, or
        *compute()*, kept for the next time (a directory of a ``.npy``
        file per array, the arrays numbers or fixed-width texts). As for
        a table, a directory that cannot be written to, or files that
        cannot be read, only mean that they are computed again.
        """
        key = digest(kind, parameters, *map(self._digest, frames))
        path = self.directory / f"{kind}-{key}"
        if path.is_dir():
            try:
                return self._read_arrays(path)
            except Exception:  # damaged, or written by another version
                pass
        arrays = compute()
        try:
            _write_directory(path, arrays)
            return self._read_arrays(path)
        except Exception:
            return arrays

    @staticmethod
    def _read_arrays(path: Path):
        names = json.loads((path / "names.json").read_text())
        return {
            name: np.load(
                path / f"{number}.npy", mmap_mode="r", allow_pickle=False
            )
            for number, name in enumerate(names)
        }

    @staticmethod
    def _write(file, frame) -> None:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        table = pa.Table.from_pandas(frame, preserve_index=False)
        with ipc.new_file(file, table.schema) as writer:
            writer.write_table(table)

    @staticmethod
    def _read(path: Path):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        table = ipc.open_file(pa.memory_map(str(path))).read_all()
        return table.to_pandas(split_blocks=True)


def keep_table(program, kind: str, frames, parameters: Callable, compute):
    """
    Return the table *compute()* gives, found as *kind* from the tables
    *frames* and the *parameters()* (texts, numbers, :func:`file_digest`...),
    which are all it depends on: the one kept by the ``derived_tables`` of
    *program* (a :class:`DerivedTables`, while an engine is built), or else
    *compute()*, without naming it (in a browser, the parameters are not
    computed).
    """
    tables = getattr(program, "derived_tables", None)
    if tables is None:
        return compute()
    return tables(kind, frames, parameters(), compute)


def prebuild(
    engine_name: str, data_dir, resolution: Optional[float] = None,
    yaml_path=None,
) -> List[Path]:
    """
    Build the engine so that its tables are kept in the data directory, and
    return the files they are in. They are all found and written, none read
    from the Arrow files of the builds (:class:`DerivedTables`), which the
    browser does not read.
    """
    from neurolang_apps.utils import engine_registry

    engine_registry.build_engine(
        engine_name, Path(data_dir), resolution, yaml_path, use_cache=True,
        map_tables=False,
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
