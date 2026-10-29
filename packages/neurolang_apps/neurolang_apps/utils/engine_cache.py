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
VERSION = 1


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
                part = np.array([str(item) for item in part.ravel()])
                hasher.update(repr(part.tolist()).encode())
                continue
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


def prebuild(
    engine_name: str, data_dir, resolution: Optional[float] = None,
    yaml_path=None,
) -> List[Path]:
    """
    Build the engine so that its tables are kept in the data directory, and
    return the files they are in.
    """
    from neurolang_apps.utils import engine_registry

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
