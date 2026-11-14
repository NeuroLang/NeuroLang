"""
Writing the dataset: matrices in parquet, and the long tables of the
NeuroLang relations in gzipped TSV.
"""

from pathlib import Path

import pandas as pd

# gzip without the time, so that a file made again is the same
GZIP = {"method": "gzip", "mtime": 0}


def matrix(values, regions: list) -> pd.DataFrame:
    """The square matrix *values* over *regions*, its index ``region``."""
    index = pd.Index(regions, name="region")
    return pd.DataFrame(values, index=index, columns=list(index))


def write_matrix(frame: pd.DataFrame, path: Path) -> None:
    """Write the matrix *frame* in parquet, its 64-bit values whole: zstd
    on their bytes split by significance (byte_stream_split), which
    compresses averages, whose last bits are noise, as numpy's compressed
    npz does."""
    frame.to_parquet(path, compression="zstd", use_dictionary=False,
                     use_byte_stream_split=list(frame.columns))


def pairs(frame: pd.DataFrame, rows, columns, names, value: str,
          decimals: int) -> pd.DataFrame:
    """The long table of *frame* between *rows* and *columns*, both ways
    round when they are the same regions, the diagonal, the NaN and the
    values rounding to 0 left out: columns *names* and *value*."""
    block = frame.loc[rows, columns].round(decimals).stack().reset_index()
    block.columns = [*names, value]
    return block[(block[names[0]] != block[names[1]]) & (block[value] != 0)]


def write_tsv(table: pd.DataFrame, path: Path, float_format=None) -> None:
    table.to_csv(path, sep="\t", index=False, compression=GZIP,
                 float_format=float_format)
