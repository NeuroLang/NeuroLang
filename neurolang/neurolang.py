from __future__ import absolute_import, division, print_function

from neurolang.frontend import (
    ExplicitVBR,
    ExplicitVBROverlay,
    NeurolangDL,
    NeurolangPDL
)

from . import exceptions

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError


try:
    __version__ = version("neurolang")
except PackageNotFoundError:
    __version__ = None
    # package is not installed


__all__ = [
    "__version__",
    "NeurolangDL",
    "NeurolangPDL",
    "ExplicitVBR",
    "ExplicitVBROverlay",
    "exceptions",
]
