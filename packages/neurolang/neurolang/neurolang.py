from __future__ import absolute_import, division, print_function

import importlib

from nl_expr.config import config
from nl_expr import exceptions

from . import _aliases

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError


try:
    __version__ = version("neurolang")
except PackageNotFoundError:
    __version__ = None
    # package is not installed

# The frontends are imported when first used, so that importing the
# expression engine or the Datalog core, e.g. ``neurolang.expressions``,
# imports neither the frontends nor their parsing and neuroimaging
# libraries.
_FRONTEND_NAMES = (
    "NeurolangDL",
    "NeurolangPDL",
    "ExplicitVBR",
    "ExplicitVBROverlay",
)

__all__ = [
    "__version__",
    *_FRONTEND_NAMES,
    "exceptions",
    "config",
]


def __getattr__(name):
    if name in _FRONTEND_NAMES:
        from . import frontend

        return getattr(frontend, name)
    if name in _aliases.MOVED:
        return importlib.import_module(f"neurolang.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
