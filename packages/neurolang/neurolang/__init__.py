# The modules moved to nl_expr, nl_core and neurolang_apps keep their names
# under neurolang: see _aliases.
from . import _aliases

_aliases.install()

# Import the module neurolang.config first: importing it sets the
# attribute ``config`` of this package, which is the configuration.
from . import config as _config_module  # noqa: E402,F401

# ``__getattr__`` imports the frontends, and the moved modules, when first
# used.
from .neurolang import (  # noqa: E402,F401
    __all__,
    __getattr__,
    config,
    exceptions,
)

from importlib.metadata import version, PackageNotFoundError  # noqa: E402
try:
    __version__ = version('neurolang')
except PackageNotFoundError:
    __version__ = 'unknown'
