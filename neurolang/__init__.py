# ``__getattr__`` imports the frontends when first used.
from .neurolang import __all__, __getattr__, config, exceptions  # noqa: F401

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version('neurolang')
except PackageNotFoundError:
    __version__ = 'unknown'
