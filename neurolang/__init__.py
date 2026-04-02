from .neurolang import *

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version('neurolang')
except PackageNotFoundError:
    __version__ = 'unknown'
