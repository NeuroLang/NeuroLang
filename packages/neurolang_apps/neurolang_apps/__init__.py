"""
NeuroLang's applications: the command line, the terminal interface, the
query assistant and the engines of predefined data.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
