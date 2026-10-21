"""
NeuroLang's expression engine: the expressions of its intermediate
representation, their types, the pattern-matching walkers that rewrite
them, and first-order logic.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
