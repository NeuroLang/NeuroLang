"""
NeuroLang's core: relational algebra and its sets, Datalog with its chase,
and probabilistic Datalog with its solvers, on the expression engine
:mod:`nl_expr`.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
