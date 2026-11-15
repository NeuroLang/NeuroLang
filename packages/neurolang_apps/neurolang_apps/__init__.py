"""
NeuroLang's applications: the command line, the terminal interface, the
query assistant and the engines of predefined data.
"""
import os
import sys
from importlib.metadata import PackageNotFoundError, version

# Apple's Accelerate, the BLAS of numpy on macOS, crashes a forked child
# (SIGSEGV) once the parent has run it with several threads; the query
# assistant checks its queries in forked children (utils/llm/validation.py).
# One thread keeps them safe, if set before numpy loads, as the REPL and the
# server import this package first. A value of the environment is kept.
_BLAS_THREADS = os.environ.get("VECLIB_MAXIMUM_THREADS")
if sys.platform == "darwin":
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# Whether a forked child may run numpy's BLAS (validation._in_child).
BLAS_FORK_SAFE = (
    sys.platform != "darwin" or "numpy" not in sys.modules
    or _BLAS_THREADS == "1"
)

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
