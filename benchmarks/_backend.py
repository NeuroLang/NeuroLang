"""Select the relational algebra set backend for a benchmark."""
import importlib
import sys

BACKENDS = ["pandas", "polars"]


def use_backend(backend):
    """
    Select ``backend`` and re-import NeuroLang so that it uses it.

    Many NeuroLang modules bind the relational algebra set classes of the
    configured backend when they are imported, and
    ``config.set_query_backend`` only reloads a few of them.  asv imports the
    benchmark modules, and with them NeuroLang, before ``setup`` runs, so
    every ``neurolang`` module except the configuration is dropped from
    ``sys.modules`` and imported again on demand.  Each asv benchmark runs in
    its own process, so this does not leak into other benchmarks.

    Parameters
    ----------
    backend : str
        ``"pandas"`` or ``"polars"``.

    Raises
    ------
    NotImplementedError
        If the backend's dependencies are not installed; asv then skips the
        benchmark.

    """
    if backend == "polars":
        try:
            importlib.import_module("polars")
        except ImportError:
            raise NotImplementedError("polars is not installed")
    from neurolang.config import config

    config.set_query_backend(backend)
    # set_query_backend switches the join ordering heuristic off for polars
    # but does not switch it back on, so set it explicitly.
    config["CHASE"]["sort_join_predicates"] = str(backend != "polars")
    stale = [
        name for name in sys.modules
        if (name == "neurolang" or name.startswith("neurolang."))
        and not name.startswith("neurolang.config")
    ]
    for name in stale:
        del sys.modules[name]
