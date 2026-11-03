import configparser
import logging
import os
import sys
from importlib import reload

LOG = logging.getLogger(__name__)

VALID_QUERY_BACKENDS = ("pandas", "polars")
REMOVED_QUERY_BACKENDS = {
    "dask": "the dask-sql backend was removed from NeuroLang",
}


def check_query_backend(backend):
    """
    Check that ``backend`` names an available relational algebra set backend.

    Parameters
    ----------
    backend : str
        Name of the backend, e.g. ``"pandas"`` or ``"polars"``.

    Raises
    ------
    ValueError
        If ``backend`` is not one of ``VALID_QUERY_BACKENDS``. Backends that
        were removed get an error message saying so.

    """
    if backend in VALID_QUERY_BACKENDS:
        return
    valid = ", ".join(repr(b) for b in VALID_QUERY_BACKENDS)
    if backend in REMOVED_QUERY_BACKENDS:
        raise ValueError(
            f"Query backend {backend!r} is no longer available: "
            f"{REMOVED_QUERY_BACKENDS[backend]}. "
            f"Use one of {valid} instead."
        )
    raise ValueError(
        f"Unknown query backend {backend!r}. "
        f"The query backend option should be one of {valid}."
    )


class NeurolangConfigParser(configparser.ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backend_dependent_modules = []

    def set_query_backend(self, backend):
        """
        Convenience method to set the backend used by Neurolang.
        """
        check_query_backend(backend)
        LOG.info(f"Setting new query backend : {backend}")
        self["RAS"]["backend"] = backend
        # Polars has its own query optimizer — disable the chase's
        # manual join ordering heuristic to avoid unnecessary len() calls
        # that would force materialization of lazy frames.
        if backend == "polars":
            self["CHASE"]["sort_join_predicates"] = "False"
        self.switch_backend()

    def reload_on_backend_switch(self, module_name):
        """
        Register the module ``module_name`` to be reloaded when the backend
        changes, since it chooses the classes of the backend when imported.
        Modules are reloaded in the order they registered.
        """
        if module_name not in self._backend_dependent_modules:
            self._backend_dependent_modules.append(module_name)

    def switch_backend(self):
        """
        Changing the backend value in the config object is not enough to
        switch the backends since this config value is evaluated at import
        time in some modules to figure out which classes to import.
        So we need to reimport these modules, which registered with
        :meth:`reload_on_backend_switch`, after changing the backend value
        in the config.
        """
        for module in list(self._backend_dependent_modules):
            module = sys.modules.get(module)
            if module is not None:
                reload(module)
                LOG.debug(f"Reloading module : {module}")

    def enable_expression_type_printing(self):
        self.set("DEFAULT", "expressionTypePrinting", "True")

    def disable_expression_type_printing(self):
        self.set("DEFAULT", "expressionTypePrinting", "False")

    def expression_type_printing(self):
        return self["DEFAULT"].get("expressionTypePrinting", "True") == "True"

    def switch_expression_type_printing(self):
        old_value = self.expression_type_printing()
        self["DEFAULT"]["expressionTypePrinting"] = str(not old_value)
        return self["DEFAULT"]["expressionTypePrinting"] == "True"

    def set_structural_knowledge_namespace(self, name):
        self.set("DEFAULT", "structural_knowledge_namespace", name)

    def get_structural_knowledge_namespace(self):
        return self["DEFAULT"].get("structural_knowledge_namespace", "neurolang:")

    def get_probabilistic_solver_check_unate(self):
        return self.getboolean("PROBABILISTIC_SOLVER", "check_unate", fallback=True)

    def enable_probabilistic_solver_check_unate(self):
        self.set("PROBABILISTIC_SOLVER", "check_unate", "True")

    def disable_probabilistic_solver_check_unate(self):
        self.set("PROBABILISTIC_SOLVER", "check_unate", "False")

    def get_chase_max_iterations(self):
        return self.getint("CHASE", "max_iterations", fallback=10000)

    def set_chase_max_iterations(self, value):
        self.set("CHASE", "max_iterations", str(value))

    def get_chase_sort_join_predicates(self):
        return self.getboolean("CHASE", "sort_join_predicates", fallback=True)


config = NeurolangConfigParser()

config_dirs = [
    os.path.join(sys.prefix, "config"),
    os.path.dirname(os.path.realpath(__file__)),
]
for d in config_dirs:
    config_file = os.path.join(d, "config.ini")
    if os.path.isfile(config_file):
        LOG.info(f"Reading configuration file for Neurolang: {config_file}")
        config.read(config_file)
        LOG.info(f"Read config file with sections {config.sections()}")
        break
