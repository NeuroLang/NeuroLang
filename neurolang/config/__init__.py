import configparser
import logging
import os
import sys
from importlib import reload

LOG = logging.getLogger(__name__)


class NeurolangConfigParser(configparser.ConfigParser):
    def set_query_backend(self, backend):
        """
        Convenience method to set the backend used by Neurolang.
        """
        valid_backends = {"pandas", "dask"}
        if backend not in valid_backends:
            raise ValueError(
                f"The query backend option should be one of {valid_backends}"
            )
        LOG.info(f"Setting new query backend : {backend}")
        self["RAS"]["backend"] = backend
        self.switch_backend()

    def switch_backend(self):
        """
        Changing the backend value in the config object is not enough to
        switch the backends since this config value is evaluated at import
        time in some modules to figure out which classes to import.
        So we need to reimport these modules after changing the backend value
        in the config.
        """
        modules = [
            "neurolang.probabilistic.weighted_model_counting",
            "neurolang.utils.relational_algebra_set",
            "neurolang.utils",
        ]
        for module in modules:
            module = sys.modules.get(module)
            if module is not None:
                reload(module)
                LOG.debug(f"Reloading module : {module}")


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
