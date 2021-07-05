import configparser
import logging
import os
import sys

LOG = logging.getLogger(__name__)


class NeurolangConfigParser(configparser.ConfigParser):
    def set_backend(self, backend):
        """
        Convenience method to set the backend used by Neurolang.
        """
        valid_backends = {"pandas", "dask"}
        assert (
            backend in valid_backends
        ), f"The backend option should be one of {valid_backends}"
        self["RAS"]["backend"] = backend


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
