import configparser
import os
import sys
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from pathlib import Path

# Workaround for editable install with pip
# and versioneer
sys.path.append(os.path.dirname(__file__))

import versioneer


def update_config_file():
    """
    Read the config file in `neurolang/utils/config/config.ini`
    and update the value for ["RAS"]["Backend"] to dask.
    This config file will then be copied during install as a
    setuptools data_files.
    """
    config_file = Path("neurolang/config/config.ini")
    config = configparser.ConfigParser(
        allow_no_value=True, comment_prefixes="//"
    )
    config.optionxform = str
    config.read(config_file)
    config["RAS"]["backend"] = "dask"
    with open(config_file, "w") as configfile:
        config.write(configfile)


class DevelopCommand(develop):
    """
    Custom develop option to edit the config file and set
    dask-sql as the backend when `--dask` flag is set on install.

    Usage
    -----
    Run
    $ python setup.py develop --dask
    or
    $ pip install -e . --install-option='--dask'

    This will only work when installing a source distribution zip or tarball,
    or installing in editable mode from a source tree. **It will not work when
    installing from a binary wheel.**
    """

    description = "develop option to set dask-sql as backend"

    user_options = develop.user_options + [
        ("dask", None, None),
    ]

    def initialize_options(self):
        develop.initialize_options(self)
        self.dask = None

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        if self.dask:
            update_config_file()
        develop.run(self)


class InstallCommand(install):
    """
    Custom install option to edit the config file and set
    dask-sql as the backend when `--dask` flag is set on install.

    Usage
    -----
    Run
    $ python setup.py install --dask
    or
    $ pip install . --install-option='--dask'

    This will only work when installing a source distribution zip or tarball,
    or installing in editable mode from a source tree. **It will not work when
    installing from a binary wheel.**
    """

    description = "install option to set dask-sql as backend"

    user_options = install.user_options + [
        ("dask", None, None),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.dask = None

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        if self.dask:
            update_config_file()
        install.run(self)


if __name__ == "__main__":
    version = (versioneer.get_version(),)
    cmdclass = (versioneer.get_cmdclass(),)
    setup(
        use_scm_version=True,
        packages=find_packages(),
        cmdclass={
            "install": InstallCommand,
            "develop": DevelopCommand,
        },
    )
