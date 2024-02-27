import configparser
import distutils
import os
import sys
import shutil
import subprocess
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build_py import build_py
from pathlib import Path

# Workaround for editable install with pip
# and versioneer
sys.path.append(os.path.dirname(__file__))

import versioneer


def update_config_file():
    """
    Read the config file in `neurolang/config/config.ini`
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
    Custom develop command.

    - Option to edit the config file and set
    dask-sql as the backend when `--dask` flag is set on install.

    Usage
    -----
    Run
    $ python setup.py develop --dask
    or
    $ pip install -e . --install-option='--dask'

    - Option to build the frontend app using the npm_build command

    Usage
    -----
    Run
    $ python setup.py develop --npm-build
    or
    $ pip install -e . --install-option='--npm-build'

    This will only work when installing a source distribution zip or tarball,
    or installing in editable mode from a source tree. **It will not work when
    installing from a binary wheel.**
    """

    description = "develop option to set dask-sql as backend"

    user_options = develop.user_options + [
        ("dask", None, None),
        ("npm-build", None, "build the frontend html/js app using npm-build"),
    ]

    def initialize_options(self):
        develop.initialize_options(self)
        self.dask = None
        self.npm_build = False

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        import nltk
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        if self.dask:
            update_config_file()
        if self.npm_build:
            self.run_command("npm_build")
        develop.run(self)


class InstallCommand(install):
    """
    Custom install command.

    - Option to edit the config file and set
    dask-sql as the backend when `--dask` flag is set on install.

    Usage
    -----
    Run
    $ python setup.py install --dask
    or
    $ pip install . --install-option='--dask'

    - Option to build the frontend app using the npm_build command

    Usage
    -----
    Run
    $ python setup.py install --npm-build
    or
    $ pip install . --install-option='--npm-build'

    This will only work when installing a source distribution zip or tarball,
    or installing in editable mode from a source tree. **It will not work when
    installing from a binary wheel.**
    """

    description = "install option to set dask-sql as backend"

    user_options = install.user_options + [
        ("dask", None, None),
        ("npm-build", None, "build the frontend html/js app using npm-build"),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.dask = None
        self.npm_build = False

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        if self.dask:
            update_config_file()
        if self.npm_build:
            self.run_command("npm_build")
        install.run(self)


class NPMBuildCommand(distutils.cmd.Command):
    """Run the npm build command"""

    description = "run the npm build command for the web server"
    user_options = [
        ("force", "f", "force npm build even if dist directory already exists")
    ]

    def initialize_options(self):
        """Set default values for options."""
        self.force = False

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run the npm build command"""
        npm_command = shutil.which("npm")
        if not npm_command:
            raise OSError(
                "Could not find the npm binary required to build the frontend "
                "app. NPM is Node.js' package manager and can be installed "
                "with node from https://nodejs.org/en/download/."
            )

        web_dir = (
            Path(__file__).resolve().parent
            / "neurolang"
            / "utils"
            / "server"
            / "neurolang-web"
        )
        if not web_dir.exists():
            raise OSError(f"{web_dir} directory does not exist.")

        dist_dir = web_dir / "dist"
        if dist_dir.exists() and not self.force:
            self.announce(
                f"Build directory already exists for frontend app. Not rebuilding.",
                level=distutils.log.INFO,
            )
            return

        command = [npm_command, "install"]
        self.announce(
            f"Running command: [{web_dir}]$ {' '.join(command)}",
            level=distutils.log.INFO,
        )
        subprocess.check_call(
            command, cwd="neurolang/utils/server/neurolang-web"
        )

        command = [npm_command, "run", "build", "--", "--mode", "dev"]
        self.announce(
            f"Running command: [{web_dir}]$ {' '.join(command)}",
            level=distutils.log.INFO,
        )
        subprocess.check_call(
            command, cwd="neurolang/utils/server/neurolang-web"
        )


class BuildPyCommand(build_py):
    """Customize the build command and add the npm build"""

    def run(self):
        """Run the npm build before the normal build"""
        try:
            self.run_command("npm_build")
        except (OSError, subprocess.CalledProcessError) as e:
            self.announce(
                f"Skipping build of frontend app because: {e}.",
                level=distutils.log.WARN,
            )
        build_py.run(self)


if __name__ == "__main__":
    version = (versioneer.get_version(),)
    cmdclass = (versioneer.get_cmdclass(),)
    setup(
        use_scm_version=True,
        packages=find_packages(),
        cmdclass={
            "install": InstallCommand,
            "develop": DevelopCommand,
            "npm_build": NPMBuildCommand,
            "build_py": BuildPyCommand,
        },
    )
