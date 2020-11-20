import os
import sys
from setuptools import find_packages, setup

# Workaround for editable install with pip
# and versioneer
sys.path.append(os.path.dirname(__file__))

import versioneer

if __name__ == "__main__":
    version = versioneer.get_version(),
    cmdclass = versioneer.get_cmdclass(),
    setup(use_scm_version=True, packages=find_packages())
