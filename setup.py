from setuptools import setup
import versioneer

if __name__ == "__main__":
    version = versioneer.get_version(),
    cmdclass = versioneer.get_cmdclass(),
    setup(use_scm_version=True)
