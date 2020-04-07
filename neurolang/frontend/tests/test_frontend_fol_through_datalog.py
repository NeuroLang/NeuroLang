from unittest.mock import patch

from . import test_frontend
from .. import RegionFrontendFolThroughDatalog


current_module = __import__(__name__)

# Programmatically copy the tests from test_frontend using
# RegionFrontendFolThroughDatalog instead of RegionFrontend
for name in dir(test_frontend):
    if not name.startswith("test_"):
        continue
    test = getattr(test_frontend, name)
    if not callable(test):
        continue

    def test_wrapper():
        with patch(
            "neurolang.frontend.RegionFrontend",
            new=RegionFrontendFolThroughDatalog,
        ):
            test()

    setattr(current_module, name, test_wrapper)
