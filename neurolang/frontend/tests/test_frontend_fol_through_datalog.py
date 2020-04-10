from unittest.mock import patch
from .. import RegionFrontendFolThroughDatalog
from . import test_frontend as tf


patch_frontend = patch(
    "neurolang.frontend.RegionFrontend", new=RegionFrontendFolThroughDatalog
)


@patch_frontend
def test_tuple_symbol_multiple_types_query():
    tf.test_tuple_symbol_multiple_types_query()


@patch_frontend
def test_quantifier_expressions():
    tf.test_quantifier_expressions()


#
# current_module = __import__(__name__)
#
# Programmatically copy the tests from test_frontend using
# RegionFrontendFolThroughDatalog instead of RegionFrontend
# def borrow_tests():
#     from . import test_frontend
#
#     for name in dir(test_frontend):
#         if not name.startswith("test_"):
#             continue
#         test = getattr(test_frontend, name)
#         if not callable(test):
#             continue
#
#         def test_wrapper():
#             with patch(
#                 "neurolang.frontend.RegionFrontend",
#                 new=RegionFrontendFolThroughDatalog,
#             ):
#                 test()
#
#         setattr(current_module, name, test_wrapper)
#
# borrow_tests()
#
