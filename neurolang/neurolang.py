from __future__ import absolute_import, division, print_function

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

from .exceptions import NeuroLangException
from .expression_pattern_matching import *
from .expression_walker import *
from .expression_walker import (ExpressionBasicEvaluator, PatternMatcher,
                                add_match)
from .expressions import *
from .expressions import (Constant, Expression, FunctionApplication, Lambda,
                          NeuroLangTypeException, Projection, Query, Statement,
                          Symbol, TypedSymbolTable, Unknown,
                          is_leq_informative, unify_types)
from .neurolang_compiler import NeuroLangIntermediateRepresentationCompiler
from .solver import *


try:
    __version__ = version('neurolang')
except PackageNotFoundError:
    __version__ = None
    # package is not installed
    pass


__all__ = [
    '__version__',
    'add_match',
    'NeuroLangException',
    'PatternMatcher',
    'add_match', 'NeuroLangIntermediateRepresentationCompiler', 'Expression',
    'Constant', 'Symbol', 'FunctionApplication', 'Lambda', 'Projection',
    'Statement', 'Query', 'ExistentialPredicate', 'UniversalPredicate',
    'TypedSymbolTable', 'Unknown', 'is_leq_informative', 'unify_types',
    'ExpressionBasicEvaluator', 'NeuroLangTypeException'
]
