from typing import Callable
import numpy as np
from nl_expr.expressions import Constant
from ..utils.euclidean_join import euclidean_distance

class NumpyFunctionsMixin():
    """
    This Mixin adds common mathematical functions in numpy to a
    NeurolangFrontend instance's DatalogProgram.

    The constant_X fields will be made available as functions to
    the frontend. ``euclidean`` is the Euclidean distance between two
    points given by their coordinates, those of the first point followed by
    those of the second one (see
    :func:`~neurolang.utils.euclidean_join.euclidean_distance`); bounding
    it by a number in a rule is translated by
    :class:`~neurolang.frontend.datalog.sugar.spatial.TranslateEuclideanDistanceBoundMatrixMixin`.
    """

    constant_exp = Constant[Callable[[float], float]](np.exp)
    constant_log = Constant[Callable[[float], float]](np.log)
    constant_log10 = Constant[Callable[[float], float]](np.log10)
    constant_cos = Constant[Callable[[float], float]](np.cos)
    constant_sin = Constant[Callable[[float], float]](np.sin)
    constant_tan = Constant[Callable[[float], float]](np.tan)
    constant_euclidean = Constant[
        Callable[[float, float, float, float, float, float], float]
    ](euclidean_distance, verify_type=False)