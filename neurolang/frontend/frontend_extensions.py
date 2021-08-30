from typing import Callable
import numpy as np
from ..expressions import Constant

class NumpyFunctionsMixin():
    """
    This Mixin adds common mathematical functions in numpy to a
    NeurolangFrontend instance's DatalogProgram.

    The constant_X fields will be made available as functions to
    the frontend.
    """

    constant_exp = Constant[Callable[[float], float]](np.exp)
    constant_log = Constant[Callable[[float], float]](np.log)
    constant_log10 = Constant[Callable[[float], float]](np.log10)
    constant_cos = Constant[Callable[[float], float]](np.cos)
    constant_sin = Constant[Callable[[float], float]](np.sin)
    constant_tan = Constant[Callable[[float], float]](np.tan)