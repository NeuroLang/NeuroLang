import logging
import time
from contextlib import contextmanager
from itertools import chain, combinations

import numpy as np


@contextmanager
def log_performance(
    logger, init_message, init_args=None,
    end_message=None, end_args=None,
    level=logging.INFO
):
    """Context manager to log the performance
    of executed commands in the context.

    Parameters
    ----------
    logger : logging.Logger
        Logger to use for the message
    init_message : str
        Message to display before executing the
        code within the context.
    init_args : tuple, optional
        Tuple with the arguments for the init
        message, by default None
    end_message : str, optional
        Message to display when code has finished
        first parameter is the elapsed seconds,
        by default None
    end_args : tuple, optional
        more arguments for the end message,
        by default None
    level : logging level, optional
        level to log, by default logging.INFO
    """
    if init_args is None:
        init_args = tuple()
    start = time.perf_counter()
    logger.log(level, init_message, *init_args)
    yield start
    end = time.perf_counter()
    lapse = end - start
    if end_message is None:
        end_message = '\t%2.2fs'
    if end_args is None:
        end_args = tuple()
    end_args = (lapse,) + end_args
    logger.log(level, end_message, *end_args)


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class FrozenNDArray(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj.setflags(write=False)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.setflags(write=False)

    def __hash__(self):
        return hash(self.tobytes())

    def __eq__(self, other):
        if isinstance(other, (np.ndarray, FrozenNDArray)):
            return np.array_equal(self, other)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, (np.ndarray, FrozenNDArray)):
            return not np.array_equal(self, other)
        return NotImplemented
