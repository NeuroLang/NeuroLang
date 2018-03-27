from collections import OrderedDict
from itertools import chain
import inspect
from inspect import isclass
import logging
from typing import Tuple

from . import expressions


__all__ = ['add_match', 'PatternMatcher']


class PatternMatchingMetaClass(type):
    @classmethod
    def __prepare__(self, name, bases):
        return OrderedDict()

    def __new__(self, name, bases, classdict):
        classdict['__ordered__'] = [
            key for key in classdict.keys()
            if key not in ('__module__', '__qualname__')
        ]
        patterns = []
        for k, v in classdict.items():
            if callable(v) and hasattr(v, 'pattern') and hasattr(v, 'guard'):
                patterns.append(
                    (getattr(v, 'pattern'), getattr(v, 'guard'), v)
                )
        classdict['__patterns__'] = patterns
        return type.__new__(self, name, bases, classdict)


def add_match(pattern, guard=None):
    def bind_match(f):
        f.pattern = pattern
        f.guard = guard
        return f
    return bind_match


class PatternMatcher(object, metaclass=PatternMatchingMetaClass):
    def __init__(self):
        pass

    @property
    def patterns(self):
        return chain(*(
                pm.__patterns__ for pm in self.__class__.mro()
                if hasattr(pm, '__patterns__')
        ))

    def match(self, expression):
        for pattern, guard, action in self.patterns:
            if logging.getLogger().getEffectiveLevel() >= logging.DEBUG:
                pattern_match = self.pattern_match(pattern, expression)
                guard_match = pattern_match and (
                    guard is None or guard(expression)
                )
                logging.debug("test {}:{} | {}:{}".format(
                    pattern, pattern_match,
                    guard, guard_match
                ))

            if (
                self.pattern_match(pattern, expression) and
                (guard is None or guard(expression))
            ):
                logging.debug("* match {} | {}".format(pattern, guard))
                return action(self, expression)
        else:
            raise ValueError()

    def pattern_match(self, pattern, expression):
        if pattern is ...:
            return True
        elif isclass(pattern):
            if issubclass(pattern, expressions.Expression):
                logging.debug("Match type")
                return isinstance(expression, pattern)
            else:
                raise ValueError(
                    'Class pattern matching only implemented '
                    'for Expression subclasses'
                )
        elif isinstance(pattern, expressions.Expression):
            logging.debug("Match expression instance")
            if not isinstance(expression, pattern.__class__):
                return False
            elif isclass(pattern.type) and issubclass(pattern.type, Tuple):
                logging.debug("Match tuple")
                if (
                    isclass(expression.type) and
                    issubclass(expression.type, Tuple)
                ):
                    if (
                        len(pattern.type.__args__) !=
                        len(expression.type.__args__)
                    ):
                        return False
                    for p, e in zip(pattern.value, expression.value):
                        if not self.pattern_match(p, e):
                            return False
                    else:
                        return True
                else:
                    return False
            else:
                parameters = inspect.signature(pattern.__class__).parameters
                logging.debug("Match parameters {}".format(parameters))
                for argname, arg in parameters.items():
                    if arg.default != inspect._empty:
                        continue
                    p = getattr(pattern, argname)
                    e = getattr(expression, argname)
                    match = self.pattern_match(p, e)
                    logging.debug("\t {} vs {}: {}".format(p, e, match))
                    if not match:
                        return False
                else:
                    return True
        elif isinstance(pattern, tuple) and isinstance(expression, tuple):
            logging.debug("Match tuples {} vs {}".format(pattern, expression))

            if len(pattern) != len(expression):
                return False
            for p, e in zip(pattern, expression):
                if not self.pattern_match(p, e):
                    return False
            else:
                return True
        else:
            logging.debug("Match other {} vs {}".format(pattern, expression))
            return pattern == expression
