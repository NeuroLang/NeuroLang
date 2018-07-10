from collections import OrderedDict
from itertools import chain
import inspect
from inspect import isclass
import logging
from typing import Tuple, TypeVar

from . import expressions
from .symbols_and_types import replace_type_variable

__all__ = ['add_match', 'PatternMatcher']


class PatternMatchingMetaClass(expressions.ParametricTypeClassMeta):
    @classmethod
    def __prepare__(self, name, bases):
        return OrderedDict()

    def __new__(cls, name, bases, classdict):
        classdict['__ordered__'] = [
            key for key in classdict.keys()
            if key not in ('__module__', '__qualname__')
        ]
        patterns = []
        if (
            '__generic_class__' in classdict and
            hasattr(classdict['__generic_class__'], 'type') and
            isinstance(classdict['__generic_class__'].type, TypeVar)
        ):
            needs_replacement = True
            src_type = classdict['__generic_class__'].type
            dst_type = classdict['type']
        else:
            needs_replacement = False

        for k, v in classdict.items():
            if callable(v) and hasattr(v, 'pattern') and hasattr(v, 'guard'):
                pattern = getattr(v, 'pattern')
                if needs_replacement:
                    pattern = __pattern_replace_type__(
                        pattern, src_type, dst_type
                    )
                patterns.append(
                    (pattern, getattr(v, 'guard'), v)
                )
        classdict['__patterns__'] = patterns
        return type.__new__(cls, name, bases, classdict)


def __pattern_replace_type__(pattern, src_type, dst_type):
    if (
        isclass(pattern) and
        issubclass(pattern, expressions.Expression) and
        hasattr(pattern, '__generic_class__')
    ):
        pattern = pattern.__generic_class__[
            replace_type_variable(
                dst_type, pattern.type, type_var=src_type
            )
        ]
    elif isinstance(pattern, expressions.Expression):
        print(f"Replace {pattern}")
        parameters = inspect.signature(
            pattern.__class__
        ).parameters
        args = []

        for argname, arg in parameters.items():
            if arg.default != inspect._empty:
                continue
            args.append(
                __pattern_replace_type__(
                    getattr(pattern, argname),
                    src_type,
                    dst_type
                )
            )

        pattern_class = type(pattern)
        if hasattr(pattern_class, '__generic_class__'):
            pattern_class = pattern_class[
                replace_type_variable(
                    dst_type, pattern.type, type_var=src_type
                )
            ]
        pattern = pattern_class(*args)
    elif isinstance(pattern, tuple):
        pattern = tuple(
            __pattern_replace_type__(p, src_type, dst_type)
            for p in pattern
        )
    return pattern


def add_match(pattern, guard=None):
    '''Decorator adding patterns to a :class:`PatternMatcher` class.

    Should be used as
    `@add_match(PATTERN, GUARD)` to turn the decorated method, receiving
    an instance of :class:`Expression` into a matching case. See
    :py:meth:`PatternMatcher.pattern_match` for more details on patterns.
    '''
    def bind_match(f):
        f.pattern = pattern
        f.guard = guard
        return f
    return bind_match


class PatternMatcher(metaclass=PatternMatchingMetaClass):
    '''Class for expression pattern matching.
    '''
    def __init__(self):
        pass

    @property
    def patterns(self):
        '''Property holding an iterator of triplets ``(pattern, guard, action)``

            - ``pattern``: is an Expression class, or instance where
              construction parameters and the type can be replaced
              by an ellipsis `...` to signal a wildcard. See
              :py:meth:`pattern_match` for more details.
            - ``guard``: is a function mapping an `Expression` instance
              to a boolean or ``None``.
            - ``action``: is the method receiving the matching ``expression``
              instance to be executed upon pattern and match being ``True``.
        '''
        return chain(*(
                pm.__patterns__ for pm in self.__class__.mro()
                if hasattr(pm, '__patterns__')
        ))

    def match(self, expression):
        '''Find the action for a given expression by going through the ``patterns``.
        Goes through the triplets in in ``patterns`` and calls the action
        specified by the first satisfied triplet.
        '''
        for pattern, guard, action in self.patterns:
            if logging.getLogger().getEffectiveLevel() >= logging.DEBUG:
                pattern_match = self.pattern_match(pattern, expression)
                guard_match = pattern_match and (
                    guard is None or guard(expression)
                )
                logging.debug("test {} {}:{} | {}:{}".format(
                    self.__class__.__name__, pattern, pattern_match,
                    guard, guard_match
                ))

            if (
                self.pattern_match(pattern, expression) and
                (guard is None or guard(expression))
            ):
                logging.debug("**** match {} | {}".format(pattern, guard))
                return action(self, expression)
        else:
            raise ValueError()

    def pattern_match(self, pattern, expression):
        '''Returns ``True`` if ``pattern`` matches ``expression``.

        Patterns are of the following form:

        - ``...`` matches everything.
        - :class:`Expression` class matches if ``expression`` is
          instance of the given class for any ``type``.
        - :class:`Expression` ``[T]`` matches if ``expression`` is
          instance of the given class with ``expression.type``
          a subtype of ``T``.
        - :class:`Expression` ``[T](Pattern_0, Pattern_1, .....,
          Pattern_N)`` matches if
          ``expression`` is instance of the given class with
          ``expression.type`` a subtype of ``T`` and recursively matches the
          construction parameters of the instance on the types given as
          ``Pattern_0``, ``Pattern_1``, etc.
        - ``(Pattern_0, ....., Pattern_N)`` matches when expression is a
          ``tuple`` of length `N + 1` where ``expression[0]`` matches
          ``Pattern_0``, ``expression[1]`` matches ``Pattern_1``, etc.
        - ``instance`` an instance of a python class not subclassing
          :class:`Expression` matches when
          ``instance == expression``
        '''
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
