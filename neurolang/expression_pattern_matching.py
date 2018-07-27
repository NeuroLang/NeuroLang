from collections import OrderedDict
import copy
from itertools import chain
import inspect
from inspect import isclass
import logging
from typing import Tuple, TypeVar
import types
from warnings import warn

from . import expressions
from .symbols_and_types import replace_type_variable

__all__ = ['add_match', 'PatternMatcher']



class NeuroLangPatternMatchingNoMatch(expressions.NeuroLangException):
    pass


class PatternMatchingMetaClass(expressions.ParametricTypeClassMeta):
    @classmethod
    def __prepare__(self, name, bases):
        return OrderedDict()

    def __new__(cls, name, bases, classdict):
        overwriteable_properties = (
            '__module__', '__patterns__', '__doc__',
            'type', 'plural_type_name', '__no_explicit_type__',
            '__generic_class__',
        )

        for base in bases:
            repeated_methods = set(dir(base)).intersection(classdict)
            repeated_methods.difference_update(overwriteable_properties)
            if '__init__' in repeated_methods:
                if getattr(base, '__init__') is object.__init__:
                    repeated_methods.remove('__init__')
            if len(repeated_methods) > 1:
                warn_message = (
                    f"Warning in class {name} "
                    f"overwrites {repeated_methods} from base {base}"
                )
                warn(warn_message)

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

        new_cls = super().__new__(cls, name, bases, classdict)
        if needs_replacement:
            for attribute_name in dir(new_cls):
                attribute = getattr(new_cls, attribute_name, None)
                if (
                    attribute is None or
                    not hasattr(attribute, '__annotations__')
                ):
                    continue
                if isinstance(
                    attribute,
                    (types.FunctionType, types.MethodType)
                ):
                    new_attribute = types.FunctionType(
                        attribute.__code__, attribute.__globals__,
                        name=attribute.__name__,
                        argdefs=attribute.__defaults__,
                        closure=attribute.__closure__
                    )
                else:
                    new_attribute = copy.copy(attribute)
                annotations = getattr(attribute, '__annotations__')
                if annotations:
                    new_annotations = {
                        k: replace_type_variable(
                            dst_type, v, type_var=src_type
                        )
                        for k, v in annotations.items()
                    }
                    setattr(new_attribute, '__annotations__', new_annotations)
                    setattr(new_cls, attribute_name, new_attribute)

        return new_cls


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
        parameters = inspect.signature(
            pattern.__class__
        ).parameters
        args = []

        for argname, arg in parameters.items():
            if arg.default is not inspect.Parameter.empty:
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
                logging.info(
                    f"**** match {pattern} | {guard} with {expression}"
                )
                return action(self, expression)
        else:
            raise NeuroLangPatternMatchingNoMatch(f'No match for {expression}')

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
        logging.debug(f"Match try {expression} with pattern {pattern}")
        if pattern is ...:
            return True
        elif isclass(pattern):
            if issubclass(pattern, expressions.Expression):
                res = isinstance(expression, pattern)
                if res:
                    logging.debug(f"Match type {expression} {pattern}")
                return res
            else:
                raise ValueError(
                    'Class pattern matching only implemented '
                    'for Expression subclasses'
                )
        elif isinstance(pattern, expressions.Expression):
            if not (
                (
                    hasattr(type(pattern), '__generic_class__') and
                    isinstance(expression, type(pattern).__generic_class__) and
                    pattern.type is expressions.ToBeInferred
                ) or
                isinstance(expression, type(pattern))
            ):
                logging.debug(
                    f"\texpression is not instance of pattern "
                    f"class {pattern.__class__}"
                )
                return False

            if isclass(pattern.type) and issubclass(pattern.type, Tuple):
                logging.debug("\tMatch tuple")
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
                        logging.debug(
                            f"\t\tMatched tuple's expression instance "
                            f"{expression} with {pattern}"
                        )
                        return True
                else:
                    return False
            else:
                parameters = inspect.signature(pattern.__class__).parameters
                logging.debug(
                    f"\t\tTrying to match parameters "
                    f"{expression} with {pattern}"
                )
                for argname, arg in parameters.items():
                    if arg.default is not inspect.Parameter.empty:
                        continue
                    p = getattr(pattern, argname)
                    e = getattr(expression, argname)
                    match = self.pattern_match(p, e)
                    if not match:
                        return False
                    else:
                        logging.debug(f"\t\t\tmatch {p} vs {e}")
                else:
                    return True
        elif isinstance(pattern, tuple) and isinstance(expression, tuple):
            if len(pattern) != len(expression):
                return False
            for p, e in zip(pattern, expression):
                if not self.pattern_match(p, e):
                    return False
            else:
                logging.debug(f"Match tuples {expression} with {pattern}")
                return True
        else:
            logging.debug("Match other {} vs {}".format(pattern, expression))
            return pattern == expression
