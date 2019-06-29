"""Module implementing expression pattern matching."""

from collections import OrderedDict
import copy
from itertools import chain
import inspect
from inspect import isclass
from functools import lru_cache
import logging
from typing import Any, Tuple, TypeVar
import types
from warnings import warn

from . import expressions
from .type_system import replace_type_variable

logging.addLevelName(logging.DEBUG - 1, 'FINEDEBUG')
FINEDEBUG = logging.DEBUG - 1


__all__ = ['add_match', 'PatternMatcher']


UndeterminedType = TypeVar('UndeterminedType')


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
            if (
                '__init__' in repeated_methods and
                getattr(base, '__init__') is object.__init__
            ):
                repeated_methods.remove('__init__')
            if len(repeated_methods) > 1:
                warn_message = (
                    f"Warning in class {name} "
                    f"overwrites {repeated_methods} from base {base}"
                )
                warn(warn_message)

        src_type, dst_type, needs_replacement = cls.__infer_patterns__(
            classdict
        )

        current_type = cls.__infer_type__(classdict, bases)

        classdict['type'] = current_type

        new_cls = super().__new__(cls, name, bases, classdict)
        if needs_replacement:
            cls.__replace_type_in_patterns__(new_cls, src_type, dst_type)

        return new_cls

    def __infer_type__(classdict, bases):
        current_type = classdict.get('type', Any)
        for base in bases:
            if hasattr(base, 'type'):
                if (
                    current_type is Any or
                    base.type is Any or
                    isinstance(base.type, TypeVar) or
                    current_type is base.type
                ):
                    if current_type is Any:
                        current_type = base.type
                else:
                    current_type = UndeterminedType
                    break
        return current_type

    def __infer_patterns__(classdict):
        src_type = None
        dst_type = None
        needs_replacement = False
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

        patterns = []
        for v in classdict.values():
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
        return src_type, dst_type, needs_replacement

    def __replace_type_in_patterns__(new_cls, src_type, dst_type):
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
            pattern_class = pattern_class.__generic_class__[
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
    """Decorate by adding patterns to a :class:`PatternMatcher` class.

    Should be used as
    `@add_match(PATTERN, GUARD)` to turn the decorated method, receiving
    an instance of :class:`Expression` into a matching case. See
    :py:meth:`PatternMatcher.pattern_match` for more details on patterns.
    """
    def bind_match(f):
        f.pattern = pattern
        f.guard = guard
        return f
    return bind_match


class PatternMatcher(metaclass=PatternMatchingMetaClass):
    """Class for expression pattern matching."""

    @property
    def patterns(self):
        """Property holding an iterator of triplets ``(pattern, guard, action)``.

        - ``pattern``: is an Expression class, or instance where
          construction parameters and the type can be replaced
          by an ellipsis `...` to signal a wildcard. See
          :py:meth:`pattern_match` for more details.
        - ``guard``: is a function mapping an `Expression` instance
          to a boolean or ``None``.
        - ``action``: is the method receiving the matching ``expression``
          instance to be executed upon pattern and match being ``True``.
        """
        return chain(*(
                pm.__patterns__ for pm in self.__class__.mro()
                if hasattr(pm, '__patterns__')
        ))

    def match(self, expression):
        """Find the action for a given expression by going through the ``patterns``.

        Goes through the triplets in in ``patterns`` and calls the action
        specified by the first satisfied triplet.
        """

        logging.info(
            '\033[1m\033[91mExpression\033[0m: %(expression)s',
            {'expression': expression}
        )
        for pattern, guard, action in self.patterns:
            name = '\033[1m\033[91m' + action.__qualname__ + '\033[0m'
            pattern_match = self.pattern_match(pattern, expression)
            guard_match = pattern_match and (
                guard is None or guard(expression)
            )
            if pattern_match and guard_match:
                logging.info('\tMATCH %(name)s', {'name': name})
                logging.info('\t\tpattern: %(pattern)s', {'pattern': pattern})
                logging.info('\t\tguard: %(guard)s', {'guard': guard})
                result_expression = action(self, expression)
                logging.info(
                    '\t\tresult: %(result_expression)s',
                    {'result_expression': result_expression}
                )
                return result_expression
            else:
                logging.debug('\tNOMATCH %(name)s', {'name': name})
                logging.debug(
                    '\t\tpattern: %(pattern)s %(pattern_match)s',
                    {'pattern': pattern, 'pattern_match': pattern_match}
                )
                logging.debug(
                    '\t\tguard: %(guard)s %(guard_match)s',
                    {'guard': guard, 'guard_match': guard_match}
                )
        else:
            raise NeuroLangPatternMatchingNoMatch(f'No match for {expression}')

    def pattern_match(self, pattern, expression):
        """Return ``True`` if ``pattern`` matches ``expression``.

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
        """
        result = False
        if pattern is ...:
            result = True
        elif isclass(pattern):
            if issubclass(pattern, expressions.Expression):
                result = isinstance(expression, pattern)
                if result:
                    logging.log(FINEDEBUG, "\t\tmatch type")
            else:
                raise ValueError(
                    'Class pattern matching only implemented '
                    'for Expression subclasses'
                )
        elif isinstance(pattern, expressions.Expression):
            result = self.pattern_match_expression(pattern, expression)
        elif isinstance(pattern, tuple) and isinstance(expression, tuple):
            result = self.pattern_match_tuple(pattern, expression)
        else:
            logging.log(
                FINEDEBUG,
                "\t\t\t\tMatch other %(pattern)s vs %(expression)s",
                {'expression': expression, 'pattern': pattern}
            )
            result = pattern == expression
        return result

    def pattern_match_expression(self, pattern, expression):
        if not (
            (
                hasattr(type(pattern), '__generic_class__') and
                isinstance(expression, type(pattern).__generic_class__) and
                expressions.is_leq_informative(
                    expression.type, pattern.type
                )
            ) or
            isinstance(expression, type(pattern))
        ):
            logging.log(
                FINEDEBUG,
                "\t\t\t\t%(expression)s is not instance of pattern "
                "class %(class)s",
                {'expression': expression, 'class': pattern.__class__}
            )
            result = False
        elif isclass(pattern.type) and issubclass(pattern.type, Tuple):
            logging.log(FINEDEBUG, "\t\t\t\tMatch tuple")
            if (
                isclass(expression.type) and
                issubclass(expression.type, Tuple)
            ):
                result = self.pattern_match_expression_tuple(expression, pattern)
            else:
                result = False
        else:
            result = self.pattern_match_expression_parameters(pattern, expression)
        return result

    def pattern_match_expression_parameters(self, pattern, expression):
        parameters = signature(pattern.__class__)
        logging.log(
            FINEDEBUG,
            "\t\t\t\tTrying to match parameters "
            "%(expression)s with %(pattern)s",
            {'expression': expression, 'pattern': pattern}
        )
        for argname, arg in parameters.items():
            if arg.default is not inspect.Parameter.empty:
                continue
            p = getattr(pattern, argname)
            e = getattr(expression, argname)
            match = self.pattern_match(p, e)
            if not match:
                result = False
                break
            else:
                logging.log(
                    FINEDEBUG,
                    "\t\t\t\t\tmatch %(p)s vs %(e)s",
                    {'p': p, 'e': e}
                )
        else:
            result = True
        return result

    def pattern_match_expression_tuple(self, expression, pattern):
        result = True
        if (
            len(pattern.type.__args__) !=
            len(expression.type.__args__)
        ):
            result = False
        else:
            if pattern.value is ...:
                pattern_value = (
                    expressions.Constant[pattern.type.__args__[i]](...)
                    for i in range(len(pattern.type.__args__))
                )
            else:
                pattern_value = pattern.value
            for p, e in zip(pattern_value, expression.value):
                if not self.pattern_match(p, e):
                    result = False
                    break
            else:
                logging.log(
                    FINEDEBUG,
                    "\t\t\t\t\tMatched tuple's expression instance "
                    "%(expression)s with %(pattern)s",
                    {'expression': expression, 'pattern': pattern}
                )
                result = True
        return result

    def pattern_match_tuple(self, pattern, expression):
        if len(pattern) != len(expression):
            result = False
        else:
            for p, e in zip(pattern, expression):
                if not self.pattern_match(p, e):
                    result = False
                    break
            else:
                logging.log(
                    FINEDEBUG,
                    "\t\t\t\tMatch tuples %(expression)s with %(pattern)s",
                    {'expression': expression, 'pattern': pattern}
                )
                result = True
        return result


@lru_cache(maxsize=128)
def signature(cls):
    return inspect.signature(cls).parameters
