from collections import OrderedDict
from itertools import chain
import inspect

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
            if (
                self.pattern_match(pattern, expression) and
                (guard is None or guard(expression))
            ):
                return action(self, expression)
        else:
            raise ValueError()

    def pattern_match(self, pattern, expression):
        if isinstance(pattern, type(...)):
            return True
        elif type(pattern) == type:
            return isinstance(expression, pattern)
        elif isinstance(pattern, expressions.Expression):
            if (
                not isinstance(expression, pattern.__class__) or
                not expressions.is_subtype(pattern.type, expression.type)
            ):
                return False
            elif isinstance(pattern, expressions.FunctionApplication):
                return (
                    self.pattern_match(pattern.functor, expression.functor) and
                    (
                        pattern.args is None or
                        self.pattern_match(pattern.args, expression.args)
                    )
                )
            else:
                parameters = inspect.signature(pattern.__class__).parameters
                return all(
                    self.pattern_match(
                        getattr(pattern, argname), getattr(expression, argname)
                    )
                    for argname, arg in parameters.items()
                    if arg.default == inspect._empty
                )
        else:
            return pattern == expression
