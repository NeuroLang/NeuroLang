'''
Implementation of the type system based on Siek and Vachharajani,
"Gradual Typing with Unification-based Inference", DLS 2008.
'''


import inspect
import operator
import sys
import types
from itertools import islice
from typing import (AbstractSet, Any, Callable, Generic, Iterable, Mapping,
                    Sequence, Set, Text, Tuple, TypeVar)

import numpy as np
from typing_inspect import (get_origin, is_callable_type, is_generic_type,
                            is_tuple_type, is_typevar, is_union_type)

from ..exceptions import NeuroLangException

NEW_TYPING = sys.version_info[:3] >= (3, 7, 0)


class NeuroLangTypeException(NeuroLangException):
    pass


if sys.version_info < (3, 6, 0):
    raise ImportError("Only python 3.6 and over compatible")
if not NEW_TYPING:
    from typing import _FinalTypingBase

    class _Unknown(_FinalTypingBase, _root=True):
        """Special type indicating an unknown type.

        - Unknown is compatible with every type.
        - Unknown is less informative than all types.
        """

        __slots__ = ()

        def __instancecheck__(self, obj):
            raise TypeError("Unknown cannot be used with isinstance().")

        def __subclasscheck__(self, cls):
            raise TypeError("Unknown cannot be used with issubclass().")

    Unknown = _Unknown(_root=True)
else:
    from typing import _SpecialForm, _Final, _Immutable, _GenericAlias

    Unknown = _SpecialForm(
        'Unknown', doc="""
        Special type indicating an unknown type.

        - Unknown is compatible with every type.
        - Unknown is less informative than all types.
        """
    )

    class Unknown(_Final, _Immutable, _root=True):
        """Special type indicating an unknown type.

        - Unknown is compatible with every type.
        - Unknown is less informative than all types.
        """

        __slots__ = ()

        def __instancecheck__(self, obj):
            raise TypeError("Unknown cannot be used with isinstance().")

        def __subclasscheck__(self, cls):
            raise TypeError("Unknown cannot be used with issubclass().")


type_order = {
    np.int64: (np.int,),
    np.bool_: (bool,),
    bool: (np.bool_,),
    int: (np.int64, float, complex),
    float: (np.float64, complex,),
    Set: (AbstractSet,)
}


def is_consistent(type1, type2):
    if not isinstance(type1, type) and isinstance(type2, type):
        raise ValueError('Both parameters need to be types')

    if type1 is Unknown or type2 is Unknown:
        return True
    elif is_parameterized(type1) and is_parameterized(type2):
        generic1 = get_origin(type1)
        generic2 = get_origin(type2)

        if not is_consistent(generic1, generic2):
            return False

        type_parameters1 = get_args(type1)
        type_parameters2 = get_args(type2)

        return len(type_parameters1) == len(type_parameters2) and all(
            is_consistent(t1, t2)
            for t1, t2 in zip(type_parameters1, type_parameters2)
        )
    else:
        return type1 is type2


def is_leq_informative(left, right):
    if not (is_type(left) and is_type(right)):
        raise ValueError('Both parameters need to be types')
    if (
        get_origin(left) is Generic or
        get_origin(right) is Generic
    ):
        raise ValueError("typing Generic not supported")
    if left is right:
        result = True
    elif left is Unknown:
        result = True
    elif right is Unknown:
        result = False
    elif right is Any:
        result = True
    elif left is Any:
        result = False
    elif is_union_type(right):
        result = is_leq_informative_union(left, right)
    elif is_union_type(left):
        result = False
    elif is_parameterized(right):
        result = is_leq_informative_parameterized_right(left, right)
    elif is_parametrical(right):
        if is_parameterized(left):
            left = get_origin(left)
        result = issubclass(left, right)
    elif is_parametrical(left) or is_parameterized(left):
        result = False
    elif left in type_order and right in type_order[left]:
        result = True
    else:
        result = issubclass(left, right)

    return result


def is_leq_informative_union(left, right):
    type_parameters_right = get_args(right)
    if is_union_type(left):
        type_parameters_left = get_args(left)
        return all(
            any(
                is_leq_informative(l, r)
                for r in type_parameters_right
            )
            for l in type_parameters_left
        )
    else:
        return any(
            is_leq_informative(left, parameter)
            for parameter in get_args(right)
        )


def is_leq_informative_parameterized_right(left, right):
    generic_right = get_origin(right)

    if is_parameterized(left):
        if not is_leq_informative(get_origin(left), generic_right):
            return False

        type_parameters_left = get_args(left)
        type_parameters_right = get_args(right)

        if len(type_parameters_left) != len(type_parameters_right):
            return False

        return all(
            is_leq_informative(t_left, t_right)
            for t_left, t_right in
            zip(type_parameters_left, type_parameters_right)
        )
    elif is_parametrical(left):
        return False
    else:
        return is_leq_informative(left, generic_right)


def is_type(type_):
    return (
        isinstance(type_, type) or
        type_ is Unknown or
        type_ is Any or
        is_parameterized(type_) or
        is_parametrical(type_) or
        is_typevar(type_) or
        is_union_type(type_)

    )


def is_parametrical(type_):
    is_parametrical_generic = any(
        p(type_)
        for p in
        (is_generic_type, is_callable_type, is_tuple_type, is_union_type)
    ) and not getattr(type_, '_is_protocol', False)

    if is_parametrical_generic:
        if NEW_TYPING:
            return getattr(type_, '_special', False) or (
                is_union_type(type_) and not hasattr(type_, '__args__')
            )
        else:
            return type_.__args__ is None
    else:
        return False


def is_parameterized(type_):
    is_parametrical_generic = any(
        p(type_)
        for p in
        (is_generic_type, is_callable_type, is_tuple_type, is_union_type)
    )

    if is_parametrical_generic:
        if NEW_TYPING:
            return not (
                getattr(type_, '_special', False) or (
                    is_union_type(type_) and not hasattr(type_, '__args__')
                ) or
                getattr(type_, '_is_protocol', False)
            )
        else:
            return get_origin(type_) is not type_
    else:
        return False


def unify_types(t1, t2):
    if t1 is Unknown:
        return t2
    elif t2 is Unknown:
        return t1
    elif is_leq_informative(t1, t2):
        return t2
    elif is_leq_informative(t2, t1):
        return t1
    else:
        raise NeuroLangTypeException(
            "The types {} and {} can't be unified".format(
                t1, t2
            )
        )


def infer_type(value, deep=False, recursive_callback=None):
    if recursive_callback is None:
        recursive_callback = infer_type

    if isinstance(value, (types.FunctionType, types.MethodType)):
        result = typing_callable_from_annotated_function(value)
    elif isinstance(value, types.BuiltinFunctionType):
        result = infer_type_builtins(value)
    elif isinstance(value, Tuple):
        inner_types = tuple(
            recursive_callback(v)
            for v in value
        )
        result = Tuple[inner_types]
    elif isinstance(value, Text):
        result = type(value)
    elif isinstance(value, (AbstractSet, Sequence)):
        result = infer_type_iterables(
            value, deep=deep, recursive_callback=recursive_callback
        )
    elif isinstance(value, Mapping):
        result = infer_type_mapping(
            value, deep=deep, recursive_callback=recursive_callback
        )
    else:
        result = type(value)

    return result


def infer_type_iterables(value, deep=True, recursive_callback=infer_type):
    inner_type = Unknown
    it = iter(value)
    if not deep:
        it = islice(it, 1)

    for element in it:
        inner_type = unify_types(
            recursive_callback(element), inner_type
        )
    if isinstance(value, AbstractSet):
        return AbstractSet[inner_type]
    elif isinstance(value, Sequence):
        return Sequence[inner_type]


def infer_type_mapping(value,  deep=True, recursive_callback=infer_type):
    ktype = Unknown
    vtype = Unknown
    it = iter(value.items())
    if not deep:
        it = islice(it, 1)

    for k, v in it:
        ktype = unify_types(recursive_callback(k), ktype)
        vtype = unify_types(recursive_callback(v), vtype)
    return Mapping[ktype, vtype]


def replace_type_variable(type_, type_hint, type_var=None):
    if (
        isinstance(type_hint, TypeVar) and
        type_hint == type_var
    ):
        return type_
    elif is_parameterized(type_hint):
        new_args = replace_type_variable(
            type_, get_args(type_hint), type_var=type_var
        )
        new_args = tuple(new_args)
        origin = get_origin(type_hint)
        return replace_type_variable_fix_python36_37(
            type_hint, origin, new_args
        )
    elif isinstance(type_hint, Iterable):
        return [
            replace_type_variable(type_, arg, type_var=type_var)
            for arg in type_hint
        ]
    else:
        return type_hint


def replace_type_variable_fix_python36_37(type_hint, origin, new_args):
    if NEW_TYPING and isinstance(type_hint, _GenericAlias):
        new_type = type_hint.copy_with(new_args)
    else:
        new_type = origin[new_args]
    return new_type


def typing_callable_from_annotated_function(function):
    """Get typing.Callable type representing the annotated function type."""
    signature = inspect.signature(function)
    parameter_types = [
        v.annotation if v.annotation is not inspect.Parameter.empty
        else Unknown
        for v in signature.parameters.values()
    ]

    if signature.return_annotation is inspect.Parameter.empty:
        return_annotation = Unknown
    else:
        return_annotation = signature.return_annotation
    return Callable[
        parameter_types,
        return_annotation
    ]


_BINARY_OPERATORS = (
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        operator.lt,
        operator.le,
        operator.and_,
        operator.or_,
        operator.xor,
        operator.contains,
        operator.matmul,
        operator.mul,
        operator.add,
        operator.sub,
        operator.pow,
        operator.lshift,
        operator.rshift,
        operator.truediv,
        operator.ior,
        operator.iand,
)


_UNARY_OPERATORS = (
    operator.neg,
    operator.pos,
    operator.invert,
)


_NARY_OPERATORS = (
    sum,
)

_RELATIVE_OPERATORS = (
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
    operator.lt,
    operator.le,
    operator.contains,
)


_BOOLEAN_OPERATORS = (
    operator.and_,
    operator.or_,
    operator.xor,
    operator.invert
)


def infer_type_builtins(builtin):
    if builtin in _BINARY_OPERATORS:
        n_params = 2
    elif builtin in _UNARY_OPERATORS:
        n_params = 1
    elif builtin in _NARY_OPERATORS:
        n_params = None
    else:
        try:
            signature = inspect.signature(builtin)
            n_params = len(signature.parameters)
        except ValueError:
            return Callable[..., Unknown]

    if builtin in _BOOLEAN_OPERATORS:
        params_type = [bool] * n_params
        return_type = bool
    elif builtin in _RELATIVE_OPERATORS:
        params_type = [Unknown] * n_params
        return_type = bool
    elif builtin in _NARY_OPERATORS:
        params_type = [Unknown]
        return_type = Unknown
    else:
        params_type = [Unknown] * n_params
        return_type = Unknown

    return Callable[params_type, return_type]


def get_args(type_):
    if is_parameterized(type_):
        ret = type_.__args__
        if ret is None:
            ret = tuple()
    elif is_parametrical(type_):
        ret = tuple()
    else:
        raise ValueError(f"Not {type_} is not a generic type")
    return ret


def get_generic_type(typ):
    return getattr(typ, "__generic_class__", typ)
