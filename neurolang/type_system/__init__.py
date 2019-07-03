'''
Implementation of the type system based on Siek and Vachharajani,
"Gradual Typing with Unification-based Inference", DLS 2008.
'''


import inspect
import types
from typing import (
    Callable, Tuple, Set, AbstractSet, Mapping, TypeVar,
    Iterable, Sequence, Any, Generic, Text, _FinalTypingBase
)

from typing_inspect import (
    get_origin,
    is_union_type, is_tuple_type, is_callable_type, is_generic_type,
    is_typevar
)


from ..exceptions import NeuroLangException


class NeuroLangTypeException(NeuroLangException):
    pass


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


type_order = {
    int: (float, complex),
    float: (complex,),
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
        return True
    elif left is Unknown:
        return True
    elif right is Unknown:
        return False
    elif right is Any:
        return True
    elif left is Any:
        return False
    elif is_union_type(right):
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
    elif is_union_type(left):
        return False
    elif is_parameterized(right):
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
    elif left in type_order and right in type_order[left]:
        return True
    else:
        return issubclass(left, right)


def is_type(type_):
    return (
        isinstance(type_, type) or
        type_ is Unknown or
        type_ is Any or
        is_typevar(type_) or
        is_union_type(type_)
    )


def is_parametrical(type_):
    return any(
        p(type_)
        for p in
        (is_generic_type, is_callable_type, is_tuple_type, is_union_type)
    )


def is_parameterized(type_):
    return any(
        p(type_)
        for p in
        (is_generic_type, is_callable_type, is_tuple_type, is_union_type)
    ) and get_origin(type_) is not type_


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
        return typing_callable_from_annotated_function(value)
    elif isinstance(value, Tuple):
        inner_types = tuple(
            recursive_callback(v)
            for v in value
        )
        return Tuple[inner_types]
    elif isinstance(value, Text):
        return type(value)
    elif isinstance(value, (AbstractSet, Sequence)):
        if len(value) == 0:
            inner_type = Unknown
        else:
            it = iter(value)
            element = next(it)
            inner_type = recursive_callback(element)
            if deep:
                for element in it:
                    inner_type = unify_types(
                        inner_type, recursive_callback(it)
                    )
        if isinstance(value, AbstractSet):
            return AbstractSet[inner_type]
        elif isinstance(value, Sequence):
            return Sequence[inner_type]
    elif isinstance(value, Mapping):
        it = iter(value.items())
        k, v = next(it)
        ktype = recursive_callback(k)
        vtype = recursive_callback(v)
        if deep:
            for element in it:
                ktype = unify_types(recursive_callback(k), ktype)
                vtype = unify_types(recursive_callback(v), vtype)
        return Mapping[ktype, vtype]
    else:
        return type(value)


def replace_type_variable(type_, type_hint, type_var=None):
    if (
        isinstance(type_hint, TypeVar) and
        type_hint == type_var
    ):
        return type_
    elif hasattr(type_hint, '__args__') and type_hint.__args__ is not None:
        new_args = []
        for arg in get_args(type_hint):
            new_args.append(
                replace_type_variable(type_, arg, type_var=type_var)
            )
        return type_hint.__origin__[tuple(new_args)]
    elif isinstance(type_hint, Iterable):
        return [
            replace_type_variable(type_, arg, type_var=type_var)
            for arg in type_hint
        ]
    else:
        return type_hint


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


def get_args(type_):
    ret = type_.__args__
    if ret is None:
        return ()
    else:
        return ret
