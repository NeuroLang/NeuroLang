import typing
from typing_inspect import (
    get_args, get_origin,
    is_union_type, is_tuple_type, is_callable_type, is_generic_type,
    is_typevar
)

Unknown = typing.Any


type_order = {
    int: (float, complex),
    float: (complex,)
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
        get_origin(left) is typing.Generic or
        get_origin(right) is typing.Generic
    ):
        raise ValueError("typing Generic not supported")
    if left is right:
        return True
    elif left is Unknown:
        return True
    elif right is Unknown:
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

            return (
                len(type_parameters_left) == len(type_parameters_right) and
                all(
                    is_leq_informative(t_left, t_right)
                    for t_left, t_right in
                    zip(type_parameters_left, type_parameters_right)
                )
            )
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
        is_typevar(type_) or
        is_union_type(type_)
    )


def is_parameteritrical(type_):
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
