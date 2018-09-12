import typing


Unknown = typing.Any


def is_consistent(type1, type2):
    if not isinstance(type1, type) and isinstance(type2, type):
        raise ValueError('Both parameters need to be types')

    if type1 is Unknown or type2 is Unknown:
        return True
    elif is_parameterized(type1) and is_parameterized(type2):
        generic1 = generic_type(type1)
        generic2 = generic_type(type2)

        if not is_consistent(generic1, generic2):
            return False

        type_parameters1 = get_type_parameters(type1)
        type_parameters2 = get_type_parameters(type2)

        return len(type_parameters1) == len(type_parameters2) and all(
            is_consistent(t1, t2)
            for t1, t2 in zip(type_parameters1, type_parameters2)
        )
    else:
        return type1 is type2


def is_leq_informative(left, right):
    if not (is_type(left) and is_type(right)):
        raise ValueError('Both parameters need to be types')

    if left is Unknown:
        return True
    elif right is Unknown:
        return False
    elif generic_type(right) is typing.Union:
        return any(
            is_leq_informative(left, parameter)
            for parameter in get_type_parameters(right)
        )
    elif generic_type(left) is typing.Union:
        return False
    elif is_parameterized(right):
        generic_right = generic_type(right)

        if is_parameterized(left):
            if not is_leq_informative(generic_type(left), generic_right):
                return False

            type_parameters_left = get_type_parameters(left)
            type_parameters_right = get_type_parameters(right)

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
    else:
        return issubclass(left, right)


def is_type(type_):
    return (
        isinstance(type_, type) or
        type_ is Unknown or
        type(type_) is type(typing.Union)
    )


def is_parameterized(type_):
    return getattr(type_, '__origin__', None) is not None


def generic_type(type_):
    return getattr(type_, '__origin__', None)


def get_type_parameters(type_):
    return getattr(type_, '__args__', None)
