from . import expressions as exp


def most_general_unifier(expression1, expression2):
    '''
    Obtain the most general unifier (MGU) between two function applications.
    If the MGU exists it returns the substitution and the unified expression.
    If the MGU doesn't exist it returns None.
    '''
    if not (
        isinstance(expression1, exp.FunctionApplication) and
        isinstance(expression2, exp.FunctionApplication) and
        len(expression1.args) == len(expression2.args)
    ):
        return ValueError("We can only unify function applications")

    if not (
        expression1.functor == expression2.functor and
        len(expression1.args) == len(expression2.args)
    ):
        return None

    unifier = most_general_unifier_arguments(
        expression1.args, expression2.args
    )

    if unifier is None:
        return unifier
    else:
        return (
            unifier[0],
            expression1.apply(expression1.functor, unifier[1])
        )


def apply_substitution(function_application, substitution):
    return exp.FunctionApplication[function_application.type](
        function_application.functor,
        apply_substitution_arguments(function_application.args, substitution)
    )


def most_general_unifier_arguments(args1, args2):
    '''
    Obtain the most general unifier (MGU) between argument tuples.
    If the MGU exists it returns the substitution and the unified arguments.
    If the MGU doesn't exist it returns None.
    '''
    if not (
        isinstance(args1, tuple) and
        isinstance(args2, tuple)
    ):
        return ValueError("We can only unify argument tuples")

    if len(args1) != len(args2):
        return None

    substitution = dict()
    while True:
        for arg1, arg2 in zip(args1, args2):
            if arg1 != arg2:
                break
        else:
            return substitution, args1

        if isinstance(arg1, exp.Symbol):
            substitution[arg1] = arg2
        elif isinstance(arg2, exp.Symbol):
            substitution[arg2] = arg1
        else:
            return None

        args1 = apply_substitution_arguments(args1, substitution)
        args2 = apply_substitution_arguments(args2, substitution)


def apply_substitution_arguments(arguments, substitution):
    return tuple(substitution.get(a, a) for a in arguments)


def merge_substitutions(subs1, subs2):
    if len(subs1) > len(subs2):
        aux = subs1
        subs1 = subs2
        subs2 = aux

    if any(
        v != subs2[k]
        for k, v in subs1.items()
        if k in subs2
    ):
        return None
    res = subs1.copy()
    res.update(subs2)
    return res


def compose_substitutions(subs1, subs2):
    new_subs = dict()
    new_subs = {
        k: v for k, v in subs2.items()
        if k not in subs1
    }

    for k, v in subs1.items():
        if isinstance(v, exp.Symbol):
            new_value = subs2.get(k, v)
            if new_value != k:
                new_subs[k] = new_value
        else:
            new_subs[k] = v

    return new_subs
