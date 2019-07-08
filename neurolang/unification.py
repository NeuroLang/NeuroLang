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
        raise ValueError("We can only unify function applications")

    if not (
        expression1.functor == expression2.functor and
        len(expression1.args) == len(expression2.args)
    ):
        return None

    substitution = dict()
    while True:
        print(expression1, expression2)
        for arg1, arg2 in zip(expression1.args, expression2.args):
            if arg1 != arg2:
                break
        else:
            return substitution, expression1

        if isinstance(arg1, exp.Symbol):
            substitution[arg1] = arg2
        elif isinstance(arg2, exp.Symbol):
            substitution[arg2] = arg1
        else:
            return None

        expression1 = apply_substitution(expression1, substitution)
        expression2 = apply_substitution(expression2, substitution)


def apply_substitution(function_application, substitution):
    return exp.FunctionApplication[function_application.type](
        function_application.functor,
        tuple(substitution.get(a, a) for a in function_application.args)
    )


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
