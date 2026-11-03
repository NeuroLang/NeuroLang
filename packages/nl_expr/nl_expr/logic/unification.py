from .. import expressions as exp
from . import UnaryLogicOperator, NaryLogicOperator


def most_general_unifier(expression1, expression2):
    '''
    Obtain the most general unifier (MGU) between two function applications.
    If the MGU exists it returns the substitution and the unified expression.
    If the MGU doesn't exist it returns None.
    '''
    args1, args2 = most_general_unifier_extract_arguments(
        expression1, expression2
    )

    if args1 is None or args2 is None:
        return None

    unifier = most_general_unifier_arguments(args1, args2)

    if unifier is None:
        return unifier
    else:
        return (
            unifier[0],
            apply_substitution(expression1, unifier[0])
        )


def most_general_unifier_extract_arguments(expression1, expression2):
    expression_stack = [(expression1, expression2)]
    args1 = tuple()
    args2 = tuple()
    while expression_stack:
        expression1, expression2 = expression_stack.pop(0)
        if not (
            expression1.functor == expression2.functor and
            len(expression1.args) == len(expression2.args)
        ):
            return None, None

        for arg1, arg2 in zip(expression1.args, expression2.args):
            is_application1 = isinstance(arg1, exp.FunctionApplication)
            is_application2 = isinstance(arg2, exp.FunctionApplication)
            if is_application1 and is_application2:
                expression_stack.append((arg1, arg2))
            elif is_application1 or is_application2:
                return None, None
            else:
                args1 += (arg1,)
                args2 += (arg2,)

    return args1, args2


def apply_substitution(function_application, substitution):
    if isinstance(function_application, UnaryLogicOperator):
        return function_application.apply(*(
            apply_substitution(formula, substitution)
            for formula in function_application.unapply()
        ))
    if isinstance(function_application, NaryLogicOperator):
        return function_application.apply(
            *(
                apply_substitution_arguments(formula, substitution)
                for formula in function_application.unapply()
            )
        )
    return type(function_application)(
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
    new_args = tuple()
    for a in arguments:
        if isinstance(a, exp.FunctionApplication):
            new_args += (apply_substitution(a, substitution),)
        else:
            new_args += (substitution.get(a, a),)
    return new_args


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
