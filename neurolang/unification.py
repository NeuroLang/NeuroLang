from .expressions import FunctionApplication, Symbol
from .generative_datalog import DeltaTerm, DeltaAtom


def check_type_and_get_terms(exp):
    if isinstance(exp, FunctionApplication):
        terms = exp.args
    elif isinstance(exp, DeltaAtom):
        terms = exp.terms
    else:
        raise NeuroLangException(
            'Expression was expected to be an atom or a Δ-atom, '
            'but got {}'.format(type(exp))
        )
    return terms


def most_general_unifier(exp1, exp2):
    '''
    Obtain the most general unifier (MGU) between two literals.
    If the MGU exists it returns the substitution and the unified expression.
    If the MGU doesn't exist it returns None.
    '''
    terms1 = check_type_and_get_terms(exp1)
    terms2 = check_type_and_get_terms(exp2)

    if exp1.functor != exp2.functor or len(terms1) != len(terms2):
        return None

    unifier = most_general_unifier_arguments(terms1, terms2)

    if unifier is None:
        return unifier
    else:
        return (unifier[0], exp1.apply(exp1.functor, unifier[1]))


def apply_substitution(literal, substitution):
    return literal.__class__(
        literal.functor,
        apply_substitution_args(
            check_type_and_get_terms(literal), substitution
        )
    )


def most_general_unifier_arguments(args1, args2):
    '''Obtain the most general unifier (MGU) between argument tuples.

    If the MGU exists it returns the substitution and the unified arguments.
    If the MGU doesn't exist it returns None.
    '''
    if not isinstance(args1, tuple) or not isinstance(args2, tuple):
        return ValueError('We can only unify argument tuples')

    if len(args1) != len(args2):
        return None

    substitution = dict()
    while True:
        for arg1, arg2 in zip(args1, args2):
            if arg1 != arg2:
                break
        else:
            return substitution, args1

        if isinstance(arg1, Symbol):
            substitution[arg1] = arg2
        elif isinstance(arg2, Symbol):
            substitution[arg2] = arg1
        elif isinstance(arg1, DeltaTerm) and isinstance(arg2, DeltaTerm):
            for p1, p2 in zip(arg1.dist_parameters, arg2.dist_parameters):
                if p1 != p2:
                    substitution[p1] = p2
                    break
        else:
            return None

        expression1 = apply_substitution(expression1, substitution)
        expression2 = apply_substitution(expression2, substitution)


def apply_substitution_to_delta_term(delta_term, substitution):
    return DeltaTerm(
        delta_term.dist_name,
        *tuple(substitution.get(p, p) for p in delta_term.dist_parameters)
    )


def apply_substitution_args(args, substitution):
    return tuple(
        apply_substitution_to_delta_term(arg, substitution)
        if isinstance(arg, DeltaTerm) else substitution.get(arg, arg)
        for arg in args
    )


def merge_substitutions(subs1, subs2):
    if len(subs1) > len(subs2):
        aux = subs1
        subs1 = subs2
        subs2 = aux

    if any(v != subs2[k] for k, v in subs1.items() if k in subs2):
        return None
    res = subs1.copy()
    res.update(subs2)
    return res


def compose_substitutions(subs1, subs2):
    new_subs = dict()
    new_subs = {k: v for k, v in subs2.items() if k not in subs1}

    for k, v in subs1.items():
        if isinstance(v, Symbol):
            new_value = subs2.get(k, v)
            if new_value != k:
                new_subs[k] = new_value
        else:
            new_subs[k] = v

    return new_subs
