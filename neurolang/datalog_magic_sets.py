'''
Magic Sets [1] rewriting implementation for Datalog.

[1] F. Bancilhon, D. Maier, Y. Sagiv, J. D. Ullman, in ACM PODS ’86, pp. 1–15.
'''


from .solver_datalog_naive import (
    Implication, Symbol,
    ExpressionBlock, Constant,
    extract_datalog_predicates,
)


class SymbolAdorned(Symbol):
    """Symbol with adornments"""

    def __init__(self, name, adornment, number):
        self.adornment = adornment
        self.number = number
        super().__init__(name)

    def __eq__(self, other):
        return (
            isinstance(other, SymbolAdorned) and
            hash(self) == hash(other)
        )

    def __hash__(self):
        return hash((self.name, self.adornment, self.number))

    def __repr__(self):
        if len(self.adornment) > 0:
            superindex = f'^{self.adornment}'
        else:
            superindex = ''

        if self.number is not None:
            subindex = f'_{self.number}'
        else:
            subindex = ''

        return (
            f'S{{{self.name}{superindex}{subindex}: '
            f'{self.__type_repr__}}}'
        )


def magic_rewrite(query, datalog):
    adorned_code = reachable_adorned_code(query, datalog)
    # assume that the query rule is the first
    adorned_query = adorned_code.expressions[0]
    goal = adorned_query.consequent.functor

    idb = datalog.intensional_database()
    edb = datalog.extensional_database()
    magic_rules = create_magic_rules(adorned_code, idb, edb)
    modified_rules = create_modified_rules(adorned_code, edb)
    complementary_rules = create_complementary_rules(adorned_code, idb)

    return goal, ExpressionBlock(
        magic_rules +
        modified_rules +
        complementary_rules
    )


def create_complementary_rules(adorned_code, idb):
    complementary_rules = []
    for i, rule in enumerate(adorned_code.expressions):
        for predicate in extract_datalog_predicates(rule.antecedent):
            if predicate.functor.name in idb:
                magic_consequent = magic_predicate(predicate)
                magic_antecedent = magic_predicate(predicate, i)
                complementary_rules.append(Implication(
                    magic_consequent, magic_antecedent
                ))

    return complementary_rules


def create_magic_rules(adorned_code, idb, edb):
    magic_rules = []
    for i, rule in enumerate(adorned_code.expressions):
        consequent = rule.consequent
        new_consequent = magic_predicate(consequent)
        if len(new_consequent.args) == 0:
            new_consequent = Constant(True)
        antecedent = rule.antecedent
        predicates = extract_datalog_predicates(antecedent)

        edb_antecedent = create_magic_rules_create_edb_antecedent(
            predicates, edb
        )
        new_antecedent = new_consequent
        for predicate in edb_antecedent:
            new_antecedent = new_antecedent & predicate

        magic_rules += create_magic_rules_create_rules(
            new_antecedent, predicates, idb, i
        )
    return magic_rules


def create_magic_rules_create_edb_antecedent(predicates, edb):
    edb_antecedent = []
    for predicate in predicates:
        functor = predicate.functor
        if (
            functor.name in edb and
            isinstance(functor, SymbolAdorned) and
            'b' in functor.adornment
        ):
            predicate = Symbol(predicate.functor.name)(*predicate.args)
            edb_antecedent.append(predicate)
    return edb_antecedent


def create_magic_rules_create_rules(new_antecedent, predicates, idb, i):
    magic_rules = []
    for predicate in predicates:
        functor = predicate.functor
        is_adorned = isinstance(functor, SymbolAdorned)
        if (
            functor.name in idb and
            is_adorned and 'b' in functor.adornment
        ):
            new_predicate = magic_predicate(predicate, i)
            magic_rules.append(
                Implication(
                    new_predicate,
                    new_antecedent
                )
            )
    return magic_rules


def create_modified_rules(adorned_code, edb):
    modified_rules = []
    for i, rule in enumerate(adorned_code.expressions):
        new_antecedent = []
        for predicate in extract_datalog_predicates(rule.antecedent):
            functor = predicate.functor
            if functor.name in edb:
                new_antecedent.append(
                    Symbol(functor.name)(*predicate.args)
                )
            else:
                m_p = magic_predicate(predicate, i)
                update = [m_p, predicate]
                if functor == rule.consequent.functor:
                    new_antecedent = update + new_antecedent
                else:
                    new_antecedent += update

        if len(new_antecedent) > 0:
            new_antecedent_ = new_antecedent[0]
            for predicate in new_antecedent[1:]:
                new_antecedent_ = new_antecedent_ & predicate

            modified_rules.append(Implication(
                rule.consequent, new_antecedent_
            ))

    return modified_rules


def magic_predicate(predicate, i=None):
    name = predicate.functor.name
    adornment = predicate.functor.adornment
    if i is not None:
        new_name = f'magic_r{i}_{name}'
    else:
        new_name = f'magic_{name}'

    new_args = [
        arg for arg, ad in
        zip(predicate.args, adornment)
        if ad == 'b'
    ]
    new_functor = SymbolAdorned(
        new_name, adornment, predicate.functor.number
    )
    return new_functor(*new_args)


def reachable_adorned_code(query, datalog):
    adorned_code = adorn_code(query, datalog)
    reachable_code = []
    adorned_datalog = type(datalog)()
    adorned_datalog.walk(adorned_code)
    adorned_idb = adorned_datalog.intensional_database()
    # assume that the query rule is the first
    adorned_query = adorned_code.expressions[0]
    to_reach = [adorned_query.consequent.functor]
    reached = set()
    while to_reach:
        p = to_reach.pop()
        reached.add(p)
        rules = adorned_idb[p]
        for rule in rules.expressions:
            reachable_code.append(rule)
            for predicate in extract_datalog_predicates(rule.antecedent):
                functor = predicate.functor
                if functor not in reached and functor in adorned_idb:
                    to_reach.append(functor)

    return ExpressionBlock(reachable_code)


def adorn_code(query, datalog):
    """
    Produce the rewritten datalog program according to the
    Magic Sets technique.

    :param query Implication: query to solve
    :param datalog DatalogBasic: processed datalog program.

    Returns
    -------
    ExpressionBlock

        adorned code where the query rule is the first expression
        in the block.
    """

    adornment = ''
    for a in query.args:
        if isinstance(a, Symbol):
            adornment += 'f'
        else:
            adornment += 'b'

    query = SymbolAdorned(query.functor.name, adornment, 0)(*query.args)
    adorn_stack = [query]

    edb = datalog.extensional_database()
    idb = datalog.intensional_database()
    rewritten_program = []
    rewritten_rules = set()

    while adorn_stack:
        consequent = adorn_stack.pop()

        if isinstance(consequent.functor, SymbolAdorned):
            adornment = consequent.functor.adornment
        else:
            adornment = ''

        rules = idb.get(consequent.functor.name, None)
        if rules is None:
            continue

        for rule in rules.expressions:
            adorned_antecedent = adorn_antecedent(
                rule, adornment,
                edb, adorn_stack, rewritten_rules
            )
            new_consequent = consequent.functor(*rule.consequent.args)
            rewritten_program.append(
                Implication(new_consequent, adorned_antecedent)
            )
            rewritten_rules.add(consequent.functor)

    return ExpressionBlock(rewritten_program)


def adorn_antecedent(
    rule, adornment, edb, adorn_stack, rewritten_rules
):
    consequent = rule.consequent
    antecedent = rule.antecedent

    bound_variables = {
        arg for arg, ad in zip(consequent.args, adornment)
        if isinstance(arg, Symbol) and ad == 'b'
    }

    predicates = extract_datalog_predicates(antecedent)
    checked_predicates = {}
    adorned_antecedent = None

    for predicate in predicates:
        if (
            predicate.functor.name in edb and
            len(bound_variables.intersection(predicate.args)) > 0
        ):
            bound_variables.update(
                arg for arg in predicate.args
                if isinstance(arg, Symbol)
            )

    for predicate in predicates:
        predicate_number = checked_predicates.get(predicate, 0)
        checked_predicates[predicate] = predicate_number + 1
        in_edb = predicate.functor.name in edb

        adorned_predicate = adorn_predicate(
            predicate, bound_variables, predicate_number, in_edb
        )

        is_adorned = isinstance(adorned_predicate.functor, SymbolAdorned)
        if (
            not in_edb and is_adorned and
            adorned_predicate.functor not in rewritten_rules
        ):
            adorn_stack.append(adorned_predicate)

        if adorned_antecedent is None:
            adorned_antecedent = adorned_predicate
        else:
            adorned_antecedent = adorned_antecedent & adorned_predicate
    return adorned_antecedent


def adorn_predicate(
    predicate, bound_consequent_variables,
    predicate_number, in_edb
):

    adornment = ''
    has_b = False
    for arg in predicate.args:
        if (
            isinstance(arg, Constant) or
            arg in bound_consequent_variables
        ):
            adornment += 'b'
            has_b = True
        else:
            adornment += 'f'

    if in_edb and has_b:
        adornment = 'b' * len(adornment)

    if not has_b:
        adornment = ''

    p = SymbolAdorned(predicate.functor.name, adornment, predicate_number)
    return p(*predicate.args)
