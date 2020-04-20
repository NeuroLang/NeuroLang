'''
Magic Sets [1] rewriting implementation for Datalog.

[1] F. Bancilhon, D. Maier, Y. Sagiv, J. D. Ullman, in ACM PODS ’86, pp. 1–15.
'''

from ..expressions import Constant, Symbol
from ..type_system import Unknown
from . import expression_processing, extract_logic_predicates
from .expressions import Conjunction, Implication, Union


class AdornedExpression(Symbol):
    def __init__(self, expression, adornment, number):
        self.expression = expression
        self.adornment = adornment
        self.number = number
        self._symbols = expression._symbols
        if self.type is Unknown:
            self.type = self.expression.type

    @property
    def name(self):
        if isinstance(self.expression, Symbol):
            return self.expression.name
        else:
            raise NotImplementedError()

    def __eq__(self, other):
        return (
            hash(self) == hash(other) and
            isinstance(other, AdornedExpression)
        )

    def __hash__(self):
        return hash((self.expression, self.adornment, self.number))

    def __repr__(self):
        if isinstance(self.expression, Symbol):
            rep = self.expression.name
        elif isinstance(self.expression, Constant):
            rep = self.expression.value

        if len(self.adornment) > 0:
            superindex = f'^{self.adornment}'
        else:
            superindex = ''

        if self.number is not None:
            subindex = f'_{self.number}'
        else:
            subindex = ''

        return (
            f'S{{{rep}{superindex}{subindex}: '
            f'{self.__type_repr__}}}'
        )


def magic_rewrite(query, datalog):
    adorned_code = reachable_adorned_code(query, datalog)
    # assume that the query rule is the last
    adorned_query = adorned_code.formulas[-1]
    goal = adorned_query.consequent.functor

    idb = datalog.intensional_database()
    edb = datalog.extensional_database()
    magic_rules = create_magic_rules(adorned_code, idb, edb)
    modified_rules = create_modified_rules(adorned_code, edb)
    complementary_rules = create_complementary_rules(adorned_code, idb)

    return goal, Union(
        magic_rules +
        modified_rules +
        complementary_rules
    )


def create_complementary_rules(adorned_code, idb):
    complementary_rules = []
    for i, rule in enumerate(adorned_code.formulas):
        for predicate in extract_logic_predicates(rule.antecedent):
            if (
                not (
                    isinstance(predicate.functor, AdornedExpression) and
                    isinstance(predicate.functor.expression, Constant)
                ) and
                predicate.functor.name in idb
            ):
                magic_consequent = magic_predicate(predicate)
                magic_antecedent = magic_predicate(predicate, i)
                complementary_rules.append(Implication(
                    magic_consequent, magic_antecedent
                ))

    return complementary_rules


def create_magic_rules(adorned_code, idb, edb):
    magic_rules = []
    for i, rule in enumerate(adorned_code.formulas):
        consequent = rule.consequent
        new_consequent = magic_predicate(consequent)
        if len(new_consequent.args) == 0:
            new_consequent = Constant(True)
        antecedent = rule.antecedent
        predicates = extract_logic_predicates(antecedent)

        edb_antecedent = create_magic_rules_create_edb_antecedent(
            predicates, edb
        )
        new_antecedent = (new_consequent,)
        for predicate in edb_antecedent:
            new_antecedent += (predicate,)

        if len(new_antecedent) == 1:
            new_antecedent = new_antecedent[0]
        else:
            new_antecedent = Conjunction(new_antecedent)

        magic_rules += create_magic_rules_create_rules(
            new_antecedent, predicates, idb, i
        )
    return magic_rules


def create_magic_rules_create_edb_antecedent(predicates, edb):
    edb_antecedent = []
    for predicate in predicates:
        functor = predicate.functor
        if (
            (
                isinstance(functor.expression, Constant) or
                functor.name in edb
            ) and
            isinstance(functor, AdornedExpression) and
            'b' in functor.adornment
        ):
            predicate = Symbol(predicate.functor.name)(*predicate.args)
            edb_antecedent.append(predicate)
    return edb_antecedent


def create_magic_rules_create_rules(new_antecedent, predicates, idb, i):
    magic_rules = []
    for predicate in predicates:
        functor = predicate.functor
        is_adorned = isinstance(functor, AdornedExpression)
        if (
            is_adorned and
            not isinstance(functor.expression, Constant) and
            functor.name in idb and
            'b' in functor.adornment
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
    for i, rule in enumerate(adorned_code.formulas):
        new_antecedent = obtain_new_antecedent(rule, edb, i)

        if len(new_antecedent) > 0:
            if len(new_antecedent) == 1:
                new_antecedent_ = new_antecedent[0]
            else:
                new_antecedent_ = Conjunction(tuple(new_antecedent))

            modified_rules.append(Implication(
                rule.consequent, new_antecedent_
            ))

    return modified_rules


def obtain_new_antecedent(rule, edb, rule_number):
    new_antecedent = []
    for predicate in extract_logic_predicates(rule.antecedent):
        functor = predicate.functor
        if (
            isinstance(functor, AdornedExpression) and
            isinstance(functor.expression, Constant)
        ):
            new_antecedent.append(functor.expression(*predicate.args))
        elif functor.name in edb:
            new_antecedent.append(
                Symbol(functor.name)(*predicate.args)
            )
        else:
            m_p = magic_predicate(predicate, rule_number)
            update = [m_p, predicate]
            if functor == rule.consequent.functor:
                new_antecedent = update + new_antecedent
            else:
                new_antecedent += update
    return new_antecedent


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
    new_functor = AdornedExpression(
        Symbol(new_name), adornment, predicate.functor.number
    )
    return new_functor(*new_args)


def reachable_adorned_code(query, datalog):
    adorned_code = adorn_code(query, datalog)
    adorned_datalog = type(datalog)()
    adorned_datalog.walk(adorned_code)
    # assume that the query rule is the first
    adorned_query = adorned_code.formulas[0]
    return expression_processing.reachable_code(adorned_query, adorned_datalog)


def adorn_code(query, datalog):
    """
    Produce the rewritten datalog program according to the
    Magic Sets technique.

    :param query Implication: query to solve
    :param datalog DatalogBasic: processed datalog program.

    Returns
    -------
    Union

        adorned code where the query rule is the first expression
        in the block.
    """

    adornment = ''
    for a in query.args:
        if isinstance(a, Symbol):
            adornment += 'f'
        else:
            adornment += 'b'

    query = AdornedExpression(query.functor, adornment, 0)(*query.args)
    adorn_stack = [query]

    edb = datalog.extensional_database()
    idb = datalog.intensional_database()
    rewritten_program = []
    rewritten_rules = set()

    while adorn_stack:
        consequent = adorn_stack.pop()

        if isinstance(consequent.functor, AdornedExpression):
            adornment = consequent.functor.adornment
            name = consequent.functor.expression.name
        else:
            adornment = ''
            name = consequent.functor.name

        rules = idb.get(name, None)
        if rules is None:
            continue

        for rule in rules.formulas:
            adorned_antecedent, to_adorn = adorn_antecedent(
                rule, adornment,
                edb, rewritten_rules
            )
            adorn_stack += to_adorn
            new_consequent = consequent.functor(*rule.consequent.args)
            rewritten_program.append(
                Implication(new_consequent, adorned_antecedent)
            )
            rewritten_rules.add(consequent.functor)

    return Union(rewritten_program)


def adorn_antecedent(
    rule, adornment, edb, rewritten_rules
):
    consequent = rule.consequent
    antecedent = rule.antecedent
    to_adorn = []

    bound_variables = {
        arg for arg, ad in zip(consequent.args, adornment)
        if isinstance(arg, Symbol) and ad == 'b'
    }

    predicates = extract_logic_predicates(antecedent)
    checked_predicates = {}
    adorned_antecedent = None

    for predicate in predicates:
        if (
            (
                isinstance(predicate.functor, Constant) or
                predicate.functor.name in edb
            ) and
            len(bound_variables.intersection(predicate.args)) > 0
        ):
            bound_variables.update(
                arg for arg in predicate.args
                if isinstance(arg, Symbol)
            )

    for predicate in predicates:
        predicate_number = checked_predicates.get(predicate, 0)
        checked_predicates[predicate] = predicate_number + 1
        in_edb = (
            isinstance(predicate.functor, Constant) or
            predicate.functor.name in edb
        )

        adorned_predicate = adorn_predicate(
            predicate, bound_variables, predicate_number, in_edb
        )

        is_adorned = isinstance(adorned_predicate.functor, AdornedExpression)
        if (
            not in_edb and is_adorned and
            adorned_predicate.functor not in rewritten_rules
        ):
            to_adorn.append(adorned_predicate)

        if adorned_antecedent is None:
            adorned_antecedent = (adorned_predicate,)
        else:
            adorned_antecedent += (adorned_predicate,)

    if len(adorned_antecedent) == 1:
        adorned_antecedent = adorned_antecedent[0]
    else:
        adorned_antecedent = Conjunction(adorned_antecedent)
    return adorned_antecedent, to_adorn


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

    p = AdornedExpression(predicate.functor, adornment, predicate_number)
    return p(*predicate.args)
