'''
Magic Sets [1] rewriting implementation for Datalog.

[1] F. Bancilhon, D. Maier, Y. Sagiv, J. D. Ullman, in ACM PODS ’86, pp. 1–15.
'''

from typing import Iterable, Tuple, Type
from ..config import config
from ..expressions import Constant, Expression, Symbol
from ..logic import Negation
from ..probabilistic.expressions import Condition, ProbabilisticQuery
from ..type_system import Unknown
from . import expression_processing, extract_logic_predicates, DatalogProgram
from .exceptions import (
    BoundAggregationApplicationError,
    NegationInMagicSetsRewriteError,
    NonConjunctiveAntecedentInMagicSetsError,
)
from .expressions import (
    AggregationApplication,
    Conjunction,
    Implication,
    Union,
)

class AdornedSymbol(Symbol):
    def __init__(self, expression, adornment, number):
        self.expression = expression
        self.adornment = adornment
        self.number = number
        self._symbols = {self}
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
            isinstance(other, AdornedSymbol) and
            self.unapply() == other.unapply()
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

        if config.expression_type_printing():
            return (
                f'S{{{rep}{superindex}{subindex}: '
                f'{self.__type_repr__}}}'
            )
        else:
            return f'S{{{rep}{superindex}{subindex}}}'


class SIPS:
    """
    Sideways Information Passing Strategy (SIPS). A SIPS defines how
    bounded variables are passed from the head of a rule to the body
    predicates of the rule. SIPS formally describe what information
    (bounded variables) is passed by one literal, or a conjunction of
    literals, to another literal.

    This default SIPS considers that for a body predicate P, and head
    predicate H with adornment a, a variable v of P is bound iif it
    corresponds to a bound variable of H in a.
    """

    def __init__(self, rule, adornment, edb) -> None:
        self.rule = rule
        self.adornment = adornment
        self.edb = edb
        self.bound_variables = {
            arg
            for arg, ad in zip(rule.consequent.args, adornment)
            if isinstance(arg, Symbol) and ad == "b"
        }
        bounded_aggregates = {
            arg
            for arg, ad in zip(rule.consequent.args, adornment)
            if isinstance(arg, AggregationApplication) and ad == "b"
        }
        if len(bounded_aggregates) > 0:
            bounded_aggregate = AdornedSymbol(rule.consequent.functor, adornment, None)
            bounded_aggregate = bounded_aggregate(*rule.consequent.args)
            raise BoundAggregationApplicationError(
                "Magic Sets rewrite would lead to aggregation application"
                " being bound. Problematic adorned expression is: "
                f"{bounded_aggregate}"
            )

    def adorn_predicate(self, predicate, predicate_number, in_edb):
        adornment = ""
        has_b = False
        for arg in predicate.args:
            if isinstance(arg, Constant) or arg in self.bound_variables:
                adornment += "b"
                has_b = True
            else:
                adornment += "f"

        if in_edb and has_b:
            adornment = "b" * len(adornment)

        if not has_b:
            adornment = ""

        p = AdornedSymbol(predicate.functor, adornment, predicate_number)
        return p(*predicate.args)


class LeftToRightSIPS(SIPS):
    """
    LeftToRightSIPS which corresponds to the default SIPS as specified in
    Balbin et al. [1].

    For a given body predicate P and head predicate H with adornment a,
    a variable v of P is bound iif:
        - it corresponds to a bound variable of H in a.
        - or it is a variable of a positive body literal left of P in the
          rule.

    .. [1] Isaac Balbin, Graeme S. Port, Kotagiri Ramamohanarao,
       Krishnamurthy Meenakshi. 1991. Efficient Bottom-UP Computation of
       Queries on Stratified Databases. J. Log. Program. 11(3&4). p. 305.
    """

    def __init__(self, rule, adornment, edb) -> None:
        super().__init__(rule, adornment, edb)

    def adorn_predicate(self, predicate, predicate_number, in_edb):
        if in_edb or isinstance(predicate.functor, Constant):
            return predicate
        adorned_predicate = super().adorn_predicate(
            predicate, predicate_number, in_edb
        )
        self.bound_variables.update(
            arg for arg in predicate.args if isinstance(arg, Symbol)
        )
        return adorned_predicate


class CeriSIPS(SIPS):
    """
    In this SIPS, the initial set of bounded variables in the head predicate H
    with adornment a is augmented with all the variables of EDB predicates and
    constants of the rule.
    """

    def __init__(self, rule, adornment, edb) -> None:
        super().__init__(rule, adornment, edb)

        for predicate in extract_logic_predicates(rule.antecedent):
            if (
                isinstance(predicate.functor, Constant)
                or predicate.functor.name in edb
            ) and len(self.bound_variables.intersection(predicate.args)) > 0:
                self.bound_variables.update(
                    arg for arg in predicate.args if isinstance(arg, Symbol)
                )


def magic_rewrite_ceri(
    query: Expression, datalog: DatalogProgram
) -> Tuple[Symbol, Union]:
    """
    Implementation of the Magic Sets method of optimization for datalog
    programs using Ceri et al [1] algorithm.

    .. [1] Stefano Ceri, Georg Gottlob, and Letizia Tanca. 1990.
       Logic programming and databases. Springer-Verlag, Berlin, Heidelberg.
       p. 168.

    query : Expression
        the head symbol of the query rule in the program
    datalog : DatalogProgram
        a processed datalog program to optimize

    Returns
    -------
    Tuple[Symbol, Union]
        the rewritten query symbol and program
    """
    adorned_code, _ = reachable_adorned_code(query, datalog, CeriSIPS)
    # assume that the query rule is the last
    adorned_query = adorned_code.formulas[-1]
    goal = adorned_query.consequent.functor

    idb = datalog.intensional_database()
    edb = datalog.extensional_database()
    magic_rules = create_magic_rules(adorned_code, idb, edb)
    modified_rules = create_modified_rules(adorned_code, edb)
    complementary_rules = create_complementary_rules(adorned_code, idb)
    return goal, Union(magic_rules + modified_rules + complementary_rules)


def magic_rewrite(
    query: Expression, datalog: DatalogProgram
) -> Tuple[Symbol, Union]:
    """
    Implementation of the Magic Sets method of optimization for datalog
    programs using Balbin et al [1] algorithm.

    .. [1] Isaac Balbin, Graeme S. Port, Kotagiri Ramamohanarao,
       Krishnamurthy Meenakshi. 1991. Efficient Bottom-UP Computation of
       Queries on Stratified Databases. J. Log. Program. 11(3&4). p. 311.

    query : Expression
        the head symbol of the query rule in the program
    datalog : DatalogProgram
        a processed datalog program to optimize

    Returns
    -------
    Tuple[Symbol, Union]
        the rewritten query symbol and program
    """
    adorned_code, constant_predicates = reachable_adorned_code(
        query, datalog, LeftToRightSIPS
    )
    # assume that the query rule is the last
    adorned_query = adorned_code.formulas[-1]
    goal = adorned_query.consequent.functor

    magic_rules = create_balbin_magic_rules(adorned_code.formulas[:-1])
    magic_inits = create_magic_query_inits(constant_predicates)
    return goal, Union(
        magic_inits + [adorned_query] + magic_rules,
    )


def create_magic_query_inits(constant_predicates: Iterable[AdornedSymbol]):
    """
    Create magic initialization predicates from the set of adorned predicates
    with at least one argument constant, according to Balbin et al.'s magic
    set algorithm.
    For each adorned predicate p^a(t) in the set, return an
    initialization rule of the form magic_p^a(t_d) :- True, where t_d is the
    vector of arguments which are bound in the adornment a of p.

    Parameters
    ----------
    constant_predicates : Iterable[AdornedExpression]
        the set of adorned predicates where at least one argument is a constant

    Returns
    -------
    Iterable[Expression]
        the set of magic initialization rules
    """
    magic_init_rules = []
    for predicate in constant_predicates:
        magic_init_rules.append(
            Implication(
                magic_predicate(predicate, adorned=False),
                Constant(True),
            )
        )
    return magic_init_rules


def create_balbin_magic_rules(adorned_rules):
    """
    Create the set of Magic Set rules according to Algorithm 2 of
    Balbin et al.

    This method creates the ensemble Pm of rewritten rules from the ensemble
    P^a of adorned rules in the program (not including the adorned query rule).

    The pseudo-code algorithm for this method is:

    ```
    for each adorned rule Ra in P^a of the form head :- body
        add the rule head :- Magic(head), body to Pm
        for each adorned predicate p in body :
            add the rule Magic(p) :- Magic(h), ^(body predicates left of p)
    ```
    """
    magic_rules = []
    for rule in adorned_rules:
        consequent = rule.consequent
        if isinstance(rule.antecedent, Condition):
            magic_rules.append(rule)
            continue
        magic_head = magic_predicate(consequent, adorned=False)
        if len(magic_head.args) == 0:
            magic_rules.append(rule)
            continue
        body_predicates = (magic_head,)
        for predicate in extract_logic_predicates(rule.antecedent):
            functor = predicate.functor
            if isinstance(functor, AdornedSymbol) and isinstance(
                functor.expression, Constant
            ):
                body_predicates += (functor.expression(*predicate.args),)
            elif (
                isinstance(functor, AdornedSymbol)
                and "b" in functor.adornment
            ):
                new_predicate = magic_predicate(predicate, adorned=False)
                if not (
                    len(body_predicates) == 1
                    and new_predicate == body_predicates[0]
                ):
                    # avoid adding rules of the form magic_p(x) :- magic_p(x)
                    magic_rules.append(
                        Implication(
                            new_predicate, Conjunction(body_predicates)
                        )
                    )
                body_predicates += (predicate,)
            else:
                body_predicates += (predicate,)

        magic_rules.append(
            Implication(consequent, Conjunction(body_predicates))
        )

    return magic_rules


def create_complementary_rules(adorned_code, idb):
    complementary_rules = []
    for i, rule in enumerate(adorned_code.formulas):
        for predicate in extract_logic_predicates(rule.antecedent):
            if (
                not (
                    isinstance(predicate.functor, AdornedSymbol) and
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
            isinstance(functor, AdornedSymbol) and
            'b' in functor.adornment
        ):
            predicate = Symbol(predicate.functor.name)(*predicate.args)
            edb_antecedent.append(predicate)
    return edb_antecedent


def create_magic_rules_create_rules(new_antecedent, predicates, idb, i):
    magic_rules = []
    for predicate in predicates:
        functor = predicate.functor
        is_adorned = isinstance(functor, AdornedSymbol)
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
            isinstance(functor, AdornedSymbol) and
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


def magic_predicate(predicate, i=None, adorned=True):
    """
    Given a predicate of the form P^bf(x, y), create a magic predicate of the
    form magic_P^bf(x).
    If adorned is True, the magic predicate returned is an AdornedSymbol,
    otherwise it is a Symbol with the name magic_P^bf.

    Parameters
    ----------
    predicate : Expression
        a predicate with an adorned symbol
    i : int, optional
        a rule number, by default None
    adorned : bool, optional
        whether to return an AdornedSymbol, by default True

    Returns
    -------
    Expression
        a new predicate.
    """
    name = predicate.functor.name
    adornment = predicate.functor.adornment
    if i is not None:
        new_name = f"magic_r{i}_{name}"
    else:
        new_name = f"magic_{name}"

    new_args = [arg for arg, ad in zip(predicate.args, adornment) if ad == "b"]
    if adorned:
        new_functor = AdornedSymbol(
            Symbol(new_name), adornment, predicate.functor.number
        )
    else:
        superindex = f"^{adornment}" if len(adornment) > 0 else ""
        subindex = (
            f"_{predicate.functor.number}"
            if predicate.functor.number is not None
            else ""
        )
        new_name = f"{new_name}{superindex}{subindex}"
        new_functor = Symbol(new_name)
    return new_functor(*new_args)


def reachable_adorned_code(query, datalog, sips_class: Type[SIPS]):
    adorned_code, constant_predicates = adorn_code(query, datalog, sips_class)
    adorned_datalog = type(datalog)()
    adorned_datalog.walk(adorned_code)
    # assume that the query rule is the first
    adorned_query = adorned_code.formulas[0]
    return (
        expression_processing.reachable_code(adorned_query, adorned_datalog),
        constant_predicates,
    )


def adorn_code(
    query: Expression, datalog: DatalogProgram, sips_class: Type[SIPS]
) -> Union:
    """
    Produce the rewritten datalog program according to the
    Magic Sets technique.

    Parameters
    ----------
    query : Expression
        query to solve
    datalog : DatalogProgram
        processed datalog program
    sips_class : Type[SIPS]
        the SIPS class to use for adornment of predicates

    Returns
    -------
    Union
        adorned code where the query rule is the first expression
        in the block.
    """
    adornment = ''
    for a in query.args:
        if isinstance(a, Symbol) or isinstance(a, ProbabilisticQuery):
            adornment += 'f'
        else:
            adornment += 'b'

    query = AdornedSymbol(query.functor, adornment, 0)(*query.args)
    adorn_stack = [query]

    edb = edb_with_prob_symbols(datalog)
    idb = datalog.intensional_database()
    rewritten_program = []
    rewritten_rules = set()
    constant_predicates = set()

    while adorn_stack:
        consequent = adorn_stack.pop()

        if isinstance(consequent.functor, AdornedSymbol):
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
                edb, rewritten_rules, sips_class
            )
            adorn_stack += to_adorn
            new_consequent = consequent.functor(*rule.consequent.args)
            rewritten_program.append(
                Implication(new_consequent, adorned_antecedent)
            )
            rewritten_rules.add(consequent.functor)
            for predicate in to_adorn:
                for arg in predicate.args:
                    if isinstance(arg, Constant):
                        constant_predicates.add(predicate)

    return Union(rewritten_program), constant_predicates


def adorn_antecedent(
    rule, adornment, edb, rewritten_rules, sips_class: Type[SIPS]
):
    antecedent = rule.antecedent
    to_adorn = []

    sips = sips_class(rule, adornment, edb)

    predicates = extract_logic_predicates(antecedent)
    checked_predicates = {}
    adorned_antecedent = None

    for predicate in predicates:
        if isinstance(predicate, Negation):
            raise NegationInMagicSetsRewriteError(
                "Magic sets rewrite does not work with negative predicates."
            )
        predicate_number = checked_predicates.get(predicate, 0)
        checked_predicates[predicate] = predicate_number + 1
        in_edb = (
            isinstance(predicate.functor, Constant)
            or predicate.functor.name in edb
        )

        adorned_predicate = sips.adorn_predicate(
            predicate, predicate_number, in_edb
        )

        is_adorned = isinstance(adorned_predicate.functor, AdornedSymbol)
        if (
            not in_edb
            and is_adorned
            and adorned_predicate.functor not in rewritten_rules
        ):
            to_adorn.append(adorned_predicate)

        if adorned_antecedent is None:
            adorned_antecedent = (adorned_predicate,)
        else:
            adorned_antecedent += (adorned_predicate,)

    if len(adorned_antecedent) == 1:
        adorned_antecedent = adorned_antecedent[0]
    elif len(adorned_antecedent) == 2 and isinstance(antecedent, Condition):
        adorned_antecedent = Condition(
            adorned_antecedent[0], adorned_antecedent[1]
        )
    elif not isinstance(antecedent, Conjunction):
        raise NonConjunctiveAntecedentInMagicSetsError(
            "Magic Set rewrite does not work with "
            f"non-conjunctive antecedent {antecedent}"
        )
    else:
        adorned_antecedent = Conjunction(adorned_antecedent)
    return adorned_antecedent, to_adorn


def edb_with_prob_symbols(datalog: DatalogProgram) -> Iterable[Symbol]:
    edb = set(datalog.extensional_database())
    try:
        edb = (
            edb
            | set(datalog.probabilistic_facts())
            | set(datalog.probabilistic_choices())
        )
    except AttributeError:
        pass
    return edb
