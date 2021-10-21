"""
Magic Sets [1] rewriting implementation for Datalog.

[1] F. Bancilhon, D. Maier, Y. Sagiv, J. D. Ullman, in ACM PODS ’86, pp. 1–15.
"""

from abc import ABC, abstractmethod
from typing import Iterable, List, Set, Tuple
from ..config import config
from ..expressions import Constant, Expression, FunctionApplication, Symbol
from ..expression_walker import ExpressionWalker
from ..expression_pattern_matching import add_match
from ..logic import Negation
from ..probabilistic.expressions import ProbabilisticQuery
from . import expression_processing, extract_logic_predicates, DatalogProgram
from .exceptions import (
    BoundAggregationApplicationError,
    NegationInMagicSetsRewriteError,
    NonConjunctiveAntecedentInMagicSetsError,
    NoConstantPredicateFoundError,
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

    @property
    def name(self):
        if isinstance(self.expression, Symbol):
            return self.expression.name
        else:
            raise NotImplementedError()

    def __eq__(self, other):
        return (
            hash(self) == hash(other)
            and isinstance(other, AdornedSymbol)
            and self.unapply() == other.unapply()
        )

    def __hash__(self):
        return hash((self.expression, self.adornment, self.number))

    def __str__(self) -> str:
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
        return f'{rep}{superindex}{subindex}'

    def __repr__(self):
        if config.expression_type_printing():
            return (
                f'S{{{self}: '
                f'{self.__type_repr__}}}'
            )
        else:
            return f'S{{{self}}}'


class ReplaceAdornedSymbolWalker(ExpressionWalker):
    @add_match(Symbol)
    def replace_adorned_symbol(self, symbol):
        if isinstance(symbol, AdornedSymbol) and isinstance(
            symbol.expression, Symbol
        ):
            superindex = (
                f"^{symbol.adornment}" if len(symbol.adornment) > 0 else ""
            )
            subindex = f"_{symbol.number}" if symbol.number is not None else ""
            new_name = f"{symbol.name}{superindex}{subindex}"
            return Symbol(new_name)
        else:
            return symbol

    @add_match(Implication)
    def implication(self, exp):
        return Implication(
            self.walk(exp.consequent), self.walk(exp.antecedent)
        )


class SIPS(ABC):
    """
    Sideways Information Passing Strategy (SIPS). A SIPS defines how
    bound variables are passed from the head of a rule to the body
    predicates of the rule. SIPS formally describe what information
    (bound variables) is passed by one literal, or a conjunction of
    literals, to another literal.
    """

    def __init__(self, datalog) -> None:
        self.datalog = datalog
        self.edb = edb_with_prob_symbols(datalog)
        try:
            self.prob_symbols = datalog.probabilistic_predicate_symbols
        except AttributeError:
            self.prob_symbols = set()

    def _check_for_bound_aggregates(self, rule, adornment):
        """
        Check if there are aggregate functions in the consequent's arguments
        and raise an exception if these arguments are bound in the adornment.
        """
        bound_aggregates = {
            arg
            for arg, ad in zip(rule.consequent.args, adornment)
            if isinstance(arg, AggregationApplication) and ad == "b"
        }
        if len(bound_aggregates) > 0:
            bound_aggregate = AdornedSymbol(
                rule.consequent.functor, adornment, None
            )
            bound_aggregate = bound_aggregate(*rule.consequent.args)
            raise BoundAggregationApplicationError(
                "Magic Sets rewrite would lead to aggregation application"
                " being bound. Problematic adorned expression is: "
                f"{bound_aggregate}"
            )

    def creates_arcs(self, rule, adorned_head):
        """
        For a given rule and an adorned head predicate, create the
        arcs corresponding to this SIPS.

        For each predicate in the rule's antecedent, this method will try to
        create an arc by calling `self._create_arc`, which should return None
        if no arc can be created for this predicate, or return the head and
        tail of the arc to be added.

        Returns
        -------
        Union[Dict[Expression, Tuple[Expression]], Tuple[Expression]]
            returns the created arcs as a dict mapping of head -> tails and
            the adorned antecedent of the rule
        """
        self._check_for_bound_aggregates(rule, adorned_head.functor.adornment)

        tail_predicates = [adorned_head]
        bound_variables = {
            arg
            for arg, ad in zip(
                adorned_head.args, adorned_head.functor.adornment
            )
            if isinstance(arg, Symbol) and ad == "b"
        }
        arcs = dict()
        adorned_antecedent = ()
        checked_predicates = {}

        predicates = extract_logic_predicates(rule.antecedent)
        for predicate in predicates:
            predicate_number = checked_predicates.get(predicate, 0)
            checked_predicates[predicate] = predicate_number + 1
            arc = self._create_arc(
                    predicate,
                    predicate_number,
                    bound_variables,
                    tail_predicates,
                )
            if arc is not None:
                tail, head = arc
                arcs[head] = tail
                adorned_antecedent += (head,)
            else:
                adorned_antecedent += (predicate,)
        return arcs, adorned_antecedent

    def _adorn_predicate(
        self, predicate, predicate_number, bound_variables
    ):
        adornment = ""
        has_b = False
        for arg in predicate.args:
            if isinstance(arg, Constant) or arg in bound_variables:
                adornment += "b"
                has_b = True
            else:
                adornment += "f"

        if not has_b:
            adornment = ""

        p = AdornedSymbol(predicate.functor, adornment, predicate_number)
        p = p(*predicate.args)
        return p

    @abstractmethod
    def _create_arc(
        self,
        predicate: Expression,
        predicate_number: int,
        bound_variables: Set[Symbol],
        adorned_predicates: List[Expression],
    ) -> Union[Tuple[Expression], Expression, None]:
        """
        Create an arc for the given predicate. An arc is a tuple (tail, head)
        where tail is a tuple of adorned expressions, and head is the input
        predicate adorned with the bound variables for this arc.
        This method should return None if no arc should be created for the
        given predicate.
        """
        pass


class LeftToRightSIPS(SIPS):
    """
    LeftToRightSIPS which corresponds to the default SIPS as specified in
    Balbin et al. [1].

    For a given body predicate P and head predicate H with adornment a,
    a variable v of P is bound iif:
        - it corresponds to a bound variable of H in a.
        - or it is a variable of a positive non probabilistic body literal
        left of P in the rule.

    .. [1] Isaac Balbin, Graeme S. Port, Kotagiri Ramamohanarao,
       Krishnamurthy Meenakshi. 1991. Efficient Bottom-UP Computation of
       Queries on Stratified Databases. J. Log. Program. 11(3&4). p. 305.
    """

    def __init__(self, datalog) -> None:
        super().__init__(datalog)

    def _create_arc(
        self,
        predicate: Expression,
        predicate_number: int,
        bound_variables: Set[Symbol],
        tail_predicates: List[Expression],
    ) -> Union[Tuple[Expression], Expression]:
        """
        For each arc, the tail of the arc is composed of all the predicates
        that have been adorned before it in the rule.
        This is achieved by updating the list of tail_predicates with each new
        adorned predicate.
        """
        # 1. unwrap negative predicates
        is_neg = isinstance(predicate, Negation)
        if is_neg:
            pred = predicate.formula
            if not hasattr(pred, "functor"):
                raise NegationInMagicSetsRewriteError(
                    "MagicSets can only handle negations on single predicates."
                    f"Negation {predicate} cannot be handled"
                )
        else:
            pred = predicate
        # 2. Constants and predicates in the EDB are already bound and should
        # not be adorned.
        if isinstance(pred.functor, Constant) or pred.functor.name in self.edb:
            return None

        # 3. Adorn the predicate
        p = self._adorn_predicate(pred, predicate_number, bound_variables)
        tail = tuple(tail_predicates)

        # 4. Update the tail_predicates and list of bound_variables with the
        # variables of this predicate. We only add bound variables for
        # positive, non probabilistic predicates
        if p.functor.name not in self.prob_symbols and not is_neg:
            bound_variables.update(
                arg for arg in predicate.args if isinstance(arg, Symbol)
            )
            tail_predicates.append(p)
        
        if is_neg:
            p = Negation(p)
        return tail, p


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
    # 1. Create a SIPS and use it to adorn the code
    sips = LeftToRightSIPS(datalog)
    adorned_code, constant_predicates = reachable_adorned_code(
        query, datalog, sips
    )
    # 2. Check that there is a constant in the code
    if len(constant_predicates) == 0:
        # No constants present in the code, magic sets is not usefull
        raise NoConstantPredicateFoundError(
            "No predicate with constant argument found."
        )
    # assume that the query rule is the last
    adorned_query = adorned_code.formulas[-1]

    # 3. Create magic rules
    magic_rules = create_balbin_magic_rules(
        adorned_code.formulas[:-1], sips
    )
    magic_inits = create_magic_query_inits(constant_predicates)

    # 4. Unadorn the magic rules + query to replace AdornedSymbols by
    # regular symbols (makes it easier to check symbol equality later on)
    unadorned_code = ReplaceAdornedSymbolWalker().walk(
        tuple(magic_inits) + tuple(magic_rules) + (adorned_query,)
    )
    # Ordered of the rules should be preserved, so the unadorned query should
    # be the last of the unadorned_code
    goal = unadorned_code[-1].consequent.functor
    return goal, Union(unadorned_code)


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


def create_balbin_magic_rules(adorned_rules, sips):
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
        magic_head = magic_predicate(consequent, adorned=False)
        if len(magic_head.args) == 0:
            magic_rules.append(rule)
            continue

        arcs, adorned_antecedent = sips.creates_arcs(rule, consequent)

        # Add the rule head :- magic(head), body
        magic_rules.append(
            Implication(
                rule.consequent,
                Conjunction((magic_head,) + adorned_antecedent),
            )
        )
        # Add the magic rules for each arc of the sips
        for head, tail in arcs.items():
            if isinstance(head, Negation):
                head = head.formula
            if "b" in head.functor.adornment:
                new_predicate = magic_predicate(head, adorned=False)
                body_lits = [
                    p if p != consequent else magic_head for p in tail
                ]
                if not (len(body_lits) == 1 and new_predicate == body_lits[0]):
                    # avoid adding rules of the form magic_p(x) :- magic_p(x)
                    magic_rules.append(
                        Implication(new_predicate, Conjunction(body_lits))
                    )

    return magic_rules


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


def reachable_adorned_code(query, datalog, sips: SIPS):
    adorned_code, constant_predicates = adorn_code(query, datalog, sips)
    adorned_datalog = type(datalog)()
    adorned_datalog.walk(adorned_code)
    # assume that the query rule is the first
    adorned_query = adorned_code.formulas[0]
    return (
        expression_processing.reachable_code(adorned_query, adorned_datalog),
        constant_predicates,
    )


def adorn_code(
    query: Expression, datalog: DatalogProgram, sips: SIPS
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
    sips_class : SIPS
        the SIPS instance to use for adornment of predicates

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
            new_consequent = consequent.functor(*rule.consequent.args)
            adorned_antecedent, to_adorn = adorn_antecedent(
                rule, new_consequent, rewritten_rules, sips
            )
            adorn_stack += to_adorn
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
    rule, adorned_head, rewritten_rules, sips: SIPS
):
    to_adorn = []
    arcs, adorned_antecedent = sips.creates_arcs(rule, adorned_head)
    for adorned_predicate in arcs.keys():
        if isinstance(adorned_predicate, Negation):
            adorned_predicate = adorned_predicate.formula
        if adorned_predicate.functor not in rewritten_rules:
            to_adorn.append(adorned_predicate)

    antecedent = rule.antecedent

    if len(adorned_antecedent) == 1:
        adorned_antecedent = adorned_antecedent[0]
    elif not isinstance(antecedent, Conjunction):
        raise NonConjunctiveAntecedentInMagicSetsError(
            "Magic Set rewrite does not work with "
            f"non-conjunctive antecedent {antecedent}"
        )
    else:
        adorned_antecedent = Conjunction(adorned_antecedent)
    return adorned_antecedent, to_adorn


def edb_with_prob_symbols(datalog: DatalogProgram) -> Iterable[Symbol]:
    edb = set(datalog.extensional_database()) | set(datalog.included_functions)
    try:
        edb = (
            edb
            | set(datalog.probabilistic_facts())
            | set(datalog.probabilistic_choices())
        )
    except AttributeError:
        pass
    return edb
