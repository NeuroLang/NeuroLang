"""
Magic Sets [1] rewriting implementation for Datalog.

[1] F. Bancilhon, D. Maier, Y. Sagiv, J. D. Ullman, in ACM PODS ’86, pp. 1–15.
"""

from abc import ABC, abstractmethod
from typing import Iterable, List, Set, Tuple


from ..expressions import Constant, Expression, FunctionApplication, Symbol
from ..expression_walker import ExpressionWalker
from ..expression_pattern_matching import add_match
from ..logic import Negation
from ..probabilistic.expressions import ProbabilisticQuery
from . import extract_logic_predicates, DatalogProgram
from .exceptions import (
    BoundAggregationApplicationError,
    NegationInMagicSetsRewriteError,
    NonConjunctiveAntecedentInMagicSetsError,
    NoConstantPredicateFoundError,
)
from .expressions import (
    AdornedSymbol,
    AggregationApplication,
    Conjunction,
    Implication,
    Union,
)
from .constraints_representation import (
    RightImplication,
    get_list_of_intensional_rules,
    reachable_code,
)
from .expression_processing import (
    any_body_preds_in_set,
    extract_logic_free_variables,
    order_conjunction_by_shared_variables,
)


def order_conjunction_constant_atoms_first(
    formulas: Tuple[Expression, ...],
) -> Tuple[Expression, ...]:
    """
    Order a rule body so constant-bearing atoms are adorned first.

    Magic sets adorns rule bodies left to right; a constant-bearing atom
    only seeds the magic set (``magic_P(c) :- ()``) while its shared
    variables are still free, i.e. when no atom sharing them precedes it.
    Bodies where a constant-bearing atom would be adorned with an already
    bound shared variable are reordered: constant-bearing atoms first,
    in their original relative order, then the remaining atoms joined
    greedily on shared variables (``order_conjunction_by_shared_variables``),
    keeping every magic init rule range restricted -- no free variable
    in a seed consequent.

    Only flat conjunctions of predicate applications are considered;
    bodies with negation or other non-predicate formulas, and bodies
    whose constant atoms share nothing with earlier atoms, are returned
    unchanged.
    """
    if not all(isinstance(f, FunctionApplication) for f in formulas):
        return formulas
    constant_atoms = [
        f for f in formulas
        if any(isinstance(arg, Constant) for arg in f.args)
    ]
    if not constant_atoms:
        return formulas
    prior_vars = set()
    for f in formulas:
        f_vars = extract_logic_free_variables(f)
        if f in constant_atoms and f_vars & prior_vars:
            break
        prior_vars |= f_vars
    else:
        return formulas
    free_atoms = [
        f for f in formulas
        if not any(isinstance(arg, Constant) for arg in f.args)
    ]
    return tuple(constant_atoms) + order_conjunction_by_shared_variables(
        tuple(free_atoms)
    )


def unadorned_predicate_symbol(symbol: Symbol) -> Symbol:
    """
    Return the predicate symbol ``symbol`` is a magic-sets adornment of.

    Magic-sets adornments are numbered ``AdornedSymbol`` instances, and
    adorning an adorned rule nests them. Other adorned symbols -- such as
    the ``q^cond_num`` of the conditional query sugar, whose number is
    None -- are predicate symbols of their own.
    """
    while isinstance(symbol, AdornedSymbol) and symbol.number is not None:
        symbol = symbol.expression
    return symbol


def adorned_base_name(symbol: Symbol) -> str:
    """
    Name the predicate symbol of ``symbol`` as the base of an adornment.

    A predicate symbol which is itself an ``AdornedSymbol``, such as
    ``q^cond_num``, is named with its own adornment, so that adorning it
    does not give it the name of an adornment of ``q``.
    """
    symbol = unadorned_predicate_symbol(symbol)
    if isinstance(symbol, AdornedSymbol):
        return str(symbol)
    return symbol.name


class ReplaceAdornedSymbolWalker(ExpressionWalker):
    @add_match(Symbol)
    def replace_adorned_symbol(self, symbol):
        if isinstance(symbol, AdornedSymbol) and isinstance(
            symbol.expression, Symbol
        ) and symbol.number is not None:
            superindex = (
                f"^{symbol.adornment}" if len(symbol.adornment) > 0 else ""
            )
            subindex = f"_{symbol.number}"
            base = unadorned_predicate_symbol(symbol.expression)
            new_symbol = Symbol(f"{base.name}{superindex}{subindex}")
            if isinstance(base, AdornedSymbol):
                # keep the adornment of the predicate symbol, e.g. cond_num
                new_symbol = AdornedSymbol(
                    new_symbol, base.adornment, base.number
                )
            return new_symbol
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

    def __init__(self, datalog, seed_from_edb=True) -> None:
        self.datalog = datalog
        self.edb = edb_with_prob_symbols(datalog)
        self.prob_symbols = probabilistic_symbols(datalog)
        self.seed_from_edb = seed_from_edb
        self.seeding_edb_atoms = set()

    def _check_for_bound_aggregates(self, rule, adornment):
        """
        Check if there are aggregate functions, or probabilistic queries,
        in the consequent's arguments and raise an exception if these
        arguments are bound in the adornment.
        """
        bound_aggregates = {
            arg
            for arg, ad in zip(rule.consequent.args, adornment)
            if isinstance(arg, (AggregationApplication, ProbabilisticQuery))
            and ad == "b"
        }
        if len(bound_aggregates) > 0:
            bound_aggregate = AdornedSymbol(
                rule.consequent.functor, adornment, None
            )
            bound_aggregate = bound_aggregate(*rule.consequent.args)
            raise BoundAggregationApplicationError(
                "Magic Sets rewrite would lead to aggregation application"
                " or probabilistic query being bound. Problematic adorned"
                " expression is: "
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
        tail_predicates: List[Expression],
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
        - or, if ``seed_from_edb``, it is a variable of a positive literal
        of a deterministic relation of the extensional database with a
        constant argument, left of P in the rule, as ``d`` in
        ``task(t, "stroop", d) & p(d, x)``. That literal joins the
        bodies of the magic rules of the literals right of it.

    Probabilistic literals are those of probabilistic facts and choices,
    of within-language probabilistic queries, and of the predicates
    depending on probabilistic facts or choices: magic predicates are thus
    deterministic, and restricting a probabilistic rule with them does
    not change its probabilities.

    .. [1] Isaac Balbin, Graeme S. Port, Kotagiri Ramamohanarao,
       Krishnamurthy Meenakshi. 1991. Efficient Bottom-UP Computation of
       Queries on Stratified Databases. J. Log. Program. 11(3&4). p. 305.
    """

    def __init__(self, datalog, seed_from_edb=True) -> None:
        super().__init__(datalog, seed_from_edb)

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
        pred, is_neg = _unwrap_negation(predicate)
        # Constants, constraints and predicates in the EDB are already bound
        # and should not be adorned.
        if self._is_bound_literal(pred):
            if not is_neg and self._seeds_magic_sets(pred):
                self._seed(pred, bound_variables, tail_predicates)
            return None

        p = self._adorn_predicate(pred, predicate_number, bound_variables)
        tail = tuple(tail_predicates)

        # Update the tail_predicates and list of bound_variables with the
        # variables of this predicate. We only add bound variables for
        # positive, non probabilistic predicates
        if (
            unadorned_predicate_symbol(pred.functor) not in self.prob_symbols
            and not is_neg
        ):
            _bind_shared_variables(p, tail, bound_variables)
            tail_predicates.append(p)

        if is_neg:
            p = Negation(p)
        return tail, p

    def _seed(self, predicate, bound_variables, tail_predicates):
        """Bind the variables of *predicate*, a literal of the extensional
        database with a constant, and add it to the tails of the arcs of
        the literals right of it."""
        self.seeding_edb_atoms.add(predicate)
        bound_variables.update(
            arg for arg in predicate.args if isinstance(arg, Symbol)
        )
        tail_predicates.append(predicate)

    def _is_bound_literal(self, predicate):
        """Whether *predicate* is a constant function, a predicate of the
        extensional database or a constraint, which are not adorned."""
        return (
            isinstance(predicate.functor, Constant) or
            predicate.functor.name in self.edb or
            (
                hasattr(self.datalog, 'categorized_constraints') and
                predicate.functor in self.datalog.categorized_constraints
            )
        )

    def _seeds_magic_sets(self, predicate):
        """
        Whether *predicate*, a positive literal, binds its variables: it
        is of a deterministic relation of the extensional database, not a
        function, with a constant argument.
        """
        return (
            self.seed_from_edb and
            isinstance(predicate.functor, Symbol) and
            predicate.functor in self.datalog.extensional_database() and
            predicate.functor not in self.prob_symbols and
            any(isinstance(arg, Constant) for arg in predicate.args)
        )


def _bind_shared_variables(predicate, tail, bound_variables):
    """Bind the variables of *predicate* which are in the *tail*."""
    tail_variables = {arg for t in tail for arg in t.args}
    bound_variables.update(
        arg for arg in predicate.args
        if isinstance(arg, Symbol) and arg in tail_variables
    )


def _unwrap_negation(predicate):
    """The predicate of the literal *predicate* and whether it is negated."""
    if not isinstance(predicate, Negation):
        return predicate, False
    if not hasattr(predicate.formula, "functor"):
        raise NegationInMagicSetsRewriteError(
            "MagicSets can only handle negations on single predicates."
            f"Negation {predicate} cannot be handled"
        )
    return predicate.formula, True


def magic_rewrite(
    query: Expression, datalog: DatalogProgram, seed_from_edb: bool = True
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
    seed_from_edb : bool
        whether a literal of the extensional database with a constant
        binds the variables of the literals right of it, see
        :class:`LeftToRightSIPS`

    Returns
    -------
    Tuple[Symbol, Union]
        the rewritten query symbol and program
    """
    # 1. Create a SIPS and use it to adorn the code
    sips = LeftToRightSIPS(datalog, seed_from_edb)
    adorned_code, constant_predicates = reachable_adorned_code(
        query, datalog, sips
    )
    # 2. Check that there is a constant in the code
    if len(constant_predicates) == 0 and len(sips.seeding_edb_atoms) == 0:
        # No constants present in the code, magic sets is not useful
        raise NoConstantPredicateFoundError(
            "No predicate with constant argument found."
        )
    # assume that the query rule is the last
    adorned_query = adorned_code.formulas[-1]

    # 3. Create magic rules
    magic_rules = create_balbin_magic_rules(
        adorned_code.formulas[:-1], sips
    )
    query_arcs, _ = sips.creates_arcs(adorned_query, adorned_query.consequent)
    magic_rules += create_magic_arc_rules(
        query_arcs, adorned_query.consequent, None
    )
    magic_inits = create_magic_query_inits(constant_predicates)

    # 4. Unadorn the magic rules + query to replace AdornedSymbols by
    # regular symbols (makes it easier to check symbol equality later on)
    unadorned_code = ReplaceAdornedSymbolWalker().walk(
        tuple(magic_inits) + tuple(magic_rules) + (adorned_query,)
    )
    unadorned_code = order_rules_definitions_first(unadorned_code[:-1]) + (
        unadorned_code[-1],
    )
    # Ordered of the rules should be preserved, so the unadorned query should
    # be the last of the unadorned_code
    goal = unadorned_code[-1].consequent.functor
    return goal, Union(unadorned_code)


def order_rules_definitions_first(rules):
    """
    Order ``rules`` so that predicates are defined before they are used.

    A rule comes after the rules defining the predicates of its body,
    those of recursive predicates aside; otherwise, the order of
    ``rules`` is kept. Walking the rules in this order, the types of
    the predicates of a rule body are known when the rule is walked.
    Unlike :func:`.expression_processing.stratify`, the predicates of
    the bodies not defined in ``rules``, such as probabilistic ones, are
    taken as known, and the rules after a recursive one are still
    ordered.
    """
    pending = list(rules)
    ordered = []
    while pending:
        heads = {rule.consequent.functor for rule in pending}
        ready = [
            rule for rule in pending
            if not any_body_preds_in_set(
                rule, heads - {rule.consequent.functor}
            )
        ] or pending[:1]
        ordered += ready
        pending = [rule for rule in pending if rule not in ready]
    return tuple(ordered)


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
        magic_head = magic_predicate(predicate, adorned=False)
        # bound variables get their magic set from the SIPS arc instead
        if all(isinstance(arg, Constant) for arg in magic_head.args):
            magic_init_rules.append(
                Implication(magic_head, Conjunction(()))
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
        arcs, adorned_antecedent = sips.creates_arcs(rule, consequent)
        if len(magic_head.args) == 0:
            magic_rules.append(rule)
            magic_head = None
        else:
            # Add the rule head :- magic(head), body
            magic_rules.append(
                Implication(
                    rule.consequent,
                    Conjunction((magic_head,) + adorned_antecedent),
                )
            )
        magic_rules += create_magic_arc_rules(arcs, consequent, magic_head)

    return magic_rules


def create_magic_arc_rules(arcs, consequent, magic_head):
    """
    Create the rule Magic(p) :- Magic(h), ^(body predicates left of p)
    for each adorned predicate p heading an arc of the sips.

    When magic_head is None, the rule head h is not bound and Magic(h)
    is left out of the rule bodies. Arcs whose body is then empty are
    skipped, their magic predicate being seeded by the magic
    initialization rules.
    """
    magic_rules = []
    for head, tail in arcs.items():
        if isinstance(head, Negation):
            head = head.formula
        if "b" in head.functor.adornment:
            new_predicate = magic_predicate(head, adorned=False)
            body_lits = [p for p in tail if p != consequent]
            if magic_head is not None:
                body_lits = [magic_head] + body_lits
            elif len(body_lits) == 0:
                continue
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
    name = adorned_base_name(predicate.functor)
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
        reachable_code(adorned_query, adorned_datalog),
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
            predicate_symbol = consequent.functor.expression
        else:
            predicate_symbol = consequent.functor

        rules = idb.get(predicate_symbol, None)
        if rules is None:
            continue

        for rule in rules.formulas:
            if isinstance(rule, RightImplication):
                continue
            if isinstance(rule.antecedent, Conjunction):
                formulas = order_conjunction_constant_atoms_first(
                    rule.antecedent.formulas
                )
                if formulas != rule.antecedent.formulas:
                    rule = rule.apply(
                        rule.consequent, Conjunction(formulas)
                    )
            new_consequent = consequent.functor(*rule.consequent.args)
            adorned_antecedent, predicates = adorn_antecedent(
                rule, new_consequent, sips
            )
            adorn_stack += [
                predicate for predicate in predicates
                if predicate.functor not in rewritten_rules
            ]
            rewritten_program.append(
                Implication(new_consequent, adorned_antecedent)
            )
            rewritten_rules.add(consequent.functor)
            # Predicates already adorned still need the magic init of
            # their constants, e.g. r(2, y) after r(1, y) was adorned.
            for predicate in predicates:
                if any([isinstance(arg, Constant) for arg in predicate.args]):
                    constant_predicates.add(predicate)

    return Union(rewritten_program), constant_predicates


def adorn_antecedent(rule, adorned_head, sips: SIPS):
    predicates = []
    arcs, adorned_antecedent = sips.creates_arcs(rule, adorned_head)
    for adorned_predicate in arcs.keys():
        if isinstance(adorned_predicate, Negation):
            adorned_predicate = adorned_predicate.formula
        predicates.append(adorned_predicate)

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
    return adorned_antecedent, predicates


def probabilistic_symbols(datalog: DatalogProgram) -> Set[Symbol]:
    """
    Return the symbols of the predicates of ``datalog`` which are random.

    These are the within-language probabilistic queries (WLQs), the
    :func:`random_predicate_symbols` of ``datalog`` and the predicates
    depending on them, e.g. ``q`` in ``q(x, p) :- q^cond_num(x, p1) &
    q^cond_den(p2) & (p == p1 / p2)``; none if ``datalog`` is not
    probabilistic. Magic predicates derived from these would not be
    deterministic, and restricting a probabilistic rule by them could
    change probabilities.
    """
    try:
        wlq_symbols = set(datalog.within_language_prob_queries())
    except AttributeError:
        return set()
    return _symbols_depending_on(
        get_list_of_intensional_rules(datalog),
        random_predicate_symbols(datalog) | wlq_symbols,
    )


def random_predicate_symbols(datalog) -> Set[Symbol]:
    """
    Return the symbols of the probabilistic predicates of ``datalog``.

    These are the probabilistic facts and choices, and the intensional
    predicates which depend on probabilistic facts or choices other than
    through a within-language probabilistic query (WLQ). ``datalog`` is a
    probabilistic program: one with probabilistic facts, choices and WLQs.
    """
    random_symbols = datalog.pfact_pred_symbs | datalog.pchoice_pred_symbs
    wlq_symbols = set(datalog.within_language_prob_queries())
    rules = [
        rule for rule in get_list_of_intensional_rules(datalog)
        if rule.consequent.functor not in wlq_symbols
    ]
    return _symbols_depending_on(rules, random_symbols)


def _symbols_depending_on(rules, symbols):
    """Return ``symbols`` and the heads of ``rules`` depending on them."""
    new_symbols = symbols
    while new_symbols:
        new_symbols = {
            rule.consequent.functor for rule in rules
            if any_body_preds_in_set(rule, symbols)
        } - symbols
        symbols = symbols | new_symbols
    return symbols


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
