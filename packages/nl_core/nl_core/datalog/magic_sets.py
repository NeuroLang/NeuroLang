"""
Magic Sets [1] rewriting implementation for Datalog.

[1] F. Bancilhon, D. Maier, Y. Sagiv, J. D. Ullman, in ACM PODS ’86, pp. 1–15.
"""

from collections import Counter, defaultdict
from abc import ABC, abstractmethod
from typing import Iterable, List, Set, Tuple


from nl_expr.expressions import (
    Constant,
    Expression,
    FunctionApplication,
    Symbol,
)
from nl_expr.config import config
from nl_expr.expression_walker import ExpressionWalker
from nl_expr.expression_pattern_matching import (
    NeuroLangPatternMatchingNoMatch,
    add_match,
)
from nl_expr.logic import Negation
from ..probabilistic.expressions import ProbabilisticQuery
from . import extract_logic_predicates, DatalogProgram
from .exceptions import (
    BoundAggregationApplicationError,
    NegationInMagicSetsRewriteError,
    NonConjunctiveAntecedentInMagicSetsError,
    NewRecursionInMagicSetsRewriteError,
    NoConstantPredicateFoundError,
    UnstratifiedMagicSetsRewriteError,
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
from .range_restriction import unbound_variables_error
from .expression_processing import (
    FlattenQueryInNonRecursiveUCQ,
    _body_predicates,
    any_body_preds_in_set,
    extract_logic_free_variables,
    order_conjunction_by_shared_variables,
    recursive_components,
    strongly_connected_components,
)


AGGREGATES = (AggregationApplication, ProbabilisticQuery)


def order_conjunction_constant_atoms_first(
    formulas: Tuple[Expression, ...], bound: Set[Symbol] = frozenset(),
    binding=lambda atom: False, last=lambda atom: 0,
) -> Tuple[Expression, ...]:
    """
    Order a rule body so constant-bearing atoms are adorned first.

    The variables ``bound`` by the adorned head count as constants: in
    ``p(q, i) :- a(i, s), sim(q, s)`` adorned ``p^bf``, ``sim(q, s)`` is
    adorned first and restricts ``a`` to its studies.

    Builtins, such as ``x > 0``, are not constant-bearing atoms: they
    bind nothing. The atoms ``binding`` accepts (the stored relations of
    the query's rule, or those binding a predicate that needs it, see
    :class:`LeftToRightSIPS`) count as constant-bearing atoms. They
    are put in the order of their rank by ``last``, then in their
    original order: the rule ``task(t)`` restricting ``definition(t, d)``,
    then ``definition(t, d)``, then ``sim(d, 'x')``, which needs ``d``.

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
    constant_atoms = sorted((
        f for f in formulas
        if isinstance(f.functor, Symbol)
        and (
            binding(f) or
            any(isinstance(arg, Constant) or arg in bound for arg in f.args)
        )
    ), key=last)
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
    free_atoms = [f for f in formulas if f not in constant_atoms]
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


def program_invariants(datalog) -> dict:
    """What a SIPS reads of the program *datalog*, which a rewrite does not
    change: computed once for the attempts of one rewrite, not once each
    (44 attempts, 25 s of a query of tracts)."""
    return {
        "edb": edb_with_prob_symbols(datalog),
        "prob_symbols": probabilistic_symbols(datalog),
        "aggregate_positions": aggregate_positions(datalog),
        "needed_positions": needed_positions(datalog),
    }


class SIPS(ABC):
    """
    Sideways Information Passing Strategy (SIPS). A SIPS defines how
    bound variables are passed from the head of a rule to the body
    predicates of the rule. SIPS formally describe what information
    (bound variables) is passed by one literal, or a conjunction of
    literals, to another literal.
    """

    def __init__(self, datalog, seed_from_edb=True, invariants=None) -> None:
        self.datalog = datalog
        if invariants is None:
            invariants = program_invariants(datalog)
        self.edb = invariants["edb"]
        self.prob_symbols = invariants["prob_symbols"]
        self.seed_from_edb = seed_from_edb
        self.seeding_edb_atoms = set()
        # the query's predicate (adorn_code sets it), and whether the rule
        # whose arcs are being created is the query's own
        self.query_functor = None
        self.query_is_recursive = False
        self.in_query_rule = False
        self.aggregate_positions = invariants["aggregate_positions"]
        # predicates left unbound, computed whole, see magic_rewrite
        self.unbound_predicates = set()
        # the arguments of the predicates that need bindings, and the
        # variables at them in the rule whose arcs are being created
        self.needed_positions = invariants["needed_positions"]
        self.needed_variables = frozenset()

    def variables_needed(self, rule):
        """The variables of the body of *rule* at the arguments of a
        predicate that need bindings, see :func:`needed_positions`."""
        return frozenset(
            atom.args[i] for atom, _ in _literals(rule.antecedent)
            if isinstance(atom, FunctionApplication)
            for i in self.needed_positions.get(
                unadorned_predicate_symbol(atom.functor), ()
            )
        )

    def _check_for_bound_aggregates(self, rule, adornment):
        """
        Check if there are aggregate functions, or probabilistic queries,
        in the consequent's arguments and raise an exception if these
        arguments are bound in the adornment.
        """
        bound_aggregates = {
            arg
            for arg, ad in zip(rule.consequent.args, adornment)
            if isinstance(arg, AGGREGATES)
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
        self.in_query_rule = (
            unadorned_predicate_symbol(adorned_head.functor)
            == self.query_functor and not self.query_is_recursive
        )
        self.needed_variables = self.variables_needed(rule)

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
        symbol = unadorned_predicate_symbol(predicate.functor)
        free = self.aggregate_positions.get(symbol, ())
        for i, arg in enumerate(predicate.args):
            if symbol in self.unbound_predicates:
                pass
            elif isinstance(arg, Constant) or (
                arg in bound_variables and i not in free
            ):
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
        of a deterministic relation of the extensional database left of P
        in the rule, with a constant argument, as ``d`` in
        ``task(t, "stroop", d) & p(d, x)``, or, in the query's own rule,
        of a small relation (``[MAGIC_SETS] seed_rows`` tuples at most), as
        ``t`` in ``ans(t, x) :- chosen(t) & p(t, x)`` (the terms the query
        keeps). That literal joins the bodies of the magic rules of the
        literals right of it; in the query's rule, it is put first.
        - or it is a variable of a positive literal of a deterministic
        relation of the extensional database sharing a variable with an
        argument a predicate of the rule needs bound, as ``q`` of
        ``sim(q, s) :- study(s), v == f(q, s), v > .5`` (see
        :func:`needed_positions`): ``d`` in ``task(t, d), sim(d, s)``,
        whatever ``seed_from_edb``. That literal is put first, after the
        constants and the rules sharing its variables, which restrict
        it: ``ri_task(t), definition(t, d), sim(d, s)``.

    Probabilistic literals are those of probabilistic facts and choices,
    of within-language probabilistic queries, and of the predicates
    depending on probabilistic facts or choices: magic predicates are thus
    deterministic, and restricting a probabilistic rule with them does
    not change its probabilities.

    .. [1] Isaac Balbin, Graeme S. Port, Kotagiri Ramamohanarao,
       Krishnamurthy Meenakshi. 1991. Efficient Bottom-UP Computation of
       Queries on Stratified Databases. J. Log. Program. 11(3&4). p. 305.
    """

    def __init__(self, datalog, seed_from_edb=True, invariants=None) -> None:
        super().__init__(datalog, seed_from_edb, invariants)

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
            bound_variables.update(
                arg for arg in p.args if isinstance(arg, Symbol)
            )
            tail_predicates.append(p)

        if is_neg:
            p = Negation(p)
        return tail, p

    def _seed(self, predicate, bound_variables, tail_predicates):
        """Bind the variables of *predicate*, a literal of the extensional
        database with a constant or of the query's rule, and add it to the
        tails of the arcs of the literals right of it."""
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
        function, with a constant argument or in the query's rule, or
        with a variable a predicate of the rule needs bound
        (:func:`needed_positions`), whatever ``seed_from_edb``.
        """
        return self.binds(predicate) or (
            self.seed_from_edb and
            self._stored(predicate.functor) and
            (
                any(isinstance(arg, Constant) for arg in predicate.args) or
                (self.in_query_rule and self.seeds(predicate.functor))
            )
        )

    def binds(self, predicate) -> bool:
        """Whether *predicate*, a literal of a stored relation, has a
        variable that a predicate of its rule needs bound."""
        return self._stored(predicate.functor) and not (
            self.needed_variables.isdisjoint(predicate.args)
        )

    def _stored(self, functor) -> bool:
        """Whether *functor* is a deterministic relation of the extensional
        database."""
        return (
            isinstance(functor, Symbol) and
            functor in self.datalog.extensional_database() and
            functor not in self.prob_symbols
        )

    def seeds(self, functor) -> bool:
        """Whether a literal of *functor* in the query's rule binds its
        variables without a constant: a deterministic relation of the
        extensional database of at most ``[MAGIC_SETS] seed_rows``
        tuples."""
        if not self._stored(functor):
            return False
        relation = self.datalog.extensional_database()[functor].value
        return len(relation) <= config.getint(
            "MAGIC_SETS", "seed_rows", fallback=100
        )


def needed_positions(datalog):
    """
    Return the positions of the arguments of each predicate of *datalog*
    bound only through its head, which magic sets bind: ``q`` in
    ``sim(q, s) :- study(s), v == f(q, s), v > .5``, or in a rule
    passing it on, ``p(q, PROB) :- sim(q, s) & selected(s)``. Such a
    predicate cannot be computed whole: its relation is over the values
    the rules using it bind.
    """
    rules = get_list_of_intensional_rules(datalog)
    functions = datalog.builtins()
    positions = {}
    for rule in rules:
        head = rule.consequent.args
        if unbound_variables_error(rule.antecedent, functions) is None:
            continue
        positions.setdefault(rule.consequent.functor, set()).update(
            i for i, arg in enumerate(head)
            if isinstance(arg, Symbol) and unbound_variables_error(
                rule.antecedent, functions,
                exempt=set(head) - {arg},
            ) is not None
        )
    changed = True
    while changed:
        changed = False
        for rule in rules:
            needed, bound = set(), set()
            for atom, negated in _literals(rule.antecedent):
                if negated or not isinstance(atom, FunctionApplication) or (
                    isinstance(atom.functor, Constant) or
                    atom.functor in functions
                ):
                    continue  # a negation, a comparison, a function
                at = positions.get(atom.functor, ())
                for i, arg in enumerate(atom.args):
                    (needed if i in at else bound).add(arg)
            new = {
                i for i, arg in enumerate(rule.consequent.args)
                if arg in needed - bound
            }
            known = positions.setdefault(rule.consequent.functor, set())
            if not new <= known:
                known |= new
                changed = True
    return {
        functor: frozenset(at) for functor, at in positions.items() if at
    }


def aggregate_positions(datalog):
    """
    Return the positions of the arguments of each predicate of *datalog*
    computed by an aggregation or a probabilistic query, in its rules or
    in those of the predicates they use: a variable there is not bound
    by magic sets, which would bind the aggregate.
    """
    rules = get_list_of_intensional_rules(datalog)
    positions = {}
    changed = True
    while changed:
        changed = False
        for rule in rules:
            aggregated = {
                atom.args[i]
                for atom in extract_logic_predicates(rule.antecedent)
                if hasattr(atom, "functor")
                for i in positions.get(atom.functor, ())
            }
            new = {
                i for i, arg in enumerate(rule.consequent.args)
                if arg in aggregated or isinstance(arg, AGGREGATES)
            }
            known = positions.setdefault(rule.consequent.functor, set())
            if not new <= known:
                known |= new
                changed = True
    return positions


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


def _literals(antecedent):
    """The ``(atom, negated)`` of the literals of an antecedent."""
    formulas = (
        antecedent.formulas if isinstance(antecedent, Conjunction)
        else (antecedent,)
    )
    for formula in formulas:
        if isinstance(formula, Negation):
            yield formula.formula, True
        else:
            yield formula, False


def check_stratified(rules, origins=None) -> None:
    """Raise :class:`UnstratifiedMagicSetsRewriteError` if an aggregation or
    a negation of the rules is in a cycle of their dependencies; its
    ``predicate`` is the predicate of *origins* (head symbol -> predicate)
    of the rule with it, or None.

    The dependencies are those of a rule's head on the predicates of its
    body; the *strict* ones are those on a negated literal and, in a rule
    that aggregates, on any literal. A strict dependency between two
    predicates of one strongly connected component is the cycle. Those of
    aggregations are checked first, then those of negations, each in the
    order of the rules: the predicate reported, which the rewrite leaves
    unbound, does not depend on the order of a set (it did on the hashes
    of the symbols: a functional connectivity query was planned with or
    without its aggregation bound, run to run, in 5 or 17 s), and is
    the aggregation's when there is one, computed once whole rather than
    once for each binding.
    """
    edges = {}
    by_aggregation, by_negation = {}, {}
    for rule in rules:
        if not isinstance(rule, Implication):
            continue
        head = rule.consequent.functor
        aggregates = any(
            isinstance(arg, AGGREGATES) for arg in rule.consequent.args
        )
        edges.setdefault(head, set())
        for atom, negated in _literals(rule.antecedent):
            functor = getattr(atom, "functor", None)
            if not isinstance(functor, Symbol):
                continue  # a comparison, a built-in function
            edges[head].add(functor)
            if aggregates:
                by_aggregation[head, functor] = None
            elif negated:
                by_negation[head, functor] = None
    component = strongly_connected_components(edges)
    for head, functor in list(by_aggregation) + list(by_negation):
        if component.get(head) is not None and (
            component.get(head) == component.get(functor)
        ) and (head != functor or functor in edges.get(head, ())):
            error = UnstratifiedMagicSetsRewriteError(
                f"The rewrite would have {head} depend on {functor} through "
                "a cycle, with an aggregation or a negation in it: the "
                "rewritten program would not be stratified"
            )
            error.predicate = (origins or {}).get(head)
            raise error


def _origins(adorned_code):
    """``{symbol: predicate}`` of the adorned and the magic predicate
    symbols of the rewrite of *adorned_code*, once unadorned, and of the
    predicate of the program each is of."""
    origins = {}
    for rule in adorned_code.formulas:
        functor = rule.consequent.functor
        origin = unadorned_predicate_symbol(functor)
        origins[ReplaceAdornedSymbolWalker().walk(functor)] = origin
        origins[magic_predicate(rule.consequent, adorned=False).functor] = (
            origin
        )
    return origins


def check_no_new_recursion(rules, origins, recursive) -> None:
    """
    Raise :class:`NewRecursionInMagicSetsRewriteError` if the rewritten
    *rules* have a cycle of predicates none of which is of a predicate of
    *recursive*, by *origins* (head symbol -> predicate): a recursion of
    the rewrite's own, not the program's.

    Balbin's magic sets give all the call sites of a predicate with one
    adornment one adorned predicate and one magic predicate. In
    ``a(x) :- p(x, y), q(y)`` and ``b(x) :- q(x), p(x, y)``, with ``p``
    computed before ``q`` in the first and after in the second,
    ``magic_q^b`` reads ``p^bf`` and ``magic_p^bf`` reads ``q^b``, which
    reads ``magic_q^b``: a cycle, though neither ``p`` nor ``q`` is
    recursive. The whole program is then evaluated as one recursive
    stratum, rule by rule until nothing changes, which costs more than
    the bindings save. A cycle with a recursive predicate in it is the
    recursion of the program, which magic sets keep.
    """
    for component in recursive_components(rules):
        if recursive.isdisjoint(origins.get(p, p) for p in component):
            names = sorted({origins.get(p, p).name for p in component})
            raise NewRecursionInMagicSetsRewriteError(
                "The rewrite would put in a cycle predicates that are in "
                f"none in the program: {', '.join(names)}"
            )


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
    # A predicate whose binding would put an aggregation or a negation in a
    # cycle of the rewritten program is left unbound, and the rewrite done
    # again: it is then computed whole, the others still bound. A predicate
    # recursive in the program is left unbound whenever it is reported; one
    # that is not, once: a cycle of such predicates is one of the rewrite's
    # own (see check_no_new_recursion), which one more predicate left
    # unbound rarely removes, and each attempt is a whole rewrite.
    # When that does not help, only the predicates that cannot be computed
    # whole are bound, the others left unbound, and the cycles of that
    # rewrite kept: unrewritten, the program has no answer (the relation of
    # similarities to a query text can be the magic predicate of its own
    # percentile). Without such predicates, the program is solved
    # unrewritten.
    rules = datalog.intensional_database().get(query.functor)
    reachable = reachable_code(getattr(rules, "formulas", ()), datalog)
    recursive = set().union(*recursive_components(reachable.formulas))
    unbound, unbinds_left, checked = set(), 1, recursive
    invariants = program_invariants(datalog)
    while True:
        try:
            return _magic_rewrite(
                query, datalog, seed_from_edb, unbound, invariants, checked
            )
        except UnstratifiedMagicSetsRewriteError as error:
            predicate = error.predicate
            if predicate in recursive and predicate not in unbound:
                unbound.add(predicate)
                continue
            if predicate is not None and predicate not in recursive:
                if unbinds_left and predicate not in unbound:
                    unbinds_left -= 1
                    unbound.add(predicate)
                    continue
                failure = error
                if checked is not None:  # not the needed bindings' rewrite
                    failure = NewRecursionInMagicSetsRewriteError(str(error))
            else:
                failure = error
        except NewRecursionInMagicSetsRewriteError as error:
            failure = error
        needed = needed_positions(datalog)
        others = set(datalog.intensional_database()) - set(needed)
        if others <= unbound or not any(
            rule.consequent.functor in needed for rule in reachable.formulas
        ):
            raise failure
        unbound |= others
        checked = None


def _magic_rewrite(
    query,
    datalog,
    seed_from_edb,
    unbound_predicates,
    invariants=None,
    recursive=None,
):
    """:func:`magic_rewrite`, the *unbound_predicates* left unbound; the
    rewrite must not add a cycle without a predicate of *recursive*, the
    recursive predicates of the program, when it is given."""
    # 1. Create a SIPS and use it to adorn the code
    sips = LeftToRightSIPS(datalog, seed_from_edb, invariants)
    sips.unbound_predicates = unbound_predicates
    adorned_code, constant_predicates = reachable_adorned_code(
        query, datalog, sips
    )
    # 2. Check that some predicate is bound: by a constant, or by the
    # literals left of it in a rule, as ``d`` in ``q(t, d), p(d, s)``
    if not any(
        "b" in rule.consequent.functor.adornment
        for rule in adorned_code.formulas
    ):
        # Nothing is bound, magic sets is not useful
        raise NoConstantPredicateFoundError(
            "No predicate with a bound argument found."
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
    # 5. The rewrite must leave the program stratified
    check_stratified(unadorned_code, {
        ReplaceAdornedSymbolWalker().walk(rule.consequent.functor):
        unadorned_predicate_symbol(rule.consequent.functor)
        for rule in magic_rules
    })
    if recursive is not None:
        check_no_new_recursion(
            unadorned_code, _origins(adorned_code), recursive
        )
    # 6. Restrictions of stored relations are left to the rules using them
    return goal, Union(unfold_restrictions(unadorned_code, datalog, goal))


def _body_functors(rule):
    """The functors of the atoms of the body of *rule*, negated or not."""
    return {
        atom.functor for atom, _ in _literals(rule.antecedent)
        if isinstance(atom, FunctionApplication)
    }


class _Restrictions:
    """The rules of the restrictions to unfold, as a program's."""

    def __init__(self, rules):
        self._idb = {
            rule.consequent.functor: Union((rule,)) for rule in rules
        }

    def intensional_database(self):
        return self._idb


def _restriction(rule, rules_of, edb, probabilistic, goal):
    """Whether *rule* is a restriction of a stored relation: the only rule
    of its predicate, not the goal, of a head of distinct variables, its
    body the magic predicate of its head and one positive atom of a
    deterministic relation of the extensional database."""
    head = rule.consequent
    body = rule.antecedent
    atoms = body.formulas if isinstance(body, Conjunction) else ()
    return (
        head.functor != goal
        and rules_of[head.functor] == 1
        and all(isinstance(arg, Symbol) for arg in head.args)
        and len(set(head.args)) == len(head.args)
        and len(atoms) == 2
        and all(
            isinstance(atom, FunctionApplication)
            and isinstance(atom.functor, Symbol)
            for atom in atoms
        )
        and atoms[0].functor.name == f"magic_{head.functor.name}"
        and atoms[1].functor in edb
        and atoms[1].functor not in probabilistic
    )


def unfold_restrictions(rules, datalog, goal):
    """
    Return *rules*, a program rewritten by magic sets of goal *goal*, with
    the restrictions of stored relations unfolded into the rules using
    them.

    Magic sets restrict a predicate to the bindings of its magic predicate
    with a rule, ``p^bf(x, y) :- magic_p^bf(x), r(x, y)`` of a stored
    relation ``r``, which a bottom-up evaluation computes as a new
    relation, a copy of the rows of ``r`` it keeps. Substituting the
    rule's body for each use of ``p^bf`` (an equivalence: the rule is the
    only one of its predicate, and not recursive) leaves the join to the
    plan of each rule using it, where a backend applies it as it applies
    any join: as a semijoin, or, of a relation held as a sparse matrix, by
    a product selecting its rows, without the copy. A restriction used
    under negation, or by a rule the substitution does not reach, is kept.
    """
    rules_of = {}
    for rule in rules:
        functor = rule.consequent.functor
        rules_of[functor] = rules_of.get(functor, 0) + 1
    edb = set(datalog.extensional_database())
    probabilistic = probabilistic_symbols(datalog)
    restrictions = [
        rule for rule in rules
        if _restriction(rule, rules_of, edb, probabilistic, goal)
    ]
    negated = {
        atom.functor
        for rule in rules for atom, is_negated in _literals(rule.antecedent)
        if is_negated and isinstance(atom, FunctionApplication)
    }
    restrictions = [
        rule for rule in restrictions
        if rule.consequent.functor not in negated
    ]
    if not restrictions:
        return tuple(rules)
    flatten = FlattenQueryInNonRecursiveUCQ(_Restrictions(restrictions))
    unfolded = set(rule.consequent.functor for rule in restrictions)
    result = []
    for rule in rules:
        if rule not in restrictions and _body_functors(rule) & unfolded:
            try:
                rule = Implication(
                    rule.consequent, flatten.walk(rule.antecedent)
                )
            except NeuroLangPatternMatchingNoMatch:  # a body it does not
                pass  # reach: kept, with the restriction
        result.append(rule)
    used = {
        functor for rule in result if rule not in restrictions
        for functor in _body_functors(rule)
    }
    return tuple(
        rule for rule in result
        if rule not in restrictions or rule.consequent.functor in used
    )


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
    # Passes: each takes, in order, the rules whose body has no predicate
    # defined by a rule still to take but their own, or the first rule
    # left when none has. A rule waits on a count of such predicates,
    # updated when the last rule of one is taken: each body is read once,
    # not once a pass (a pass compared thousands of symbols, 16 s of a
    # query of tracts). Rules go by position, as comparing rules is
    # comparing their expressions.
    rules = list(rules)
    heads = [rule.consequent.functor for rule in rules]
    rules_of = Counter(heads)
    waiting_on = [_body_predicates(rule) for rule in rules]
    users = defaultdict(list)
    for i, body in enumerate(waiting_on):
        waiting_on[i] = {p for p in body if p in rules_of and p != heads[i]}
        for predicate in waiting_on[i]:
            users[predicate].append(i)
    pending = dict.fromkeys(range(len(rules)))
    ordered = []
    while pending:
        ready = [i for i in pending if not waiting_on[i]] or [
            next(iter(pending))
        ]
        for i in ready:
            del pending[i]
            ordered.append(rules[i])
            rules_of[heads[i]] -= 1
        for predicate in {heads[i] for i in ready}:
            if rules_of[predicate] == 0:
                for j in users[predicate]:
                    waiting_on[j].discard(predicate)
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

    The body keeps only the predicates connected to the bound arguments
    of p, see :func:`connected_literals`: the others would only multiply
    the magic set by their rows, then be projected out.
    """
    magic_rules = []
    for head, tail in arcs.items():
        if isinstance(head, Negation):
            head = head.formula
        if "b" in head.functor.adornment:
            rule = _magic_arc_rule(head, tail, consequent, magic_head)
            if rule is not None:
                magic_rules.append(rule)
    return magic_rules


def _magic_arc_rule(head, tail, consequent, magic_head):
    """The magic rule of the arc of *tail* to *head*, see
    :func:`create_magic_arc_rules`, or None if it is not needed."""
    new_predicate = magic_predicate(head, adorned=False)
    body_lits = [p for p in tail if p != consequent]
    if magic_head is not None:
        body_lits = [magic_head] + body_lits
    body_lits = connected_literals(new_predicate, body_lits)
    if magic_head is None and len(body_lits) == 0:
        return None
    if len(body_lits) == 1 and new_predicate == body_lits[0]:
        # avoid adding rules of the form magic_p(x) :- magic_p(x)
        return None
    return Implication(new_predicate, Conjunction(body_lits))


def connected_literals(predicate, literals):
    """
    The *literals* sharing a variable with *predicate*, directly or
    through other literals, in their order.

    Leaving the others out of the body of a magic rule of head
    *predicate* is sound: a magic predicate restricts a rule whose body
    has these literals already, and a larger magic set restricts less.
    """
    variables = _symbols_of(predicate)
    connected = set()
    added = True
    while added:
        added = False
        for i, literal in enumerate(literals):
            if i not in connected and _symbols_of(literal) & variables:
                variables |= _symbols_of(literal)
                connected.add(i)
                added = True
    return [literal for i, literal in enumerate(literals) if i in connected]


def _symbols_of(literal):
    return {arg for arg in literal.args if isinstance(arg, Symbol)}


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
    """
    The rules of the adorned code reachable from the adorned ``query``,
    the query rule last, and the adorned predicates with a constant.

    The rules are those of :func:`adorn_code`, with their bodies in the
    order they were adorned in: walking them in ``datalog``'s class may
    reorder the bodies (as the probabilistic frontend does to avoid cross
    products), and the arcs of the SIPS, created from the bodies, would
    not follow the adornments.
    """
    adorned_code, constant_predicates = adorn_code(query, datalog, sips)
    # assume that the query rule is the first
    adorned_query = adorned_code.formulas[0]
    reachable = dict.fromkeys(
        rule.consequent.functor for rule in reachable_code(
            adorned_query, _RulesOf(adorned_code.formulas)
        ).formulas
    )
    # a predicate adorned twice before its rules were has them twice
    rules = dict.fromkeys(adorned_code.formulas)
    return (
        Union(tuple(
            rule for functor in reachable for rule in rules
            if rule.consequent.functor == functor
        )),
        constant_predicates,
    )


class _RulesOf:
    """The rules of a rewrite as a program for
    :func:`.constraints_representation.reachable_code`, which reads its
    intensional database alone: walking them into a program of their own,
    to that end, ran its type resolution and checks on each rule of each
    attempt of a rewrite (1 s each)."""

    def __init__(self, rules):
        rules_of = {}
        for rule in rules:
            rules_of.setdefault(rule.consequent.functor, []).append(rule)
        self.rules_of = {
            functor: Union(tuple(rules)) for functor, rules in rules_of.items()
        }

    def intensional_database(self):
        return self.rules_of


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
    adornment = ''.join(
        'b' if isinstance(a, Constant) else 'f' for a in query.args
    )
    sips.query_functor = query.functor
    # a query defined recursively (a rule of a predicate unfolded into the
    # query's) has no relation it keeps: its stored relations are steps
    sips.query_is_recursive = any(
        query.functor in _body_functors(rule)
        for rule in getattr(
            datalog.intensional_database().get(query.functor), "formulas", ()
        )
    )

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

        in_query_rule = (
            predicate_symbol == sips.query_functor
            and not sips.query_is_recursive
        )
        for rule in rules.formulas:
            if isinstance(rule, RightImplication):
                continue
            if isinstance(rule.antecedent, Conjunction):
                needed = sips.variables_needed(rule)
                seeds = in_query_rule and sips.seed_from_edb
                head_bound = {
                    arg for arg, a in zip(
                        rule.consequent.args,
                        getattr(consequent.functor, "adornment", ""),
                    ) if a == "b"
                }

                def binds(atom, needed=needed):
                    return (
                        sips._stored(atom.functor) and
                        not needed.isdisjoint(atom.args)
                    )
                # The constants bind first, then the rules restricting
                # the variables of those stored relations, the stored
                # relations, and the literals needing their variables:
                # task(t), definition(t, d), sim(d, s), whatever the
                # order of the body.
                stored_vars = {
                    arg for atom in rule.antecedent.formulas
                    if isinstance(atom, FunctionApplication) and binds(atom)
                    for arg in atom.args
                }

                def binding(atom, seeds=seeds, binds=binds,
                            stored_vars=stored_vars):
                    return binds(atom) or (
                        seeds and sips.seeds(atom.functor)
                    ) or (
                        atom.functor in idb and
                        atom.functor not in sips.needed_positions and
                        atom.functor not in sips.prob_symbols and
                        not stored_vars.isdisjoint(atom.args)
                    )

                def last(atom, binds=binds, head_bound=head_bound):
                    if any(
                        isinstance(atom.args[i], Symbol) and
                        atom.args[i] not in head_bound
                        for i in sips.needed_positions.get(atom.functor, ())
                    ):
                        return 3
                    if any(
                        isinstance(arg, Constant) or arg in head_bound
                        for arg in atom.args
                    ):
                        return 0
                    return 2 if binds(atom) else int(atom.functor in idb)
                formulas = order_conjunction_constant_atoms_first(
                    rule.antecedent.formulas, head_bound, binding, last,
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
