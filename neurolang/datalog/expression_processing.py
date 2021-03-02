"""
Utilities to process intermediate representations of
Datalog programs.
"""


import collections
import operator
import typing
from typing import Iterable

import numpy as np

from ..exceptions import (
    ForbiddenExpressionError,
    RuleNotFoundError,
    SymbolNotFoundError,
    UnsupportedProgramError,
)
from ..expression_pattern_matching import (
    NeuroLangPatternMatchingNoMatch,
    add_match,
)
from ..expression_walker import (
    ExpressionWalker,
    PatternWalker,
    ReplaceExpressionWalker,
    ReplaceSymbolWalker,
)
from ..expressions import Constant, Expression, FunctionApplication, Symbol
from ..logic import (
    FALSE,
    TRUE,
    Conjunction,
    Disjunction,
    Implication,
    Negation,
    Quantifier,
    Union,
)
from ..logic import expression_processing as elp
from ..logic.transformations import CollapseConjunctions
from ..logic.unification import most_general_unifier
from .expressions import AggregationApplication, TranslateToLogic

EQ = Constant(operator.eq)


class TranslateToDatalogSemantics(TranslateToLogic, ExpressionWalker):
    pass


def implication_has_existential_variable_in_antecedent(implication):
    """
    Whether an implication has at least one existential variable in its
    antecedent.
    """
    c_free_vars = set(extract_logic_free_variables(implication.consequent))
    a_free_vars = set(extract_logic_free_variables(implication.antecedent))
    return a_free_vars > c_free_vars


def is_conjunctive_expression(expression):
    if isinstance(expression, Conjunction):
        formulas = expression.formulas
    else:
        formulas = [expression]

    return all(
        expression == Constant(True)
        or expression == Constant(False)
        or (
            isinstance(expression, FunctionApplication)
            and not any(
                isinstance(arg, FunctionApplication) for arg in expression.args
            )
        )
        for expression in formulas
    )


def is_conjunctive_expression_with_nested_predicates(expression):
    tr = TranslateToDatalogSemantics()
    expression = tr.walk(expression)
    stack = [expression]
    while stack:
        exp = stack.pop()
        if exp == Constant(True) or exp == Constant(False):
            pass
        elif isinstance(exp, FunctionApplication):
            stack += [
                arg for arg in exp.args if isinstance(arg, FunctionApplication)
            ]
        elif isinstance(exp, Conjunction):
            stack += exp.formulas
        elif isinstance(exp, Quantifier):
            stack.append(exp.body)
        else:
            return False

    return True


def is_linear_rule(rule):
    """Check if a rule is linear

    Parameters
    ----------
    rule : Implication
        rule to analyse

    Returns
    -------
    bool
        True if the rule is linear

    """
    predicates = extract_logic_predicates(rule.antecedent)
    return (
        sum(
            int(
                (predicate.formula.functor == rule.consequent.functor)
                if isinstance(predicate, Negation)
                else predicate.functor == rule.consequent.functor
            )
            for predicate in predicates
        )
        < 2
    )


def all_body_preds_in_set(implication, predicate_set):
    """Checks wether all predicates in the antecedent
    are in the functor_set or are the consequent functor.

    Parameters
    ----------
    implication :
        Implication
    predicate_set :
        set or functors of the consequent

    Returns
    -------
    bool
        True is all predicates in the antecedent are
        in the prediacte_set

    """
    preds = (e.functor for e in extract_logic_atoms(implication.antecedent))
    predicate_set = predicate_set | {implication.consequent.functor}
    return all(not isinstance(e, Symbol) or e in predicate_set for e in preds)


def any_body_preds_in_set(implication, predicate_set):
    """Checks wether any predicates in the antecedent
    are in the predicate_set.

    Parameters
    ----------
    implication :
        Implication
    predicate_set :
        set of functors

    Returns
    -------
    bool
        True is any predicates in the antecedent are
        in the predicate_set

    """
    preds = (e.functor for e in extract_logic_atoms(implication.antecedent))
    return any(e in predicate_set for e in preds)


def extract_logic_free_variables(expression):
    """Extract variables from expression assuming it's in datalog format.

    Parameters
    ----------
    expression : Expression


    Returns
    -------
        OrderedSet
            set of all free variables in the expression.
    """
    translator = TranslateToDatalogSemantics()
    datalog_code = translator.walk(expression)
    return elp.extract_logic_free_variables(datalog_code)


def extract_logic_predicates(expression):
    """Extract predicates from expression
    knowing that it's in Datalog format

    Parameters
    ----------
    expression : Expression
        expression to extract predicates from


    Returns
    -------
    OrderedSet
        set of all predicates in the expression in lexicographical
        order.

    """
    return elp.extract_logic_predicates(expression)


def extract_logic_atoms(expression):
    """Extract atoms from expression
    knowing that it's in Datalog format

    Parameters
    ----------
    expression : Expression
        expression to extract atoms from


    Returns
    -------
    OrderedSet
        set of all predicates in the atoms in lexicographical
        order.

    """
    return elp.extract_logic_atoms(expression)


def stratify(union, datalog_instance):
    """Given an expression block containing `Implication` instances
     and a datalog instance, return the stratification of the formulas
     in the block as a list of lists..

    Parameters
    ----------
    union : Union
        union of implications to be stratified.

    datalog_instance : DatalogProgram
        Datalog instance containing the EDB and IDB databases


    Returns
    -------
        list of lists of `Implications`, boolean
            Strata and wether it was stratisfiable.
            If it was not, all non-stratified predicates
            will be in the last strata.

    """
    strata = []
    seen = set(k for k in datalog_instance.extensional_database().keys())
    seen |= set(k for k in datalog_instance.builtins())
    to_process = union.formulas
    stratifiable = True

    stratum, new_to_process = stratify_obtain_facts_stratum(to_process, seen)

    if len(stratum) > 0:
        strata.append(stratum)
    to_process = new_to_process

    while len(to_process) > 0:
        new_seen, new_to_process, stratum = stratify_obtain_new_stratum(
            to_process, seen
        )
        to_process = new_to_process
        if len(new_seen) > 0:
            strata.append(stratum)
            seen |= new_seen
        else:
            strata.append(to_process)
            stratifiable = False
            break

    return strata, stratifiable


def stratify_obtain_facts_stratum(to_process, seen):
    new_to_process = []
    stratum = []
    true_ = Constant(True)
    for r in to_process:
        if r.antecedent == true_:
            stratum.append(r)
            seen.add(r.consequent.functor)
        else:
            new_to_process.append(r)
    return stratum, new_to_process


def stratify_obtain_new_stratum(to_process, seen):
    # 1. add all the rule for which we've seen the body predicates
    temp_stratum = []
    new_to_process = []
    for r in to_process:
        if all_body_preds_in_set(r, seen):
            temp_stratum.append(r)

        else:
            new_to_process.append(r)

    # 2. remove all the rules which have body predicates that will be updated
    # by rules in later strata.
    will_see_later = {r.consequent.functor for r in new_to_process}
    stratum = []
    new_seen = set()
    for r in temp_stratum:
        if any_body_preds_in_set(r, will_see_later):
            new_to_process.insert(0, r)
        else:
            stratum.append(r)
            new_seen.add(r.consequent.functor)

    return new_seen, new_to_process, stratum


def reachable_code(query, datalog):
    """Produces the code reachable by a query

    Parameters
    ----------
    query : Implication
        Rule to figure out the reachable program from
    datalog : DatalogProgram
        datalog instance containing the EDB and IDB.

    Returns
    -------
    ExpressionBlock
        Code needed to solve the query.
    """
    if not isinstance(query, Iterable):
        query = [query]

    reachable_code = []
    idb = datalog.intensional_database()
    to_reach = [q.consequent.functor for q in query]
    reached = set()
    seen_rules = set()
    while to_reach:
        p = to_reach.pop()
        reached.add(p)
        rules = idb[p]
        for rule in rules.formulas:
            if rule in seen_rules:
                continue
            seen_rules.add(rule)
            reachable_code.append(rule)
            for predicate in extract_logic_atoms(rule.antecedent):
                functor = predicate.functor
                if functor not in reached and functor in idb:
                    to_reach.append(functor)

    return Union(reachable_code[::-1])


def dependency_matrix(datalog, rules=None, instance=None):
    """Produces the dependecy matrix for a datalog's
    instance intensional database (IDB).

    Parameters
    ----------
    datalog : DatalogProgram
        datalog instance containing the EDB and IDB.
    rules : None or Union of rules
        an optional subset of rules from the datalog
        program's IDB.
    instance: Instance
        an optional instance, meant to represent the current
        DatalogProgram resolution, and whose's symbols won't raise
        a SymbolNotFoundError if found as a dependency in the given
        rules.

    Returns
    -------
    idb_symbols: list
        A list of IDB symbols
        in the dependency matrix.
    dependency_matrix: ndarray
        The dependency matrix
        where row is the origin symbol and column is the
        dependency. It is the adjacency matrix of the
        graph where each node is a predicate of the IDB.

    Raises
    ------
    SymbolNotFoundError
        If there is a predicate in the antecedent of a rule which
        is not a constant or an extensiona/intensional predicate.
    """

    if rules is None:
        idb = datalog.intensional_database()
        to_reach = []
        for rule_union in idb.values():
            to_reach += rule_union.formulas
        idb_symbols = idb.keys()
    else:
        if isinstance(rules, Union):
            to_reach = list(rules.formulas)
        else:
            to_reach = list(rules)
        idb_symbols = set()
        for rule in to_reach:
            functor = rule.consequent.functor
            if rule not in datalog.intensional_database()[functor].formulas:
                raise RuleNotFoundError(
                    f"Rule {rule} not contained in the datalog " "instance."
                )
            idb_symbols.add(functor)

    idb_symbols = tuple(sorted(idb_symbols, key=lambda s: s.name))
    edb = datalog.extensional_database()
    previous_symbols = set()
    if instance is not None:
        previous_symbols = instance.as_map().keys()
    if hasattr(datalog, "constraints"):
        constraint_symbols = set(
            formula.consequent.functor
            for formula in datalog.constraints().formulas
        )
    else:
        constraint_symbols = set()

    dependency_matrix = np.zeros(
        (len(idb_symbols), len(idb_symbols)), dtype=int
    )

    while to_reach:
        rule = to_reach.pop()
        head_functor = rule.consequent.functor
        ix_head = idb_symbols.index(head_functor)
        for predicate in extract_logic_atoms(rule.antecedent):
            functor = predicate.functor
            if functor in edb or functor in constraint_symbols:
                continue
            elif functor in idb_symbols:
                ix_functor = idb_symbols.index(functor)
                dependency_matrix[ix_head, ix_functor] += 1
            elif (
                isinstance(functor, Symbol)
                and functor not in previous_symbols
                and (
                    functor not in datalog.symbol_table
                    or functor in datalog.intensional_database()
                )
            ):
                raise SymbolNotFoundError(f"Symbol not found {functor.name}")

    return idb_symbols, dependency_matrix


def program_has_loops(program_representation):
    if not isinstance(program_representation, np.ndarray):
        _, program_representation = dependency_matrix(program_representation)
    reachable = program_representation
    for _ in range(len(program_representation)):
        if any(np.diag(reachable)):
            return True
        else:
            reachable = np.dot(reachable, program_representation)

    return False


def conjunct_if_needed(formulas):
    """
    Only conjunct the given list of formulas if there is more than one. If
    an empty list of formulas is passed, `Constant[bool](True)` is returned
    instead.

    """
    if len(formulas) == 0:
        return TRUE
    if len(formulas) == 1:
        return formulas[0]
    return Conjunction(formulas)


def conjunct_formulas(f1, f2):
    """Conjunct two logical formulas."""
    if isinstance(f1, Conjunction) and isinstance(f2, Conjunction):
        return Conjunction(tuple(f1.formulas) + tuple(f2.formulas))
    elif isinstance(f1, Conjunction):
        return Conjunction(tuple(f1.formulas) + (f2,))
    elif isinstance(f2, Conjunction):
        return Conjunction((f1,) + tuple(f2.formulas))
    else:
        return Conjunction((f1, f2))


def is_ground_predicate(predicate):
    """Whether all the predicate's terms are all constant."""
    return all(isinstance(arg, Constant) for arg in predicate.args)


def enforce_conjunction(expression):
    if isinstance(expression, Conjunction):
        return expression
    elif isinstance(expression, (FunctionApplication, Negation)):
        return Conjunction((expression,))
    raise ForbiddenExpressionError(
        "Cannot conjunct expression of type {}".format(type(expression))
    )


def enforce_conjunctive_antecedent(implication):
    return implication.apply(
        implication.consequent,
        remove_conjunction_duplicates(
            enforce_conjunction(implication.antecedent)
        ),
    )


def maybe_deconjunct_single_pred(expression):
    """
    Remove the conjunction from single-conjunct conjunctions.

    The conjunction remains unchanged if it has multiple conjuncts or if it
    already is a single predicate.

    """
    if isinstance(expression, FunctionApplication):
        return expression
    if len(expression.formulas) == 1:
        return expression.formulas[0]
    return expression


def maybe_disjunct(
    formulas: typing.Iterable[Expression],
) -> typing.Union[Expression, Disjunction]:
    formulas = tuple(formulas)
    if len(formulas) > 1:
        return Disjunction(formulas)
    return formulas[0]


class HeadConstantToBodyEquality(PatternWalker):
    """
    Transform rules whose head (consequent) predicate contains constant terms
    into an equivalent rule where the head predicate only contains variable
    terms and the body (antecedent) of the rule contains variable-to-constant
    equalities. A fresh variable is generated for each constant term in the
    head predicate.

    Examples
    --------
    The rule `Q(x, 2) :- P(x, y)` is transformed into the rule `Q(x, _f_) :-
    P(x, y), _f_ = 2`, where `_f_` is a fresh variable.

    """

    @add_match(
        Implication(FunctionApplication, ...),
        lambda implication: any(
            isinstance(term, Constant) for term in implication.consequent.args
        ),
    )
    def implication_with_constant_term_in_head(self, implication):
        body_formulas = list(
            enforce_conjunction(implication.antecedent).formulas
        )
        new_consequent_vars = list()
        for term in implication.consequent.args:
            if isinstance(term, Constant):
                var = Symbol.fresh()
                body_formulas.append(EQ(var, term))
            else:
                var = term
            new_consequent_vars.append(var)
        new_consequent = implication.consequent.functor(*new_consequent_vars)
        new_antecedent = Conjunction(tuple(body_formulas))
        new_antecedent = maybe_deconjunct_single_pred(new_antecedent)
        return self.walk(Implication(new_consequent, new_antecedent))


class HeadRepeatedVariableToBodyEquality(PatternWalker):
    """
    Transform rules whose head (consequent) predicate contains repeated
    variables (symbols) to an equivalent rule without repeated variables and
    corresponding variable equalities. New fresh variables are generated to
    replace variable repetitions.

    Examples
    --------
    The rule `Q(x, x) :- P(x, y)` is transformed into the rule `Q(x, _f_) :-
    P(x, y), _f_ = x`, where `_f_` is a fresh variable.

    """

    @add_match(
        Implication(FunctionApplication, ...),
        lambda implication: max(
            collections.Counter(
                (
                    arg
                    for arg in implication.consequent.args
                    if isinstance(arg, Symbol)
                )
            ).values()
        )
        > 1,
    )
    def implication_with_repeated_variable_in_head(self, implication):
        seen_args = set()
        new_csqt_args = list()
        vareq_formulas = list()
        for csqt_arg in implication.consequent.args:
            if isinstance(csqt_arg, Symbol) and csqt_arg in seen_args:
                new_csqt_arg = Symbol.fresh()
                vareq_formulas.append(EQ(new_csqt_arg, csqt_arg))
            else:
                new_csqt_arg = csqt_arg
                seen_args.add(csqt_arg)
            new_csqt_args.append(new_csqt_arg)
        new_consequent = implication.consequent.functor(*new_csqt_args)
        new_antecedent = conjunct_formulas(
            implication.antecedent, Conjunction(tuple(vareq_formulas))
        )
        new_antecedent = maybe_deconjunct_single_pred(new_antecedent)
        return self.walk(Implication(new_consequent, new_antecedent))


class FreshenFreeVariables(PatternWalker):
    @add_match(
        Implication(FunctionApplication, ...),
        lambda implication: any(
            not var.is_fresh
            for var in extract_logic_free_variables(implication)
        ),
    )
    def implication_with_free_variables(self, implication):
        replacements = {
            var: Symbol.fresh()
            for var in extract_logic_free_variables(implication)
        }
        implication = ReplaceExpressionWalker(replacements).walk(implication)
        return self.walk(implication)


def flatten_query(query, program):
    """
    Construct the conjunction corresponding to a query on a program.

    TODO: currently this only handles programs without conjunctions.

    Parameters
    ----------
    query : predicate or conjunction of predicates
        The query for which the conjunction is constructed.
    program : a program with an intensional database
        Program with logical rules that will be used to construct the
        conjunction corresponding to the given query.

    Returns
    -------
    disjunction or conjunction of predicates

    """
    if not hasattr(program, "intensional_database"):
        raise UnsupportedProgramError(
            "Only program with an intensional database are supported"
        )
    try:
        res = FlattenQueryInNonRecursiveUCQ(program).walk(query)
        if isinstance(res, FunctionApplication):
            res = Conjunction((res,))
    except RecursionError:
        raise UnsupportedProgramError(
            "Flattening of recursive programs is not supported."
        )
    except NeuroLangPatternMatchingNoMatch:
        raise UnsupportedProgramError("Expression not supported.")
    return res


class FlattenQueryInNonRecursiveUCQ(PatternWalker):
    """Flatten a query defined by a non-recursive
    union of conjunctive queries (UCQ)
    """

    def __init__(self, program):
        self.program = program
        self.idb = self.program.intensional_database()
        self._rule_normaliser = FlattenQueryInNonRecursiveUCQ._RuleNormaliser()

    class _RuleNormaliser(
        HeadConstantToBodyEquality,
        HeadRepeatedVariableToBodyEquality,
        FreshenFreeVariables,
        ExpressionWalker,
    ):
        pass

    @add_match(
        FunctionApplication(Symbol, ...),
        lambda fa: all(isinstance(arg, (Constant, Symbol)) for arg in fa.args),
    )
    def query_predicate(self, qpred):
        pred_symb = qpred.functor
        if pred_symb not in self.idb:
            return qpred
        ucq = self.idb[pred_symb]
        cqs = []
        for cq in ucq.formulas:
            cq = self._rule_normaliser.walk(cq)
            exp = self._unify_cq_antecedent(cq, qpred)
            if exp != FALSE:
                exp = self.walk(exp)
            cqs.append(exp)
        return maybe_disjunct(cqs)

    @add_match(FunctionApplication(Constant, ...))
    def builtin_application(self, fa):
        return fa

    def _unify_cq_antecedent(self, cq, qpred):
        mgu = most_general_unifier(cq.consequent, qpred)
        # if we cannot unify, this is always a false statement
        if mgu is None:
            return FALSE
        antecedent = ReplaceExpressionWalker(mgu[0]).walk(cq.antecedent)
        if isinstance(
            antecedent, Conjunction
        ) and not self._check_compatibility_symb_to_const_eqs(antecedent):
            return FALSE
        return maybe_deconjunct_single_pred(antecedent)

    @staticmethod
    def _check_compatibility_symb_to_const_eqs(conjunction):
        """
        Statically analyse variable-to-constant equalities within the
        conjunction to check for compatibility. For example, if the two
        conjuncts `x = 2` and `x = 3` are present, this function will return
        `False`. It also handles equalities where the order is `const =
        symbol`, such as `2 = 3, x = 3`.

        """
        symb_to_const = dict()
        symb_to_const_eq_formulas = (
            conjunct
            for conjunct in conjunction.formulas
            if is_symb_to_const_equality(conjunct)
        )
        for equality in symb_to_const_eq_formulas:
            symbol = next(
                arg for arg in equality.args if isinstance(arg, Symbol)
            )
            constant = next(
                arg for arg in equality.args if isinstance(arg, Constant)
            )
            if symbol in symb_to_const:
                return False
            symb_to_const[symbol] = constant
        return True

    @add_match(Conjunction)
    def conjunction(self, expression):
        formulas = list(expression.formulas)
        new_formulas = tuple()
        while len(formulas) > 0:
            formula = formulas.pop()
            new_formula = self.walk(formula)
            if isinstance(new_formula, Conjunction):
                new_formulas += new_formula.formulas
            else:
                new_formulas += (new_formula,)
        if len(new_formulas) > 0:
            res = Conjunction(tuple(new_formulas))
        else:
            res = new_formulas[0]
        return res

    @add_match(Negation)
    def negation(self, expression):
        return Negation(self.walk(expression.formula))


def is_rule_with_builtin(rule, known_builtins=None):
    if known_builtins is None:
        known_builtins = set()
    return any(
        isinstance(pred.functor, Constant) or pred.functor in known_builtins
        for pred in extract_logic_predicates(rule.antecedent)
    )


def remove_conjunction_duplicates(conjunction):
    return maybe_deconjunct_single_pred(
        Conjunction(tuple(set(conjunction.formulas)))
    )


def is_aggregation_rule(rule):
    return is_aggregation_predicate(rule.consequent)


def is_aggregation_predicate(predicate):
    return any(
        isinstance(arg, AggregationApplication) for arg in predicate.args
    )


def is_var_equality_to_var_or_const(formula):
    return (
        isinstance(formula, FunctionApplication)
        and formula.functor == EQ
        and len(formula.args) == 2
        and all(isinstance(arg, (Symbol, Constant)) for arg in formula.args)
        and any(isinstance(arg, Symbol) for arg in formula.args)
    )


def is_symb_to_const_equality(formula):
    return (
        isinstance(formula, FunctionApplication)
        and formula.functor == EQ
        and len(formula.args) == 2
        and any(isinstance(arg, Symbol) for arg in formula.args)
        and any(isinstance(arg, Constant) for arg in formula.args)
    )


class RemoveDuplicatedAntecedentPredicates(PatternWalker):
    @add_match(
        Implication(FunctionApplication, Conjunction),
        lambda implication: any(
            count > 1
            for count in collections.Counter(
                implication.antecedent.formulas
            ).values()
        ),
    )
    def implication_with_duplicated_antecedent_predicate(self, implication):
        return self.walk(
            implication.apply(
                implication.consequent,
                remove_conjunction_duplicates(implication.antecedent),
            )
        )


class UnifyVariableEqualitiesMixin(PatternWalker):
    @add_match(
        Implication(FunctionApplication(Symbol, ...), ...),
        lambda implication: all(
            isinstance(arg, (Symbol, Constant))
            for arg in implication.consequent.args
        )
        and any(
            is_var_equality_to_var_or_const(formula)
            for formula in extract_logic_predicates(implication.antecedent)
        ),
    )
    def extract_and_unify_var_eqs_in_implication(self, implication):
        eq_sets = self.extract_variable_equalities(implication.antecedent)
        substitutions = self.build_substitutions_from_equalities(eq_sets)
        replacer = ReplaceSymbolWalker(substitutions)
        antecedent = Conjunction(
            tuple(
                formula
                for formula in extract_logic_predicates(implication.antecedent)
                if not is_var_equality_to_var_or_const(formula)
            )
        )
        antecedent = replacer.walk(antecedent)
        if len(antecedent.formulas) == 0:
            antecedent = TRUE
        else:
            antecedent = remove_conjunction_duplicates(antecedent)
        consequent = replacer.walk(implication.consequent)
        new_implication = Implication(consequent, antecedent)
        return self.walk(new_implication)

    @add_match(
        Implication(FunctionApplication(Symbol, ...), ...),
        lambda implication: is_aggregation_rule(implication)
        and any(
            is_var_equality_to_var_or_const(formula)
            for formula in extract_logic_predicates(implication.antecedent)
        ),
    )
    def unsupported_unification_for_aggregation(self, rule_with_aggregation):
        raise ForbiddenExpressionError(
            "Unification of rules with aggregation not supported"
        )

    @staticmethod
    def build_substitutions_from_equalities(eq_sets):
        substitutions = dict()
        for eq_set in eq_sets:
            if any(isinstance(term, Constant) for term in eq_set):
                const = next(
                    term for term in eq_set if isinstance(term, Constant)
                )
                update = {
                    term: const for term in eq_set if isinstance(term, Symbol)
                }
            else:
                chosen_symb = max(eq_set, key=lambda symb: symb.name)
                update = {
                    symb: chosen_symb for symb in eq_set if symb != chosen_symb
                }
            substitutions.update(update)
        return substitutions

    @staticmethod
    def extract_variable_equalities(expression):
        equality_predicates = set(
            predicate
            for predicate in extract_logic_predicates(expression)
            if is_var_equality_to_var_or_const(predicate)
        )
        eq_sets = list()
        for equality_predicate in equality_predicates:
            if all(isinstance(arg, Symbol) for arg in equality_predicate.args):
                UnifyVariableEqualities.add_equality_with_symbol(
                    eq_sets, *equality_predicate.args
                )
            elif isinstance(equality_predicate.args[0], Symbol):
                UnifyVariableEqualities.add_equality_with_constant(
                    eq_sets, *equality_predicate.args
                )
            else:
                UnifyVariableEqualities.add_equality_with_constant(
                    eq_sets, *reversed(equality_predicate.args)
                )
        return eq_sets

    @staticmethod
    def add_equality_with_constant(eq_sets, symb, const):
        found_eq_set = False
        for eq_set in eq_sets:
            if any(term == symb for term in eq_set):
                eq_set.add(const)
                found_eq_set = True
        if not found_eq_set:
            eq_sets.append({symb, const})

    @staticmethod
    def add_equality_with_symbol(eq_sets, first, second):
        found_eq_set = False
        for eq_set in eq_sets:
            if any(term == first for term in eq_set):
                eq_set.add(second)
                found_eq_set = True
            elif any(term == second for term in eq_set):
                eq_set.add(first)
                found_eq_set = True
        if not found_eq_set:
            eq_sets.append({first, second})


class UnifyVariableEqualities(UnifyVariableEqualitiesMixin, ExpressionWalker):
    pass


class CollapseConjunctiveAntecedents(CollapseConjunctions):
    @add_match(
        Implication(FunctionApplication, Conjunction),
        lambda implication: any(
            isinstance(formula, Conjunction)
            for formula in implication.antecedent.formulas
        ),
    )
    def implication_with_collapsable_conjunctive_antecedent(self, implication):
        return self.walk(
            implication.apply(
                implication.consequent, self.walk(implication.antecedent)
            )
        )


def is_to_be_sorted_equality(formula):
    return (
        formula.functor == Constant(operator.eq)
        and isinstance(formula.args[1], Symbol)
        and not isinstance(formula.args[0], Symbol)
    )


class EqualitySymbolLeftHandSideNormaliseMixin(PatternWalker):
    @add_match(
        Implication(FunctionApplication, Conjunction),
        lambda implication: any(
            is_to_be_sorted_equality(conjunct)
            for conjunct in extract_logic_atoms(implication.antecedent)
        ),
    )
    def sort_antecedent_equalities(self, implication):
        new_conjuncts = tuple(
            self.reverse_equality(conjunct)
            if is_to_be_sorted_equality(conjunct)
            else conjunct
            for conjunct in implication.antecedent.formulas
        )
        new_implication = Implication(
            implication.consequent, Conjunction(new_conjuncts)
        )
        return self.walk(new_implication)

    @staticmethod
    def reverse_equality(equality):
        new_type = typing.Callable[
            [equality.args[1].type, equality.args[0].type], bool
        ]
        return Constant[new_type](operator.eq, auto_infer_type=False)(
            *reversed(equality.args)
        )
