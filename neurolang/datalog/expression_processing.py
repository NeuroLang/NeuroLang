"""
Utilities to process intermediate representations of
Datalog programs.
"""


import operator
from itertools import chain
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
    IdentityWalker,
    PatternWalker,
    ReplaceExpressionWalker,
    ReplaceSymbolWalker,
)
from ..expressions import Constant, Expression, FunctionApplication, Symbol
from ..logic import (
    TRUE,
    Conjunction,
    Disjunction,
    Implication,
    Negation,
    Quantifier,
    Union,
)
from ..logic import expression_processing as elp
from .expressions import TranslateToLogic

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
    preds = (
        e.functor for e in extract_logic_predicates(implication.antecedent)
    )
    predicate_set = predicate_set | {implication.consequent.functor}
    return all(not isinstance(e, Symbol) or e in predicate_set for e in preds)


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
    stratum = []
    new_to_process = []
    new_seen = set()
    for r in to_process:
        if all_body_preds_in_set(r, seen):
            stratum.append(r)
            new_seen.add(r.consequent.functor)
        else:
            new_to_process.append(r)
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
            for predicate in extract_logic_predicates(rule.antecedent):
                functor = predicate.functor
                if functor not in reached and functor in idb:
                    to_reach.append(functor)

    return Union(reachable_code[::-1])


def dependency_matrix(datalog, rules=None):
    """Produces the dependecy matrix for a datalog's
    instance intensional database (IDB).

    Parameters
    ----------
    datalog : DatalogProgram
        datalog instance containing the EDB and IDB.
    rules : None or Union of rules
        an optional subset of rules from the datalog
        program's IDB.

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
        for predicate in extract_logic_predicates(rule.antecedent):
            functor = predicate.functor
            if functor in edb or functor in constraint_symbols:
                continue
            elif functor in idb_symbols:
                ix_functor = idb_symbols.index(functor)
                dependency_matrix[ix_head, ix_functor] += 1
            elif isinstance(functor, Symbol) and (
                functor not in datalog.symbol_table
                or functor in datalog.intensional_database()
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
    """Only conjunct the given list of formulas if there is more than one."""
    if len(formulas) == 1:
        return formulas[0]
    else:
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
    elif isinstance(expression, FunctionApplication):
        return Conjunction((expression,))
    raise ForbiddenExpressionError(
        "Cannot conjunct expression of type {}".format(type(expression))
    )


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

    @add_match(
        FunctionApplication,
        lambda fa: all(isinstance(arg, (Constant, Symbol)) for arg in fa.args),
    )
    def function_application(self, expression):
        functor = expression.functor
        if functor not in self.idb:
            return expression

        args = expression.args
        ucq = self.idb[functor]
        cqs = []
        for cq in ucq.formulas:
            cq = self._normalise_arguments(cq, args)
            antecedent = self.walk(cq.antecedent)
            cqs.append(antecedent)

        if len(cqs) > 1:
            res = Disjunction(tuple(cqs))
        else:
            res = cqs[0]
        return res

    def _normalise_arguments(self, cq, args):
        free_variables = extract_logic_free_variables(cq)
        linked_variables = cq.consequent.args
        cq = ReplaceExpressionWalker(
            dict(
                chain(
                    zip(linked_variables, args),
                    zip(
                        free_variables,
                        (Symbol.fresh() for _ in range(len(free_variables))),
                    ),
                )
            )
        ).walk(cq)
        return cq

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


def is_rule_with_builtin(rule, known_builtins=None):
    if known_builtins is None:
        known_builtins = set()
    return any(
        isinstance(pred.functor, Constant) or pred.functor in known_builtins
        for pred in extract_logic_predicates(rule.antecedent)
    )


def remove_conjunction_duplicates(conjunction):
    return Conjunction(tuple(set(conjunction.formulas)))


class ExtractedEquality(FunctionApplication):
    pass


class PropagatedEquality(FunctionApplication):
    pass


def is_equality_between_symbol_and_symbol_or_constant(formula):
    return (
        isinstance(formula, FunctionApplication)
        and formula.functor == EQ
        and len(formula.args) == 2
        and all(isinstance(arg, (Constant, Symbol)) for arg in formula.args)
    )


class VariableEqualityExtractor(PatternWalker):
    def __init__(self):
        self._equality_sets = list()

    @property
    def substitutions(self):
        substitutions = dict()
        for eq_set in self._equality_sets:
            if any(isinstance(term, Constant) for term in eq_set):
                update = self._get_const_equality_substitutions(eq_set)
            else:
                update = self._get_between_symbs_equality_substitutions(eq_set)
            substitutions.update(update)
        return substitutions

    @add_match(
        Conjunction,
        lambda conj: any(
            is_equality_between_symbol_and_symbol_or_constant(formula)
            and not isinstance(
                formula, (ExtractedEquality, PropagatedEquality)
            )
            for formula in conj.formulas
        ),
    )
    def conjunction_with_variable_equality(self, conjunction):
        conjuncts = tuple(
            self.walk(formula) for formula in conjunction.formulas
        )
        return self.walk(Conjunction(conjuncts))

    @add_match(FunctionApplication(EQ, (Symbol, Symbol)))
    def variable_equality_between_variables(self, function_application):
        first, second = function_application.args
        self._add_equality_with_symbol(first, second)
        return ExtractedEquality(*function_application.unapply())

    @add_match(FunctionApplication(EQ, (Constant, Symbol)))
    def variable_equality_with_constant_reversed(self, function_application):
        functor, (const, symb) = function_application.unapply()
        return function_application.apply(functor, (symb, const))

    @add_match(FunctionApplication(EQ, (Symbol, Constant)))
    def variable_equality_with_constant(self, function_application):
        symb, const = function_application.args
        self._add_equality_with_constant(symb, const)
        return ExtractedEquality(*function_application.unapply())

    def _add_equality_with_constant(self, symb, const):
        found_eq_set = False
        for eq_set in self._equality_sets:
            if any(term == symb for term in eq_set):
                eq_set.add(const)
                found_eq_set = True
        if not found_eq_set:
            self._equality_sets.append({symb, const})

    def _add_equality_with_symbol(self, first, second):
        found_eq_set = False
        for eq_set in self._equality_sets:
            if any(term == first for term in eq_set):
                eq_set.add(second)
                found_eq_set = True
            elif any(term == second for term in eq_set):
                eq_set.add(first)
                found_eq_set = True
        if not found_eq_set:
            self._equality_sets.append({first, second})

    @staticmethod
    def _get_const_equality_substitutions(eq_set):
        const = next(term for term in eq_set if isinstance(term, Constant))
        return {term: const for term in eq_set if isinstance(term, Symbol)}

    @staticmethod
    def _get_between_symbs_equality_substitutions(eq_set):
        iterator = iter(eq_set)
        chosen_symb = next(iterator)
        return {symb: chosen_symb for symb in iterator}


class VariableEqualityPropagator(PatternWalker):
    @add_match(
        Implication(FunctionApplication(Symbol, ...), Conjunction),
        lambda implication: any(
            is_equality_between_symbol_and_symbol_or_constant(formula)
            and not isinstance(
                formula, (ExtractedEquality, PropagatedEquality)
            )
            for formula in implication.antecedent.formulas
        ),
    )
    def extract_and_replace_var_eqs_in_implication(self, implication):
        (
            new_antecedent,
            replacer,
        ) = self._extract_and_propagate_equalities(implication.antecedent)
        new_consequent = replacer.walk(implication.consequent)
        new_implication = Implication[implication.type](
            new_consequent, new_antecedent
        )
        return self.walk(new_implication)

    @add_match(
        Conjunction,
        lambda conjunction: any(
            is_equality_between_symbol_and_symbol_or_constant(formula)
            and not isinstance(
                formula, (ExtractedEquality, PropagatedEquality)
            )
            for formula in conjunction.formulas
        ),
    )
    def extract_and_replace_var_eqs_in_conjunction(self, conjunction):
        (
            new_conjunction,
            _,
        ) = self._extract_and_propagate_equalities(conjunction)
        return self.walk(new_conjunction)

    @staticmethod
    def _extract_and_propagate_equalities(conjunction):
        extractor = VariableEqualityPropagator._Extractor()
        new_conjunction = extractor.walk(conjunction)
        replacer = ReplaceSymbolWalker(extractor.substitutions)
        conjuncts = list()
        for formula in new_conjunction.formulas:
            if is_equality_between_symbol_and_symbol_or_constant(formula):
                conjuncts.append(PropagatedEquality(*formula.unapply()))
            else:
                conjuncts.append(replacer.walk(formula))
        new_conjunction = Conjunction(conjuncts)
        return new_conjunction, replacer

    class _Extractor(VariableEqualityExtractor, IdentityWalker):
        pass


class TautologyRemover(PatternWalker):
    @add_match(
        Conjunction,
        lambda conjunction: any(
            formula == TRUE for formula in conjunction.formulas
        ),
    )
    def simplifiable_conjunction(self, conjunction):
        return Conjunction[bool](
            tuple(
                formula for formula in conjunction.formulas if formula != TRUE
            )
        )


class PropagatedEqualityRemover(TautologyRemover):
    @add_match(PropagatedEquality)
    def replace_propagated_equality_with_true(self, equality):
        return TRUE

    @add_match(
        Implication(..., Conjunction),
        lambda implication: any(
            isinstance(formula, PropagatedEquality)
            for formula in implication.antecedent.formulas
        ),
    )
    def implication_with_propagated_equality(self, implication):
        return self.walk(
            implication.apply(
                implication.consequent, self.walk(implication.antecedent)
            )
        )

    @add_match(
        Conjunction,
        lambda conjunction: any(
            isinstance(formula, PropagatedEquality)
            for formula in conjunction.formulas
        ),
    )
    def conjunction_with_propagated_equality(self, conjunction):
        return self.walk(
            conjunction.apply(
                tuple(self.walk(formula) for formula in conjunction.formulas)
            )
        )
