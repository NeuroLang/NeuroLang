"""
Utilities to process intermediate representations of
Datalog programs.
"""

from typing import Iterable

import numpy as np

from ..exceptions import NeuroLangException
from ..expression_walker import ExpressionWalker
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import Conjunction, Negation, Quantifier, Union
from ..logic import expression_processing as elp
from .expressions import TranslateToLogic


class SymbolNotFoundException(NeuroLangException):
    pass


class RuleNotFoundException(NeuroLangException):
    pass


class TranslateToDatalogSemantics(TranslateToLogic, ExpressionWalker):
    pass


def implication_has_existential_variable_in_antecedent(implication):
    '''
    Whether an implication has at least one existential variable in its
    antecedent.
    '''
    c_free_vars = set(extract_logic_free_variables(implication.consequent))
    a_free_vars = set(extract_logic_free_variables(implication.antecedent))
    return a_free_vars > c_free_vars


def is_conjunctive_expression(expression):
    if isinstance(expression, Conjunction):
        formulas = expression.formulas
    else:
        formulas = [expression]

    return all(
        expression == Constant(True) or expression == Constant(False) or (
            isinstance(expression, FunctionApplication) and not any(
                isinstance(arg, FunctionApplication) for arg in expression.args
            )
        ) for expression in formulas
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
                arg for arg in exp.args
                if isinstance(arg, FunctionApplication)
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
    return sum(
        int(
            (predicate.formula.functor == rule.consequent.functor)
            if isinstance(predicate, Negation)
            else predicate.functor == rule.consequent.functor
        )
        for predicate in predicates
    ) < 2


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
    SymbolNotFoundException
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
                raise RuleNotFoundException(
                    f"Rule {rule} not contained in the datalog "
                    "instance."
                )
            idb_symbols.add(functor)

    idb_symbols = tuple(sorted(idb_symbols, key=lambda s: s.name))
    edb = datalog.extensional_database()

    dependency_matrix = np.zeros(
        (len(idb_symbols), len(idb_symbols)),
        dtype=int
    )

    while to_reach:
        rule = to_reach.pop()
        head_functor = rule.consequent.functor
        ix_head = idb_symbols.index(head_functor)
        for predicate in extract_logic_predicates(rule.antecedent):
            functor = predicate.functor
            if functor in edb:
                continue
            elif functor in idb_symbols:
                ix_functor = idb_symbols.index(functor)
                dependency_matrix[ix_head, ix_functor] += 1
            elif (
                isinstance(functor, Symbol) and
                (
                    functor not in datalog.symbol_table or
                    functor in datalog.intensional_database()
                )
            ):
                raise SymbolNotFoundException(
                    f'Symbol not found {functor.name}'
                )

    return idb_symbols, dependency_matrix


def program_has_loops(program_representation):
    if not isinstance(program_representation, np.ndarray):
        _, program_representation = dependency_matrix(program_representation)
    reachable = program_representation
    for _ in range(len(program_representation)):
        if any(np.diag(reachable)):
            return True
        else:
            reachable = np.dot(
               reachable, program_representation
            )

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
        return Conjunction(tuple(f1.formulas) + (f2, ))
    elif isinstance(f2, Conjunction):
        return Conjunction((f1, ) + tuple(f2.formulas))
    else:
        return Conjunction((f1, f2))


def is_ground_predicate(predicate):
    """Whether all the predicate's terms are all constant."""
    return all(isinstance(arg, Constant) for arg in predicate.args)
