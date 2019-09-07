"""
Utilities to process intermediate representations of
Datalog programs.
"""

from operator import and_, invert, or_, xor
from typing import Iterable

from ..expression_walker import PatternWalker, add_match, expression_iterator
from ..expressions import (Constant, ExpressionBlock, FunctionApplication,
                           NeuroLangException, Quantifier, Symbol)
from ..utils import OrderedSet
from .expressions import Implication


class ExtractDatalogFreeVariablesWalker(PatternWalker):
    @add_match(FunctionApplication(Constant(and_), ...))
    def conjunction(self, expression):
        fvs = OrderedSet()
        for arg in expression.args:
            fvs |= self.walk(arg)
        return fvs

    @add_match(FunctionApplication)
    def extract_variables_fa(self, expression):
        args = expression.args

        variables = OrderedSet()
        for a in args:
            if isinstance(a, Symbol):
                variables.add(a)
            elif isinstance(a, FunctionApplication):
                variables |= self.walk(a)
            elif isinstance(a, Constant):
                pass
            else:
                raise NeuroLangException('Not a Datalog function application')
        return variables

    @add_match(Quantifier)
    def extract_variables_q(self, expression):
        return self.walk(expression.body) - expression.head._symbols

    @add_match(Implication)
    def extract_variables_s(self, expression):
        return (
            self.walk(expression.antecedent) -
            self.walk(expression.consequent)
        )

    @add_match(ExpressionBlock)
    def extract_variables_eb(self, expression):
        res = set()
        for exp in expression.expressions:
            res |= self.walk(exp)

        return res

    @add_match(Symbol)
    def extract_variables_symbol(self, expression):
        return OrderedSet((expression,))

    @add_match(...)
    def _(self, expression):
        return OrderedSet()


def extract_datalog_free_variables(expression):
    """Extract variables from expression assuming it's in Datalog format.

    Parameters
    ----------
    expression : Expression


    Returns
    -------
        OrderedSet
            set of all free variables in the expression.
    """
    efvw = ExtractDatalogFreeVariablesWalker()
    return efvw.walk(expression)


def is_conjunctive_expression(expression):
    return all(
        not isinstance(exp, FunctionApplication) or
        (
            isinstance(exp, FunctionApplication) and
            (
                (
                    isinstance(exp.functor, Constant) and
                    exp.functor.value is and_
                ) or all(
                    not isinstance(arg, FunctionApplication)
                    for arg in exp.args
                )
            )
        )
        for _, exp in expression_iterator(expression)
    )


def is_conjunctive_expression_with_nested_predicates(expression):
    stack = [expression]
    while stack:
        exp = stack.pop()
        if isinstance(exp, FunctionApplication):
            if isinstance(exp.functor, Constant):
                if exp.functor.value is and_:
                    stack += exp.args
                    continue
                elif any(exp.functor.value is op for op in (or_, invert, xor)):
                    return False
            stack += [
                arg for arg in exp.args
                if isinstance(arg, FunctionApplication)
            ]
    return True


class ExtractDatalogPredicates(PatternWalker):
    @add_match(Symbol)
    def symbol(self, expression):
        return OrderedSet()

    @add_match(Constant)
    def constant(self, expression):
        return OrderedSet()

    @add_match(FunctionApplication(Constant(and_), ...))
    def conjunction(self, expression):
        res = OrderedSet()
        for arg in expression.args:
            res |= self.walk(arg)
        return res

    @add_match(FunctionApplication)
    def extract_predicates_fa(self, expression):
        return OrderedSet([expression])

    @add_match(ExpressionBlock)
    def expression_block(self, expression):
        res = OrderedSet()
        for exp in expression.expressions:
            res |= self.walk(exp)
        return res


def extract_datalog_predicates(expression):
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
    edp = ExtractDatalogPredicates()
    return edp.walk(expression)


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
    predicates = extract_datalog_predicates(rule.antecedent)
    return sum(
        int(predicate.functor == rule.consequent.functor)
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
        e.functor for e in
        extract_datalog_predicates(implication.antecedent)
    )
    predicate_set = predicate_set | {implication.consequent.functor}
    return all(
        not isinstance(e, Symbol) or e in predicate_set
        for e in preds
    )


def stratify(expression_block, datalog_instance):
    """Given an expression block containing `Implication` instances
     and a datalog instance, return the stratification of the expressions
     in the block as a list of lists..

    Parameters
    ----------
    expression_block : ExpressionBlock
        code block to be stratified.

    datalog_instance : DatalogProgram
        Datalog instance containing the EDB and IDB databases
        

    Returns
    -------
        list of lists of `Implications`, boolean
            Strata and wether it was stratisfiable.
            If it was not all non-stratified predicates
            will be in the last strata.

    """
    strata = []
    seen = set(k for k in datalog_instance.extensional_database().keys())
    seen |= set(k for k in datalog_instance.builtins())
    to_process = expression_block.expressions
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
    to_reach = [
        q.consequent.functor
        for q in query
    ]
    reached = set()
    seen_rules = set()
    while to_reach:
        p = to_reach.pop()
        reached.add(p)
        rules = idb[p]
        for rule in rules.expressions:
            if rule in seen_rules:
                continue
            seen_rules.add(rule)
            reachable_code.append(rule)
            for predicate in extract_datalog_predicates(rule.antecedent):
                functor = predicate.functor
                if functor not in reached and functor in idb:
                    to_reach.append(functor)

    return ExpressionBlock(reachable_code)
