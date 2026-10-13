from typing import AbstractSet

from ...datalog.basic_representation import DatalogProgram
from ...datalog.chase import (
    ChaseGeneral,
    ChaseNaive,
    ChaseNamedRelationalAlgebraMixin,
)
from ...datalog.expression_processing import reachable_code
from ...exceptions import ForbiddenDisjunctionError
from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import Constant, ExpressionBlock, Symbol
from ...logic import Implication
from ...logic.expression_processing import (
    TranslateToLogic,
    extract_logic_atoms,
    extract_logic_predicates,
)
from ...relational_algebra import (
    ColumnInt,
    NamedRelationalAlgebraFrozenSet,
    Projection,
    RelationalAlgebraSolver,
)
from ..expression_processing import is_probabilistic_fact
from ..expressions import (
    Grounding,
    ProbabilisticChoiceGrounding,
    ProbabilisticFact,
)


class Datalog(TranslateToLogic, DatalogProgram, ExpressionBasicEvaluator):
    pass


def remove_probability_column(relation):
    new_columns = tuple(
        Constant[ColumnInt](ColumnInt(i))
        for i in list(range(relation.value.arity))[1:]
    )
    return RelationalAlgebraSolver().walk(Projection(relation, new_columns))


def cplogic_to_datalog(cpl_program, predicate_symbols=None):
    """
    Convert a CP-Logic program to a deterministic Datalog program.

    Probabilistic facts and choices are turned into extensional
    predicates by dropping their probability column, so that the chase
    of the resulting program computes every ground atom that can hold in
    some possible world.

    Parameters
    ----------
    cpl_program : CPLogicProgram
        The CP-Logic program to convert.
    predicate_symbols : set of Symbol, optional
        If given, only these predicates (rules and facts) are included
        in the Datalog program. By default, all the predicates of the
        program are included.

    Returns
    -------
    Datalog
        The deterministic Datalog program.

    """
    if predicate_symbols is None:
        predicate_symbols = cpl_program.predicate_symbols
    dl = Datalog()
    for pred_symb in predicate_symbols:
        if pred_symb in cpl_program.intensional_database():
            if len(cpl_program.symbol_table[pred_symb].formulas) > 1:
                raise ForbiddenDisjunctionError(
                    "CP-Logic programs do not support disjunctions"
                )
            for formula in cpl_program.symbol_table[pred_symb].formulas:
                dl.walk(formula)
        else:
            if pred_symb in cpl_program.extensional_database():
                relation = cpl_program.symbol_table[pred_symb]
            else:
                relation = remove_probability_column(
                    cpl_program.symbol_table[pred_symb]
                )
            dl.add_extensional_predicate_from_tuples(pred_symb, relation.value)
    return dl


def get_query_reachable_predicates(cpl_program, query_predicate):
    """
    Compute the predicates a query transitively depends on.

    Starting from the query's predicate symbol, the rules of the
    program's intensional database are followed through their
    antecedents (including negated atoms), collecting every predicate
    symbol of the program that the query may depend on.

    Parameters
    ----------
    cpl_program : CPLogicProgram
        The CP-Logic program.
    query_predicate : FunctionApplication
        The query predicate, e.g. ``P(x, y)``.

    Returns
    -------
    set of Symbol
        Predicate symbols of ``cpl_program`` reachable from the query,
        including the query's own predicate symbol.

    """
    query_symb = query_predicate.functor
    reachable = {query_symb}
    if query_symb in cpl_program.intensional_database():
        query_rule = Implication(query_predicate, Constant[bool](True))
        for rule in reachable_code(query_rule, cpl_program).formulas:
            reachable.add(rule.consequent.functor)
            reachable |= {
                atom.functor for atom in extract_logic_atoms(rule.antecedent)
            }
    return reachable & cpl_program.predicate_symbols


def build_extensional_grounding(pred_symb, tuple_set):
    args = tuple(Symbol.fresh() for _ in range(tuple_set.value.arity))
    cols = tuple(arg.name for arg in args)
    return Grounding(
        expression=Implication(pred_symb(*args), Constant[bool](True)),
        relation=Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                columns=cols, iterable=tuple_set.value
            )
        ),
    )


def build_rule_grounding(pred_symb, st_item, tuple_set):
    rule = st_item.formulas[0]
    cols = tuple(arg.name for arg in rule.consequent.args)
    return Grounding(
        expression=rule,
        relation=Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                columns=cols, iterable=tuple_set.value
            )
        ),
    )


def build_probabilistic_grounding(pred_symb, relation, grounding_cls):
    # construct the grounded expression with fresh symbols
    # fresh symbol for the probability p in ( P(x) : p ) <- T
    prob_symb = Symbol.fresh()
    # fresh symbols for the terms x1, ..., xn in ( P(x1, ..., xn) : p ) <- T
    args = tuple(Symbol.fresh() for _ in range(relation.value.arity - 1))
    # grounded expression
    expression = Implication(
        ProbabilisticFact(prob_symb, pred_symb(*args)),
        Constant[bool](True),
    )
    # build the new relation (TODO: this could be done with a RenameColumns)
    new_relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=(prob_symb.name,) + tuple(arg.name for arg in args),
            iterable=relation.value,
        )
    )
    # finally construct the grounding using the given class
    return grounding_cls(expression=expression, relation=new_relation)


def build_pchoice_grounding(pred_symb, relation):
    return build_probabilistic_grounding(
        pred_symb, relation, ProbabilisticChoiceGrounding
    )


def build_pfact_grounding_from_set(pred_symb, relation):
    return build_probabilistic_grounding(pred_symb, relation, Grounding)


def build_grounding(cpl_program, dl_instance, predicate_symbols=None):
    if predicate_symbols is None:
        predicate_symbols = cpl_program.predicate_symbols
    groundings = []
    for pred_symb in predicate_symbols:
        relation = cpl_program.symbol_table[pred_symb]
        if pred_symb in cpl_program.pfact_pred_symbs:
            groundings.append(
                build_pfact_grounding_from_set(pred_symb, relation)
            )
        elif pred_symb in cpl_program.pchoice_pred_symbs:
            groundings.append(build_pchoice_grounding(pred_symb, relation))
        elif isinstance(relation, Constant[AbstractSet]):
            groundings.append(build_extensional_grounding(pred_symb, relation))
        else:
            groundings.append(
                build_rule_grounding(
                    pred_symb, relation, dl_instance[pred_symb]
                )
            )
    return ExpressionBlock(groundings)


class Chase(ChaseNaive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
    pass


def ground_cplogic_program(cpl_program):
    dl_program = cplogic_to_datalog(cpl_program)
    chase = Chase(dl_program)
    dl_instance = chase.build_chase_solution()
    return build_grounding(cpl_program, dl_instance)


def ground_cplogic_program_for_query(cpl_program, query_predicate):
    """
    Ground the part of a CP-Logic program a query depends on.

    Only the rules and facts of predicates transitively reachable from
    ``query_predicate`` are grounded, which avoids the cost of grounding
    the whole program when the query only needs a small part of it. The
    groundings obtained for the reachable predicates are the same as
    the ones obtained with :func:`ground_cplogic_program`.

    Parameters
    ----------
    cpl_program : CPLogicProgram
        The CP-Logic program to ground.
    query_predicate : FunctionApplication
        The query predicate (e.g. ``P(x, y)``) whose transitive
        dependencies determine which rules and facts are grounded.

    Returns
    -------
    ExpressionBlock
        Grounding expressions of the predicates reachable from
        ``query_predicate``.

    """
    reachable = get_query_reachable_predicates(cpl_program, query_predicate)
    dl_program = cplogic_to_datalog(cpl_program, reachable)
    chase = Chase(dl_program)
    dl_instance = chase.build_chase_solution()
    return build_grounding(cpl_program, dl_instance, reachable)


def get_grounding_predicate(grounded_exp):
    if is_probabilistic_fact(grounded_exp):
        return grounded_exp.consequent.body
    else:
        return grounded_exp.consequent


def get_grounding_pred_symb(grounded_exp):
    return get_grounding_predicate(grounded_exp).functor


def get_grounding_dependencies(grounding):
    predicates = extract_logic_predicates(grounding.expression.antecedent)
    return set(pred.functor for pred in predicates)


def topological_sort_groundings_util(pred_symb, dependencies, visited, result):
    for dep_symb in dependencies[pred_symb]:
        if dep_symb not in visited:
            topological_sort_groundings_util(
                dep_symb, dependencies, visited, result
            )
    if pred_symb not in visited:
        result.append(pred_symb)
    visited.add(pred_symb)


def topological_sort_groundings(groundings):
    dependencies = dict()
    pred_symb_to_grounding = dict()
    for grounding in groundings:
        pred_symb = get_grounding_pred_symb(grounding.expression)
        pred_symb_to_grounding[pred_symb] = grounding
        dependencies[pred_symb] = get_grounding_dependencies(grounding)
    result = list()
    visited = set()
    for pred_symb, grounding in pred_symb_to_grounding.items():
        topological_sort_groundings_util(
            pred_symb, dependencies, visited, result
        )
    return [pred_symb_to_grounding[pred_symb] for pred_symb in result]
