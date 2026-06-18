from typing import AbstractSet

from ...datalog.basic_representation import DatalogProgram
from ...datalog.chase import (
    ChaseGeneral,
    ChaseNaive,
    ChaseNamedRelationalAlgebraMixin,
)
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


def cplogic_to_datalog(cpl_program):
    dl = Datalog()
    for pred_symb in cpl_program.predicate_symbols:
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


def cplogic_to_datalog_for_query(cpl_program, query_reachable_predicates):
    """
    Convert a CP-Logic program to a Datalog program restricted
    to predicates transitively reachable from a query.

    Parameters
    ----------
    cpl_program : CPLogicProgram
        The full CP-Logic program.
    query_reachable_predicates : set of Symbol
        Predicate symbols that are transitively reachable
        from the query (including the query predicate itself).

    Returns
    -------
    Datalog
        Restricted Datalog program containing only rules and
        extensional facts needed to answer the query.
    """
    dl = Datalog()
    for pred_symb in cpl_program.predicate_symbols:
        if pred_symb not in query_reachable_predicates:
            continue
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


def _get_reachable_predicates_from_query(cpl_program, query_predicate):
    from ...datalog.expression_processing import reachable_code
    query_rule = Implication(query_predicate, Constant[bool](True))
    reachable_code_union = reachable_code(query_rule, cpl_program)
    reachable_preds = {query_predicate.functor}
    for rule in reachable_code_union.formulas:
        reachable_preds.add(rule.consequent.functor)
        for pred in extract_logic_atoms(rule.antecedent):
            if isinstance(pred.functor, Symbol):
                reachable_preds.add(pred.functor)
    return reachable_preds


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


def build_grounding(cpl_program, dl_instance, restricted_predicates=None):
    groundings = []
    pred_iter = (
        cpl_program.predicate_symbols
        if restricted_predicates is None
        else restricted_predicates
    )
    for pred_symb in pred_iter:
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
    Ground only the subset of a CP-Logic program transitively
    reachable from a query predicate.

    This avoids the cost of grounding the entire program when only
    a small fraction of its predicates are needed to answer a query.

    Parameters
    ----------
    cpl_program : CPLogicProgram
        The CP-Logic program to ground.
    query_predicate : FunctionApplication
        The query predicate (e.g. ``P(x, y)``) whose transitive
        dependencies determine which rules/facts to ground.

    Returns
    -------
    ExpressionBlock
        Grounding expressions restricted to predicates reachable
        from ``query_predicate``.
    """
    reachable_preds = _get_reachable_predicates_from_query(
        cpl_program, query_predicate
    )
    # Ensure extensional / probabilistic predicates that are directly
    # referenced (but never occur as rule consequents) are still included.
    for pred in reachable_preds.copy():
        if pred in cpl_program.extensional_database():
            continue
        if pred in cpl_program.pfact_pred_symbs or \
           pred in cpl_program.pchoice_pred_symbs:
            continue
        relation = cpl_program.symbol_table.get(pred)
        if relation is None:
            continue
        for atom in extract_logic_atoms(relation):
            if isinstance(atom.functor, Symbol):
                reachable_preds.add(atom.functor)

    dl_program = cplogic_to_datalog_for_query(cpl_program, reachable_preds)
    chase = Chase(dl_program)
    dl_instance = chase.build_chase_solution()
    return build_grounding(cpl_program, dl_instance, restricted_predicates=reachable_preds)


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
