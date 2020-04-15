from typing import AbstractSet

from ...datalog.basic_representation import DatalogProgram
from ...datalog.chase import (
    ChaseGeneral,
    ChaseNaive,
    ChaseNamedRelationalAlgebraMixin,
)
from ...exceptions import NeuroLangException
from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import Constant, ExpressionBlock, Symbol
from ...logic import Implication
from ...logic.expression_processing import (
    TranslateToLogic,
    extract_logic_predicates,
)
from ...relational_algebra import (
    ColumnInt,
    NamedRelationalAlgebraFrozenSet,
    Projection,
    RelationalAlgebraSolver,
)
from ..expressions import (
    Grounding,
    ProbabilisticChoice,
    ProbabilisticPredicate,
)
from .program import CPLogicProgram


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


def build_pchoice_grounding(pred_symb, relation):
    args = tuple(Symbol.fresh() for _ in range(relation.value.arity - 1))
    predicate = pred_symb(*args)
    expression = ProbabilisticChoice(predicate)
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=(Symbol.fresh().name,) + tuple(a.name for a in args),
            iterable=relation.value,
        )
    )
    return Grounding(expression=expression, relation=relation)


def build_pfact_grounding_from_set(pred_symb, relation):
    param_symb = Symbol.fresh()
    args = tuple(Symbol.fresh() for _ in range(relation.value.arity - 1))
    expression = Implication(
        ProbabilisticPredicate(param_symb, pred_symb(*args)),
        Constant[bool](True),
    )
    return Grounding(
        expression=expression,
        relation=Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                columns=(param_symb.name,) + tuple(arg.name for arg in args),
                iterable=relation.value,
            )
        ),
    )


def build_grounding(cpl_program, dl_instance):
    groundings = []
    for pred_symb in cpl_program.predicate_symbols:
        st_item = cpl_program.symbol_table[pred_symb]
        if pred_symb in cpl_program.pfact_pred_symbs:
            groundings.append(
                build_pfact_grounding_from_set(pred_symb, st_item)
            )
        elif pred_symb in cpl_program.pchoice_pred_symbs:
            groundings.append(build_pchoice_grounding(pred_symb, st_item))
        elif isinstance(st_item, Constant[AbstractSet]):
            groundings.append(build_extensional_grounding(pred_symb, st_item))
        else:
            groundings.append(
                build_rule_grounding(
                    pred_symb, st_item, dl_instance[pred_symb]
                )
            )
    return ExpressionBlock(groundings)


class Chase(ChaseNaive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
    pass


def ground_cplogic_program(cpl_code, **sets):
    cpl_program = CPLogicProgram()
    cpl_program.walk(cpl_code)
    for prefix in ["probfacts", "extensional_predicate", "probchoice"]:
        if f"{prefix}_sets" not in sets:
            continue
        add_fun = getattr(cpl_program, f"add_{prefix}_from_tuples")
        for symb, the_set in sets[f"{prefix}_sets"].items():
            add_fun(symb, the_set)
    for disjunction in cpl_program.intensional_database().values():
        if len(disjunction.formulas) > 1:
            raise NeuroLangException(
                "Programs with several rules with the same head predicate "
                "symbol are not currently supported"
            )
    dl_program = cplogic_to_datalog(cpl_program)
    chase = Chase(dl_program)
    dl_instance = chase.build_chase_solution()
    return build_grounding(cpl_program, dl_instance)


def get_predicate_from_grounded_expression(expression):
    if isinstance(expression, ProbabilisticChoice):
        return expression.predicate
    elif is_probabilistic_fact(expression):
        return expression.consequent.body
    elif isinstance(expression, FunctionApplication):
        return expression
    else:
        return expression.consequent


def get_grounding_pred_symb(grounding):
    if isinstance(grounding.expression, ProbabilisticChoice):
        return grounding.expression.predicate.functor
    elif isinstance(grounding.expression.consequent, ProbabilisticPredicate):
        return grounding.expression.consequent.body.functor
    return grounding.expression.consequent.functor


def get_grounding_dependencies(grounding):
    if isinstance(grounding.expression, ProbabilisticChoice):
        return set()
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
        pred_symb = get_grounding_pred_symb(grounding)
        pred_symb_to_grounding[pred_symb] = grounding
        dependencies[pred_symb] = get_grounding_dependencies(grounding)
    result = list()
    visited = set()
    for grounding in groundings:
        pred_symb = get_grounding_pred_symb(grounding)
        topological_sort_groundings_util(
            pred_symb, dependencies, visited, result
        )
    return [pred_symb_to_grounding.get(pred_symb) for pred_symb in result]
