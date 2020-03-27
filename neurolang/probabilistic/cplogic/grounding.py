from typing import AbstractSet

from ...exceptions import NeuroLangException
from ...expressions import Constant, Symbol, ExpressionBlock
from ...expression_walker import ExpressionBasicEvaluator
from ...logic import Implication, Union
from ...logic.expression_processing import TranslateToLogic
from ...typed_symbol_table import TypedSymbolTable
from ...datalog.basic_representation import DatalogProgram
from ...datalog.chase import (
    ChaseNaive,
    ChaseGeneral,
    ChaseNamedRelationalAlgebraMixin,
)
from ...relational_algebra import (
    ColumnInt,
    RelationalAlgebraSolver,
    Projection,
    NamedRelationalAlgebraFrozenSet,
)
from ..expressions import (
    Grounding,
    ProbabilisticPredicate,
    ProbabilisticChoice,
)
from .program import CPLogicProgram


class Datalog(TranslateToLogic, DatalogProgram, ExpressionBasicEvaluator):
    pass


def cplogic_to_datalog(cpl_program):
    new_symbol_table = TypedSymbolTable()
    solver = RelationalAlgebraSolver()
    for pred_symb in cpl_program.symbol_table:
        value = cpl_program.symbol_table[pred_symb]
        if (
            pred_symb
            in cpl_program.pfact_pred_symbs | cpl_program.pchoice_pred_symbs
        ):
            columns = tuple(
                Constant[ColumnInt](ColumnInt(i))
                for i in list(range(value.value.arity))[1:]
            )
            new_symbol_table[pred_symb] = solver.walk(
                Projection(value, columns)
            )
        else:
            new_symbol_table[pred_symb] = value
    return Datalog(new_symbol_table)


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
    extensional_groundings = []
    probfact_groundings = []
    probchoice_groundings = []
    intensional_groundings = []
    for pred_symb in cpl_program.predicate_symbols:
        st_item = cpl_program.symbol_table[pred_symb]
        if pred_symb in cpl_program.pfact_pred_symbs:
            grounding = build_pfact_grounding_from_set(pred_symb, st_item)
            probfact_groundings.append(grounding)
        elif pred_symb in cpl_program.pchoice_pred_symbs:
            probchoice_groundings.append(
                build_pchoice_grounding(pred_symb, st_item)
            )
        elif isinstance(st_item, Constant[AbstractSet]):
            extensional_groundings.append(
                build_extensional_grounding(pred_symb, st_item)
            )
        else:
            intensional_groundings.append(
                build_rule_grounding(
                    pred_symb, st_item, dl_instance[pred_symb]
                )
            )
    return ExpressionBlock(
        probfact_groundings
        + probchoice_groundings
        + extensional_groundings
        + intensional_groundings
    )


class Chase(ChaseNaive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
    pass


def ground_cplogic_program(cpl_code, **sets):
    cpl_program = CPLogicProgram()
    cpl_program.walk(cpl_code)
    for prefix in ["probfact", "extensional_predicate", "probchoice"]:
        if f"{prefix}_sets" not in sets:
            continue
        add_fun = getattr(cpl_program, f"add_{prefix}_from_tuples")
        for symb, the_set in sets[f"{prefix}_sets"]:
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
