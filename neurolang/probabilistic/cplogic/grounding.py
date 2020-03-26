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


def probdatalog_to_datalog(pd_program):
    new_symbol_table = TypedSymbolTable()
    for pred_symb in pd_program.symbol_table:
        value = pd_program.symbol_table[pred_symb]
        if (
            pred_symb
            in pd_program.pfact_pred_symbs | pd_program.pchoice_pred_symbs
        ):
            if not isinstance(value, Constant[AbstractSet]):
                raise NeuroLangException(
                    "Expected grounded probabilistic facts"
                )
            columns = tuple(
                Constant[ColumnInt](ColumnInt(i))
                for i in list(range(value.value.arity))[1:]
            )
            new_symbol_table[pred_symb] = RelationalAlgebraSolver().walk(
                Projection(value, columns)
            )
        else:
            new_symbol_table[pred_symb] = value

    class Datalog(TranslateToLogic, DatalogProgram, ExpressionBasicEvaluator):
        pass

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
    if isinstance(st_item, Union):
        st_item = st_item.formulas[0]
    elif isinstance(st_item, ExpressionBlock):
        st_item = st_item.expressions[0]
    if isinstance(st_item.consequent, ProbabilisticPredicate):
        pred = st_item.consequent.body
    else:
        pred = st_item.consequent
    cols = tuple(arg.name for arg in pred.args)
    return Grounding(
        expression=st_item,
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


def build_grounding(pd_program, dl_instance):
    extensional_groundings = []
    probfact_groundings = []
    probchoice_groundings = []
    intensional_groundings = []
    for pred_symb in pd_program.predicate_symbols:
        st_item = pd_program.symbol_table[pred_symb]
        if pred_symb in pd_program.pfact_pred_symbs:
            if isinstance(st_item, Constant[AbstractSet]):
                grounding = build_pfact_grounding_from_set(pred_symb, st_item)
            else:
                grounding = build_rule_grounding(
                    pred_symb, st_item, dl_instance[pred_symb]
                )
            probfact_groundings.append(grounding)
        elif pred_symb in pd_program.pchoice_pred_symbs:
            probchoice_groundings.append(
                build_pchoice_grounding(pred_symb, st_item)
            )
        else:
            if isinstance(st_item, Constant[AbstractSet]):
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


def ground_probdatalog_program(
    pd_code, probfact_sets=None, extensional_sets=None, probchoice_sets=None,
):
    pd_program = CPLogicProgram()
    pd_program.walk(pd_code)
    if probfact_sets is not None:
        for symb, probabilistic_set in probfact_sets.items():
            pd_program.add_probfacts_from_tuples(symb, probabilistic_set)
    if extensional_sets is not None:
        for symb, extensional_set in extensional_sets.items():
            pd_program.add_extensional_predicate_from_tuples(
                symb, extensional_set
            )
    if probchoice_sets is not None:
        for symb, probchoice_set in probchoice_sets.items():
            pd_program.add_probchoice_from_tuples(symb, probchoice_set)
    for disjunction in pd_program.intensional_database().values():
        if len(disjunction.formulas) > 1:
            raise NeuroLangException(
                "Programs with several rules with the same head predicate "
                "symbol are not currently supported"
            )
    dl_program = probdatalog_to_datalog(pd_program)

    class Chase(ChaseNaive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
        pass

    chase = Chase(dl_program)
    dl_instance = chase.build_chase_solution()
    return build_grounding(pd_program, dl_instance)
