import contextlib

import numpy as np

from ...expressions import Constant, Symbol
from ...rap_to_latex import (
    LaTeXReinitialiser,
    RAPLaTeXWriter,
    RAPToLaTeX,
    save_latex,
)
from ...relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    Projection,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
)
from .cplogic_to_gm import CPLogicGroundingToGraphicalModelTranslator
from .gm_provenance_solver import (
    TRUE,
    CPLogicGraphicalModelProvenanceSolver,
    ProbabilityOperation,
    SelectionOutPusher,
    UnionRemover,
    rename_columns_for_args_to_match,
)
from .grounding import get_grounding_predicate, ground_cplogic_program


class LaTeXSelectionOutPusher(RAPLaTeXWriter, SelectionOutPusher):
    pass


class LaTeXUnionRemover(RAPLaTeXWriter, UnionRemover):
    pass


class LaTeXRAPSolver(
    RAPLaTeXWriter, RelationalAlgebraProvenanceCountingSolver
):
    def __init__(self, *args, **kwargs):
        self.symbol_table = {}
        super().__init__(*args, **kwargs)


def build_gm(cpl_program):
    grounded = ground_cplogic_program(cpl_program)
    translator = CPLogicGroundingToGraphicalModelTranslator()
    gm = translator.walk(grounded)
    return gm


def get_named_relation_tuples(relation):
    if isinstance(relation, Constant):
        relation = relation.value
    return set(tuple(x) for x in relation)


def eq_prov_relations(pas1, pas2):
    assert isinstance(pas1, ProvenanceAlgebraSet)
    assert isinstance(pas2, ProvenanceAlgebraSet)
    assert (
        pas1.value.projection(*(c.value for c in pas1.non_provenance_columns))
    ) == (
        pas2.value.projection(*(c.value for c in pas2.non_provenance_columns))
    )
    # ensure the prov col names are different so we can join the sets
    c1 = Symbol.fresh().name
    c2 = Symbol.fresh().name
    x1 = pas1.value.rename_column(pas1.provenance_column.value, c1)
    x2 = pas2.value.rename_column(pas2.provenance_column.value, c2)
    joined = x1.naturaljoin(x2)
    probs = list(joined.projection(*(c1, c2)))
    for p1, p2 in probs:
        if not np.isclose(p1, p2):
            return False
    return True


def make_prov_set(iterable, columns):
    return ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(columns, iterable),
        str2columnstr_constant(columns[0]),
    )


def inspect_resolution(qpred, cpl_program, tex_out_path=None):
    grounded = ground_cplogic_program(cpl_program)
    translator = CPLogicGroundingToGraphicalModelTranslator()
    gm = translator.walk(grounded)
    qpred_symb, qpred_args = qpred.functor, qpred.args
    solver = CPLogicGraphicalModelProvenanceSolver(gm)
    query_node = gm.get_node(qpred_symb)
    exp = solver.walk(ProbabilityOperation((query_node, TRUE), tuple()))
    result_args = get_grounding_predicate(query_node.expression).args
    exp = rename_columns_for_args_to_match(exp, result_args, qpred_args)
    gm = build_gm(cpl_program)
    reinitialiser = LaTeXReinitialiser()
    latex_translator = RAPToLaTeX(cpl_program, gm)
    spusher = LaTeXSelectionOutPusher(translator=latex_translator)
    latex = spusher.latex
    sexp = reinitialiser.walk(spusher.walk(exp))
    uremover = LaTeXUnionRemover(translator=latex_translator, latex=latex)
    latex = uremover.latex
    uexp = reinitialiser.walk(uremover.walk(sexp))
    result = Projection(
        uexp, tuple(str2columnstr_constant(arg.name) for arg in qpred_args)
    )
    solver = LaTeXRAPSolver(translator=latex_translator, latex=latex)
    result = solver.walk(result)
    latex = solver.latex
    if tex_out_path is not None:
        save_latex(latex, tex_out_path)
    return exp, result


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
