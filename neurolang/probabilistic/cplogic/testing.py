import contextlib
import itertools

import numpy as np

from ...expression_pattern_matching import add_match
from ...expression_walker import PatternWalker
from ...expressions import Constant, Symbol
from ...relational_algebra import (
    ColumnStr,
    NamedRelationalAlgebraFrozenSet,
    NaturalJoin,
    Projection,
    RenameColumn,
    Selection,
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
    TupleEqualSymbol,
    TupleSymbol,
    UnionOverTuples,
    UnionRemover,
    rename_columns_for_args_to_match,
)
from .grounding import get_grounding_predicate, ground_cplogic_program


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
        pas1.value.projection(*pas1.non_provenance_columns).to_unnamed()
        == pas2.value.projection(*pas2.non_provenance_columns).to_unnamed()
    )
    # ensure the prov col names are different so we can join the sets
    c1 = Symbol.fresh().name
    c2 = Symbol.fresh().name
    x1 = pas1.value.rename_column(pas1.provenance_column, c1)
    x2 = pas2.value.rename_column(pas2.provenance_column, c2)
    joined = x1.naturaljoin(x2)
    probs = list(joined.projection(*(c1, c2)))
    for p1, p2 in probs:
        if isinstance(p1, float) and isinstance(p2, float):
            if not np.isclose(p1, p2):
                return False
        elif p1 != p2:
            return False
    return True


def make_prov_set(iterable, columns):
    return ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(columns, iterable),
        ColumnStr(columns[0]),
    )


class TestRAPToLaTeXTranslator(PatternWalker):
    def __init__(self, cpl_program, graphical_model):
        self.cpl_program = cpl_program
        self.graphical_model = graphical_model
        self.fresh_symbol_renames = dict()
        self.fresh_symbol_rename_count = 1
        self.colors = itertools.cycle(
            ["blue", "red", "pink", "teal", "olive", "magenta", "cyan"]
        )

    def prettify(self, exp):
        name = exp.value if isinstance(exp, Constant) else exp.name
        if not name.startswith("fresh_"):
            return name
        if name in self.fresh_symbol_renames:
            return self.fresh_symbol_renames[name]
        if isinstance(exp, TupleSymbol):
            prefix = "\\nu"
        elif isinstance(exp, Constant[ColumnStr]):
            prefix = "c"
        else:
            prefix = "s"
        new_name = "{}_{{{}}}".format(prefix, self.fresh_symbol_rename_count)
        self.fresh_symbol_renames[name] = new_name
        self.fresh_symbol_rename_count += 1
        return new_name

    @add_match(Projection)
    def projection(self, op):
        inner = self.walk(op.relation)
        inner = "\n".join("  " + x for x in inner.split("\n"))
        return (
            "\\pi_{"
            + ", ".join(self.prettify(c) for c in op.attributes)
            + "}"
            + "\n\\left(\n"
            + inner
            + "\n\\right)"
        )

    @add_match(RenameColumn)
    def rename_column(self, op):
        inner = self.walk(op.relation)
        inner = "\n".join("  " + x for x in inner.split("\n"))
        return (
            "\\rho_{"
            + self.prettify(op.src)
            + " / "
            + self.prettify(op.dst)
            + "}"
            + "\n\\left(\n"
            + inner
            + "\n\\right)"
        )

    @add_match(Selection(..., TupleEqualSymbol))
    def selection_by_tuple_symbol(self, op):
        inner = self.walk(op.relation)
        inner = "\n".join("  " + x for x in inner.split("\n"))
        return (
            "\\sigma_{"
            + "({}) = {}".format(
                ", ".join(self.prettify(c) for c in op.formula.columns),
                self.prettify(op.formula.tuple_symbol),
            )
            + "}"
            + "\n\\left(\n"
            + inner
            + "\n\\right)"
        )

    @add_match(Selection)
    def selection(self, op):
        inner = self.walk(op.relation)
        inner = "\n".join("  " + x for x in inner.split("\n"))
        return (
            "\\sigma_{"
            + "{} = {}".format(
                self.prettify(op.formula.args[0]),
                self.prettify(op.formula.args[1]),
            )
            + "}"
            + "\n\\left(\n"
            + inner
            + "\n\\right)"
        )

    @add_match(NaturalJoin)
    def naturaljoin(self, op):
        left = self.walk(op.relation_left)
        right = self.walk(op.relation_right)
        left = "\n".join("  " + x for x in left.split("\n"))
        right = "\n".join("  " + x for x in right.split("\n"))
        color1 = next(self.colors)
        color2 = next(self.colors)
        return (
            "\\left[\n"
            + "{\\color{"
            + color1
            + "}\n"
            + left
            + "}"
            + "\n\\right]\n"
            + "\\bowtie\n"
            + "\\left[\n"
            + "{\\color{"
            + color2
            + "}\n"
            + right
            + "}"
            + "\n\\right]"
        )

    @add_match(UnionOverTuples)
    def union_over_tuples(self, op):
        pred_symb = get_grounding_predicate(
            op.__debug_expression__
        ).functor.name
        inner = self.walk(op.relation)
        inner = "\n".join("  " + x for x in inner.split("\n"))
        return (
            "\\bigcup_{"
            + self.prettify(op.tuple_symbol)
            + "\\in \\mathcal{{{}}}}}".format(pred_symb)
            + "\n\\left\\{\n"
            + inner
            + "\n\\right\\}"
        )

    @add_match(ProvenanceAlgebraSet)
    def provenance_algebra_set(self, prov_set):
        if not hasattr(prov_set, "__debug_expression__"):
            raise RuntimeError(
                "Cannot convert to LaTeX without expression information "
                "stored in __debug_expression__ attribute"
            )
        pred = get_grounding_predicate(prov_set.__debug_expression__)
        string = f"\\mathcal{{{pred.functor.name}}}"
        if hasattr(prov_set, "__debug_alway_true__"):
            string += "_1"
        return string


def rap_expression_to_latex(exp, cpl_program, graphical_model):
    walker = TestRAPToLaTeXTranslator(cpl_program, graphical_model)
    latex = walker.walk(exp)
    return latex


def inspect_resolution(qpred, cpl_program, tex_out_path=None):
    grounded = ground_cplogic_program(cpl_program)
    translator = CPLogicGroundingToGraphicalModelTranslator()
    gm = translator.walk(grounded)
    qpred_symb = qpred.functor
    qpred_args = qpred.args
    solver = CPLogicGraphicalModelProvenanceSolver(gm)
    query_node = gm.get_node(qpred_symb)
    exp = solver.walk(ProbabilityOperation((query_node, TRUE), tuple()))
    result_args = get_grounding_predicate(query_node.expression).args
    exp = rename_columns_for_args_to_match(exp, result_args, qpred_args)
    gm = build_gm(cpl_program)
    spusher = SelectionOutPusher()
    sexp = spusher.walk(exp)
    uremover = UnionRemover()
    uexp = uremover.walk(sexp)
    if tex_out_path is not None:
        latex = rap_expression_to_latex(exp, cpl_program, gm)
        slatex = rap_expression_to_latex(sexp, cpl_program, gm)
        ulatex = rap_expression_to_latex(uexp, cpl_program, gm)
        with open(tex_out_path, "w") as f:
            f.write("\n\\\\\n".join([latex, slatex, ulatex]) + "\n")
    result = Projection(
        uexp, tuple(str2columnstr_constant(arg.name) for arg in qpred_args)
    )
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(result)
    return exp, result


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
