import itertools
from typing import AbstractSet

from .expression_pattern_matching import add_match
from .expression_walker import ExpressionWalker, PatternWalker
from .expressions import Constant
from .probabilistic.cplogic.grounding import get_grounding_predicate
from .relational_algebra import (
    ColumnStr,
    NaturalJoin,
    Projection,
    RelationalAlgebraOperation,
    RenameColumn,
    Selection,
)
from .relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    TupleEqualSymbol,
    TupleSymbol,
    UnionOverTuples,
    is_provenance_operation,
)


def preserve_debug_symbols(prev, new):
    attrs = ["__debug_expression__", "__debug_alway_true__"]
    for attr in attrs:
        if hasattr(prev, attr):
            setattr(new, attr, getattr(prev, attr))
    return new


class RAPToLaTeX(PatternWalker):
    def __init__(self):
        self.fresh_symbol_renames = dict()
        self.fresh_symbol_rename_count = 1
        self.colors = itertools.cycle(
            ["blue", "red", "pink", "teal", "olive", "magenta", "cyan"]
        )

    def add_pretty_name(self, name, pretty_name):
        if name in self.fresh_symbol_renames:
            return self.fresh_symbol_renames[name]
        self.fresh_symbol_renames[name] = pretty_name
        self.fresh_symbol_rename_count += 1
        return pretty_name

    def prettify(self, exp):
        name = exp.value if isinstance(exp, Constant) else exp.name
        if not name.startswith("fresh_"):
            return name
        if isinstance(exp, TupleSymbol):
            prefix = "\\nu"
        elif isinstance(exp, Constant[ColumnStr]):
            prefix = "c"
        else:
            prefix = "s"
        pretty = "{}_{{{}}}".format(prefix, self.fresh_symbol_rename_count)
        return self.add_pretty_name(name, pretty)

    def prettify_pred_symb(self, name):
        if not name.startswith("fresh_"):
            if len(name) == 1:
                return f"\\mathcal{{{name}}}"
            else:
                return f"\\mathcal{{R}}^{{\\text{{{name}}}}}"
        pretty = "\\mathcal{{R}}^{{{}}}".format(self.fresh_symbol_rename_count)
        return self.add_pretty_name(name, pretty)

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
        if hasattr(op, "__debug_expression__"):
            set_name = self.prettify_pred_symb(
                get_grounding_predicate(op.__debug_expression__).functor.name
            )
        else:
            set_name = "?"
        inner = self.walk(op.relation)
        inner = "\n".join("  " + x for x in inner.split("\n"))
        return (
            "\\bigcup_{"
            + self.prettify(op.tuple_symbol)
            + "\\in {}}}".format(set_name)
            + "\n\\left\\{\n"
            + inner
            + "\n\\right\\}"
        )

    @add_match(ProvenanceAlgebraSet)
    def provenance_algebra_set(self, prov_set):
        if hasattr(prov_set, "__debug_expression__"):
            pred = get_grounding_predicate(prov_set.__debug_expression__)
            string = self.prettify_pred_symb(pred.functor.name)
        else:
            string = "\\text{?}"
        if hasattr(prov_set, "__debug_alway_true__"):
            string += "_1"
        simple = string
        table = "\\begin{{array}}{{c|{}}}\n".format(
            "c" * len(prov_set.non_provenance_columns)
        )
        np_cols = list(
            sorted(
                list(prov_set.non_provenance_columns), key=lambda c: c.value
            )
        )
        table += "{} & {}\\\\\n".format(
            simple, " & ".join(self.prettify(col) for col in np_cols),
        )
        table += "\\hline\n"
        for _, tupl in prov_set.value._container[
            [prov_set.provenance_column.value] + [c.value for c in np_cols]
        ].iterrows():
            table += (
                " & ".join(r"\texttt{{{}}}".format(x) for x in tupl) + "\\\\\n"
            )
        table += "\\end{array}"
        return table

    @add_match(Constant[AbstractSet])
    def constant_relation(self, relation):
        relation = relation.value
        string = "\\begin{{array}}{{{}}}".format("c" * relation.arity)
        for tupl in relation:
            string += " & ".join(str(x) for x in tupl) + "\\\\\n"
        string += "\\end{array}"
        return string


HEADER = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=100cm,paperheight=20cm, margin=0.5in]{geometry}
\usepackage[usenames]{color} %used for font color
\usepackage{amssymb} %maths
\usepackage{amsmath} %maths
\usepackage{wasysym}
\usepackage[utf8]{inputenc} %useful to type directly diacritic characters
\usepackage{xcolor}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\delimitershortfall=-1pt
\DeclareMathOperator*{\bigbowtie}{\bowtie}
"""


def save_latex(latex_strings, tex_out_path):
    latex = HEADER
    latex += "\\begin{document}\n"
    for s in latex_strings:
        latex += "\\begin{equation}\n"
        latex += s + "\n"
        latex += "\\end{equation}\n"
    latex += "\\end{document}"
    with open(tex_out_path, "w") as f:
        f.write(latex)


def already_latexed(exp):
    return hasattr(exp, "__latexed__") and exp.__latexed__


class RAPLaTeXWriter(PatternWalker):
    def __init__(self, latex=None, translator=None, intermediate=False):
        self.latex = [] if latex is None else latex
        if translator is None:
            self.translator = RAPToLaTeX()
        else:
            self.translator = translator
        self.intermediate = intermediate

    @add_match(
        RelationalAlgebraOperation,
        lambda exp: is_provenance_operation(exp) and not already_latexed(exp),
    )
    def no_latex_op(self, op):
        op_latex = self.translator.walk(op)
        op.__latexed__ = True
        result = self.walk(op)
        result = preserve_debug_symbols(op, result)
        result.__latexed__ = True
        for walker_cls in self.__class__.__mro__:
            if "LaTeX" not in walker_cls.__name__:
                walker_name = walker_cls.__name__
                break
        self.latex.append(f"\\texttt{{{walker_name}}}")
        self.latex.append(op_latex)
        if self.intermediate:
            intermediate = op.apply(*(self.walk(arg) for arg in op.unapply()))
            intermediate_latex = self.translator.walk(intermediate)
            self.latex.append(intermediate_latex)
        self.latex.append("\\rightarrow " + self.translator.walk(result))
        save_latex(self.latex, "/tmp/lol.tex")
        return self.walk(result)


class LaTeXReinitialiser(ExpressionWalker):
    @add_match(
        RelationalAlgebraOperation,
        lambda exp: is_provenance_operation(exp) and already_latexed(exp),
    )
    def remove_tag(self, op):
        op.__latexed__ = False
        new_op = self.walk(op)
        preserve_debug_symbols(op, new_op)
        return new_op
