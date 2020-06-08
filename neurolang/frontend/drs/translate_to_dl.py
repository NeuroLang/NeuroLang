from ...expressions import (
    Symbol,
    Constant,
    ExpressionBlock,
    FunctionApplication,
)
from ...datalog.expressions import Fact
from ...logic import (
    Implication,
    Conjunction,
    UniversalPredicate,
)
from ...logic.horn_clauses import fol_query_to_datalog_program
from .drs_builder import DRSBuilder, DRS2FOL
from .chart_parser import ChartParser
from .english_grammar import EnglishGrammar, EnglishBaseLexicon
import operator


_equals = Constant(operator.eq)


def translate_to_dl(string):
    grammar = EnglishGrammar(EnglishBaseLexicon())
    parser = ChartParser(grammar)
    builder = DRSBuilder(grammar)
    into_fol = DRS2FOL()

    program = ExpressionBlock(())

    for sentence in string.split("."):
        sentence = sentence.strip()
        if not sentence:
            continue
        t = parser.parse(sentence)[0]

        drs = builder.walk(t)
        exp = into_fol.walk(drs)

        intensional_rule = _as_intensional_rule(exp)
        if intensional_rule:
            program = ExpressionBlock(
                program.expressions + intensional_rule.expressions
            )
            continue

        fact = _as_fact(exp)
        if fact:
            program = ExpressionBlock(
                program.expressions + fact.expressions
            )
            continue

        raise Exception(f"Unsupported expression: {repr(exp)}")

    return program


def _as_intensional_rule(exp):
    ucv = ()

    while isinstance(exp, UniversalPredicate):
        ucv += (exp.head,)
        exp = exp.body

    if not isinstance(exp, Implication):
        return None

    con = exp.consequent
    ant = exp.antecedent

    if not isinstance(con, FunctionApplication):
        return None

    args = ()

    for a in con.args:
        if isinstance(a, Constant):
            s = Symbol.fresh()
            ant = Conjunction((ant, _equals(s, a)))
            args += (s,)
        elif a in ucv:
            args += (a,)
        else:
            return None

    con = con.functor(*args)

    return fol_query_to_datalog_program(con, ant)


def _as_fact(exp):
    if not isinstance(exp, FunctionApplication):
        return None

    return ExpressionBlock((Fact(exp),))
