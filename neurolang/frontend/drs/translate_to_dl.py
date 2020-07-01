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


class TranslateToDatalog:
    def __init__(self):
        self.grammar = EnglishGrammar(EnglishBaseLexicon())
        self.parser = ChartParser(self.grammar)
        self.builder = DRSBuilder(self.grammar)
        self.into_fol = DRS2FOL()

    def translate_block(self, string):
        program = ExpressionBlock(())

        for sentence in string.split("."):
            sentence = sentence.strip()
            if not sentence:
                continue
            exp_block = self.translate_sentence(sentence)
            program = ExpressionBlock(
                program.expressions + exp_block.expressions
            )

        return program

    def translate_sentence(self, sentence):
        t = self.parser.parse(sentence)[0]

        drs = self.builder.walk(t)
        exp = self.into_fol.walk(drs)

        intensional_rule = _as_intensional_rule(exp)
        if intensional_rule:
            return ExpressionBlock(intensional_rule.expressions)

        fact = _as_fact(exp)
        if fact:
            return ExpressionBlock(fact.expressions)

        raise Exception(f"Unsupported expression: {repr(exp)}")


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
