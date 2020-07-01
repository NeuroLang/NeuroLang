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

        try:
            return _as_intensional_rule(exp)
        except TranslateToDatalogError:
            pass

        try:
            return _as_fact(exp)
        except TranslateToDatalogError:
            pass

        raise Exception(f"Unsupported expression: {repr(exp)}")


class TranslateToDatalogError(Exception):
    pass


def _as_intensional_rule(exp):
    ucv, exp = _strip_universal_quantifiers(exp)

    if not isinstance(exp, Implication):
        raise TranslateToDatalogError("A Datalog rule must be an implication")

    head = exp.consequent
    body = exp.antecedent

    if not isinstance(head, FunctionApplication):
        raise TranslateToDatalogError(
            "The head of a Datalog rule must be a function application"
        )

    head, body, ucv = _constrain_using_head_constants(head, body, ucv)

    if any(a not in ucv for a in head.args):
        raise TranslateToDatalogError(
            "All rule head arguments must be universally quantified"
        )

    return fol_query_to_datalog_program(head, body)


def _strip_universal_quantifiers(exp):
    ucv = ()

    while isinstance(exp, UniversalPredicate):
        ucv += (exp.head,)
        exp = exp.body

    return ucv, exp


def _constrain_using_head_constants(head, body, ucv):
    args = ()

    for a in head.args:
        if isinstance(a, Constant):
            s = Symbol.fresh()
            body = Conjunction((body, _equals(s, a)))
            ucv += (s,)
            args += (s,)
        else:
            args += (a,)

    head = head.functor(*args)
    return head, body, ucv


def _as_fact(exp):
    if not isinstance(exp, FunctionApplication):
        raise TranslateToDatalogError(
            "A fact must be a single function application"
        )

    return ExpressionBlock((Fact(exp),))
