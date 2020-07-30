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
from ...expression_walker import ExpressionWalker
from ...logic.transformations import (
    CollapseConjunctions,
    DistributeUniversalQuantifiers,
    DistributeImplicationsWithConjunctiveHeads,
)


_equals = Constant(operator.eq)


def cnl_initialized(method):
    def new_method(self, *args, **kwargs):
        if not hasattr(self, "_lexicon"):
            self._initialize_cnl()
        return method(self, *args, **kwargs)

    return new_method


class CnlFrontendMixin:
    def _initialize_cnl(self):
        self._lexicon = EnglishBaseLexicon()
        self._grammar = EnglishGrammar
        self._parser = ChartParser(self._grammar, self._lexicon)
        self._drs_builder = DRSBuilder(self._grammar)
        self._into_fol = DRS2FOL()
        self._into_cos = IntoConjunctionOfSentences()

    @cnl_initialized
    def execute_cnl_code(self, code):
        for sentence in code.split("."):
            sentence = sentence.strip()
            if not sentence:
                continue
            self.execute_cnl_sentence(sentence)

    def execute_cnl_sentence(self, sentence):
        t = self._parser.parse(sentence)[0]
        drs = self._drs_builder.walk(t)
        exp = self._into_fol.walk(drs)
        exp = self._into_cos.walk(exp)
        lsentences = exp.formulas if isinstance(exp, Conjunction) else (exp,)
        for s in lsentences:
            self.execute_fol_sentence(s)

    def execute_fol_sentence(self, exp):
        program = self._translate_logical_sentence(exp)
        self.solver.walk(program)

    def _translate_logical_sentence(self, exp):
        try:
            return _as_intensional_rule(exp)
        except TranslateToDatalogError:
            pass

        try:
            return _as_fact(exp)
        except TranslateToDatalogError:
            pass

        raise TranslateToDatalogError(f"Unsupported expression: {repr(exp)}")


class IntoConjunctionOfSentences(
    DistributeImplicationsWithConjunctiveHeads,
    DistributeUniversalQuantifiers,
    CollapseConjunctions,
    ExpressionWalker,
):
    pass


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


def _add_universal_quantifiers(exp, ucv):
    ucv = list(ucv)
    while ucv:
        v = ucv.pop()
        exp = UniversalPredicate(v, exp)
    return exp


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
