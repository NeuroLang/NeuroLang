from ...expression_walker import PatternWalker
from ...expression_pattern_matching import add_match
from ...expressions import Symbol, Constant, ExpressionBlock
from ...logic import Implication
from ...datalog.expression_processing import conjunct_formulas
from ..ppdl import is_ppdl_rule, get_dterm, DeltaTerm
from ..expressions import ProbabilisticPredicate
from .program import CPLogicProgram


class PPDLToCPLogicTranslator(PatternWalker):
    """
    Translate a PPDL program to a CP-Logic program.

    A PPDL probabilistic rule whose delta term's distribution is finite can
    be represented as a CP law.

    """

    @add_match(Implication, is_ppdl_rule)
    def rule(self, rule):
        datom = rule.consequent
        dterm = get_dterm(datom)
        predicate = datom.functor
        if not dterm.functor.name == "bernoulli":
            raise ValueError("Only the Bernoulli distribution is supported")
        probability = dterm.args[0]
        pfact_pred_symb = Symbol.fresh()
        terms = tuple(
            arg for arg in datom.args if not isinstance(arg, DeltaTerm)
        )
        probfact_atom = pfact_pred_symb(*terms)
        new_rule = Implication(
            predicate(*terms),
            conjunct_formulas(rule.antecedent, probfact_atom),
        )
        return self.walk(
            ExpressionBlock(
                [
                    Implication(
                        ProbabilisticPredicate(probability, probfact_atom),
                        Constant[bool](True),
                    ),
                    new_rule,
                ]
            )
        )

    @add_match(ExpressionBlock)
    def expression_block(self, block):
        expressions = []
        for expression in block.expressions:
            result = self.walk(expression)
            if isinstance(result, ExpressionBlock):
                expressions += list(result.expressions)
            else:
                expressions.append(result)
        return ExpressionBlock(expressions)


class PPDLToCPLogic(PPDLToCPLogicTranslator, CPLogicProgram):
    pass
