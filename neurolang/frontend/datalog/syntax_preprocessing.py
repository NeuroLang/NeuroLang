from ...datalog.negation import is_conjunctive_negation
from ...exceptions import ExpressionIsNotSafeRange, NeuroLangException
from ...expression_walker import PatternWalker, add_match
from ...logic import Implication, ExistentialPredicate
from ...logic.horn_clauses import (
    Fol2DatalogTranslationException,
    fol_query_to_datalog_program
)
from ...probabilistic.exceptions import ForbiddenConditionalQueryNonConjunctive
from ...probabilistic.expressions import Condition


def _is_existentially_quantified_condition(expression):
    antecedent = expression.antecedent
    while isinstance(antecedent, ExistentialPredicate):
        antecedent = antecedent.body

    return isinstance(antecedent, Condition)


class ProbFol2DatalogMixin(PatternWalker):
    """Mixin to translate first order logic expressions
    to datalog expressions, including MARG queries
    """
    @add_match(
        Implication(..., ExistentialPredicate),
        _is_existentially_quantified_condition 
    )
    def eliminate_existential_on_marg_query(self, imp):
        antecedent = imp.antecedent
        while isinstance(antecedent, ExistentialPredicate):
            antecedent = antecedent.body
        new_imp = imp.apply(imp.consequent, antecedent)
        return self.walk(new_imp)

    @add_match(
        Implication(..., Condition),
        lambda imp: any(
            not is_conjunctive_negation(arg)
            for arg in imp.antecedent.unapply()
        ),
    )
    def translate_marg_query(self, imp):
        raise ForbiddenConditionalQueryNonConjunctive(
            "Conditions on a MARG query need to be conjunctive"
        )

    @add_match(
        Implication,
        lambda imp: not isinstance(imp.antecedent, Condition)
        and not is_conjunctive_negation(imp.antecedent),
    )
    def translate_implication(self, imp):
        try:
            program = fol_query_to_datalog_program(
                imp.consequent, imp.antecedent
            )
        except ExpressionIsNotSafeRange:
            raise
        except NeuroLangException as e:
            raise Fol2DatalogTranslationException from e
        return self.walk(program)
