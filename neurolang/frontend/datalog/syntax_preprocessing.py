from ...datalog.negation import is_conjunctive_negation
from ...expression_walker import PatternWalker, add_match
from ...expressions import NeuroLangException
from ...logic import Implication
from ...logic.horn_clauses import (
    Fol2DatalogTranslationException,
    fol_query_to_datalog_program
)
from ...probabilistic.exceptions import ForbiddenConditionalQueryNonConjunctive
from ...probabilistic.expressions import Condition


class ProbFol2DatalogMixin(PatternWalker):
    """Mixin to translate first order logic expressions
    to datalog expressions, including MARG queries
    """

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
        except NeuroLangException as e:
            raise Fol2DatalogTranslationException from e
        return self.walk(program)
