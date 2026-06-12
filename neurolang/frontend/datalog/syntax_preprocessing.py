from ...datalog.expression_processing import extract_logic_free_variables
from ...datalog.negation import is_conjunctive_negation
from .sugar import TranslateProbabilisticQueryMixin
from ...expression_walker import PatternWalker, add_match
from ...expressions import ExpressionBlock, NeuroLangException, Symbol
from ...logic import Implication, ExistentialPredicate
from ...logic.horn_clauses import (
    Fol2DatalogTranslationException,
    fol_query_to_datalog_program
)
from ...probabilistic.expressions import Condition


class ProbFol2DatalogMixin(PatternWalker):
    """Mixin to translate first order logic expressions
    to datalog expressions, including MARG queries
    """

    @add_match(
        Implication(..., ExistentialPredicate(..., Condition)),
        lambda imp: (
            is_conjunctive_negation(imp.antecedent.body.conditioned)
            and is_conjunctive_negation(imp.antecedent.body.conditioning)
        ),
    )
    def strip_ep_over_conjunctive_condition(self, imp):
        return self.walk(Implication(
            imp.consequent, imp.antecedent.body
        ))

    @add_match(
        Implication(..., ExistentialPredicate(..., Condition)),
        lambda imp: any(
            not is_conjunctive_negation(arg)
            for arg in imp.antecedent.body.unapply()
        ),
    )
    def translate_marg_query_lifted(self, imp):
        """Handle ``Implication(..., ExistentialPredicate(?v, Condition(...)))``.

        The ``AnaphoraResolutionWalker`` lifts the anaphoric variable to wrap
        the ``Condition``.  Unwrap it, call ``rewrite_conditional_query``
        directly on the inner ``Condition``, then re-apply the existential
        quantifier to each resulting rule.
        """
        var = imp.antecedent.head
        inner_imp = Implication(imp.consequent, imp.antecedent.body)
        result = TranslateProbabilisticQueryMixin.rewrite_conditional_query(
            self, inner_imp
        )
        if isinstance(result, tuple):
            return tuple(
                Implication(
                    ExistentialPredicate(var, r.consequent),
                    r.antecedent
                ) if isinstance(r, Implication) else r
                for r in result
            )
        if isinstance(result, Implication):
            return Implication(
                ExistentialPredicate(var, result.consequent),
                result.antecedent
            )
        return result

    @add_match(
        Implication(..., Condition),
        lambda imp: any(
            not is_conjunctive_negation(arg)
            for arg in imp.antecedent.unapply()
        ),
    )
    def decompose_non_conjunctive_condition(self, imp):
        condition = imp.antecedent
        results = []
        new_args = []
        for arg in (condition.conditioned, condition.conditioning):
            if is_conjunctive_negation(arg):
                new_args.append(arg)
            else:
                fv = extract_logic_free_variables(arg)
                fresh_head = Symbol.fresh()(*tuple(fv))
                aux = fol_query_to_datalog_program(fresh_head, arg)
                results.append(aux)
                new_args.append(fresh_head)
        new_condition = Condition(*new_args)
        main_impl = self.walk(Implication(imp.consequent, new_condition))
        if results:
            return ExpressionBlock([main_impl] + results)
        return main_impl

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
