from ...datalog.expression_processing import extract_logic_free_variables
from ...datalog.negation import is_conjunctive_negation
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
    )
    def strip_ep_over_condition(self, imp):
        """Strip ``ExistentialPredicate`` that wraps a ``Condition`` in the
        antecedent of an ``Implication``.

        The ``AnaphoraResolutionWalker`` lifts the existential quantifier
        (from ``a/an/every``) to wrap the ``Condition`` after cross-boundary
        resolution.  Datalog semantics already treat free variables in the
        body as existentially quantified, so the quantifier wrapper is
        unnecessary and can be removed.  The inner ``Condition`` is then
        handled naturally by downstream mixins such as
        ``rewrite_conditional_query`` or
        ``decompose_non_conjunctive_condition``.
        """
        return self.walk(Implication(
            imp.consequent, imp.antecedent.body
        ))

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
        new_condition = Condition(new_args[0], new_args[1])
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
