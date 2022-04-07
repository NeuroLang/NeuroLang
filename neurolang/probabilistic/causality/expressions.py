from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker
from ...logic import Conjunction, Implication, NaryLogicOperator, Negation, Union
from ..exceptions import MalformedCausalOperatorError
from ..expressions import Condition
from ...expressions import Constant, Symbol

class CausalIntervention(NaryLogicOperator):
    def __init__(self, formulas):
        self.formulas = tuple(formulas)

        self._symbols = set()
        for formula in self.formulas:
            if any([not isinstance(arg, Constant) for arg in formula.args]):
                raise MalformedCausalOperatorError('''The atoms intervened by the
                DO operator can only contain constants''')
            self._symbols |= formula._symbols

    def __repr__(self):
        return f"DO{self.formulas}"


class CausalInterventionIdentification(ExpressionWalker):

    @add_match(Implication)
    def process_imp(self, implication):
        self.walk(implication.antecedent)

        return implication

    @add_match(Condition)
    def valid_intervention(self, condition):
        n_interventions = [
            formula
            for formula in condition.conditioning.formulas
            if isinstance(formula, CausalIntervention)
        ]
        if len(n_interventions) > 1:
            raise MalformedCausalOperatorError('''The use of more than one DO operator
            is not allowed. All interventions must be combined in a single DO operator.
            ''')
        elif len(n_interventions) == 1:
            self.intervention = n_interventions[0]

class CausalInterventionRewriter(ExpressionWalker):

    def __init__(self, intervention):
        self.intervention = intervention
        self.new_facts = set()

        self.intervention_replacement = {}
        self.new_head_replacement = {}

    def has_causal_intervention(rule):
        for formula in rule.antecedent.conditioning.formulas:
            if isinstance(formula, CausalIntervention):
                return True
        return False

    @add_match(Implication(..., Condition), has_causal_intervention)
    def rewrite_do_operator(self, rule):
        if self.intervention is not None:
            old_antecedent = rule.antecedent
            new_conditioning = []
            for formula in old_antecedent.conditioning.formulas:
                if isinstance(formula, CausalIntervention):
                    for ci in formula.formulas:
                        new_conditioning.append(ci)
                else:
                    new_conditioning.append(formula)

            new_rule = Implication(
                rule.consequent,
                Condition(
                    old_antecedent.conditioned,
                    Conjunction(tuple(new_conditioning))
                )
            )
            return new_rule

    @add_match(Implication, lambda rule: not rule.consequent.functor.is_fresh)
    def rewrite_implication(self, rule):
        if self.intervention is not None:
            head = rule.consequent
            matched_atom = [atom for atom in self.intervention.formulas if head.functor == atom.functor]

            if len(matched_atom) == 1:
                matched_functor = matched_atom[0].functor
                if matched_functor not in self.intervention_replacement.keys():
                    new_int_functor = Symbol.fresh()
                    # symbol f1(x)
                    new_int_atom = new_int_functor(*head.args)
                    # symbol f1(a)
                    new_fact = new_int_functor(*matched_atom[0].args)
                    self.intervention_replacement[matched_functor] = new_int_functor
                    # added f1(a) as fact
                    self.new_facts.add(new_fact)

                    # symbol f2(x)
                    new_head_functor = Symbol.fresh()
                    new_head = new_head_functor(*head.args)
                    self.new_head_replacement[matched_functor] = new_head_functor

                    # added rule f2(x) <- f1(x)
                    f1impf2 = Implication(new_head, new_int_atom)
                    # added rule f2(x) <- old and not(f1(a))
                    notf1impf2 = Implication(new_head, Conjunction((rule.antecedent, Negation(new_int_atom))))
                    rule = Union((f1impf2, notf1impf2))
                else:
                    # symbol f1(x)
                    new_int_functor = self.intervention_replacement[matched_functor]
                    new_int_atom = new_int_functor(*head.args)

                    # symbol f2(x)
                    new_head_functor = self.new_head_replacement[matched_functor]
                    new_head = new_head_functor(*head.args)

                    # rule f2(x) <- f1(x) already exists
                    # added rule f2(x) <- old and not(f1(a))
                    rule = Implication(new_head, Conjunction((rule.antecedent, Negation(new_int_atom))))

        return rule


