import uuid
import itertools

from ..expressions import (
    Expression, Constant, Symbol, FunctionApplication, ExpressionBlock
)
from ..unification import apply_substitution
from ..datalog.expressions import Fact, Implication, Disjunction, Conjunction
from ..exceptions import NeuroLangException
from ..datalog.chase import Chase
from ..datalog import DatalogProgram
from ..expression_pattern_matching import add_match
from ..expression_walker import PatternWalker
from ..probabilistic.ppdl import is_gdatalog_rule
from .ppdl import (
    concatenate_to_expression_block, get_antecedent_formulas, get_dterm,
    DeltaTerm
)


class ProbFact(Fact):
    '''
    A fact that occurs with a certain probability.

    A probabilistic fact is of the form p :: R(x). where R(x) is an atom and p
    is a probability.

    Notes
    -----
    - If x is a variable, its possible values are determined a typing unary
      predicate within the rules of the program.
    - If p is a variable, then it's a learnable parameter whose most probable
      value should be inferred from data.
    '''
    def __init__(self, probability, consequent):
        super().__init__(consequent)
        if not isinstance(probability, Expression):
            raise NeuroLangException('The probability must be an expression')
        self.probability = probability

    def __repr__(self):
        return 'ProbFact{{{} : {} \u2190 {}}}'.format(
            repr(self.probability), repr(self.consequent), True
        )


class ProbChoice(Implication):
    '''
    A choice over probabilistic facts.

    A probabilistic choice is of the form p_1 :: R(x_1), ..., p_n :: R(x_n).
    where p_1, ..., p_n are probabilities whose sum must be less than or equal
    to 1, where x_1, ..., x_n are tuples of terms and where R is a predicate.

    One of the probabilistic fact is chosen to be true with its corresponding
    probability while the other will be false (i.e. not 2 of the probabilistic
    facts can be true at the same time).
    '''
    def __init__(self, probfacts):
        sum_const_probs = 0.
        for probfact in probfacts:
            if isinstance(probfact.probability, Constant[float]):
                sum_const_probs += probfact.probability.value
        if sum_const_probs > 1:
            raise NeuroLangException(
                'Sum of probabilities in probabilistic '
                'choice cannot be greater than 1'
            )
        consequent = Disjunction(probfacts)
        super().__init__(consequent, Constant[bool](True))


class ProbDatalogProgram(DatalogProgram):
    '''
    Datalog extended with probabilistic facts semantics from ProbLog.

    It adds a probabilistic database which is a set of probabilistic facts.

    Probabilistic facts are stored in the symbol table of the program such that
    the key in the symbol table is the symbol of the predicate of the
    probabilsitic fact and the value is the probabilistic fact itself.
    '''
    @add_match(ProbFact)
    def probabilistic_fact(self, probfact):
        predicate = probfact.consequent.functor
        if predicate in self.symbol_table:
            raise NeuroLangException(f'Predicate {predicate} already defined')
        self.symbol_table[predicate] = probfact
        return probfact

    def probabilistic_database(self):
        '''Returns probabilistic facts of the symbol table.'''
        return {
            k: v
            for k, v in self.symbol_table.items()
            if isinstance(v, ProbFact)
        }

    @property
    def probabilistic_rules(self):
        '''
        Rules in the program with at least one atom in their antecedent whose
        predicate is defined via a probabilistic fact.
        '''
        probabilistic_predicates = set(self.probabilistic_database().keys())
        return {
            k: v
            for k, v in self.intensional_database().items()
            if (
                isinstance(v.antecedent, Conjunction) and any(
                    formula.functor in probabilistic_predicates
                    for formula in v.antecedent.formulas
                )
            ) or v.antecedent.functor in probabilistic_predicates
        }

    @property
    def parametric_probfacts(self):
        '''
        Probabilistic facts in the program whose probabilities are parameters.
        '''
        return {
            k: v
            for k, v in self.symbol_table.items()
            if isinstance(v, ProbFact) and isinstance(v.probability, Symbol)
        }


class GDatalogToProbDatalogTranslator(PatternWalker):
    '''
    Translate a GDatalog program to a ProbDatalog program.

    A GDatalog probabilsitic rule whose delta term's distribution is finite can
    be represented as a probabilistic choice. If the distribution is a
    bernoulli distribution, it can be represented as probabilistic fact.
    '''
    @add_match(Implication, is_gdatalog_rule)
    def rule(self, rule):
        '''
        Translate a GDatalog rule whose delta term is bernoulli distributed to
        an expression block containing a probabilistic fact and a
        (deterministic) rule.

        Example
        -------
        Let tau be the following walked GDatalog rule (with syntactic sugar)

            Q(x_1, ..., x_{i-1}, B[[0.3]], x_{i+1}, ..., x_n) :- P(x).

        where x is the tuple (x_1, ..., x_{i-1}, x_{i+1}, ..., x_n) and where
        P(x) is a conjunction of atoms over x.

        The following block is returned

            0.3 :: ProbFact_Q_<uid_1>(x).
            Q(x) :- P(x), ProbFact_Q_<uid_1>(x).

        '''
        datom = rule.consequent
        dterm = get_dterm(datom)
        predicate = datom.functor
        if not dterm.functor.name == 'bernoulli':
            raise NeuroLangException(
                'Other distributions than bernoulli are not supported'
            )
        probability = dterm.args[0]
        probfact_predicate = Symbol(
            'ProbFact_{}_{}'.format(predicate.name, uuid.uuid1())
        )
        terms = tuple(
            arg for arg in datom.args if not isinstance(arg, DeltaTerm)
        )
        probfact_atom = probfact_predicate(*terms)
        new_rule = Implication(
            predicate(*terms),
            conjunct_formulas(rule.antecedent, probfact_atom)
        )
        return ExpressionBlock((
            self.walk(ProbFact(probability,
                               probfact_atom)), self.walk(new_rule)
        ))

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


def conjunct_formulas(f1, f2):
    '''Conjunct two logical formulas.'''
    if isinstance(f1, Conjunction) and isinstance(f2, Conjunction):
        return Conjunction(list(f1.formulas) + list(f2.formulas))
    elif isinstance(f1, Conjunction):
        return Conjunction(list(f1.formulas) + [f2])
    elif isinstance(f2, Conjunction):
        return Conjunction([f1] + list(f2.formulas))
    else:
        return Conjunction([f1, f2])


class GDatalogToProbDatalog(
    GDatalogToProbDatalogTranslator, ProbDatalogProgram
):
    pass


def get_antecedent_predicates(rule):
    antecedent_literals = get_antecedent_formulas(rule)
    return [literal.functor for literal in antecedent_literals]


def substitute_dterm(datom, substitute):
    new_args = tuple(
        substitute if isinstance(arg, DeltaTerm) else arg for arg in datom.args
    )
    return FunctionApplication[datom.type](datom.functor, new_args)


def get_antecedent_atom_matching_predicate(predicate, rule):
    if isinstance(rule.antecedent, FunctionApplication):
        if rule.antecedent.functor == predicate:
            return rule.antecedent
        else:
            raise NeuroLangException(f'No atom with predicate {predicate}')
    elif isinstance(rule.antecedent, Conjunction):
        for formula in rule.antecedent.formulas:
            if formula.functor == predicate:
                return formula
        else:
            raise NeuroLangException(f'No atom with predicate {predicate}')


def get_possible_ground_substitutions(probfact, rule, interpretation):
    '''
    Get all possible substitutions that ground a given probabilistic fact in a
    given interpretation based on a rule where the predicate of the
    probabilistic fact occurs.
    '''
    predicate = probfact.consequent.functor
    probfact_antecedent_atom = get_antecedent_atom_matching_predicate(
        predicate, rule
    )
    variables = set(
        arg for arg in probfact_antecedent_atom.args
        if isinstance(arg, Symbol)
    )
    typing_atoms = set(
        formula for formula in get_antecedent_formulas(rule)
        if len(formula.args) == 1 and formula.args[0] in variables
    )
    substitutions_per_variable = {
        atom.args[0]: frozenset(
            fact.consequent.args[0] for fact in interpretation
            if fact.consequent.functor == atom.functor
        )
        for atom in typing_atoms
    }
    return frozenset({
        frozenset(zip(substitutions_per_variable.keys(), values))
        for values in itertools.product(*substitutions_per_variable.values())
    })


def full_observability_parameter_estimation(program, interpretations):
    '''
    Estimate parametric probabilities of the probabilistic facts in a given
    ProbDatalog program using the given fully-observable interpretations.

    This computation relies on a the domain of each variable occurring in the
    probabilistic facts to be defined by each interpretation using unary
    predicates, as explained in [1]_.

    .. [1] Gutmann et al., "Learning the Parameters of Probabilistic Logic
       Programs", section 3.1.

    '''
    estimations = dict()
    probabilistic_rules = program.probabilistic_rules
    parametric_probfacts = program.parametric_probfacts
    for parameter, probfact in parametric_probfacts.keys():
        rule = probabilistic_rules[probfact.consequent.functor].formulas[0]
        count = 0.
        normaliser = 0.
        for interpretation in interpretations:
            substitutions = get_possible_ground_substitutions(
                probfact, rule, interpretation
            )
            normaliser += len(substitutions)
            for substitution in substitutions:
                ground_fact = apply_substitution(
                    probfact.consequent, substitution
                )
                if ground_fact in interpretation:
                    count += 1
        estimations[probfact.consequent.functor] = count / normaliser
    return estimations
