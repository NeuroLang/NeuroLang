import itertools

from ..expressions import ExpressionBlock
from .ppdl import concatenate_to_expression_block, get_antecedent_literals
from ..datalog.expressions import Fact
from ..datalog.chase import Chase


class ProbabilisticFact(Fact):
    def __init__(self, consequent, probability):
        super().__init__(consequent)
        self.probability = probability

    def __repr__(self):
        return 'ProbFact{{{} : {} \u2190 {}}}'.format(
            repr(self.probability), repr(self.consequent), True
        )


def get_antecedent_predicates(rule):
    antecedent_literals = get_antecedent_literals(rule)
    return [literal.functor for literal in antecedent_literals]


def split_probfacts_and_background_knowledge(program):
    '''
    Given a Datalog program with probabilistic facts, seperate
    probabilistic facts from the background knowledege.

    The background knowledge contains all rules and ground
    non-probabilistic facts.
    '''
    probabilistic_facts = ExpressionBlock(tuple())
    background_knowledge = ExpressionBlock(tuple())
    for expression in program:
        if isinstance(expression, ProbabilisticFact):
            probabilistic_facts = concatenate_to_expression_block(
                probabilistic_facts, expression
            )
        else:
            background_knowledge = concatenate_to_expression_block(
                background_knowledge, expression
            )
    return probabilistic_facts, background_knowledge


def get_probfacts_possible_ground_substitutions_in_interpretation(
    probfacts, background_knowledge, interpretation
):
    '''
    Generate all possible substitutions that ground probabilistic facts in a
    given interpretation.

    It is assumed that the interpretation contains an explicit definition of
    the different types in the form of fully-observable unary predicates (see
    section 3.1 of "Learning the Parameters of Probabilistic Logic Programs"
    by Gutmann et al.).

    For example, let `p :: P(x)` be a probabilistic fact and let `Q(x) :- T(x),
    P(x)` be a rule in the background knowledge of the program. `T` is the
    unary predicate that defines the type of the variable x which is also the
    first argument of the literal P(x) whose predicate is the same as the one
    of the probabilistic fact. The interpretation (a set of ground facts) must
    contain the `k` ground facts `T(a_1), ..., T(a_k)`.
    '''
    pass


def generate_possible_outcomes(probabilistic_program):
    pfacts, base_program = split_program_probabilistic_deterministic(
        probabilistic_program
    )
    for facts in itertools.product(
        *[(({pfact.fact}, pfact.probability), (set(), 1.0 - pfact.probability))
          for pfact in pfacts]
    ):
        probability = None
        facts = set.union(*facts)
        program = concatenate_to_expression_block(base_program, facts)
        chase = Chase()
        solution = chase.build_chase_solution(program)
