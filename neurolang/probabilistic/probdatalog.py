import itertools

from ..expressions import ExpressionBlock, Expression
from .ppdl import concatenate_to_expression_block, get_antecedent_formulas
from ..datalog.expressions import Fact
from ..exceptions import NeuroLangException
from ..datalog.chase import Chase
from ..datalog import DatalogProgram
from ..expression_pattern_matching import add_match
from ..expression_walker import PatternWalker
from ..probabilistic.ppdl import is_gdatalog_rule


class ProbFact(Fact):
    '''
    A fact that occurs with a certain probability.

    A probabilistic fact is of the form p :: P(x) where P(x) is an atom and p
    is a probability.

    Notes
    -----
    - If x is a variable, its possible values are determined a typing unary
      predicate within the rules of the program.
    - If p is a variable, then it's a learnable parameter whose most probable
      value will be inferred from data.
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

class PPDLToProbDatalogTranslator(PatternWalker):
    '''
    Translate a PPDL program to a ProbDatalog program.

    A PPDL probabilsitic rule whose delta term's distribution is finite can be
    represented as a probabilistic choice. A probabilistic choice can be
    represented within the probabilistic facts formalism (as noted by Vennekens
    et al., 2004). Thus, the rule can also be represented with probabilistic
    facts and standard Datalog rulees.
    '''
    @add_match(Implication, is_gdatalog_rule)
    def rule(self, rule):
        pass


def get_antecedent_predicates(rule):
    antecedent_literals = get_antecedent_formulas(rule)
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
