import uuid
import itertools
from collections import defaultdict
from typing import Mapping, Set

from ..expressions import (
    Expression, Constant, Symbol, FunctionApplication, ExpressionBlock
)
from ..unification import apply_substitution
from ..datalog.expressions import (
    Fact, Implication, Disjunction, Conjunction, TranslateToLogic
)
from ..exceptions import NeuroLangException
from ..datalog import DatalogProgram
from ..expression_pattern_matching import add_match
from ..expression_walker import (
    PatternWalker, ExpressionWalker, ExpressionBasicEvaluator
)
from ..probabilistic.ppdl import is_gdatalog_rule
from ..datalog.instance import SetInstance
from ..datalog.expression_processing import (
    extract_datalog_predicates,
    implication_has_existential_variable_in_antecedent
)
from .ppdl import concatenate_to_expression_block, get_dterm, DeltaTerm
from ..datalog.chase import (
    ChaseNamedRelationalAlgebraMixin, ChaseGeneral, ChaseNaive
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

    @property
    def probfacts(self):
        return self.consequent.formulas


def get_pfact_pred_symbols(rule, pfact_pred_symbs):
    return set(
        p.functor
        for p in extract_datalog_predicates(rule.antecedent)
        if p.functor in pfact_pred_symbs
    )


def _put_probfacts_in_front(code_block):
    probfacts = []
    non_probfacts = []
    for expression in code_block.expressions:
        if isinstance(expression, ProbFact):
            probfacts.append(expression)
        else:
            non_probfacts.append(expression)
    return ExpressionBlock(probfacts + non_probfacts)


class ProbDatalogProgram(DatalogProgram):
    '''
    Datalog extended with probabilistic facts semantics from ProbLog.

    It adds a probabilistic database which is a set of probabilistic facts.

    Probabilistic facts are stored in the symbol table of the program such that
    the key in the symbol table is the symbol of the predicate of the
    probabilsitic fact and the value is the probabilistic fact itself.
    '''

    typing_symbol = \
        Symbol[Mapping[Symbol, Mapping[int, Set[Symbol]]]]('__pfacts_typing__')

    @add_match(ExpressionBlock)
    def program_code(self, code):
        # TODO: this relies on the class inheriting from ExpressionWalker
        super().process_expression(_put_probfacts_in_front(code))

    @add_match(ProbFact)
    def probabilistic_fact(self, probfact):
        self.protected_keywords.add(self.typing_symbol.name)
        predicate = probfact.consequent.functor
        if predicate not in self.symbol_table:
            self.symbol_table[predicate] = ExpressionBlock(tuple())
        self.symbol_table[predicate] = concatenate_to_expression_block(
            self.symbol_table[predicate], [probfact]
        )
        return probfact

    def _update_pfact_typing(self, symbol, typing):
        '''
        Update typing information for a probabilistic fact's terms.

        Parameters
        ----------
        symbol : Symbol
            Probabilistic fact's predicate symbol.
        typing : Mapping[int, Set[Symbol]]
            New typing information that will be integrated.

        '''
        if self.typing_symbol not in self.symbol_table:
            self.symbol_table[self.typing_symbol] = Constant(dict())
        if symbol not in self.symbol_table[self.typing_symbol].value:
            self.symbol_table[self.typing_symbol].value[symbol] = dict()
        prev_typing = self.symbol_table[self.typing_symbol].value[symbol]
        _check_typing_consistency(prev_typing, typing)
        new_typing = _combine_typings(prev_typing, typing)
        self.symbol_table[self.typing_symbol].value[symbol] = new_typing

    @add_match(
        Implication(FunctionApplication[bool](Symbol, ...), Expression),
        lambda exp: exp.antecedent != Constant[bool](True)
    )
    def statement_intensional(self, expression):
        '''
        Ensure that the typing of the probabilistic facts in the given rule
        stays consistent with the typing from previously seen rules.

        Raises
        ------
        NeuroLangException
            If the implication's antecedent has an existentially quantified
            variable. See [1]_ for the definition of the syntax of CP-Logic
            (the syntax of Prob(Data)Log can be viewed as a subset of the
            syntax of CP-Logic).

        .. [1] Vennekens, "Algebraic and logical study of constructive
        processes in knowledge representation", section 5.2.1 Syntax.

        '''
        if implication_has_existential_variable_in_antecedent(expression):
            raise NeuroLangException(
                'Existentially quantified variables are '
                'forbidden in Prob(Data)log'
            )
        pfact_pred_symbols = self.probabilistic_facts()
        rule_pfact_pred_symbols = get_pfact_pred_symbols(
            expression, pfact_pred_symbols
        )
        for pred_symb in rule_pfact_pred_symbols:
            typing = infer_pfact_typing_predicate_symbols(
                pred_symb, expression
            )
            self._update_pfact_typing(pred_symb, typing)
        return super().statement_intensional(expression)

    def probabilistic_facts(self):
        '''Return probabilistic facts of the symbol table.'''
        return {
            k: v
            for k, v in self.symbol_table.items()
            if isinstance(v, ExpressionBlock) and
            any(isinstance(exp, ProbFact) for exp in v.expressions)
        }

    def probabilistic_rules(self):
        '''
        Rules in the program with at least one atom in their antecedent whose
        predicate is defined via a probabilistic fact.
        '''
        pfact_pred_symbs = set(self.probabilistic_facts().keys())
        prob_rules = defaultdict(set)
        for rule_disjunction in self.intensional_database().values():
            for rule in rule_disjunction.formulas:
                prob_rules.update({
                    symbol: prob_rules[symbol] | {rule}
                    for symbol in
                    get_pfact_pred_symbols(rule, pfact_pred_symbs)
                })
        return prob_rules

    def parametric_probfacts(self):
        '''
        Probabilistic facts in the program whose probabilities are parameters.
        '''
        result = dict()
        for block in self.probabilistic_facts().values():
            result.update({
                probfact.probability: probfact
                for probfact in block.expressions
                if isinstance(probfact.probability, Symbol)
            })
        return result


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
        pfact_pred_symb = Symbol(
            'ProbFact_{}_{}'.format(predicate.name, uuid.uuid1())
        )
        terms = tuple(
            arg for arg in datom.args if not isinstance(arg, DeltaTerm)
        )
        probfact_atom = pfact_pred_symb(*terms)
        new_rule = Implication(
            predicate(*terms),
            conjunct_formulas(rule.antecedent, probfact_atom)
        )
        return self.walk(
            ExpressionBlock([ProbFact(probability, probfact_atom), new_rule])
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
    GDatalogToProbDatalogTranslator, ProbDatalogProgram, ExpressionWalker
):
    pass


def _check_typing_consistency(typing, local_typing):
    if any(
        not typing[i] & local_typing[i] for i in local_typing if i in typing
    ):
        raise NeuroLangException(
            'Inconsistent typing of probabilistic fact variables'
        )


def _combine_typings(typing_a, typing_b):
    '''
    Combine two typings of a probabilistic fact's terms.

    Parameters
    ----------
    typing_a : Dict[int, Set[Symbol]]
        First typing.
    typing_b : Dict[int, Set[Symbol]]
        Second typing.

    Returns
    -------
    Dict[int, Set[Symbol]]
        Resulting combined typing.

    '''
    new_typing = dict()
    for idx, pred_symbols in typing_a.items():
        new_typing[idx] = pred_symbols
    for idx, pred_symbols in typing_b.items():
        if idx in new_typing:
            new_typing[idx] &= pred_symbols
        else:
            new_typing[idx] = pred_symbols
    return new_typing


def infer_pfact_typing_predicate_symbols(pfact_pred_symbol, rule):
    '''
    Infer a probabilistic fact's typing from a rule whose antecedent contains
    the probabilistic fact's predicate symbol.

    There can be several typing predicate symbol candidates in the rule. For
    example, let `Q(x) :- A(x), Pfact(x), B(x)` be a rule where `Pfact` is the
    probabilistic fact's predicate symbol. Both `A` and `B` can be the typing
    predicate symbols for the variable `x` occurring in `Pfact(x)`. The output
    will thus be `{0: {A, B}}`, `0` being the index of `x` in `Pfact(x)`.

    Parameters
    ----------
    pfact_pred_symbol : Symbol
        Predicate symbol of the probabilistic fact.
    rule : Implication
        Rule that contains an atom with the probabilistic fact's predicate
        symbol in its antecedent.

    Returns
    -------
    typing : Mapping[int, Set[Symbol]]
        Mapping from term indices in the probabilistic fact's literal to the
        typing predicate symbol candidates found in the rule.

    '''
    antecedent_atoms = extract_datalog_predicates(rule.antecedent)
    rule_pfact_atoms = [
        atom for atom in antecedent_atoms if atom.functor == pfact_pred_symbol
    ]
    if not rule_pfact_atoms:
        raise NeuroLangException(
            'Expected rule with atom whose predicate symbol is the '
            'probabilistic fact\'s predicate symbol'
        )
    typing = dict()
    for rule_pfact_atom in rule_pfact_atoms:
        idx_to_var = {
            i: arg
            for i, arg in enumerate(rule_pfact_atom.args)
            if isinstance(arg, Symbol)
        }
        local_typing = {
            i: {
                atom.functor
                for atom in antecedent_atoms
                if atom.args == (var, ) and atom.functor != pfact_pred_symbol
            }
            for i, var in idx_to_var.items()
        }
        _check_typing_consistency(typing, local_typing)
        typing = _combine_typings(typing, local_typing)
    return typing


def get_possible_ground_substitutions(probfact, typing, interpretation):
    '''
    Get all possible substitutions that ground a given probabilistic fact in a
    given interpretation based on a rule where the predicate of the
    probabilistic fact occurs.

    This works under the following assumptions:
    (1) for each probabilistic fact p :: P(x_1, ..., x_n), there exists at
        least one rule in the program such that an atom P(y_1, ..., y_n) occurs
        in its antecedent;
    (2) the antecedent conjunction of that rule also contains an atom Y_i(y_i)
        for each variable y_i in (y_1, ..., y_n), where Y_i is a unary
        extensional predicate that defines the domain (or the type) of the
        variable y_i
    (3) and if several of those rules are part of the program, the same typing
        is applied in each one of them, such that it does not matter which rule
        is used for inferring the possible ground substitutions for the
        probabilistic fact.

    '''
    pfact_args = probfact.consequent.args
    facts_per_variable = {
        pfact_args[var_idx]: set(
            tupl[0]
            for tupl in interpretation.elements[next(iter(typing_pred_symbs))]
        )
        for var_idx, typing_pred_symbs in typing.items()
        if isinstance(pfact_args[var_idx], Symbol)
    }
    return frozenset({
        frozenset(zip(facts_per_variable, values))
        for values in itertools.product(*facts_per_variable.values())
    })


def _count_ground_instances_in_interpretation(
    pfact, substitutions, interpretation
):
    return sum(
        apply_substitution(pfact.consequent, dict(substitution)) in
        interpretation for substitution in substitutions
    )


def _probfact_parameter_estimation(pfact, typing, interpretations):
    n_ground_instances = 0
    n_possible_substitutions = 0
    for interpretation in interpretations:
        substitutions = get_possible_ground_substitutions(
            pfact, typing, interpretation
        )
        n_possible_substitutions += len(substitutions)
        n_ground_instances += _count_ground_instances_in_interpretation(
            pfact, substitutions, interpretation
        )
    return n_ground_instances / n_possible_substitutions


def full_observability_parameter_estimation(prog, interpretations):
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
    parametric_probfacts = prog.parametric_probfacts()
    for parameter, probfact in parametric_probfacts.items():
        pfact_pred_symbol = probfact.consequent.functor
        typing = prog.symbol_table[prog.typing_symbol].value[pfact_pred_symbol]
        estimations[parameter] = _probfact_parameter_estimation(
            probfact, typing, interpretations
        )
    return estimations


class ProbfactAsFactWalker(ExpressionWalker):
    @add_match(ProbFact)
    def probfact(self, pfact):
        if any(not isinstance(arg, Constant) for arg in pfact.consequent.args):
            raise NeuroLangException(
                'Variables in probabilistic facts are currently unsupported'
            )
        return Fact(pfact.consequent)


class Datalog(TranslateToLogic, DatalogProgram, ExpressionBasicEvaluator):
    pass


class Chase(ChaseNaive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
    pass


def conjunct_if_needed(formulas):
    '''Only conjunct the given formulas if there is more than one.'''
    if len(formulas) == 1:
        return formulas[0]
    else:
        return Conjunction(formulas)


def get_rule_groundings(rule, instance):
    '''
    Find all groundings of a rule based on an instance.

    TODO: speed up with tabular substitutions
    '''
    head_pred_symb = rule.consequent.functor
    if head_pred_symb not in instance.elements:
        return set()
    grounded_rules = set()
    for tupl in instance.elements[head_pred_symb]:
        substitution = {
            arg: tupl[i]
            for i, arg in enumerate(rule.consequent.args)
            if isinstance(arg, Symbol)
        }
        substituted_antecedent_atoms = [
            apply_substitution(atom, substitution)
            for atom in extract_datalog_predicates(rule.antecedent)
        ]
        if any(
            atom not in instance for atom in substituted_antecedent_atoms
        ):
            continue
        grounded_rules.add(
            Implication(
                apply_substitution(rule.consequent, substitution),
                conjunct_if_needed(substituted_antecedent_atoms)
            )
        )
    return grounded_rules


def dict_to_instance(dict_instance):
    '''Convert a `Dict[Symbol, Constant[Set[Tuple]]]` to a `SetInstance`.'''
    return SetInstance({
        pred_symb:
        frozenset({const_tuple.value
                   for const_tuple in const_set.value})
        for pred_symb, const_set in dict_instance.items()
    })


def ground_probdatalog_program(probdatalog_code):
    '''
    Ground a Prob(Data)Log program by considering all its probabilistic facts
    to be true.

    This is a 3 steps process:
        1. convert all ProbFact in the program to Fact without
        probabilities, thereby obtaining a deterministic Datalog program,
        2. solving that program to obtain a Datalog instance containing all
        extensional and intensional facts,
        3. using this instance to find all possible groundings of rules in the
        intensional database.

    '''
    datalog_code = ProbfactAsFactWalker().walk(probdatalog_code)
    dl = Datalog()
    dl.walk(datalog_code)
    chase = Chase(dl)
    solution_instance = dict_to_instance(chase.build_chase_solution())
    grounded_rules = set.union(
        *[set()] + [
            set.union(
                *[
                    get_rule_groundings(rule, solution_instance)
                    for rule in disjunction.formulas
                ]
            )
            for disjunction in dl.intensional_database().values()
        ]
    )
    new_expressions = []
    for exp in probdatalog_code.expressions:
        if isinstance(exp, (Fact, ProbFact)):
            new_expressions.append(exp)
    new_expressions += grounded_rules
    return ExpressionBlock(new_expressions)
