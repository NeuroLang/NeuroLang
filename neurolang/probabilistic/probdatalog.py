import itertools
from collections import defaultdict
from typing import Mapping, Set

from ..expressions import (
    Definition,
    Expression,
    Constant,
    Symbol,
    FunctionApplication,
    ExpressionBlock,
    ExistentialPredicate,
)
from ..unification import apply_substitution
from ..datalog.expressions import (
    Fact,
    Implication,
    Disjunction,
    Conjunction,
    TranslateToLogic,
)
from ..exceptions import NeuroLangException
from ..datalog import DatalogProgram
from ..expression_pattern_matching import add_match
from ..expression_walker import (
    PatternWalker,
    ExpressionWalker,
    ExpressionBasicEvaluator,
)
from .ppdl import is_gdatalog_rule
from ..datalog.instance import SetInstance, FrozenMapInstance
from ..datalog.expression_processing import (
    extract_datalog_predicates,
    is_ground_predicate,
    implication_has_existential_variable_in_antecedent,
    conjunct_if_needed,
    conjunct_formulas,
)
from .ppdl import concatenate_to_expression_block, get_dterm, DeltaTerm
from ..datalog.chase import (
    ChaseNamedRelationalAlgebraMixin,
    ChaseGeneral,
    ChaseNaive,
)
from .expressions import ProbabilisticPredicate
from ..utils.relational_algebra_set import (
    RelationalAlgebraFrozenSet,
    NamedRelationalAlgebraFrozenSet,
)
from ..relational_algebra import NameColumns


def is_probabilistic_fact(expression):
    return (
        isinstance(expression, Implication)
        and isinstance(expression.consequent, ProbabilisticPredicate)
        and isinstance(expression.consequent.body, FunctionApplication)
        and expression.antecedent == Constant[bool](True)
    )


def is_existential_probabilistic_fact(expression):
    return (
        isinstance(expression, Implication)
        and isinstance(expression.consequent, ExistentialPredicate)
        and isinstance(expression.consequent.body, ProbabilisticPredicate)
        and isinstance(expression.consequent.body.body, FunctionApplication)
        and expression.antecedent == Constant[bool](True)
    )


def _extract_probfact_probability(expression):
    if is_probabilistic_fact(expression):
        return expression.consequent.probability
    elif is_existential_probabilistic_fact(expression):
        return expression.consequent.body.probability
    else:
        raise NeuroLangException("Invalid probabilistic fact")


def get_rule_pfact_pred_symbs(rule, pfact_pred_symbs):
    return set(
        p.functor
        for p in extract_datalog_predicates(rule.antecedent)
        if p.functor in pfact_pred_symbs
    )


def _put_probfacts_in_front(code_block):
    probfacts = []
    non_probfacts = []
    for expression in code_block.expressions:
        if is_probabilistic_fact(
            expression
        ) or is_existential_probabilistic_fact(expression):
            probfacts.append(expression)
        else:
            non_probfacts.append(expression)
    return ExpressionBlock(probfacts + non_probfacts)


def _check_existential_probfact_validity(expression):
    qvar = expression.consequent.head
    if qvar in expression.consequent.body.body._symbols:
        raise NeuroLangException(
            "Existentially quantified variable can only be used as the "
            "probability of the probability fact"
        )


def _extract_probfact_or_eprobfact_pred_symb(expression):
    if is_existential_probabilistic_fact(expression):
        return expression.consequent.body.body.functor
    else:
        return expression.consequent.body.functor


def _get_pfact_var_idxs(pfact):
    if is_probabilistic_fact(pfact):
        atom = pfact.consequent.body
    else:
        atom = pfact.consequent.body.body
    return {i for i, arg in enumerate(atom.args) if isinstance(arg, Symbol)}


class ProbDatalogProgram(DatalogProgram, ExpressionWalker):
    """
    Datalog extended with probabilistic facts semantics from ProbLog.

    It adds a probabilistic database which is a set of probabilistic facts.

    Probabilistic facts are stored in the symbol table of the program such that
    the key in the symbol table is the symbol of the predicate of the
    probabilsitic fact and the value is the probabilistic fact itself.
    """

    typing_symbol = Symbol("__pfacts_typing__")

    def _check_all_probfacts_variables_have_been_typed(self):
        """
        Check that the type of all the variables occurring in all the
        probabilistic facts was correctly inferred from the rules of the
        program.

        Several candidate typing predicate symbols can be found during the
        static analysis of the rules of the program. If at the end of the
        static analysis several candidates remain, the type inference failed
        and an exception is raised.

        """
        for pfact_pred_symb, pfact_block in self.probabilistic_facts().items():
            typing = self.symbol_table[self.typing_symbol].value[
                pfact_pred_symb
            ]
            if any(
                not (var_idx in typing and len(typing[var_idx]) == 1)
                for var_idx in _get_pfact_var_idxs(pfact_block.expressions[0])
            ):
                raise NeuroLangException(
                    f"Types of variables of probabilistic facts with "
                    f"predicate symbol {pfact_pred_symb} could not be "
                    f"inferred from the program"
                )

    @add_match(ExpressionBlock)
    def program_code(self, code):
        super().process_expression(_put_probfacts_in_front(code))
        self._check_all_probfacts_variables_have_been_typed()

    @add_match(
        Implication,
        lambda exp: is_probabilistic_fact(exp)
        or is_existential_probabilistic_fact(exp),
    )
    def probfact_or_existential_probfact(self, expression):
        if is_existential_probabilistic_fact(expression):
            _check_existential_probfact_validity(expression)
        self.protected_keywords.add(self.typing_symbol.name)
        pred_symb = _extract_probfact_or_eprobfact_pred_symb(expression)
        if pred_symb not in self.symbol_table:
            self.symbol_table[pred_symb] = ExpressionBlock(tuple())
        self.symbol_table[pred_symb] = concatenate_to_expression_block(
            self.symbol_table[pred_symb], [expression]
        )
        return expression

    def _update_pfact_typing(self, symbol, typing):
        """
        Update typing information for a probabilistic fact's terms.

        Parameters
        ----------
        symbol : Symbol
            Probabilistic fact's predicate symbol.
        typing : Mapping[int, Set[Symbol]]
            New typing information that will be integrated.

        """
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
        lambda exp: exp.antecedent != Constant[bool](True),
    )
    def statement_intensional(self, expression):
        """
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

        """
        if implication_has_existential_variable_in_antecedent(expression):
            raise NeuroLangException(
                "Existentially quantified variables are "
                "forbidden in Prob(Data)log"
            )
        pfact_pred_symbs = set(self.probabilistic_facts())
        for pred_symb in get_rule_pfact_pred_symbs(
            expression, pfact_pred_symbs
        ):
            typing = _infer_pfact_typing_pred_symbs(pred_symb, expression)
            self._update_pfact_typing(pred_symb, typing)
        return super().statement_intensional(expression)

    def probabilistic_facts(self):
        """Return probabilistic facts of the symbol table."""
        return {
            k: v
            for k, v in self.symbol_table.items()
            if isinstance(v, ExpressionBlock)
            and any(
                is_probabilistic_fact(exp)
                or is_existential_probabilistic_fact(exp)
                for exp in v.expressions
            )
        }

    def probabilistic_rules(self):
        """
        Rules in the program with at least one atom in their antecedent whose
        predicate is defined via a probabilistic fact.
        """
        pfact_pred_symbs = set(self.probabilistic_facts().keys())
        prob_rules = defaultdict(set)
        for rule_disjunction in self.intensional_database().values():
            for rule in rule_disjunction.formulas:
                prob_rules.update(
                    {
                        symbol: prob_rules[symbol] | {rule}
                        for symbol in get_rule_pfact_pred_symbs(
                            rule, pfact_pred_symbs
                        )
                    }
                )
        return prob_rules

    def parametric_probfacts(self):
        """
        Probabilistic facts in the program whose probabilities are parameters.
        """
        result = dict()
        for block in self.probabilistic_facts().values():
            result.update(
                {
                    probfact.consequent.probability: probfact
                    for probfact in block.expressions
                    if isinstance(probfact.consequent.probability, Symbol)
                }
            )
        return result


class GDatalogToProbDatalogTranslator(PatternWalker):
    """
    Translate a GDatalog program to a ProbDatalog program.

    A GDatalog probabilsitic rule whose delta term's distribution is finite can
    be represented as a probabilistic choice. If the distribution is a
    bernoulli distribution, it can be represented as probabilistic fact.
    """

    @add_match(Implication, is_gdatalog_rule)
    def rule(self, rule):
        """
        Translate a GDatalog rule whose delta term is bernoulli distributed to
        an expression block containing a probabilistic fact and a
        (deterministic) rule.
        """
        datom = rule.consequent
        dterm = get_dterm(datom)
        predicate = datom.functor
        if not dterm.functor.name == "bernoulli":
            raise NeuroLangException(
                "Other distributions than bernoulli are not supported"
            )
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


class GDatalogToProbDatalog(
    GDatalogToProbDatalogTranslator, ProbDatalogProgram
):
    pass


def _check_typing_consistency(typing, local_typing):
    if any(
        not typing[i] & local_typing[i] for i in local_typing if i in typing
    ):
        raise NeuroLangException(
            "Inconsistent typing of probabilistic fact variables"
        )


def _combine_typings(typing_a, typing_b):
    """
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

    """
    new_typing = dict()
    for idx, pred_symbols in typing_a.items():
        new_typing[idx] = pred_symbols
    for idx, pred_symbols in typing_b.items():
        if idx in new_typing:
            new_typing[idx] &= pred_symbols
        else:
            new_typing[idx] = pred_symbols
    return new_typing


def _infer_pfact_typing_pred_symbs(pfact_pred_symb, rule):
    """
    Infer a probabilistic fact's typing from a rule whose antecedent contains
    the probabilistic fact's predicate symbol.

    There can be several typing predicate symbol candidates in the rule. For
    example, let `Q(x) :- A(x), Pfact(x), B(x)` be a rule where `Pfact` is the
    probabilistic fact's predicate symbol. Both `A` and `B` can be the typing
    predicate symbols for the variable `x` occurring in `Pfact(x)`. The output
    will thus be `{0: {A, B}}`, `0` being the index of `x` in `Pfact(x)`.

    Parameters
    ----------
    pfact_pred_symb : Symbol
        Predicate symbol of the probabilistic fact.
    rule : Implication
        Rule that contains an atom with the probabilistic fact's predicate
        symbol in its antecedent.

    Returns
    -------
    typing : Mapping[int, Set[Symbol]]
        Mapping from term indices in the probabilistic fact's literal to the
        typing predicate symbol candidates found in the rule.

    """
    antecedent_atoms = extract_datalog_predicates(rule.antecedent)
    rule_pfact_atoms = [
        atom for atom in antecedent_atoms if atom.functor == pfact_pred_symb
    ]
    if not rule_pfact_atoms:
        raise NeuroLangException(
            "Expected rule with atom whose predicate symbol is the "
            "probabilistic fact's predicate symbol"
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
                if atom.args == (var,) and atom.functor != pfact_pred_symb
            }
            for i, var in idx_to_var.items()
        }
        _check_typing_consistency(typing, local_typing)
        typing = _combine_typings(typing, local_typing)
    return typing


def get_possible_ground_substitutions(probfact, typing, interpretation):
    """
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

    """
    pfact_args = probfact.consequent.body.args
    facts_per_variable = {
        pfact_args[var_idx]: set(
            tupl.value[0]
            for tupl in interpretation.as_map()[
                next(iter(typing_pred_symbs))
            ].value
        )
        for var_idx, typing_pred_symbs in typing.items()
        if isinstance(pfact_args[var_idx], Symbol)
    }
    return frozenset(
        {
            frozenset(zip(facts_per_variable, values))
            for values in itertools.product(*facts_per_variable.values())
        }
    )


def _count_ground_instances_in_interpretation(
    pfact, substitutions, interpretation
):
    return sum(
        apply_substitution(pfact.consequent.body, dict(substitution))
        in interpretation
        for substitution in substitutions
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
    """
    Estimate parametric probabilities of the probabilistic facts in a given
    ProbDatalog program using the given fully-observable interpretations.

    This computation relies on a the domain of each variable occurring in the
    probabilistic facts to be defined by each interpretation using unary
    predicates, as explained in [1]_.

    .. [1] Gutmann et al., "Learning the Parameters of Probabilistic Logic
       Programs", section 3.1.

    """
    estimations = dict()
    parametric_probfacts = prog.parametric_probfacts()
    for parameter, probfact in parametric_probfacts.items():
        pfact_pred_symb = _extract_probfact_or_eprobfact_pred_symb(probfact)
        typing = prog.symbol_table[prog.typing_symbol].value[pfact_pred_symb]
        estimations[parameter] = _probfact_parameter_estimation(
            probfact, typing, interpretations
        )
    return estimations


class RuleGrounding(Definition):
    def __init__(self, rule, algebra_set):
        self.rule = rule
        self.algebra_set = algebra_set


class RemoveProbabilitiesWalker(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table
        self.typing_symbol = ProbDatalogProgram.typing_symbol

    @add_match(
        Implication,
        lambda exp: is_probabilistic_fact(exp)
        and is_ground_predicate(exp.consequent.body),
    )
    def ground_probabilistic_fact(self, pfact):
        return Fact(pfact.consequent.body)

    @add_match(
        Implication,
        lambda exp: is_existential_probabilistic_fact(exp)
        and is_ground_predicate(exp.consequent.body.body),
    )
    def ground_existential_probabilistic_fact(self, existential_pfact):
        return Fact(existential_pfact.consequent.body.body)

    @add_match(Implication, is_probabilistic_fact)
    def notground_probabilist_fact(self, pfact):
        return self._construct_pfact_intensional_rule(
            pfact, pfact.consequent.body
        )

    @add_match(Implication, is_existential_probabilistic_fact)
    def notground_existential_probabilist_fact(self, existential_pfact):
        return self._construct_pfact_intensional_rule(
            existential_pfact, existential_pfact.consequent.body.body
        )

    def _construct_pfact_intensional_rule(self, pfact, pfact_pred):
        """
        Construct an intensional rule from a probabilistic fact that can later
        be used to obtain the possible groundings of the probabilistic fact.

        Let `p :: P(x_1, ..., x_n)` be a probabilistic fact and let `T_1, ...,
        T_n` be the relations (predicate symbols) that type the variables `x_1,
        ..., x_n` (respectively). This method will return the intensional rule
        `P(x_1, ..., x_n) :- T_1(x_1), ..., T_n(x_n)`.

        """
        pfact_pred_symb = pfact_pred.functor
        typing = self.symbol_table[self.typing_symbol].value[pfact_pred_symb]
        antecedent = conjunct_if_needed(
            [
                typing.value[Constant[int](var_idx)](var_symb)
                for var_idx, var_symb in enumerate(pfact_pred.args)
                if isinstance(var_symb, Symbol)
            ]
        )
        return Implication(pfact_pred, antecedent)


class GroundProbDatalogProgram(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table


class Datalog(TranslateToLogic, DatalogProgram, ExpressionBasicEvaluator):
    pass


class Chase(ChaseNaive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
    pass
