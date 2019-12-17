from collections import defaultdict
from typing import Mapping, AbstractSet
import itertools

from ..expressions import (
    Expression,
    Constant,
    Symbol,
    FunctionApplication,
    ExpressionBlock,
)
from ..datalog.expressions import Fact, TranslateToLogic
from ..logic import Union, Implication, Conjunction, ExistentialPredicate
from ..logic.expression_processing import extract_logic_predicates
from ..exceptions import NeuroLangException
from ..datalog import DatalogProgram
from ..expression_pattern_matching import add_match
from ..expression_walker import (
    PatternWalker,
    ExpressionWalker,
    ExpressionBasicEvaluator,
)
from .ppdl import is_gdatalog_rule
from ..datalog.expression_processing import (
    is_ground_predicate,
    conjunct_if_needed,
    conjunct_formulas,
)
from .ppdl import concatenate_to_expression_block, get_dterm, DeltaTerm
from ..datalog.chase import (
    ChaseNamedRelationalAlgebraMixin,
    ChaseGeneral,
    ChaseNaive,
)
from .expressions import (
    ProbabilisticPredicate,
    Grounding,
    PfactGrounding,
    make_numerical_col_symb,
)
from ..utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ..relational_algebra import ColumnStr, RelationalAlgebraSolver, Projection


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


class Datalog(TranslateToLogic, DatalogProgram, ExpressionBasicEvaluator):
    pass


class Chase(ChaseNaive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
    pass


def get_rule_pfact_pred_symbs(rule, pfact_pred_symbs):
    return set(
        p.functor
        for p in extract_logic_predicates(rule.antecedent)
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


def _group_probfacts_by_pred_symb(code_block):
    probfacts = defaultdict(list)
    non_probfacts = list()
    for expression in code_block.expressions:
        if is_probabilistic_fact(expression):
            probfacts[expression.consequent.body.functor].append(expression)
        else:
            non_probfacts.append(expression)
    return probfacts, non_probfacts


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


def _const_or_symb_as_python_type(exp):
    if isinstance(exp, Constant):
        return exp.value
    else:
        return exp.name


def _build_pfact_set(pred_symb, pfacts):
    some_pfact = next(iter(pfacts))
    cols = [ColumnStr(Symbol.fresh().name)] + [
        ColumnStr(arg.name) if isinstance(arg, Symbol) else Symbol.fresh().name
        for arg in some_pfact.consequent.body.args
    ]
    iterable = [
        (_const_or_symb_as_python_type(pf.consequent.probability),)
        + tuple(
            _const_or_symb_as_python_type(arg)
            for arg in pf.consequent.body.args
        )
        for pf in pfacts
    ]
    return Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(cols, iterable)
    )


class ProbDatalogProgram(DatalogProgram, ExpressionWalker):
    """
    Datalog extended with probabilistic facts semantics from ProbLog.

    It adds a probabilistic database which is a set of probabilistic facts.

    Probabilistic facts are stored in the symbol table of the program such that
    the key in the symbol table is the symbol of the predicate of the
    probabilsitic fact and the value is the probabilistic fact itself.
    """

    pfact_pred_symbs_symb = Symbol("__probabilistic_predicate_symbols__")
    typing_symbol = Symbol("__pfacts_typing__")

    @property
    def pfact_pred_symbs(self):
        return self.symbol_table.get(
            self.pfact_pred_symbs_symb, Constant[AbstractSet](set())
        ).value

    def get_pfact_typing(self, pfact_pred_symb):
        return self.symbol_table.get(
            self.typing_symbol, Constant[Mapping](dict())
        ).value.get(pfact_pred_symb, Constant[Mapping](dict()))

    @add_match(ExpressionBlock)
    def program_code(self, code):
        probfacts, other_expressions = _group_probfacts_by_pred_symb(code)
        for pred_symb, pfacts in probfacts.items():
            self._register_probfact_pred_symb(pred_symb)
            if pred_symb in self.symbol_table:
                raise NeuroLangException(
                    "Probabilistic fact predicate symbol already seen"
                )
            if len(pfacts) > 1:
                self.symbol_table[pred_symb] = _build_pfact_set(
                    pred_symb, pfacts
                )
            else:
                self.walk(list(pfacts)[0])
        super().process_expression(ExpressionBlock(other_expressions))
        self._check_all_probfacts_variables_have_been_typed()

    def _register_probfact_pred_symb(self, pred_symb):
        self.protected_keywords.add(self.pfact_pred_symbs_symb.name)
        self.symbol_table[self.pfact_pred_symbs_symb] = Constant[AbstractSet](
            self.pfact_pred_symbs | {pred_symb}
        )

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
        for pred_symb in get_rule_pfact_pred_symbs(
            expression, self.pfact_pred_symbs
        ):
            typing = _infer_pfact_typing_pred_symbs(pred_symb, expression)
            self._update_pfact_typing(pred_symb, typing)
        return super().statement_intensional(expression)

    def _update_pfact_typing(self, pfact_pred_symb, typing):
        """
        Update typing information for a probabilistic fact's terms.

        Parameters
        ----------
        symbol : Symbol
            Probabilistic fact's predicate symbol.
        typing : Mapping[int, AbstractSet[Symbol]]
            New typing information that will be integrated.

        """
        if self.typing_symbol not in self.symbol_table:
            self.symbol_table[self.typing_symbol] = Constant[Mapping](dict())
        prev_typing = self.get_pfact_typing(pfact_pred_symb)
        new_pfact_typing = _combine_typings(prev_typing, typing)
        new_typing = Constant[Mapping](
            {
                pred_symb: (
                    self.symbol_table[self.typing_symbol].value[pred_symb]
                    if pred_symb != pfact_pred_symb
                    else new_pfact_typing
                )
                for pred_symb in (
                    set(self.symbol_table[self.typing_symbol].value)
                    | {pfact_pred_symb}
                )
            }
        )
        self.symbol_table[self.typing_symbol] = new_typing

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
        for pfact_pred_symb, pfacts in self.probabilistic_facts().items():
            if isinstance(pfacts, Constant[AbstractSet]):
                continue
            typing = self.get_pfact_typing(pfact_pred_symb)
            if any(
                not (
                    var_idx in typing.value
                    and len(typing.value[var_idx].value) == 1
                )
                for var_idx in _get_pfact_var_idxs(pfacts.expressions[0])
            ):
                raise NeuroLangException(
                    f"Types of variables of probabilistic facts with "
                    f"predicate symbol {pfact_pred_symb} could not be "
                    f"inferred from the program"
                )

    def probabilistic_facts(self):
        """Return probabilistic facts of the symbol table."""
        return {
            k: v
            for k, v in self.symbol_table.items()
            if k in self.pfact_pred_symbs
        }

    def extensional_database(self, exclude_predicates=None):
        return super().extensional_database(self.pfact_pred_symbs)


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
        not typing.value[i].value & local_typing.value[i].value
        for i in local_typing.value
        if i in typing.value
    ):
        raise NeuroLangException(
            "Inconsistent typing of probabilistic fact variables"
        )


def _combine_typings(typing_a, typing_b):
    """
    Combine two typings of a probabilistic fact's terms.

    Parameters
    ----------
    typing_a : Dict[int, AbstractSet[Symbol]]
        First typing.
    typing_b : Dict[int, AbstractSet[Symbol]]
        Second typing.

    Returns
    -------
    Dict[int, AbstractSet[Symbol]]
        Resulting combined typing.

    """
    return Constant[Mapping](
        {
            idx: Constant[AbstractSet](
                (
                    typing_a.value[idx].value
                    if idx in typing_a.value
                    else typing_b.value[idx].value
                )
                & (
                    typing_b.value[idx].value
                    if idx in typing_b.value
                    else typing_a.value[idx].value
                )
            )
            for idx in set(typing_a.value) | set(typing_b.value)
        }
    )


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
    typing : Mapping[int, AbstractSet[Symbol]]
        Mapping from term indices in the probabilistic fact's literal to the
        typing predicate symbol candidates found in the rule.

    """
    antecedent_atoms = extract_logic_predicates(rule.antecedent)
    rule_pfact_atoms = [
        atom for atom in antecedent_atoms if atom.functor == pfact_pred_symb
    ]
    if not rule_pfact_atoms:
        raise NeuroLangException(
            "Expected rule with atom whose predicate symbol is the "
            "probabilistic fact's predicate symbol"
        )
    typing = Constant[Mapping](dict())
    for rule_pfact_atom in rule_pfact_atoms:
        idx_to_var = {
            i: arg
            for i, arg in enumerate(rule_pfact_atom.args)
            if isinstance(arg, Symbol)
        }
        local_typing = Constant[Mapping](
            {
                Constant[int](i): Constant[AbstractSet](
                    {
                        atom.functor
                        for atom in antecedent_atoms
                        if atom.args == (var,)
                        and atom.functor != pfact_pred_symb
                    }
                )
                for i, var in idx_to_var.items()
            }
        )
        _check_typing_consistency(typing, local_typing)
        typing = _combine_typings(typing, local_typing)
    return typing


def probdatalog_to_datalog(pd_program):
    new_symbol_table = dict()
    for pred_symb in pd_program.symbol_table:
        value = pd_program.symbol_table[pred_symb]
        if pred_symb in pd_program.pfact_pred_symbs:
            if isinstance(value, Constant[AbstractSet]):
                new_symbol_table[pred_symb] = RelationalAlgebraSolver().walk(
                    Projection(value, value.value.columns[1:])
                )
            elif isinstance(value, ExpressionBlock):
                new_symbol_table[pred_symb] = ExpressionBlock(
                    [
                        _construct_pfact_intensional_rule(
                            pfact,
                            pfact.consequent.body,
                            pd_program.get_pfact_typing(pred_symb),
                        )
                        for pfact in value.expressions
                    ]
                )
            else:
                raise NeuroLangException("Unhandled pfacts type")
        else:
            new_symbol_table[pred_symb] = value
    return Datalog(new_symbol_table)


def _construct_pfact_intensional_rule(pfact, pfact_pred, typing):
    """
    Construct an intensional rule from a probabilistic fact that can later
    be used to obtain the possible groundings of the probabilistic fact.

    Let `p :: P(x_1, ..., x_n)` be a probabilistic fact and let `T_1, ...,
    T_n` be the relations (predicate symbols) that type the variables `x_1,
    ..., x_n` (respectively). This method will return the intensional rule
    `P(x_1, ..., x_n) :- T_1(x_1), ..., T_n(x_n)`.

    """
    pfact_pred_symb = pfact_pred.functor
    antecedent = conjunct_if_needed(
        [
            next(iter(typing.value[Constant[int](var_idx)].value))(var_symb)
            for var_idx, var_symb in enumerate(pfact_pred.args)
            if isinstance(var_symb, Symbol)
        ]
    )
    return Implication(pfact_pred, antecedent)


def _construct_fact_variable_predicate(fact):
    new_args = (Symbol.fresh() for arg in fact.consequent.args)
    return fact.consequent.functor(*new_args)


def _construct_pfact_variable_predicate(pfact):
    new_args = tuple(Symbol.fresh() for arg in pfact.consequent.body.args)
    return ProbabilisticPredicate(
        Symbol.fresh(), pfact.consequent.body.functor(*new_args)
    )


def _split_and_group_grounded_pfacts(block):
    grouped_ground_pfacts = defaultdict(list)
    other_expressions = list()
    for exp in block.expressions:
        if is_probabilistic_fact(exp) and is_ground_predicate(
            exp.consequent.body
        ):
            pred_symb = exp.consequent.body.functor
            grouped_ground_pfacts[pred_symb].append(exp)
        else:
            other_expressions.append(exp)
    return grouped_ground_pfacts, other_expressions


class ProbDatalogGrounder(PatternWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table
        self.walked_extensional_pred_symbs = set()

    @add_match(ExpressionBlock)
    def expression_block(self, block):
        grouped_gpfacts, other_exps = _split_and_group_grounded_pfacts(block)
        return ExpressionBlock(
            list(
                self._construct_ground_pfacts_grounding(group)
                for group in grouped_gpfacts.values()
            )
            + list(
                itertools.chain(
                    *[
                        [self.walk(exp)]
                        if not isinstance(exp, Fact)
                        else (
                            []
                            if exp.consequent.functor
                            in self.walked_extensional_pred_symbs
                            else [self._construct_fact_grounding(exp)]
                        )
                        for exp in other_exps
                    ]
                )
            )
        )

    @add_match(Implication, is_probabilistic_fact)
    def probabilistic_fact(self, pfact):
        return self._construct_grounding(pfact, pfact.consequent.body)

    @add_match(Implication, is_existential_probabilistic_fact)
    def existential_probabilistic_fact(self, existential_pfact):
        return self._construct_grounding(
            existential_pfact, existential_pfact.consequent.body.body
        )

    @add_match(
        Implication(FunctionApplication[bool](Symbol, ...), Expression),
        lambda exp: exp.antecedent != Constant[bool](True),
    )
    def statement_intensional(self, rule):
        return self._construct_grounding(rule, rule.consequent)

    def _construct_ground_pfacts_grounding(self, pfacts):
        new_pred = _construct_pfact_variable_predicate(next(iter(pfacts)))
        iterable = set(
            tuple(arg.value for arg in pfact.consequent.body.args)
            for pfact in pfacts
        )
        params_iterable = set(
            tuple(arg.value for arg in pfact.consequent.body.args)
            + (
                pfact.consequent.probability.value
                if isinstance(pfact.consequent.probability, Constant)
                else pfact.consequent.probability.name,
            )
            for pfact in pfacts
        )
        relation = Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                tuple(c.name for c in new_pred.body.args), iterable
            )
        )
        params_relation = Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                tuple(c.name for c in new_pred.body.args)
                + (new_pred.probability.name,),
                params_iterable,
            )
        )
        new_pfact = Implication(new_pred, Constant[bool](True))
        return PfactGrounding(new_pfact, relation, params_relation)

    def _construct_fact_grounding(self, fact):
        if fact.consequent.functor not in self.walked_extensional_pred_symbs:
            self.walked_extensional_pred_symbs.add(fact.consequent.functor)
            new_pred = _construct_fact_variable_predicate(fact)
            return self._construct_grounding(new_pred, new_pred)

    def _construct_grounding(self, expression, predicate):
        return Grounding(
            expression,
            Constant[AbstractSet](
                NamedRelationalAlgebraFrozenSet(
                    iterable=self.symbol_table[predicate.functor].value,
                    columns=[
                        arg.name
                        if isinstance(arg, Symbol)
                        else Symbol.fresh().name
                        for arg in predicate.args
                    ],
                )
            ),
        )


def ground_probdatalog_program(pd_code):
    """
    Ground a Prob(Data)Log program.

    This is a 3 steps process:
    (1) Create a Datalog program based on the Prob(Data)Log program, such that
        each probabilistic fact `p :: P(x_1, ..., x_n)` is converted into a
        rule `P(x_1, ..., x_n) :- T_1(x_1), ..., T_n(x_n)` where `T_i` is the
        typing predicate symbol (relation) for the variable `x_i`. Note that if
        `x_i` is a constant there is no variable to type and thus no
        `T_i(x_i)` in the antecedent of the newly created rule.
    (2) Solve the Datalog program obtained from (1), thereby obtaining all
        inferrable intensional ground facts.
    (3) Use the intensional ground facts obtained from (2) and the initial
        Prob(Data)Log program to obtain the grounding of all the rules in the
        program. We then have a grounded Prob(Data)Log program.

    """
    pd_program = ProbDatalogProgram()
    pd_program.walk(pd_code)
    for disjunction in pd_program.intensional_database().values():
        if len(disjunction.formulas) > 1:
            raise NeuroLangException(
                "Programs with several rules with the same head predicate "
                "symbol are not currently supported"
            )
    dl_program = probdatalog_to_datalog(pd_program)
    chase = Chase(dl_program)
    dl_instance = chase.build_chase_solution()
    grounder = ProbDatalogGrounder(symbol_table=dl_instance)
    grounding = grounder.walk(pd_code)
    return grounding
