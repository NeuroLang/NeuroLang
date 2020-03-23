from collections import defaultdict
from typing import Mapping, AbstractSet, Tuple
import itertools

import numpy as np

from ..expressions import (
    Expression,
    Constant,
    Symbol,
    FunctionApplication,
    ExpressionBlock,
)
from ..datalog.expressions import Fact, TranslateToLogic
from ..logic import Union, Implication, Conjunction, ExistentialPredicate
from ..logic.expression_processing import (
    extract_logic_predicates,
    extract_logic_free_variables
)
from ..exceptions import NeuroLangException
from ..type_system import Unknown
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
from ..datalog.wrapped_collections import WrappedRelationalAlgebraSet
from .expressions import (
    ProbabilisticPredicate,
    ProbabilisticChoice,
    Grounding,
    make_numerical_col_symb,
)
from ..utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ..relational_algebra import (
    ColumnStr,
    ColumnInt,
    RelationalAlgebraSolver,
    Projection,
)
from ..typed_symbol_table import TypedSymbolTable
from ..logic import Union

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

def is_existential_predicate(expression):
    free_vars = extract_logic_free_variables(expression)
    if len(free_vars) > 0:
        return True
    return False


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

def _rewrite_existential_predicate(expression):
    ext_free_vars = extract_logic_free_variables(expression)
    new_head = _fill_variables_in_head(ext_free_vars, expression.consequent)
    new_exp = Implication(new_head, expression.antecedent)
    new_rule = Implication(expression.consequent, new_head)

    return new_rule, new_exp

def _fill_variables_in_head(free_vars, exp_head):
    new_functor = Symbol.fresh()
    args = exp_head.args + tuple(free_vars)

    return FunctionApplication(new_functor, args)


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
    return Constant[AbstractSet](WrappedRelationalAlgebraSet(iterable))


class ProbDatalogExistentialTranslator(ExpressionWalker):

    @add_match(ExpressionBlock)
    def expression_block(self, code):
        new_code = tuple()
        for expression in code.expressions:
            new_code += self.walk(expression)

        return ExpressionBlock(new_code)

    @add_match(Implication(Expression, Constant))
    def existential_fact(self, expression):
        return (expression,)

    @add_match(
        Implication,
        lambda exp: is_existential_predicate(exp)
    )
    def existential_predicate(self, expression):
        modified_rule, new_rule = _rewrite_existential_predicate(expression)
        return tuple([modified_rule, new_rule])

    @add_match(Expression)
    def definition(self, expression):
        return (expression,)

class ProbDatalogProgram(DatalogProgram, ExpressionWalker):
    """
    Datalog extended with probabilistic facts semantics from ProbLog.

    It adds a probabilistic database which is a set of probabilistic facts.

    Probabilistic facts are stored in the symbol table of the program such that
    the key in the symbol table is the symbol of the predicate of the
    probabilsitic fact and the value is the probabilistic fact itself.
    """

    pfact_pred_symbs_symb = Symbol("__pfact_pred_symbs__")
    pchoice_pred_symbs_symb = Symbol("__pchoice_pred_symbs__")

    @property
    def predicate_symbols(self):
        return (
            set(self.intensional_database())
            | set(self.extensional_database())
            | set(self.pfact_pred_symbs)
            | set(self.pchoice_pred_symbs)
        )

    @property
    def pfact_pred_symbs(self):
        return self.symbol_table.get(
            self.pfact_pred_symbs_symb, Constant[AbstractSet](set())
        ).value

    @property
    def pchoice_pred_symbs(self):
        return self.symbol_table.get(
            self.pchoice_pred_symbs_symb, Constant[AbstractSet](set())
        ).value

    def extensional_database(self):
        exclude = (
            self.protected_keywords
            | self.pfact_pred_symbs
            | self.pchoice_pred_symbs
        )
        ret = self.symbol_table.symbols_by_type(AbstractSet)
        for keyword in exclude:
            if keyword in ret:
                del ret[keyword]
        return ret

    def add_probfacts_from_tuples(self, symbol, iterable, type_=Unknown):
        self._register_probfact_pred_symb(symbol)
        if type_ is Unknown:
            type_, iterable = self.infer_iterable_type(iterable)
        _check_iterable_prob_type(type_)
        constant = Constant[AbstractSet[type_]](
            self.new_probability_set(list(iterable)),
            auto_infer_type=False,
            verify_type=False,
        )
        symbol = symbol.cast(constant.type)
        self.symbol_table[symbol] = constant

    def add_probchoice_from_tuples(self, symbol, iterable, type_=Unknown):
        """
        Add a probabilistic choice to the symbol table.

        """
        self._register_probchoice_pred_symb(symbol)
        if type_ is Unknown:
            type_, iterable = self.infer_iterable_type(iterable)
        _check_iterable_prob_type(type_)
        if symbol in self.symbol_table:
            raise NeuroLangException("Symbol already used")
        ra_set = Constant[AbstractSet](
            self.new_probability_set(list(iterable)),
            auto_infer_type=False,
            verify_type=False,
        )
        _check_probchoice_probs_sum_to_one(ra_set)
        self.symbol_table[symbol] = ra_set

    @staticmethod
    def new_probability_set(iterable=None):
        return WrappedRelationalAlgebraSet(iterable=iterable)

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

    def _register_probfact_pred_symb(self, pred_symb):
        self.protected_keywords.add(self.pfact_pred_symbs_symb.name)
        self.symbol_table[self.pfact_pred_symbs_symb] = Constant[AbstractSet](
            self.pfact_pred_symbs | {pred_symb}
        )

    def _register_probchoice_pred_symb(self, pred_symb):
        self.protected_keywords.add(self.pchoice_pred_symbs_symb.name)
        self.symbol_table[self.pchoice_pred_symbs_symb] = Constant[
            AbstractSet
        ](self.pchoice_pred_symbs | {pred_symb})

    @add_match(
        Implication,
        lambda exp: is_probabilistic_fact(exp)
        or is_existential_probabilistic_fact(exp),
    )
    def probfact_or_existential_probfact(self, expression):
        if is_existential_probabilistic_fact(expression):
            _check_existential_probfact_validity(expression)
        pred_symb = _extract_probfact_or_eprobfact_pred_symb(expression)
        if pred_symb not in self.symbol_table:
            self.symbol_table[pred_symb] = ExpressionBlock(tuple())
        self.symbol_table[pred_symb] = concatenate_to_expression_block(
            self.symbol_table[pred_symb], [expression]
        )
        return expression

    def probabilistic_facts(self):
        """Return probabilistic facts of the symbol table."""
        return {
            k: v
            for k, v in self.symbol_table.items()
            if k in self.pfact_pred_symbs
        }


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


def probdatalog_to_datalog(pd_program):
    new_symbol_table = TypedSymbolTable()
    for pred_symb in pd_program.symbol_table:
        value = pd_program.symbol_table[pred_symb]
        if (
            pred_symb
            in pd_program.pfact_pred_symbs | pd_program.pchoice_pred_symbs
        ):
            if not isinstance(value, Constant[AbstractSet]):
                raise NeuroLangException(
                    "Expected grounded probabilistic facts"
                )
            columns = tuple(
                Constant[ColumnInt](ColumnInt(i))
                for i in list(range(value.value.arity))[1:]
            )
            new_symbol_table[pred_symb] = RelationalAlgebraSolver().walk(
                Projection(value, columns)
            )
        else:
            new_symbol_table[pred_symb] = value
    return Datalog(new_symbol_table)


def build_extensional_grounding(pred_symb, tuple_set):
    args = tuple(Symbol.fresh() for _ in range(tuple_set.value.arity))
    cols = tuple(arg.name for arg in args)
    return Grounding(
        expression=Implication(pred_symb(*args), Constant[bool](True)),
        relation=Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                columns=cols, iterable=tuple_set.value
            )
        ),
    )


def build_rule_grounding(pred_symb, st_item, tuple_set):
    if isinstance(st_item, Union):
        st_item = st_item.formulas[0]
    elif isinstance(st_item, ExpressionBlock):
        st_item = st_item.expressions[0]
    if isinstance(st_item.consequent, ProbabilisticPredicate):
        pred = st_item.consequent.body
    else:
        pred = st_item.consequent
    cols = tuple(arg.name for arg in pred.args)
    return Grounding(
        expression=st_item,
        relation=Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                columns=cols, iterable=tuple_set.value
            )
        ),
    )


def build_pchoice_grounding(pred_symb, relation):
    args = tuple(Symbol.fresh() for _ in range(relation.value.arity - 1))
    predicate = pred_symb(*args)
    expression = ProbabilisticChoice(predicate)
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=(Symbol.fresh().name,) + tuple(a.name for a in args),
            iterable=relation.value,
        )
    )
    return Grounding(expression=expression, relation=relation)


def build_pfact_grounding_from_set(pred_symb, relation):
    param_symb = Symbol.fresh()
    args = tuple(Symbol.fresh() for _ in range(relation.value.arity - 1))
    expression = Implication(
        ProbabilisticPredicate(param_symb, pred_symb(*args)),
        Constant[bool](True),
    )
    return Grounding(
        expression=expression,
        relation=Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                columns=(param_symb.name,) + tuple(arg.name for arg in args),
                iterable=relation.value,
            )
        ),
    )


def build_grounding(pd_program, dl_instance):
    extensional_groundings = []
    probfact_groundings = []
    probchoice_groundings = []
    intensional_groundings = []
    for pred_symb in pd_program.predicate_symbols:
        st_item = pd_program.symbol_table[pred_symb]
        if pred_symb in pd_program.pfact_pred_symbs:
            if isinstance(st_item, Constant[AbstractSet]):
                probfact_groundings.append(
                    build_pfact_grounding_from_set(pred_symb, st_item)
                )
            else:
                probfact_groundings.append(
                    build_rule_grounding(
                        pred_symb, st_item, dl_instance[pred_symb]
                    )
                )
        elif pred_symb in pd_program.pchoice_pred_symbs:
            probchoice_groundings.append(
                build_pchoice_grounding(pred_symb, st_item)
            )
        else:
            if isinstance(st_item, Constant[AbstractSet]):
                extensional_groundings.append(
                    build_extensional_grounding(pred_symb, st_item)
                )
            else:
                intensional_groundings.append(
                    build_rule_grounding(
                        pred_symb, st_item, dl_instance[pred_symb]
                    )
                )
    return ExpressionBlock(
        probfact_groundings
        + probchoice_groundings
        + extensional_groundings
        + intensional_groundings
    )


def ground_probdatalog_program(
    pd_code, probfact_sets=None, extensional_sets=None, probchoice_sets=None,
):
    pd_program = ProbDatalogProgram()
    pd_program.walk(pd_code)
    if probfact_sets is not None:
        for symb, probabilistic_set in probfact_sets.items():
            pd_program.add_probfacts_from_tuples(symb, probabilistic_set)
    if extensional_sets is not None:
        for symb, extensional_set in extensional_sets.items():
            pd_program.add_extensional_predicate_from_tuples(
                symb, extensional_set
            )
    if probchoice_sets is not None:
        for symb, probchoice_set in probchoice_sets.items():
            pd_program.add_probchoice_from_tuples(symb, probchoice_set)
    for disjunction in pd_program.intensional_database().values():
        if len(disjunction.formulas) > 1:
            raise NeuroLangException(
                "Programs with several rules with the same head predicate "
                "symbol are not currently supported"
            )
    dl_program = probdatalog_to_datalog(pd_program)
    chase = Chase(dl_program)
    dl_instance = chase.build_chase_solution()
    return build_grounding(pd_program, dl_instance)


def _check_iterable_prob_type(iterable_type):
    if not (
        issubclass(iterable_type.__origin__, Tuple)
        and iterable_type.__args__[0] is float
    ):
        raise NeuroLangException(
            "Expected tuples to have a probability as their first element"
        )


def _check_probchoice_probs_sum_to_one(ra_set):
    probs_sum = sum(v.value[0].value for v in ra_set.value)
    if not np.isclose(probs_sum, 1.0):
        raise NeuroLangException(
            "Probabilities of probabilistic choice should sum to 1"
        )
