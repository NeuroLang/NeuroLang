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
from ..logic.expression_processing import (
    extract_logic_predicates,
    extract_logic_free_variables
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

    pfact_pred_symbs_symb = Symbol("__probabilistic_predicate_symbols__")
    typing_symbol = Symbol("__pfacts_typing__")

    @property
    def predicate_symbols(self):
        return (
            set(self.intensional_database())
            | set(self.extensional_database())
            | set(self.pfact_pred_symbs)
        )

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

    def extensional_database(self):
        exclude = self.protected_keywords | self.pfact_pred_symbs
        ret = self.symbol_table.symbols_by_type(AbstractSet)
        for keyword in exclude:
            if keyword in ret:
                del ret[keyword]
        return ret


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


def probdatalog_to_datalog(pd_program):
    new_symbol_table = TypedSymbolTable()
    for pred_symb in pd_program.symbol_table:
        value = pd_program.symbol_table[pred_symb]
        if pred_symb in pd_program.pfact_pred_symbs:
            if isinstance(value, Constant[AbstractSet]):
                columns = tuple(
                    Constant[ColumnInt](ColumnInt(i))
                    for i in list(range(value.value.arity))[1:]
                )
                new_symbol_table[pred_symb] = RelationalAlgebraSolver().walk(
                    Projection(value, columns)
                )
            else:
                new_symbol_table[pred_symb] = Union(
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
        probfact_groundings + extensional_groundings + intensional_groundings
    )


def ground_probdatalog_program(pd_code):
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
    return build_grounding(pd_program, dl_instance)
