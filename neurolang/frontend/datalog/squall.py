from collections import Counter
from operator import add, eq, mul, ne, neg, pow, sub, truediv
from re import I
from typing import Callable, List, TypeVar

from ...datalog.aggregation import AggregationApplication
from ...exceptions import NeuroLangFrontendException
from ...expression_walker import (
    ChainedWalker,
    ExpressionWalker,
    IdentityWalker,
    PatternWalker,
    ReplaceExpressionWalker,
    add_match,
    expression_iterator
)
from ...expressions import (
    Command,
    Constant,
    Definition,
    Expression,
    FunctionApplication,
    Lambda,
    Projection,
    Query as EQuery,
    Symbol
)
from ...logic import (
    TRUE,
    Conjunction,
    ExistentialPredicate,
    Implication,
    LogicOperator,
    NaryLogicOperator,
    Negation,
    Quantifier,
    Union,
    UniversalPredicate
)
from ...logic.expression_processing import extract_logic_free_variables
from ...logic.transformations import (
    CollapseConjunctionsMixin,
    CollapseDisjunctionsMixin,
    FactorQuantifiersMixin,
    LogicExpressionWalker,
    PushExistentialsDownMixin,
    PushUniversalsDown,
    RemoveTrivialOperationsMixin
)
from ...probabilistic.expressions import (
    Condition,
    ProbabilisticFact,
    ProbabilisticChoice,
    ProbabilisticPredicate
)
from ...type_system import get_args
from .sugar.spatial import EUCLIDEAN


class RepeatedLabelException(NeuroLangFrontendException):
    pass


S = bool
E = TypeVar("Entity")
P1 = Callable[[E], S]
P2 = Callable[[E, E], S]
PN = Callable[[E, List[E]], S]
S1 = Callable[[P1], S]
S2 = Callable[[P1], S1]


def M(type_):
    return Callable[[type_], type_]


def K(type_):
    return Callable[[Callable[[E], type_]], type_]


TO = Constant("to")
FROM = Constant("from")

EQ = Constant(eq)
NE = Constant(ne)
ADD = Constant(add)
DIV = Constant(truediv)
MUL = Constant(mul)
NEG = Constant(neg)
POW = Constant(pow)
SUB = Constant(sub)

ProbabilisticFactSymbol = Symbol("ProbabilisticFactSymbol")
ProbabilisticChoiceSymbol = Symbol("ProbabilistiChoiceSymbol")


class Label(FunctionApplication):
    def __init__(self, variable, label):
        self.functor = Constant(None)
        if isinstance(label, tuple):
            self.args = (variable,) + label
        else:
            self.args = (variable, label)
        self.variable = variable
        self.label = label

    def __repr__(self):
        return "Label[{}:={}]".format(self.label, self.variable)


class Aggregation(Definition):
    def __init__(self, functor, criteria, values):
        self.functor = functor
        self.criteria = criteria
        self.values = values

    def __repr__(self):
        return "Aggregate[{}, {}, {}]".format(
            self.functor, self.criteria, self.values
        )


class ExpandListArgument(Definition):
    def __init__(self, expression, list_to_replace):
        self._symbols = expression._symbols | list_to_replace._symbols
        self.expression = expression
        self.list_to_replace = list_to_replace

    def __repr__(self):
        return "ExpandListArgument[{}, {}]".format(
            self.expression, self.list_to_replace
        )


class Ref(Expression):
    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return "Ref[{}]".format(self.label)


class Arg(Definition):
    def __init__(self, preposition, args):
        self.preposition = preposition
        self.args = args

    def __repr__(self):
        return "Arg[{}, {}]".format(self.preposition, self.args)


class From(FunctionApplication):
    def __init__(self, args):
        self.functor = "TO"
        self.args = args

    def __repr__(self):
        return "To[{}]".format(self.args)


class ForArg(Definition):
    def __init__(self, preposition, arg):
        self.preposition = preposition
        self.arg = arg

    def __repr__(self):
        return "ForArg[{}, {}]".format(self.preposition, self.arg)


class Expr(Definition):
    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        return "Expr({})".format(self.arg)


class The(Quantifier):
    def __init__(self, head, arg1, arg2):
        self.head = head
        self.arg1 = arg1
        self.arg2 = arg2

    def __repr__(self):
        return "The[{}; {}, {}]".format(self.head, self.arg1, self.arg2)


class TheMarker(Expression):
    pass


class TheToUniversal(TheMarker):
    def __init__(self, expression):
        self.expression = expression
        self._symbols = expression._symbols
    
    def __repr__(self):
        return "The->Universal[{}]".format(self.expression)


class TheToExistential(TheMarker):
    def __init__(self, expression):
        self.expression = expression
        self._symbols = expression._symbols

    def __repr__(self):
        return "The->Existential[{}]".format(self.expression)


class Query(Expression):
    def __init__(self):
        pass

    def __repr__(self):
        return "Query?"


class ExtendedLogicExpressionWalker(LogicExpressionWalker):
    @add_match(ProbabilisticPredicate)
    def probabilistic_predicate(self, expression):
        return expression

    @add_match(Query)
    def query(self, expression):
        return expression

    @add_match(EQuery)
    def query(self, expression):
        return expression

    @add_match(Command)
    def command(self, expression):
        return expression


class LambdaSolverMixin(PatternWalker):
    @add_match(Lambda, lambda exp: len(exp.args) > 1)
    def explode_lambda(self, expression):
        res = expression.function_expression
        for arg in expression.args:
            res = Lambda((arg,), res)
        return res

    @add_match(FunctionApplication(Lambda, ...))
    def solve_lambda(self, expression):
        functor = self.walk(expression.functor)
        args = self.walk(expression.args)
        lambda_args = functor.args
        lambda_fun = self.walk(functor.function_expression)
        replacements = {arg: value for arg, value in zip(lambda_args, args)}
        res = LambdaReplacementsWalker(replacements).walk(lambda_fun)
        if len(lambda_args) < len(args):
            res = FunctionApplication(res, args[len(lambda_args):])
        elif len(lambda_args) > len(args):
            res = Lambda(lambda_args[len(args):], res)
        return self.walk(res)


class LambdaReplacementsWalker(ExpressionWalker):
    def __init__(self, replacements):
        self.replacements = replacements

    @add_match(Lambda)
    def process_lambda(self, expression):
        new_replacements = {
            k: v for k, v in self.replacements.items()
            if k not in expression.args
        }
        if new_replacements:
            new_function_expression = (
                LambdaReplacementsWalker(new_replacements)
                .walk(expression.function_expression)
            )
            if new_function_expression != expression.function_expression:
                expression = Lambda(
                    expression.args,
                    new_function_expression
                )
        return expression

    @add_match(Symbol)
    def process_symbol(self, expression):
        return self.replacements.get(expression, expression)


class PullUniversalUpImplicationMixin(PatternWalker):
    @add_match(Implication(UniversalPredicate, ...))
    def push_universal_up_implication(self, expression):
        consequent = expression.consequent
        antecedent = expression.antecedent
        new_head = consequent.head
        new_consequent = consequent.body
        if new_head in antecedent._symbols:
            new_head = new_head.fresh()
            new_consequent = ReplaceExpressionWalker(
                {consequent.head: new_head}
            ).walk(consequent.body)
        return UniversalPredicate(
            new_head, Implication(new_consequent, antecedent)
        )


class LogicPreprocessing(FactorQuantifiersMixin, PullUniversalUpImplicationMixin, ExtendedLogicExpressionWalker):
    @add_match(Quantifier, lambda exp: isinstance(exp.head, tuple))
    def explode_quantifier_tuples(self, expression):
        head = expression.head
        res = expression.body
        for h in sorted(head, key=repr):
            res = expression.apply(h, res)
        return self.walk(res)


def _label_in_quantifier_body(expression):
    res = any(
        isinstance(l, Label) and l.variable == expression.head
        for _, l in expression_iterator(expression.body)
    )
    return res


class ExplodeTupleArguments(ExtendedLogicExpressionWalker):
    #@add_match(
    #    Quantifier(Symbol, ...),
    #    lambda exp: any(
    #        isinstance(e, FunctionApplication) and
    #        e.functor == EQ and
    #        any(arg == exp.head for arg in e.args) and
    #        any(isinstance(arg, tuple) for arg in e.args)
    #        for _, e in expression_iterator(exp.body)
    #    )
    #)
    def tuple_equality(self, expression):
        replacements = {}
        for _, exp in expression_iterator(expression.body):
            if (
                isinstance(exp, FunctionApplication) and
                exp.functor == EQ and
                any(arg == expression.head for arg in exp.args) and
                any(isinstance(arg, tuple) for arg in exp.args)
            ):
                if isinstance(exp.args[0], tuple):
                    label = exp.args[1]
                    value = exp.args[0]
                else:
                    label = exp.args[0]
                    value = exp.args[1]
                
                if label in replacements:
                    replacements[replacements[label]] = value
                replacements[label] = value
                replacements[exp] = TRUE
        return expression.apply(value, ReplaceExpressionWalker(replacements).walk(expression.body))

    @add_match(Quantifier, lambda exp: isinstance(exp.head, tuple))
    def explode_quantifier_tuples(self, expression):
        head = expression.head
        res = expression.body
        for h in sorted(head, key=repr):
            res = expression.apply(h, res)
        return self.walk(res)

    @add_match(
        FunctionApplication,
        lambda exp: any(isinstance(arg, tuple) for arg in exp.args)
    )
    def explode_function_applications(self, expression):
        new_args = tuple()
        for arg in expression.args:
            if not isinstance(arg, tuple):
                arg = (arg,)
            new_args += arg
            type_args = get_args(expression.functor.type)
            new_type = Callable[[arg.type for arg in new_args], type_args[-1]]
        return self.walk(expression.apply(
            expression.functor.cast(new_type), new_args
        ))


def _label_expressions(expression):
    return [
        e for _, e in expression_iterator(expression)
        if isinstance(e, Label)
    ]


def _repeated_labels(expression):
    labels = _label_expressions(expression)
    label_counts = Counter(label.label for label in labels)
    return [l for l in labels if label_counts[l] > 1]


class DuplicatedLabelsVerification(ExtendedLogicExpressionWalker):
    @add_match(
        LogicOperator,
        lambda exp: _repeated_labels(exp)
    )
    def solve_repeated_labels(self, expression):
        raise RepeatedLabelException("Repeated appositions are not permitted")


class FactorQuantifiers(FactorQuantifiersMixin, ExtendedLogicExpressionWalker):
    @add_match(Condition(..., Quantifier))
    def factor_existential_out_of_condition(self, expression):
        return expression.conditioning.apply(
            expression.conditioning.head,
            expression.apply(
                expression.conditioned,
                expression.conditioning.body
            )
        )

    @add_match(Condition(Quantifier, ...))
    def factor_existential_out_of_conditional(self, expression):
        return expression.conditioned.apply(
            expression.conditioned.head,
            expression.apply(
                expression.conditioned.body,
                expression.conditioning
            )
        )


class SolveLabels(ExtendedLogicExpressionWalker):
    @add_match(
        Quantifier,
        _label_in_quantifier_body
    )
    def solve_label(self, expression):
        labels = [
            l for l in _label_expressions(expression.body)
            if l.variable == expression.head
        ]
        if labels:
            expression = ReplaceExpressionWalker(
                {labels[0]: TRUE, labels[0].variable: labels[0].label}
            ).walk(expression)
        return self.walk(expression.apply(expression.head, self.walk(expression.body)))


class SimplifyNestedImplicationsMixin(PullUniversalUpImplicationMixin):
    @add_match(Implication(Implication, ...))
    def implication_implication_other(self, expression):
        consequent = expression.consequent.consequent
        antecedent = Conjunction(
            (expression.consequent.antecedent, expression.antecedent)
        )
        return Implication(consequent, antecedent)

    @add_match(
        Implication(Conjunction, ...),
        lambda exp: (
            any(isinstance(f, Implication) for f in exp.consequent.formulas) and
            any(isinstance(f, FunctionApplication) for f in exp.consequent.formulas)
        )
    )
    def implication_conjunction_implication_other(self, expression):
        consequent_implications = []
        implication_functors = []
        consequent_other = []
        for formula in expression.consequent.formulas:
            if isinstance(formula, Implication):
                consequent_implications.append(formula)
                implication_functors.append(formula.consequent.functor)
            else:
                consequent_other.append(formula)

        consequent_final = []
        move_to_antecedent = []
        for formula in consequent_other:
            if (
                isinstance(formula, FunctionApplication) and
                formula.functor in implication_functors
            ):
                move_to_antecedent.append(formula)
            else:
                consequent_final.append(formula)

        res = Union[S](
            tuple(consequent_implications) +
            (expression.apply(
                Conjunction[S](tuple(consequent_final)),
                Conjunction[S]((expression.antecedent,) + tuple(move_to_antecedent))
            ),)
        )
        return res


class PrepositionSolverMixin(PatternWalker):
    @add_match(Arg(..., (..., Arg)))
    def arg_arg(self, expression):
        internal = self.walk(expression.args[1])
        if internal is not expression.args[1]:
            expression = Arg(expression.preposition, (expression.args[0], internal))
        return expression

    @add_match(
        Arg(..., (..., ForArg)),
        lambda exp: exp.preposition == exp.args[1].preposition
    )
    def arg_forarg(self, expression):
        return expression.args[1].arg(expression.args[0])

    @add_match(Arg(..., (..., Lambda)))
    def arg(self, expression):
        lambda_exp = expression.args[1]
        new_lambda_exp = Lambda(
            lambda_exp.args,
            Arg(expression.preposition, (expression.args[0], lambda_exp.function_expression))
        )

        return new_lambda_exp


class TheToExistentialWalker(ExpressionWalker):
    @add_match(The)
    def replace_the_existential(self, expression):
        head, d1, d2 = expression.unapply()
        return ExistentialPredicate[S](head, Conjunction[S]((d2, d1)))

    @add_match(TheMarker)
    def stop_replacing_the(self, expression):
        return expression


class TheToUniversalWalker(ExpressionWalker):
    @add_match(The)
    def replace_the_universal(self, expression):
        head, d1, d2 = expression.unapply()
        return UniversalPredicate[S](head, Implication[S](d1, d2))

    @add_match(TheMarker)
    def stop_replacing_the(self, expression):
        return expression


class SquallIntermediateSolver(PatternWalker):
    @add_match(TheToExistential)
    def replace_the_existential(self, expression):
        return TheToExistentialWalker().walk(expression.expression)

    @add_match(TheToUniversal)
    def replace_the_universal(self, expression):
        return TheToUniversalWalker().walk(expression.expression)

    @add_match(Aggregation(..., Lambda, Symbol))
    def translate_aggregation(self, expression):
        functor = expression.functor
        criteria = expression.criteria
        value = expression.values
        groups = Symbol[List[E]].fresh()
        aggregate_var = Symbol[E].fresh()
        solved_criteria = self.walk(criteria(groups)(aggregate_var))

        n_groups = max(
            -1, -1, *(
                s.item.value
                for _, s in expression_iterator(solved_criteria)
                if isinstance(s, Projection) and s.collection == groups
            )
        ) + 1
        predicate = Symbol[Callable[[E] * (n_groups + 1), S]].fresh()
        group_vars = tuple(Symbol[E].fresh() for _ in range(n_groups))
        solved_criteria = ReplaceExpressionWalker({
            Projection(groups, Constant(i)): var
            for i, var in enumerate(group_vars)
        }).walk(solved_criteria)

        group_var_labels = tuple(
            l for _, l in expression_iterator(solved_criteria)
            if (
                isinstance(l, FunctionApplication) and
                l.functor == EQ and
                l.args[0] in group_vars
            )
        )

        solved_criteria = ReplaceExpressionWalker({l: TRUE for l in group_var_labels}).walk(solved_criteria)
        solved_criteria = ReplaceExpressionWalker({l.args[1]: l.args[0] for l in group_var_labels}).walk(solved_criteria)

        aggregation_rule = UniversalPredicate(aggregate_var, Implication(
            FunctionApplication(predicate, group_vars + (AggregationApplication(functor, (aggregate_var,)),)),
            solved_criteria
        ))
        for var in group_vars:
            aggregation_rule = UniversalPredicate(var, aggregation_rule)
        aggregation_predicate = FunctionApplication(predicate, group_vars + (value,))
        formulas = (aggregation_predicate, aggregation_rule)

        formulas += group_var_labels

        return Conjunction[S](formulas)

    @add_match(ExpandListArgument)
    def expand_list_argument(self, expression):
        exp = self.walk(expression.expression)
        list_to_replace = expression.list_to_replace

        labeled_projections = list(sorted(
            (
                s for _, s in expression_iterator(exp)
                if (
                    isinstance(s, Label) and
                    isinstance(s.label, Projection) and
                    s.label.collection == list_to_replace
                )
            ),
            key=lambda s: s.label.item.value
        ))

        built_list = tuple(s.variable for s in labeled_projections[::-1])
        exp = ReplaceExpressionWalker({s: TRUE for s in labeled_projections}).walk(exp)
        exp = ReplaceExpressionWalker({list_to_replace: built_list}).walk(exp)

        return exp

    @add_match(FunctionApplication(Symbol, ...))
    def replace_type_symbols(self, expression):
        sims = getattr(self, 'type_predicate_symbols', {})
        if expression.functor.name in sims:
            return TRUE
        else:
            return expression


class Cons(Expression):
    def __init__(self, head, tail):
        self._symbols = head._symbols | tail._symbols
        self.head = head
        self.tail = tail

    def __repr__(self):
        return f"{self.head}::{self.tail}"


class SimplifyNestedProjectionMixin(PatternWalker):
    @add_match(
        Projection(Projection(..., Constant[slice]), Constant[int]),
        lambda exp: (
            exp.collection.item.value.start is not None and
            exp.collection.item.value.stop is None and
            exp.collection.item.value.step is None
        )
    )
    def simplify_nested_projection(self, expression):
        item = expression.item
        start = expression.collection.item.value.start
        collection = expression.collection.collection
        new_item = Constant(item.value + start)

        return self.walk(Projection(collection, new_item))

    @add_match(
        Cons(
            Projection(..., Constant[int](0)),
            Projection(..., Constant(slice(1, None)))
        ),
        lambda exp: exp.head.collection == exp.tail.collection
    )
    def cons_head_tail(self, expression):
        return expression.head.collection


class SquallSolver(
    SimplifyNestedProjectionMixin,
    LambdaSolverMixin,
    PrepositionSolverMixin,
    SquallIntermediateSolver,
    ExpressionWalker
):
    def __init__(self, type_predicate_symbols=None):
        if type_predicate_symbols is None:
            type_predicate_symbols = {}
        self.type_predicate_symbols = type_predicate_symbols


class LambdaSolver(LambdaSolverMixin, ExpressionWalker):
    pass


class MergeQuantifiersMixin(PatternWalker):
    @add_match(UniversalPredicate(..., UniversalPredicate))
    def merge_universals(self, expression):
        new_head = tuple(sorted(expression.head + expression.body.head))
        new_body = expression.body.body
        return UniversalPredicate(new_head, new_body)


class SplitNestedImplicationsMixin(PatternWalker):
    @add_match(
        Implication(..., Conjunction),
        lambda exp: any(
            isinstance(formula, UniversalPredicate) and
            isinstance(formula.body, Implication)
            for formula in exp.antecedent.formulas
        )
    )
    def nested_implications_to_union(self, expression):
        code = tuple()
        new_formulas = tuple()
        for formula in expression.antecedent.formulas:
            if (
                isinstance(formula, UniversalPredicate) and
                isinstance(formula.body, Implication)
            ):
                new_formulas += (formula.body.consequent,)
                code += (formula,)
        new_rule = Implication(expression.consequent, Conjunction(new_formulas))

        code += (new_rule,)

        res = Union(code) 
        return res


class SplitQuantifiersMixin(PatternWalker):
    @add_match(
        UniversalPredicate, lambda exp:
        isinstance(exp.head, tuple) and len(exp.head) > 1
    )
    def split_universals(self, expression):
        exp = expression.body
        for head in expression.head[::-1]:
            exp = UniversalPredicate((head,), exp)
        return exp


class SimplifiyEqualitiesMixin(PatternWalker):
    @add_match(
        Conjunction,
        lambda exp: any(
            isinstance(formula, FunctionApplication) and
            formula.functor == EQ and
            any(
                isinstance(formula_, FunctionApplication) and
                formula_.functor == EQ and
                formula_.args[1] in formula_._symbols
                for formula_ in exp.formulas
                if formula_ != formula
            )
            for formula in exp.formulas
        )
    )
    def simplify_equalities(self, expression):
        equalities = []
        non_equalities = []
        for formula in expression.formulas:
            if (
                isinstance(formula, FunctionApplication) and
                formula.functor == EQ
            ):
                equalities.append(formula)
            else:
                non_equalities.append(formula)
        equality_replacements = {
            formula.args[1]: formula.args[0]
            for formula in equalities
        }
        rew = ReplaceExpressionWalker(equality_replacements)
        changed = True
        while changed:
            changed = False
            new_equalities = []
            for equality in equalities:
                left, right = equality.args
                left = rew.walk(left)
                if left != equality.args[0]:
                    equality = EQ(left, right)
                    changed = True
                new_equalities.append(equality)
            equalities = new_equalities

        return Conjunction(tuple(non_equalities + equalities))

    @add_match(Negation(FunctionApplication(EQ, ...)))
    def negation_eq_to_ne(self, expression):
        return FunctionApplication(
            NE, expression.formula.args
        )


class LogicSimplifier(
    SimplifyNestedImplicationsMixin,
    RemoveTrivialOperationsMixin,
    CollapseConjunctionsMixin,
    CollapseDisjunctionsMixin,
    FactorQuantifiersMixin,
    SimplifiyEqualitiesMixin,
    ExtendedLogicExpressionWalker
):
    @add_match(Quantifier, lambda exp: isinstance(exp.head, tuple))
    def explode_quantifiers(self, expression):
        res = expression.body
        for h in expression.head:
            res = expression.apply(h, res)

        return res


NONE = Constant(None)


class EliminateSpuriousEqualities(
    PushExistentialsDownMixin,
    CollapseConjunctionsMixin,
    CollapseDisjunctionsMixin,
    RemoveTrivialOperationsMixin,
    ExtendedLogicExpressionWalker
):
    @add_match(ExistentialPredicate(..., EQ(..., ...)))
    def eliminate_trivial_existential(self, _):
        return NONE

    @add_match(
        NaryLogicOperator,
        lambda exp: any(e == NONE for e in exp.formulas)
    )
    def eliminate_element(self, expression):
        return expression.apply(
            tuple(e for e in expression.formulas if e != NONE)
        )


class PushUniversalsDownWithImplications(PushUniversalsDown):
    @add_match(UniversalPredicate(..., Implication(FunctionApplication, ...)))
    def push_universal_down_implication(self, expression):
        if expression.head in expression.body.consequent.args:
            new_antecedent = self.walk(expression.body.antecedent)
            expression = self.walk(Implication(expression.body.consequent, new_antecedent))
        else:
            expression = self.walk(
                Implication(
                    expression.body.consequent,
                    ExistentialPredicate(
                        expression.head,
                        expression.body.antecedent
                    )
                )
            )
        return expression


class GuaranteeUnion(IdentityWalker):
    @add_match(..., lambda e: not isinstance(e, Union))
    def guarantee_conjunction(self, expression):
        return Union((expression,))


class CollapseUnions(IdentityWalker):
    @add_match(
        Union,
        lambda exp: any(isinstance(f, Union) for f in exp.formulas)
    )
    def collapse_union(self, expression):
        formulas = tuple()
        for formula in expression.formulas:
            if isinstance(formula, Union):
                formulas += formula.formulas
            else:
                formulas += (formula,)

        return self.walk(Union(formulas))


def _is_aggregation_implication(expression):
    return (
        isinstance(expression, Implication) and
        isinstance(expression.consequent, FunctionApplication) and
        any(
            isinstance(arg, AggregationApplication)
            for arg in expression.consequent.args
        )
    )


class SquallExpressionsToNeuroLang(ExpressionWalker):
    @add_match(UniversalPredicate(..., Union))
    def push_universal_down_union(self, expression):
        new_formulas = tuple()
        count = 0
        for formula in expression.body.formulas:
            if expression.head in extract_logic_free_variables(formula):
                formula = UniversalPredicate(expression.head, formula)
                count += 1
            new_formulas += (formula,)

        if count <= 1:
            return self.walk(expression.body.apply(new_formulas))
        else:
            return expression

    @add_match(ExistentialPredicate(..., Union))
    def push_existential_down_union(self, expression):
        new_formulas = tuple()
        for formula in expression.body.formulas:
            if expression.head in extract_logic_free_variables(formula):
                if _is_aggregation_implication(formula):
                    formula = UniversalPredicate(expression.head, formula)
                else:
                    formula = ExistentialPredicate(expression.head, formula)
            new_formulas += (formula,)
        return self.walk(expression.body.apply(new_formulas))

    @add_match(
        UniversalPredicate(..., Implication),
        lambda exp: (
            exp.head in exp.body.consequent._symbols or
            exp.head in exp.body.antecedent._symbols
        )
    )
    def make_datalog_rule_universal(self, expression):
        return self.walk(expression.body)

    @add_match(
        UniversalPredicate(..., EQuery),
        lambda exp: (
            exp.head in exp.body.head._symbols or
            exp.head in exp.body.body._symbols
        )
    )
    def make_datalog_query_universal(self, expression):
        return self.walk(expression.body)

    @add_match(
        ExistentialPredicate(..., Implication(..., Conjunction)),
        lambda exp: (
            exp.head not in exp.body.consequent._symbols and
            any(
                exp.head in formula.antecedent._symbols
                for formula in exp.body.antecedent.formulas
                if _is_aggregation_implication(formula)
            )
        )
    )
    def make_datalog_rule_existential_aggregation(self, expression):
        return self.walk(expression.body)

    @add_match(FunctionApplication(ProbabilisticFactSymbol, (..., ...)))
    def probabilistic_fact_symbol(self, expression):
        return ProbabilisticFact(*expression.args)

    @add_match(FunctionApplication(ProbabilisticChoiceSymbol, (..., ...)))
    def probabilistic_choice_symbol(self, expression):
        return ProbabilisticChoice(*expression.args)

    @add_match(
        ExistentialPredicate(..., Implication(Conjunction, ...)),
        lambda expression: any(
            isinstance(formula, FunctionApplication) and
            formula.functor == EQ and
            expression.head in formula.args
            for formula in expression.body.consequent.formulas
        )
    )
    def move_head_equalities_to_body(self, expression):
        head = expression.head
        eq_formulas = tuple()
        other_formulas = tuple()
        for formula in expression.body.consequent.formulas:
            if (
                isinstance(formula, FunctionApplication) and
                formula.functor == EQ and
                head in formula.args
            ):
                eq_formulas += (formula,)
            else:
                other_formulas += (formula,)

        if len(other_formulas) == 1:
            consequent = other_formulas[0]
        else:
            consequent = Conjunction[S](other_formulas)
        expression = expression.body.apply(
            consequent,
            Conjunction[S](eq_formulas + (expression.body.antecedent,))
        )

        return self.walk(expression)

    @add_match(
        Implication(..., Conjunction),
        lambda exp: any(
            _is_aggregation_implication(e)
            for e in exp.antecedent.formulas
        )
    )
    def nested_implication_aggregation(self, expression):
        antecedent_formulas = expression.antecedent.formulas
        formulas_to_keep = tuple()
        aggregation_formulas = tuple()
        for formula in antecedent_formulas:
            if _is_aggregation_implication(formula):
                aggregation_formulas += (formula,)
            else:
                formulas_to_keep += (formula,)

        rule = expression.apply(
            expression.consequent,
            expression.antecedent.apply(formulas_to_keep)
        )

        aggregation_formulas += (rule,)

        return self.walk(Union(aggregation_formulas))

    @add_match(Implication(FunctionApplication(Query, ...), ...))
    def query_implication(self, expression):
        antecedent = expression.antecedent
        consequent = expression.consequent
        d = Symbol[consequent.functor.type].fresh()
        res = self.walk(EQuery(
            d(*consequent.args), antecedent
        ))
        return res


def squall_to_fol(expression, type_predicate_symbols=None):
    if type_predicate_symbols is None:
        type_predicate_symbols = {}
    cw = ChainedWalker(
        SquallSolver(type_predicate_symbols),
        ExplodeTupleArguments(),
        DuplicatedLabelsVerification(),
        FactorQuantifiers(),
        SolveLabels(),
        ExplodeTupleArguments(),
        LogicPreprocessing(),
        LogicSimplifier(),
        EliminateSpuriousEqualities(),
        SquallExpressionsToNeuroLang(),
        CollapseUnions(),
        LogicSimplifier(),
        GuaranteeUnion(),
        ReplaceExpressionWalker({
            Symbol("euclidean"): EUCLIDEAN,
            Symbol("euclidean distance"): EUCLIDEAN
        })
    )

    return cw.walk(expression)


LS = LambdaSolver()
