from collections import Counter
from functools import cmp_to_key
from operator import add, eq, mul, ne, neg, pow, sub, truediv
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
    def equery(self, expression):
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
        for src, dst in zip(lambda_args, args):
            lambda_fun = LambdaReplacementsWalker(src, dst).walk(lambda_fun)
        res = lambda_fun
        if len(lambda_args) < len(args):
            res = FunctionApplication(res, args[len(lambda_args):])
        elif len(lambda_args) > len(args):
            res = Lambda(lambda_args[len(args):], res)
        return self.walk(res)


class LambdaReplacementsWalker(ExpressionWalker):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    @add_match(Lambda)
    def process_lambda(self, expression):
        if self.src not in expression.args:
            function_expression = self.walk(expression.function_expression)
            if function_expression is not expression.function_expression:
                expression = expression.apply(
                    expression.args,
                    self.walk(expression.function_expression)
                )
        return expression

    def process_lambda_(self, expression):
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
        if expression == self.src:
            return self.dst
        else:
            return expression


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


class LogicPreprocessing(
    FactorQuantifiersMixin,
    PullUniversalUpImplicationMixin,
    ExtendedLogicExpressionWalker
):
    @add_match(ExistentialPredicate, lambda exp: isinstance(exp.head, tuple))
    def explode_existential_tuples(self, expression):
        head = expression.head
        res = expression.body
        for h in sorted(head, key=repr):
            res = expression.apply(h, res)
        return self.walk(res)

    @add_match(UniversalPredicate, lambda exp: isinstance(exp.head, tuple))
    def explode_universal_tuples(self, expression):
        return self.explode_existential_tuples(expression)


def _label_in_quantifier_body(expression):
    res = any(
        isinstance(l, Label) and l.variable == expression.head
        for _, l in expression_iterator(expression.body)
    )
    return res


class ExplodeTupleArguments(ExtendedLogicExpressionWalker):
    # @add_match(
    #    Quantifier(Symbol, ...),
    #    lambda exp: any(
    #        isinstance(e, FunctionApplication) and
    #        e.functor == EQ and
    #        any(arg == exp.head for arg in e.args) and
    #        any(isinstance(arg, tuple) for arg in e.args)
    #        for _, e in expression_iterator(exp.body)
    #    )
    # )
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
        return expression.apply(
            value,
            ReplaceExpressionWalker(replacements).walk(expression.body)
        )

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

    @add_match(
        FunctionApplication,
        lambda exp: any(isinstance(arg, Constant[List[E]]) for arg in exp.args)
    )
    def explode_function_applications_list(self, expression):
        new_args = tuple()
        for arg in expression.args:
            if isinstance(arg, Constant[List[E]]):
                arg = tuple(arg.value)
            else:
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


class FactorQuantifierConditionMixin(PatternWalker):
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
    
    @add_match(ExistentialPredicate(..., Condition))
    def remove_existential_on_condition(self, expression):
        return expression.body


class FactorQuantifiers(
    FactorQuantifierConditionMixin,
    FactorQuantifiersMixin,
    ExtendedLogicExpressionWalker
):
    pass


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
        return self.walk(
            expression.apply(
                expression.head,
                self.walk(expression.body)
            )
        )


class SimplifyNestedImplicationsMixin(PullUniversalUpImplicationMixin):
    @add_match(Implication(Implication, ...))
    def implication_implication_other(self, expression):
        consequent = expression.consequent.consequent
        antecedent = Conjunction(
            (expression.consequent.antecedent, expression.antecedent)
        )
        return self.walk(Implication(consequent, antecedent))

    @add_match(
        Implication(Conjunction, ...),
        lambda exp: (
            any(isinstance(f, Implication) for f in exp.consequent.formulas)
            and
            any(
                isinstance(f, FunctionApplication)
                for f in exp.consequent.formulas
            )
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
                Conjunction[S](
                    (expression.antecedent,) +
                    tuple(move_to_antecedent)
                )
            ),)
        )
        return self.walk(res)

    @add_match(
        Implication(..., Implication),
        lambda exp: exp.consequent == exp.antecedent.consequent
    )
    def nested_implication_equal_consequent(self, expression):
        return self.walk(expression.antecedent)

    @add_match(Implication(..., Implication(TRUE, ...)))
    def nested_implication_2nd_True(self, expression):
        return self.walk(expression.apply(
            expression.consequent,
            expression.antecedent.antecedent
        ))

    @add_match(Implication(TRUE, Implication))
    def nested_implication_1st_True(self, expression):
        return self.walk(expression.antecedent)


class TheToExistentialWalker(ExpressionWalker):
    @add_match(The)
    def replace_the_existential(self, expression):
        head, d1, d2 = expression.unapply()
        return self.walk(ExistentialPredicate[S](head, Conjunction[S]((d2, d1))))

    @add_match(TheMarker)
    def stop_replacing_the(self, expression):
        return self.walk(expression)


class TheToUniversalWalker(ExpressionWalker):
    @add_match(The)
    def replace_the_universal(self, expression):
        head, d1, d2 = expression.unapply()
        return self.walk(UniversalPredicate[S](head, Implication[S](d1, d2)))

    @add_match(TheMarker)
    def stop_replacing_the(self, expression):
        return self.walk(expression)


class SquallIntermediateSolver(PatternWalker):
    @add_match(TheToExistential)
    def replace_the_existential(self, expression):
        expression = self.walk(expression.expression)
        return self.walk(TheToExistentialWalker().walk(expression))

    @add_match(TheToUniversal)
    def replace_the_universal(self, expression):
        expression = self.walk(expression.expression)
        return self.walk(TheToUniversalWalker().walk(expression))

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

        solved_criteria = ReplaceExpressionWalker(
            {l: TRUE for l in group_var_labels}
        ).walk(solved_criteria)
        solved_criteria = ReplaceExpressionWalker(
            {l.args[1]: l.args[0] for l in group_var_labels}
        ).walk(solved_criteria)

        application = self.walk(FunctionApplication(functor, (aggregate_var,)))

        aggregation_rule = UniversalPredicate(aggregate_var, Implication(
            FunctionApplication(
                predicate,
                group_vars +
                (AggregationApplication(*application.unapply()),)
            ),
            solved_criteria
        ))
        for var in group_vars:
            aggregation_rule = UniversalPredicate(var, aggregation_rule)
        aggregation_predicate = FunctionApplication(
            predicate,
            group_vars + (value,)
        )
        formulas = (aggregation_predicate, aggregation_rule)

        formulas += group_var_labels

        return self.walk(Conjunction[S](formulas))

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
        built_list = Constant[List[E]](built_list, verify_type=False)
        exp = ReplaceExpressionWalker(
            {s: TRUE for s in labeled_projections}
        ).walk(exp)
        exp = ReplaceExpressionWalker({list_to_replace: built_list}).walk(exp)

        return self.walk(exp)

    @add_match(FunctionApplication(Symbol, ...))
    def replace_type_symbols(self, expression):
        sims = getattr(self, 'type_predicate_symbols', {})
        if expression.functor.name in sims:
            return TRUE
        else:
            functor = self.walk(expression.functor)
            args = self.walk(expression.args)
            if (
                functor is expression.functor and
                self._tuples_are_the_same(args, expression.args)
            ):
                return expression
            else:
                return self.walk(expression.apply(functor, args))

    def _tuples_are_the_same(self, t1, t2):
        for e1, e2, in zip(t1, t2):
            if isinstance(e1, tuple) and isinstance(e2, tuple):
                if not self._tuples_are_the_same(e1, e2):
                    return False
            elif e1 is not e2:
                return False
        return True


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


class SquallSolver(
    RemoveTrivialOperationsMixin,
    SimplifyNestedProjectionMixin,
    LambdaSolverMixin,
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
        return self.walk(UniversalPredicate(new_head, new_body))


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
        new_rule = Implication(
            expression.consequent,
            Conjunction(new_formulas)
        )

        code += (new_rule,)

        res = Union(code)
        return self.walk(res)


class SplitQuantifiersMixin(PatternWalker):
    @add_match(
        UniversalPredicate, lambda exp:
        isinstance(exp.head, tuple) and len(exp.head) > 1
    )
    def split_universals(self, expression):
        exp = expression.body
        for head in expression.head[::-1]:
            exp = UniversalPredicate((head,), exp)
        return self.walk(exp)


class NormalizeEqualities(PatternWalker):
    @add_match(
        EQ(..., Symbol),
        lambda expression: (
            not isinstance(expression.args[0], Symbol) or
            expression.args[0].name > expression.args[1].name
        )
    )
    def flip_equality(self, expression):
        return self.walk(expression.apply(
            expression.functor,
            expression.args[::-1]
        ))


def _cmp_equalities(left, right):
    if left.args[0] in right.args[1]._symbols:
        return -1
    elif right.args[0] in left.args[1]._symbols:
        return 1
    else:
        return 0


class SimplifiyEqualitiesMixin(NormalizeEqualities):
    @add_match(
        Conjunction,
        lambda exp: any(
            any(
                _cmp_equalities(formula, formula_) != 0
                for formula_ in exp.formulas[i + 1:]
                if (
                    isinstance(formula_, FunctionApplication) and
                    formula_.functor == EQ
                )
            )
            for i, formula in enumerate(exp.formulas)
            if (
                isinstance(formula, FunctionApplication) and
                formula.functor == EQ
            )
        )
    )
    def simplify_equalities(self, expression):
        equalities = []
        non_equalities = []
        for formula in expression.formulas:
            if isinstance(formula, FunctionApplication) and formula.functor == EQ:
                equalities.append(formula)
            else:
                non_equalities.append(formula)

        equalities = list(sorted(equalities, key=cmp_to_key(_cmp_equalities)))
        equality_replacements = {equalities[0].args[0]: equalities[0].args[1]}
        rew = ReplaceExpressionWalker(equality_replacements)
        new_equalities = [rew.walk(equality) for equality in equalities[1:]]
        return self.walk(Conjunction(tuple(non_equalities + new_equalities)))

    @add_match(Negation(FunctionApplication(EQ, ...)))
    def negation_eq_to_ne(self, expression):
        return self.walk(expression.formula.apply(
            NE.cast(expression.functor.type), expression.formula.args
        ))


class LogicSimplifier(
    SimplifyNestedImplicationsMixin,
    RemoveTrivialOperationsMixin,
    CollapseConjunctionsMixin,
    CollapseDisjunctionsMixin,
    SimplifiyEqualitiesMixin,
    ExtendedLogicExpressionWalker
):
    @add_match(Quantifier, lambda exp: isinstance(exp.head, tuple))
    def explode_quantifiers(self, expression):
        res = expression.body
        for h in expression.head:
            res = expression.apply(h, res)

        return self.walk(res)

    @add_match(
        ExistentialPredicate(..., Implication),
        lambda expression: (
            expression.head
            not in extract_logic_free_variables(expression.body.consequent)
        )
    )
    def push_existential_down_implication(self, expression):
        return self.walk(expression.body.apply(
            expression.body.consequent,
            expression.apply(expression.head, expression.body.antecedent)
        ))


class LogicSimplifierPost(
    SimplifyNestedImplicationsMixin,
    RemoveTrivialOperationsMixin,
    CollapseConjunctionsMixin,
    CollapseDisjunctionsMixin,
    SimplifiyEqualitiesMixin,
    FactorQuantifiersMixin,
    ExtendedLogicExpressionWalker
):
    @add_match(UniversalPredicate(..., Implication))
    def univesal_implication_to_rule(self, expression):
        return self.walk(expression.body)

    @add_match(Implication(..., ExistentialPredicate))
    def existential_predicate_antecedent_remove(self, expression):
        return self.walk(expression.apply(
            expression.consequent,
            expression.antecedent.body
        ))


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
        return self.walk(expression.apply(
            tuple(e for e in expression.formulas if e != NONE)
        ))


class PushUniversalsDownWithImplications(PushUniversalsDown):
    @add_match(UniversalPredicate(..., Implication(FunctionApplication, ...)))
    def push_universal_down_implication(self, expression):
        if expression.head in expression.body.consequent.args:
            new_antecedent = self.walk(expression.body.antecedent)
            expression = self.walk(
                Implication(expression.body.consequent, new_antecedent)
            )
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
    while isinstance(expression, ExistentialPredicate):
        expression = expression.body

    return (
        isinstance(expression, Implication) and
        isinstance(expression.consequent, FunctionApplication) and
        any(
            isinstance(arg, AggregationApplication)
            for arg in expression.consequent.args
        )
    )


class SquallExpressionsToNeuroLang(
    FactorQuantifierConditionMixin,
    ExpressionWalker
):
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
        if expression.head in expression.body.consequent._symbols:
            expression = expression.body
        else:
            expression = expression.body.apply(
                expression.body.consequent,
                ExistentialPredicate[expression.type](
                    expression.head,
                    expression.body.antecedent
                )
            )
        return self.walk(expression)

    @add_match(
        ExistentialPredicate(Symbol, Conjunction),
        lambda expression: any(
            expression.head not in extract_logic_free_variables(formula)
            for formula in expression.body.formulas
        )
    )
    def push_existential_down_conjunction(self, expression):
        new_formulas_in = tuple()
        new_formulas_out = tuple()
        head = expression.head
        for formula in expression.body.formulas:
            if head in extract_logic_free_variables(formula):
                new_formulas_in += (formula,)
            else:
                new_formulas_out += (formula,)
        e_formula = expression.apply(
            head,
            expression.body.apply(new_formulas_in)
        )
        new_formulas = new_formulas_out + (e_formula,)
        return expression.body.apply(new_formulas)

    @add_match(
        UniversalPredicate(..., EQuery),
        lambda exp: (
            exp.head in exp.body.head._symbols or
            exp.head in exp.body.body._symbols
        )
    )
    def make_datalog_query_universal(self, expression):
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
                while isinstance(formula, ExistentialPredicate):
                    formula = formula.body
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

    @add_match(Condition(Implication, ...))
    def factor_implication_from_conditioned(self, expression):
        return Implication[S](
            expression.conditioned.consequent,
            Condition[S](expression.conditioned.antecedent, expression.conditioning)
        )

    @add_match(Condition(..., Implication))
    def factor_implication_from_conditioning(self, expression):
        return Implication[S](
            expression.conditioning.consequent,
            Condition[S](expression.conditioned, expression.conditioning.antecedent)
        )

    @add_match(
        Implication(..., Implication),
        lambda exp: exp.consequent == exp.antecedent.consequent
    )
    def nested_implication_equal_consequent(self, expression):
        return expression.antecedent

    @add_match(Implication(..., Implication(TRUE, ...)))
    def nested_implication_2nd_True(self, expression):
        return expression.apply(
            expression.consequent,
            expression.antecedent.antecedent
        )

    @add_match(Implication(TRUE, Implication))
    def nested_implication_1st_True(self, expression):
        return expression.antecedent


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
        LogicSimplifierPost(),
        GuaranteeUnion(),
        ReplaceExpressionWalker({
            Symbol[EUCLIDEAN.type]("euclidean"): EUCLIDEAN,
            Symbol[EUCLIDEAN.type]("euclidean distance"): EUCLIDEAN
        })
    )

    return cw.walk(expression)


LS = LambdaSolver()
