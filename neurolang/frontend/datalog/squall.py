from collections import Counter
from operator import add, eq, mul, ne, neg, pow, sub, truediv
from typing import Callable, List, NewType, TypeVar

from ...datalog.aggregation import AggregationApplication
from ...exceptions import NeuroLangFrontendException
from ...expression_walker import (
    ChainedWalker,
    ExpressionWalker,
    PatternWalker,
    ReplaceExpressionWalker,
    add_match,
    expression_iterator
)
from ...expressions import (
    Constant,
    Definition,
    Expression,
    FunctionApplication,
    Lambda,
    Projection,
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
from ...logic.transformations import (
    CollapseConjunctionsMixin,
    CollapseDisjunctionsMixin,
    FactorQuantifiersMixin,
    LogicExpressionWalker,
    PushExistentialsDownMixin,
    PushUniversalsDown,
    RemoveTrivialOperationsMixin
)
from ...type_system import get_args


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


class LogicPreprocessing(FactorQuantifiersMixin, PullUniversalUpImplicationMixin, LogicExpressionWalker):
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


class ExplodeTupleArguments(LogicExpressionWalker):
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
        return self.walk(FunctionApplication(
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


class DuplicatedLabelsVerification(LogicExpressionWalker):
    @add_match(
        LogicOperator,
        lambda exp: _repeated_labels(exp)
    )
    def solve_repeated_labels(self, expression):
        raise RepeatedLabelException("Repeated appositions are not permitted")


class SolveLabels(LogicExpressionWalker):
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


class SquallIntermediateSolver(PatternWalker):
    @add_match(The)
    def replace_the_existential(self, expression):
        head, d1, d2 = expression.unapply()
        return ExistentialPredicate[S](head, Conjunction[S]((d2, d1)))

    @add_match(Aggregation(..., Lambda, Symbol))
    def translate_aggregation(self, expression):
        functor = expression.functor
        criteria = expression.criteria
        value = expression.values
        predicate = Symbol.fresh()
        groups = Symbol.fresh()
        aggregate_var = Symbol.fresh()
        solved_criteria = self.walk(criteria(groups)(aggregate_var))

        n_groups = max(
            s.item.value
            for _, s in expression_iterator(solved_criteria)
            if isinstance(s, Projection) and s.collection == groups
        ) + 1

        group_vars = tuple(Symbol[E].fresh() for _ in range(n_groups))
        solved_criteria = ReplaceExpressionWalker({
            Projection(groups, Constant(i)): var
            for i, var in enumerate(group_vars)
        }).walk(solved_criteria)

        aggregate_var_label = tuple(
            l for _, l in expression_iterator(solved_criteria)
            if isinstance(l, Label) and l.variable == aggregate_var
        )

        group_var_labels = tuple(
            l for _, l in expression_iterator(solved_criteria)
            if isinstance(l, Label) and l.variable in group_vars
        )

        solved_criteria = ReplaceExpressionWalker({l: TRUE for l in group_var_labels}).walk(solved_criteria)

        aggregation_rule = UniversalPredicate(aggregate_var, Implication(
            FunctionApplication(predicate, group_vars + (AggregationApplication(functor, (aggregate_var,)),)),
            solved_criteria
        ))
        for var in group_vars:
            aggregation_rule = UniversalPredicate(var, aggregation_rule)
        aggregation_predicate = FunctionApplication(predicate, group_vars + (value,))
        formulas = (aggregation_predicate, aggregation_rule)

        formulas += tuple(EQ(l.label, l.variable) for l in group_var_labels)

        if aggregate_var_label and isinstance(aggregate_var_label[0].label, tuple):
            new_tuple = tuple(Symbol[E].fresh() for _ in aggregate_var_label[0].label)
            aggregate_var_label = ReplaceExpressionWalker(
                {aggregate_var_label[0]: Label(value, new_tuple)}
            ).walk(aggregate_var_label)
            formulas += aggregate_var_label

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
    pass


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
    LogicExpressionWalker
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
    LogicExpressionWalker
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


def squall_to_fol(expression):
    cw = ChainedWalker(
        SquallSolver(),
        ExplodeTupleArguments(),
        DuplicatedLabelsVerification(),
        LogicPreprocessing(),
        SolveLabels(),
        ExplodeTupleArguments(),
        LogicSimplifier(),
        EliminateSpuriousEqualities(),
        PushUniversalsDownWithImplications()
    )

    return cw.walk(expression)


LS = LambdaSolver()
