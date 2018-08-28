from itertools import product
from operator import and_

# from . import solver_datalog_naive
from .expression_walker import (
    PatternWalker, add_match, expression_iterator,
    ReplaceSymbolWalker
)
from .expressions import (
    Constant, Symbol,
    Lambda, FunctionApplication
)


class DatalogSeminaiveEvaluator(PatternWalker):
    def __init__(self, datalog_walker):
        self.datalog = datalog_walker
        self.instance = self.datalog.extensional_database()

    @add_match(FunctionApplication(Lambda, ...))
    def evaluate_lambda(self, expression):
        args = list(expression.args)
        predicates = []
        join_arguments = []

        rsv = ReplaceSymbolWalker(zip(expression.functor.args, args))
        function = rsv.walk(expression.functor.function_expression)
        for _, p in expression_iterator(function):
            if (
                isinstance(p, FunctionApplication) and
                not (
                    isinstance(p.functor, Constant) and
                    p.functor.value is and_
                )
            ):
                predicates.append(p)
                join_arguments.extend(p.args)

        joins = dict()
        for i, p in enumerate(predicates):
            args_map = dict()
            for j, a in enumerate(p.args):
                if isinstance(a, Symbol):
                    args_map[j] = join_arguments.index(a)
            joins[i] = args_map

        res_set = [
            self.walk(p)
            for p in predicates
        ]

        res = self.multijoin(
            res_set,
            joins
        )
        final_project_args = [join_arguments.index(a) for a in args]
        res = self.project(res, final_project_args)
        return res

    @add_match(FunctionApplication(Symbol, ...))
    def fa(self, expression):
        functor = expression.functor
        args = expression.args
        constants = {
            i: arg
            for i, arg in enumerate(args)
            if isinstance(arg, Constant)
        }

        res = set()
        if functor in self.instance:
            res = self.select(self.instance[functor].value, constants)

        if functor in self.datalog.intensional_database():
            rules = self.datalog.intensional_database()[functor]
            res_intensional = self.walk(rules.expressions[0](*args))
            for rule in rules.expressions[1:]:
                res_intensional |= self.walk(rule(*args))
            res = res | res_intensional

        return res

    @staticmethod
    def select(set_, constants):
        return set(
            t for t in set_
            if all(
                t.value[i] == v
                for i, v in constants.items()
            )
        )

    @staticmethod
    def project(set_, indices):
        return set(
            Constant(tuple(t.value[i] for i in indices))
            for t in set_
        )

    @staticmethod
    def multijoin(sets, joins):
        res = []
        for s in product(*sets):
            t = s[0].value
            for t_ in s[1:]:
                t += t_.value

            if all(
                s[i].value[j].value == t[k].value
                for i, arg_map in joins.items()
                for j, k in arg_map.items()
             ):
                res.append(Constant(t))

        return set(res)
