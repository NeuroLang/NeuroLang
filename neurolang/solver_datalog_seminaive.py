from itertools import product

from .expression_walker import (
    PatternWalker, add_match,
    ReplaceSymbolWalker
)
from .expressions import (
    Constant, Symbol, Query,
    Lambda, FunctionApplication
)
from .solver_datalog_naive import (
    extract_datalog_free_variables,
    extract_datalog_predicates
)


class DatalogSeminaiveEvaluator(PatternWalker):
    @property
    def instance(self):
        return self.extensional_database()

    @add_match(FunctionApplication(Lambda, ...))
    def evaluate_lambda(self, expression):
        args = list(expression.args)
        join_arguments = []

        rsv = ReplaceSymbolWalker(dict(zip(expression.functor.args, args)))
        function = rsv.walk(expression.functor.function_expression)
        predicates = extract_datalog_predicates(function)
        for p in predicates:
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
        final_project_args = [
            join_arguments.index(a) for a in args
            if a in join_arguments
        ]
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

        if functor in self.intensional_database():
            rules = self.intensional_database()[functor]
            res_intensional = self.walk(rules.expressions[0](*args))
            for rule in rules.expressions[1:]:
                res_intensional |= self.walk(rule(*args))
            res = res | res_intensional

        return res

    @add_match(Query)
    def query(self, expression):
        if isinstance(expression.head, Symbol):
            head = (expression.head,)
        else:
            head = expression.head

        res = self.walk(expression.body)
        body_fv = extract_datalog_free_variables(expression.body)
        indices_to_project = tuple(
            body_fv.index(v)
            for v in head
        )

        return self.project(res, indices_to_project)

    @staticmethod
    def select(set_, constants):
        if len(constants) == 0:
            return set_

        select_str = (
            'lambda t, constants=constants: ' +
            ' and '.join(
                f't.value[{i}] == constants[{i}].value '
                for i in constants
            )
        )

        sel = eval(select_str)

        return set(
            t for t in set_
            if sel(t)
        )

    @staticmethod
    def project(set_, indices):
        project_str = (
            'lambda t: (' +
            ' ,'.join(f't.value[{i}]' for i in indices) +
            ',)'
        )

        proj = eval(project_str)
        return set(
            Constant(proj(t))
            for t in set_
        )

    @staticmethod
    def multijoin(sets, joins):
        res = []

        sel_str = (
            'lambda s, t: ' +
            ' and '.join(
                f's[{i}].value[{j}].value == t[{k}].value'
                for i, arg_map in joins.items()
                for j, k in arg_map.items()
            )
        )
        sel = eval(sel_str)

        for s in product(*sets):
            t = s[0].value
            for t_ in s[1:]:
                t += t_.value
            if sel(s, t):
                res.append(Constant(t))

        return set(res)
