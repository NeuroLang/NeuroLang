from itertools import product
from uuid import uuid4
import typing

import pandas

from .expression_walker import (
    PatternWalker, add_match,
    ReplaceSymbolWalker
)
from .expressions import (
    Constant, Symbol, Query,
    Lambda, FunctionApplication,
    ToBeInferred, is_subtype,
    NeuroLangTypeException
)
from .solver_datalog_naive import (
    extract_datalog_free_variables,
    extract_datalog_predicates
)
from .exceptions import NeuroLangException


MAX_RECURSION = 10000


class MaxRecursionNeurolangException(NeuroLangException):
    pass


class DatalogSeminaiveEvaluator(PatternWalker):
    '''
    Query evaluation based on relational algebra operations
    implemented over Python sets.
    Recursive evaluation based on fixpoint semantics.
    Indirect recursion is not supported and will break.
    '''

    @add_match(FunctionApplication(Lambda, ...))
    def evaluate_lambda(self, expression):
        args = list(expression.args)

        rsv = ReplaceSymbolWalker(dict(zip(expression.functor.args, args)))
        function = rsv.walk(expression.functor.function_expression)
        predicates = extract_datalog_predicates(function)

        # Merging predicate by predicate progressively reduces the
        # length of the crossproduct.
        # The better implementation would be to use an IR on relational
        # algebra and then apply known optimizations there.
        # Or better implement it over SQL or NoSQL.

        ext_pred = []
        int_pred = []
        other_pred = []
        edb = self.extensional_database()
        idb = self.intensional_database()
        for p in predicates:
            if p.functor in edb:
                ext_pred.append(p)
            elif p.functor in idb:
                int_pred.append(p)
            else:
                other_pred.append(p)

        predicates = ext_pred + int_pred

        if len(predicates) == 0 and len(other_pred) > 0:
            raise NotImplemented(
                "Predicates not in EDB or IDB can't bre alone"
            )

        cur_args = predicates[0].args
        res = self.multijoin_named(
            [self.walk(predicates[0])],
            [cur_args]
        )

        for pred in predicates[1:]:
            res = self.multijoin_named(
                [res, self.walk(pred)],
                [cur_args, pred.args]
            )

            cur_args = cur_args + pred.args

        for pred in other_pred:
            new_res = set()
            for t in res:
                replacement = dict(zip(cur_args, t.value))
                rsv = ReplaceSymbolWalker(replacement)
                r = self.walk(
                    self.symbol_table[pred.functor](*rsv.walk(pred.args))
                )
                if isinstance(r, Constant) and r.value is True:
                    new_res.add(t)
            res = new_res

        final_project_args = [
            cur_args.index(a) for a in args
            if a in cur_args
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

        if functor in self.extensional_database():
            res = self.select(
                self.symbol_table[functor].value,
                constants
            )

        elif functor in self.intensional_database():
            fresh_name = functor.name + str(uuid4())
            while fresh_name in self.symbol_table:
                fresh_name = functor.name + str(uuid4())

            fresh_symbol = Symbol[functor.type](fresh_name)
            self.symbol_table[fresh_symbol] = Constant[functor.type](set())

            replace_symbol_variable = ReplaceSymbolWalker(
                {functor: fresh_symbol}
            )

            rules = self.intensional_database()[functor]
            rules = replace_symbol_variable.walk(rules)

            old_res = self.symbol_table[fresh_symbol].value

            res = set()
            for i in range(MAX_RECURSION):
                old_res |= res

                res = self.walk(rules.expressions[0](*args))
                for rule in rules.expressions[1:]:
                    res |= self.walk(rule(*args))

                if res == old_res:
                    break
            else:
                raise MaxRecursionNeurolangException(
                    'Maximum number of recursions reached ' +
                    f'for symbol {functor}'
                )

            del self.symbol_table[fresh_symbol]
        else:
            raise NeuroLangException(f'Symbol {functor} not defined')

        return res

    @add_match(
        FunctionApplication(Constant(...), ...),
        lambda e: all(
            isinstance(arg, Constant)
            for arg in e.args
        )
    )
    def evaluate_function(self, expression):
        functor = expression.functor
        args = expression.args
        if functor.type is not ToBeInferred:
            if not is_subtype(functor.type, typing.Callable):
                raise NeuroLangTypeException(
                    'Function {} is not of callable type'.format(functor)
                )
            result_type = functor.type.__args__[-1]
        else:
            if not callable(functor.value):
                raise NeuroLangTypeException(
                    'Function {} is not of callable type'.format(functor)
                )
            result_type = ToBeInferred

        args = (a.value for a in args)
        result = Constant[result_type](
            functor.value(*args)
        )
        return result

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
    def multijoin_named(sets, sets_args):
        if len(sets) > 2:
            raise NotImplemented()

        import pdb; pdb.set_trace()
        sets = [
            pandas.DataFrame.from_records(iter(s))
            for s in sets
        ]
    
        join_arguments = sum(sets_args, tuple())

        joins = dict()
        for i, args in enumerate(sets_args):
            args_map = dict()
            for j, a in enumerate(args):
                if isinstance(a, Symbol):
                    args_map[j] = join_arguments.index(a)
            joins[i] = args_map

        sel_str = (
            'lambda s, t: ' +
            ' and '.join(
                f's[{i}].value[{j}].value == t[{k}].value'
                for i, arg_map in joins.items()
                for j, k in arg_map.items()
            )
        )
        sel = eval(sel_str)

        res = []
        for s in product(*sets):
            t = s[0].value
            for t_ in s[1:]:
                t += t_.value
            if sel(s, t):
                res.append(Constant(t))

        return set(res)

    @staticmethod
    def multijoin_named_old(sets, sets_args):
        join_arguments = sum(sets_args, tuple())

        joins = dict()
        for i, args in enumerate(sets_args):
            args_map = dict()
            for j, a in enumerate(args):
                if isinstance(a, Symbol):
                    args_map[j] = join_arguments.index(a)
            joins[i] = args_map

        sel_str = (
            'lambda s, t: ' +
            ' and '.join(
                f's[{i}].value[{j}].value == t[{k}].value'
                for i, arg_map in joins.items()
                for j, k in arg_map.items()
            )
        )
        sel = eval(sel_str)

        res = []
        for s in product(*sets):
            t = s[0].value
            for t_ in s[1:]:
                t += t_.value
            if sel(s, t):
                res.append(Constant(t))

        return set(res)
