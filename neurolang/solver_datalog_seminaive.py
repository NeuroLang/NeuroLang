from uuid import uuid4
import typing

from .expression_walker import (
    add_match,
    ReplaceSymbolWalker
)
from .expressions import (
    Constant, Symbol, Query,
    Lambda, FunctionApplication,
    Unknown, is_leq_informative,
    NeuroLangTypeException
)
from .solver_datalog_naive import (
    extract_datalog_free_variables,
    extract_datalog_predicates,
    RelationalAlgebraSetIR,
    DatalogBasic
)
from .exceptions import NeuroLangException


MAX_RECURSION = 10000


class MaxRecursionNeurolangException(NeuroLangException):
    pass


class DatalogSeminaiveEvaluator(DatalogBasic):
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
        res = self.walk(predicates[0])

        for pred in predicates[1:]:
            join_left = []
            join_right = []
            for i, arg in enumerate(pred.args):
                try:
                    join_left.append(cur_args.index(arg))
                    join_right.append(i)
                except ValueError:
                    pass

            res = res.join_by_columns(self.walk(pred), join_left, join_right)
            cur_args = cur_args + pred.args

        for pred in other_pred:
            new_res = set()
            for t in res:
                replacement = dict(zip(cur_args, t.value))
                rsv = ReplaceSymbolWalker(replacement)
                r = self.walk(
                    FunctionApplication(
                        self.symbol_table[pred.functor],
                        rsv.walk(pred.args)
                    )
                )
                if isinstance(r, Constant) and r.value is True:
                    new_res.add(t)
            res = RelationalAlgebraSetIR(new_res)

        final_project_args = [
            cur_args.index(a) for a in args
            if a in cur_args
        ]
        res = res.project(final_project_args)
        return res

    @add_match(FunctionApplication(Symbol, ...))
    def fa(self, expression):
        functor = expression.functor
        args = expression.args

        if functor in self.extensional_database():
            constants = dict()
            parameter_equalities = dict()
            arglist = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, Constant):
                    constants[i] = arg
                elif isinstance(arg, Symbol):
                    ix = arglist.index(arg)
                    if ix < i:
                        parameter_equalities[i] = ix

            res = self.symbol_table[functor].value.select_equality(constants)
            res = res.select_columns(parameter_equalities)

        elif functor in self.intensional_database():
            fresh_name = functor.name + str(uuid4())
            while fresh_name in self.symbol_table:
                fresh_name = functor.name + str(uuid4())

            fresh_symbol = Symbol[functor.type](fresh_name)
            self.symbol_table[fresh_symbol] = Constant[functor.type](
                RelationalAlgebraSetIR()
            )

            replace_symbol_variable = ReplaceSymbolWalker(
                {functor: fresh_symbol}
            )

            rules = self.intensional_database()[functor]
            rules = replace_symbol_variable.walk(rules)

            old_res = self.symbol_table[fresh_symbol].value

            res = RelationalAlgebraSetIR()
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
        if functor.type is not Unknown:
            if not is_leq_informative(functor.type, typing.Callable):
                raise NeuroLangTypeException(
                    'Function {} is not of callable type'.format(functor)
                )
            result_type = functor.type.__args__[-1]
        else:
            if not callable(functor.value):
                raise NeuroLangTypeException(
                    'Function {} is not of callable type'.format(functor)
                )
            result_type = Unknown

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

        return res.project(indices_to_project)
