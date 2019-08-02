from .expressions import (
    Symbol, Constant, ExpressionBlock,
    FunctionApplication,
)
from .expression_walker import (
    PatternWalker, add_match, expression_iterator
)
from .solver_datalog_naive import (
    Implication, Fact
)

from .exceptions import NeuroLangException

class NeuroLangDataLogNonWarded(NeuroLangException):
    pass

class WardedDatalog(PatternWalker):

    def __init__(self):
        self.can_be_dangerous = dict({})


    @add_match(ExpressionBlock)
    def warded_expression_block(self, expression):
        for rule in expression.expressions:
            self.walk(rule)

        cdv = CheckDangerousVariables(self.can_be_dangerous)
        cdv.walk(expression)

        return True


    @add_match(FunctionApplication(Constant, ...))
    def warded_function_constant(self, expression):
        symbols = set()
        for arg in expression.args:
            temp = self.walk(arg)
            symbols = symbols.union(temp)

        return symbols


    @add_match(FunctionApplication)
    def warded_function_application(self, expression):
        symbols = set()
        for arg in expression.args:
            symbol = self.walk(arg)
            symbols = symbols.union(symbol)

        return symbols


    @add_match(Fact)
    def warded_fact(self, expression):
        pass
        #symbols = set()
        #for arg in expression.fact.args:
        #    symbol = self.walk(arg)
        #    symbols = symbols.union(symbol)

        #return symbols


    @add_match(Implication)
    def warded_implication(self, expression):
        antecedent = self.walk(expression.antecedent)
        consequent = self.walk(expression.consequent)

        free_vars = antecedent.symmetric_difference(consequent)

        for var in free_vars:
            if var in consequent:
                position = self.calc_position(var, expression.consequent)
                self.can_be_dangerous = self.merge_dicts(self.can_be_dangerous, position)


    @add_match(Symbol)
    def warded_symbol(self, expression):
        return set(expression.name)


    @add_match(Constant)
    def warded_constant(self, expression):
        pass


    def calc_position(self, var, expression):
        for exp in expression_iterator(expression):
            if var in exp[1].args:
                return dict({exp[1].functor: [exp[1].args.index(var)]})


    def merge_dicts(self, to_update_dic, new_dict):
        for key, value in new_dict.items():
            if key in to_update_dic:
                old_values = to_update_dic[key]
                old_values.append(value)
                to_update_dic[key] = old_values
            else:
                to_update_dic[key] = [value]

        return to_update_dic


class CheckDangerousVariables(PatternWalker):

    def __init__(self, can_be_dangerous):
        self.can_be_dangerous = can_be_dangerous
        self.dangerous_vars = {}


    @add_match(ExpressionBlock)
    def check_dangerous_block(self, expression):
        for rule in expression.expressions:
            self.walk(rule)


    @add_match(Fact)
    def check_dangerous_fact(self, expression):
        pass


    @add_match(Implication)
    def check_dangerous_implication(self, expression):
        antecedent = self.check_dangerous(expression.antecedent)
        consequent = self.check_dangerous(expression.consequent)

        dangerous_symbol = antecedent.intersection(consequent)
        if len(dangerous_symbol) == 1 and next(iter(dangerous_symbol)) in expression.antecedent._symbols:
            var = dangerous_symbol.pop()
            dangerous_pos = self.can_be_dangerous[var].pop()

            dangerous_var = self.get_name(expression.consequent, dangerous_pos)

            single_body = self.check_var_single_body(dangerous_var, expression.antecedent)
            if not single_body:
                raise NeuroLangDataLogNonWarded(
                    f'The program is not warded: there are dangerous variables outside the ward in {expression.antecedent}'
                )


    def check_dangerous(self, expression):
        dangerous = set()
        for key in self.can_be_dangerous.keys():
            if key in expression.functor:
                dangerous.add(key)

        if len(dangerous) > 1:
            raise NeuroLangDataLogNonWarded(
                f'The program is not warded: there are dangerous variables that appear in more than one atom of the body in {expression}'
            )

        return dangerous

    def get_name(self, expression, position):
        names = [expression.args[index] for index in position]
        #TODO Remove this conditional
        if len(names) > 1:
            raise NeuroLangException(
                f'DEBUG: Unexpected length'
            )
        return names[0]

    def check_var_single_body(self, var, expression):
        founded = False
        for exp in expression_iterator(expression):
            if exp[0] is 'args':
                if not founded and var in exp[1]:
                    founded = True
                elif founded and var in exp[1]:
                    return False

        return True