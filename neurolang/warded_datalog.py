from .expressions import (
    Symbol, Constant, ExpressionBlock, FunctionApplication,
    ExistentialPredicate
)
from .expression_walker import (PatternWalker, add_match, expression_iterator)
from .solver_datalog_naive import (Implication, Fact)

from .exceptions import NeuroLangException
from .utils.orderedset import OrderedSet


class NeuroLangNonWardedException(NeuroLangException):
    pass


class WardedDatalogDangerousVariableExtraction(PatternWalker):
    @add_match(ExpressionBlock)
    def warded_expression_block(self, expression):
        can_be_dangerous = dict({})
        for rule in expression.expressions:
            symbols = self.walk(rule)
            can_be_dangerous = self.merge_dicts(can_be_dangerous, symbols)

        return can_be_dangerous

    @add_match(FunctionApplication(Constant, ...))
    def warded_function_constant(self, expression):
        symbols = OrderedSet()
        for arg in expression.args:
            temp = self.walk(arg)
            symbols |= temp

        return symbols

    @add_match(FunctionApplication)
    def warded_function_application(self, expression):
        symbols = OrderedSet()
        for arg in expression.args:
            symbol = self.walk(arg)
            symbols |= symbol

        return symbols

    @add_match(Fact)
    def warded_fact(self, expression):
        return dict({})

    @add_match(Implication(ExistentialPredicate, ...))
    def warded_existential(self, expression):
        new_implication = Implication(
            expression.consequent.body, expression.antecedent
        )
        return self.walk(new_implication)

    @add_match(Implication)
    def warded_implication(self, expression):
        antecedent = self.walk(expression.antecedent)
        consequent = self.walk(expression.consequent)

        free_vars = antecedent._set.symmetric_difference(consequent._set)
        can_be_dangerous = dict({})
        for var in free_vars:
            if var in consequent:
                position = self.calc_position(var, expression.consequent)
                can_be_dangerous = self.merge_position(can_be_dangerous, position)

        return can_be_dangerous

    @add_match(Symbol)
    def warded_symbol(self, expression):
        return OrderedSet(expression.name)

    @add_match(Constant)
    def warded_constant(self, expression):
        pass

    def calc_position(self, var, expression):
        for exp in expression_iterator(expression):
            if var in exp[1].args:
                return dict({exp[1].functor: exp[1].args.index(var)})

    def merge_position(self, to_update_dic, new_dict):
        for key, value in new_dict.items():
            if key in to_update_dic:
                old_values = to_update_dic[key]
                old_values.add(value)
                to_update_dic[key] = old_values
            else:
                to_update_dic[key] = set([value])

        return to_update_dic

    def merge_dicts(self, to_update_dic, new_dict):
        for key, value in new_dict.items():
            if key in to_update_dic:
                old_values = to_update_dic[key]
                old_values.append(value)
                to_update_dic[key] = old_values
            else:
                to_update_dic[key] = [value]

        return to_update_dic


class WardedDatalogDangerousVariableCheck(PatternWalker):
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

    @add_match(Implication(ExistentialPredicate, ...))
    def warded_existential(self, expression):
        new_implication = Implication(
            expression.consequent.body, expression.antecedent
        )
        self.walk(new_implication)

    @add_match(Implication)
    def check_dangerous_implication(self, expression):
        antecedent = self.check_dangerous(expression.antecedent)
        consequent = self.check_dangerous(expression.consequent)

        dangerous_symbol = antecedent.intersection(consequent)
        if len(dangerous_symbol) == 1 and next(
            iter(dangerous_symbol)
        ) in expression.antecedent._symbols:
            var = dangerous_symbol.pop()
            dangerous_pos = self.can_be_dangerous[var].pop()

            dangerous_vars = self.get_name(
                expression.consequent, dangerous_pos
            )
            for dangerous_var in dangerous_vars:
                single_body = self.check_var_single_body(
                    dangerous_var, expression.antecedent
                )
                if not single_body:
                    raise NeuroLangNonWardedException(
                        f'The program is not warded: \
                            there are dangerous variables \
                                outside the ward in {expression.antecedent}'
                    )

    def check_dangerous(self, expression):
        dangerous = set()
        for key in self.can_be_dangerous.keys():
            if key in expression.functor:
                dangerous.add(key)

        if len(dangerous) > 1:
            raise NeuroLangNonWardedException(
                f'The program is not warded: \
                    there are dangerous variables \
                        that appear in more than \
                            one atom of the body in {expression}'
            )

        return dangerous

    def get_name(self, expression, position):
        return [expression.args[index] for index in position]

    def check_var_single_body(self, var, expression):
        founded = False
        for exp in expression_iterator(expression):
            if exp[0] is 'args' and not founded and var in exp[1]:
                founded = True
            elif exp[0] is 'args' and founded and var in exp[1]:
                return False

        return True
