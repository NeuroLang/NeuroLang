from .expressions import Expression, Symbol, NeuroLangException


class ExistentialRule(Expression):
    def __init__(self, eq_variables, statement):
        # if only one variable given, make a tuple
        if isinstance(eq_variables, Symbol): eq_variables = (eq_variables, )
        if not all(isinstance(v, Symbol) for v in eq_variables):
            raise NeuroLangException('E-Quantified variables must be symbols')
        if any(v in statement.rhs._symbols for v in eq_variables):
            raise NeuroLangException(
                'E-Quantified variables cannot occur in rhs'
            )
        if not all(v in statement.lhs._symbols for v in eq_variables):
            raise NeuroLangException(
                'All E-Quantified variables must occur in lhs'
            )
        self.eq_variables = eq_variables
        self.statement = statement

    def __repr__(self):
        eq_var_str = '\u2203 {}'.format(
            ', '.join([v.name for v in self.eq_variables])
        )
        return f'ExistentialRule({eq_var_repr}, {statement_repr})'
