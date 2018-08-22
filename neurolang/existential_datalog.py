from .expressions import Statement, ExistentialPredicate, NeuroLangException
from .solver_datalog_naive import NaiveDatalog
from .expression_pattern_matching import add_match

class ExistentialDatalog(NaiveDatalog):
    @add_match(Statement(ExistentialPredicate, ...))
    def existential_predicate_in_head(self, expression):
        eq_variable = expression.lhs.head
        if eq_variable in expression.rhs._symbols:
            raise NeuroLangException(
                '\u2203-quantified variable cannot occur in rhs'
            )
