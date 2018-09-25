from .expressions import NeuroLangException
from .expressions import (
    Expression, ExpressionBlock, FunctionApplication, Symbol, Constant,
    Definition, ExistentialPredicate
)
from .solver_datalog_naive import DatalogBasic, NaiveDatalog
from .expression_pattern_matching import add_match
from .existential_datalog import (
    Implication, NonRecursiveExistentialDatalog,
    SolverNonRecursiveExistentialDatalog
)


class DeltaTerm(Expression):
    def __init__(self, dist_name, dist_parameters, event_signature):
        self.dist_name = dist_name
        self.dist_parameters = dist_parameters
        self.event_signature = event_signature
        self._symbols = set.union(*(e._symbols for e in self.event_signature))

    def repr(self):
        return f'Δ-term {self.name}({self.parameters})'


class DeltaAtom(Definition):
    def __init__(self, name, terms):
        delta_terms = [t for t in terms if isinstance(t, DeltaTerm)]
        if len(delta_terms) != 1:
            raise NeuroLangException('A Δ-atom must contain a single Δ-term')
        if not all(
            not isinstance(t, DeltaTerm) and
            (isinstance(t, Constant) or isinstance(t, Symbol))
            for t in terms
            if not isinstance(t, DeltaTerm)
        ):
            raise NeuroLangException(
                'All terms in a Δ-atom that are not the Δ-term '
                'must be symbols or constants'
            )
        self.name = name
        self.terms = terms
        self.delta_term = next(iter(delta_terms))
        self._symbols = set.union(*(term._symbols for term in self.terms))


class NonRecursiveGenerativeDatalog(NonRecursiveExistentialDatalog):
    @add_match(
        Implication(FunctionApplication, ...),
        lambda expression: any(
            isinstance(arg, DeltaTerm)
            for arg in expression.consequent.args
        )
    )
    def from_fa_to_delta_atom(self, expression):
        """
        Convert to Δ-atom any function application on terms where at
        least one term is a Δ-term (with a parameterized distribution).

        Note
        ----
        A valid Δ-atom must contain one and only one Δ-term.
        An exception is raised if more than one Δ-term is present

        """
        if not isinstance(expression.consequent.functor, Symbol):
            raise NeuroLangException(
                'Cannot get DeltaAtom name if functor not a symbol'
            )
        return self.walk(
            Implication(
                DeltaAtom(
                    expression.consequent.functor.name,
                    expression.consequent.functor.args
                ), expression.antecedent
            )
        )

    @add_match(Implication(DeltaAtom, ...))
    def gdatalog_rule(self, expression):
        """
        Add a definition of a GDatalog[Δ] rule to the GDB (Generative
        Database).

        """
        consequent_name = get_gd_rule_consequent_name(expression)
        if consequent_name in self.symbol_table:
            raise NeuroLangException('GDatalog[Δ] rule already defined')
        else:
            self.symbol_table[consequent_name] = expression

    def generative_database(self):
        return {
            k: v
            for k, v in self.symbol_table.items()
            if (
                isinstance(v, Implication) and
                isinstance(v.consequent, FunctionApplication) and
                any(isinstance(arg, DeltaTerm) for arg in v.consequent.args)
            )
        }


class SolverNonRecursiveGenerativeDatalog(
    SolverNonRecursiveExistentialDatalog, NonRecursiveGenerativeDatalog
):
    pass


def add_to_expression_block(eb, to_add):
    expressions = eb.expressions
    if isinstance(to_add, Expression):
        expressions += (to_add, )
    else:
        if (
            not isinstance(to_add, (list, tuple)) or
            not all(isinstance(e, Expression) for e in to_add)
        ):
            raise NeuroLangException(
                'to_add must be expression or list|tuple of expressions'
            )
        expressions += tuple(to_add)
    return ExpressionBlock(expressions)


def is_gdatalog_rule(exp):
    return (
        isinstance(exp, Implication) and
        isinstance(exp.consequent, DeltaAtom) and
        # check that there is one and only one delta term in consequent
        sum(isinstance(t, DeltaTerm) for t in exp.consequent.terms) == 1
    )


def get_gd_rule_consequent_name(gd_rule):
    if not is_gdatalog_rule:
        raise NeuroLangException('Must be a GDatalog[Δ] rule')
    else:
        body = gd_rule.consequent
        while isinstance(body, ExistentialPredicate):
            body = body.body
        if not isinstance(body, DeltaAtom):
            raise NeuroLangException('Consequent core must be a Δ-atom')
        else:
            return body.name
