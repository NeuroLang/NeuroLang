from uuid import uuid1
from typing import Tuple

from .expressions import NeuroLangException
from .expressions import (
    Expression, ExpressionBlock, FunctionApplication, Symbol, Constant,
    Definition, ExistentialPredicate
)
from .expression_walker import ExpressionBasicEvaluator
from .expression_pattern_matching import add_match
from .existential_datalog import (
    Implication, ExistentialDatalog, SolverNonRecursiveExistentialDatalog
)


class DeltaSymbol(Symbol):
    def __init__(self, dist_name, n_terms):
        self.dist_name = dist_name
        self.n_terms = n_terms
        super().__init__('Result')

    def __repr__(self):
        return f'Δ-Symbol{{{self.name}({self.dist_name}, {self.n_terms})}}'

    def __hash__(self):
        return hash((self.dist_name, self.n_terms))


class DeltaTerm(Expression):
    def __init__(self, dist_name, *dist_parameters):
        self.dist_name = dist_name
        self.dist_parameters = dist_parameters
        if len(self.dist_parameters) > 0:
            self._symbols = set.union(
                *(p._symbols for p in self.dist_parameters)
            )
        else:
            self._symbols = set()

    def __eq__(self, other):
        return (
            super().__eq__(other) and self.dist_name == other.dist_name and
            self.dist_parameters == other.dist_parameters
        )

    def __repr__(self):
        return f'Δ-term{{{self.dist_name}({self.dist_parameters})}}'


class DeltaAtom(Definition):
    def __init__(self, name, terms):
        if not isinstance(terms, Tuple):
            raise NeuroLangException('Expected terms to be a tuple')
        delta_terms = [t for t in terms if isinstance(t, DeltaTerm)]
        if len(delta_terms) != 1:
            raise NeuroLangException(
                'A Δ-atom must contain one and only one Δ-term'
            )
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
        self._symbols = set.union(*(term._symbols for term in self.terms))

    @property
    def delta_term(self):
        return next(t for t in self.terms if isinstance(t, DeltaTerm))

    def __repr__(self):
        terms_str = ', '.join(repr(t) for t in self.terms)
        return f'Δ-atom{{{self.name}({terms_str})}}'


class GenerativeDatalogSugarRemover(ExpressionBasicEvaluator):
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
                    expression.consequent.args
                ), expression.antecedent
            )
        )


class GenerativeDatalog(GenerativeDatalogSugarRemover, ExistentialDatalog):
    @add_match(Implication(DeltaAtom, ...))
    def add_gdatalog_rule_to_symbol_table(self, expression):
        """
        Add a definition of a GDatalog[Δ] rule to the GDB (Generative
        Database).

        """
        consequent_name = get_gd_rule_consequent_name(expression)
        if consequent_name in self.symbol_table:
            block = self.symbol_table[consequent_name]
            if consequent_name in self.intensional_database():
                raise NeuroLangException(
                    f"`{consequent_name}' already defined in IDB"
                )
            elif consequent_name in self.existential_intensional_database():
                raise NeuroLangException(
                    f"`{consequent_name}' already defined in E-IDB"
                )
            elif consequent_name in self.extensional_database():
                raise NeuroLangException(
                    f"`{consequent_name}' already defined in EDB"
                )
            new_block = add_to_expression_block(block, expression)
            self.symbol_table[consequent_name] = new_block
        else:
            self.symbol_table[consequent_name] = \
                ExpressionBlock((expression, ))
        return expression

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


class TranslateGDatalogToEDatalog(ExpressionBasicEvaluator):
    @add_match(
        ExpressionBlock,
        lambda block: any(
            is_gdatalog_rule(e) or is_gdatalog_rule_wannabe(e)
            for e in block.expressions
        )
    )
    def convert_expression_block_to_edatalog(self, block):
        expressions = tuple()
        for exp in block.expressions:
            res = self.walk(exp)
            if isinstance(res, ExpressionBlock):
                expressions += res.expressions
            else:
                expressions += (res, )
        return self.walk(ExpressionBlock(expressions))

    @add_match(Implication(DeltaAtom, ...))
    def convert_gdatalog_rule_to_edatalog_rules(self, expression):
        delta_atom = expression.consequent
        y = Symbol[delta_atom.delta_term.type]('y_' + str(uuid1()))
        result_terms = (
            delta_atom.delta_term.dist_parameters +
            (Constant(delta_atom.name), ) +
            tuple(t for t in delta_atom.terms
                  if not isinstance(t, DeltaTerm)) + (y, )
        )
        result_atom = FunctionApplication(
            DeltaSymbol(
                delta_atom.delta_term.dist_name, len(delta_atom.terms)
            ), result_terms
        )
        first_rule = Implication(
            ExistentialPredicate(y, result_atom), expression.antecedent
        )
        second_rule = Implication(
            FunctionApplication(
                Symbol(delta_atom.name),
                tuple(
                    term if not isinstance(term, DeltaTerm) else y
                    for term in delta_atom.terms
                )
            ), expression.antecedent & result_atom
        )
        return self.walk(ExpressionBlock((first_rule, second_rule)))

    @add_match(Expression)
    def other_expressions(self, expression):
        return expression


class SolverNonRecursiveGenerativeDatalog(
    GenerativeDatalog, SolverNonRecursiveExistentialDatalog
):
    pass


def add_to_expression_block(eb, to_add):
    expressions = eb.expressions
    if isinstance(to_add, ExpressionBlock):
        expressions += to_add.expressions
    elif isinstance(to_add, Expression):
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


def is_gdatalog_rule_wannabe(exp):
    return (
        isinstance(exp, Implication) and
        isinstance(exp.consequent, FunctionApplication) and
        sum(isinstance(arg, DeltaTerm) for arg in exp.consequent.args) == 1
    )


def get_gd_rule_consequent_name(gd_rule):
    if not is_gdatalog_rule(gd_rule):
        raise NeuroLangException('Must be a GDatalog[Δ] rule')
    else:
        body = gd_rule.consequent
        while isinstance(body, ExistentialPredicate):
            body = body.body
        if not isinstance(body, DeltaAtom):
            raise NeuroLangException('Consequent core must be a Δ-atom')
        else:
            return body.name
