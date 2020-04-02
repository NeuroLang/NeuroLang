from functools import reduce
import operator
from . import (
    Symbol,
    Constant,
    FunctionApplication,
    Implication,
    Union,
    Conjunction,
    Disjunction,
    Negation,
    UniversalPredicate,
    ExistentialPredicate,
    LogicOperator,
    Quantifier,
)
from ..datalog.expressions import Fact
from ..exceptions import NeuroLangException
from ..expressions import ExpressionBlock
from ..expression_walker import (
    add_match,
    PatternWalker,
    ChainedWalker,
)
from .expression_processing import extract_logic_free_variables


from .transformations import (
    LogicExpressionWalker,
    EliminateImplications,
    MoveNegationsToAtoms,
    MoveQuantifiersUp,
    DesambiguateQuantifiedVariables,
    DistributeDisjunctions,
    CollapseDisjunctions,
    CollapseConjunctions,
    RemoveUniversalPredicates,
)


class NeuroLangTranslateToHornClauseException(NeuroLangException):
    pass


class HornClause(LogicOperator):
    """
    Expression of the form `P(X) \u2190 (Q(X) \u22C0 ... \u22C0 S(X))`
    """

    def __init__(self, head, body=None):
        self._validate(head, body)
        self.head = head
        self.body = body
        self._symbols = head._symbols or set()
        if body is not None:
            for l in body:
                self._symbols |= l._symbols

    def _validate(self, head, body):
        if not (head is None or self._is_literal(head)):
            raise NeuroLangException(
                f"Head must be a literal or None, {head} given"
            )
        if not head and not body:
            raise NeuroLangException(f"Head and body can not both be empty")
        if not (
            body is None
            or (
                isinstance(body, tuple)
                and all(self._is_literal(l) for l in body)
            )
        ):
            raise NeuroLangException(
                f"Body must be a tuple of literals or None, {body} given"
            )

    def _is_literal(self, exp):
        return (
            isinstance(exp, FunctionApplication)
            or isinstance(exp, Symbol)
            or isinstance(exp, Constant)
            or (
                isinstance(exp, Negation)
                and (
                    isinstance(exp.formula, FunctionApplication)
                    or isinstance(exp.formula, Symbol)
                    or isinstance(exp.formula, Constant)
                )
            )
        )

    def __repr__(self):
        r = "HornClause{"
        if self.head is not None:
            r += repr(self.head)
            if self.body is not None:
                r += " :- "
        else:
            r += "?- "

        if self.body is not None:
            r += ", ".join(repr(l) for l in self.body)
        r += ".}"
        return r


class MoveNegationsToAtomsOrExistentialQuantifiers(MoveNegationsToAtoms):
    @add_match(Negation(ExistentialPredicate(..., ...)))
    def negated_existential(self, negation):
        quantifier = negation.formula
        h = quantifier.head
        b = self.walk(quantifier.body)
        exp = Negation(ExistentialPredicate(h, b))
        if b != quantifier.body:
            exp = self.walk(exp)
        return exp


def convert_to_srnf(e):
    w = ChainedWalker(
        DesambiguateQuantifiedVariables,
        EliminateImplications,
        RemoveUniversalPredicates,
        MoveNegationsToAtomsOrExistentialQuantifiers,
    )
    return w.walk(e)


def is_safe_range(expression):
    try:
        return extract_logic_free_variables(
            expression
        ) == range_restricted_variables(expression)
    except NeuroLangException:
        return False


def range_restricted_variables(e):
    return RangeRestrictedVariables().walk(e)


class RangeRestrictedVariables(LogicExpressionWalker):
    @add_match(FunctionApplication)
    def match_function(self, exp):
        return set([a for a in exp.args if isinstance(a, Symbol)])

    @add_match(Negation)
    def match_negation(self, exp):
        self.walk(exp.formula)
        return set()

    @add_match(Disjunction)
    def match_disjunction(self, exp):
        return set.intersection(*self.walk(exp.formulas))

    @add_match(Conjunction)
    def match_conjunction(self, exp):
        return set.union(*self.walk(exp.formulas))

    @add_match(ExistentialPredicate)
    def match_existential(self, exp):
        r = self.walk(exp.body)
        if exp.head in r:
            return r - {exp.head}
        # This better could be a return value
        raise NeuroLangException(
            "Some quantified variable is not range restricted"
        )


def convert_srnf_to_horn_clauses(head, expression):
    if not is_safe_range(expression):
        raise NeuroLangTranslateToHornClauseException(
            "Expression is not safe range: {}".format(expression)
        )
    if set(head.args) != range_restricted_variables(expression):
        raise NeuroLangTranslateToHornClauseException(
            "Variables in head ({}) must be present in body ({})".format(
                head, expression
            )
        )

    queue = [(head, expression)]
    processed = []

    while queue:
        head, exp = queue.pop()
        body, remainder = ConvertSRNFToHornClause().walk(exp)
        processed.append(HornClause(head, body))
        queue = remainder + queue

    return Union(tuple(reversed(processed)))


class ConvertSRNFToHornClause(PatternWalker):
    @add_match(Conjunction)
    def match_conjunction(self, exp):
        bodies, remainders = zip(*map(self.walk, exp.formulas))
        return (
            tuple(
                sorted(reduce(operator.add, bodies), key=self._negation_order)
            ),
            reduce(operator.add, remainders),
        )

    def _negation_order(self, exp):
        return 1 if isinstance(exp, Negation) else 0

    @add_match(Disjunction)
    def match_disjunction(self, exp):
        nh = self._new_head_for(exp)
        return (nh,), [(nh, f) for f in exp.formulas]

    @add_match(ExistentialPredicate)
    def match_existential(self, exp):
        return self.walk(exp.body)

    @add_match(Negation(FunctionApplication))
    def match_negated_atom(self, exp):
        return (exp,), []

    @add_match(Negation(ExistentialPredicate))
    def match_negated_existential(self, exp):
        nh = self._new_head_for(exp.formula)
        return (Negation(nh),), [(nh, exp.formula)]

    @add_match(FunctionApplication)
    def match_atom(self, exp):
        return (exp,), []

    @add_match(...)
    def match_unknown(self, exp):
        raise NeuroLangTranslateToHornClauseException(
            "Expression not in safe range normal form: {}".format(exp)
        )

    def _new_head_for(self, exp):
        fv = extract_logic_free_variables(exp)
        S = Symbol.fresh()
        return S(*tuple(fv))


def translate_horn_clauses_to_datalog(horn_clauses):
    return TranslateHornClausesToDatalog().walk(horn_clauses)


class TranslateHornClausesToDatalog(PatternWalker):
    @add_match(Union)
    def match_union(self, exp):
        return ExpressionBlock(self.walk(exp.formulas))

    @add_match(Quantifier)
    def match_quantifier(self, exp):
        return self.walk(exp.body)

    @add_match(HornClause, lambda e: not e.body)
    def match_horn_fact(self, exp):
        return Fact(exp.head)

    @add_match(HornClause, lambda e: e.body and len(e.body) == 1)
    def match_horn_rule_1(self, exp):
        return Implication(exp.head, self.walk(exp.body[0]))

    @add_match(HornClause, lambda e: e.body and len(e.body) > 1)
    def match_horn_rule_2(self, exp):
        return Implication(exp.head, Conjunction(self.walk(exp.body)))

    @add_match(Negation)
    def match_negation(self, exp):
        return Negation(self.walk(exp.formula))

    @add_match(FunctionApplication)
    def match_fa(self, exp):
        return exp

    @add_match(Symbol)
    def match_symbol(self, exp):
        return exp


def fol_query(head, exp):
    exp = convert_to_srnf(exp)
    horn_clauses = convert_srnf_to_horn_clauses(head, exp)
    program = translate_horn_clauses_to_datalog(horn_clauses)
    return program
