"""
Set of syntactic sugar processors at the intermediate level.
"""

import operator as op
import typing
from typing import AbstractSet, Any, Callable, DefaultDict

from .... import expression_walker as ew
from .... import expressions as ir
from ....datalog.expression_processing import (
    conjunct_formulas,
    conjunction_needs_reordering,
    extract_logic_atoms,
    extract_logic_free_variables,
    order_conjunction_by_shared_variables,
)
from ....datalog.expressions import AdornedSymbol
from ....exceptions import ForbiddenExpressionError, SymbolNotFoundError
from ....expression_pattern_matching import NeuroLangPatternMatchingNoMatch
from ....expression_walker import ReplaceExpressionWalker, ReplaceSymbolWalker
from ....expressions import Constant, FunctionApplication, Symbol
from ....logic import (
    TRUE,
    Conjunction,
    ExistentialPredicate,
    Implication,
    Negation,
)
from ....logic.transformations import ExtractBoundVariables
from ....probabilistic.expressions import (
    PROB,
    Condition,
    ProbabilisticFact,
    ProbabilisticChoice,
    ProbabilisticQuery,
)
from ....type_system import (
    Unknown,
    get_args,
    get_generic_type,
    is_leq_informative,
)


class Column(ir.Definition):
    def __init__(self, set_symbol, column_position):
        self.set_symbol = set_symbol
        self.column_position = column_position
        self._symbols = (
            self.set_symbol._symbols | self.column_position._symbols
        )


def _has_column_sugar(conjunction, marker):
    return any(
        isinstance(arg, marker)
        for atom in extract_logic_atoms(conjunction)
        for arg in atom.args
    )


class TranslateColumnsToAtoms(ew.PatternWalker):
    """
    Syntactic sugar to handle cases where the first column is used
    as a selector. Specifically cases such as

    >>> Implication(B(z), C(z, Column(A, 1)))

    is transformed to

    >>> Implication(B(z), Conjunction((A(fresh0, fresh1), C(z, fresh1))))

    If this syntactic sugar is on the head, then every atom that, by type
    has two arguments, but only one is used, will be replaced by the case
    where first argument is the sugared element.

    >>> Implication(Column(A, 2), B(Column(A, 2), x)))

    is transformed to

    >>> Implication(A(fresh0, fresh1, fresh2), B(fresh2, x)))
    """

    @ew.add_match(
        ir.FunctionApplication,
        lambda exp: any(isinstance(arg, Column) for arg in exp.args),
    )
    def application_column_sugar(self, expression):
        return self.walk(Conjunction((expression,)))

    @ew.add_match(Conjunction, lambda exp: _has_column_sugar(exp, Column))
    def conjunction_column_sugar(self, expression):
        (
            replacements,
            new_atoms,
        ) = self._obtain_new_atoms_and_column_replacements(expression)

        replaced_expression = ew.ReplaceExpressionWalker(replacements).walk(
            expression
        )
        new_formulas = replaced_expression.formulas + tuple(new_atoms)
        return self.walk(Conjunction(new_formulas))

    @ew.add_match(Implication, lambda exp: _has_column_sugar(exp, Column))
    def implication_column_sugar(self, expression):
        (
            replacements,
            new_atoms,
        ) = self._obtain_new_atoms_and_column_replacements(expression)

        replaced_expression = ew.ReplaceExpressionWalker(replacements).walk(
            expression
        )
        new_antecedent = conjunct_formulas(
            replaced_expression.antecedent, Conjunction(new_atoms)
        )

        return self.walk(
            Implication(replaced_expression.consequent, new_antecedent)
        )

    def _obtain_new_atoms_and_column_replacements(self, expression):
        sugared_columns = self._obtain_sugared_columns(expression)

        new_atoms = []
        replacements = {}
        for k, v in sugared_columns.items():
            k_constant = ew.ReplaceSymbolsByConstants(self.symbol_table).walk(
                k
            )
            args = (
                v.get(Column(k, ir.Constant[int](i)), ir.Symbol.fresh())
                for i in range(k_constant.value.arity)
            )
            new_atoms.append(k(*args))
            replacements.update(v)
        return replacements, tuple(new_atoms)

    def _obtain_sugared_columns(self, expression):
        sugared_columns = DefaultDict(dict)
        for atom in extract_logic_atoms(expression):
            for arg in atom.args:
                if isinstance(arg, Column):
                    sugared_columns[arg.set_symbol][arg] = ir.Symbol.fresh()
        return sugared_columns


class SelectByFirstColumn(ir.Definition):
    def __init__(self, set_symbol, selector):
        self.set_symbol = set_symbol
        self.selector = selector
        self._symbols = self.set_symbol._symbols | self.selector._symbols

    def __repr__(self):
        return f"{self.set_symbol}.{self.selector}"


class TranslateSelectByFirstColumn(ew.PatternWalker):
    """
    Syntactic sugar to handle cases where the first column is used
    as a selector. Specifically cases such as

    >>> Implication(B(z), C(z, SelectByFirstColumn(A, c)))

    is transformed to

    >>> Implication(B(z), Conjunction((A(c, fresh), C(z, fresh))))

    If this syntactic sugar is on the head, then every atom that, by type
    has two arguments, but only one is used, will be replaced by the case
    where first argument is the sugared element.

    >>> Implication(SelectByFirstColumn(A, c), Conjunction((eq(x), B(x))))

    is transformed to

    >>> Implication(A(c, fresh), Conjunction((eq(fresh, x), B(x))))
    """

    @ew.add_match(
        ir.FunctionApplication,
        lambda exp: any(
            isinstance(arg, SelectByFirstColumn) for arg in exp.args
        ),
    )
    def application_column_sugar(self, expression):
        return self.walk(Conjunction((expression,)))

    @ew.add_match(
        Conjunction, lambda exp: _has_column_sugar(exp, SelectByFirstColumn)
    )
    def conjunction_column_sugar(self, expression):
        (
            replacements,
            new_atoms,
        ) = self._obtain_new_atoms_and_select_by_first_column_replacements(
            expression
        )

        replaced_expression = ew.ReplaceExpressionWalker(replacements).walk(
            expression
        )
        new_formulas = replaced_expression.formulas + tuple(new_atoms)
        return self.walk(Conjunction(new_formulas))

    @ew.add_match(Implication(SelectByFirstColumn, ...))
    def implication_select_by_left_head(self, expression):
        consequent = expression.consequent
        head_fresh = ir.Symbol.fresh()
        new_consequent = consequent.set_symbol(consequent.selector, head_fresh)

        replacements = {consequent: new_consequent}
        for atom in extract_logic_atoms(expression.antecedent):
            if len(atom.args) == 1 and self._theoretical_arity(atom) == 2:
                replacements[atom] = atom.functor(head_fresh, atom.args[0])

        new_rule = Implication(
            new_consequent,
            ReplaceExpressionWalker(replacements).walk(expression.antecedent),
        )

        return self.walk(new_rule)

    @ew.add_match(
        Implication, lambda exp: _has_column_sugar(exp, SelectByFirstColumn)
    )
    def implication_column_sugar(self, expression):
        (
            replacements,
            new_atoms,
        ) = self._obtain_new_atoms_and_select_by_first_column_replacements(
            expression
        )

        replaced_expression = ew.ReplaceExpressionWalker(replacements).walk(
            expression
        )
        new_antecedent = conjunct_formulas(
            replaced_expression.antecedent, Conjunction(new_atoms)
        )

        return self.walk(
            Implication(replaced_expression.consequent, new_antecedent)
        )

    def _obtain_new_atoms_and_select_by_first_column_replacements(
        self, expression
    ):
        sugared_columns = DefaultDict(dict)
        for atom in extract_logic_atoms(expression):
            for arg in atom.args:
                if isinstance(arg, SelectByFirstColumn):
                    sugared_columns[arg] = ir.Symbol.fresh()

        new_atoms = []
        for k, v in sugared_columns.items():
            new_atom = k.set_symbol(k.selector, v)
            new_atoms.append(new_atom)
        return sugared_columns, tuple(new_atoms)

    def _theoretical_arity(self, atom):
        functor = atom.functor
        if isinstance(functor, ir.Symbol) and functor.type is Unknown:
            try:
                functor = self.symbol_table[functor]
            except KeyError:
                raise SymbolNotFoundError(f"Symbol {functor} not found")
        if functor.type is Unknown:
            arity = None
        elif is_leq_informative(functor.type, Callable):
            arity = len(get_args(functor.type)[0])
        elif is_leq_informative(functor.type, AbstractSet):
            arity = len(get_args(get_args(functor.type)[0]))
        else:
            arity = None
        return arity


GETATTR = ir.Constant[Callable[[Unknown, str], Unknown]](getattr)


class ConvertAttrSToSelectByColumn(ew.ExpressionWalker):
    """
    Convert terms such as P.s[c]
    or `getattr(P, s)[c]` at `SelectByFirstColumn(P, s)`.
    """

    @ew.add_match(
        FunctionApplication(
            FunctionApplication(GETATTR, (..., Constant[str]("s"))), (...,)
        )
    )
    def conversion(self, expression):
        return self.walk(
            SelectByFirstColumn(expression.functor.args[0], expression.args[0])
        )


class RecogniseSSugar(ew.PatternWalker):
    """
    Recognising datalog terms such as P.s[c]
    or `getattr(P, s)[c]`.
    """

    @ew.add_match(Constant)
    def constant(self, expression):
        return False

    @ew.add_match(Symbol)
    def symbol(self, expression):
        return False

    @ew.add_match(
        FunctionApplication(
            FunctionApplication(GETATTR, (..., Constant[str]("s"))), (...,)
        )
    )
    def s_sugar(self, expression):
        return True

    @ew.add_match(...)
    def others(self, expression):
        params = list(expression.unapply())
        while params:
            param = params.pop()
            if isinstance(param, tuple):
                params += param
            else:
                if self.walk(param):
                    return True
        return False


_RECOGNISE_S_SUGAR = RecogniseSSugar()


class TranslateSSugarToSelectByColumn(ew.PatternWalker):
    """
    Syntactic sugar to convert datalog terms P.s[c] to
    SelectByFirstColumn(P, s).
    """

    _convert_attr_s__to_SelectByColumn = ConvertAttrSToSelectByColumn()

    @ew.add_match(Implication, _RECOGNISE_S_SUGAR.walk)
    def replace_s_getattr_by_first_column(self, expression):
        new_expression = (
            TranslateSSugarToSelectByColumn
            ._convert_attr_s__to_SelectByColumn
            .walk(
                expression
            )
        )
        if new_expression is not expression:
            new_expression = self.walk(new_expression)
        return new_expression


EQ = Constant(op.eq)


class TranslateHeadConstantsToEqualities(ew.PatternWalker):
    """
    Syntactic sugar to convert datalog rules having constants
    in the head to having equalities in the body.
    """

    @ew.add_match(
        Implication,
        lambda imp: any(
            isinstance(arg, ir.Constant) for arg in imp.consequent.args
        )
        and (imp.antecedent != TRUE),
    )
    def head_constants_to_equalities(self, expression):
        new_equalities = {}
        new_args = tuple()
        for arg in expression.consequent.args:
            if isinstance(arg, ir.Constant):
                new_param = ir.Symbol.fresh()
                new_equalities[new_param] = arg
                arg = new_param
            new_args += (arg,)
        new_consequent = expression.consequent.functor(*new_args)
        new_antecedent = Conjunction(
            tuple(
                [expression.antecedent]
                + [EQ(k, v) for k, v in new_equalities.items()]
            )
        )

        new_implication = Implication(new_consequent, new_antecedent)

        return self.walk(new_implication)


def _as_conjuncts(expression):
    """Top-level conjuncts of `expression`: its formulas if it is a
    Conjunction, or the single-element list [expression] otherwise.
    """
    if isinstance(expression, Conjunction):
        return list(expression.formulas)
    return [expression]


def _rebuild_conjunction(conjuncts):
    if len(conjuncts) == 1:
        return conjuncts[0]
    return Conjunction(tuple(conjuncts))


def delegate_to_next_match(walker, expression, skip_action):
    """Re-dispatch `expression` through `walker`'s pattern list, skipping
    `skip_action` (the match currently executing). Lets a handler that
    decided not to transform an expression hand it to whichever pattern
    would have matched next, instead of returning it unchanged (which
    would re-trigger its own guard forever) or re-walking it (same
    problem). Shared by every mixin in this package that needs this
    "decline and fall through" behavior -- see also
    `TranslateRegionDestroy._delegate_to_next_match`'s former local copy
    in spatial.py, now using this one.
    """
    for pattern, guard, action in walker.patterns:
        if action is skip_action:
            continue
        if walker.pattern_match(pattern, expression) and (
            guard is None or guard(expression)
        ):
            return action(walker, expression)
    raise NeuroLangPatternMatchingNoMatch(f"No match for {expression}")


def _has_distinguished_variable_negation(impl):
    """True iff `impl`'s antecedent is a Condition whose conditioned or
    conditioning side has, at its top level, a Negation literal sharing a
    free variable with the implication's consequent (a "distinguished"
    variable -- one that stays free in the query's result).
    """
    if not isinstance(impl.antecedent, Condition):
        return False
    head_vars = extract_logic_free_variables(impl.consequent)
    for side in (impl.antecedent.conditioned, impl.antecedent.conditioning):
        for conjunct in _as_conjuncts(side):
            if isinstance(conjunct, Negation):
                inner_vars = extract_logic_free_variables(conjunct.formula)
                if inner_vars & head_vars:
                    return True
    return False


class TranslateProbabilisticQueryMixin(ew.PatternWalker):
    """
    Hoist a negated literal whose argument is a free "distinguished"
    variable (shared with the enclosing query head) out of a
    probabilistic conditional query, into a plain deterministic rule,
    before `rewrite_conditional_query` below ever sees it.

    `rewrite_conditional_query` builds a numerator within-language
    query by flatly conjoining the conditioned and conditioning sides
    of a `//` query (`F1(x, y, z, PROB) :- Q(x) & R(y) & T(z)`). When
    one of those conjuncts is a negated literal whose argument
    includes a distinguished variable (e.g. `~Active(s, r)` with `r`
    free in the query head), that numerator query returns an
    incorrect probability -- confirmed wrong even as a bare SUCC
    query with no conditional-probability machinery involved, and
    independent of the WMC-vs-lifted-solver choice. The denominator
    query is unaffected.

    The fix: compute the negation as a plain deterministic rule
    first (no probabilistic atom in scope yet, so the distinguished
    variable is still a free Datalog variable, not yet part of any
    probabilistic computation), then substitute the resulting
    positive relation back into the query.

    These methods are defined here, ahead of `rewrite_conditional_query`
    in this same class, rather than in a separate mixin composed as a
    base class -- pattern precedence in this framework follows
    `type(self).mro()`, which checks a *subclass's own* patterns
    before any it inherits, so a separate base-class mixin would run
    *after* `rewrite_conditional_query`'s always-true guard, never
    getting a chance to fire. Defining the hoist here instead (the
    same precedence mechanism already used by `lift_ep_from_conditioned`
    / `lift_ep_from_conditioning` below) gives it priority, and
    guarantees every consumer of `TranslateProbabilisticQueryMixin`
    gets the fix with no separate opt-in -- unlike composing a sibling
    mixin that a call site could forget to include in the right place.
    """

    @ew.add_match(
        Implication(..., Condition),
        _has_distinguished_variable_negation,
    )
    def hoist_negated_distinguished_variable_literal(self, impl):
        head_vars = extract_logic_free_variables(impl.consequent)
        condition = impl.antecedent
        hoisted_rules = []
        new_conditioned = self._hoist_side(
            condition.conditioned, head_vars, hoisted_rules
        )
        new_conditioning = self._hoist_side(
            condition.conditioning, head_vars, hoisted_rules
        )
        if not hoisted_rules:
            # Every negated distinguished-variable literal found by the
            # guard turned out to be unsafe to hoist (no deterministic
            # domain relation available for it in this conjunction -- see
            # _hoist_side). The antecedent is therefore unchanged, and
            # re-walking it would just re-match this same guard forever.
            # Defer to whatever the rest of the MRO does with it as-is.
            return delegate_to_next_match(
                self,
                impl,
                type(self).hoist_negated_distinguished_variable_literal,
            )
        new_impl = self.walk(
            Implication(
                impl.consequent, Condition(new_conditioned, new_conditioning)
            )
        )
        return tuple(hoisted_rules) + (new_impl,)

    def _hoist_side(self, side, head_vars, hoisted_rules):
        conjuncts = _as_conjuncts(side)
        positive = [c for c in conjuncts if not isinstance(c, Negation)]
        # A hoisted rule must stay purely deterministic -- including a
        # probabilistic atom (e.g. the uniform choice over studies) as a
        # range-restrictor would reintroduce a probabilistic dependency
        # into what must be resolved *before* any probabilistic atom is
        # involved, defeating the fix.
        prob_symbs = self._safe_probabilistic_predicate_symbols()
        deterministic_positive = [
            p for p in positive if p.functor not in prob_symbs
        ]
        new_conjuncts = list(positive)
        for conjunct in conjuncts:
            if not isinstance(conjunct, Negation):
                continue
            inner_vars = extract_logic_free_variables(conjunct.formula)
            if not (inner_vars & head_vars):
                # No distinguished variable involved: this negation is
                # not the pattern that triggers the bug, leave it as-is.
                new_conjuncts.append(conjunct)
                continue
            # Range-restrict the hoisted rule with whatever DETERMINISTIC
            # positive atoms on this same side already mention the
            # negated literal's free variables, so the rule stays
            # safe-range without depending on a probabilistic atom.
            restrictors = [
                p
                for p in deterministic_positive
                if extract_logic_free_variables(p) & inner_vars
            ]
            restricted_vars = set()
            for p in restrictors:
                restricted_vars |= extract_logic_free_variables(p)
            if not (inner_vars <= restricted_vars):
                # Not every free variable of the negated literal can be
                # range-restricted deterministically on this side (e.g.
                # the only candidate restrictor is itself probabilistic).
                # Hoisting would produce an unsafe rule, so leave this
                # negation untouched rather than emit something wrong;
                # the pre-existing (buggy) behavior is unchanged for this
                # case, which is reported separately as a documented
                # engine limitation when it is hit in practice.
                new_conjuncts.append(conjunct)
                continue
            fresh_functor = Symbol.fresh()
            fresh_args = tuple(
                sorted(inner_vars, key=lambda v: v.name)
            )
            new_head = fresh_functor(*fresh_args)
            new_body = _rebuild_conjunction([conjunct] + restrictors)
            hoisted_rules.append(self.walk(Implication(new_head, new_body)))
            new_conjuncts.append(new_head)
        return _rebuild_conjunction(new_conjuncts)

    def _safe_probabilistic_predicate_symbols(self):
        """self.pfact_pred_symbs / pchoice_pred_symbs raise KeyError when no
        probabilistic fact (resp. choice) has been registered yet in this
        program -- their backing EDB relation symbol was never added to the
        symbol table. Treat that as "no predicates of that kind" rather
        than letting the lookup fail.
        """
        symbs = set()
        for attr in ("pfact_pred_symbs", "pchoice_pred_symbs"):
            try:
                symbs |= set(getattr(self, attr))
            except (KeyError, AttributeError):
                pass
        return symbs

    @ew.add_match(
        Implication(..., Conjunction),
        lambda impl: order_conjunction_by_shared_variables(
            impl.antecedent.formulas
        ) != impl.antecedent.formulas,
    )
    def order_conjunction_for_join(self, impl):
        """
        Order implication-conjunction bodies to avoid cross products.

        Relational algebra plans evaluate a conjunction as a left-deep
        natural join in atom order; joining two atoms that share no
        variable first materialises their cross product.  The order is
        only changed when the current one contains such a cross product
        (see ``conjunction_needs_reordering``), so already-efficient
        conjunctions are left untouched and handled by the other rules.
        """
        return self.walk(
            impl.apply(
                impl.consequent,
                Conjunction(
                    order_conjunction_by_shared_variables(
                        impl.antecedent.formulas
                    )
                ),
            )
        )

    @ew.add_match(Implication(..., Condition(ExistentialPredicate(..., ...), ...)))
    def lift_ep_from_conditioned(self, impl):
        condition = impl.antecedent
        ep = condition.conditioned
        other_side = condition.conditioning
        other_bound = ExtractBoundVariables().walk(other_side)
        if ep.head in other_bound:
            fresh_var = Symbol[ep.head.type].fresh()
            fresh_ep = ReplaceSymbolWalker({ep.head.name: fresh_var}).walk(ep)
            lifted_ep = fresh_ep
        else:
            lifted_ep = ep
        lifted = ExistentialPredicate(
            lifted_ep.head,
            Condition(lifted_ep.body, condition.conditioning)
        )
        return self.walk(Implication(impl.consequent, lifted))

    @ew.add_match(Implication(..., Condition(..., ExistentialPredicate(..., ...))))
    def lift_ep_from_conditioning(self, impl):
        condition = impl.antecedent
        ep = condition.conditioning
        other_side = condition.conditioned
        other_bound = ExtractBoundVariables().walk(other_side)
        if ep.head in other_bound:
            fresh_var = Symbol[ep.head.type].fresh()
            fresh_ep = ReplaceSymbolWalker({ep.head.name: fresh_var}).walk(ep)
            lifted_ep = fresh_ep
        else:
            lifted_ep = ep
        lifted = ExistentialPredicate(
            lifted_ep.head,
            Condition(condition.conditioned, lifted_ep.body)
        )
        return self.walk(Implication(impl.consequent, lifted))

    @ew.add_match(
        Implication(
            ..., FunctionApplication(Constant[typing.Any](op.floordiv), ...)
        )
    )
    def conditional_query(self, implication):
        new_implication = Implication(
            implication.consequent, Condition(*implication.antecedent.args)
        )
        if not (
            any(
                isinstance(arg, ProbabilisticQuery)
                or (
                    get_generic_type(type(arg)) is FunctionApplication
                    and arg.functor == PROB
                )
                for arg in new_implication.consequent.args
            )
        ):
            raise ForbiddenExpressionError(
                "Missing probabilistic term in consequent's head"
            )
        if (
            sum(
                isinstance(arg, ProbabilisticQuery)
                for arg in new_implication.consequent.args
            )
            > 1
        ):
            raise ForbiddenExpressionError(
                "Can only contain one probabilistic term in consequent's head"
            )
        return self.walk(new_implication)

    @ew.add_match(
        Implication,
        lambda implication: isinstance(
            implication.consequent, FunctionApplication
        )
        and any(
            get_generic_type(type(arg)) is FunctionApplication
            and arg.functor == PROB
            for arg in implication.consequent.args
        ),
    )
    def within_language_prob_query(self, implication):
        csqt_args = tuple()
        for arg in implication.consequent.args:
            if (
                get_generic_type(type(arg)) is FunctionApplication
                and arg.functor == PROB
            ):
                arg = ProbabilisticQuery(*arg.unapply())
            csqt_args += (arg,)
        consequent = implication.consequent.functor(*csqt_args)
        return self.walk(Implication(consequent, implication.antecedent))

    @ew.add_match(
        Implication(..., Condition),
        lambda implication: len(
            extract_logic_free_variables(implication.consequent)
            & extract_logic_free_variables(implication.antecedent.conditioning)
        )
        >= 0,
    )
    def rewrite_conditional_query(self, impl):
        """
        Translate an expression of the form
            P(x, y, z) :- Q(x) & R(y) // T(z)

        to its equivalent expression
            F1(x, y, z, PROB) :- Q(x) & R(y) &T(z)
            F2(z, PROB) :- T(z)
            P(x, y, z, p) :- F1(x, y, z, p1) & F2(z, p2) & (p == p1 / p2)
        """
        left = impl.antecedent.conditioned
        right = impl.antecedent.conditioning

        head_args = extract_logic_free_variables(impl.consequent)
        num_head = AdornedSymbol(
            impl.consequent.functor, "cond_num", None
        )(*impl.consequent.args)
        num_body = conjunct_formulas(left, right)
        new_num = self.walk(
            Implication(
                num_head,
                num_body,
            )
        )

        new_denum_args = extract_logic_free_variables(right) & head_args
        new_denum_prob_arg = ProbabilisticQuery(PROB, tuple(new_denum_args))
        new_denum_args.add(new_denum_prob_arg)
        denum_head = AdornedSymbol(
            impl.consequent.functor, "cond_den", None
        )(*new_denum_args)
        new_denum = self.walk(
            Implication(denum_head, right)
        )

        p = Symbol.fresh()
        p1 = Symbol.fresh()
        p2 = Symbol.fresh()
        new_impl = self.walk(
            Implication(
                self._replace_prob_query_arg_by_var(impl.consequent, p),
                Conjunction(
                    (
                        self._replace_prob_query_arg_by_var(
                            num_head, p1
                        ),
                        self._replace_prob_query_arg_by_var(
                            denum_head, p2
                        ),
                        EQ(
                            p,
                            Constant(op.truediv)(p1, p2),
                        ),
                    )
                ),
            )
        )

        return (new_num, new_denum, new_impl)

    def _replace_prob_query_arg_by_var(self, expr, var):
        new_args = tuple(
            arg if not isinstance(arg, ProbabilisticQuery) else var
            for arg in expr.args
        )
        return expr.functor(*new_args)


class TranslateQueryBasedProbabilisticFactMixin(ew.PatternWalker):
    """
    Translate an expression of the form

        (P @ y)(x) :- Q(x)

    to its equivalent query-based probabilistic fact

        P(x) : y :- Q(x)

    This is useful when the rule was defined at the frontend level using the
    sugar syntax `(P @ y)[x] = Q[x]`.

    """

    @ew.add_match(
        Implication(
            FunctionApplication(
                FunctionApplication(
                    Constant[Callable[[Any, Any], Any]](op.matmul),
                    ...,
                ),
                ...,
            ),
            ...,
        ),
    )
    def query_based_probfact_wannabe(self, impl):
        pred_symb, probability = impl.consequent.functor.args
        body = pred_symb(*impl.consequent.args)
        new_consequent = ProbabilisticFact(probability, body)
        return self.walk(Implication(new_consequent, impl.antecedent))

    @ew.add_match(
        Implication(
            FunctionApplication(
                FunctionApplication(
                    Constant[Callable[[Any, Any], Any]](op.xor),
                    ...,
                ),
                ...,
            ),
            ...,
        ),
    )
    def query_based_probchoice_wannabe(self, impl):
        pred_symb, probability = impl.consequent.functor.args
        body = pred_symb(*impl.consequent.args)
        new_consequent = ProbabilisticChoice(probability, body)
        return self.walk(Implication(new_consequent, impl.antecedent))
