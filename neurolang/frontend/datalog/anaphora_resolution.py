from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker, ReplaceSymbolWalker
from ...expressions import FunctionApplication, Symbol
from ...logic import AnaphoraPredicate, Conjunction, ExistentialPredicate
from ...probabilistic.expressions import Condition


def _collect_anaphoras(expr, result):
    if isinstance(expr, AnaphoraPredicate):
        result.append((expr.head, expr.noun_name.name))
        return
    if isinstance(expr, Conjunction):
        for f in expr.formulas:
            _collect_anaphoras(f, result)
    elif hasattr(expr, 'body') and not isinstance(expr, Condition):
        _collect_anaphoras(expr.body, result)


def _find_matching_noun_var(expr, noun_name):
    if isinstance(expr, FunctionApplication):
        if (
            isinstance(expr.functor, Symbol)
            and expr.functor.name == noun_name
            and len(expr.args) > 0
            and isinstance(expr.args[0], Symbol)
        ):
            return expr.args[0]
        return None
    if isinstance(expr, Conjunction):
        for f in expr.formulas:
            result = _find_matching_noun_var(f, noun_name)
            if result is not None:
                return result
        return None
    if hasattr(expr, 'body'):
        return _find_matching_noun_var(expr.body, noun_name)
    return None


def _strip_anaphora_markers(expr, resolved_heads=None):
    if isinstance(expr, AnaphoraPredicate):
        body = _strip_anaphora_markers(expr.body, resolved_heads)
        if resolved_heads is not None and expr.head in resolved_heads:
            return body
        return ExistentialPredicate(expr.head, body)
    if isinstance(expr, Conjunction):
        new_formulas = tuple(
            _strip_anaphora_markers(f, resolved_heads) for f in expr.formulas
        )
        return Conjunction(new_formulas)
    if hasattr(expr, 'body') and hasattr(expr, 'head'):
        new_head = expr.head
        new_body = _strip_anaphora_markers(expr.body, resolved_heads)
        return expr.apply(new_head, new_body)
    if hasattr(expr, 'formulas'):
        new_formulas = tuple(
            _strip_anaphora_markers(f, resolved_heads) for f in expr.formulas
        )
        return expr.apply(new_formulas)
    return expr


def _unpack_existential(expr, head_symbol):
    """Remove an ExistentialPredicate(head=head_symbol) and flatten its body.

    When a variable referenced by an anaphora on the conditioning side is
    bound by an existential on the conditioned side, that quantifier must
    be lifted to scope over the entire Condition. This strips it from the
    conditioned side, merging the body into the parent conjunction.
    """
    if isinstance(expr, ExistentialPredicate) and expr.head == head_symbol:
        return expr.body
    if isinstance(expr, Conjunction):
        new_formulas = []
        for f in expr.formulas:
            if isinstance(f, ExistentialPredicate) and f.head == head_symbol:
                if isinstance(f.body, Conjunction):
                    new_formulas.extend(f.body.formulas)
                else:
                    new_formulas.append(f.body)
            else:
                new_formulas.append(_unpack_existential(f, head_symbol))
        if not new_formulas:
            return None
        if len(new_formulas) == 1:
            return new_formulas[0]
        return Conjunction(tuple(new_formulas))
    if hasattr(expr, 'body'):
        return expr.apply(
            expr.head, _unpack_existential(expr.body, head_symbol)
        )
    return expr


def _has_anaphora(expr):
    if isinstance(expr, AnaphoraPredicate):
        return True
    if isinstance(expr, Conjunction):
        return any(_has_anaphora(f) for f in expr.formulas)
    if hasattr(expr, 'body'):
        return _has_anaphora(expr.body)
    if hasattr(expr, 'formulas'):
        return any(_has_anaphora(f) for f in expr.formulas)
    if hasattr(expr, 'conditioned'):
        return _has_anaphora(expr.conditioned) or _has_anaphora(expr.conditioning)
    return False


def _anaphora_guard(expression):
    return _has_anaphora(expression.conditioned) or _has_anaphora(expression.conditioning)


class AnaphoraResolutionWalker(ExpressionWalker):
    """Resolve 'the X' anaphora across Condition boundaries.

    When 'the X' appears on the conditioning side of 'given' and X was
    not in symbol scope during parsing, it produces an AnaphoraPredicate
    node. This walker finds those markers on the conditioning side, looks
    for a matching noun predicate on the conditioned side, and unifies
    the variables by replacing the conditioning-side symbol with the
    conditioned-side one.
    """

    @add_match(Condition, _anaphora_guard)
    def resolve_condition_anaphora(self, expression):
        anaphoras = []
        _collect_anaphoras(expression.conditioning, anaphoras)

        new_conditioned = self.walk(expression.conditioned)
        new_conditioning = self.walk(expression.conditioning)

        replacement = {}
        for head_sym, noun_name in anaphoras:
            match = _find_matching_noun_var(new_conditioned, noun_name)
            if match is not None:
                replacement[head_sym] = match

        if replacement:
            new_conditioning = ReplaceSymbolWalker(replacement).walk(
                new_conditioning
            )

        new_conditioning = _strip_anaphora_markers(
            new_conditioning, resolved_heads=set(replacement.values())
        )

        lifted_vars = set()
        for matched_sym in set(replacement.values()):
            new_conditioned = _unpack_existential(new_conditioned, matched_sym)
            lifted_vars.add(matched_sym)

        result = Condition(new_conditioned, new_conditioning)
        for var in lifted_vars:
            result = ExistentialPredicate(var, result)
        return result
