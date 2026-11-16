from nl_expr.expression_walker import (
    ReplaceExpressionWalker,
    ReplaceSymbolWalker,
    add_match,
)
from nl_expr.expressions import FunctionApplication, Symbol
from nl_expr.logic import (
    Constant, Implication, NaryLogicOperator, Negation, Union
)
from nl_expr.logic.expression_processing import ExtractFreeVariablesWalker
from nl_expr.logic.transformations import CollapseConjunctions
from nl_expr.logic.unification import apply_substitution, most_general_unifier
from .expression_processing import extract_logic_atoms
from .ontologies_parser import RightImplication


def constraints_that_can_hold(constraints, holds):
    """
    Return the ``constraints``, indexed by the name of their consequent's
    predicate as ``get_constraints`` does, whose antecedent can hold: each
    of its predicates is in ``holds`` or is the consequent of such a
    constraint. A rewriting step with another constraint gives a rule
    whose body never holds.
    """
    holding = set(holds)
    while True:
        kept = {
            name: {
                sigma for sigma in sigmas
                if all(
                    atom.functor in holding
                    for atom in extract_logic_atoms(sigma.antecedent)
                )
            }
            for name, sigmas in constraints.items()
        }
        heads = {
            sigma.consequent.functor
            for sigmas in kept.values() for sigma in sigmas
        }
        if heads <= holding:
            return {name: sigmas for name, sigmas in kept.items() if sigmas}
        holding |= heads


def valued_rules(constraints, holds, rules):
    """
    Return the ``rules`` without an atom over a consequent of the
    ``constraints`` that has no values, the predicates with values, from
    ``holds`` and the heads of these rules, and the constraints without
    existential variables that can hold from them.
    """
    without_existentials = {
        name: {s for s in sigmas if not _existential_variables(s)}
        for name, sigmas in constraints.items()
    }
    valued, heads = set(holds), None
    while heads is None or not heads <= valued:
        valued |= heads or set()
        full = constraints_that_can_hold(without_existentials, valued)
        unvalued = {
            sigma.consequent.functor for sigmas in constraints.values()
            for sigma in sigmas if sigma.consequent.functor.name not in full
        } - valued
        kept = [
            rule for rule in rules if not unvalued & {
                atom.functor for atom in extract_logic_atoms(rule.antecedent)
            }
        ]
        heads = {
            rule.consequent.functor for rule in kept
            if rule.consequent.functor.name in constraints
        }
    return kept, valued, full


def chase_rules(constraints, holds, rules, edb):
    """
    Return ``rules`` with, as Datalog rules, the ``constraints`` without
    existential variables that can hold from ``holds`` and that ``rules``
    depend on, as ``valued_rules`` keeps them. A predicate of ``edb`` these
    derive is replaced by a copy of its facts, as a rule cannot derive
    into it.
    """
    rules, _, full = valued_rules(constraints, holds, rules)
    chase, seen, todo = [], set(), list(rules)
    while todo:
        for atom in extract_logic_atoms(todo.pop().antecedent):
            name = getattr(atom.functor, "name", None)
            if name not in seen:
                seen.add(name)
                for sigma in full.get(name, ()):
                    chase.append(
                        Implication(sigma.consequent, sigma.antecedent)
                    )
                    todo.append(sigma)
    heads = {
        rule.consequent.functor: rule.consequent for rule in rules + chase
    }
    copies = {p.name: Symbol.fresh() for p in heads if p in edb}
    copy_rules = []
    for name, copy in copies.items():
        args = tuple(Symbol.fresh() for _ in heads[Symbol(name)].args)
        copy_rules.append(Implication(copy(*args), Symbol(name)(*args)))
    return tuple(
        ReplaceSymbolWalker(copies).walk(Union(tuple(rules) + tuple(chase)))
        .formulas
    ) + tuple(copy_rules)


class ExtractFreeVariablesRightImplicationWalker(ExtractFreeVariablesWalker):
    @add_match(RightImplication)
    def extract_variables_s(self, expression):
        return self.walk(expression.consequent) - self.walk(expression.antecedent)


def _arguments(formula):
    """The arguments of an atom, the symbols of a negated one."""
    return getattr(formula, "args", formula._symbols)


def _existential_variables(sigma):
    return ExtractFreeVariablesRightImplicationWalker().walk(sigma)


def _projects(sigma):
    """Whether the body of ``sigma`` is one atom with a variable out of its
    head, as ``p(x, y) → C(x)``: the head may hold of named values only
    while the body holds of an unnamed one."""
    return isinstance(sigma.antecedent, FunctionApplication) and bool(
        set(sigma.antecedent.args) - set(sigma.consequent.args)
    )


def projecting_rules(constraints):
    """The ``constraints`` whose body is one atom with a variable out of
    their head, as the domain ``p(x, y) → D(x)``, as rules. Their head can
    hold of named values through an unnamed one, which only the rewriting
    of their body finds: the chase applies their rewritings."""
    return tuple(
        Implication(sigma.consequent, sigma.antecedent)
        for sigmas in constraints.values() for sigma in sigmas
        if _projects(sigma)
    )


def _repeated_functors(q):
    """The predicates of two atoms or more of ``q``."""
    seen, repeated = set(), set()
    for f in getattr(q.antecedent, "formulas", (q.antecedent,)):
        functor = getattr(f, "functor", None)
        (repeated if functor in seen else seen).add(functor)
    return repeated


def _chased_only(sigma):
    """Whether the body of ``sigma`` has two atoms of its head's predicate,
    as ``p(x, y) ∧ p(y, z) → p(x, z)``: a rewriting with it never ends."""
    return sum(
        atom.functor == sigma.consequent.functor
        for atom in extract_logic_atoms(sigma.antecedent)
    ) > 1


class OntologyRewriter:
    """
    Rewrite ``query`` with ``union_of_constraints`` (Xrewrite).

    With ``chased``, the constraints without existential variables hold
    on the named values, as the chase applied them as rules. They then
    rewrite only atoms with a variable not in the rule's head, the only
    ones that can stand for a value an existential introduces.
    Transitivity is left to the chase.

    With ``holds``, the predicates with facts or rules, a rewriting with
    an atom that is never rewritten and has no values is left out, with
    all the rewritings it would give (see :meth:`_dead`).
    """

    def __init__(
        self, query, union_of_constraints, chased=False, holds=None
    ):
        self.query = query
        self.union_of_constraints = union_of_constraints
        self.chased = chased
        self.holds = holds
        self._null_positions_cache = {}
        self._backward_cache = {}
        self._moves_cache = {}
        self._sigma_cache = {}
        self._kills_cache = {}
        self._occurrences_cache = {}

    def _sigma(self, sigma):
        """What a rewriting step needs of ``sigma``, which is the same at
        every step: its existential variables, its body's atoms, and
        whether its head's arguments are distinct variables."""
        key = id(sigma)
        if key not in self._sigma_cache:
            body = extract_logic_atoms(sigma.antecedent)
            args = sigma.consequent.args
            self._sigma_cache[key] = (
                _existential_variables(sigma),
                tuple(body) if all(
                    isinstance(atom, FunctionApplication) for atom in body
                ) else None,
                all(isinstance(arg, Symbol) for arg in args)
                and len(set(args)) == len(args),
                sigma,
            )
        return self._sigma_cache[key]

    def Xrewrite(self):
        """Algorithm based on the one proposed in
        G. Gottlob, G. Orsi, and A. Pieris,
        “Query Rewriting and Optimization for Ontological Databases,”
        ACM Transactions on Database Systems, vol. 39, May 2014."""
        rename_count = 0
        Q_rew = set({})
        for t in self.query.formulas:
            Q_rew.add((t, "r", "u"))

        Q_temp = set({})
        Q_explored = set({})
        while Q_rew != Q_temp:
            Q_temp = Q_rew.copy()
            for q in Q_temp:
                q0 = q[0]
                if q0 in Q_explored:
                    Q_rew.remove(q)
                    continue
                selected_sigmas = self._get_related_sigmas(
                    q0, self.union_of_constraints
                )
                repeated = _repeated_functors(q0)
                for sigma in selected_sigmas:
                    list_q0 = self.rewriting_step(q0, sigma, rename_count)
                    for new_q0 in list_q0:
                        if (
                            self._is_new_rewriting(new_q0, Q_rew, Q_explored)
                            and not self._dead(new_q0)
                        ):
                            Q_rew.add((new_q0, "r", "u"))
                    if sigma.consequent.functor not in repeated:
                        # a factorization needs two atoms of sigma's head
                        continue
                    list_q0 = self.factorization_step(q0, sigma)
                    for new_q0 in list_q0:
                        if (
                            self._is_new_factorization(
                                new_q0, Q_rew, Q_explored
                            )
                            and not self._dead(new_q0)
                        ):
                            Q_rew.add((new_q0, "f", "u"))

                Q_rew.remove(q)
                Q_explored.add(q[0])

        return Q_explored

    def _dead(self, q):
        """
        Whether no rewriting of ``q`` can be evaluated.  An atom ``P(…)``
        whose predicate has no facts or rules, derived only by constraints
        with an existential in position ``i``, is rewritten only once its
        term ``t`` in position ``i`` is a variable that occurs nowhere
        else.  A constant or a variable of the head stays one in every
        rewriting, as does an occurrence of ``t`` in a negated atom, which
        is not rewritten.  Rewriting an atom keeps ``t``, at the positions
        of the constraint's variable, unless ``t`` is in its existential
        position, which is blocked while ``t`` is shared.  Hence ``t``
        leaves the other atoms only into an atom ``P(…)`` with ``t`` in
        position ``i``, factorised with this one: if an occurrence of ``t``
        cannot reach it, the atom over ``P``, which has no values, is in
        every rewriting of ``q``.
        """
        formulas = getattr(q.antecedent, "formulas", (q.antecedent,))
        return self._dead_atoms(
            [(f.functor, f.args) for f in formulas if hasattr(f, "functor")],
            [f for f in formulas if not hasattr(f, "functor")],
            q.consequent._symbols,
        )

    def _dead_atoms(self, atoms, others, distinguished):
        """:meth:`_dead` of the query of ``atoms``, ``(functor, args)``
        pairs, the formulas ``others`` that are not atoms, and the
        variables of its head, ``distinguished``."""
        if self.holds is None:
            return False
        for k, (functor, args) in enumerate(atoms):
            for i in self._null_positions(functor):
                term = args[i]
                if (
                    isinstance(term, Constant)
                    or term in distinguished
                    or any(term in f._symbols for f in others)
                ):
                    return True
                if any(
                    (functor, i) not in self._backward(other, j)
                    for m, (other, other_args) in enumerate(atoms)
                    for j, arg in enumerate(other_args)
                    if arg == term and (m != k or j != i)
                ):
                    return True
        return False

    def _kills(self, sigma):
        """The ``(j, B, i)`` of ``sigma``: its head's variable in position
        ``j`` is, in its body, in position ``i`` of an atom over ``B``,
        a position with no values (:meth:`_null_positions`).  None when
        its head is not of distinct variables."""
        key = id(sigma)
        if key not in self._kills_cache:
            _, body, plain_head, _ = self._sigma(sigma)
            self._kills_cache[key] = None if body is None or not plain_head \
                else tuple(
                    (j, atom.functor, i)
                    for j, v in enumerate(sigma.consequent.args)
                    for atom in body
                    for i in self._null_positions(atom.functor)
                    if atom.args[i] == v
                )
        return self._kills_cache[key]

    def _surely_dead(self, q, S, sigma):
        """
        Whether rewriting the atom ``S`` of ``q`` with ``sigma`` is left
        out, told from what is precomputed of ``sigma`` (:meth:`_kills`):
        the term of ``S`` that its body puts in a position with no values
        is a constant, a variable of the head or of a negated formula, or
        occurs in another atom of ``q`` from which no single step moves it
        there (:meth:`_moves`).  Steps on distinct atoms commute, so the
        step waits until each other occurrence is one step away; the
        rewritings with values are the same (see doc/ontologies.rst).
        :meth:`_builds_dead` tests the other steps in full.
        """
        kills = self._kills(sigma) if self.holds is not None else None
        if not kills:
            return False
        occurrences, blocked = self._occurrences(q)
        for j, functor, i in kills:
            term = S.args[j]
            if isinstance(term, Constant) or term in blocked:
                return True
            if any(
                (functor, i) not in self._moves(other, k)
                for atom, other, k in occurrences.get(term, ())
                if atom is not S and atom != S
            ):
                return True
        return False

    def _occurrences(self, q):
        """Where each term of ``q`` occurs, ``(atom, functor, position)``
        triples, and the variables of its head and of its negated
        formulas, computed once for each query explored."""
        key = id(q)
        cached = self._occurrences_cache.get(key)
        if cached is None or cached[0] is not q:
            occurrences, blocked = {}, set(q.consequent._symbols)
            for f in getattr(q.antecedent, "formulas", (q.antecedent,)):
                if hasattr(f, "functor"):
                    for k, arg in enumerate(f.args):
                        occurrences.setdefault(arg, []).append(
                            (f, f.functor, k)
                        )
                else:
                    blocked |= f._symbols
            cached = self._occurrences_cache[key] = (
                q, occurrences, blocked
            )
        return cached[1], cached[2]

    def _builds_dead(self, q, S, sigma):
        """
        Whether rewriting the atom ``S`` of ``q`` with ``sigma`` gives a
        query that :meth:`_dead` drops, told before building it: the head
        of ``sigma``, of distinct variables, maps them to the terms of
        ``S``, and its other variables get names no query has, as
        renaming makes them distinct.
        Other heads, which bind the terms of ``q``, are built and tested.
        """
        _, body, plain_head, _ = self._sigma(sigma)
        if self.holds is None or body is None or not plain_head:
            return False
        terms = dict(zip(sigma.consequent.args, S.args))
        formulas = getattr(q.antecedent, "formulas", (q.antecedent,))
        atoms = [
            (f.functor, f.args) for f in formulas
            if hasattr(f, "functor") and f != S
        ] + [
            (atom.functor, tuple(
                terms.get(arg, Symbol("\0" + arg.name))
                if isinstance(arg, Symbol) else arg
                for arg in atom.args
            ))
            for atom in body
        ]
        others = [f for f in formulas if not hasattr(f, "functor")]
        return self._dead_atoms(atoms, others, q.consequent._symbols)

    def _null_positions(self, functor):
        """The positions of the existential of every constraint deriving
        ``functor``, none if one has none or ``functor`` has values."""
        if functor not in self._null_positions_cache:
            positions = set()
            sigmas = self.union_of_constraints.get(functor, ())
            if sigmas and functor not in self.holds:
                positions = set.intersection(*(
                    {
                        i for v in _existential_variables(sigma)
                        for i in self._get_position_existential(
                            sigma.consequent, v
                        )
                    }
                    for sigma in sigmas
                ))
            self._null_positions_cache[functor] = positions
        return self._null_positions_cache[functor]

    def _backward(self, functor, position):
        """The predicates and positions a term in ``position`` of an atom
        over ``functor`` can be moved to by rewriting steps that keep it."""
        key = (functor, position)
        if key not in self._backward_cache:
            reach, todo = {key}, [key]
            while todo:
                for move in self._moves(*todo.pop()) - reach:
                    reach.add(move)
                    todo.append(move)
            self._backward_cache[key] = reach
        return self._backward_cache[key]

    def _moves(self, functor, position):
        """``(functor, position)`` and the positions of the bodies of the
        constraints deriving ``functor`` one rewriting step moves its term
        to."""
        key = (functor, position)
        if key not in self._moves_cache:
            self._moves_cache[key] = {key} | {
                (body.functor, k)
                for sigma in self.union_of_constraints.get(functor, ())
                if isinstance(v := sigma.consequent.args[position], Symbol)
                and v not in _existential_variables(sigma)
                for body in extract_logic_atoms(sigma.antecedent)
                for k, arg in enumerate(body.args) if arg == v
            }
        return self._moves_cache[key]

    def _get_related_sigmas(self, q0, sigma_free_vars):
        new_sigmas = []
        if isinstance(q0.antecedent, NaryLogicOperator):
            for f in q0.antecedent.formulas:
                if isinstance(f, Negation):
                    continue
                if f.functor in sigma_free_vars.keys():
                    new_sigmas = new_sigmas + list(sigma_free_vars[f.functor])
        else:
            if q0.antecedent.functor in sigma_free_vars.keys():
                new_sigmas = new_sigmas + list(sigma_free_vars[q0.antecedent.functor])

        return new_sigmas

    def rewriting_step(self, q0, sigma, rename_count):
        if all(
            self._surely_dead(q0, S, sigma)
            for S in self._get_term(q0.antecedent, sigma.consequent)
        ):
            return []
        S_applicable = self._get_applicable(sigma, q0)
        list_q0 = []
        # The renamed variables of sigma must not be those of q0.
        names = {symbol.name for symbol in q0._symbols}
        while any(
            f"{symbol.name}{rename_count + 1}" in names
            for symbol in sigma._symbols
        ):
            rename_count += 1
        for S in S_applicable:
            rename_count += 1
            if self._builds_dead(q0, S, sigma):
                continue
            sigma_i = self._rename(sigma, rename_count)
            qS = most_general_unifier(sigma_i.consequent, S)
            if qS:
                new_q0 = self._combine_rewriting(q0, qS, S, sigma_i)
                list_q0.append(new_q0)

        return list_q0

    def _is_new_rewriting(self, new_q0, Q_rew, Q_explored):
        return (new_q0, "r", "u") not in Q_rew and (new_q0, "r", "e",) not in Q_explored

    def factorization_step(self, q0, sigma):
        """Unify the atoms of ``q0`` that ``sigma`` factorizes, in all of
        ``q0``, and keep one copy of each atom."""
        functor = sigma.consequent.functor
        if len({
            f for f in getattr(q0.antecedent, "formulas", (q0.antecedent,))
            if getattr(f, "functor", None) == functor
        }) < 2:
            # one atom unifies only with itself: q0, never a new query
            return []
        S = self._get_factorizable(sigma, q0.antecedent)
        unified = self._full_unification(S) if len(S) > 1 else None
        if unified is None:
            return []
        q = ReplaceSymbolWalker({
            arg.name: u
            for atom in S for arg, u in zip(atom.args, unified.args)
            if isinstance(arg, Symbol) and arg != u
        }).walk(q0)
        body = q.antecedent.apply(tuple(dict.fromkeys(q.antecedent.formulas)))
        return [Implication(q.consequent, CollapseConjunctions().walk(body))]

    def _is_new_factorization(self, new_q0, Q_rew, Q_explored):
        return (
            (new_q0, "r", "u") not in Q_rew
            and (new_q0, "r", "e") not in Q_explored
            and (new_q0, "f", "u") not in Q_rew
            and (new_q0, "f", "e") not in Q_explored
        )

    def _full_unification(self, S):
        acum = S[0]
        for term in S:
            temp = most_general_unifier(term, acum)
            if temp is None:
                return None
            acum = apply_substitution(temp[1], temp[0])

        return acum

    def _get_factorizable(self, sigma, q):
        factorizable = []
        for fv in self._sigma(sigma)[0]:
            pos = self._get_position_existential(sigma.consequent, fv)
            S = self._get_term(q, sigma.consequent)
            if (
                S
                and self._is_factorizable(S, pos)
                and self._var_same_position(pos, q, S)
            ):
                factorizable.append(S)

        return sum(factorizable, [])

    def _var_same_position(self, pos, q, S):
        eq_vars = self._equivalent_var(pos, S)
        for var in eq_vars:
            if self._free_var_other_term(var, q, S):
                return False

            if self._free_var_same_term_other_position(var, pos, q):
                return False

        return True

    def _equivalent_var(self, pos, S):
        eq_vars = []
        for elem in pos:
            eq_vars.append(S[0].args[elem])

        return eq_vars

    def _free_var_other_term(self, free_var, q, S):
        if isinstance(q, NaryLogicOperator):
            return any(
                formula not in S and free_var in _arguments(formula)
                for formula in q.formulas
            )
        else:
            if q != S and free_var in q.args:
                return True

        return False

    def _free_var_same_term_other_position(self, free_var, pos, q):
        if isinstance(q, NaryLogicOperator):
            return any(
                sub_arg == free_var and index not in pos
                for formula in q.formulas
                for index, sub_arg in enumerate(_arguments(formula))
            )
        else:
            return any(arg == free_var and i != pos for i, arg in enumerate(q.args))

    def _get_applicable(self, sigma, q):
        S = self._get_term(q.antecedent, sigma.consequent)
        if self.chased and not self._sigma(sigma)[0]:
            if _chased_only(sigma):
                return []
            S = [
                s for s in S if any(
                    isinstance(arg, Symbol)
                    and not self._is_distinguished(arg, q)
                    for arg in s.args
                )
            ]
        if self._is_applicable(sigma, q, S):
            return S

        return []

    def _get_term(self, q, sigma_con):
        q_args = []
        # Only atoms are rewritten: a negated atom has no functor.
        if isinstance(q, NaryLogicOperator):
            q_args = [
                formula
                for formula in q.formulas
                if getattr(formula, "functor", None) == sigma_con.functor
            ]
        else:
            if getattr(q, "functor", None) == sigma_con.functor:
                q_args.append(q)

        return q_args

    def _is_applicable(self, sigma, q, S):
        return self._unifies(S, sigma.consequent) and self._not_in_existential(
            q, S, sigma
        )

    def _is_factorizable(self, S, pos):
        return all(most_general_unifier(term, S[0]) for term in S) or pos

    def _unifies(self, S, sigma):
        return all(most_general_unifier(term, sigma) for term in S)

    def _not_in_existential(self, q, S, sigma):
        for fv in self._sigma(sigma)[0]:
            existential_position = self._get_position_existential(sigma.consequent, fv)
            if self._position_shared_or_constant(q, S, existential_position):
                return False

        return True

    def _get_position_existential(self, sigma, free_var):
        return [pos for pos, symbol in enumerate(sigma.args) if symbol == free_var]

    def _position_shared_or_constant(self, q, S, positions):
        return any(
            isinstance(term.args[pos], Constant) or self._is_shared(term.args[pos], q.antecedent)
            or self._is_distinguished(term.args[pos], q) for pos in positions for term in S
        )

    def _is_distinguished(self, var, q):
        return var in q.consequent._symbols

    def _is_shared(self, a, q):
        if isinstance(q, NaryLogicOperator):
            count = sum(
                a == var for term in q.formulas for var in _arguments(term)
            )
        else:
            count = sum(a == term for term in q.args)
        return count > 1

    def _rename(self, sigma, index):
        renamed = set({})
        a, renamed = self._replace(sigma.antecedent, index, renamed)
        b, renamed = self._replace(sigma.consequent, index, renamed)
        sus = {**a, **b}
        sigma = ReplaceSymbolWalker(sus).walk(sigma)

        return sigma

    def _replace(self, sigma, index, renamed):
        new_args = {}
        if isinstance(sigma, NaryLogicOperator):
            for app in sigma.formulas:
                new_arg, renamed = self._replace(app, index, renamed)
                new_args = {**new_args, **new_arg}
        else:
            for arg in sigma.args:
                if arg not in renamed and isinstance(arg, Symbol):
                    # A new symbol: renaming one would leave stale hashes.
                    temp = type(arg)(arg.name + str(index))
                    temp.is_fresh = arg.is_fresh
                    new_args[arg] = temp
                renamed.add(arg)
        return new_args, renamed

    def _combine_rewriting(self, q, qS, S, sigma):
        """Replace ``S`` in ``q`` by the body of ``sigma`` and apply their
        unifier ``qS`` to all of ``q``: a query variable it binds, as to
        the constant of ``C(x) → p(x, v)``, is bound everywhere."""
        substitution = {}
        for symbol, value in qS[0].items():
            for _ in qS[0]:
                if value not in qS[0]:
                    break
                value = qS[0][value]
            substitution[symbol.name] = value
        q = ReplaceSymbolWalker(substitution).walk(Implication(
            q.consequent,
            ReplaceExpressionWalker({S: sigma.antecedent}).walk(q.antecedent),
        ))
        return Implication(
            q.consequent, CollapseConjunctions().walk(q.antecedent)
        )

