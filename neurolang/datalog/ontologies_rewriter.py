from ..expression_walker import (
    ReplaceExpressionWalker,
    ReplaceSymbolWalker,
    add_match,
)
from ..expressions import Symbol
from ..logic import Constant, Implication, NaryLogicOperator
from ..logic.expression_processing import ExtractFreeVariablesWalker
from ..logic.transformations import CollapseConjunctions
from ..logic.unification import apply_substitution, most_general_unifier
from .ontologies_parser import RightImplication


class ExtractFreeVariablesRightImplicationWalker(ExtractFreeVariablesWalker):
    @add_match(RightImplication)
    def extract_variables_s(self, expression):
        return self.walk(expression.consequent) - self.walk(
            expression.antecedent
        )


class OntologyRewriter:
    def __init__(self, query, union_of_constraints):
        self.query = query
        self.union_of_constraints = union_of_constraints

    def Xrewrite(self):
        """Algorithm based on the one proposed in
        G. Gottlob, G. Orsi, and A. Pieris,
        “Query Rewriting and Optimization for Ontological Databases,”
        ACM Transactions on Database Systems, vol. 39, May 2014."""
        rename_count = 0
        Q_rew = set({})
        for t in self.query.formulas:
            Q_rew.add((t, "r", "u"))

        sigma_free_vars = self._extract_free_variables()

        Q_temp = set({})
        while Q_rew != Q_temp:
            Q_temp = Q_rew.copy()
            for q in Q_temp:
                if q[2] == "e":
                    continue
                q0 = q[0]
                for sigma in sigma_free_vars:
                    Q_rew = self.rewriting_step(q0, sigma, rename_count, Q_rew)
                    Q_rew = self.factorization_step(q0, sigma, Q_rew)

                Q_rew.remove(q)
                Q_rew.add((q[0], q[1], "e"))

        return {x for x in Q_rew if x[2] == "e"}

    def _extract_free_variables(self):
        sigma_free_vars = []
        for sigma in self.union_of_constraints.formulas:
            if isinstance(sigma, RightImplication):
                efvw = ExtractFreeVariablesRightImplicationWalker()
                free_vars = efvw.walk(sigma)
                sigma_free_vars.append((sigma, free_vars))

        return sigma_free_vars

    def rewriting_step(self, q0, sigma, rename_count, Q_rew):
        body_q = q0.antecedent
        S_applicable = self._get_applicable(sigma, body_q)
        for S in S_applicable:
            rename_count += 1
            sigma_i = self._rename(sigma[0], rename_count)
            qS = most_general_unifier(sigma_i.consequent, S)
            if qS:
                new_q0 = self._combine_rewriting(q0, qS, S, sigma_i.antecedent)
                if self._is_new_rewriting(new_q0, Q_rew):
                    Q_rew.add((new_q0, "r", "u"))

        return Q_rew

    def _is_new_rewriting(self, new_q0, Q_rew):
        return (new_q0, "r", "u") not in Q_rew and (
            new_q0,
            "r",
            "e",
        ) not in Q_rew

    def factorization_step(self, q0, sigma, Q_rew):
        body_q = q0.antecedent
        S_factorizable = self._get_factorizable(sigma, body_q)
        if len(S_factorizable) > 1:
            qS = self._full_unification(S_factorizable)
            if qS:
                new_q0 = Implication(q0.consequent, qS)
                if self._is_new_factorization(new_q0, Q_rew):
                    Q_rew.add((new_q0, "f", "u"))

        return Q_rew

    def _is_new_factorization(self, new_q0, Q_rew):
        return (
            (new_q0, "r", "u") not in Q_rew
            and (new_q0, "r", "e") not in Q_rew
            and (new_q0, "f", "u") not in Q_rew
            and (new_q0, "f", "e") not in Q_rew
        )

    def _full_unification(self, S):
        acum = S[0]
        for term in S:
            temp = most_general_unifier(term, acum)
            new_term = apply_substitution(temp[1], temp[0])
            acum = new_term

        return acum

    def _get_factorizable(self, sigma, q):
        factorizable = []
        for free_var in sigma[1]._list:
            pos = self._get_position_existential(sigma[0].consequent, free_var)
            S = self._get_term(q, sigma[0].consequent)
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
                formula not in S and free_var in formula.args
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
                for index, sub_arg in enumerate(formula.args)
            )
        else:
            return any(
                arg == free_var and i != pos for i, arg in enumerate(q.args)
            )

        return False

    def _get_applicable(self, sigma, q):
        S = self._get_term(q, sigma[0].consequent)
        if self._is_applicable(sigma, q, S):
            return S

        return []

    def _get_term(self, q, sigma_con):
        q_args = []
        if isinstance(q, NaryLogicOperator):
            q_args = [
                formula
                for formula in q.formulas
                if formula.functor == sigma_con.functor
            ]
        else:
            if q.functor == sigma_con.functor:
                q_args.append(q)

        return q_args

    def _is_applicable(self, sigma, q, S):
        return self._unifies(
            S, sigma[0].consequent
        ) and self._not_in_existential(q, S, sigma)

    def _is_factorizable(self, S, pos):
        return all(most_general_unifier(term, S[0]) for term in S) or pos

    def _unifies(self, S, sigma):
        return all(most_general_unifier(term, sigma) for term in S)

    def _not_in_existential(self, q, S, sigma):
        for free_var in sigma[1]._list:
            existential_position = self._get_position_existential(
                sigma[0].consequent, free_var
            )
            if self._position_shared_or_constant(q, S, existential_position):
                return False

        return True

    def _get_position_existential(self, sigma, free_var):
        return [
            pos for pos, symbol in enumerate(sigma.args) if symbol == free_var
        ]

    def _position_shared_or_constant(self, q, S, positions):
        return any(
            isinstance(term.args[pos], Constant)
            or self._is_shared(term.args[pos], q)
            for pos in positions
            for term in S
        )

    def _is_shared(self, a, q):
        if isinstance(q, NaryLogicOperator):
            count = sum(a in term.args for term in q.formulas)
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
                    temp = arg.fresh()
                    temp.name = arg.name + str(index)
                    new_args[arg] = temp
                renamed.add(arg)
        return new_args, renamed

    def _combine_rewriting(self, q, qS, S, sigma_ant):
        sigma_ant = apply_substitution(sigma_ant, qS[0])
        replace = dict({S: sigma_ant})
        rsw = ReplaceExpressionWalker(replace)
        sigma_ant = rsw.walk(q.antecedent)
        sigma_ant = CollapseConjunctions().walk(sigma_ant)

        q_cons = apply_substitution(q.consequent, qS[0])

        return Implication(q_cons, sigma_ant)
