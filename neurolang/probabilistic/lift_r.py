from functools import lru_cache, reduce
from itertools import chain, combinations

from .. import relational_algebra_provenance as rap
from ..datalog import DatalogProgram, Fact, chase
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..exceptions import NonLiftableException
from ..expression_walker import (
    ChainedWalker,
    PatternWalker,
    ReplaceExpressionWalker,
    add_match
)
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    NaryLogicOperator,
    UnaryLogicOperator,
    Union,
    UniversalPredicate
)
from ..logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables
)
from ..logic.transformations import (
    CollapseConjunctions,
    CollapseDisjunctions,
    DistributeConjunctions,
    DistributeDisjunctions,
    LogicExpressionWalker,
    convert_to_pnf_with_cnf_matrix,
    convert_to_pnf_with_dnf_matrix
)
from ..logic.unification import compose_substitutions, most_general_unifier
from ..relational_algebra import (
    BinaryRelationalAlgebraOperation,
    NAryRelationalAlgebraOperation,
    NaturalJoin,
    RelationalAlgebraOperation,
    UnaryRelationalAlgebraOperation
)

__all__ = [
    "LiftRAlgorithm",
]


def has_separator_variables(query):
    '''
    Returns true if `query` has a separator variable.

    According to Dalvi and Suciu [1]_ if `query` is in DNF,
    a variable z is called a separator variable if Q starts with ∃z,
    that is, Q = ∃z.Q1, for some query expression Q1, and (a) z
    is a root variable (i.e. it appears in every atom),
    (b) for every relation symbol R, there exists an attribute (R, iR)
    such that every atom with symbol R has z in position iR. This is
    equivalent, in datalog syntax, to Q ≡ Q0←∃z.Q1.

    Also, according to Suciu [2]_ the dual is also true,
    if `query` is in CNF i.e. the separation variable z needs to
    be universally quantified, that is Q = ∀x.Q1. But this is not
    implemented.

    [1] Dalvi, N. & Suciu, D. The dichotomy of probabilistic inference
    for unions of conjunctive queries. J. ACM 59, 1–87 (2012).
    [2] Suciu, D. Probabilistic Databases for All. in Proceedings of the
    39th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems
    19–31 (ACM, 2020).
    '''

    return len(find_separator_variables(query)[0]) > 0


def all_free_variables_the_same(expression):
    head_vars = set(expression.consequent.args)
    return all(
        head_vars.issubset(
            extract_logic_free_variables(formula)
        )
        for formula in expression.antecedent.formulas
    )


def some_free_variables_the_same(expression):
    head_vars = set(expression.consequent.args)
    return all(
        not head_vars.isdisjoint(
            extract_logic_free_variables(formula)
        )
        for formula in expression.antecedent.formulas
    )


class WeightedNaturalJoin(NAryRelationalAlgebraOperation):
    def __init__(self, relations, weights):
        self.relations = relations
        self.weights = weights

    def __repr__(self):
        return (
            "\N{Greek Capital Letter Sigma}"
            f"_{self.weights}({self.relations})"
        )


class NonLiftable(RelationalAlgebraOperation):
    def __init__(self, non_liftable_query):
        self.non_liftable_query = non_liftable_query

    def __repr__(self):
        return (
            "NonLiftable"
            f"({self.non_liftable_query})"
        )


def descomposable_independent_disjunction(expression):
    consequent, antecedent = expression.unapply()
    antecedent = convert_to_pnf_with_dnf_matrix(antecedent)
    expression = Implication(consequent, antecedent)
    return is_syntactically_independent(antecedent)


def can_split_conjunctive_query_independent(expression):
    expression = convert_to_cnf_rule(expression)
    if not isinstance(expression.antecedent, (Conjunction, Disjunction)):
        return False
    _, right = split_conjunctive_query_independent(
        expression
    )

    return len(right) > 0


class LiftRAlgorithm(PatternWalker):
    '''
    Transforms a probabilistic query expressed an extended Datalog rule.
    Implements the "Safe Query Plan" algorithm from [1]_, chapter 4,
    Algorithm 1.

    [1] Suciu, D., Olteanu, D., Koch, C. & Koch,
    C. Probabilistic Databases. (Morgan & Claypool Publishers, 2011).
    '''

    @add_match(
        Implication(..., NaryLogicOperator),
        lambda e: len(e.antecedent.formulas) == 1
    )
    def nary_simple(self, expression):
        return self.walk(Implication(
            expression.consequent,
            expression.antecedent.formulas[0]
        ))

    @add_match(
        Implication(..., NaryLogicOperator),
        lambda e: len(set(e.antecedent.formulas)) != len(e.antecedent.formulas)
    )
    def nary_repeated(self, expression):
        return self.walk(Implication(
            expression.consequent,
            expression.antecedent.apply(tuple(
                set(expression.antecedent.formulas)
            ))
        ))

    @add_match(
        Implication(..., Disjunction),
        descomposable_independent_disjunction
    )
    def decomposable_disjunction(self, expression):
        consequent = expression.consequent
        q = syntactically_independent_components(
           convert_to_pnf_with_dnf_matrix(expression.antecedent)
        )
        walked_formulas = tuple(
            self.walk(Implication(consequent, q_))
            for q_ in q.formulas
        )
        return reduce(
            rap.Union,
            walked_formulas[1:], walked_formulas[0]
        )

    @add_match(
        Implication,
        can_split_conjunctive_query_independent
    )
    def decomposable_comjunction(self, expression):
        left, right = split_conjunctive_query_independent(expression)
        left = Conjunction(left)
        right = Conjunction(right)
        head_functor = expression.consequent.functor
        head_vars = set(expression.consequent.args)
        left = Implication(
            head_functor(
                tuple(head_vars & extract_logic_free_variables(left))
            ),
            left
        )
        right = Implication(
            head_functor(
                tuple(head_vars & extract_logic_free_variables(right))
            ),
            right
        )
        return rap.Projection(
            rap.NaturalJoin(
                self.walk(left),
                self.walk(right)
            ),
            tuple(head_vars)
        )
    # @add_match(
    #     Implication(..., Conjunction),
    #     lambda e: is_syntactically_independent(e.antecedent)
    # )
    # def decomposable_conjunction(self, expression):
    #     consequent = expression.consequent
    #     q = syntactically_independent_components(expression.antecedent)
    #     common_fvs, q_fvs = self._extract_formulas_common_fvs(q)

    #     common_args = common_fvs | set(consequent.args)
    #     walked_formulas = tuple(
    #         self.walk(
    #             Implication(
    #                 FunctionApplication(
    #                     Symbol.fresh(),
    #                     tuple(q_fv & common_args)
    #                 ),
    #                 q_
    #             )
    #         )
    #         for q_, q_fv in zip(q.formulas, q_fvs)
    #     )
    #     return rap.Projection(
    #         reduce(
    #             rap.NaturalJoin,
    #             walked_formulas[1:], walked_formulas[0]
    #         ),
    #         consequent.args
    #     )

    @staticmethod
    def _extract_formulas_common_fvs(q):
        q_fv = [
            extract_logic_free_variables(q_)
            for q_ in q.formulas
        ]
        common_args = (
            set().union(*(
                q_fv[i] & q_fv[j]
                for i in range(len(q_fv))
                for j in range(i + 1, len(q_fv))
            ))
        )
        return common_args, q_fv

    @add_match(Implication, has_separator_variables)
    def separator_variable(self, expression):
        svs, expression = find_separator_variables(expression)
        consequent, antecedent = expression.unapply()
        svs = tuple(svs)
        return rap.Projection(
            self.walk(
                Implication(
                    FunctionApplication(
                        consequent.functor,
                        consequent.args + svs
                    ),
                    antecedent
                )
            ),
            consequent.args
        )

    @add_match(
        Implication(..., NaryLogicOperator),
        all_free_variables_the_same
    )
    def inclusion_exclusion(self, expression):
        expression = convert_to_cnf_rule(expression)
        consequent, antecedent = expression.unapply()
        args = consequent.args
        if isinstance(antecedent, Conjunction):
            # antecedent = convert_to_pnf_with_cnf_matrix(antecedent)
            operation = Disjunction
            containment_function = is_contained_dnf
            cc = convert_to_pnf_with_dnf_matrix
        else:
            # antecedent = convert_to_pnf_with_dnf_matrix(antecedent)
            operation = Conjunction
            containment_function = is_contained
            #cc = CollapseConjunctions()
            cc = convert_to_pnf_with_cnf_matrix
        formula_powerset = [
            Implication(
                Symbol.fresh()(*args),
                cc(operation(tuple(formula)))
            )
            for formula in powerset(antecedent.formulas)
            if len(formula) > 0
        ]
        formulas_weights = self._formulas_weights(
            formula_powerset, containment_function
        )
        new_formulas, weights = zip(*(
            (self.walk(formula), weight)
            for formula, weight in formulas_weights.items()
            if weight != 0
        ))

        return WeightedNaturalJoin(tuple(new_formulas), tuple(weights))

    def _formulas_weights(self, formula_powerset, containment_function):
        formula_containments = {
            formula: set()
            for formula in formula_powerset
        }
        for i, f0 in enumerate(formula_powerset):
            for f1 in formula_powerset[i + 1:]:
                for c0, c1 in ((f0, f1), (f1, f0)):
                    if (
                        (c1 not in formula_containments[f0]) &
                        containment_function(c0, c1)
                    ):
                        formula_containments[c0].add(c1)
                        formula_containments[c0] |= (
                            formula_containments[c1] -
                            {c0}
                        )
                        break

        formulas_weights = mobius_weights(formula_containments)
        return formulas_weights

    @add_match(Implication(..., FunctionApplication))
    def function_application(self, expression):
        return TranslateToNamedRA().walk(expression.antecedent)

    @add_match(...)
    def default(self, expression):
        return NonLiftable(expression)


def mobius_weights(formula_containments):
    _mobius_weights = {}
    for formula in formula_containments:
        _mobius_weights[formula] = mobius_function(
            formula, formula_containments, _mobius_weights
        )
    return _mobius_weights


def mobius_function(formula, formula_containments, known_weights={}):
    if formula in known_weights:
        return known_weights[formula]
    res = -sum(
        (
            known_weights.setdefault(
                f,
                mobius_function(f, formula_containments)
            )
            for f in formula_containments[formula]
            if f != formula
        ),
        1
    )
    return res


@lru_cache
def split_conjunctive_query_independent(expression):
    if not isinstance(expression, Implication):
        raise ValueError("Only works conjunctive queries")
    # antecedent = convert_to_pnf_with_cnf_matrix(expression.antecedent)

    expression = convert_to_cnf_rule(expression)
    antecedent = expression.antecedent
    head_vars = set(expression.consequent.args)
    formulas = list(antecedent.formulas)
    left = (formulas.pop(),)
    right = tuple()
    left_symbols = set(left[0]._symbols) - head_vars
    while formulas:
        formula = formulas.pop()
        f_symbols = set(formula._symbols)
        if left_symbols.isdisjoint(f_symbols):
            right += (formula,)
        else:
            left += (formula,)
            left_symbols |= f_symbols - head_vars
            formulas += right
            right = tuple()
    return left, right


def convert_to_cnf_rule(implication):
    consequent, antecedent = implication.unapply()
    head_vars = set(consequent.args)
    existential_vars = (
        extract_logic_free_variables(antecedent) -
        set(head_vars)
    )
    for a in existential_vars:
        antecedent = ExistentialPredicate(a, antecedent)
    c = ChainedWalker(
        PushExistentialsDown,
        DistributeDisjunctions,
        CollapseConjunctions,
        CollapseDisjunctions,
        RemoveExistentials,
        CollapseConjunctions,
    )
    antecedent = c.walk(antecedent)
    return Implication(consequent, antecedent)


def distribute_disjunctions(antecedent, head_vars):
    existential_vars = (
        extract_logic_free_variables(antecedent) -
        set(head_vars)
    )
    for a in existential_vars:
        antecedent = ExistentialPredicate(a, antecedent)
    # antecedent = disjunction_to_conjunction(antecedent)
    c = ChainedWalker(
        PushExistentialsDown,
        DistributeDisjunctions,
    )
    antecedent = c.walk(antecedent)
    antecedent = RemoveExistentials().walk(antecedent)
    return antecedent


class RemoveExistentials(LogicExpressionWalker):
    @add_match(ExistentialPredicate)
    def existential(self, expression):
        return self.walk(expression.body)


def is_syntactically_independent(expression):
    operation = type(expression)
    if operation not in (Conjunction, Disjunction):
        raise ValueError(
            "Syntactic independence only valid for conjunction or disjunctions"
        )
    _, splittable = \
        compute_syntactically_independent_splits_if_possible(
            expression, operation
        )
    return splittable


def syntactically_independent_components(expression):
    operation = type(expression)
    if operation not in (Conjunction, Disjunction):
        raise ValueError(
            "Syntactic independence only valid for conjunction or disjunctions"
        )
    res, is_syntactically_independent = \
        compute_syntactically_independent_splits_if_possible(
            expression, operation
        )
    if not is_syntactically_independent:
        raise ValueError(f"{expression} not syntactically independent")
    return res


@lru_cache
def compute_syntactically_independent_splits_if_possible(query, operation):
    '''
    Group formulas into independent subformula splits
    using the associativity property of `operation`.
    The definition of independent is that subformulas don't
    share relational symbols.
    '''
    if len(query.formulas) == 1:
        return query.formulas[0], True

    if operation is Conjunction:
        query = convert_to_pnf_with_cnf_matrix(query)
    else:
        query = convert_to_pnf_with_dnf_matrix(query)

    atom_groups_indices = _compute_independent_split_indices(query)

    if len(atom_groups_indices) == 1:
        return query, False
    new_query = _splits_to_expression(atom_groups_indices, query, operation)
    return new_query, True


def _compute_independent_split_indices(query):
    atom_groups_indices = [
        (
            set(
                a.functor
                for a in extract_logic_atoms(q)
            ),
            (i,)
        )
        for i, q in enumerate(query.formulas)
    ]
    changed = True
    while changed:
        outer_keep = []
        changed = False
        while atom_groups_indices:
            current_atoms, current_idxs = atom_groups_indices.pop()
            inner_keep = []
            while atom_groups_indices:
                atoms, idxs = atom_groups_indices.pop()
                if not current_atoms.isdisjoint(atoms):
                    current_idxs += idxs
                    current_atoms |= atoms
                    changed = True
                else:
                    inner_keep.append((atoms, idxs))
            atom_groups_indices = inner_keep
            outer_keep.append((current_atoms, current_idxs))
        atom_groups_indices = outer_keep
    return [ixs for _, ixs in atom_groups_indices]


def _splits_to_expression(atom_groups_indices, query, operation):
    new_formulas = []
    for formula_indices in atom_groups_indices:
        new_formula = tuple(
           query.formulas[i]
           for i in formula_indices
        )
        if len(new_formula) == 1:
            new_formula = new_formula[0]
        else:
            new_formula = operation(tuple(new_formula))
        new_formulas.append(new_formula)
    new_query = operation(tuple(new_formulas))
    return new_query


def inclusion_exclusion_formulas(query):
    query_powerset = powerset(query.formulas)
    clean_powerset = set(
        set(q) if len(q) > 1 else set((q,))
        for q in query_powerset
    )
    result = tuple(
        Conjunction(tuple(q)) if len(q) > 1 else q.pop()
        for q in clean_powerset
    )
    return result


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


@lru_cache
def find_separator_variables(query):
    '''
    According to Dalvi and Suciu [1]_ if `query` is in DNF,
    a variable z is called a separator variable if Q starts with ∃z,
    that is, Q = ∃z.Q1, for some query expression Q1, and (a) z
    is a root variable (i.e. it appears in every atom),
    (b) for every relation symbol R, there exists an attribute (R, iR)
    such that every atom with symbol R has z in position iR. This is
    equivalent, in datalog syntax, to Q ≡ Q0←∃z.Q1.

    Also, according to Suciu [2]_ the dual is also true,
    if `query` is in CNF i.e. the separation variable z needs to
    be universally quantified, that is Q = ∀x.Q1.

    This algorithm assumes that Q1 can't be splitted into independent
    formulas.

    [1] Dalvi, N. & Suciu, D. The dichotomy of probabilistic inference
    for unions of conjunctive queries. J. ACM 59, 1–87 (2012).
    [2] Suciu, D. Probabilistic Databases for All. in Proceedings of the
    39th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems
    19–31 (ACM, 2020).
    '''

    if isinstance(query, Implication):
        exclude_variables = set(query.consequent.args)
        consequent = query.consequent
        query = convert_to_pnf_with_dnf_matrix(query.antecedent)
        query = _unify_existential_variables(query, exclude_variables)
        svs, rewritten_query = find_separator_variables(query)
        return (
            svs - exclude_variables,
            Implication(consequent, rewritten_query)
        )
    elif isinstance(query, ExistentialPredicate):
        raise ValueError(
            "Separator variable for explicitely existentially "
            " quantified cases not implemented."
        )
        query = _unify_existential_variables(query, {})
        query = convert_to_pnf_with_dnf_matrix(query.body)
        svs, rewritten_query = find_separator_variables(query)
        return find_separator_variables(query)
    elif isinstance(query, UniversalPredicate):
        raise ValueError(
            "Separator variable for universally quantified "
            "cases not implemented."
        )

    if isinstance(query, NaryLogicOperator):
        formulas = query.formulas
    else:
        formulas = [query]

    candidates = None
    all_atoms = set()
    for i, formula in enumerate(formulas):
        atoms = extract_logic_atoms(formula)
        all_atoms |= atoms
        root_variables = reduce(
            lambda y, x: set(x.args) & y,
            atoms[1:],
            set(atoms[0].args)
        )
        if candidates is None:
            candidates = root_variables
        else:
            candidates &= root_variables

    separator_variables = set()
    for var in candidates:
        atom_positions = {}
        for atom in all_atoms:
            functor = atom.functor
            pos_ = {i for i, v in enumerate(atom.args) if v == var}
            if any(
                pos_.isdisjoint(pos)
                for pos in atom_positions.setdefault(functor, [])
            ):
                break
            atom_positions[functor].append(pos_)
        else:
            separator_variables.add(var)

    return separator_variables, query


def _unify_existential_variables(query, exclude_variables):
    if not isinstance(query, Disjunction):
        return query
    for variable in (
        extract_logic_free_variables(query) - exclude_variables
    ):
        query = ExistentialPredicate(variable, query)
    query = PushExistentialsDown().walk(query)
    while isinstance(query, ExistentialPredicate):
        query = query.body
    unifiers = []
    for i, clause in enumerate(query.formulas):
        atoms = extract_logic_atoms(clause)
        for clause_ in query.formulas[i + 1:]:
            atoms_ = extract_logic_atoms(clause_)
            unifiers += [
                most_general_unifier(a, a_)
                for a in atoms
                for a_ in atoms_
            ]
    unifiers = reduce(
        compose_substitutions,
        (u[0] for u in unifiers if u is not None),
        {}
    )
    unifiers = {
        k: v for k, v in unifiers.items()
        if exclude_variables.isdisjoint((k, v))
    }
    for i in range(len(unifiers)):
        query = ReplaceExpressionWalker(unifiers).walk(query)
    new_formulas = tuple()
    for formula in query.formulas:
        while isinstance(formula, ExistentialPredicate):
            formula = formula.body
        new_formulas += (formula,)
    query = Disjunction(new_formulas)
    return query


class PushExistentialsDown(LogicExpressionWalker):
    @add_match(ExistentialPredicate(..., Disjunction))
    def push_existential_down_disjunction(self, expression):
        variable = expression.head
        changed = False
        new_formulas = tuple()
        for formula in expression.body.formulas:
            if variable in extract_logic_free_variables(formula):
                changed = True
                formula = self.walk(ExistentialPredicate(variable, formula))
            new_formulas += (formula,)
        if changed:
            res = self.walk(Disjunction(new_formulas))
        else:
            res = expression
        return res

    @add_match(ExistentialPredicate(..., Conjunction))
    def push_existential_down(self, expression):
        variable = expression.head
        in_ = tuple()
        out_ = tuple()
        for formula in expression.body.formulas:
            if variable in extract_logic_free_variables(formula):
                in_ += (formula,)
            else:
                out_ += (formula,)

        if len(out_) == 0:
            res = ExistentialPredicate(variable, self.walk(Conjunction(in_)))
        elif len(in_) == 0:
            res = self.walk(Conjunction(out_))
        if len(in_) > 0 and len(out_) > 0:
            res = self.walk(
                Conjunction((
                    ExistentialPredicate(variable, Conjunction(in_)),
                    Conjunction(out_)
                ))
            )
        return res

    @add_match(ExistentialPredicate(..., ExistentialPredicate))
    def nested_existential(self, expression):
        outer_var = expression.head
        new_body = self.walk(expression.body)
        if new_body != expression.body:
            expression = self.walk(
                ExistentialPredicate(outer_var, new_body)
            )
        else:
            inner_var = expression.body.head
            inner_body = expression.body.body
            swapped_body = ExistentialPredicate(
                outer_var, inner_body
            )
            new_body = self.walk(swapped_body)
            if new_body != swapped_body:
                expression = self.walk(
                    ExistentialPredicate(inner_var, new_body)
                )
        return expression


def disjunction_to_conjunction(expression):
    c = ChainedWalker(
        PushExistentialsDown,
        DistributeDisjunctions,
        CollapseConjunctions,
        CollapseDisjunctions,
    )
    res = c.walk(expression)
    return res


def _clause_separator_variable_candidates(
    formula, separator_variable_position
):
    '''
    Variables in a clause are candidates to be a separator variable if
    they appear in every atom in the clause once; and
    for each predicate in the query, they appear always in the same position.
    '''
    atoms = extract_logic_atoms(formula)
    clause_separator_variables = _separator_variable_candidates_atom(
        atoms[0], separator_variable_position
    )
    for atom in atoms[1:]:
        clause_separator_variables &= _separator_variable_candidates_atom(
            atom, separator_variable_position
        )
        if len(clause_separator_variables) == 0:
            break
    return clause_separator_variables


def _separator_variable_candidates_atom(atom, separator_variable_position):
    '''
    Obtains the separator variable candidates defined as all variables
    which appear once in the arguments and for each appearance of the
    relational symbol, they appear in the same position.
    '''
    functor = atom.functor
    args = atom.args
    res = set()
    for arg in atom.args:
        ix = args.index(arg)
        if (
            args.count(arg) == 1 and
            separator_variable_position.setdefault((functor, arg), ix) == ix
        ):
            res.add(arg)
    return res


def _assemble_separator_variables(
    separator_variable_per_clause, separator_variable_position
):
    separator_variables = set(
        v for _, v in separator_variable_position.keys()
    )
    for sv_clause in separator_variable_per_clause:
        separator_variables &= sv_clause
        if len(separator_variables) == 0:
            break
    return set(separator_variables)


class IsPureLiftedPlan(PatternWalker):
    @add_match(NonLiftable)
    def non_liftable(self, expression):
        return False

    @add_match(NAryRelationalAlgebraOperation)
    def nary(self, expression):
        return all(
            self.walk(relation)
            for relation in expression.relations
        )

    @add_match(BinaryRelationalAlgebraOperation)
    def binary(self, expression):
        return (
            self.walk(expression.relation_left) &
            self.walk(expression.relation_right)
        )

    @add_match(UnaryRelationalAlgebraOperation)
    def unary(self, expression):
        return self.walk(expression.relation)

    @add_match(...)
    def other(self, expression):
        return True


def is_pure_lifted_plan(query):
    return IsPureLiftedPlan().walk(query)


def freeze_atom(atom):
    args = (
        Constant(s.name)
        for s in atom.args
    )
    return atom.functor(*args)


def canonical_database_program(q1):
    consequent, antecedent = q1.unapply()
    return Union(tuple(
        Fact(freeze_atom(atom))
        for atom in extract_logic_atoms(antecedent)
    )), freeze_atom(consequent)


def is_contained(q1, q2):
    '''
    Computes if q1 is contained in q2. Specifically,
    for 2 non-recursive Datalog queries, computes wether
    q1←q2.
    '''
    s = Symbol.fresh()
    q1 = Implication(
        s(*q1.consequent.args), q1.antecedent
    )
    q2 = Implication(
        s(*q2.consequent.args), q2.antecedent
    )
    d_q1, frozen_head = canonical_database_program(q1)
    dp = DatalogProgram()
    for f in d_q1.formulas:
        dp.walk(f)
    dp.walk(q2)
    solution = chase.Chase(dp).build_chase_solution()
    contained = (
        frozen_head.functor in solution and
        (
            tuple(a.value for a in frozen_head.args)
            in solution[frozen_head.functor].value.unwrap()
        )
    )
    return contained


def is_contained_dnf(q1, q2):
    '''
    Computes if q1 is contained in q2. Specifically,
    for 2 non-recursive Datalog queries, computes wether
    q1←q2.
    '''
    s = Symbol.fresh()
    args1 = set(q1.consequent.args)
    args2 = set(q2.consequent.args)
    antecedent1 = convert_to_pnf_with_dnf_matrix(q1.antecedent)
    antecedent2 = convert_to_pnf_with_dnf_matrix(q2.antecedent)
    return all(
        any(
            is_contained(
                Implication(
                    s(*(args1 & extract_logic_free_variables(q1_))), q1_
                ),
                Implication(
                    s(*(args2 & extract_logic_free_variables(q2_))), q2_
                )
            )
            for q1_ in antecedent1.formulas
        )
        for q2_ in antecedent2.formulas
    )
