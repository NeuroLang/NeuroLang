from itertools import product
from operator import eq

from ..expression_walker import (
    ExpressionWalker,
    ReplaceExpressionWalker,
    add_match,
)
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import Conjunction, Disjunction, ExistentialPredicate, Implication
from ..logic.expression_processing import extract_logic_atoms
from ..logic.unification import most_general_unifier
from .probabilistic_ra_utils import (
    generate_probabilistic_symbol_table_for_query,
    is_atom_a_probabilistic_choice_relation,
)
from .transforms import convert_rule_to_ucq, convert_ucq_to_ccq


class SelfjoinChoiceSimplification(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(Conjunction)
    def match_conjunction(self, conjunction):

        choice_atoms = GetChoiceInConjunctionOrExistential(
            self.symbol_table
        ).walk(conjunction)

        seen = set()
        dups_functors = [
            a.functor
            for a in choice_atoms
            if a.functor in seen or seen.add(a.functor)
        ]
        if len(dups_functors) > 0:
            for functor in dups_functors:
                match_atoms = [a for a in choice_atoms if a.functor == functor]

                replacements = {}
                eq_vars = []
                # product between choices
                for i1, atom1 in enumerate(match_atoms):
                    for i2, atom2 in enumerate(match_atoms):
                        if i1 >= i2 or atom1.functor != atom2.functor:
                            continue

                        new_atom, eq_var = translate_with_mgu(atom1, atom2)
                        # new_atom can't be false. Check it.
                        replacements[atom1] = new_atom
                        replacements[atom2] = new_atom
                        eq_vars.append(eq_var)

                conjunction = ReplaceExpressionInConjunctionWalker(
                    replacements
                ).walk(conjunction)

                conj = NestedExistentialChoiceSimplification(
                    self.symbol_table, eq_vars
                ).walk(conjunction)

        return conjunction


class EqualityVarsDetection(ExpressionWalker):
    @add_match(Conjunction)
    def match_conjuntion(self, conjunction):
        vars = tuple()
        for formula in conjunction.formulas:
            eq_vars = self.walk(formula)
            if eq_vars:
                vars += (eq_vars,)

        return vars

    @add_match(ExistentialPredicate)
    def match_existential(self, existential):
        return self.walk(existential.body)

    @add_match(FunctionApplication, lambda fa: fa.functor == Constant(eq))
    def match_equality(self, equality):
        return list(equality._symbols)

    @add_match(...)
    def no_match(self, _):
        return tuple()


class NestedExistentialChoiceSimplification(ExpressionWalker):
    def __init__(self, symbol_table, eq_vars):
        self.symbol_table = symbol_table
        self.eq_vars = eq_vars

    @add_match(Conjunction)
    def match_conjuntion(self, conjunction):
        evd = EqualityVarsDetection()
        forms = tuple()
        for formula in conjunction.formulas:
            eq_vars = evd.walk(formula)
            for vars_pair in eq_vars:
                if vars_pair in self.eq_vars:
                    forms += ReplaceNestedExistential(vars_pair).walk(formula)
                else:
                    forms += formula

        return Conjunction(forms)

    @add_match(ExistentialPredicate)
    def match_existential(self, existential):
        evd = EqualityVarsDetection()
        eq_vars = evd.walk(existential.body)
        for vars_pair in eq_vars:
            if vars_pair in self.eq_vars:
                existential = ReplaceNestedExistential(vars_pair).walk(
                    existential
                )

        return existential


class ReplaceNestedExistential(ExpressionWalker):
    def __init__(self, eq_vars):
        self.eq_vars = eq_vars

    @add_match(Conjunction)
    def match_conjuntion(self, conjunction):
        forms = tuple()
        for formula in conjunction.formulas:
            forms += self.walk(formula)

        return Conjunction(forms)

    @add_match(ExistentialPredicate)
    def match(self, existential):
        if existential.head in self.eq_vars and isinstance(
            existential.body, ExistentialPredicate
        ):
            if existential.body.head in self.eq_vars:
                inner_formula = self.walk(existential.body.body)
                existential = ExistentialPredicate(
                    self.eq_vars[1], inner_formula
                )

        return existential

    @add_match(FunctionApplication, lambda fa: fa.functor == Constant(eq))
    def match_equality(self, _):
        return ()

    @add_match(FunctionApplication)
    def match_function_application(self, fa):
        new_args = tuple()
        for arg in fa.args:
            if arg == self.eq_vars[0]:
                new_args += (self.eq_vars[1],)
            else:
                new_args += arg

        return fa.functor(*new_args)

    @add_match(...)
    def no_match(self, expression):
        return expression


class ReplaceExpressionInConjunctionWalker(ExpressionWalker):
    def __init__(self, replacements):
        self.replacements = replacements

    @add_match(Conjunction)
    def match_conjuntion(self, conjunction):
        forms = tuple()
        for formula in conjunction.formulas:
            new_formula = self.walk(formula)
            forms += (new_formula,)

        return Conjunction(forms)

    @add_match(ExistentialPredicate)
    def match_existential(self, existential):
        match = self.walk(existential.body)
        return ExistentialPredicate(existential.head, match)

    @add_match(FunctionApplication)
    def match_function(self, func_app):
        if func_app in self.replacements:
            # as replacements can never be Constant(False),
            # it always returns a tuple.
            # I should modify it in the initial method
            return Conjunction(self.replacements[func_app])

        return func_app


class GetChoiceInConjunctionOrExistential(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(Conjunction)
    def match_conjuntion(self, conjunction):
        p_choices = tuple()
        for formula in conjunction.formulas:
            match = self.walk(formula)
            if match:
                p_choices += match

        return p_choices

    @add_match(ExistentialPredicate)
    def match_existential(self, existential):
        match = self.walk(existential.body)
        return match

    @add_match(FunctionApplication)
    def match_function(self, func_app):
        if is_atom_a_probabilistic_choice_relation(
            func_app, self.symbol_table
        ):
            return (func_app,)

        return tuple()

    @add_match(...)
    def no_match(self, _):
        return tuple()


class ReplaceFunctionApplicationArgsWalker(ExpressionWalker):
    def __init__(self, symbol_replacements):
        self.symbol_replacements = symbol_replacements

    @add_match(FunctionApplication)
    def replace_args(self, symbol):
        new_args = tuple()
        old_vars = self.symbol_replacements.keys()
        for arg in symbol.args:
            if arg in old_vars:
                new_args += (self.symbol_replacements[arg],)
            else:
                new_args += (arg,)

        return symbol.functor(*new_args)


def pchoice_constants_as_head_variables(query, cpl_program):
    """
    First step of the antishattering strategy.
    Given an implication, body constants that are associated with
    probabilistic choices are removed and replaced by variables
    that are also included in the consequent.

    Parameters
    ----------
    query : Implication
        Query to be transformed.
    cpl_program : CPLogicProgram
        CP-Logic program on which the query should be solved.

    Returns
    -------
    Implication
        Implication with the constants of the pchoices
        replaced by variables, which are also added in the consequent.
    """
    symbol_table = generate_probabilistic_symbol_table_for_query(
        cpl_program, query.antecedent
    )
    query_ucq = convert_rule_to_ucq(query)
    query_ucq = convert_ucq_to_ccq(query_ucq, transformation="DNF")

    query_ucq = remove_selfjoins_between_pchoices(query_ucq, symbol_table)

    parameter_variable = {}
    args_amount = {}

    for clause in query_ucq.formulas:
        for atom in extract_logic_atoms(clause):
            if not isinstance(
                atom, Constant
            ) and is_atom_a_probabilistic_choice_relation(atom, symbol_table):
                for i, arg in enumerate(atom.args):
                    if isinstance(arg, Constant):
                        if atom not in parameter_variable:
                            parameter_variable[atom] = set([(i, arg)])
                            args_amount[atom.functor] = len(
                                [
                                    arg
                                    for arg in atom.args
                                    if isinstance(arg, Constant)
                                ]
                            )
                        else:
                            tmp = parameter_variable[atom]
                            tmp.add((i, arg))
                            parameter_variable[atom] = tmp

    new_atoms = tuple()
    rewritten_atoms = {}
    for atom, var_pos in parameter_variable.items():
        n_args = args_amount[atom.functor]
        filtered = [
            clean_tuple(elem)
            for elem in product(var_pos, repeat=n_args)
            if ordered_atoms(elem, atom)
        ]
        new_atom = Symbol.fresh()
        cpl_program.add_extensional_predicate_from_tuples(new_atom, filtered)
        atom_new_vars = [Symbol.fresh() for _ in range(len(filtered[0]))]

        new_atoms += tuple([new_atom(*atom_new_vars)])
        rewritten_atoms[atom] = ReplaceFunctionApplicationArgsWalker(
            dict(zip(*filtered, atom_new_vars))
        ).walk(atom)

    query = ReplaceExpressionWalker(rewritten_atoms).walk(query)

    query = Implication(
        query.consequent, Conjunction((*new_atoms, query.antecedent))
    )

    return query


def remove_selfjoins_between_pchoices(rule_dnf, symbol_table):
    """
    Verify the existence of selfjoins between pchoices.
    Parameters
    ----------
    rule_dnf : bool
        Rule to analyze in DNF form.
    symbol_table : Mapping
        Mapping from symbols to probabilistic
        or deterministic sets to solve the query.
    Returns
    -------
    bool
        True if there is a join between pchoices
        in the same conjunction
    """

    disj_formulas = []
    for formula in rule_dnf.formulas:
        conj_formulas = tuple()
        atoms = extract_logic_atoms(formula)
        for i1, atom1 in enumerate(atoms):
            if (
                is_atom_a_probabilistic_choice_relation(atom1, symbol_table)
                and len(atoms) > 1
            ):
                for i2, atom2 in enumerate(atoms):
                    if i1 >= i2 or atom1.functor != atom2.functor:
                        continue

                    new_atom, _ = translate_with_mgu(atom1, atom2)
                    conj_formulas += new_atom
            else:
                conj_formulas += (atom1,)

        disj_formulas.append(Conjunction(conj_formulas))

    return Disjunction(tuple(disj_formulas))


def translate_with_mgu(atom1, atom2):
    mgu = most_general_unifier(atom1, atom2)
    if mgu is None:
        return (Constant(False),), tuple()
    else:
        res = tuple()
        eq_vars = tuple()
        for var1, var2 in mgu[0].items():
            if isinstance(var1, Constant) or isinstance(var2, Constant):
                return (atom1, atom2), tuple()
            else:
                res += (Constant(eq)(var1, var2),)
                eq_vars += (var1, var2)

        res += (mgu[1],)
        return (Conjunction(res),), eq_vars


def ordered_atoms(product, atom):
    for pos, value in product:
        if atom.args[pos] != value:
            return False

    return True


def clean_tuple(atoms):
    return tuple([atom[1].value for atom in atoms])
