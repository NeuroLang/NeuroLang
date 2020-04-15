import itertools
from typing import AbstractSet, Mapping


from ...datalog.expressions import Implication
from ...exceptions import NeuroLangException
from ...expression_pattern_matching import add_match
from ...expression_walker import PatternWalker
from ...expressions import (
    Constant,
    ExpressionBlock,
    FunctionApplication,
    Symbol,
)
from ...logic.expression_processing import extract_logic_predicates
from ...relational_algebra import (
    ColumnStr,
    Difference,
    NaturalJoin,
    Projection,
    RelationalAlgebraSolver,
    RenameColumn,
    Selection,
    Union,
    eq_,
)
from ...utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
)
from ..expression_processing import is_probabilistic_fact
from ..expressions import (
    ChoiceDistribution,
    GraphicalModel,
    Grounding,
    ProbabilisticChoice,
    ProbabilisticPredicate,
    RandomVariableValuePointer,
    SuccQuery,
    VectorisedTableDistribution,
)
from .grounding import (
    get_predicate_from_grounded_expression,
    ground_cplogic_program,
    topological_sort_groundings,
)


def bernoulli_vect_table_distrib(p, grounding):
    if not isinstance(p, Constant[float]):
        raise NeuroLangException(
            "Bernoulli's parameter must be Constant[float]"
        )
    return VectorisedTableDistribution(
        Constant[Mapping](
            {
                Constant[bool](False): Constant[float](1.0 - p.value),
                Constant[bool](True): p,
            }
        ),
        grounding,
    )


def multi_bernoulli_vect_table_distrib(grounding):
    if not isinstance(grounding.relation, Constant[AbstractSet]):
        raise NeuroLangException(
            "Bernoulli's parameter must be Constant[AbstractSet]"
        )
    prob_col = Constant(
        ColumnStr(grounding.expression.consequent.probability.name)
    )
    prob_num_col = Constant(ColumnStr(make_numerical_col_symb().name))
    rename = RenameColumn(grounding.relation, prob_col, prob_num_col)
    return VectorisedTableDistribution(
        Constant[Mapping](
            {
                Constant[bool](True): rename,
                Constant[bool](False): NegateProbability(rename),
            }
        ),
        grounding,
    )


def probchoice_distribution(grounding, choice_rv_symb):
    """
    Given the value of a choice variable
        c_0 | c_1  | ... | c_k
        -----------------------
        p_i | a_i1 | ... | a_ik
    and the grounding of the probabilistic choice
        c_0 | c_1  | ... | c_k
        -----------------------
        p_1 | a_11 | ... | a_1k
         .     .      .     .
        p_n | a_n1 | ... | a_nk
    construct the set
        c_0 | c_1  | ... | c_k
        -----------------------
        0   | a_11 | ... | a_1k
        .      .      .     .
        1   | a_i1 | ... | a_ik
        .      .      .     .
        0   | a_n1 | ... | a_nk
    where column c_0 contains the realised boolean values of the probabilistic
    choice's head predicates given the realised choice variable.

    """
    columns = tuple(
        Constant(ColumnStr(arg.name))
        for arg in grounding.expression.predicate.args
    )
    shared_num_col = Constant(ColumnStr(make_numerical_col_symb().name))
    truth_prob = Union(
        AddRepeatedValueColumn(
            Difference(
                Projection(grounding.relation, columns),
                Projection(
                    RandomVariableValuePointer(choice_rv_symb), columns
                ),
            ),
            Constant[float](0.0),
            shared_num_col,
        ),
        AddRepeatedValueColumn(
            Projection(RandomVariableValuePointer(choice_rv_symb), columns),
            Constant[float](1.0),
            shared_num_col,
        ),
    )
    return VectorisedTableDistribution(
        Constant[Mapping](
            {
                Constant[bool](True): truth_prob,
                Constant[bool](False): NegateProbability(truth_prob),
            }
        ),
        grounding,
    )


def extensional_vect_table_distrib(grounding):
    return bernoulli_vect_table_distrib(Constant[float](1.0), grounding)


def and_vect_table_distribution(rule_grounding, parent_groundings):
    antecedent_preds = extract_logic_predicates(
        rule_grounding.expression.antecedent
    )
    to_join = tuple(
        make_rv_value_pointer(pred, parent_groundings[pred.functor])
        for pred in antecedent_preds
    )
    return MultiplyColumns(MultipleNaturalJoin(to_join))


class CPLogicToGraphicalModelTranslator(PatternWalker):
    def __init__(self):
        self.edges = dict()
        self.cpds = dict()
        self.groundings = dict()

    @add_match(
        ExpressionBlock,
        lambda block: all(
            isinstance(exp, Grounding) for exp in block.expressions
        ),
    )
    def block_of_groundings(self, block):
        for grounding in topological_sort_groundings(block.expressions):
            self.walk(grounding)
        return GraphicalModel(
            Constant[Mapping](self.edges),
            Constant[Mapping](self.cpds),
            Constant[Mapping](self.groundings),
        )

    @add_match(
        Grounding, lambda exp: isinstance(exp.expression, ProbabilisticChoice)
    )
    def probchoice_grounding(self, grounding):
        rv_symb = grounding.expression.predicate.functor
        choice_rv_symb = Symbol("__choice__{}".format(rv_symb.name))
        self.add_grounding(rv_symb, grounding)
        self.add_random_variable(
            rv_symb, probchoice_distribution(grounding, choice_rv_symb)
        )
        self.add_random_variable(choice_rv_symb, ChoiceDistribution(grounding))
        self.add_grounding(choice_rv_symb, grounding)
        self.add_edges(rv_symb, {choice_rv_symb})

    @add_match(
        Grounding,
        lambda exp: isinstance(exp.expression, Implication)
        and isinstance(exp.expression.consequent, FunctionApplication)
        and exp.expression.antecedent == Constant[bool](True),
    )
    def extensional_grounding(self, grounding):
        rv_symb = grounding.expression.consequent.functor
        self.add_grounding(rv_symb, grounding)
        self.add_random_variable(
            rv_symb, extensional_vect_table_distrib(grounding)
        )

    @add_match(
        Grounding,
        lambda grounding: (
            is_probabilistic_fact(grounding.expression)
            and len(grounding.relation.value.columns)
            == (len(grounding.expression.consequent.body.args) + 1)
        ),
    )
    def ground_probfact_grounding(self, grounding):
        rv_symb = grounding.expression.consequent.body.functor
        self.add_grounding(rv_symb, grounding)
        self.add_random_variable(
            rv_symb, multi_bernoulli_vect_table_distrib(grounding)
        )

    @add_match(Grounding, lambda exp: is_probabilistic_fact(exp.expression))
    def probfact_grounding(self, grounding):
        rv_symb = grounding.expression.consequent.body.functor
        self.add_grounding(rv_symb, grounding)
        self.add_random_variable(
            rv_symb,
            bernoulli_vect_table_distrib(
                grounding.expression.consequent.probability, grounding
            ),
        )

    @add_match(Grounding)
    def rule_grounding(self, rule_grounding):
        rv_symb = rule_grounding.expression.consequent.functor
        self.add_grounding(rv_symb, rule_grounding)
        parent_groundings = {
            predicate.functor: self.groundings[predicate.functor]
            for predicate in extract_logic_predicates(
                rule_grounding.expression.antecedent
            )
        }
        self.add_random_variable(
            rv_symb,
            and_vect_table_distribution(rule_grounding, parent_groundings),
        )
        parent_rv_symbs = {
            pred.functor
            for pred in extract_logic_predicates(
                rule_grounding.expression.antecedent
            )
        }
        self.add_edges(rv_symb, parent_rv_symbs)

    def add_edges(self, rv_symb, parent_rv_symbs):
        if rv_symb not in self.edges:
            self.edges[rv_symb] = set()
        self.edges[rv_symb] |= parent_rv_symbs

    def add_grounding(self, rv_symb, grounding):
        self.groundings[rv_symb] = grounding

    def add_random_variable(self, rv_symb, cpd_factory):
        self.check_random_variable_not_already_defined(rv_symb)
        self.cpds[rv_symb] = cpd_factory

    def check_random_variable_not_already_defined(self, rv_symb):
        if rv_symb in self.cpds:
            raise NeuroLangException(
                f"Already processed predicate symbol {rv_symb}"
            )


def succ_query(program_code, query_pred):
    grounded = ground_cplogic_program(program_code)
    gm = CPLogicToGraphicalModelTranslator().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    return solver.walk(SuccQuery(query_pred))


class QueryGraphicalModelSolver(PatternWalker):
    def __init__(self, graphical_model):
        self.graphical_model = graphical_model

    @add_match(SuccQuery)
    def succ_query(self, query):
        predicate = get_predicate_from_grounded_expression(
            self.graphical_model.groundings.value[
                query.predicate.functor
            ].expression
        )
        rv_symb = predicate.functor
        if rv_symb not in self.graphical_model.edges.value:
            marginal = self.compute_marg_distrib(rv_symb, {}, {})
        else:
            parent_symbs = self.graphical_model.edges.value[rv_symb]
            exp = self.graphical_model.groundings.value[rv_symb].expression
            if isinstance(exp, ProbabilisticChoice):
                parent_marginal_distribs = {
                    parent_symb: self.compute_marg_distrib(parent_symb, {}, {})
                    for parent_symb in parent_symbs
                }
            else:
                rule = self.graphical_model.groundings.value[
                    rv_symb
                ].expression
                parent_marginal_distribs = {
                    pred.functor: self.walk(SuccQuery(pred))
                    for pred in extract_logic_predicates(rule.antecedent)
                }
            parent_groundings = {
                parent_symb: self.graphical_model.groundings.value[parent_symb]
                for parent_symb in parent_symbs
            }
            marginal = self.compute_marg_distrib(
                rv_symb, parent_marginal_distribs, parent_groundings,
            )
        result = marginal
        for qpred_arg, marginal_arg in zip(
            query.predicate.args, predicate.args
        ):
            if isinstance(qpred_arg, Constant):
                result = Selection(
                    result,
                    eq_(
                        Constant[ColumnStr](
                            ColumnStr(
                                get_column_name_from_expression(marginal_arg)
                            )
                        ),
                        qpred_arg,
                    ),
                )
            elif qpred_arg != marginal_arg:
                result = RenameColumn(
                    result,
                    Constant(ColumnStr(marginal_arg.name)),
                    Constant(ColumnStr(qpred_arg.name)),
                )

        return ExtendedRelationalAlgebraSolver({}).walk(result)

    def compute_marg_distrib(
        self, rv_symb, parent_marg_distribs, parent_groundings
    ):
        cpd = self.graphical_model.cpds.value.get(rv_symb)
        if not parent_marg_distribs:
            return ExtendedRelationalAlgebraSolver({}).walk(cpd)
        else:
            terms = []
            for parent_values, parent_probs in self.iter_parents(
                parent_marg_distribs
            ):
                solver = ExtendedRelationalAlgebraSolver(parent_values)
                terms.append(
                    MultiplyColumns(
                        MultipleNaturalJoin(
                            (solver.walk(cpd),) + tuple(parent_probs.values())
                        )
                    )
                )
            return ExtendedRelationalAlgebraSolver({}).walk(
                SumColumns(MultipleNaturalJoin(tuple(terms)))
            )

    def iter_parents(self, parent_marg_distribs):
        parent_symbs = sorted(parent_marg_distribs, key=lambda symb: symb.name)
        parent_iterables = dict()
        for parent_symb in parent_symbs:
            if parent_symb.name.startswith("__choice__"):
                parent_iterables[parent_symb] = iter_choice_variable(
                    self.graphical_model.groundings.value[parent_symb]
                )
            else:
                parent_iterables[parent_symb] = self.iter_bool_variable(
                    parent_symb, parent_marg_distribs[parent_symb]
                )
        for parents in itertools.product(
            *[parent_iterables[parent_symb] for parent_symb in parent_symbs]
        ):
            parent_values, parent_probs = zip(*parents)
            yield (
                dict(zip(parent_symbs, parent_values)),
                dict(zip(parent_symbs, parent_probs)),
            )

    def iter_bool_variable(self, pred_symb, marg_distrib):
        grounding = self.graphical_model.groundings.value[pred_symb]
        predicate = get_predicate_from_grounded_expression(
            grounding.expression
        )
        relation = Projection(
            grounding.relation,
            tuple(Constant(ColumnStr(arg.name)) for arg in predicate.args),
        )
        true_val = AddRepeatedValueColumn(relation, Constant[int](1))
        false_val = AddRepeatedValueColumn(relation, Constant[int](0))
        true_prob = marg_distrib
        false_prob = NegateProbability(marg_distrib)
        yield true_val, true_prob
        yield false_val, false_prob


def iter_choice_variable(grounding):
    for tupl in grounding.relation.value:
        relation = Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                iterable=[tupl], columns=grounding.relation.value.columns,
            )
        )
        value = Projection(
            relation,
            tuple(
                Constant(ColumnStr(arg.name))
                for arg in grounding.expression.predicate.args
            ),
        )
        probs = RenameColumn(
            grounding.relation,
            Constant(ColumnStr(grounding.relation.value.columns[0])),
            Constant(ColumnStr(make_numerical_col_symb().name)),
        )
        yield value, probs


def make_rv_value_pointer(pred, grounding):
    rv_name = pred.functor.name
    result = RandomVariableValuePointer(rv_name)
    if is_probabilistic_fact(grounding.expression):
        grounding_expression_args = grounding.expression.consequent.body.args
    elif isinstance(grounding.expression, ProbabilisticChoice):
        grounding_expression_args = grounding.expression.predicate.args
    else:
        grounding_expression_args = [
            Symbol(col) for col in grounding.relation.value.columns
        ]
    if len(grounding_expression_args) != len(pred.args):
        raise NeuroLangException(
            "Number of args should be the same in "
            "the grounded expression and predicate"
        )
    for arg1, arg2 in zip(grounding_expression_args, pred.args):
        if isinstance(arg2, Constant):
            result = Selection(
                result, eq_(Constant[ColumnStr](ColumnStr(arg1.name)), arg2)
            )
        else:
            result = RenameColumn(
                result,
                Constant(ColumnStr(arg1.name)),
                Constant(ColumnStr(arg2.name)),
            )
    return result
