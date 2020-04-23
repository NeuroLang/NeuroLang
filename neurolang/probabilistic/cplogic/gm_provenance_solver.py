from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker
from ...expressions import Definition, Symbol
from ...relational_algebra import NaturalJoin, RenameColumn, str2columnstr
from ...relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
)
from ...logic.expression_processing import extract_logic_predicates
from .cplogic_to_gm import (
    AndCPDFactory,
    BernoulliCPDFactory,
    CPLogicGroundingToGraphicalModelTranslator,
)
from .grounding import (
    get_predicate_from_grounded_expression,
    ground_cplogic_program,
)


def rename_columns_for_args_to_match(relation, current_args, desired_args):
    """
    Rename the columns of a relation so that they match the targeted args.

    Parameters
    ----------
    relation : ProvenanceAlgebraSet or RelationalAlgebraOperation
        The relation on which the renaming of the columns should happen.
    current_args : tuple of Symbols
        The predicate's arguments currently matching the columns.
    desired_args : tuple of Symbols
        New args that the naming of the columns should match.

    Returns
    -------
    RelationalAlgebraOperation
        The unsolved nested operations that apply the renaming scheme.

    """
    result = relation
    for src, dst in zip(current_args, desired_args):
        result = RenameColumn(
            result, str2columnstr(src.name), str2columnstr(dst.name)
        )
    return result


def solve_succ_query(query_predicate, cplogic_code, **sets):
    grounded = ground_cplogic_program(cplogic_code, **sets)
    translator = CPLogicGroundingToGraphicalModelTranslator()
    gm = translator.walk(grounded)
    qpred_symb, qpred_args = query_predicate.unapply()
    solver = CPLogicGraphicalModelProvenanceSolver(gm)
    marginal = solver.walk(Marginalise(qpred_symb))
    args = get_predicate_from_grounded_expression(
        gm.expressions.value[qpred_symb]
    ).args
    result = rename_columns_for_args_to_match(marginal, args, qpred_args)
    solver = RelationalAlgebraProvenanceCountingSolver()
    return solver.walk(result)


class ProbabilityOperation(Definition):
    """
    Operation representing probability calculations on sets of random
    variables.
    """

    def __init__(self, rv_symbol):
        self.rv_symbol = rv_symbol


class ApplyCPD(ProbabilityOperation):
    """
    Application of a conditional probability distribution.

    Attributes
    ----------
    rv_symb : Symbol
        Random variable symbol.
    cpd_factory : CPDFactory
        Object used to produce the conditional probability distributions
        from the values of the parents.
    parent_values : Dict[Symbol, ProvenanceAlgebraSet]
        Values of the parents defining the distribution.

    """

    def __init__(self, rv_symbol, cpd_factory, parent_values):
        super().__init__(rv_symbol)
        self.cpd_factory = cpd_factory
        self.parent_values = parent_values


class Marginalise(ProbabilityOperation):
    """
    Operation representing the calculation of the marginal
    probability of a set of random variables.
    """


class CPLogicGraphicalModelProvenanceSolver(ExpressionWalker):
    """
    Solver that constructs an RAP expression that calculates probabilities
    of sets of random variables in a graphical model.

    Walking a probability calculation operation will result in the
    creation of an expression with nested provenance operations on
    provenance relations.

    To calculate the actual probabilities, one must walk the resulting
    expression using a provenance solver of her choice.

    """

    def __init__(self, graphical_model):
        self.graphical_model = graphical_model

    @add_match(Marginalise)
    def marginalise(self, marg_op):
        """
        Construct the provenance expression that calculates
        the marginal distribution of a random variable in
        the graphical model.
        """
        rv_symb = marg_op.rv_symbol
        cpd_factory = self.graphical_model.cpd_factories.value[rv_symb]
        parent_rv_symbs = self.graphical_model.edges.value.get(rv_symb, set())
        if isinstance(cpd_factory, BernoulliCPDFactory):
            parent_values = tuple()
        elif isinstance(cpd_factory, AndCPDFactory):
            parent_values = {
                parent_rv_symb: self.walk(Marginalise(parent_rv_symb))
                for parent_rv_symb in parent_rv_symbs
            }
        else:
            raise NotImplementedError("Unknown CPD")
        result = self.walk(ApplyCPD(rv_symb, cpd_factory, parent_values))
        return result

    @add_match(ApplyCPD(Symbol, BernoulliCPDFactory, ...))
    def apply_bernoulli_cpd(self, app_op):
        """
        Construct the provenance algebra set that represents
        the truth probabilities of a set of independent
        Bernoulli-distributed random variables.
        """
        return ProvenanceAlgebraSet(
            app_op.cpd_factory.relation.value,
            app_op.cpd_factory.probability_column,
        )

    @add_match(ApplyCPD(Symbol, AndCPDFactory, ...))
    def apply_and_cpd(self, app_op):
        """
        Construct the provenance expression that calculates
        the truth probabilities of an AND random variable
        in the network.
        """
        rv_symb = app_op.rv_symbol
        expression = self.graphical_model.expressions.value[rv_symb]
        antecedent_pred_symb_to_args = {
            pred.functor: pred.args
            for pred in extract_logic_predicates(expression.antecedent)
        }
        result = None
        for parent_symb, parent_value in app_op.parent_values.items():
            parent_expression = self.graphical_model.expressions.value[
                parent_symb
            ]
            current_args = get_predicate_from_grounded_expression(
                parent_expression
            ).args
            desired_args = antecedent_pred_symb_to_args[parent_symb]
            parent_value = rename_columns_for_args_to_match(
                parent_value, current_args, desired_args
            )
            if result is None:
                result = parent_value
            else:
                result = NaturalJoin(result, parent_value)
        return result
