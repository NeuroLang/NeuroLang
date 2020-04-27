from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker
from ...expressions import Definition, Symbol
from ...logic.expression_processing import extract_logic_predicates
from ...relational_algebra import (
    NaturalJoin,
    RenameColumns,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
)
from .cplogic_to_gm import (
    AndCPDFactory,
    BernoulliCPDFactory,
    NaryChoiceCPDFactory,
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
    return RenameColumns(
        relation,
        tuple(
            (
                str2columnstr_constant(src_arg.name),
                str2columnstr_constant(dst_arg.name),
            )
            for src_arg, dst_arg in zip(current_args, desired_args)
        ),
    )


def solve_succ_query(query_predicate, cpl_program):
    """
    Obtain the solution of a SUCC query on a CP-Logic program.

    Solving a SUCC query is a multi-step process:

        1.  First, the program is grounded. That is because the
            algorithm used to convert it to a graphical model in
            the next step only works on grounded CP-Logic programs.
            A grounding of a program is an ExpressionBlock where
            each expression is a Grounding expression. Each of this
            Grounding contains both the grounded expression itself
            and a relation representing all the replacements of the
            variables in the grounded expression. This way, we can
            represent many rules of the program using relations.

        2.  Then, a graphical model representation of that grounded
            program is constructed, using the algorithm detailed in [1]_.
            In the resulting graphical model, each ground atom in the
            grounded program is associated with a random variable.
            The random variables are represented using relations for
            compactness and make it possible to leverage relational
            algebra operations to solve queries on the graphical model,
            as is detailed in the next step. The links between the
            random variables of the graphical model anad their CPDs are
            determined by the structure of the initial program. A view
            of the initial grounded expressions is maintained because it
            is necessary for solving queries.

        3.  Finally, the query is solved from the joint probability
            distribution defined by the graphical model. Only the parts
            of the graphical model necessary to solve the query are
            calculated. The solver generates a provenance relational
            expression that is left un-evaluated (lazy calculation) and
            that is solved only at the end. This separation between the
            representation of the operations necessary to solve the query
            and the actual resolution makes it possible to choose between
            multiple solvers.

    .. [1]: Meert, Wannes, Jan Struyf, and Hendrik Blockeel. “Learning Ground
    CP-Logic Theories by Leveraging Bayesian Network Learning Techniques,”
    n.d., 30.

    """
    grounded = ground_cplogic_program(cpl_program)
    translator = CPLogicGroundingToGraphicalModelTranslator()
    gm = translator.walk(grounded)
    qpred_symb = query_predicate.functor
    qpred_args = query_predicate.args
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
            result = self.walk(ApplyCPD(rv_symb, cpd_factory, tuple()))
        elif isinstance(cpd_factory, AndCPDFactory):
            parent_values = {
                parent_rv_symb: self.walk(Marginalise(parent_rv_symb))
                for parent_rv_symb in parent_rv_symbs
            }
            result = self.walk(ApplyCPD(rv_symb, cpd_factory, parent_values))
        elif isinstance(cpd_factory, NaryChoiceCPDFactory):
            parent_values = {}
        else:
            raise NotImplementedError("Unknown CPD")
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
        the truth conditional probabilities of an AND random
        variable in the graphical model.

        Given an implication rule of the form

            Z(x, y)  <-  Q(x, y), P(x, y)

        The AND conditional probability distribution is obtained by
        applying a provenance natural join to the parent value
        relations

               Q(x, y)        P(x, y)         Z(x, y)

            _p_ | x | y     _p_ | x | y     _p_ | x | y
            ====|===|===    ============    ====|===|===
            1.0 | a | a     1.0 | a | a     1.0 | a | a
            0.0 | a | b     1.0 | b | b

        where the probabilities in the provenance column _p_ always
        are 1.0 or 0.0 because this is a deterministic CPD and all
        random variables that play a role here are boolean.

        """
        rv_symb = app_op.rv_symbol
        # retrieve the expression of the intensional rule corresponding
        # to the AND node for which we wish to calculate the probabilities
        expression = self.graphical_model.expressions.value[rv_symb]
        # map the predicate symbol of each antecedent predicate to the names
        # of its argument (TODO: handle constant terms in the arguments
        # instead of assuming all the arguments are quantified variables)
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
            # ensure the names of the columns match before the natural join
            # the renaming scheme is based on the antecedent of the
            # intensional rule attached to the AND node for which
            # we wish to compute the conditional probabilities
            parent_value = rename_columns_for_args_to_match(
                parent_value, current_args, desired_args
            )
            if result is None:
                result = parent_value
            else:
                result = NaturalJoin(result, parent_value)
        return result

    @add_match(ApplyCPD(Symbol, NaryChoiceCPDFactory, ...))
    def nary_choice_cpd_app(self, app_op):
        """
        Construct the provenance expression that calculates
        the truth probabilities of a n-ary choice random
        variable in the graphical model.

        Given the probabilistic choice for predicate symbol P

            P_i : p_1  v  ...  v  P_n : p_n  :-  T

        """
        return ProvenanceAlgebraSet(
            app_op.cpd_factory.relation.value,
            app_op.cpd_factory.probability_column,
        )
