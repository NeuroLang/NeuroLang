import itertools
from typing import AbstractSet

from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker
from ...expressions import Constant, Definition, Symbol
from ...logic.expression_processing import extract_logic_predicates
from ...relational_algebra import (
    ConcatenateConstantColumn,
    Difference,
    NaturalJoin,
    Projection,
    RelationalAlgebraSolver,
    RenameColumns,
    Union,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
)
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from .cplogic_to_gm import (
    AndPlateNode,
    BernoulliPlateNode,
    CPLogicGroundingToGraphicalModelTranslator,
    NaryChoicePlateNode,
    NaryChoiceResultPlateNode,
    ProbabilisticPlateNode,
    PlateNode,
)
from .grounding import (
    get_predicate_from_grounded_expression,
    ground_cplogic_program,
)

TRUE = Constant[bool](True, verify_type=False, auto_infer_type=False)


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

    The SUCC query must take the form

        SUCC[ P(x) ]

    where

        -   P(x) is a positive literal
        -   P is any (probabilistic, intensional, extensional)
            predicate symbol
        -   x is a universally-quantified variable

    TODO: add support for SUCC[ P(a) ] where a is a constant term.

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
    query_node = gm.get_node(qpred_symb)
    marginal_provenance_expression = solver.walk(
        ProbabilityOperation((query_node, TRUE), tuple())
    )
    result_args = get_predicate_from_grounded_expression(
        query_node.expression
    ).args
    result = rename_columns_for_args_to_match(
        marginal_provenance_expression, result_args, qpred_args
    )
    solver = RelationalAlgebraProvenanceCountingSolver()
    return solver.walk(result)


class ProbabilityOperation(Definition):
    """
    Operation representing a probability calculation on multiple sets
    of random variables (possibly with conditioning random variables).

    Attributes
    ----------
    valued_node : tuple of (PlateNode, Constant) tuples
        Plate nodes containing the random variables whose probability
        is calculated for specific values. If TRUE is
        given, it is seen as all the random variables being True.
    condition_valued_nodes : tuple of (PlateNode, Constant) tuples
        Plate nodes containing the random variables conditioning the
        evaluated probabilities and their values.

    """

    def __init__(self, valued_node, condition_valued_nodes):
        self.valued_node = valued_node
        self.condition_valued_nodes = condition_valued_nodes

    def __repr__(self):
        return "P[ {}{} ]".format(
            "{} = {}".format(
                self.valued_node[0].node_symbol.name, self.valued_node[1].value
            ),
            (
                ""
                if not self.condition_valued_nodes
                else " | "
                + ", ".join(
                    "{} = {}".format(cnode.node_symbol.name, cnode_value.value)
                    for cnode, cnode_value in self.condition_valued_nodes
                )
            ),
        )


def ra_binary_to_nary(op):
    def nary_op(relations):
        it = iter(relations)
        result = next(it)
        for relation in it:
            result = op(result, relation)
        return result

    return nary_op


def get_node_support(node):
    if isinstance(node, NaryChoiceResultPlateNode):
        return tuple(
            Constant[AbstractSet](
                NamedRelationalAlgebraFrozenSet(
                    columns=node.relation.value.columns, iterable=[tupl],
                )
            )
            for tupl in node.relation.value
        )
    elif isinstance(node, NaryChoicePlateNode):
        relation = RelationalAlgebraSolver().walk(
            Projection(
                node.relation,
                (
                    str2columnstr_constant(col)
                    for col in node.relation.value.columns
                    if col != node.probability_column.value
                ),
            )
        )
        return tuple(
            Constant[AbstractSet](
                NamedRelationalAlgebraFrozenSet(
                    columns=relation.value.columns, iterable=[tupl],
                )
            )
            for tupl in relation.value
        )
    else:
        return (TRUE,)


def is_bernoulli_truth_probability(operation):
    return (
        len(operation.condition_valued_nodes) == 0
        and isinstance(operation.valued_node[0], BernoulliPlateNode)
        and operation.valued_node[1] == TRUE
    )


def is_and_truth_conditional_probability(operation):
    return (
        len(operation.condition_valued_nodes) > 0
        and isinstance(operation.valued_node[0], AndPlateNode)
        and operation.valued_node[1] == TRUE
    )


def is_nary_choice_probability(operation):
    return (
        len(operation.condition_valued_nodes) == 0
        and isinstance(operation.valued_node[0], NaryChoicePlateNode)
        and isinstance(operation.valued_node[1], Constant)
        and operation.valued_node[1].type is AbstractSet
    )


def is_nary_choice_result_truth_conditional_probability(operation):
    return (
        len(operation.condition_valued_nodes) == 1
        and isinstance(operation.valued_node[0], NaryChoiceResultPlateNode)
        and isinstance(operation.valued_node[1], Constant)
        and operation.valued_node[1].type is AbstractSet
    )


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

    @add_match(ProbabilityOperation, is_bernoulli_truth_probability)
    def bernoulli_truth_probability(self, operation):
        """
        Construct the provenance algebra set that represents
        the truth probabilities of a set of independent
        Bernoulli-distributed random variables.

        """
        node = operation.valued_node[0]
        relation = node.relation.value
        prov_col = node.probability_column
        return ProvenanceAlgebraSet(relation, prov_col)

    @add_match(ProbabilityOperation, is_and_truth_conditional_probability)
    def and_truth_conditional_probability(self, operation):
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
        node = operation.valued_node[0]
        expression = node.expression
        # map the predicate symbol of each antecedent predicate to the symbols
        # in its arguments (TODO: handle constant terms in the arguments
        # instead of assuming all the arguments are quantified variables)
        pred_symb_to_child_args = {
            pred.functor: pred.args
            for pred in extract_logic_predicates(expression.antecedent)
        }
        result = None
        for cnode, cnode_value in operation.condition_valued_nodes:
            if cnode_value != TRUE:
                raise ValueError(
                    "Expected conditioning node to have a T value"
                )
            parent_args = get_predicate_from_grounded_expression(
                cnode.expression
            ).args
            child_args = pred_symb_to_child_args[cnode.node_symbol]
            parent_relation = cnode.relation
            if isinstance(cnode, ProbabilisticPlateNode):
                parent_relation = Projection(
                    parent_relation,
                    tuple(
                        str2columnstr_constant(col)
                        for col in parent_relation.value.columns
                        if col != cnode.probability_column.value
                    ),
                )
            # ensure the names of the columns match before the natural join
            # the renaming scheme is based on the antecedent of the
            # intensional rule attached to the AND node for which
            # we wish to compute the conditional probabilities
            parent_relation = rename_columns_for_args_to_match(
                parent_relation, parent_args, child_args
            )
            if result is None:
                result = parent_relation
            else:
                result = NaturalJoin(result, parent_relation)
        prov_col = str2columnstr_constant(Symbol.fresh().name)
        result = ConcatenateConstantColumn(
            result, prov_col, Constant[float](1.0)
        )
        result = RelationalAlgebraSolver().walk(result)
        return ProvenanceAlgebraSet(result.value, prov_col)

    @add_match(ProbabilityOperation, is_nary_choice_probability)
    def nary_choice_probability(self, operation):
        """
        Construct the provenance relation that represents the truth
        probability of a specific value of a n-ary choice random
        variable in the graphical model.

        Let the probabilistic choice for predicate symbol P be

            P_i : p_1  v  ...  v  P_n : p_n  :-  T

        where P_i = P(a_i1, ..., a_im) and a_i1, ..., a_im are
        constant terms.

        The distribution of the choice variable associated with this
        probabilistic choice is represented by the relation

            _p_ | x_1  | ... | x_m
            ====|======|=====|=====
            p_1 | a_11 | ... | a_1m
            ... | ...  | ... | ...
            p_n | a_n1 | ... | a_nm

        Given a specific value of the choice variable

             x_1  | ... | x_m
            ======|=====|=====
             a_i1 | ... | a_im

        this function returns the provenance relation

            _p_ | x_1  | ... | x_m
            ====|======|=====|=====
            p_i | a_i1 | ... | a_im

        representing that Pr[ c_P = i ] = p_i.

        """
        choice_node, choice_value = operation.valued_node
        relation = NaturalJoin(choice_node.relation, choice_value)
        # TODO: optimise by nesting non-provenance operation?
        relation = RelationalAlgebraSolver().walk(relation)
        prov_col = choice_node.probability_column
        return ProvenanceAlgebraSet(relation.value, prov_col)

    @add_match(
        ProbabilityOperation,
        is_nary_choice_result_truth_conditional_probability,
    )
    def nary_choice_result_truth_conditional_probability(self, operation):
        result_value = operation.valued_node[1]
        choice_value = operation.condition_valued_nodes[0][1]
        relation = NaturalJoin(result_value, choice_value)
        prov_col = str2columnstr_constant(Symbol.fresh().name)
        relation = ConcatenateConstantColumn(
            relation, prov_col, Constant[float](1.0),
        )
        relation = RelationalAlgebraSolver().walk(relation)
        return ProvenanceAlgebraSet(relation.value, prov_col)

    @add_match(ProbabilityOperation((PlateNode, TRUE), tuple()))
    def single_node_truth_probability(self, operation):
        start_node = operation.valued_node[0]
        (
            cpd_templates,
            cross_product_node_values,
        ) = self._construct_marg_calculation(start_node,)
        node_symb_to_cpd_template = {
            node_symb: (node_symb, cnode_symbs)
            for node_symb, cnode_symbs in cpd_templates
        }
        jpd_marg_terms = []
        for valued_nodes in itertools.product(
            *(
                tuple((node_symb, value) for value in values)
                for node_symb, values in cross_product_node_values.items()
            )
        ):
            node_symb_to_value = dict(valued_nodes)
            term = self._construct_jpd_marg_term(
                cpd_templates[0],
                node_symb_to_cpd_template,
                node_symb_to_value,
            )
            jpd_marg_terms.append(term)
        return ra_binary_to_nary(Union)(jpd_marg_terms)

    @add_match(ProbabilityOperation)
    def capture_unsolved_probability_operation(self, op):
        raise ValueError(f"Could not solve operation {op}")

    def _construct_marg_calculation(self, start_node):
        gm = self.graphical_model
        cpd_templates = list()
        cross_product_node_values = {start_node.node_symbol: {TRUE}}
        visited_node_symbs = set()
        node_symbs_stack = {start_node.node_symbol}
        while node_symbs_stack:
            node_symb = node_symbs_stack.pop()
            if node_symb in visited_node_symbs:
                continue
            visited_node_symbs.add(node_symb)
            node = gm.get_node(node_symb)
            parent_node_symbs = gm.get_parent_node_symbols(node_symb)
            node_symbs_stack |= parent_node_symbs
            cpd_templates.append((node_symb, tuple(parent_node_symbs)))
            cross_product_node_values[node_symb] = get_node_support(node)
        return cpd_templates, cross_product_node_values

    def _construct_jpd_marg_term(
        self, cpd_template, node_symb_to_cpd_template, node_symb_to_value
    ):
        node_symb, cnode_symbs = cpd_template
        node = self.graphical_model.get_node(node_symb)
        expression = node.expression
        args = get_predicate_from_grounded_expression(expression).args
        valued_node = (node, node_symb_to_value[node_symb])
        valued_cnodes = tuple(
            (
                self.graphical_model.get_node(cnode_symb),
                node_symb_to_value[cnode_symb],
            )
            for cnode_symb in cnode_symbs
        )
        term = ProbabilityOperation(valued_node, valued_cnodes)
        term = self.walk(term)
        terms = [term]
        for cnode_symb in cnode_symbs:
            cnode_cpd_template = node_symb_to_cpd_template[cnode_symb]
            cnode = self.graphical_model.get_node(cnode_symb)
            cnode_exp = cnode.expression
            cnode_args = get_predicate_from_grounded_expression(cnode_exp).args
            term = rename_columns_for_args_to_match(
                self._construct_jpd_marg_term(
                    cnode_cpd_template,
                    node_symb_to_cpd_template,
                    node_symb_to_value,
                ),
                cnode_args,
                args,
            )
            terms.append(term)
        return ra_binary_to_nary(NaturalJoin)(terms)
