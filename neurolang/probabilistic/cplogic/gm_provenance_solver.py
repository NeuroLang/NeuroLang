import operator
from typing import Callable

from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker
from ...expressions import Constant, Definition, FunctionApplication, Symbol
from ...logic.expression_processing import extract_logic_predicates
from ...relational_algebra import (
    ColumnStr,
    ConcatenateConstantColumn,
    NaturalJoin,
    Projection,
    RelationalAlgebraOperation,
    RelationalAlgebraSolver,
    RenameColumns,
    Selection,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
)
from .cplogic_to_gm import (
    AndPlateNode,
    BernoulliPlateNode,
    CPLogicGroundingToGraphicalModelTranslator,
    NaryChoicePlateNode,
    NaryChoiceResultPlateNode,
    PlateNode,
    ProbabilisticPlateNode,
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


def build_always_true_provenance_relation(relation, prob_col):
    # remove the probability column if it is already there
    if prob_col.value in relation.value.columns:
        kept_cols = tuple(
            str2columnstr_constant(col)
            for col in relation.value.columns
            if col != prob_col.value
        )
        relation = Projection(relation, kept_cols)
    # add a new probability column with name `prob_col` and ones everywhere
    cst_one_probability = Constant[float](
        1.0, auto_infer_type=False, verify_type=False
    )
    relation = ConcatenateConstantColumn(
        relation, prob_col, cst_one_probability
    )
    relation = RelationalAlgebraSolver().walk(relation)
    return ProvenanceAlgebraSet(relation.value, prob_col)


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


class UnionOverTuples(RelationalAlgebraOperation):
    def __init__(self, relation, tuple_symbols):
        self.relation = relation
        # ( (x, \nu_1), (y, \nu_2) )
        self.tuple_symbols = tuple_symbols


class UnionOverTuplesSymbol(Symbol):
    pass


def ra_binary_to_nary(op):
    def nary_op(relations):
        it = iter(relations)
        result = next(it)
        for relation in it:
            result = op(result, relation)
        return result

    return nary_op


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
                    "{} = {}".format(cnode.node_symbol.name, cnode_value)
                    for cnode, cnode_value in self.condition_valued_nodes
                )
            ),
        )


def is_bernoulli_probability(operation):
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
    return len(operation.condition_valued_nodes) == 0 and isinstance(
        operation.valued_node[0], NaryChoicePlateNode
    )


def is_nary_choice_result_conditional_probability(operation):
    return len(operation.condition_valued_nodes) == 1 and isinstance(
        operation.valued_node[0], NaryChoiceResultPlateNode
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

    @add_match(ProbabilityOperation, is_bernoulli_probability)
    def bernoulli_probability(self, operation):
        """
        Construct the provenance algebra set that represents
        the truth probabilities of a set of independent
        Bernoulli-distributed random variables.

        """
        node = operation.valued_node[0]
        relation = node.relation.value
        prov_col = node.probability_column
        prov_set = ProvenanceAlgebraSet(relation, prov_col)
        prov_set.__debug_expression__ = node.expression
        return prov_set

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
        and_node = operation.valued_node[0]
        expression = and_node.expression
        # map the predicate symbol of each antecedent predicate to the symbols
        # in its arguments (TODO: handle constant terms in the arguments
        # instead of assuming all the arguments are quantified variables)
        pred_symb_to_child_args = {
            pred.functor: pred.args
            for pred in extract_logic_predicates(expression.antecedent)
        }
        result = None
        for cnode, cnode_value in operation.condition_valued_nodes:
            parent_args = get_predicate_from_grounded_expression(
                cnode.expression
            ).args
            child_args = pred_symb_to_child_args[cnode.node_symbol]
            if isinstance(cnode, ProbabilisticPlateNode):
                prob_col = cnode.probability_column
            else:
                prob_col = str2columnstr_constant(Symbol.fresh().name)
            parent_relation = build_always_true_provenance_relation(
                cnode.relation, prob_col,
            )
            parent_relation.__debug_expression__ = cnode.expression
            parent_relation.__debug_alway_true__ = True
            if isinstance(cnode, NaryChoiceResultPlateNode):
                parent_relation = _choice_tuple_selection(
                    parent_relation, cnode_value,
                )
            # ensure the names of the columns match before the natural join.
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
        return result

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
        choice_node = operation.valued_node[0]
        choice_value = operation.valued_node[1]
        prov_set = ProvenanceAlgebraSet(
            choice_node.relation.value, choice_node.probability_column,
        )
        prov_set.__debug_expression__ = choice_node.expression
        return _choice_tuple_selection(prov_set, choice_value)

    @add_match(
        ProbabilityOperation, is_nary_choice_result_conditional_probability,
    )
    def nary_choice_result_truth_conditional_probability(self, operation):
        choice_node = operation.condition_valued_nodes[0][0]
        choice_value = operation.condition_valued_nodes[0][1]
        relation = build_always_true_provenance_relation(
            choice_node.relation, choice_node.probability_column,
        )
        relation.__debug_expression__ = choice_node.expression
        relation.__debug_alway_true__ = True
        relation = _choice_tuple_selection(relation, choice_value)
        return relation

    @add_match(ProbabilityOperation((PlateNode, TRUE), tuple()))
    def single_node_truth_probability(self, operation):
        the_node_symb = operation.valued_node[0].node_symbol
        # get symbolic representations of the chosen values of the choice
        # nodes on which the node depends
        chosen_tuple_symbs = self._build_chosen_tuple_symbols(
            self._get_choice_node_symb_dependencies(the_node_symb)
        )
        # keep track of which nodes have been visited, to prevent their
        # CPD terms from occurring multiple times in the sum's term
        visited = set()
        symbolic_sum_term_exp = self._build_symbolic_marg_sum_term_exp(
            the_node_symb, chosen_tuple_symbs, visited
        )
        relation = symbolic_sum_term_exp
        for choice_node_symb, tuple_symbols in chosen_tuple_symbs.items():
            relation = UnionOverTuples(relation, tuple_symbols)
            relation.__debug_expression__ = self.graphical_model.get_node(
                the_node_symb
            ).expression
        return relation

    def _build_symbolic_marg_sum_term_exp(
        self, node_symb, chosen_tuple_symbs, visited
    ):
        visited.add(node_symb)
        node = self.graphical_model.get_node(node_symb)
        args = get_predicate_from_grounded_expression(node.expression).args
        cnode_symbs = self.graphical_model.get_parent_node_symbols(node_symb)
        cnodes = [self.graphical_model.get_node(cns) for cns in cnode_symbs]
        valued_cnodes = tuple(
            (cn, self._get_node_value(cn.node_symbol, chosen_tuple_symbs))
            for cn in cnodes
        )
        node_value = self._get_node_value(node_symb, chosen_tuple_symbs)
        node_cpd = ProbabilityOperation((node, node_value), valued_cnodes)
        node_cpd = self.walk(node_cpd)
        relations = [node_cpd]
        for cnode_symb in cnode_symbs:
            if cnode_symb in visited:
                continue
            relation = self._build_symbolic_marg_sum_term_exp(
                cnode_symb, chosen_tuple_symbs, visited
            )
            cnode = self.graphical_model.get_node(cnode_symb)
            cargs = get_predicate_from_grounded_expression(
                cnode.expression
            ).args
            relation = rename_columns_for_args_to_match(relation, cargs, args)
            relations.append(relation)
        return ra_binary_to_nary(NaturalJoin)(relations)

    def _get_choice_node_symb_dependencies(self, start_node_symb):
        """
        Retrieve the symbols of choice nodes a given node depends on.

        """
        choice_node_symbs = set()
        visited = set()
        stack = [start_node_symb]
        while stack:
            node_symb = stack.pop()
            if node_symb in visited:
                continue
            visited.add(node_symb)
            parent_node_symbs = self.graphical_model.get_parent_node_symbols(
                node_symb
            )
            stack += sorted(list(parent_node_symbs), key=lambda s: s.name)
            node = self.graphical_model.get_node(node_symb)
            if isinstance(node, NaryChoicePlateNode):
                choice_node_symbs.add(node_symb)
        return choice_node_symbs

    def _get_node_value(self, node_symb, chosen_tuple_symbs):
        node = self.graphical_model.get_node(node_symb)
        if isinstance(node, NaryChoicePlateNode):
            return chosen_tuple_symbs[node_symb]
        elif isinstance(node, NaryChoiceResultPlateNode):
            choice_node_symb = next(
                iter(self.graphical_model.get_parent_node_symbols(node_symb))
            )
            return chosen_tuple_symbs[choice_node_symb]
        return TRUE

    def _build_chosen_tuple_symbols(self, choice_node_symbs):
        """
        Generate a symbolic representation for the chosen tuple of a
        choice random variable.

        """
        result = dict()
        for choice_node_symb in choice_node_symbs:
            node = self.graphical_model.get_node(choice_node_symb)
            args = get_predicate_from_grounded_expression(node.expression).args
            result[choice_node_symb] = tuple(
                (arg, UnionOverTuplesSymbol.fresh()) for arg in args
            )
        return result

    @add_match(ProbabilityOperation)
    def capture_unsolvable_probability_op(self, op):
        raise RuntimeError(f"Cannot solve operation: {op}")


def _choice_tuple_selection(prov_set, chosen_tuple_symbs):
    for arg, symbol in chosen_tuple_symbs:
        args = (str2columnstr_constant(arg.name), symbol)
        selection_formula = FunctionApplication(Constant(operator.eq), args)
        prov_set = Selection(prov_set, selection_formula)
    return prov_set


class ProvenanceExpressionSimplifier(ExpressionWalker):
    @add_match(
        RenameColumns(
            Selection(
                ...,
                FunctionApplication(
                    Constant(operator.eq), (Constant[ColumnStr], Symbol),
                ),
            ),
            ...,
        )
    )
    def swap_rename_selection(self, rename):
        renames = dict(rename.renames)
        selection = rename.relation
        selection_col, selection_cst = selection.formula.args
        new_selection_col = renames.get(selection_col, selection_col)
        new_rename = RenameColumns(selection.relation, rename.renames)
        new_rename = self.walk(new_rename)
        new_selection = Selection(
            new_rename,
            FunctionApplication[selection.formula.type](
                selection.formula.functor, (new_selection_col, selection_cst),
            ),
        )
        return new_selection

    @add_match(
        NaturalJoin(
            Selection(
                ...,
                FunctionApplication(
                    Constant(operator.eq),
                    (Constant[ColumnStr], UnionOverTuplesSymbol),
                ),
            ),
            ...,
        )
    )
    def natural_join_selection_with_union_symbol_rhs_left(self, njoin):
        return Selection(
            self.walk(
                NaturalJoin(njoin.relation_left.relation, njoin.relation_right)
            ),
            njoin.relation_left.formula,
        )

    @add_match(
        NaturalJoin(
            ...,
            Selection(
                ...,
                FunctionApplication(
                    Constant(operator.eq),
                    (Constant[ColumnStr], UnionOverTuplesSymbol),
                ),
            ),
        )
    )
    def natural_join_selection_with_union_symbol_rhs_right(self, njoin):
        return Selection(
            self.walk(
                NaturalJoin(njoin.relation_left, njoin.relation_right.relation)
            ),
            njoin.relation_right.formula,
        )

    @add_match(Selection(Selection, ...))
    def merge_selections_with_same_rhs(self, op):
        formulas_grpby_rhs = dict()
        while isinstance(op, Selection):
            rhs = op.formula.args[1]
            if rhs not in formulas_grpby_rhs:
                formulas_grpby_rhs[rhs] = set()
            formulas_grpby_rhs[rhs].add(op.formula)
            op = op.relation
        for rhs, formulas in formulas_grpby_rhs.items():
            formula_it = iter(formulas)
            prev_formula = next(formula_it)
            if len(formulas) == 1:
                op = Selection(op, prev_formula)
            else:
                for formula in formula_it:
                    op = Selection(
                        op,
                        FunctionApplication(
                            Constant(operator.eq),
                            (prev_formula.args[0], formula.args[0]),
                        ),
                    )
                    prev_formula = formula
        return op

    @add_match(
        UnionOverTuples(
            Selection(
                ...,
                FunctionApplication(
                    Constant(operator.eq),
                    (Constant[ColumnStr], Constant[ColumnStr]),
                ),
            ),
            ...,
        )
    )
    def move_selection_outside_union(self, op):
        selection = op.relation
        new_union = UnionOverTuples(selection.relation, op.tuple_symbols)
        new_union.__debug_expression__ = op.__debug_expression__
        return Selection(new_union, selection.formula)

    @add_match(RelationalAlgebraOperation)
    def ra_operation(self, op):
        new_op = op.apply(*(self.walk(arg) for arg in op.unapply()))
        new_op.__debug_expression__ = getattr(op, "__debug_expression__", None)
        if new_op == op:
            return new_op
        else:
            return self.walk(new_op)
