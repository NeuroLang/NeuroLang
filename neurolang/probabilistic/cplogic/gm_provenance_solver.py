import collections
import operator
from typing import AbstractSet, Tuple

from ...datalog.expression_processing import conjunct_formulas
from ...datalog.expressions import Fact
from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker, PatternWalker
from ...expressions import Constant, Definition, FunctionApplication, Symbol
from ...logic import TRUE, Conjunction, Implication, Union
from ...logic.expression_processing import extract_logic_predicates
from ...rap_to_latex import preserve_debug_symbols
from ...relational_algebra import (
    ColumnStr,
    ConcatenateConstantColumn,
    Difference,
    NaturalJoin,
    Projection,
    RelationalAlgebraOperation,
    RelationalAlgebraSolver,
    RenameColumn,
    RenameColumns,
    Selection,
)
from ...relational_algebra import Union as RAUnion
from ...relational_algebra import str2columnstr_constant
from ...relational_algebra_provenance import (
    NaturalJoinInverse,
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
    TheOperation,
    TupleEqualSymbol,
    TupleSymbol,
    UnionOverTuples,
    ra_binary_to_nary,
)
from .cplogic_to_gm import (
    AndPlateNode,
    BernoulliPlateNode,
    CPLogicGroundingToGraphicalModelTranslator,
    NaryChoicePlateNode,
    NaryChoiceResultPlateNode,
    PlateNode,
)
from .grounding import get_grounding_predicate, ground_cplogic_program
from .program import remove_constants_from_pred

EQUAL = Constant(operator.eq)


def rename_columns_for_args_to_match(relation, src_args, dst_args):
    """
    Rename the columns of a relation so that they match the targeted args.

    Parameters
    ----------
    relation : ProvenanceAlgebraSet or RelationalAlgebraOperation
        The relation on which the renaming of the columns should happen.
    src_args : tuple of Symbols
        The predicate's arguments currently matching the columns.
    dst_args : tuple of Symbols
        New args that the naming of the columns should match.

    Returns
    -------
    RelationalAlgebraOperation
        The unsolved nested operations that apply the renaming scheme.

    """
    src_cols = list(str2columnstr_constant(arg.name) for arg in src_args)
    dst_cols = list(str2columnstr_constant(arg.name) for arg in dst_args)
    result = relation
    for dst_col in set(dst_cols):
        idxs = [i for i, c in enumerate(dst_cols) if c == dst_col]
        result = RenameColumn(result, src_cols[idxs[0]], dst_col)
        for idx in idxs[1:]:
            result = Selection(result, EQUAL(src_cols[idx], dst_col))
    return result


def build_always_true_provenance_relation(relation, prob_col=None):
    """
    Construct a provenance set from a relation with probabilities of 1
    for all tuples in the relation.

    The provenance column is named after the ``prob_col`` argument. If
    ``prob_col`` is already in the columns of the relation, it is
    removed before being re-added.

    Parameters
    ----------
    relation : NamedRelationalAlgebraFrozenSet
        The relation containing the tuples that will be in the
        resulting provenance set.
    prob_col : Constant[ColumnStr]
        Name of the provenance column that will contain constant
        probabilities of 1.

    Returns
    -------
    ProvenanceAlgebraSet

    """
    if prob_col is None:
        prob_col = str2columnstr_constant(Symbol.fresh().name)
    # remove the probability column if it is already there
    elif prob_col.value in relation.value.columns:
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
    result_args = get_grounding_predicate(query_node.expression).args
    result = rename_columns_for_args_to_match(
        marginal_provenance_expression, result_args, qpred_args
    )
    selection_pusher = SelectionOutPusher()
    result = selection_pusher.walk(result)
    union_remover = UnionRemover()
    result = union_remover.walk(result)
    result = Projection(
        result, tuple(str2columnstr_constant(arg.name) for arg in qpred_args)
    )
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(result)
    return result


def make_conjunction_rule(conjunction):
    if isinstance(conjunction, FunctionApplication):
        # enforce expression to be a conjunction
        conjunction = Conjunction((conjunction,))
    symbols = set()
    for pred in conjunction.formulas:
        symbols |= set(arg for arg in pred.args if isinstance(arg, Symbol))
    symbols = tuple(sorted(symbols))
    csqt_pred_symb = Symbol.fresh()
    return Implication(csqt_pred_symb(*symbols), conjunction,)


def add_query_to_program(query, program):
    assert isinstance(query, Conjunction)
    antecedent_preds = list()
    valued_args = list()
    csqt_args = set()
    for pred in query.formulas:
        pred, new_valued_args = remove_constants_from_pred(pred)
        antecedent_preds.append(pred)
        valued_args += new_valued_args
        csqt_args |= set(pred.args)
    const_preserving_pred = Symbol.fresh()(*(arg for arg, _ in valued_args))
    antecedent_preds.append(const_preserving_pred)
    csqt_pred = Symbol.fresh()(*sorted(csqt_args, key=lambda s: s.name))
    fact = Fact(
        const_preserving_pred.functor(*(value for _, value in valued_args))
    )
    rule = Implication(csqt_pred, Conjunction(antecedent_preds))
    program.walk(Union((fact, rule)))
    return csqt_pred


def solve_marg_query(query_predicate, evidence, cpl_program):
    """
    Calculate the result of a MARG probabilistic query.

    The MARG(Q | e) task is defined in [1]_. It calculates the conditional
    probability of a set of query predicates Q given some evidence e.

    .. [1] De Raedt, Luc, and Angelika Kimmig. “Probabilistic (Logic)
       Programming Concepts.” Machine Learning 100, no. 1 (July 1, 2015): 5–47.
       https://doi.org/10.1007/s10994-015-5494-z.

    Parameters
    ----------
    query_predicate : logic predicate
        Predicate q in query MARG[ q | e ]
    evidence : logic predicate or conjunction of logic predicates
        Evidence on which the query predicate is conditioned.

    """
    if isinstance(evidence, FunctionApplication):
        evidence = Conjunction((evidence,))
    joint_conjunction = conjunct_formulas(query_predicate, evidence)
    joint_qpred = add_query_to_program(joint_conjunction, cpl_program)
    evidence_qpred = add_query_to_program(evidence, cpl_program)
    # from .testing import inspect_resolution
    # inspect_resolution(joint_qpred, cpl_program, "/tmp/lol.tex")
    joint_result = solve_succ_query(joint_qpred, cpl_program)
    joint_cols = tuple(
        str2columnstr_constant(arg.name)
        for arg in query_predicate.args
        if isinstance(arg, Symbol)
    )
    joint_result = Projection(joint_result, joint_cols)
    evidence_result = solve_succ_query(evidence_qpred, cpl_program)
    evidence_cols = set()
    for formula in evidence.formulas:
        evidence_cols |= set(
            str2columnstr_constant(arg.name)
            for arg in formula.args
            if isinstance(arg, Symbol)
        )
    evidence_cols = tuple(sorted(evidence_cols, key=lambda c: c.value))
    evidence_result = Projection(evidence_result, evidence_cols)
    result = NaturalJoinInverse(joint_result, evidence_result)
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
                self.valued_node[0].node_symbol.name, self.valued_node[1]
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


class NodeValue(Definition):
    def __init__(self, node, value):
        self.node = node
        self.value = value


class CPLogicGraphicalModelProvenanceSolver(PatternWalker):
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

    @add_match(NodeValue(PlateNode, TRUE))
    def node_always_true(self, nv):
        prob_col = getattr(nv.node, "probability_column", None)
        prov_set = build_always_true_provenance_relation(
            nv.node.relation, prob_col
        )
        prov_set.__debug_expression__ = nv.node.expression
        prov_set.__debug_alway_true__ = True
        return prov_set

    @add_match(NodeValue(NaryChoiceResultPlateNode, TupleSymbol))
    def nary_choice_result(self, nv):
        prov_set = self.walk(NodeValue(nv.node, TRUE))
        prov_set = Selection(
            prov_set,
            TupleEqualSymbol(prov_set.non_provenance_columns, nv.value),
        )
        return prov_set

    @add_match(ProbabilityOperation((BernoulliPlateNode, TRUE), tuple()))
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

    @add_match(
        ProbabilityOperation((AndPlateNode, TRUE), ...),
        lambda exp: len(exp.condition_valued_nodes) > 0,
    )
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
        parent_relations = {
            node.node_symbol: self.walk(NodeValue(node, value))
            for node, value in operation.condition_valued_nodes
        }
        and_node = operation.valued_node[0]
        antecedent_preds = extract_logic_predicates(
            and_node.expression.antecedent
        )
        to_join = []
        for antecedent_pred in antecedent_preds:
            cnode = self.graphical_model.get_node(antecedent_pred.functor)
            src_args = get_grounding_predicate(cnode.expression).args
            dst_args = antecedent_pred.args
            parent_relation = rename_columns_for_args_to_match(
                parent_relations[cnode.node_symbol], src_args, dst_args
            )
            to_join.append(parent_relation)
        relation = ra_binary_to_nary(NaturalJoin)(to_join)
        return relation

    @add_match(ProbabilityOperation((NaryChoicePlateNode, ...), tuple()))
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
        return Selection(
            prov_set,
            TupleEqualSymbol(prov_set.non_provenance_columns, choice_value),
        )

    @add_match(
        ProbabilityOperation(
            (NaryChoiceResultPlateNode, ...), ((NaryChoicePlateNode, ...),)
        )
    )
    def nary_choice_result_truth_conditional_probability(self, operation):
        choice_node = operation.condition_valued_nodes[0][0]
        choice_value = operation.condition_valued_nodes[0][1]
        relation = build_always_true_provenance_relation(
            choice_node.relation, choice_node.probability_column,
        )
        relation.__debug_expression__ = choice_node.expression
        relation.__debug_alway_true__ = True
        relation = Selection(
            relation,
            TupleEqualSymbol(relation.non_provenance_columns, choice_value),
        )
        return relation

    @add_match(ProbabilityOperation((PlateNode, TRUE), tuple()))
    def single_node_truth_probability(self, operation):
        """
        Calculate the marginal truth probability of any node
        representing a set of boolean random variables.

        """
        the_node_symb = operation.valued_node[0].node_symbol
        # get symbolic representations of the chosen values of the choice
        # nodes on which the node depends
        chosen_tuple_symbs = {
            node_symb: TupleSymbol.fresh()
            for node_symb in self._get_choice_node_symb_deps(the_node_symb)
        }
        bernoulli_deps = self._get_bernoulli_deps(the_node_symb)
        # keep track of which nodes have been visited, to prevent their
        # CPD terms from occurring multiple times in the sum's term
        visited = set()
        symbolic_sum_term_exp = self._build_symbolic_marg_sum_term_exp(
            the_node_symb, chosen_tuple_symbs, bernoulli_deps, visited
        )
        relation = symbolic_sum_term_exp
        for choice_node_symb, tupl_symb in chosen_tuple_symbs.items():
            relation = UnionOverTuples(relation, tupl_symb)
            relation.__debug_expression__ = self.graphical_model.get_node(
                choice_node_symb
            ).expression
        return relation

    def _build_symbolic_marg_sum_term_exp(
        self, node_symb, chosen_tuple_symbs, bernoulli_deps, visited
    ):
        visited.add(node_symb)
        node = self.graphical_model.get_node(node_symb)
        expression = node.expression
        args = get_grounding_predicate(expression).args
        if isinstance(node, AndPlateNode):
            pred_symb_to_args = {
                pred.functor: pred.args
                for pred in extract_logic_predicates(expression.antecedent)
            }
            get_dst_args = pred_symb_to_args.get
        else:
            get_dst_args = lambda _: args
        cnode_symbs = self.graphical_model.get_parent_node_symbols(node_symb)
        cnodes = [self.graphical_model.get_node(cns) for cns in cnode_symbs]
        valued_cnodes = tuple(
            (cn, self._node_symbolic_value(cn.node_symbol, chosen_tuple_symbs))
            for cn in cnodes
        )
        node_value = self._node_symbolic_value(node_symb, chosen_tuple_symbs)
        node_cpd = ProbabilityOperation((node, node_value), valued_cnodes)
        relation = self.walk(node_cpd)
        if node_symb in bernoulli_deps:
            for args in bernoulli_deps[node_symb]:
                columns = tuple(
                    str2columnstr_constant(arg.name) for arg in args
                )
                relation = TheOperation(relation, node_symb, columns)
        relations = [relation]
        for cnode_symb in cnode_symbs:
            if cnode_symb in visited:
                continue
            relation = self._build_symbolic_marg_sum_term_exp(
                cnode_symb, chosen_tuple_symbs, bernoulli_deps, visited
            )
            cnode = self.graphical_model.get_node(cnode_symb)
            src_args = get_grounding_predicate(cnode.expression).args
            dst_args = get_dst_args(cnode_symb)
            if not isinstance(cnode, BernoulliPlateNode):
                relation = rename_columns_for_args_to_match(
                    relation, src_args, dst_args
                )
            relations.append(relation)
        relation = ra_binary_to_nary(NaturalJoin)(relations)
        if node_symb in bernoulli_deps:
            proj_cols = set()
            for bernoulli_args in bernoulli_deps[node_symb]:
                proj_cols |= set(
                    str2columnstr_constant(arg.name) for arg in bernoulli_args
                )
            proj_cols = tuple(proj_cols)
        else:
            proj_cols = tuple(str2columnstr_constant(arg.name) for arg in args)
        relation = Projection(relation, proj_cols)
        return relation

    def _get_choice_node_symb_deps(self, start_node_symb):
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

    def _node_symbolic_value(self, node_symb, chosen_tuple_symbs):
        node = self.graphical_model.get_node(node_symb)
        if isinstance(node, NaryChoicePlateNode):
            return chosen_tuple_symbs[node_symb]
        elif isinstance(node, NaryChoiceResultPlateNode):
            choice_node_symb = next(
                iter(self.graphical_model.get_parent_node_symbols(node_symb))
            )
            return chosen_tuple_symbs[choice_node_symb]
        return TRUE

    def _get_bernoulli_deps(self, node_symb):
        node = self.graphical_model.get_node(node_symb)
        if not isinstance(node, (AndPlateNode, BernoulliPlateNode)):
            return dict()
        apreds = extract_logic_predicates(node.expression.antecedent)
        deps = collections.defaultdict(set)
        for apred in apreds:
            apred_symb = apred.functor
            cnode = self.graphical_model.get_node(apred_symb)
            if isinstance(cnode, BernoulliPlateNode):
                deps[apred_symb].add(apred.args)
            renames = {
                x: y
                for x, y in zip(
                    get_grounding_predicate(cnode.expression).args, apred.args,
                )
            }
            for symbol, child_deps in self._get_bernoulli_deps(
                apred_symb
            ).items():
                deps[symbol] |= set(
                    tuple(renames.get(x, x) for x in child_dep)
                    for child_dep in child_deps
                )
        return deps


class ProvenanceExpressionTransformer(PatternWalker):
    @add_match(RelationalAlgebraOperation)
    def ra_operation(self, op):
        new_args, changed = self._walk_args(op.unapply())
        if changed:
            new_op = op.apply(*new_args)
            new_op = preserve_debug_symbols(op, new_op)
            return self.walk(new_op)
        else:
            return op

    def _walk_args(self, args):
        new_args = tuple()
        changed = False
        for arg in args:
            if isinstance(arg, Tuple):
                new_arg, new_changed = self._walk_args(arg)
            else:
                new_arg = self.walk(arg)
                new_changed = new_arg is not arg
            new_args += (new_arg,)
            changed |= new_changed
        return new_args, changed


class SelectionOutPusherMixin(PatternWalker):
    @add_match(
        Selection(Selection(..., TupleEqualSymbol), TupleEqualSymbol),
        lambda select: select.formula.tuple_symbol.name
        > select.relation.formula.tuple_symbol.name,
    )
    def sort_nested_selection_by_tuple_symbol(self, select):
        new_nested_select = Selection(select.relation.relation, select.formula)
        new_select = Selection(new_nested_select, select.relation.formula)
        return self.walk(new_select)

    @add_match(
        UnionOverTuples(UnionOverTuples, ...),
        lambda union: union.tuple_symbol.name
        > union.relation.tuple_symbol.name,
    )
    def sort_nested_union_over_tuples(self, union):
        nested_union = union.relation
        new_nested_union = UnionOverTuples(
            nested_union.relation, union.tuple_symbol
        )
        new_nested_union = preserve_debug_symbols(union, new_nested_union)
        new_union = UnionOverTuples(
            new_nested_union, nested_union.tuple_symbol
        )
        new_union = preserve_debug_symbols(nested_union, new_union)
        return self.walk(new_union)

    @add_match(
        UnionOverTuples(Selection(..., TupleEqualSymbol), ...),
        lambda union: union.tuple_symbol.name
        > union.relation.formula.tuple_symbol.name,
    )
    def sort_union_of_selection(self, union):
        select = union.relation
        new_union = UnionOverTuples(select.relation, union.tuple_symbol)
        new_union = preserve_debug_symbols(union, new_union)
        new_select = Selection(new_union, select.formula)
        return self.walk(new_select)

    @add_match(
        Selection(UnionOverTuples, TupleEqualSymbol),
        lambda select: select.formula.tuple_symbol.name
        > select.relation.tuple_symbol.name,
    )
    def sort_selection_of_union(self, select):
        union = select.relation
        new_select = Selection(union.relation, select.formula)
        new_union = UnionOverTuples(new_select, union.tuple_symbol)
        new_union = preserve_debug_symbols(union, new_union)
        return self.walk(new_union)

    @add_match(
        RenameColumn(
            Selection(..., TupleEqualSymbol),
            Constant[ColumnStr],
            Constant[ColumnStr],
        )
    )
    def swap_rename_selection(self, rename):
        new_rename = RenameColumn(
            rename.relation.relation, rename.src, rename.dst
        )
        new_selection_columns = tuple(
            rename.dst if col == rename.src else col
            for col in rename.relation.formula.columns
        )
        new_selection = Selection(
            new_rename,
            TupleEqualSymbol(
                new_selection_columns, rename.relation.formula.tuple_symbol
            ),
        )
        return self.walk(new_selection)

    @add_match(NaturalJoin(Selection(..., TupleEqualSymbol), ...))
    def njoin_left_selection(self, njoin):
        return Selection(
            NaturalJoin(njoin.relation_left.relation, njoin.relation_right),
            njoin.relation_left.formula,
        )

    @add_match(NaturalJoin(..., Selection(..., TupleEqualSymbol)))
    def njoin_right_selection(self, njoin):
        return Selection(
            NaturalJoin(njoin.relation_right.relation, njoin.relation_left),
            njoin.relation_right.formula,
        )

    @add_match(
        Selection(
            Selection(..., TupleEqualSymbol),
            FunctionApplication(
                EQUAL, (Constant[ColumnStr], Constant[ColumnStr])
            ),
        )
    )
    def nested_tuple_selection(self, op):
        return Selection(
            Selection(op.relation.relation, op.formula), op.relation.formula
        )

    @add_match(UnionOverTuples(Projection, ...))
    def union_of_projection(self, union):
        projection = union.relation
        new_union = UnionOverTuples(projection.relation, union.tuple_symbol)
        new_union = preserve_debug_symbols(union, new_union)
        new_projection = Projection(new_union, projection.attributes)
        return new_projection

    @add_match(Projection(Selection(..., TupleEqualSymbol), ...))
    def selection_in_projection(self, proj):
        select = proj.relation
        new_proj = Projection(select.relation, proj.attributes)
        new_select = Selection(new_proj, select.formula)
        return new_select

    @add_match(
        TheOperation(TheOperation, ..., ...),
        lambda to: to.symbol.name > to.relation.symbol.name,
    )
    def sort_the_operations(self, op):
        nested_op = op.relation
        new_nested_op = TheOperation(nested_op.relation, op.symbol, op.columns)
        new_op = TheOperation(
            new_nested_op, nested_op.symbol, nested_op.columns
        )
        return self.walk(new_op)


class UnionRemoverMixin(PatternWalker):
    @add_match(
        UnionOverTuples(Selection(..., TupleEqualSymbol), ...),
        lambda exp: exp.tuple_symbol == exp.relation.formula.tuple_symbol,
    )
    def union_of_selection_same_tuple_symbol(self, union):
        op = union.relation
        selection_cols = list()
        while (
            isinstance(op, Selection)
            and isinstance(op.formula, TupleEqualSymbol)
            and op.formula.tuple_symbol == union.tuple_symbol
        ):
            selection_cols.append(op.formula.columns)
            op = op.relation
        if len(selection_cols) == 1:
            return self.walk(union.relation.relation)
        for i in range(1, len(selection_cols)):
            for c1, c2 in zip(selection_cols[i - 1], selection_cols[i]):
                if c1 != c2:
                    op = Selection(op, EQUAL(c1, c2))
        return self.walk(op)

    @add_match(
        TheOperation(TheOperation(ProvenanceAlgebraSet, ..., ...), ..., ...)
    )
    def quadratic_the_operation(self, the_op):
        p = the_op.relation.relation
        p1 = build_always_true_provenance_relation(
            Constant[AbstractSet](p.value), p.provenance_column
        )
        p = RenameColumns(
            p, tuple(zip(p.non_provenance_columns, the_op.columns))
        )
        p1 = RenameColumns(
            p1, tuple(zip(p1.non_provenance_columns, the_op.columns))
        )
        renames = tuple(zip(the_op.columns, the_op.relation.columns))
        left = NaturalJoin(RenameColumns(p, renames), p1,)
        right = NaturalJoin(RenameColumns(p1, renames), p1)
        for old, new in renames:
            left = Difference(left, Selection(left, EQUAL(old, new)))
            right = Selection(right, EQUAL(old, new))
        p2 = RAUnion(left, right)
        relation = NaturalJoin(p, p2)
        return relation

    @add_match(TheOperation(ProvenanceAlgebraSet, ..., ...))
    def unary_the_operation(self, op):
        return RenameColumns(
            op.relation,
            tuple(zip(op.relation.non_provenance_columns, op.columns)),
        )


class SelectionOutPusher(
    SelectionOutPusherMixin, ExpressionWalker,
):
    pass


class UnionRemover(
    UnionRemoverMixin, ProvenanceExpressionTransformer, ExpressionWalker,
):
    pass
