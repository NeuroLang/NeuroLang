import operator
from typing import AbstractSet

from ..expression_walker import PatternWalker, add_match, ExpressionWalker
from ..exceptions import NeuroLangException
from ..utils import (
    NamedRelationalAlgebraFrozenSet,
)
from ..relational_algebra import (
    eq_, ColumnStr, Selection, Projection, Product, EquiJoin, NaturalJoin,
    Difference, NameColumns, RenameColumn, RelationalAlgebraSolver
)

from .expressions import (
    Constant,
    Symbol,
    Definition,
    FunctionApplication,
    ChoiceDistribution,
    RandomVariableValuePointer,
    NegateProbability,
    MultipleNaturalJoin,
    Aggregation,
    ExtendedProjection,
    ExtendedProjectionListMember,
    Unions,
    Union,
    SumRows,
)

FA_ = FunctionApplication
C_ = Constant

# TODO SemiRing exception
# TODO Define a better SemiRing class
# TODO Adapt _combine_relation_sum and _combine_relation_prod to use semiring


def arithmetic_operator_string(op):
    return {
        operator.add: "+",
        operator.sub: "-",
        operator.mul: "*",
        operator.truediv: "/",
    }[op]


def is_arithmetic_operation(exp):
    return (
        isinstance(exp, FunctionApplication) and
        isinstance(exp.functor, Constant) and exp.functor.value in {
            operator.add, operator.sub, operator.mul, operator.truediv
        }
    )


from ..relational_algebra import RelationalAlgebraOperation


# TODO This operation was included until we
# resolve the inheritance of the solver
class CrossProduct(RelationalAlgebraOperation):
    def __init__(self, relations):
        self.relations = tuple(relations)

    def __repr__(self):
        return '[' + f'\N{n-ary times operator}'.join(
            repr(r) for r in self.relations
        ) + ']'


# TODO This operation was included until we
# resolve the inheritance of the solver
class ProjectionNP(RelationalAlgebraOperation):
    def __init__(self, relation, attributes):
        self.relation = relation
        self.attributes = attributes

    def __repr__(self):
        return (
            f'\N{GREEK CAPITAL LETTER PI}'
            f'_{self.attributes}({self.relation})'
        )


class SemiRing:
    def __init__(self, distinguished_elements, binary_operators):
        if len(distinguished_elements) != 2:
            raise NeuroLangException("Length of Distinguished Elements != 2")

        if len(binary_operators) != 2:
            raise NeuroLangException("Length of Binary Operators != 2")

        self.distinguished_element_sum = distinguished_elements[0]
        self.distinguished_element_prod = distinguished_elements[1]
        self.binary_operator_sum = binary_operators[0]
        self.binary_operator_prod = binary_operators[1]


CountingSemiRing = SemiRing([C_(0), C_(1)], [operator.add, operator.mul])


class ProvenanceAlgebraSet(Constant):
    def __init__(self, algebra_set, provenance_column):
        self.algebra_set = algebra_set
        self.provenance_column = provenance_column

    @property
    def value(self):
        return self.algebra_set


class RelationalAlgebraProvenanceSolver(ExpressionWalker):
    """
    Mixing that walks through relational algebra expressions and
    executes the operations and provenance calculations.
    """

    def __init__(self, semiring, parent_values={}):
        self.parent_values = parent_values
        if not isinstance(semiring, SemiRing):
            raise NeuroLangException(
                "The semiring parameter does not belong to the SemiRing class"
            )
        self._semiring = semiring

    def _build_set_from_set(self, rel_set, provenance_column):
        return ProvenanceAlgebraSet(rel_set, provenance_column)

    def _build_set_from_iterable(self, iterable, columns, prov_col_name):
        return ProvenanceAlgebraSet(
            NamedRelationalAlgebraFrozenSet(
                columns=columns, iterable=iterable
            ), prov_col_name
        )

    def _separate_provenance(self, relation):
        new_prov = relation.value.projection(relation.provenance_column)
        non_provenance_col = set(relation.value.columns).difference(
            set([relation.provenance_column])
        )
        new_set = relation.value.projection(*non_provenance_col)

        return new_prov, new_set

    @add_match(Projection)
    def prov_projection(self, projection):
        relation = self.walk(projection.relation)
        cols = tuple(v.value for v in projection.attributes
                     ) + (relation.provenance_column, )
        projected_relation = relation.value.projection(*cols)
        return self._build_set_from_set(
            projected_relation, relation.provenance_column
        )

    @add_match(Product)
    def prov_product(self, product):
        rel_res = self.walk(product.relations[0])
        prov_res = rel_res.provenance_column
        for relation in product.relations[1:]:
            rel_temp = self.walk(relation)
            prov_temp = rel_temp.provenance_column

            column1 = f'{prov_res}1'
            column2 = f'{prov_temp}2'

            set_res = set(rel_res.value.columns
                          ).difference(set([rel_res.provenance_column]))
            set_temp = set(rel_temp.value.columns
                           ).difference(set([rel_temp.provenance_column]))
            common = set_res & set_temp
            if common:
                cols = set_res.difference(common)
                if len(cols) == 0:
                    cols = set_temp.difference(common)
                    rel_temp = Projection(
                        rel_temp, tuple([C_(ColumnStr(name)) for name in cols])
                    )
                    rel_temp = self.walk(rel_temp)
                else:
                    rel_res = Projection(
                        rel_res, tuple([C_(ColumnStr(name)) for name in cols])
                    )
                    rel_res = self.walk(rel_res)

            proj_columns = tuple(
                set(rel_res.value.columns) - set([column1])
            ) + tuple(set(rel_temp.value.columns) -
                      set([column2])) + tuple([rel_res.provenance_column])

            proj_columns = tuple([
                C_(ColumnStr(name)) for name in set(proj_columns)
            ])

            res = ProjectionNP(
                ExtendedProjection(
                    CrossProduct((
                        RenameColumn(
                            rel_res, C_(ColumnStr(rel_res.provenance_column)),
                            C_(ColumnStr(column1))
                        ),
                        RenameColumn(
                            rel_temp,
                            C_(ColumnStr(rel_temp.provenance_column)),
                            C_(ColumnStr(column2))
                        )
                    )),
                    tuple([
                        ExtendedProjectionListMember(
                            fun_exp=Constant(ColumnStr(column1)) *
                            Constant(ColumnStr(column2)),
                            dst_column=Constant(
                                ColumnStr(rel_res.provenance_column)
                            ),
                        )
                    ]),
                ), proj_columns
            )

            rel_res = self.walk(res)

        return rel_res

    @add_match(CrossProduct)
    def ra_product(self, product):
        if len(product.relations) == 0:
            return self._build_set_from_set(set(), product.provenance_column)

        res = self.walk(product.relations[0])
        prov_column = res.provenance_column
        res = res.value
        for relation in product.relations[1:]:
            res = res.cross_product(self.walk(relation).value)
        return self._build_set_from_set(res, prov_column)

    @add_match(ProjectionNP)
    def prov_projection_non_provenance(self, projection):
        relation = self.walk(projection.relation)
        cols = tuple(v.value for v in projection.attributes)
        projected_relation = relation.value.projection(*cols)
        return self._build_set_from_set(
            projected_relation, relation.provenance_column
        )

    @add_match(EquiJoin(ProvenanceAlgebraSet, ..., ProvenanceAlgebraSet, ...))
    def prov_equijoin(self, equijoin):
        raise NotImplementedError("Aggregations are not implemented.")

    @add_match(RenameColumn)
    def prov_rename_column(self, rename_column):
        relation = self.walk(rename_column.relation)
        src = rename_column.src.value
        dst = rename_column.dst.value
        new_set = relation.value
        if len(new_set) > 0:
            new_set = new_set.rename_column(src, dst)

        if src == relation.provenance_column:
            new_prov = dst
        else:
            new_prov = relation.provenance_column

        return self._build_set_from_set(new_set, new_prov)

    @add_match(NaturalJoin)
    def prov_naturaljoin(self, naturaljoin):
        rel_left = self.walk(naturaljoin.relation_left)
        rel_right = self.walk(naturaljoin.relation_right)

        res = Product((rel_left, rel_right))
        res = self.walk(res)

        _, non_prov_left = self._separate_provenance(rel_left)
        _, non_prov_right = self._separate_provenance(rel_right)
        comb = non_prov_left.naturaljoin(non_prov_right)
        op_applied = comb.naturaljoin(res.value)
        return self._build_set_from_set(
            op_applied, naturaljoin.relation_left.provenance_column
        )

    @add_match(Union)
    def prov_union(self, union_op):
        return self._build_set_from_set(
            self.walk(union_op.first).value | self.walk(union_op.second).value,
            union_op.first.provenance_column
        )

    @add_match(Unions)
    def prov_unions(self, unions_op):
        result = None

        for relation in unions_op.relations:
            if result is None:
                result = self.walk(relation).value
            else:
                result = result | self.walk(relation).value
        return self._build_set_from_set(
            result, unions_op.first.provenance_column
        )

    @add_match(SumRows(Projection))
    def prov_sum_projected_probability(self, sum_projected):

        projection = sum_projected.relation
        projected = self.walk(projection.relation)

        non_free_var = projection.attributes
        prob_column = sum_projected.provenance_column

        result = NegateProbability(projected)
        result = self.walk(result)

        mul = Constant[str]("prod")
        non_free_var = [Constant(ColumnStr(x.name)) for x in non_free_var]

        result = Aggregation(
            mul,
            result,
            tuple(non_free_var),
            Constant(ColumnStr(prob_column)),
            Constant(ColumnStr(prob_column)),
        )

        result = self.walk(result)

        result = NegateProbability(result)

        return self.walk(result)

    @add_match(Constant[AbstractSet])
    def prov_relation(self, relation):
        return relation

    @add_match(ChoiceDistribution)
    def prov_choice_distribution(self, distrib):
        return distrib

    @add_match(Aggregation)
    def prov_aggregation(self, agg_op):
        raise NotImplementedError("Aggregations are not implemented.")

    @add_match(ExtendedProjection)
    def prov_extended_projection(self, proj_op):
        relation = self.walk(proj_op.relation)
        str_arithmetic_walker = StringArithmeticWalker()
        pandas_eval_expressions = []
        for member in proj_op.projection_list:
            pandas_eval_expressions.append(
                "{} = {}".format(
                    member.dst_column.value,
                    str_arithmetic_walker.walk(self.walk(member.fun_exp)),
                )
            )
        new_container = relation.value._container.eval(
            "/n".join(pandas_eval_expressions)
        )
        return self._build_set_from_iterable(
            new_container.values, new_container.columns,
            relation.provenance_column
        )

    @add_match(FunctionApplication, is_arithmetic_operation)
    def prov_arithmetic_operation(self, arithmetic_op):
        return FunctionApplication[arithmetic_op.type](
            arithmetic_op.functor,
            tuple(self.walk(arg) for arg in arithmetic_op.args),
        )


class StringArithmeticWalker(PatternWalker):
    @add_match(Constant)
    def constant(self, cst):
        return cst.value

    @add_match(FunctionApplication(Constant(len), ...))
    def len(self, fa):
        if not isinstance(fa.args[0], Constant[AbstractSet]):
            raise NeuroLangException("Expected constant RA relation")
        return str(len(fa.args[0].value))

    @add_match(FunctionApplication, is_arithmetic_operation)
    def arithmetic_operation(self, fa):
        return "({} {} {})".format(
            self.walk(fa.args[0]),
            arithmetic_operator_string(fa.functor.value),
            self.walk(fa.args[1]),
        )