from typing import AbstractSet

from ...expressions import Constant, Symbol
from ...relational_algebra import (
    ColumnStr,
    ConcatenateConstantColumn,
    NamedRelationalAlgebraFrozenSet,
    Projection,
    RelationalAlgebraSolver,
    RenameColumn,
    Selection,
    eq_,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import ProvenanceAlgebraSet


def fresh_name_relation(ra_set):
    columns = tuple(Symbol.fresh().name for _ in range(ra_set.value.arity))
    relation = NamedRelationalAlgebraFrozenSet(columns, ra_set.value)
    return Constant[AbstractSet](relation)


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
            result = Selection(result, eq_(src_cols[idx], dst_col))
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
        prob_col = ColumnStr(Symbol.fresh().name)
    # remove the probability column if it is already there
    elif prob_col in relation.value.columns:
        kept_cols = tuple(
            str2columnstr_constant(col)
            for col in relation.value.columns
            if col != prob_col
        )
        relation = Projection(relation, kept_cols)
    # add a new probability column with name `prob_col` and ones everywhere
    cst_one_probability = Constant[float](
        1.0, auto_infer_type=False, verify_type=False
    )
    relation = ConcatenateConstantColumn(
        relation, str2columnstr_constant(prob_col), cst_one_probability
    )
    relation = RelationalAlgebraSolver().walk(relation)
    return ProvenanceAlgebraSet(relation.value, prob_col)
