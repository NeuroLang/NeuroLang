import numpy as np

from ...expressions import Constant, Symbol
from ...relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import ProvenanceAlgebraSet
from .cplogic_to_gm import CPLogicGroundingToGraphicalModelTranslator
from .grounding import ground_cplogic_program


def build_gm(code, **sets):
    grounded = ground_cplogic_program(code, **sets)
    translator = CPLogicGroundingToGraphicalModelTranslator()
    gm = translator.walk(grounded)
    return gm


def get_named_relation_tuples(relation):
    if isinstance(relation, Constant):
        relation = relation.value
    return set(tuple(x) for x in relation)


def eq_prov_relations(pas1, pas2):
    assert isinstance(pas1, ProvenanceAlgebraSet)
    assert isinstance(pas2, ProvenanceAlgebraSet)
    assert (
        pas1.value.projection(*(c.value for c in pas1.non_provenance_columns))
    ) == (
        pas2.value.projection(*(c.value for c in pas2.non_provenance_columns))
    )
    # ensure the prov col names are different so we can join the sets
    c1 = Symbol.fresh().name
    c2 = Symbol.fresh().name
    x1 = pas1.value.rename_column(pas1.provenance_column.value, c1)
    x2 = pas2.value.rename_column(pas2.provenance_column.value, c2)
    joined = x1.naturaljoin(x2)
    probs = list(joined.projection(*(c1, c2)))
    for p1, p2 in probs:
        if not np.isclose(p1, p2):
            return False
    return True


def make_prov_set(iterable, columns):
    return ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(columns, iterable),
        str2columnstr_constant(columns[0]),
    )
