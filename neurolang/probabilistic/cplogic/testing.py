from ...expressions import Constant
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
