from ...expressions import Constant


def get_named_relation_tuples(relation):
    if isinstance(relation, Constant):
        relation = relation.value
    return set(tuple(x) for x in relation)
