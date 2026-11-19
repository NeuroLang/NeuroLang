"""
Relational algebra as linear algebra: relations over entities encoded as
sparse matrices (:mod:`.encoding`).
"""
from .encoding import (
    Axis, EntityIndex, SparseView, ViewDeclaration, build_sparse_views,
    sparse_view, stored_relation
)

__all__ = [
    "Axis", "EntityIndex", "SparseView", "ViewDeclaration",
    "build_sparse_views", "sparse_view", "stored_relation",
]
