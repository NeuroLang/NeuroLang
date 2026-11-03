"""Hand the relations of nl_tracts to a NeuroLang engine (DESIGN.md, 8.2).

The only module of nl_tracts that is written against NeuroLang: it needs
an engine with ``add_tuple_set(DataFrame, name=...)``, such as
``NeurolangDL`` and ``NeurolangPDL``.
"""

from typing import Any, Iterable, Optional

import pandas as pd

from .indexing import RelationStore

__all__ = ["register", "RELATION_NAMES"]

#: The relations, with the names of their columns (DESIGN.md, 3).
RELATION_NAMES = {
    "tractogram": ("t",),
    "streamline": ("t", "s"),
    "atlas": ("a",),
    "label": ("a", "l", "name", "hemi"),
    "label_term": ("a", "l", "term"),
    "opposite_hemisphere": ("h1", "h2"),
    "traverses": ("a", "t", "s", "l"),
    "endpoint": ("a", "t", "s", "e", "l"),
    "streamline_extent": ("t", "s", "axis", "lo", "hi"),
    "endpoint_coord": ("t", "s", "e", "axis", "v"),
    "label_extent": ("a", "l", "axis", "lo", "hi"),
    "streamline_length": ("t", "s", "mm"),
}


def register(
    engine: Any,
    store: RelationStore,
    only: Optional[Iterable[str]] = None,
) -> None:
    """Add the relations of ``store`` to ``engine``.

    Each relation is passed as a DataFrame to ``engine.add_tuple_set``,
    under its own name (``traverses``, ...), so that no Python tuple is
    built. Relations the store does not have are skipped;
    ``opposite_hemisphere`` is always added.

    Parameters
    ----------
    engine
        A NeuroLang query builder with ``add_tuple_set``.
    store : RelationStore
        The relations to add.
    only : iterable of str, optional
        Add only these relations.
    """
    frames = dict(store.relations)
    frames["opposite_hemisphere"] = pd.DataFrame(
        {
            "h1": ["left", "right", "none"],
            "h2": ["right", "left", "none"],
        }
    )
    for name in (frames if only is None else only):
        if name in frames:
            engine.add_tuple_set(frames[name], name=name)
