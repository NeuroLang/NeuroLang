"""The id of a streamline: one int64 for the pair (tractogram, position).

``x = (index << 32) | s``, with ``index`` the load order of the tractogram
in a :class:`~.indexing.RelationStore` (0 for the first) and ``s``
the position of the streamline in its file. With one tractogram, ``x == s``.
The relation ``streamline(x, t, s)`` decodes it in a rule (DESIGN.md, 3.1).
Both functions take a scalar or an array, and are vectorised.
"""

import numpy as np

__all__ = ["SHIFT", "decode", "encode"]

#: The bits of ``s``: a tractogram has fewer than 2^32 streamlines.
SHIFT = 32
_MASK = (1 << SHIFT) - 1


def encode(index, s):
    """``x`` of the streamlines ``s`` of the tractogram of load order
    ``index``, as int64."""
    return (np.asarray(index, np.int64) << SHIFT) | np.asarray(s, np.int64)


def decode(x):
    """``(index, s)`` of the ids ``x``: the load order of the tractogram,
    and the position of the streamline in it, as int64."""
    x = np.asarray(x, np.int64)
    return x >> SHIFT, x & _MASK
