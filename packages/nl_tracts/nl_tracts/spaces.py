"""Space tags: nl_tracts does no registration (DESIGN.md, 5)."""

import warnings
from typing import NewType

__all__ = ["Space", "SpaceMismatchError", "check_same_space"]

#: A free-form tag naming a coordinate space, e.g. ``"MNI152NLin2009cAsym"``
#: or ``"subject:100307"``. Two tags are the same space iff they are equal.
Space = NewType("Space", str)


class SpaceMismatchError(ValueError):
    """An atlas and a tractogram are not in the same space."""


def check_same_space(
    atlas_space: Space, tractogram_space: Space, allow_mismatch: bool = False
) -> None:
    """Raise :class:`SpaceMismatchError` unless the spaces are equal.

    Parameters
    ----------
    atlas_space, tractogram_space : Space
        The tags of the atlas and of the tractogram.
    allow_mismatch : bool
        The explicit override: if True, warn instead of raising. The caller
        asserts that the two are aligned by other means.
    """
    if atlas_space == tractogram_space:
        return
    message = (
        f"spaces differ: {atlas_space!r} and {tractogram_space!r}; "
        "nl_tracts does no registration"
    )
    if not allow_mismatch:
        raise SpaceMismatchError(message)
    warnings.warn(message + " (allow_mismatch=True)", stacklevel=2)
