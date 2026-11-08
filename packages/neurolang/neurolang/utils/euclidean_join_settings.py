"""
Settings of the Euclidean join, read from ``euclidean_join.yaml``.

The file lists every setting under ``defaults`` and names *profiles*, which
change some of them: ``default`` balances time and memory, ``memory``
spends time to save memory, and is the one the interactive REPL uses. The
file is read with OmegaConf, as the engines' configuration is, and merged
into the structured config of :class:`JoinSettings`, which rejects the
settings it does not know and the values of the wrong type. An engine's
configuration (key ``euclidean_join`` of ``engines.yaml``) changes some
settings in the same way, see :func:`override`. The settings are the
attributes of :data:`SETTINGS`, which :mod:`neurolang.utils.euclidean_join`
reads when it joins; :func:`use_profile`, :func:`override` and
:func:`profile` change them.
"""

import contextlib
import dataclasses
import os
import pathlib
import typing

from omegaconf import OmegaConf

#: Environment variable naming a file of the layout of the packaged one.
CONFIG_ENV = "NEUROLANG_EUCLIDEAN_JOIN_CONFIG"
#: Environment variable naming the profile used at import time.
PROFILE_ENV = "NEUROLANG_EUCLIDEAN_JOIN_PROFILE"
#: The packaged configuration file.
CONFIG_PATH = pathlib.Path(__file__).with_name("euclidean_join.yaml")


@dataclasses.dataclass
class JoinSettings:
    """
    Settings of the Euclidean join; see ``euclidean_join.yaml``.

    The fields are those of its ``defaults``.
    """

    stencil_max_cells: int
    stencil_min_work: int
    stencil_work_per_point: int
    stencil_grid_cells_per_point: int
    stencil_count_first: bool
    grid_cells_per_point: int
    grid_min_cells: int
    grid_steps: typing.List[float]
    grid_max_coordinate: float
    grid_tolerance: float
    kdtree_chunk_pairs: int
    kdtree_sample: int
    block_pairs: int


def _merged(*configs):
    """
    Return the settings of configurations merged over the structured config.

    Parameters
    ----------
    *configs : mapping or DictConfig
        Configurations, the later ones changing the earlier ones.

    Returns
    -------
    JoinSettings
        The settings.

    Raises
    ------
    omegaconf.errors.OmegaConfBaseException
        If a configuration has a setting that is not one of
        :class:`JoinSettings`, or of the wrong type, or a setting is left
        without a value.
    """
    return OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(JoinSettings), *configs)
    )


def read_profile(name, path=None, overrides=None):
    """
    Return the settings of a profile.

    Parameters
    ----------
    name : str
        Profile name.
    path : str or pathlib.Path, optional
        Configuration file; the one of :data:`CONFIG_ENV`, or the packaged
        one, by default.
    overrides : mapping, optional
        Settings that change those of the profile.

    Returns
    -------
    JoinSettings
        The ``defaults`` changed by the profile and the *overrides*.

    Raises
    ------
    ValueError
        If the file has no such profile.
    """
    path = path or os.environ.get(CONFIG_ENV) or CONFIG_PATH
    config = OmegaConf.load(path)
    profiles = config.get("profiles") or {}
    if name not in profiles:
        raise ValueError(
            f"{path} has no profile {name!r}; its profiles are "
            f"{', '.join(map(repr, profiles))}"
        )
    return _merged(config.defaults, profiles[name] or {}, overrides or {})


#: The settings in use.
SETTINGS = read_profile(os.environ.get(PROFILE_ENV, "default"))


def _set(new):
    """Make *new* the settings in use, in :data:`SETTINGS` itself."""
    for field in dataclasses.fields(new):
        setattr(SETTINGS, field.name, getattr(new, field.name))


def use_profile(name, path=None, overrides=None):
    """
    Use the settings of a profile from now on.

    Parameters
    ----------
    name : str
        Profile name.
    path : str or pathlib.Path, optional
        Configuration file, see :func:`read_profile`.
    overrides : mapping, optional
        Settings that change those of the profile.
    """
    _set(read_profile(name, path, overrides))


def override(overrides):
    """
    Change some of the settings in use.

    Parameters
    ----------
    overrides : mapping
        The new value of each setting changed, as an engine's configuration
        gives them.

    Raises
    ------
    omegaconf.errors.OmegaConfBaseException
        If a setting is unknown or of the wrong type; the settings in use
        are then unchanged.
    """
    _set(_merged(OmegaConf.structured(SETTINGS), overrides or {}))


@contextlib.contextmanager
def profile(name, path=None):
    """
    Use the settings of a profile within a ``with`` block.

    Parameters
    ----------
    name : str
        Profile name.
    path : str or pathlib.Path, optional
        Configuration file, see :func:`read_profile`.
    """
    before = dataclasses.replace(SETTINGS)
    use_profile(name, path)
    try:
        yield SETTINGS
    finally:
        _set(before)
