"""Shared symbols and utilities for all neuroimaging engines."""

from typing import Callable, Iterable

import nibabel as nib
import numpy as np

from neurolang.frontend import NeurolangPDL
from neurolang.regions import ExplicitVBR, ExplicitVBROverlay


def init_base_engine(
    nl: NeurolangPDL, mask: nib.Nifti1Image
) -> None:
    """Register symbols common to all neuroimaging engines.

    Parameters
    ----------
    nl :
        Fresh :class:`~neurolang.frontend.NeurolangPDL` instance.
    mask :
        MNI template image for voxel-space coordinate mapping.
    """
    nl.add_symbol(np.exp, name="exp", type_=Callable[[float], float])
    nl.add_symbol(np.log, name="log", type_=Callable[[float], float])

    @nl.add_symbol
    def agg_count(i: Iterable) -> np.int64:
        return np.int64(len(i))

    @nl.add_symbol
    def agg_create_region(
        i: Iterable, j: Iterable, k: Iterable
    ) -> ExplicitVBR:
        voxels = np.c_[i, j, k]
        return ExplicitVBR(
            voxels, mask.affine, image_dim=mask.shape
        )

    @nl.add_symbol
    def agg_create_region_overlay(
        i: Iterable, j: Iterable, k: Iterable, p: Iterable
    ) -> ExplicitVBROverlay:
        voxels = np.c_[i, j, k]
        return ExplicitVBROverlay(
            voxels, mask.affine, p, image_dim=mask.shape
        )

    @nl.add_symbol
    def startswith(prefix: str, s: str) -> bool:
        return s.startswith(prefix)

    @nl.add_symbol
    def principal_direction(
        s: ExplicitVBR, direction: str, eps: float = 1e-6
    ) -> bool:
        c_labels = ["LR", "AP", "SI"]
        s_xyz = s.to_xyz()
        cov = np.cov(s_xyz.T)
        evals, evecs = np.linalg.eig(cov)
        i = np.argmax(np.abs(evals))
        abs_max_evec = np.abs(evecs[:, i].squeeze())
        sort_dir = np.argsort(abs_max_evec)
        if (
            np.abs(
                abs_max_evec[sort_dir[-1]]
                - abs_max_evec[sort_dir[-2]]
            )
            < eps
        ):
            return False
        else:
            main_dir = c_labels[sort_dir[-1]]
        return (direction == main_dir) or (
            direction[::-1] == main_dir
        )
