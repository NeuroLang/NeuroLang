from abc import abstractmethod, abstractproperty
from contextlib import contextmanager
from multiprocessing import BoundedSemaphore
from pathlib import Path
from typing import Callable, Iterable, Union

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import datasets, image

from neurolang.frontend import NeurolangDL, NeurolangPDL
from neurolang.frontend.neurosynth_utils import (
    StudyID,
    get_ns_mni_peaks_reported,
    get_ns_term_study_associations
)
from neurolang.regions import ExplicitVBR, ExplicitVBROverlay, region_union


class NeurolangEngineSet:
    """
    Utility class to hold a set of Neurolang engines (either NeurolangDL
    or NeurolangPDL). It keeps track of the total number of engines added
    to the set, as well as a semaphore to manage acquiring and releasing the
    engines in a safe concurrent manner.
    """

    def __init__(self, engine: Union[NeurolangDL, NeurolangPDL]) -> None:
        self.engines = set((engine,))
        self.counter = 1
        self.sema = BoundedSemaphore(value=1)

    def add_engine(self, engine: Union[NeurolangDL, NeurolangPDL]) -> None:
        """
        Add an engine to the set and increase the total number of engines
        in the set.

        Parameters
        ----------
        engine : Union[NeurolangDL, NeurolangPDL]
            the engine to add.
        """
        self.engines.add(engine)
        self.counter += 1
        self.sema = BoundedSemaphore(value=self.counter)

    @contextmanager
    def engine(self, timeout: int = None):
        """
        ContextManager to safely acquire an engine from the set.
        This method will block on the semaphore until an engine can be
        safely used from this set, and then yield it.
        At the end, the engine is put back in the set and the semaphore
        value is released.

        Parameters
        ----------
        timeout : int
            When invoked with a timeout other than None, it will block for
            at most timeout seconds.

        Yields
        -------
        Union[NeurolangDL, NeurolangPDL]
            an engine.
        """
        lock = self.sema.acquire(timeout=timeout)
        if lock:
            try:
                engine = self.engines.pop()
                yield engine
            finally:
                self.engines.add(engine)
                self.sema.release()
        else:
            yield None


class NeurolangEngineConfiguration:
    """
    A NeurolangEngineConfiguration is a combination of an id key and
    a method which returns a Neurolang instance.
    """

    @abstractproperty
    def key(self):
        pass

    @abstractproperty
    def atlas(self):
        pass

    @abstractmethod
    def create(self) -> Union[NeurolangDL, NeurolangPDL]:
        pass

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NeurolangEngineConfiguration):
            return super().__eq__(other)
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)


class NeurosynthEngineConf(NeurolangEngineConfiguration):
    def __init__(self, data_dir: Path, resolution=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.resolution = resolution
        self._mni_atlas = None
        self._mni_brain_mask = None

    @property
    def key(self):
        return "neurosynth"

    @property
    def atlas(self):
        if self._mni_atlas is None:
            self._mni_atlas = nib.load(
                datasets.fetch_icbm152_2009(
                    data_dir=str(self.data_dir / "icbm")
                )["t1"]
            )
        return self._mni_atlas

    @property
    def brain_mask(self):
        if self._mni_brain_mask is None:
            self._mni_brain_mask = nib.load(
                datasets.fetch_icbm152_2009(
                    data_dir=str(self.data_dir / "icbm")
                )["mask"]
            )
        return self._mni_brain_mask

    def create(self) -> NeurolangPDL:
        mask = self.brain_mask
        if self.resolution is not None:
            mask = image.resample_img(mask, np.eye(3) * self.resolution)
        nl = init_frontend(mask)
        add_ploting_functions(nl)
        load_neurosynth_data(self.data_dir, nl, mask)
        return nl


class NeurosynthCETEngineConf(NeurosynthEngineConf):
    def __init__(self, data_dir: Path, resolution=None) -> None:
        super().__init__(data_dir, resolution=resolution)

    @property
    def key(self):
        return "neurosynth_cet"

    def create(self) -> NeurolangPDL:
        mask = self.brain_mask
        if self.resolution is not None:
            mask = image.resample_img(mask, np.eye(3) * self.resolution)
        nl = init_frontend(mask)
        add_ploting_functions(nl)
        load_neurosynth_data_cet(self.data_dir, nl, mask)
        load_destrieux_atlas(self.data_dir, nl)
        nl.execute_datalog_program = (
            lambda query: nl.execute_controlled_english_program(
                query,
                type_symbol_names={
                    "tfidf", "probability", "quantity",
                    "studyid", "tuple", "value", "tupl"
                }
            )
        )

        self.set_completion_engine(nl)
        return nl

    def set_completion_engine(self, nl):
        def get_completions(query, line, character):
            lines = query.split("\n")
            new_query = '\n'.join(lines[:line])
            if line < len(lines):
                new_line = lines[line][:character]
                new_query += '\n' + new_line
            return nl.completion_for_controlled_english_program(new_query)
        nl.get_completions = get_completions


def load_neurosynth_data(data_dir: Path, nl, mni_mask: nib.Nifti1Image):
    term_data = get_ns_term_study_associations(data_dir / "neurosynth", convert_study_ids=True)
    peak_data = get_ns_mni_peaks_reported(data_dir / "neurosynth", convert_study_ids=True)
    study_id = peak_data[["id"]].drop_duplicates()

    nl.add_tuple_set(peak_data, name="PeakReported")
    nl.add_tuple_set(study_id, name="Study")
    nl.add_tuple_set(term_data, name="TermInStudyTFIDF")
    nl.add_uniform_probabilistic_choice_over_set(
        study_id, name="SelectedStudy"
    )
    nl.add_tuple_set(
        np.round(
            nib.affines.apply_affine(
                mni_mask.affine,
                np.transpose(mni_mask.get_fdata().astype(int).nonzero()),
            )
        ).astype(int),
        name="Voxel",
    )


def load_neurosynth_data_cet(data_dir: Path, nl, mni_mask: nib.Nifti1Image):
    term_data = get_ns_term_study_associations(data_dir / "neurosynth", convert_study_ids=True)
    peak_data = get_ns_mni_peaks_reported(data_dir / "neurosynth", convert_study_ids=True)
    study_id = peak_data[["id"]].drop_duplicates()
    term = term_data[["term"]].drop_duplicates()
    focus = peak_data[["x", "y", "z"]].drop_duplicates()

    nl.add_tuple_set(term, name="term")
    nl.add_tuple_set(focus, name="focus")
    nl.add_tuple_set(peak_data[["id", "x", "y", "z"]], name="report")
    nl.add_tuple_set(term_data[["id", "term", "tfidf"]], name="mention")
    nl.add_tuple_set(study_id, name="study")
    nl.add_uniform_probabilistic_choice_over_set(
        study_id, name="selected studi"
    )

    nl.add_tuple_set(
        np.round(
            nib.affines.apply_affine(
                mni_mask.affine,
                np.transpose(mni_mask.get_fdata().astype(int).nonzero()),
            )
        ).astype(int),
        name="voxel",
    )


def init_frontend(mni_mask):
    """
    Create a Neurolang Probabilistic engine and add some aggregation methods.

    Returns
    -------
    NeurolangPDL
        the Neurolang engine
    """
    nl = NeurolangPDL()

    @nl.add_symbol
    def percentile95(values: Iterable) -> float:
        """
        Aggregation which produces the 95th percentile
        of a set of values.
        """
        return np.percentile(values, 95)

    @nl.add_symbol
    def agg_create_region_ijk(
        i: Iterable, j: Iterable, k: Iterable
    ) -> ExplicitVBR:
        voxels = np.c_[i, j, k]
        return ExplicitVBR(voxels, mni_mask.affine, image_dim=mni_mask.shape)

    @nl.add_symbol
    def agg_create_region_overlay_ijk(
        i: Iterable, j: Iterable, k: Iterable, p: Iterable
    ) -> ExplicitVBR:
        voxels = np.c_[i, j, k]
        return ExplicitVBROverlay(
            voxels, mni_mask.affine, p, image_dim=mni_mask.shape
        )

    @nl.add_symbol
    def agg_create_region(
        x: Iterable, y: Iterable, z: Iterable
    ) -> ExplicitVBR:
        mni_coords = np.c_[x, y, z]
        voxels = np.round(
            nib.affines.apply_affine(
                np.linalg.inv(mni_mask.affine),
                mni_coords.astype(float),
            )
        ).astype(int)
        return ExplicitVBR(voxels, mni_mask.affine, image_dim=mni_mask.shape)

    @nl.add_symbol
    def agg_create_region_overlay(
        x: Iterable, y: Iterable, z: Iterable, p: Iterable
    ) -> ExplicitVBR:
        mni_coords = np.c_[x, y, z]
        voxels = np.round(
            nib.affines.apply_affine(
                np.linalg.inv(mni_mask.affine),
                mni_coords.astype(float),
            )
        ).astype(int)
        return ExplicitVBROverlay(
            voxels, mni_mask.affine, p, image_dim=mni_mask.shape
        )

    @nl.add_symbol
    def startswith(prefix: str, s: str) -> bool:
        """Describe the prefix of string `s`.

        Parameters
        ----------
        prefix : str
            prefix to query.
        s : str
            string to check whether its
            prefixed by `s`.

        Returns
        -------
        bool
            whether `s` is prefixed by
            `prefix`.
        """
        return s.startswith(prefix)

    @nl.add_symbol
    def principal_direction(s: ExplicitVBR, direction: str, eps=1e-6) -> bool:
        """Describe the principal direction of
        the extension of a volumetric region.

        Parameters
        ----------
        s : ExplicitVBR
            region to analyse the principal
            direction of its extension.
        direction : str
            principal directions, one of
            `LR`, `AP`, `SI`, for the directions
            left-right, anterio-posterior, and
            superior inferior respectively.
        eps : float, optional
            minimum difference on between
            directional standard deviations,
            by default 1e-6.

        Returns
        -------
        bool
            whether the principal variance of
            `s` is `direction`.
        """
        # Assuming RAS coding os the xyz space.
        c = ["LR", "AP", "SI"]

        s_xyz = s.to_xyz()
        cov = np.cov(s_xyz.T)
        evals, evecs = np.linalg.eig(cov)
        i = np.argmax(np.abs(evals))
        abs_max_evec = np.abs(evecs[:, i].squeeze())
        sort_dir = np.argsort(abs_max_evec)
        if (
            np.abs(abs_max_evec[sort_dir[-1]] - abs_max_evec[sort_dir[-2]])
            < eps
        ):
            return False
        else:
            main_dir = c[sort_dir[-1]]
        return (direction == main_dir) or (direction[::-1] == main_dir)

    nl.add_symbol(nl.symbols["agg_create_region_overlay"], name="create region overlay")
    nl.add_symbol(nl.symbols["agg_create_region_overlay"], name="created region overlay")
    nl.add_symbol(nl.symbols["agg_create_region"], name="create region")
    nl.add_symbol(nl.symbols["agg_create_region"], name="created region")
    return nl


class DestrieuxEngineConf(NeurolangEngineConfiguration):
    def __init__(self, data_dir: Path, resolution: int = None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.resolution = resolution
        self._mni_atlas = None

    @property
    def key(self):
        return "destrieux"

    @property
    def atlas(self):
        if self._mni_atlas is None:
            self._mni_atlas = nib.load(
                datasets.fetch_icbm152_2009(
                    data_dir=str(self.data_dir / "icbm")
                )["t1"]
            )
        return self._mni_atlas

    def create(self) -> NeurolangPDL:
        mask = self.atlas
        if self.resolution is not None:
            mask = image.resample_img(mask, np.eye(3) * self.resolution)
        nl = init_frontend(mask)
        nl.add_symbol(
            region_union,
            name="region_union",
            type_=Callable[[Iterable[ExplicitVBR]], ExplicitVBR],
        )

        load_destrieux_atlas(self.data_dir, nl)
        return nl


def load_destrieux_atlas(data_dir, nl):
    destrieux_atlas = datasets.fetch_atlas_destrieux_2009(
        data_dir=str(data_dir / "destrieux")
    )

    destrieux_atlas_images = nib.load(destrieux_atlas["maps"])
    destrieux_atlas_labels = {
        label: str(name.replace("-", " ").replace("_", " "))
        for label, name in destrieux_atlas["labels"]
        if name != b"Background"
    }
    nl.add_atlas_set(
        "destrieux", destrieux_atlas_labels, destrieux_atlas_images
    )


def add_ploting_functions(nl: Union[NeurolangDL, NeurolangPDL]):
    matplotlib.use('Agg')

    @nl.add_symbol
    def agg_kde(terms: Iterable, probs: Iterable) -> matplotlib.figure.Figure:
        """
        Create a kde plot showing prob distribution per term.

        Parameters
        ----------
        terms : Iterable[str]
            the terms
        probs: Iterable[float]
            the probs

        Returns
        -------
        matplotlib.figure.Figure
            a Figure
        """
        df = pd.DataFrame(
            {
                "terms": terms,
                "probs": probs,
            }
        )
        fig, ax = plt.subplots()
        fig.suptitle("Distribution of probs / term")
        sns.kdeplot(ax=ax, data=df[df.probs > 0.002], x="probs", hue="terms")
        return fig
