from abc import abstractmethod, abstractproperty
from contextlib import contextmanager
from multiprocessing import BoundedSemaphore
from pathlib import Path
from typing import Callable, Iterable, Union

import nibabel as nib
import numpy as np
import pandas as pd
from neurolang.frontend import NeurolangDL, NeurolangPDL
from neurolang.frontend.neurosynth_utils import StudyID
from neurolang.regions import ExplicitVBR, ExplicitVBROverlay, region_union
from nilearn import datasets, image


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
        timout : int
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

    def create(self) -> NeurolangPDL:
        mask = self.atlas
        if self.resolution is not None:
            mask = image.resample_img(mask, np.eye(3) * self.resolution)
        nl = init_frontend(mask)
        load_neurosynth_data(self.data_dir, nl, mask)
        return nl


def load_neurosynth_data(data_dir: Path, nl, mni_mask: nib.Nifti1Image):
    ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
        data_dir / "neurosynth",
        [
            (
                "database.txt",
                "https://github.com/neurosynth/neurosynth-data/raw/e8f27c4a9a44dbfbc0750366166ad2ba34ac72d6/current_data.tar.gz",
                {"uncompress": True},
            ),
            (
                "features.txt",
                "https://github.com/neurosynth/neurosynth-data/raw/e8f27c4a9a44dbfbc0750366166ad2ba34ac72d6/current_data.tar.gz",
                {"uncompress": True},
            ),
        ],
    )

    activations = pd.read_csv(ns_database_fn, sep="\t")
    activations["id"] = activations["id"].apply(StudyID)
    mni_peaks = activations.loc[activations.space == "MNI"][
        ["x", "y", "z", "id"]
    ]
    non_mni_peaks = activations.loc[activations.space == "TAL"][
        ["x", "y", "z", "id"]
    ]
    proj_mat = np.linalg.pinv(
        np.array(
            [
                [0.9254, 0.0024, -0.0118, -1.0207],
                [-0.0048, 0.9316, -0.0871, -1.7667],
                [0.0152, 0.0883, 0.8924, 4.0926],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).T
    )
    projected = np.round(
        np.dot(
            np.hstack(
                (
                    non_mni_peaks[["x", "y", "z"]].values,
                    np.ones((len(non_mni_peaks), 1)),
                )
            ),
            proj_mat,
        )[:, 0:3]
    )
    projected_df = pd.DataFrame(
        np.hstack([projected, non_mni_peaks[["id"]].values]),
        columns=["x", "y", "z", "id"],
    )
    peak_data = pd.concat([projected_df, mni_peaks]).astype(
        {"x": int, "y": int, "z": int}
    )
    study_ids = peak_data[["id"]].drop_duplicates()

    features = pd.read_csv(ns_features_fn, sep="\t")
    features.rename(columns={"pmid": "id"}, inplace=True)

    term_data = pd.melt(
        features,
        var_name="term",
        id_vars="id",
        value_name="tfidf",
    ).query("tfidf > 0")[["term", "tfidf", "id"]]
    term_data["id"] = term_data["id"].apply(StudyID)

    nl.add_tuple_set(peak_data, name="PeakReported")
    nl.add_tuple_set(study_ids, name="Study")
    nl.add_tuple_set(term_data, name="TermInStudyTFIDF")
    nl.add_uniform_probabilistic_choice_over_set(
        study_ids, name="SelectedStudy"
    )
    nl.add_tuple_set(
        np.round(
            nib.affines.apply_affine(
                mni_mask.affine, np.transpose(mni_mask.get_fdata().nonzero())
            )
        ).astype(int),
        name="Voxel",
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
            wether the principal variance of
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

    nl.new_symbol(name="destrieux")
    destrieux_atlas_image = nib.load(destrieux_atlas["maps"])
    destrieux_labels = dict(destrieux_atlas["labels"])
    destrieux_set = set()
    for k, v in destrieux_labels.items():
        if k == 0:
            continue
        destrieux_set.add(
            (
                v.decode("utf8").replace("-", " ").replace("_", " "),
                ExplicitVBR.from_spatial_image_label(destrieux_atlas_image, k),
            )
        )
    nl.add_tuple_set(destrieux_set, name="destrieux")
