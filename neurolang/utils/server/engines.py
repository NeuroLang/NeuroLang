from abc import abstractproperty, abstractstaticmethod
from contextlib import contextmanager
from multiprocessing import BoundedSemaphore
from pathlib import Path
from typing import Iterable, Union

import nibabel as nib
import numpy as np
import pandas as pd
from neurolang.frontend import NeurolangDL, NeurolangPDL
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

    @abstractstaticmethod
    def create() -> Union[NeurolangDL, NeurolangPDL]:
        pass


class NeurosynthEngineConf(NeurolangEngineConfiguration):
    @property
    def key(self):
        return "neurosynth"

    @staticmethod
    def create() -> NeurolangPDL:
        nl = init_frontend()
        load_neurosynth_data(nl)
        return nl

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NeurolangEngineConfiguration):
            return super().__eq__(other)
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)


def load_neurosynth_data(nl):
    data_dir = Path("neurolang_data")
    resolution = 2
    mni_mask = image.resample_img(
        nib.load(
            datasets.fetch_icbm152_2009(data_dir=str(data_dir / "icbm"))["gm"]
        ),
        np.eye(3) * resolution,
    )

    ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
        data_dir / "neurosynth",
        [
            (
                "database.txt",
                "https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz",
                {"uncompress": True},
            ),
            (
                "features.txt",
                "https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz",
                {"uncompress": True},
            ),
        ],
    )

    activations = pd.read_csv(ns_database_fn, sep="\t")
    mni_peaks = activations.loc[activations.space == "MNI"][
        ["x", "y", "z", "id"]
    ].rename(columns={"id": "study_id"})
    non_mni_peaks = activations.loc[activations.space != "MNI"][
        ["x", "y", "z", "id"]
    ].rename(columns={"id": "study_id"})
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
        np.hstack([projected, non_mni_peaks[["study_id"]].values]),
        columns=["x", "y", "z", "study_id"],
    )
    peak_data = pd.concat([projected_df, mni_peaks]).astype(
        {"x": int, "y": int, "z": int}
    )
    study_ids = peak_data[["study_id"]].drop_duplicates()

    ijk_positions = np.round(
        nib.affines.apply_affine(
            np.linalg.inv(mni_mask.affine),
            peak_data[["x", "y", "z"]].values.astype(float),
        )
    ).astype(int)
    peak_data["i"] = ijk_positions[:, 0]
    peak_data["j"] = ijk_positions[:, 1]
    peak_data["k"] = ijk_positions[:, 2]
    peak_data = peak_data[["i", "j", "k", "study_id"]]

    features = pd.read_csv(ns_features_fn, sep="\t")
    features.rename(columns={"pmid": "study_id"}, inplace=True)

    term_data = pd.melt(
        features,
        var_name="term",
        id_vars="study_id",
        value_name="tfidf",
    ).query("tfidf > 1e-3")[["term", "tfidf", "study_id"]]

    nl.add_tuple_set(peak_data, name="PeakReported")
    nl.add_tuple_set(study_ids, name="Study")
    nl.add_tuple_set(term_data, name="TermInStudyTFIDF")
    nl.add_uniform_probabilistic_choice_over_set(
        study_ids, name="SelectedStudy"
    )
    nl.add_tuple_set(
        np.hstack(
            np.meshgrid(
                *(np.arange(0, dim) for dim in mni_mask.get_fdata().shape)
            )
        )
        .swapaxes(0, 1)
        .reshape(3, -1)
        .T,
        name="Voxel",
    )


def init_frontend():
    """
    Create a Neurolang Probabilistic engine and add some aggregation methods.

    Returns
    -------
    NeurolangPDL
        the Neurolang engine
    """
    nl = NeurolangPDL()

    @nl.add_symbol
    def agg_count(i: Iterable) -> np.int64:
        return np.int64(len(i))

    return nl
