import os
import tarfile
from abc import abstractmethod, abstractproperty
from contextlib import contextmanager
from multiprocessing import BoundedSemaphore
from pathlib import Path
from token import NL
from typing import Callable, Iterable, Tuple, Union, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.sparse
from neurolang.frontend import NeurolangDL, NeurolangPDL
from neurolang.frontend.neurosynth_utils import StudyID
from neurolang.regions import ExplicitVBR, ExplicitVBROverlay, region_union
from nilearn import datasets, image

DIFUMO_N_COMPONENTS_TO_DOWNLOAD_ID = {
    256: "3vrct",
    512: "9b76y",
    1024: "34792",
}


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
                mni_mask.affine, np.transpose(
                    mni_mask.get_fdata().astype(int).nonzero()
                )
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

    @nl.add_symbol
    def agg_count(*iterables) -> int:
        return len(next(iter(iterables)))

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


class YeoEngineConf(NeurolangEngineConfiguration):
    def __init__(
        self,
        data_dir: Path,
        resolution=None,
        n_components=256,
        neuroquery_subsample_proportion: float = 0.5
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.resolution = resolution
        self._mni_mask = None
        self._n_components = n_components
        self._neuroquery_subsample_proportion = neuroquery_subsample_proportion

    @property
    def key(self):
        return "yeo"

    @property
    def atlas(self):
        if self._mni_mask is None:
            self._mni_mask = nib.load(datasets.fetch_icbm152_2009()["gm"])
        return self._mni_mask

    def create(self) -> NeurolangPDL:
        mask = self.atlas
        if self.resolution is not None:
            mask = image.resample_img(mask, np.eye(3) * self.resolution)

        nl = init_frontend(mask)
        load_neuroquery(self.data_dir, nl, mask)
        load_neurosynth_data(self.data_dir, nl, mask)
        load_difumo(self.data_dir, nl, mask, n_components=self._n_components, coord_type='ijk')
        load_neurosynth_topic_associations(self.data_dir, nl, 100)


        nl.add_symbol(
            np.log,
            name="log",
            type_=Callable[[float], float],
        )

        nl.add_symbol(
            lambda it: float(sum(it)),
            name="agg_sum",
            type_=Callable[[Iterable], float],
        )

        nl.add_tuple_set([("attention",), ("language",)], name="Network")
        nl.add_tuple_set(
            {
                ("FEF", "attention"),
                ("aIPS", "attention"),
                ("pIPS", "attention"),
                ("MT+", "attention"),
                ("IFG", "language"),
                ("SMG", "language"),
                ("AG", "language"),
                ("ITG", "language"),
                ("aSTS", "language"),
                ("mSTS", "language"),
                ("pSTS", "language"),
            },
            name="RegionInNetwork",
        )
        nl.add_tuple_set(
            {
                ("VWFA", -45, -57, -12),
                ("FEF", -26, -5, 50),
                ("MT+", -45, -71, -1),
                ("aIPS", -25, -62, 51),
                ("pIPS", -25, -69, 34),
                ("IFG", -53, 27, 16),
                ("SMG", -56, -43, 31),
                ("AG", -49, -57, 28),
                ("ITG", -61, -33, -15),
                ("aSTS", -54, -9, -20),
                ("mSTS", -53, -18, -10),
                ("pSTS", -52, -40, 5),
            },
            name="RegionSeedVoxel",
        )

        return nl


def load_difumo_meta(data_dir, nl, n_components: int = 256):
    out_dir = data_dir / "difumo"
    download_id = DIFUMO_N_COMPONENTS_TO_DOWNLOAD_ID[n_components]
    url = f"https://osf.io/{download_id}/download"
    labels_path = os.path.join(
        str(n_components), f"labels_{n_components}_dictionary.csv"
    )
    files = [
        (labels_path, url, {"uncompress": True}),
    ]
    files = datasets.utils._fetch_files(out_dir, files, verbose=2)
    labels = pd.DataFrame(pd.read_csv(files[0]))

    return labels

def load_difumo(
    data_dir,
    nl,
    mask: nib.Nifti1Image,
    component_filter_fun: Callable = lambda _: True,
    coord_type: str = "xyz",
    n_components: int = 256,
    with_probabilities: bool = False,
):
    out_dir = data_dir / "difumo"
    download_id = DIFUMO_N_COMPONENTS_TO_DOWNLOAD_ID[n_components]
    url = f"https://osf.io/{download_id}/download"
    csv_file = os.path.join(
        str(n_components), f"labels_{n_components}_dictionary.csv"
    )
    nifti_file = os.path.join(str(n_components), "3mm/maps.nii.gz")
    files = [
        (csv_file, url, {"uncompress": True}),
        (nifti_file, url, {"uncompress": True}),
    ]
    files = datasets.utils._fetch_files(out_dir, files, verbose=2)
    labels = pd.DataFrame(pd.read_csv(files[0]))
    img = image.load_img(files[1])
    img = image.resample_img(
        img,
        target_affine=mask.affine,
        interpolation="nearest",
    )
    img_data = img.get_fdata()
    to_concat = list()
    for i, label in enumerate(
        labels.loc[labels.apply(component_filter_fun, axis=1)].Difumo_names
    ):
        coordinates = np.argwhere(img_data[:, :, :, i] > 0)
        if coord_type == "xyz":
            coordinates = nib.affines.apply_affine(img.affine, coordinates)
        else:
            assert coord_type == "ijk"
        region_data = pd.DataFrame(coordinates, columns=list(coord_type))
        region_data["region"] = label
        cols = ["region"] + list(coord_type)
        if with_probabilities:
            probs = img_data[img_data[:, :, :, i] > 0, i] / img_data.max()
            region_data["probability"] = probs
            cols = ["probability"] + cols
        to_concat.append(region_data[cols])
    region_voxels = pd.concat(to_concat)

    RegionVoxel = nl.add_tuple_set(region_voxels, name="RegionVoxel")
    Network = nl.add_tuple_set({("ContA",), ("ContB",)}, name="Network")
    NetworkRegion = nl.add_tuple_set(
        set(
            (row["Yeo_networks17"], row["Difumo_names"])
            for _, row in labels.iterrows()
            if row["Yeo_networks17"] in ("ContA", "ContB")
        ),
        name="NetworkRegion",
    )

def load_neurosynth_topic_associations(data_dir, nl, n_topics: int) -> pd.DataFrame:
    if n_topics not in {50, 100, 200, 400}:
        raise ValueError(f"Unexpected number of topics: {n_topics}")
    ns_dir = data_dir / "neurosynth"
    ns_data_url = "https://github.com/neurosynth/neurosynth-data/blob/e8f27c4a9a44dbfbc0750366166ad2ba34ac72d6/"
    topic_data = datasets.utils._fetch_files(
        ns_dir,
        [
            (
                f"analyses/v5-topics-{n_topics}.txt",
                ns_data_url + "topics/v5-topics.tar.gz?raw=true",
                {"uncompress": True},
            ),
        ],
    )[0]
    ta = pd.read_csv(topic_data, sep="\t")
    ta.set_index("id", inplace=True)
    ta = ta.unstack().reset_index()
    ta.columns = ("topic", "study_id", "prob")
    ta = ta[["prob", "topic", "study_id"]]
    nl.add_probabilistic_facts_from_tuples(
        set(ta.itertuples(index=False, name=None)),
        name="TopicAssociation",
    )

def xyz_to_ijk(xyz, mask):
    voxels = nib.affines.apply_affine(
        np.linalg.inv(mask.affine),
        xyz,
    ).astype(int)
    return voxels

def load_neuroquery(
    data_dir,
    nl,
    mask: nib.Nifti1Image,
    tfidf_threshold: Optional[float] = None,
    coord_type: str = "xyz",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_url = "https://raw.githubusercontent.com/neuroquery/neuroquery_data/"

    tfidf_url = base_url + "8c2bd71acc6afd5e196c5bfc18bfbaf06749719d/training_data/corpus_tfidf.npz"
    coordinates_url = base_url + "8c2bd71acc6afd5e196c5bfc18bfbaf06749719d/training_data/coordinates.csv"
    feature_names_url = base_url + "8c2bd71acc6afd5e196c5bfc18bfbaf06749719d/training_data/feature_names.txt"
    study_ids_url = base_url + "8c2bd71acc6afd5e196c5bfc18bfbaf06749719d/training_data/pmids.txt"
    out_dir = data_dir / "neuroquery"
    os.makedirs(out_dir, exist_ok=True)
    (
        tfidf_fn,
        coordinates_fn,
        feature_names_fn,
        study_ids_fn,
    ) = datasets.utils._fetch_files(
        out_dir,
        [
            ("corpus_tfidf.npz", tfidf_url, dict()),
            ("coordinates.csv", coordinates_url, dict()),
            ("feature_names.txt", feature_names_url, dict()),
            ("pmids.txt", study_ids_url, dict()),
        ],
    )
    tfidf = scipy.sparse.load_npz(tfidf_fn)
    coordinates = pd.read_csv(coordinates_fn)
    assert coord_type in ("xyz", "ijk")
    if coord_type == "ijk":
        ijk = xyz_to_ijk(coordinates[["x", "y", "z"]], mask)
        coordinates["i"] = ijk[:, 0]
        coordinates["j"] = ijk[:, 1]
        coordinates["k"] = ijk[:, 2]
    coord_cols = list(coord_type)
    peak_data = coordinates[coord_cols + ["pmid"]].rename(
        columns={"pmid": "study_id"}
    )
    feature_names = pd.read_csv(feature_names_fn, header=None)
    study_ids = pd.read_csv(study_ids_fn, header=None)
    study_ids.rename(columns={0: "study_id"}, inplace=True)
    tfidf = pd.DataFrame(tfidf.todense(), columns=feature_names[0])
    tfidf["study_id"] = study_ids.iloc[:, 0]
    if tfidf_threshold is None:
        term_data = pd.melt(
            tfidf,
            var_name="term",
            id_vars="study_id",
            value_name="tfidf",
        ).query("tfidf > 0")[["term", "tfidf", "study_id"]]
    else:
        term_data = pd.melt(
            tfidf,
            var_name="term",
            id_vars="study_id",
            value_name="tfidf",
        ).query(f"tfidf > {tfidf_threshold}")[["term", "study_id"]]

    nl.add_tuple_set(peak_data, name="PeakReportedNQ")
    nl.add_tuple_set(study_ids, name="StudyNQ")
    nl.add_uniform_probabilistic_choice_over_set(
        study_ids, name="SelectedStudyNQ"
    )
    nl.add_tuple_set(
        term_data[["tfidf", "term", "study_id"]], name="NeuroQueryTFIDF"
    )

    nl.add_probabilistic_facts_from_tuples(
        set(
            term_data[["tfidf", "term", "study_id"]].itertuples(
                index=False, name=None
            )
        ),
        name="TermAssociation",
    )