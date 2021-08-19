import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from nilearn.datasets.utils import _fetch_files
from scipy import sparse
from pathlib import Path
from pkg_resources import resource_exists, resource_filename


try:
    import neurosynth as ns
except ModuleNotFoundError:
    raise ImportError("Neurosynth not installed in the system")


class StudyID(str):
    pass


class TfIDf(float):
    pass


class NeuroSynthHandler(object):
    """
    Class for the management of data provided by neurosynth.
    """

    def __init__(self, ns_dataset=None):
        self._dataset = ns_dataset

    def ns_study_id_set_from_term(self, terms, frequency_threshold=0.05):
        study_ids = self.dataset.get_studies(
            features=terms, frequency_threshold=frequency_threshold
        )
        return set(StudyID(study_id) for study_id in study_ids)

    def ns_study_tfidf_feature_for_terms(self, terms):
        feature_table = self.dataset.feature_table.data
        result_set = set()
        for term in terms:
            if term not in feature_table.columns:
                continue
            result_set |= set(
                (StudyID(tupl[0]), term, tupl[1])
                for tupl in feature_table[[term]].itertuples(
                    index=True, name=None
                )
            )
        return result_set

    def ns_load_dataset(self):

        if resource_exists(
            "neurolang.frontend", "neurosynth_data/dataset.pkl"
        ):
            file = resource_filename(
                "neurolang.frontend", "neurosynth_data/dataset.pkl"
            )
            dataset = ns.Dataset.load(file)
        else:
            path = resource_filename("neurolang.frontend", "neurosynth_data")
            logging.info(
                f"Downloading neurosynth database"
                f" and features in path: {path}"
            )
            dataset = self.download_ns_dataset(path)

        return dataset

    def ns_term_study_associations(self, threshold=1e-3, study_ids=None):
        """
        Load a 2d numpy array containing association between terms and studies
        based on thresholded tf-idf features in the database.

        """
        features = self.dataset.feature_table.data
        terms = features.columns
        if study_ids is None:
            study_ids = features.index.to_series().apply(StudyID)
        study_ids_as_int = study_ids.apply(int)
        features = features.loc[study_ids_as_int]
        features["pmid"] = study_ids
        return (
            features.melt(
                id_vars="pmid",
                var_name="term",
                value_vars=terms,
                value_name="tfidf",
            )
            .query(f"tfidf > {threshold}")[["pmid", "term"]]
            .values
        )

    def ns_reported_activations(self):
        """
        Load a 2d numpy array containing each reported activation in the
        database.

        """
        image_table = self.dataset.image_table
        vox_ids, study_ids_ix = image_table.data.nonzero()
        study_ids = (
            pd.Series(image_table.ids).apply(StudyID).iloc[study_ids_ix]
        )
        return np.transpose([study_ids, vox_ids])

    def ns_study_ids(self):
        return np.expand_dims(
            self.dataset.feature_table.data.index.astype(StudyID), axis=1
        )

    @staticmethod
    def download_ns_dataset(path):
        if not os.path.exists(path):
            os.makedirs(path)
        ns.dataset.download(path=path, unpack=True)
        dataset = ns.Dataset(os.path.join(path, "database.txt"))
        dataset.add_features(os.path.join(path, "features.txt"))
        dataset.save(os.path.join(path, "dataset.pkl"))
        return dataset

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = self.ns_load_dataset()
        return self._dataset


class NeuroSynthHandler(object):

    ns_data_url = "https://github.com/neurosynth/neurosynth-data/raw/master/"

    def fetch_study_metadata(
        self, data_dir: Path, version: int = 7, verbose: int = 1
    ) -> pd.DataFrame:
        """
        Download if needed the `metadata.tsv.gz` file from Neurosynth and load
        it into a pandas DataFrame.

        The metadata table contains the metadata for each study. Each study (ID)
        is stored on its own line. These IDs are in the same order as the id
        column of the associated `coordinates.tsv.gz` file, but the rows will
        differ because the coordinates file will contain multiple rows per
        study. They are also in the same order as the rows in the
        `features.npz` files for the same version.

        The metadata will therefore have N rows, N being the number of studies
        in the Neurosynth dataset. The columns (for version 7) are:
            - id
            - doi
            - space
            - title
            - authors
            - year
            - journal

        Parameters
        ----------
        data_dir : Path
            the path for the directory where downloaded data should be saved.
        version : int, optional
            the neurosynth data version, by default 7
        verbose : int, optional
            verbose param for nilearn's `_fetch_files`, by default 1

        Returns
        -------
        pd.DataFrame
            the study metadata dataframe
        """
        metadata_filename = (
            f"data-neurosynth_version-{version}_metadata.tsv.gz"
        )
        metadata_file = _fetch_files(
            data_dir,
            [
                (
                    metadata_filename,
                    self.ns_data_url + metadata_filename,
                    {},
                ),
            ],
        )[0]
        metadata = pd.read_table(metadata_file)
        return metadata

    def fetch_feature_data(
        self,
        data_dir: Path,
        version: int = 7,
        verbose: int = 1,
        convert_study_ids: bool = False,
    ) -> pd.DataFrame:
        """
        Download if needed the `tfidf_features.npz` file from Neurosynth and
        load it into a pandas Dataframe. The `tfidf_features` contains feature
        values for different types of "vocabularies".

        The features dataframe is stored as a compressed, sparse matrix.
        Once loaded and reconstructed into a dense matrix, it contains one row
        per study and one column per label. The associated labels are loaded,
        as well as the study ids, to reconstruct a dataframe of size N x P,
        where N is the number of studies in the Neurosynth dataset, and P is
        the number of words in the vocabulary.

        Parameters
        ----------
        data_dir : Path
            the path for the directory where downloaded data should be saved.
        version : int, optional
            the neurosynth data version, by default 7
        verbose : int, optional
            verbose param for nilearn's `_fetch_files`, by default 1
        convert_study_ids : bool, optional
            if True, cast study ids as `StudyID`, by default False

        Returns
        -------
        pd.DataFrame
            the features dataframe
        """
        file_names = [
            f"data-neurosynth_version-{version}_vocab-terms_source-abstract_type-tfidf_features.npz",
            f"data-neurosynth_version-{version}_vocab-terms_vocabulary.txt",
        ]
        files = _fetch_files(
            data_dir,
            [
                (
                    fn,
                    self.ns_data_url + fn,
                    {},
                )
                for fn in file_names
            ],
            verbose=verbose,
        )
        feature_data_sparse = sparse.load_npz(files[0])
        feature_data = feature_data_sparse.todense()
        metadata_df = self.fetch_study_metadata(data_dir, version, verbose)
        ids = metadata_df["id"].tolist()
        feature_names = np.genfromtxt(
            files[1],
            dtype=str,
            delimiter="\t",
        ).tolist()
        feature_df = pd.DataFrame(
            index=ids, columns=feature_names, data=feature_data
        )
        if convert_study_ids:
            feature_df["id"] = feature_df["id"].apply(StudyID)
        return feature_df

    def fetch_neurosynth_peak_data(
        self,
        data_dir: Path,
        version: int = 7,
        verbose: int = 1,
        convert_study_ids: bool = False,
    ) -> pd.DataFrame:
        """
        Download if needed the `coordinates.tsv.gz` file from Neurosynth and
        load it into a pandas DataFrame.

        The `coordinates.tsv.gz` contains the coordinates for the peaks
        reported by studies in the Neurosynth dataset. It contains one row per
        coordinate reported.

        The metadata for each study is also loaded to include the space in
        which the coordinates are reported. The peak_data dataframe therefore
        has PR rows, PR being the number of reported peaks in the Neurosynth
        dataset.

        The columns (for version 7) are:
            - id
            - table_id
            - table_num
            - peak_id
            - space
            - x
            - y
            - z

        Parameters
        ----------
        data_dir : Path
            the path for the directory where downloaded data should be saved.
        version : int, optional
            the neurosynth data version, by default 7
        verbose : int, optional
            verbose param for nilearn's `_fetch_files`, by default 1
        convert_study_ids : bool, optional
            if True, cast study ids as `StudyID`, by default False

        Returns
        -------
        pd.DataFrame
            the peak dataframe
        """
        coordinates_filename = (
            f"data-neurosynth_version-{version}_coordinates.tsv.gz"
        )
        coordinates_file = _fetch_files(
            data_dir,
            [
                (
                    coordinates_filename,
                    self.ns_data_url + coordinates_filename,
                    {},
                ),
            ],
            verbose=verbose,
        )[0]
        activations = pd.read_table(coordinates_file)
        metadata = self.fetch_study_metadata(data_dir, version, verbose)
        activations = activations.join(
            metadata[["id", "space"]].set_index("id"), on="id"
        )
        if convert_study_ids:
            activations["id"] = activations["id"].apply(StudyID)
        return activations

    def get_ns_term_study_associations(
        self,
        data_dir: Path,
        version: int = 7,
        verbose: int = 1,
        convert_study_ids: bool = False,
        tfidf_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Load a dataframe containing associations between term and studies.
        The dataframe contains one row for each term and study pair from the
        features table in the Neurosynth dataset. With each (term, study) pair
        comes the tfidf value for the term in the study.
        If a tfidf threshold value is passed, only (term, study) associations
        with a tfidf value > tfidf_threshold will be kept.

        Parameters
        ----------
        data_dir : Path
            the path for the directory where downloaded data should be saved.
        version : int, optional
            the neurosynth data version, by default 7
        verbose : int, optional
            verbose param for nilearn's `_fetch_files`, by default 1
        convert_study_ids : bool, optional
            if True, cast study ids as `StudyID`, by default False
        tfidf_threshold : Optional[float], optional
            the minimum tfidf value for the (term, study) associations,
            by default None

        Returns
        -------
        pd.DataFrame
            the term association dataframe
        """
        features = self.fetch_feature_data(
            data_dir, version, verbose, convert_study_ids
        )
        features.index.name = "id"
        term_data = pd.melt(
            features.reset_index(),
            var_name="term",
            id_vars="id",
            value_name="tfidf",
        )
        if tfidf_threshold is not None:
            term_data = term_data.query(f"tfidf > {tfidf_threshold}")
        else:
            term_data = term_data.query("tfidf > 0")
        return term_data

    def get_ns_mni_peaks_reported(self,
        data_dir: Path,
        version: int = 7,
        verbose: int = 1,
        convert_study_ids: bool = False,
    ) -> pd.DataFrame:
        """
        Load a dataframe containing the coordinates for the peaks reported by
        studies in the Neurosynth dataset. Coordinates for the peaks are in
        MNI space, with coordinates that are reported in Talaraich converted.

        The resulting dataframe contains one row for each peak reported. Each
        row has 4 columns:
            - id
            - x
            - y
            - z

        Parameters
        ----------
        data_dir : Path
            the path for the directory where downloaded data should be saved.
        version : int, optional
            the neurosynth data version, by default 7
        verbose : int, optional
            verbose param for nilearn's `_fetch_files`, by default 1
        convert_study_ids : bool, optional
            if True, cast study ids as `StudyID`, by default False
        
        Returns
        -------
        pd.DataFrame
            the peak dataframe
        """
        activations = self.fetch_neurosynth_peak_data(data_dir, version, verbose, convert_study_ids)
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
        return peak_data