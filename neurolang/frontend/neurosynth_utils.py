import collections
import logging
import os

import numpy as np
import pandas as pd
from pkg_resources import resource_exists, resource_filename

from ..regions import region_set_from_masked_data

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

    def ns_region_set_from_term(
        self,
        terms,
        frequency_threshold=0.05,
        q=0.01,
        prior=0.5,
        image_type=None,
    ):
        """
        Method that allows to obtain the activations related to
        a series of terms. The terms can be entered in the following formats:

        String: All expressions allowed in neurosynth. It can be
        a term, a simple logical expression, for example: (reward* | pain*).

        Iterable: A list of terms that will be calculated as a disjunction.
        This case does not support logical expressions.
        """
        if image_type is None:
            image_type = f"association-test_z_FDR_{q}"

        if not isinstance(terms, str) and isinstance(
            terms, collections.Iterable
        ):
            studies_ids = self.dataset.get_studies(
                features=terms, frequency_threshold=frequency_threshold
            )
        else:
            studies_ids = self.dataset.get_studies(
                expression=terms, frequency_threshold=frequency_threshold
            )
        ma = ns.meta.MetaAnalysis(self.dataset, studies_ids, q=q, prior=prior)
        data = ma.images[image_type]
        masked_data = self.dataset.masker.unmask(data)
        affine = self.dataset.masker.get_header().get_sform()
        dim = self.dataset.masker.dims
        region_set = region_set_from_masked_data(masked_data, affine, dim)
        return region_set

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

    def ns_load_term_study_associations(self, threshold=1e-3, study_ids=None):
        """
        Load a 2d numpy array containing association between terms and studies
        based on thresholded tf-idf features in the database.

        """
        features = self.dataset.feature_table.data
        if study_ids is not None:
            study_ids = np.array(list(study_ids)).flatten().astype(int)
            features = features.loc[study_ids]
        terms = features.columns
        features["pmid"] = features.index
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

    def ns_load_reported_activations(self):
        """
        Load a 2d numpy array containing each reported activation in the
        database.

        """
        image_table = self.dataset.image_table
        vox_ids, study_ids_ix = image_table.data.nonzero()
        study_ids = image_table.ids[study_ids_ix]
        study_id_vox_id = np.transpose([study_ids, vox_ids])
        return study_id_vox_id

    def ns_load_all_study_ids(self):
        return np.expand_dims(
            self.dataset.feature_table.data.index.values, axis=1
        )

    def ns_prob_term_in_study(self, terms):
        """
        Given a set of terms, returns the probability of that term of been relevant in the studies.
        Defined as: P(term_relevant) = cant(term_relevant) /  cant(all_terms)

        """
        res = []
        for term in terms:
            data = nsh.ns_study_tfidf_feature_for_terms([term])
            df = pd.DataFrame(data, columns=["study", "term", "tfidf"])
            res.append([len(df[df.tfidf > 0]) / len(df), term])

        return pd.DataFrame(res, columns=["probability", "term"])

    def ns_prob_voxel_activated(self):
        """
        Returns the probability of all voxels being activated 

        """
        data = nsh.ns_load_reported_activations()
        df = pd.DataFrame(data, columns=["study", "voxel"])
        gby = df.groupby("voxel").count()
        prob_voxel = gby.study / len(df.study.unique())
        return prob_voxel.reset_index(name="probability")[
            ["probability", "voxel"]
        ]

    def ns_prob_document(self):
        return 1 / len(self.ns_load_all_study_ids())

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
