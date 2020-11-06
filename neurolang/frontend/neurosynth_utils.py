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
        return study_ids.values

    def ns_study_tfidf_feature_for_terms(self, terms):
        features = self.dataset.feature_table.data
        if set(terms) - set(features.columns):
            not_found = sorted(set(terms) - set(features.columns))
            raise ValueError(
                "Could not find terms: {}".format(", ".join(not_found))
            )
        features = features[list(terms)]
        features["study_id"] = features.index.to_series().astype(int)
        return features.melt(
            id_vars="study_id",
            var_name="term",
            value_vars=list(terms),
            value_name="tfidf",
        )[["study_id", "term", "tfidf"]]

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
            study_ids = features.index.to_series()
        features = features.loc[study_ids]
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
        study_ids = pd.Series(image_table.ids).iloc[study_ids_ix]
        return np.transpose([vox_ids, study_ids])

    def ns_study_ids(self):
        return np.expand_dims(
            self.dataset.feature_table.data.index.values, axis=1
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
