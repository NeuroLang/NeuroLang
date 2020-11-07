r"""
Neuroimaging Mixins: Region and Neurosynth Mixins
================================================
Mixins provide capabilities related to brain volumes and
Neurosynth metadata manipulation
"""

from typing import AbstractSet, Dict, Iterable, List, Optional, Tuple, Union
from uuid import uuid1

import numpy as np

from nibabel.dataobj_images import DataobjImage

from .. import expressions as ir
from ..region_solver import Region
from ..regions import ExplicitVBR, ImplicitVBR, SphericalVolume
from . import query_resolution_expressions as fe
from .neurosynth_utils import NeuroSynthHandler, StudyID, TfIDf


class RegionMixin:
    """
    Mixin complementing a QueryBuilderBase
    with methods specific to the manipulation
    of brain volumes: regions, atlases, etc...
    """

    @property
    def region_names(self) -> List[str]:
        """
        Returns the list of symbol names with Region type

        Returns
        -------
        List[str]
            list of symbol names from symbol_table
        """
        return [s.name for s in self.symbol_table.symbols_by_type(Region)]

    @property
    def region_set_names(self) -> List[str]:
        """
        Returns the list of symbol names with set_type

        Returns
        -------
        List[str]
            list of symbol names from symbol_table
        """
        return [
            s.name for s in self.symbol_table.symbols_by_type(self.set_type)
        ]

    def new_region_symbol(self, name: Optional[str] = None) -> fe.Symbol:
        """
        Returns symbol with type Region

        Parameters
        ----------
        name : Optional[str], optional
            symbol's name, if None will be fresh, by default None

        Returns
        -------
        fe.Symbol
            see description
        """
        return self.new_symbol(type_=Region, name=name)

    def add_region(
        self, region: fe.Expression, name: Optional[str] = None
    ) -> fe.Symbol:
        """
        Adds region fe.Symbol to symbol_table

        Parameters
        ----------
        region : fe.Expression
            should be of the program_ir's type
        name : Optional[str], optional
            symbol's name, if None will be fresh, by default None

        Returns
        -------
        fe.Symbol
            symbol added to the symbol_table

        Raises
        ------
        ValueError
            if region is not of program_ir's type
        """
        if not isinstance(region, self.program_ir.type):
            raise ValueError(
                f"type mismatch between region and program_ir type:"
                f" {self.program_ir.type}"
            )

        return self.add_symbol(region, name)

    def add_region_set(
        self, region_set: Iterable, name: Optional[str] = None
    ) -> fe.Symbol:
        """
        Creates an AbstractSet fe.Symbol containing the elements specified in
        the iterable with a List[Tuple[Region]] format

        Parameters
        ----------
        region_set : Iterable
            Typically List[Tuple[Region]]
        name : Optional[str], optional
            symbol's name, if None will be fresh, by default None

        Returns
        -------
        fe.Symbol
            see description
        """
        return self.add_tuple_set(region_set, name=name, type_=Region)

    @staticmethod
    def create_region(
        spatial_image: DataobjImage,
        label: int = 1,
        prebuild_tree: bool = False,
    ) -> ExplicitVBR:
        """
        Creates an ExplicitVBR out of the voxels of a dense spatial_image
        with specified label

        Parameters
        ----------
        spatial_image : DataobjImage
            contains the .dataobj array of interest
        label : int, optional
            selects which voxel to put in the output, by default 1
        prebuild_tree : bool, optional
            see ExplicitVBR, by default False

        Returns
        -------
        ExplicitVBR
            see description
        """
        voxels = np.transpose(
            (np.asanyarray(spatial_image.dataobj) == label).nonzero()
        )
        if len(voxels) == 0:
            return None
        region = ExplicitVBR(
            voxels,
            spatial_image.affine,
            image_dim=spatial_image.shape,
            prebuild_tree=prebuild_tree,
        )
        return region

    def add_atlas_set(
        self,
        name: str,
        atlas_labels: Dict[int, str],
        spatial_image: DataobjImage,
    ) -> fe.Symbol:
        """
        Creates an atlas set:
        1- for each region specified by a label and name in atlas_labels,
        creates associated ExplicitVBR and symbols
        Tuple[region_name: str, Region]
        2- groups regions in an AbstractSet[Tuple[str, Region]] symbol with
        specified name

        Parameters
        ----------
        name : str
            name for the output atlas symbol
        atlas_labels : Dict[int, str]
            specifies which voxels to select in the spatial_image,
            and the associated region name
        spatial_image : DataobjImage
            contains the .dataobj array ofinterest

        Returns
        -------
        fe.Symbol
            see description
        """
        atlas_set = set()
        for label_number, label_name in atlas_labels:
            region = self.create_region(spatial_image, label=label_number)
            if region is None:
                continue
            symbol = ir.Symbol[Region](label_name)
            self.symbol_table[symbol] = ir.Constant[Region](region)
            self.symbol_table[
                self.new_symbol(type_=str).expression
            ] = ir.Constant[str](label_name)

            tuple_symbol = self.new_symbol(type_=Tuple[str, Region]).expression
            self.symbol_table[tuple_symbol] = ir.Constant[Tuple[str, Region]](
                (ir.Constant[str](label_name), symbol)
            )
            atlas_set.add(tuple_symbol)
        atlas_set = ir.Constant[AbstractSet[Tuple[str, Region]]](
            frozenset(atlas_set)
        )
        atlas_symbol = ir.Symbol[atlas_set.type](name)
        self.symbol_table[atlas_symbol] = atlas_set
        return self[atlas_symbol]

    def sphere(
        self,
        center: Union[np.ndarray, Iterable[int]],
        radius: int,
        name: Optional[str] = None,
    ) -> fe.Symbol:
        """
        Creates a Region symbol associated with the spherical
        volume described by its center and volume

        Parameters
        ----------
        center : Union[np.ndarray, Iterable[int]]
            3D center of the sphere
        radius : int
            radius of the sphere
        name : Optional[str], optional
            name of the output symbol, if None will be fresh,
            by default None

        Returns
        -------
        fe.Symbol
            see description
        """
        sr = SphericalVolume(center, radius)
        symbol = self.add_region(sr, name)
        return symbol

    def make_implicit_regions_explicit(self, affine, dim):
        """Raises NotImplementedError for now"""
        for region_symbol_name in self.region_names:
            region_symbol = self.get_symbol(region_symbol_name)
            region = region_symbol.value
            if isinstance(region, ImplicitVBR):
                self.add_region(
                    region.to_explicit_vbr(affine, dim), region_symbol_name
                )


class NeuroSynthMixin:
    """
    Neurosynth is a platform for large-scale, automated synthesis
    of functional magnetic resonance imaging (fMRI) data.
    see https://neurosynth.org/
    This Mixin complements a QueryBuilderBase with methods
    related to coordinate-based meta-analysis (CBMA) data loading.
    """

    def load_neurosynth_term_study_ids(
        self,
        term: str,
        name: Optional[str] = None,
        frequency_threshold: Optional[float] = 0.05,
    ) -> fe.Symbol:
        """
        Load study ids (PMIDs) of studies in the Neurosynth database that are
        related to a given term based on a hard thresholding of TF-IDF
        features.

        Parameters
        ----------
        term : str
            Term that studies must match.

        name : str (optional)
            Name of the symbol associated with the resulting set of studies.
            Randomly generated by default

        frequency_threshold : optional (default 0.05)
            Minimum value of the TF-IDF feature for studies to be considered
            related to the given term.

        Returns
        -------
        fe.Symbol
            fe.Symbol associated to the resulting set.

        """
        if not hasattr(self, "neurosynth_db"):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        study_set = self.neurosynth_db.ns_study_id_set_from_term(
            term, frequency_threshold
        )
        return self.add_tuple_set(
            study_set.values, type_=Tuple[StudyID], name=name
        )

    def load_neurosynth_study_tfidf_feature_for_terms(
        self, terms: Iterable[str], name: Optional[str] = None
    ) -> fe.Symbol:
        """
        Load TF-IDF features measured in each study within the Neurosynth
        database for the given terms.

        Parameters
        ----------
        terms : iterable of str
            Terms for which the TF-IDF features should be obtained.

        name : str (optional)
            Name of the symbol associated with the resulting set of (study id,
            tfidf) tuples. Randomly generated by default

        Returns
        -------
        fe.Symbol
            fe.Symbol associated to the resulting set.

        """
        if not hasattr(self, "neurosynth_db"):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        result_set = self.neurosynth_db.ns_study_tfidf_feature_for_terms(terms)
        return self.add_tuple_set(
            result_set.values, type_=Tuple[StudyID, str, TfIDf], name=name
        )

    def load_neurosynth_study_ids(
        self, name: Optional[str] = None
    ) -> fe.Symbol:
        """
        Load all study ids (PMIDs) that are part of the Neurosynth database.

        Parameters
        ----------
        name : str (optional)
            Name of the symbol associated with the resulting set of study ids.
            Randomly generated by default

        Returns
        -------
        fe.Symbol
            fe.Symbol associated to the resulting set.

        """
        if not hasattr(self, "neurosynth_db"):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        result_set = self.neurosynth_db.ns_study_ids()
        return self.add_tuple_set(result_set, type_=Tuple[StudyID], name=name)

    def load_neurosynth_reported_activations(
        self, name: Optional[str] = None
    ) -> fe.Symbol:
        """
        Load all activations reported in the Neurosynth database in the form of
        a set of (study id, voxel id) pairs.

        Parameters
        ----------
        name : str (optional)
            Name of the symbol associated with the resulting set of (study id,
            voxel id) tuples. Randomly generated by default

        Returns
        -------
        fe.Symbol
            fe.Symbol associated to the resulting set.

        """
        if not hasattr(self, "neurosynth_db"):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        result_set = self.neurosynth_db.ns_reported_activations()
        return self.add_tuple_set(
            result_set, type_=Tuple[StudyID, int], name=name
        )

    def load_neurosynth_term_study_associations(
        self,
        name: Optional[str] = None,
        threshold: Optional[float] = 1e-3,
        study_ids: Optional[List[int]] = None,
    ) -> fe.Symbol:
        """
        Load all term-to-study associations in the Neurosynth database based on
        a hard thresholding of the TF-IDF features measured from the abstracts
        of the studies.

        Parameters
        ----------
        name : str (optional)
            Name of the symbol associated with the resulting set of study ids.
            Randomly generated by default

        threshold : float (optional, default 1e-3)
            Threshold used to determine the association of a study to a given
            term.

        study_ids : iterable of int (optional)
            List of the IDs of the studies to consider. By default, all studies
            in the database are considered.

        Returns
        -------
        fe.Symbol
            fe.Symbol associated to the resulting set.

        """
        if not hasattr(self, "neurosynth_db"):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        result_set = self.neurosynth_db.ns_term_study_associations(
            threshold=threshold, study_ids=study_ids
        )
        return self.add_tuple_set(
            result_set, type_=Tuple[StudyID, str], name=name
        )
